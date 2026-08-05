import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medical_kg_sourceprep.report_pipeline import (
    ReportPipelineError,
    analyze_report,
    analyze_report_document,
    collect_metrics,
    _prompt,
    _reportable_reasoning_paths,
    _reasoning_context,
    _retrieval_queries,
    validate_model_result,
)
from medical_kg_sourceprep.graph_retrieval import GraphReasoningResult
from medical_kg_sourceprep.report_model import AbnormalFlag, Observation, ReferenceInterval
from medical_kg_sourceprep.qa import build_evidence_index
from tests.test_qa import _candidate_graph, _chunk_package


class FakeTransport:
    def __init__(self, result):
        self.result = result
        self.payload = None

    def post(self, payload):
        self.payload = payload
        return {"choices": [{"message": {"content": json.dumps(self.result, ensure_ascii=False)}}]}


def _report():
    return {
        "schema_version": "structured-report/v0.2",
        "metadata": {"patient_name": "private", "patient_identifier": "secret"},
        "observations": [
            {"raw_name": "alpha", "standard_name": "alpha", "abbreviation": "A", "value": "12", "unit": "U", "reference_interval": {"lower": "1", "upper": "10"}},
            {"raw_name": "beta", "standard_name": "beta", "value": "5", "unit": "U", "reference_interval": {"lower": "1", "upper": "10"}},
        ],
    }


def _report_with_indeterminate():
    report = _report()
    report["observations"].append({
        "raw_name": "gamma",
        "standard_name": "gamma",
        "value": "3",
        "unit": "U",
        "reference_interval": {"lower": None, "upper": None},
    })
    return report


def _model():
    return {
        "summary": "程序判定一个指标偏高，一个指标在范围内。",
        "abnormal_analyses": [{"metric_id": "alpha", "analysis": "书内证据仅支持该指标需要关注。", "evidence_ids": ["Edemo:chapter:page:0:0000"]}],
        "association_analysis": {"analysis": "知识库证据不足，无法作出书内关联解释。", "evidence_ids": []},
        "attention_suggestions": [{"text": "结合原始报告与专业人员复核。", "evidence_ids": ["Edemo:chapter:page:0:0000"]}],
        "insufficient_evidence": ["beta 未触发异常知识检索。"],
    }


class ReportPipelineTests(unittest.TestCase):
    def test_final_non_actionable_paths_are_not_reported_or_sent_to_model(self):
        paths = _reportable_reasoning_paths((
            {"graph_status": "final", "status": "final-precondition-failed"},
            {"graph_status": "final", "status": "final-no-case-match"},
            {"graph_status": "final", "status": "final-ambiguous"},
            {"graph_status": "final", "status": "final-unsupported"},
            {"graph_status": "final", "status": "final-case-match"},
            {"graph_status": "final", "status": "final-case-match-precondition-derived"},
            {"graph_status": "candidate-only", "status": "candidate-precondition-unverified"},
        ))

        self.assertEqual(
            [path["status"] for path in paths],
            [
                "final-case-match",
                "final-case-match-precondition-derived",
                "candidate-precondition-unverified",
            ],
        )

    def test_retrieval_queries_search_code_standard_name_and_raw_alias_separately(self):
        observation = Observation(
            raw_name="医院别名谷草转氨酶",
            standard_name="天冬氨酸氨基转移酶",
            abbreviation="AST",
            value="37",
            unit="U/L",
            reference_interval=ReferenceInterval("0", "31"),
        )
        self.assertEqual(
            _retrieval_queries(observation, AbnormalFlag.HIGH),
            (
                ("abbreviation", "AST 升高"),
                ("standard_name", "天冬氨酸氨基转移酶 升高"),
                ("raw_name", "医院别名谷草转氨酶 升高"),
            ),
        )

    def test_collect_metrics_invokes_each_abnormal_retrieval_channel(self):
        report = {
            "schema_version": "structured-report/v0.2",
            "observations": [{
                "raw_name": "医院别名谷草转氨酶",
                "standard_name": "天冬氨酸氨基转移酶",
                "abbreviation": "AST",
                "value": "37",
                "unit": "U/L",
                "reference_interval": {"lower": "0", "upper": "31"},
            }],
        }
        with patch("medical_kg_sourceprep.report_pipeline.query_index", return_value=[]) as query:
            metrics = collect_metrics(report, Path("unused.sqlite"))
        self.assertEqual(len(metrics), 1)
        self.assertEqual(
            [call.args[1] for call in query.call_args_list],
            ["AST 升高", "天冬氨酸氨基转移酶 升高", "医院别名谷草转氨酶 升高"],
        )

    def test_missing_unit_metric_is_compared_and_sent_to_candidate_reasoning(self):
        report = {
            "schema_version": "structured-report/v0.2",
            "observations": [{
                "raw_name": "红细胞计数",
                "standard_name": "红细胞计数",
                "abbreviation": "RBC",
                "value": "5.15",
                "unit": None,
                "reference_interval": {"lower": "3.8", "upper": "5.1"},
            }],
        }
        diagnostic = {"status": "matched", "matches": []}
        with (
            patch(
                "medical_kg_sourceprep.report_pipeline.graph_query_diagnostic",
                return_value=diagnostic,
            ),
            patch(
                "medical_kg_sourceprep.report_pipeline.query_index_with_graph",
                return_value=([], {}),
            ),
        ):
            metrics = collect_metrics(
                report, Path("unused.sqlite"), knowledge_graph=Path("unused-graph.sqlite")
            )

        self.assertEqual(metrics[0].evaluation.evidence.computed_flag, AbnormalFlag.HIGH)
        self.assertEqual(metrics[0].evaluation.normalized.unit, "10^12/L")
        self.assertEqual(metrics[0].evaluation.normalized.unit_source, "controlled_default")
        self.assertEqual(
            [error.code for error in metrics[0].evaluation.evidence.errors],
            ["default_unit_applied"],
        )
        with patch(
            "medical_kg_sourceprep.report_pipeline.graph_reasoning_paths",
            return_value=GraphReasoningResult(),
        ) as graph_paths:
            expanded, paths, rejections = _reasoning_context(
                metrics, Path("unused-graph.sqlite"), Path("unused.sqlite")
            )

        observation = graph_paths.call_args.args[2][0]
        self.assertEqual(observation["value"], "5.15")
        self.assertEqual(observation["unit"], "10^12/L")
        self.assertEqual(observation["computed_flag"], "high")
        self.assertEqual(expanded, metrics)
        self.assertEqual(paths, ())
        self.assertEqual(rejections, ())

    def test_report_code_is_resolved_before_graph_lookup(self):
        report = {
            "schema_version": "structured-report/v0.2",
            "observations": [{
                "raw_name": "NEUT%",
                "standard_name": "NEUT%",
                "abbreviation": None,
                "value": "76",
                "unit": "%",
                "reference_interval": {"lower": "40", "upper": "75"},
            }],
        }
        diagnostic = {
            "status": "matched",
            "match_mode": "exact_name",
            "matches": [{"name": "中性粒细胞百分数"}],
        }
        with (
            patch(
                "medical_kg_sourceprep.report_pipeline.graph_query_diagnostic",
                return_value=diagnostic,
            ) as graph_diagnostic,
            patch(
                "medical_kg_sourceprep.report_pipeline.query_index_with_graph",
                return_value=([], {}),
            ) as graph_query,
        ):
            metrics = collect_metrics(
                report, Path("unused.sqlite"), knowledge_graph=Path("unused-graph.sqlite")
            )

        self.assertTrue(graph_diagnostic.call_args_list)
        self.assertTrue(all(
            call.args[1] == "中性粒细胞百分数"
            for call in graph_diagnostic.call_args_list
        ))
        self.assertTrue(all(
            call.kwargs["graph_query"] == "中性粒细胞百分数"
            for call in graph_query.call_args_list
        ))
        self.assertEqual(metrics[0].metric_id, "中性粒细胞百分数")
        self.assertEqual(metrics[0].observation.abbreviation, "NEUT")
        diagnostics = {
            item["query"]: item for item in metrics[0].graph_diagnostics
        }
        self.assertEqual(set(diagnostics), {"NEUT", "NEUT%", "中性粒细胞百分数"})
        self.assertEqual(
            diagnostics["NEUT"]["resolved_query"], "中性粒细胞百分数"
        )
        self.assertEqual(
            diagnostics["NEUT%"]["resolved_query"], "中性粒细胞百分数"
        )
        self.assertEqual(diagnostics["NEUT%"]["status"], "matched")

    def test_bare_neut_uses_the_observation_name_for_graph_lookup(self):
        report = {
            "schema_version": "structured-report/v0.2",
            "observations": [{
                "raw_name": "中性粒细胞绝对数",
                "standard_name": "中性粒细胞绝对数",
                "abbreviation": "NEUT",
                "value": "7.61",
                "unit": "10^9/L",
                "reference_interval": {"lower": "1.8", "upper": "6.3"},
            }],
        }
        diagnostic = {
            "status": "matched",
            "match_mode": "normalized_variant",
            "matches": [{"name": "中性粒细胞绝对值"}],
        }
        with (
            patch(
                "medical_kg_sourceprep.report_pipeline.graph_query_diagnostic",
                return_value=diagnostic,
            ) as graph_diagnostic,
            patch(
                "medical_kg_sourceprep.report_pipeline.query_index_with_graph",
                return_value=([], {}),
            ) as graph_query,
        ):
            metrics = collect_metrics(
                report, Path("unused.sqlite"), knowledge_graph=Path("unused-graph.sqlite")
            )

        self.assertIn(
            (Path("unused-graph.sqlite"), "中性粒细胞绝对数"),
            [call.args for call in graph_diagnostic.call_args_list],
        )
        self.assertIn(
            "中性粒细胞绝对数",
            [call.kwargs["graph_query"] for call in graph_query.call_args_list],
        )
        diagnostics = {item["query"]: item for item in metrics[0].graph_diagnostics}
        self.assertEqual(diagnostics["NEUT"]["resolved_query"], "中性粒细胞绝对数")

    def test_graph_bound_evidence_displaces_unanchored_lexical_noise(self):
        report = {
            "schema_version": "structured-report/v0.2",
            "observations": [{
                "raw_name": "LYM%",
                "standard_name": "LYM%",
                "value": "19",
                "unit": "%",
                "reference_interval": {"lower": "20", "upper": "50"},
            }],
        }
        lexical_noise = {
            "chunk_id": "noise",
            "text": "unrelated B-cell evidence",
            "chunk_sha256": "noise-hash",
            "printed_page_number": 172,
            "source_pdf_page_number": 189,
            "score": 99.0,
            "retrieval_reason": "term_match",
        }
        graph_evidence = {
            "chunk_id": "chapter-01",
            "text": "淋巴细胞减少的第一章证据",
            "chunk_sha256": "graph-hash",
            "printed_page_number": 11,
            "source_pdf_page_number": 28,
            "score": 1.0,
            "retrieval_reason": "graph_path",
            "graph": {"status": "candidate-only", "matched_node_names": ["淋巴细胞百分数"]},
        }
        with (
            patch(
                "medical_kg_sourceprep.report_pipeline.graph_query_diagnostic",
                return_value={"status": "matched", "matches": []},
            ),
            patch(
                "medical_kg_sourceprep.report_pipeline.query_index_with_graph",
                return_value=([lexical_noise, graph_evidence], {}),
            ),
        ):
            metric = collect_metrics(
                report, Path("unused.sqlite"), knowledge_graph=Path("unused-graph.sqlite")
            )[0]

        self.assertEqual([item.chunk_id for item in metric.evidence], ["chapter-01"])

    def test_abnormal_only_retrieval_and_private_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            metrics = collect_metrics(_report(), index)
            self.assertLessEqual(len(metrics[0].evidence), 3)
            self.assertEqual(len({item.chunk_id for item in metrics[0].evidence}), len(metrics[0].evidence))
            self.assertEqual(metrics[1].evidence, ())
            transport = FakeTransport(_model())
            markdown = analyze_report(_report(), index, transport=transport)
            self.assertIn("evidence_id: `Edemo:chapter:page:0:0000`", markdown)
            self.assertIn("书内第 4 页；PDF 第 21 页", markdown)
            self.assertIn("](/source.pdf#page=21)", markdown)
            self.assertNotIn("private", markdown)
            self.assertEqual(transport.payload["model"], "deepseek-v4-flash")
            self.assertEqual(transport.payload["temperature"], 0)
            self.assertEqual(transport.payload["thinking"], {"type": "disabled"})
            self.assertEqual(transport.payload["response_format"], {"type": "json_object"})
            self.assertIn(
                '"abnormal_metric_ids":["alpha"]',
                transport.payload["messages"][1]["content"],
            )
            self.assertNotIn('"metric_id":"beta"', transport.payload["messages"][1]["content"])
            self.assertIn('"indeterminate":0', transport.payload["messages"][1]["content"])
            document = analyze_report_document(
                _report(), index, transport=FakeTransport(_model())
            ).to_dict()
            self.assertEqual(document["report"]["summary"], _model()["summary"])
            self.assertEqual(document["metrics"][0]["computed_flag"], "high")
            self.assertEqual(
                len({item["evidence_id"] for item in document["evidence"]}),
                len(document["evidence"]),
            )
            self.assertEqual(document["evidence"][0]["source_pdf_page_number"], 21)
            self.assertEqual(document["channels"]["mode"], "lexical")
            self.assertFalse(document["channels"]["graph"]["enabled"])

    def test_report_analysis_merges_graph_and_lexical_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _chunk_package(root)
            index = root / "evidence.sqlite"
            build_evidence_index(package, index, "2026-01-01T00:00:00Z")
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            first = manifest["chunks"][0]
            graph = _candidate_graph(
                root / "candidate.sqlite",
                first["chunk_id"],
                (package / first["chunk_path"]).read_text(encoding="utf-8"),
                node_name="alpha",
            )
            transport = FakeTransport(_model())
            document = analyze_report_document(
                _report(), index, knowledge_graph=graph, transport=transport
            ).to_dict()

        self.assertEqual(document["channels"]["mode"], "lexical+knowledge_graph")
        self.assertTrue(document["channels"]["graph"]["enabled"])
        self.assertEqual(document["channels"]["graph"]["status"], "candidate-only")
        self.assertGreaterEqual(document["channels"]["graph"]["evidence_count"], 1)
        graph_evidence = [item for item in document["evidence"] if "graph" in item]
        self.assertTrue(graph_evidence)
        self.assertIn("graph_path", graph_evidence[0]["retrieval_reason"])
        self.assertEqual(graph_evidence[0]["graph"]["matched_node_names"], ["alpha"])
        self.assertIn("第一章候选知识图谱辅助召回", document["markdown"])
        self.assertIn("candidate-only graph", transport.payload["messages"][1]["content"])

    def test_candidate_reasoning_path_is_exposed_without_executing_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _chunk_package(root)
            index = root / "evidence.sqlite"
            build_evidence_index(package, index, "2026-01-01T00:00:00Z")
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            first = manifest["chunks"][0]
            graph = _candidate_graph(
                root / "candidate.sqlite", first["chunk_id"],
                (package / first["chunk_path"]).read_text(encoding="utf-8"),
                node_name="alpha",
            )
            candidate = {
                "path_id": "candidate-path:test",
                "rule_id": "rule:test",
                "rule_name": "多指标候选关联",
                "status": "candidate-precondition-unverified",
                "graph_status": "candidate-only",
                "matched_metric_ids": ["alpha", "related"],
                "missing_inputs": [],
                "preconditions": [{"context": "前置状态", "op": "EQ", "value": "已确认"}],
                "triples": [{
                    "subject_name": "多指标候选关联",
                    "predicate": "RULE_HAS_SUBJECT",
                    "object_name": "alpha",
                }],
                "chunk_ids": [first["chunk_id"]],
            }
            transport = FakeTransport(_model())
            report = _report()
            report["metadata"]["patient_sex"] = "女"
            with patch(
                "medical_kg_sourceprep.report_pipeline.graph_reasoning_paths",
                return_value=GraphReasoningResult((candidate,), ()),
            ) as graph_paths:
                document = analyze_report_document(
                    report, index, knowledge_graph=graph, transport=transport
                ).to_dict()

        self.assertEqual(document["channels"]["graph"]["reasoning_path_count"], 1)
        self.assertTrue(document["channels"]["graph"]["reasoning_context_sent_to_model"])
        self.assertEqual(
            graph_paths.call_args.kwargs["context_facts"]["性别"]["value"], "女"
        )
        self.assertEqual(document["reasoning_paths"][0]["evidence_ids"], ["E" + first["chunk_id"]])
        self.assertIn("## 候选推理路径", document["markdown"])
        self.assertIn('"candidate_reasoning_paths"', transport.payload["messages"][1]["content"])
        self.assertIn("未经批准的候选条件求值", transport.payload["messages"][1]["content"])

    def test_final_prompt_limits_multi_metric_conclusions_to_reported_rule_paths(self):
        prompt = _prompt((), ({
            "graph_status": "final",
            "status": "final-case-match",
        },))

        self.assertIn("多指标规则结论的唯一授权来源", prompt)
        self.assertIn("不得在 association_analysis 中据此拼接疾病", prompt)

    def test_final_report_prunes_unreferenced_evidence_and_renders_association_citations(self):
        result = _model()
        result["association_analysis"] = {
            "analysis": "两个异常指标存在书内关联。",
            "evidence_ids": ["Edemo:chapter:page:0:0000"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            document = analyze_report_document(
                _report(), index, transport=FakeTransport(result)
            ).to_dict()

        self.assertEqual(
            [item["evidence_id"] for item in document["evidence"]],
            ["Edemo:chapter:page:0:0000"],
        )
        self.assertIn(
            "## 关联分析\n\n两个异常指标存在书内关联。\n\n证据：",
            document["markdown"],
        )

    def test_unknown_evidence_fails_closed(self):
        result = _model()
        result["abnormal_analyses"][0]["evidence_ids"] = ["Eforged"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            with self.assertRaisesRegex(ReportPipelineError, "unknown evidence"):
                analyze_report(_report(), index, transport=FakeTransport(result))

    def test_related_abnormal_metrics_may_share_retrieved_evidence(self):
        result = {
            "summary": "两个相关指标异常。",
            "abnormal_analyses": [
                {"metric_id": "a", "analysis": "书内关联解释。", "evidence_ids": ["E2"]},
                {"metric_id": "b", "analysis": "书内关联解释。", "evidence_ids": ["E1"]},
            ],
            "association_analysis": {"analysis": "书内关联解释。", "evidence_ids": ["E1", "E2"]},
            "attention_suggestions": [],
            "insufficient_evidence": [],
        }
        validated = validate_model_result(
            result, {"E1", "E2"}, {"a", "b"}, {"a", "b"}
        )
        self.assertIs(validated, result)

    def test_missing_citation_fails_closed(self):
        result = _model()
        result["abnormal_analyses"][0]["evidence_ids"] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            with self.assertRaisesRegex(ReportPipelineError, "no evidence citations"):
                analyze_report(_report(), index, transport=FakeTransport(result))

    def test_non_abnormal_metric_in_abnormal_analysis_fails_closed(self):
        result = _model()
        result["abnormal_analyses"].append({
            "metric_id": "beta",
            "analysis": "不应进入异常分析。",
            "evidence_ids": [],
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            with self.assertRaisesRegex(ReportPipelineError, "non-abnormal"):
                analyze_report(_report(), index, transport=FakeTransport(result))

    def test_indeterminate_metric_is_programmatically_marked_for_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            transport = FakeTransport(_model())
            document = analyze_report_document(
                _report_with_indeterminate(), index, transport=transport
            ).to_dict()
        gamma = document["metrics"][2]
        self.assertIsNone(gamma["computed_flag"])
        self.assertEqual(gamma["validation_issues"][0]["code"], "invalid_interval")
        self.assertEqual(gamma["validation_issues"][0]["label"], "缺少有效参考区间")
        self.assertIn("## 数据待核对", document["markdown"])
        self.assertIn("- gamma：缺少有效参考区间", document["markdown"])
        self.assertNotIn('"metric_id":"gamma"', transport.payload["messages"][1]["content"])

    def test_explicit_insufficient_evidence_may_decline_all_retrieved_hits(self):
        result = _model()
        result["abnormal_analyses"][0] = {
            "metric_id": "alpha",
            "analysis": "知识库证据不足，无法作出书内解释。",
            "evidence_ids": [],
        }
        result["attention_suggestions"] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            markdown = analyze_report(_report(), index, transport=FakeTransport(result))
        self.assertIn("知识库证据不足", markdown)
        self.assertNotIn("证据：[", markdown)
        self.assertIn("## 书内证据", markdown)

    def test_missing_key_is_rejected_before_transport(self):
        with self.assertRaisesRegex(ReportPipelineError, "DEEPSEEK_API_KEY"):
            from medical_kg_sourceprep.report_pipeline import DeepSeekTransport
            DeepSeekTransport(" ")


if __name__ == "__main__":
    unittest.main()
