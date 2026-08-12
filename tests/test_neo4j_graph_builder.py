import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jRelationship

from medical_kg_sourceprep.extraction import neo4j_graph_builder as graph_builder
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk


class Neo4jGraphBuilderTests(unittest.TestCase):
    def test_missing_deepseek_key_fails_without_loading_local_environment(self):
        with self.assertRaisesRegex(graph_builder.GraphBuilderConfigurationError, "DEEPSEEK_API_KEY"):
            graph_builder.load_deepseek_api_key(env={})

    def test_factory_uses_official_deepseek_contract_and_no_proxy_client(self):
        async_client = object()
        llm = object()
        with (
            patch.object(graph_builder.httpx, "AsyncClient", return_value=async_client) as client_factory,
            patch.object(graph_builder, "OpenAILLM", return_value=llm) as llm_factory,
        ):
            result = graph_builder.create_deepseek_graph_builder(
                env={"DEEPSEEK_API_KEY": "test-key", "HTTPS_PROXY": "http://proxy.invalid"}
            )

        self.assertIs(result.llm, llm)
        self.assertIs(result.http_client, async_client)
        self.assertFalse(client_factory.call_args.kwargs["trust_env"])
        self.assertEqual(
            llm_factory.call_args.kwargs,
            {
                "model_name": "deepseek-v4-flash",
                "model_params": {
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "extra_body": {"thinking": {"type": "disabled"}},
                },
                "api_key": "test-key",
                "base_url": "https://api.deepseek.com",
                "http_client": async_client,
            },
        )

    def test_smoke_parses_graph_builder_json_without_writing_any_database(self):
        class FakeLLM:
            async def ainvoke(self, prompt):
                self.prompt = prompt
                return SimpleNamespace(
                    content=(
                        '{"nodes":[{"id":"indicator","label":"LabIndicator",'
                        '"properties":{"name":"血清铁"}}],"relationships":[]}'
                    )
                )

        client = graph_builder.DeepSeekGraphBuilderClient(
            llm=FakeLLM(), http_client=SimpleNamespace()
        )
        summary = asyncio.run(graph_builder.run_smoke(client))

        self.assertEqual(summary["model"], "deepseek-v4-flash")
        self.assertEqual(summary["node_count"], 1)
        self.assertEqual(summary["relationship_count"], 0)

    def test_close_delegates_to_graph_rag_llm(self):
        class FakeLLM:
            def __init__(self):
                self.closed = False

            async def aclose(self):
                self.closed = True

        llm = FakeLLM()
        client = graph_builder.DeepSeekGraphBuilderClient(llm=llm, http_client=SimpleNamespace())
        asyncio.run(client.aclose())
        self.assertTrue(llm.closed)

    def test_response_adapter_normalizes_supported_response_shapes(self):
        class FakeLLM:
            async def ainvoke(self, prompt):
                return SimpleNamespace(
                    content=json.dumps({
                        "nodes": [{
                            "id": "table-state",
                            "label": "IndicatorState",
                            "properties": {
                                "table_state_evidence_json": {
                                    "header_exact_quote": "血清铁",
                                    "row_exact_quote": "↓",
                                },
                            },
                        }],
                        "edges": [{
                            "type": "RULE_INPUT",
                            "properties": {"exact_quote": None, "relation_cue": None},
                        }],
                    }),
                    usage=SimpleNamespace(request_tokens=100, response_tokens=20, total_tokens=120),
                )

        adapter = graph_builder._GraphRagIdCompletingLLM(FakeLLM())
        result = asyncio.run(adapter.ainvoke("prompt"))
        payload = json.loads(result.content)
        self.assertNotIn("edges", payload)
        self.assertEqual(
            json.loads(payload["nodes"][0]["properties"]["table_state_evidence_json"]),
            {"header_exact_quote": "血清铁", "row_exact_quote": "↓"},
        )
        self.assertEqual(adapter.last_response_diagnostic["usage"], {
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        })
        self.assertEqual(result.usage.total_tokens, 120)
        relationship = payload["relationships"][0]
        self.assertEqual(relationship["properties"], {})

    def test_model_table_state_uses_raw_dual_anchors_without_local_semantic_parser(self):
        header = "<tr><th>血清铁</th><th>TIBC</th><th>结论</th></tr>"
        row = "<tr><td>↓</td><td>↑</td><td>缺铁性贫血</td></tr>"
        text = f"<table>{header}{row}</table>"
        chunk = EvidenceChunk("synthetic:table-state", text, hashlib.sha256(text.encode()).hexdigest())
        graph = Neo4jGraph(nodes=[
            {"id": "indicator-serum-iron", "label": "LabIndicator", "properties": {
                "mention": "血清铁", "canonical_name_candidate": "血清铁", "exact_quote": header,
            }},
            {"id": "indicator-tibc", "label": "LabIndicator", "properties": {
                "mention": "TIBC", "canonical_name_candidate": "TIBC", "exact_quote": header,
            }},
            {"id": "state-serum-iron-low", "label": "IndicatorState", "properties": {
                "mention": "血清铁降低", "canonical_name_candidate": "血清铁降低",
                "bound_indicator_mention": "血清铁",
                "table_state_evidence_json": json.dumps({
                    "header_exact_quote": header,
                    "row_exact_quote": row,
                }, ensure_ascii=False),
            }},
            {"id": "state-tibc-high", "label": "IndicatorState", "properties": {
                "mention": "TIBC升高", "canonical_name_candidate": "TIBC升高",
                "bound_indicator_mention": "TIBC",
                "table_state_evidence_json": json.dumps({
                    "header_exact_quote": header,
                    "row_exact_quote": row,
                }, ensure_ascii=False),
            }},
        ])

        nodes, holds = graph_builder.normalize_candidate_nodes(
            graph, chunk=chunk, schema=graph_builder.load_candidate_graph_schema()
        )

        self.assertEqual(holds, [])
        states = [node for node in nodes if node["entity_type"] == "IndicatorState"]
        self.assertEqual({node["mention"] for node in states}, {"血清铁降低", "TIBC升高"})
        self.assertTrue(all(node["bound_indicator_candidate_key"] for node in states))
        self.assertTrue(all(
            [ref["role"] for ref in node["table_state_evidence_refs"]] == ["table_header", "table_row"]
            for node in states
        ))

    def test_runner_passes_raw_table_to_model_without_derived_view(self):
        header = "<tr><th>血清铁</th></tr>"
        row = "<tr><td>↓</td></tr>"
        text = f"<table>{header}{row}</table>"
        chunk = EvidenceChunk("synthetic:table-view", text, hashlib.sha256(text.encode()).hexdigest())
        prompts = []

        class FakeLLM:
            async def ainvoke(self, prompt):
                prompts.append(prompt)
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": []}))

        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk,
                schema=graph_builder.load_candidate_graph_schema(),
                output_dir=Path(temporary) / "candidate",
                source_manifest_sha256="a" * 64,
            ))

        self.assertEqual(len(prompts), 3)
        self.assertIn(text, prompts[0])
        self.assertNotIn("[DERIVED_TABLE_STATE_VIEW]", prompts[0])
        self.assertNotIn("[DERIVED_TABLE_STATE_VIEW]", prompts[1])
        self.assertNotIn("[DERIVED_TABLE_STATE_VIEW]", prompts[2])

    def test_candidate_graph_validates_nodes_relations_and_writes_no_raw_response(self):
        text = "血清铁降低。生长发育需要量多导致缺铁性贫血。"
        chunk = EvidenceChunk(
            "synthetic:0001",
            text,
            hashlib.sha256(text.encode()).hexdigest(),
        )
        indicator_ref = graph_builder._source_ref(chunk, "血清铁", "血清铁降低。")
        context_ref = graph_builder._source_ref(chunk, "生长发育需要量多", "生长发育需要量多导致缺铁性贫血。")
        disease_ref = graph_builder._source_ref(chunk, "缺铁性贫血", "生长发育需要量多导致缺铁性贫血。")
        indicator_key = graph_builder._candidate_key("LabIndicator", "血清铁", indicator_ref)
        context_key = graph_builder._candidate_key("ClinicalContext", "生长发育需要量多", context_ref)
        disease_key = graph_builder._candidate_key("Disease", "缺铁性贫血", disease_ref)

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"nodes": [
                        {"label": "LabIndicator", "properties": {
                            "mention": "血清铁", "canonical_name_candidate": "血清铁", "exact_quote": "血清铁降低。"
                        }},
                        {"label": "IndicatorState", "properties": {
                            "mention": "血清铁降低", "canonical_name_candidate": "血清铁降低", "exact_quote": "血清铁降低。",
                            "bound_indicator_mention": "血清铁"
                        }},
                        {"label": "ClinicalContext", "properties": {
                            "mention": "生长发育需要量多", "canonical_name_candidate": "生长发育需要量多", "exact_quote": "生长发育需要量多导致缺铁性贫血。"
                        }},
                        {"label": "Disease", "properties": {
                            "mention": "缺铁性贫血", "canonical_name_candidate": "缺铁性贫血", "exact_quote": "生长发育需要量多导致缺铁性贫血。"
                        }}
                    ], "relationships": []}, ensure_ascii=False))
                if self.calls == 2:
                    return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": []}))
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": [{
                    "type": "CAUSES", "start_node_id": context_key, "end_node_id": disease_key,
                    "exact_quote": "生长发育需要量多导致缺铁性贫血。", "relation_cue": "导致"
                }]}, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            schema_path = Path(temporary) / "candidate-schema.json"
            schema_value = graph_builder.load_candidate_graph_schema()
            schema_value["schema_version"] = "test-custom-schema"
            schema_path.write_text(json.dumps(schema_value, ensure_ascii=False), encoding="utf-8")
            schema = graph_builder.load_candidate_graph_schema(schema_path)
            schema_sha256 = hashlib.sha256(schema_path.read_bytes()).hexdigest()
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk,
                schema=schema,
                schema_path=schema_path,
                output_dir=output,
                source_manifest_sha256="a" * 64,
            ))
            graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
            queue = json.loads((output / "review-queue.json").read_text(encoding="utf-8"))
            judge_queue = json.loads((output / "judge-queue.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["node_count"], 4)
        self.assertEqual(summary["relationship_count"], 2, queue)
        self.assertEqual(summary["hold_count"], 0, summary)
        self.assertEqual({item["candidate_key"] for item in graph["nodes"]}, {
            indicator_key,
            context_key,
            disease_key,
            next(item["candidate_key"] for item in graph["nodes"] if item["entity_type"] == "IndicatorState"),
        })
        self.assertEqual({item["relation_type"] for item in graph["relationships"]}, {"HAS_STATE", "CAUSES"})
        self.assertEqual(manifest["configuration"]["database_write"], False)
        self.assertEqual(
            manifest["input"]["candidate_schema_sha256"],
            schema_sha256,
        )
        self.assertNotIn("raw_response", json.dumps(graph, ensure_ascii=False))
        self.assertEqual(judge_queue["counts"], {"pending": 0})
        self.assertIn("judge-queue.json", manifest["artifacts"])
        self.assertEqual(manifest["counts"]["judge_pending"], 0)

    def test_candidate_graph_routes_invalid_nodes_and_relations_to_hold(self):
        text = "HGB 参考区间。普通疾病。甲和乙导致丙。"
        chunk = EvidenceChunk("synthetic:0002", text, hashlib.sha256(text.encode()).hexdigest())
        schema = graph_builder.load_candidate_graph_schema()

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"nodes": [
                        {"id": "range", "label": "LabIndicator", "properties": {
                            "mention": "参考区间", "canonical_name_candidate": "参考区间", "exact_quote": "HGB 参考区间。"
                        }},
                        {"id": "claim", "label": "Claim", "properties": {
                            "mention": "HGB", "canonical_name_candidate": "HGB", "exact_quote": "HGB 参考区间。"
                        }},
                        {"id": "state", "label": "IndicatorState", "properties": {
                            "mention": "丙", "canonical_name_candidate": "丙", "exact_quote": "甲和乙导致丙。",
                            "bound_indicator_mention": "不存在"
                        }}
                    ], "relationships": []}, ensure_ascii=False))
                if self.calls == 2:
                    return SimpleNamespace(content=json.dumps({"nodes": [{
                        "id": "not-a-rule", "label": "RuleDefinition", "properties": {
                            "mention": "普通疾病", "canonical_name_candidate": "普通疾病", "exact_quote": "普通疾病。"
                        }
                    }], "relationships": []}, ensure_ascii=False))
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": [{
                    "type": "CAUSES", "start_node_id": "unknown", "end_node_id": "other",
                    "properties": {"exact_quote": "甲和乙导致丙。", "relation_cue": "导致"}
                }]}, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk,
                schema=schema,
                output_dir=output,
                source_manifest_sha256="b" * 64,
            ))
            queue = json.loads((output / "review-queue.json").read_text(encoding="utf-8"))
            graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
            judge_queue = json.loads((output / "judge-queue.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["node_count"], 2)
        self.assertEqual(summary["relationship_count"], 0)
        self.assertEqual(queue["status"], "candidate-only")
        self.assertEqual(queue["publication_status"], "HOLD")
        state = next(item for item in graph["nodes"] if item["entity_type"] == "IndicatorState")
        self.assertEqual(state["extraction_status"], "PARTIAL")
        self.assertEqual(state["review_status"], "REVIEW_REQUIRED")
        self.assertEqual(queue["counts"], {"review_required": 3, "rejected": 1})
        self.assertEqual({item["reason_code"] for item in queue["items"]}, {
            "entity_type_not_enabled_for_trial",
            "rule_definition_uses_business_fields",
            "state_indicator_binding_not_unique",
            "relation_endpoint_not_from_frozen_catalog",
        })
        self.assertEqual(judge_queue["counts"], {"pending": 2})
        self.assertTrue(all(item["judge_status"] == "PENDING" for item in judge_queue["items"]))

    def test_candidate_graph_preserves_dedicated_rule_definition_inputs_and_outputs(self):
        sentence = "贫血状态且MCV正常且RDW增大时，提示缺铁性贫血。"
        chunk = EvidenceChunk("synthetic:rule", sentence, hashlib.sha256(sentence.encode()).hexdigest())
        schema = graph_builder.load_candidate_graph_schema()
        entities = [
            ("ClinicalContext", "贫血状态", {}),
            ("LabIndicator", "MCV", {}),
            ("IndicatorState", "MCV正常", {"bound_indicator_mention": "MCV"}),
            ("LabIndicator", "RDW", {}),
            ("IndicatorState", "RDW增大", {"bound_indicator_mention": "RDW"}),
            ("Disease", "缺铁性贫血", {}),
        ]
        refs = {mention: graph_builder._source_ref(chunk, mention, sentence) for _type, mention, _extra in entities}
        keys = {mention: graph_builder._candidate_key(entity_type, mention, refs[mention])
                for entity_type, mention, _extra in entities}
        evidence = [{"role": "condition_sentence", "exact_quote": sentence}]
        rule_key = graph_builder._rule_candidate_key(
            chunk=chunk, rule_stage="GRAPH_COMPOSITE",
            rule_expression="缺铁性贫血=贫血形态判断(贫血状态,MCV正常,RDW增大)",
            rule_evidence_refs=[graph_builder._rule_evidence_ref(
                chunk, role="condition_sentence", exact_quote=sentence
            )],
        )

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"nodes": [
                        {"label": label, "properties": {
                            "mention": mention, "canonical_name_candidate": mention,
                            "exact_quote": sentence, **extra,
                        }} for label, mention, extra in entities
                    ], "relationships": []}, ensure_ascii=False))
                if self.calls == 2:
                    return SimpleNamespace(content=json.dumps({"nodes": [{"label": "RuleDefinition", "properties": {
                        "rule_stage_candidate": "GRAPH_COMPOSITE",
                        "rule_expression": "缺铁性贫血 = 贫血形态判断(贫血状态, MCV正常, RDW增大)",
                        "rule_name": "贫血形态判断",
                        "rule_evidence_json": json.dumps(evidence, ensure_ascii=False),
                    }}], "relationships": []}, ensure_ascii=False))
                if self.calls == 3:
                    return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": []}))
                edges = [
                    ("RULE_INPUT", keys["贫血状态"], rule_key),
                    ("RULE_INPUT", keys["MCV正常"], rule_key),
                    ("RULE_INPUT", keys["RDW增大"], rule_key),
                    ("RULE_OUTPUT", rule_key, keys["缺铁性贫血"]),
                ]
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": [
                    {"type": relation_type, "start_node_id": source, "end_node_id": target,
                     "properties": {"rule_evidence_role": "condition_sentence"}}
                    for relation_type, source, target in edges
                ]}, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk, schema=schema, output_dir=output, source_manifest_sha256="c" * 64,
            ))
            graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))

        rule = next(item for item in graph["nodes"] if item["entity_type"] == "RuleDefinition")
        self.assertEqual(rule["candidate_key"], rule["rule_candidate_key"])
        self.assertEqual(rule["rule_expression"], "缺铁性贫血=贫血形态判断(贫血状态,MCV正常,RDW增大)")
        self.assertNotIn("mention", rule)
        self.assertEqual(summary["hold_count"], 0)
        self.assertEqual({item["relation_type"] for item in summary["relationships"]},
                         {"HAS_STATE", "RULE_INPUT", "RULE_OUTPUT"})
        self.assertTrue(all(item["extraction_status"] == "VALID" for item in graph["nodes"]))
        self.assertTrue(all(item["extraction_status"] == "VALID" for item in graph["relationships"]))

    def test_model_can_extract_table_rule_with_verbatim_business_endpoints_and_role_evidence(self):
        header = "<tr><td>血清铁</td><td>TIBC</td><td>原因</td></tr>"
        row = "<tr><td>↓</td><td>↑</td><td>缺铁性贫血,铁吸收不良</td></tr>"
        text = f"<table>{header}{row}</table>"
        chunk = EvidenceChunk("synthetic:table", text, hashlib.sha256(text.encode()).hexdigest())
        schema = graph_builder.load_candidate_graph_schema()
        entities = [("LabIndicator", "血清铁"), ("LabIndicator", "TIBC"),
                    ("Disease", "缺铁性贫血"), ("ClinicalContext", "铁吸收不良")]
        keys = {
            mention: graph_builder._candidate_key(entity_type, mention, graph_builder._source_ref(chunk, mention, text))
            for entity_type, mention in entities
        }
        refs = [graph_builder._rule_evidence_ref(chunk, role="table_header", exact_quote=header),
                graph_builder._rule_evidence_ref(chunk, role="table_row", exact_quote=row)]
        expression = "[缺铁性贫血,铁吸收不良]=血清铁与TIBC联合检测(血清铁,TIBC)"
        rule_key = graph_builder._rule_candidate_key(
            chunk=chunk, rule_stage="GRAPH_COMPOSITE", rule_expression=expression, rule_evidence_refs=refs
        )

        class FakeLLM:
            def __init__(self): self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    nodes = [{"label": entity_type, "properties": {
                        "mention": mention, "canonical_name_candidate": mention, "exact_quote": text,
                    }} for entity_type, mention in entities]
                    return SimpleNamespace(content=json.dumps({"nodes": nodes, "relationships": []}, ensure_ascii=False))
                if self.calls == 2:
                    return SimpleNamespace(content=json.dumps({"nodes": [{"label": "RuleDefinition", "properties": {
                        "rule_stage_candidate": "GRAPH_COMPOSITE", "rule_expression": expression,
                        "rule_name": "血清铁与TIBC联合检测",
                        "rule_evidence_json": json.dumps([
                            {"role": "table_header", "exact_quote": header},
                            {"role": "table_row", "exact_quote": row},
                        ], ensure_ascii=False),
                    }}], "relationships": []}, ensure_ascii=False))
                if self.calls == 3:
                    return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": []}))
                edges = [("RULE_INPUT", keys["血清铁"], rule_key, "table_header"),
                         ("RULE_INPUT", keys["TIBC"], rule_key, "table_header"),
                         ("RULE_OUTPUT", rule_key, keys["缺铁性贫血"], "table_row"),
                         ("RULE_OUTPUT", rule_key, keys["铁吸收不良"], "table_row")]
                return SimpleNamespace(content=json.dumps({"nodes": [], "edges": [
                    {"type": kind, "start_node_id": source, "end_node_id": target,
                     "properties": {"rule_evidence_role": role}}
                    for kind, source, target, role in edges
                ]}, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk, schema=schema, output_dir=output, source_manifest_sha256="d" * 64,
            ))
            graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))

        rule = next(item for item in graph["nodes"] if item["entity_type"] == "RuleDefinition")
        self.assertEqual([item["role"] for item in rule["rule_evidence_refs"]], ["table_header", "table_row"])
        self.assertEqual(rule["rule_expression"], expression)
        self.assertEqual(summary["relationship_count"], 4)
        self.assertEqual(summary["hold_count"], 0)

    def test_formula_rules_keep_only_frozen_business_endpoints(self):
        ptr_formula = "PTR = PT / 正常人血浆 PT。"
        inr_formula = "国际标准化比值 (INR) = PTR × ISI。"
        text = f"{ptr_formula}{inr_formula}"
        chunk = EvidenceChunk("synthetic:formula", text, hashlib.sha256(text.encode()).hexdigest())
        schema = graph_builder.load_candidate_graph_schema()
        entities = [("LabIndicator", "PT"), ("LabIndicator", "PTR"), ("LabIndicator", "INR")]
        keys = {mention: graph_builder._candidate_key(
            entity_type, mention, graph_builder._source_refs_for_mention(chunk, mention)[0]
        ) for entity_type, mention in entities}
        ptr_expression = "PTR=PTR计算(PT)"
        inr_expression = "INR=国际标准化比值计算(PTR)"
        ptr_ref = graph_builder._rule_evidence_ref(chunk, role="formula", exact_quote=ptr_formula)
        inr_ref = graph_builder._rule_evidence_ref(chunk, role="formula", exact_quote=inr_formula)
        ptr_rule_key = graph_builder._rule_candidate_key(
            chunk=chunk, rule_stage="PREPROCESS", rule_expression=ptr_expression, rule_evidence_refs=[ptr_ref]
        )
        inr_rule_key = graph_builder._rule_candidate_key(
            chunk=chunk, rule_stage="PREPROCESS", rule_expression=inr_expression, rule_evidence_refs=[inr_ref]
        )

        class FakeLLM:
            def __init__(self): self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"nodes": [
                        {"label": entity_type, "properties": {
                            "mention": mention, "extraction_reason": "原文明示的检验指标。",
                        }} for entity_type, mention in entities
                    ], "relationships": []}, ensure_ascii=False))
                if self.calls == 2:
                    return SimpleNamespace(content=json.dumps({"nodes": [
                        {"label": "RuleDefinition", "properties": {
                            "rule_stage_candidate": "PREPROCESS", "rule_expression": ptr_expression,
                            "rule_name": "PTR计算",
                            "rule_evidence_json": json.dumps([{"role": "formula", "exact_quote": ptr_formula}], ensure_ascii=False),
                        }},
                        {"label": "RuleDefinition", "properties": {
                            "rule_stage_candidate": "PREPROCESS", "rule_expression": inr_expression,
                            "rule_name": "国际标准化比值计算",
                            "rule_evidence_json": json.dumps([{"role": "formula", "exact_quote": inr_formula}], ensure_ascii=False),
                        }},
                    ], "relationships": []}, ensure_ascii=False))
                if self.calls == 3:
                    return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": []}))
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": [
                    {"type": "RULE_INPUT", "start_node_id": keys["PT"], "end_node_id": ptr_rule_key,
                     "properties": {"rule_evidence_role": "formula"}},
                    {"type": "RULE_OUTPUT", "start_node_id": ptr_rule_key, "end_node_id": keys["PTR"],
                     "properties": {"rule_evidence_role": "formula"}},
                    {"type": "RULE_INPUT", "start_node_id": keys["PTR"], "end_node_id": inr_rule_key,
                     "properties": {"rule_evidence_role": "formula"}},
                    {"type": "RULE_OUTPUT", "start_node_id": inr_rule_key, "end_node_id": keys["INR"],
                     "properties": {"rule_evidence_role": "formula"}},
                ]}, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk, schema=schema, output_dir=output, source_manifest_sha256="e" * 64,
            ))
            queue = json.loads((output / "review-queue.json").read_text(encoding="utf-8"))
            graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["hold_count"], 0, queue)
        self.assertEqual(summary["relationship_count"], 4)
        self.assertEqual({item["mention"] for item in graph["nodes"] if item["entity_type"] == "LabIndicator"},
                         {"PT", "PTR", "INR"})
        self.assertFalse(any(item.get("mention") == "ISI" for item in graph["nodes"]))

    def test_conjunction_without_multiple_states_is_retained_for_later_evaluation(self):
        text = "甲和乙导致丙。"
        chunk = EvidenceChunk("synthetic:0003", text, hashlib.sha256(text.encode()).hexdigest())
        source_ref = graph_builder._source_ref(chunk, "甲", text)
        source = {
            "candidate_key": "candidate:context",
            "entity_type": "ClinicalContext",
            "mention": "甲",
            "canonical_name_candidate": "甲",
            "source_ref": source_ref,
        }
        target = {
            "candidate_key": "candidate:disease",
            "entity_type": "Disease",
            "mention": "丙",
            "canonical_name_candidate": "丙",
            "source_ref": graph_builder._source_ref(chunk, "丙", text),
        }
        graph = Neo4jGraph(relationships=[Neo4jRelationship(
            start_node_id=f"{chunk.chunk_id}:{source['candidate_key']}",
            end_node_id=f"{chunk.chunk_id}:{target['candidate_key']}",
            type="CAUSES",
            properties={"exact_quote": text, "relation_cue": "导致"},
        )])
        relations, holds = graph_builder.normalize_candidate_relationships(
            graph,
            chunk=chunk,
            schema=graph_builder.load_candidate_graph_schema(),
            nodes=[source, target],
        )

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["relation_type"], "CAUSES")
        self.assertEqual(relations[0]["extraction_status"], "VALID")
        self.assertEqual(holds, [])

    def test_joint_serum_iron_and_tibc_direct_edge_enters_hold(self):
        text = "血清铁降低且TIBC增高时，提示缺铁性贫血。"
        chunk = EvidenceChunk("synthetic:joint", text, hashlib.sha256(text.encode()).hexdigest())
        source_ref = graph_builder._source_ref(chunk, "血清铁降低", text)
        nodes = []
        for entity_type, mention in (("IndicatorState", "血清铁降低"), ("IndicatorState", "TIBC增高"), ("Disease", "缺铁性贫血")):
            ref = graph_builder._source_ref(chunk, mention, text)
            nodes.append({
                "candidate_key": graph_builder._candidate_key(entity_type, mention, ref),
                "entity_type": entity_type,
                "mention": mention,
                "canonical_name_candidate": mention,
                "source_ref": ref,
                **({"bound_indicator_candidate_key": f"indicator:{mention}"} if entity_type == "IndicatorState" else {}),
            })
        state_key = nodes[0]["candidate_key"]
        disease_key = nodes[2]["candidate_key"]
        graph = Neo4jGraph(relationships=[Neo4jRelationship(
            start_node_id=f"{chunk.chunk_id}:{state_key}",
            end_node_id=f"{chunk.chunk_id}:{disease_key}",
            type="INDICATES",
            properties={"exact_quote": text, "relation_cue": "提示"},
        )])
        relations, holds = graph_builder.normalize_candidate_relationships(
            graph, chunk=chunk, schema=graph_builder.load_candidate_graph_schema(), nodes=nodes
        )
        relation = next(item for item in relations if item["relation_type"] == "INDICATES")
        self.assertEqual([item["relation_type"] for item in relations], ["HAS_STATE", "HAS_STATE", "INDICATES"])
        self.assertEqual(relation["extraction_status"], "PARTIAL")
        self.assertIn("RELATION_MAY_BE_JOINT_CONDITION", relation["warnings"])
        self.assertEqual([item["reason_code"] for item in holds], ["relation_may_be_joint_condition"])

    def test_relation_endpoint_type_mismatch_is_retained_as_partial(self):
        text = "血清铁提示缺铁性贫血。"
        chunk = EvidenceChunk("synthetic:type-mismatch", text, hashlib.sha256(text.encode()).hexdigest())
        source_ref = graph_builder._source_ref(chunk, "血清铁", text)
        target_ref = graph_builder._source_ref(chunk, "缺铁性贫血", text)
        source = {
            "candidate_key": graph_builder._candidate_key("Disease", "血清铁", source_ref),
            "entity_type": "Disease", "mention": "血清铁", "canonical_name_candidate": "血清铁",
            "source_ref": source_ref,
        }
        target = {
            "candidate_key": graph_builder._candidate_key("Disease", "缺铁性贫血", target_ref),
            "entity_type": "Disease", "mention": "缺铁性贫血", "canonical_name_candidate": "缺铁性贫血",
            "source_ref": target_ref,
        }
        graph = Neo4jGraph(relationships=[Neo4jRelationship(
            start_node_id=f"{chunk.chunk_id}:{source['candidate_key']}",
            end_node_id=f"{chunk.chunk_id}:{target['candidate_key']}", type="INDICATES",
            properties={"exact_quote": text, "relation_cue": "提示"},
        )])

        relations, holds = graph_builder.normalize_candidate_relationships(
            graph, chunk=chunk, schema=graph_builder.load_candidate_graph_schema(), nodes=[source, target]
        )

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["extraction_status"], "PARTIAL")
        self.assertIn("RELATION_ENDPOINT_TYPE_INVALID", relations[0]["warnings"])
        self.assertEqual(holds[0]["status"], "REVIEW_REQUIRED")

    def test_verbatim_relation_cue_is_retained_without_local_semantic_grading(self):
        text = "血清铁降低反映铁储备不足。"
        chunk = EvidenceChunk("synthetic:unmapped-cue", text, hashlib.sha256(text.encode()).hexdigest())
        source_ref = graph_builder._source_ref(chunk, "血清铁降低", text)
        target_ref = graph_builder._source_ref(chunk, "铁储备不足", text)
        source = {
            "candidate_key": graph_builder._candidate_key("IndicatorState", "血清铁降低", source_ref),
            "entity_type": "IndicatorState",
            "mention": "血清铁降低",
            "canonical_name_candidate": "血清铁降低",
            "source_ref": source_ref,
            "bound_indicator_candidate_key": "candidate:indicator",
        }
        target = {
            "candidate_key": graph_builder._candidate_key("ClinicalContext", "铁储备不足", target_ref),
            "entity_type": "ClinicalContext",
            "mention": "铁储备不足",
            "canonical_name_candidate": "铁储备不足",
            "source_ref": target_ref,
        }
        graph = Neo4jGraph(relationships=[Neo4jRelationship(
            start_node_id=f"{chunk.chunk_id}:{source['candidate_key']}",
            end_node_id=f"{chunk.chunk_id}:{target['candidate_key']}",
            type="INDICATES",
            properties={"exact_quote": text, "relation_cue": "反映"},
        )])

        relations, holds = graph_builder.normalize_candidate_relationships(
            graph,
            chunk=chunk,
            schema=graph_builder.load_candidate_graph_schema(),
            nodes=[source, target],
        )

        relation = next(item for item in relations if item["relation_type"] == "INDICATES")
        self.assertEqual(relation["relation_cue"], "反映")
        self.assertEqual(relation["extraction_status"], "VALID")
        self.assertEqual(relation["review_status"], "PENDING")
        self.assertNotIn("warnings", relation)
        self.assertEqual(holds, [])

    def test_relation_type_is_retained_for_later_semantic_evaluation(self):
        text = "甲导致乙。"
        chunk = EvidenceChunk("synthetic:cue-mismatch", text, hashlib.sha256(text.encode()).hexdigest())
        source_ref = graph_builder._source_ref(chunk, "甲", text)
        target_ref = graph_builder._source_ref(chunk, "乙", text)
        source = {
            "candidate_key": graph_builder._candidate_key("ClinicalContext", "甲", source_ref),
            "entity_type": "ClinicalContext",
            "mention": "甲",
            "canonical_name_candidate": "甲",
            "source_ref": source_ref,
        }
        target = {
            "candidate_key": graph_builder._candidate_key("Disease", "乙", target_ref),
            "entity_type": "Disease",
            "mention": "乙",
            "canonical_name_candidate": "乙",
            "source_ref": target_ref,
        }
        graph = Neo4jGraph(relationships=[Neo4jRelationship(
            start_node_id=f"{chunk.chunk_id}:{source['candidate_key']}",
            end_node_id=f"{chunk.chunk_id}:{target['candidate_key']}",
            type="INDICATES",
            properties={"exact_quote": text, "relation_cue": "导致"},
        )])

        relations, holds = graph_builder.normalize_candidate_relationships(
            graph,
            chunk=chunk,
            schema=graph_builder.load_candidate_graph_schema(),
            nodes=[source, target],
        )

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["relation_type"], "INDICATES")
        self.assertEqual(relations[0]["extraction_status"], "VALID")
        self.assertEqual(relations[0]["review_status"], "PENDING")
        self.assertEqual(holds, [])

    def test_rule_evidence_keeps_replayable_roles_for_later_evaluation(self):
        text = "血清铁与TIBC联合检测。"
        chunk = EvidenceChunk("synthetic:holds", text, hashlib.sha256(text.encode()).hexdigest())
        graph = Neo4jGraph(nodes=[
            {"id": "missing-table-row", "label": "RuleDefinition", "properties": {
                "rule_stage_candidate": "GRAPH_COMPOSITE",
                "rule_expression": "缺铁性贫血=联合检测(血清铁,TIBC)",
                "rule_name": "联合检测",
                "rule_evidence_json": json.dumps([{"role": "table_caption", "exact_quote": text}], ensure_ascii=False),
            }},
            {"id": "rule-one", "label": "RuleDefinition", "properties": {
                "rule_stage_candidate": "PREPROCESS",
                "rule_expression": "PTR=比值计算(PT)",
                "rule_name": "比值计算",
                "rule_evidence_json": json.dumps([{"role": "formula", "exact_quote": text}], ensure_ascii=False),
            }},
            {"id": "rule-two", "label": "RuleDefinition", "properties": {
                "rule_stage_candidate": "PREPROCESS",
                "rule_expression": "PTR=比值计算(PT)",
                "rule_name": "比值计算",
                "rule_evidence_json": json.dumps([{"role": "formula", "exact_quote": text}], ensure_ascii=False),
            }},
            {"id": "unreplayable", "label": "RuleDefinition", "properties": {
                "rule_stage_candidate": "PREPROCESS",
                "rule_expression": "INR=比值计算(PTR)",
                "rule_name": "比值计算",
                "rule_evidence_json": json.dumps([
                    {"role": "formula", "exact_quote": "块中不存在的公式。"}
                ], ensure_ascii=False),
            }},
        ])
        nodes, holds = graph_builder.normalize_candidate_nodes(
            graph, chunk=chunk, schema=graph_builder.load_candidate_graph_schema()
        )
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0]["rule_evidence_refs"][0]["role"], "table_caption")
        self.assertEqual({item["reason_code"] for item in holds},
                         {"duplicate_rule_identity", "rule_evidence_quote_absent_or_ambiguous"})
        result = graph_builder.normalize_candidate_nodes(
            graph, chunk=chunk, schema=graph_builder.load_candidate_graph_schema()
        )
        self.assertEqual(len(result.judge_drafts), 1)
        self.assertEqual(result.judge_drafts[0]["reason_code"], "rule_evidence_quote_absent_or_ambiguous")

    def test_repeated_table_text_replays_with_structured_positions(self):
        header = "<tr><td>指标甲</td><td>指标乙</td><td>结果</td></tr>"
        row = "<tr><td>低</td><td>高</td><td>疾病甲</td></tr>"
        text = f"<table>{header}{row}{header}{row}</table>"
        chunk = EvidenceChunk("synthetic:duplicate-table", text, hashlib.sha256(text.encode()).hexdigest())
        header_start = text.rfind(header)
        row_start = text.rfind(row)

        source_ref = graph_builder._source_ref(
            chunk,
            "疾病甲",
            row,
            source_char_start=row_start,
            source_char_end=row_start + len(row),
        )
        header_ref = graph_builder._rule_evidence_ref(
            chunk,
            role="table_header",
            exact_quote=header,
            exact_quote_occurrence_index=1,
        )
        row_ref = graph_builder._rule_evidence_ref(
            chunk,
            role="table_row",
            exact_quote=row,
            source_char_start=row_start,
            source_char_end=row_start + len(row),
        )

        self.assertEqual(source_ref["char_start"], row_start)
        self.assertEqual(source_ref["mention_char_start"], row_start + row.find("疾病甲"))
        self.assertEqual(header_ref["char_start"], header_start)
        self.assertEqual(row_ref["char_start"], row_start)
        unique_text = "唯一指标。"
        unique_chunk = EvidenceChunk(
            "synthetic:unique-position", unique_text, hashlib.sha256(unique_text.encode()).hexdigest()
        )
        self.assertEqual(
            graph_builder._source_ref(
                unique_chunk, "唯一指标", unique_text,
                source_char_start=len(unique_text) + 1, source_char_end=len(unique_text) + 2,
            )["char_start"],
            0,
        )

    def test_invalid_rule_stage_candidate_is_retained_for_review_with_actual_value(self):
        text = "结果指标 = 计算规则(输入指标)。"
        chunk = EvidenceChunk("synthetic:invalid-rule-stage", text, hashlib.sha256(text.encode()).hexdigest())
        graph = Neo4jGraph(nodes=[{
            "id": "invalid-stage",
            "label": "RuleDefinition",
            "properties": {
                "rule_stage_candidate": "NOT_A_RULE_STAGE",
                "rule_expression": "结果指标=计算规则(输入指标)",
                "rule_name": "计算规则",
                "rule_evidence_json": json.dumps([
                    {"role": "formula", "exact_quote": text}
                ], ensure_ascii=False),
            },
        }])

        nodes, holds = graph_builder.normalize_candidate_nodes(
            graph, chunk=chunk, schema=graph_builder.load_candidate_graph_schema()
        )

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["rule_stage_candidate"], "UNKNOWN")
        self.assertEqual(nodes[0]["extraction_status"], "PARTIAL")
        hold = next(item for item in holds if item["reason_code"] == "rule_stage_candidate_invalid")
        self.assertEqual(hold["status"], "REVIEW_REQUIRED")
        self.assertEqual(hold["candidate_summary"]["rule_stage_candidate"], "NOT_A_RULE_STAGE")

    def test_rule_endpoint_not_frozen_enters_hold(self):
        text = "PTR = 受检血浆。"
        chunk = EvidenceChunk("synthetic:endpoint", text, hashlib.sha256(text.encode()).hexdigest())
        rule_ref = graph_builder._rule_evidence_ref(chunk, role="formula", exact_quote=text)
        rule_key = graph_builder._rule_candidate_key(
            chunk=chunk, rule_stage="PREPROCESS", rule_expression="PTR=比值计算(PT)", rule_evidence_refs=[rule_ref]
        )
        rule = {"candidate_key": rule_key, "rule_candidate_key": rule_key,
                "entity_type": "RuleDefinition", "rule_expression": "PTR=比值计算(PT)",
                "rule_name": "比值计算", "rule_stage_candidate": "PREPROCESS",
                "rule_evidence_refs": [rule_ref]}
        graph = Neo4jGraph(relationships=[Neo4jRelationship(
            start_node_id="candidate:unknown", end_node_id=rule_key, type="RULE_INPUT",
            properties={"rule_evidence_role": "formula"},
        )])
        relations, holds = graph_builder.normalize_candidate_relationships(
            graph, chunk=chunk, schema=graph_builder.load_candidate_graph_schema(), nodes=[rule]
        )
        self.assertEqual(relations, [])
        self.assertEqual({item["reason_code"] for item in holds}, {
            "relation_endpoint_not_from_frozen_catalog", "rule_structure_incomplete"
        })
        self.assertEqual(rule["extraction_status"], "PARTIAL")
        self.assertEqual(rule["review_status"], "REVIEW_REQUIRED")
        self.assertIn("OUTPUT_ENTITY_UNRESOLVED", rule["warnings"])
        self.assertEqual(
            next(item for item in holds if item["reason_code"] == "relation_endpoint_not_from_frozen_catalog")["status"],
            "REVIEW_REQUIRED",
        )
        self.assertEqual(
            next(item for item in holds if item["reason_code"] == "rule_structure_incomplete")["status"],
            "REVIEW_REQUIRED",
        )

    def test_rule_without_complete_input_output_subgraph_is_retained_for_review(self):
        text = "PTR = 受检血浆。"
        chunk = EvidenceChunk("synthetic:isolated-rule", text, hashlib.sha256(text.encode()).hexdigest())
        schema = graph_builder.load_candidate_graph_schema()

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"nodes": [
                        {"label": "LabIndicator", "properties": {
                            "mention": "PTR", "canonical_name_candidate": "PTR", "exact_quote": text,
                        }},
                        {"label": "LabIndicator", "properties": {
                            "mention": "受检血浆", "canonical_name_candidate": "受检血浆", "exact_quote": text,
                        }},
                    ], "relationships": []}, ensure_ascii=False))
                if self.calls == 2:
                    return SimpleNamespace(content=json.dumps({"nodes": [{
                        "label": "RuleDefinition", "properties": {
                            "rule_stage_candidate": "PREPROCESS",
                            "rule_expression": "PTR=比值计算(受检血浆)",
                            "rule_name": "比值计算",
                            "rule_evidence_json": json.dumps([
                                {"role": "formula", "exact_quote": text}
                            ], ensure_ascii=False),
                        }
                    }], "relationships": []}, ensure_ascii=False))
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": []}))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk, schema=schema, output_dir=output, source_manifest_sha256="f" * 64,
            ))
            graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
            queue = json.loads((output / "review-queue.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["relationship_count"], 0)
        self.assertEqual(summary["node_count"], 3)
        rule = next(item for item in graph["nodes"] if item["entity_type"] == "RuleDefinition")
        self.assertEqual(rule["extraction_status"], "PARTIAL")
        self.assertEqual(rule["review_status"], "REVIEW_REQUIRED")
        self.assertIn("INPUT_ENTITY_UNRESOLVED", rule["warnings"])
        self.assertIn("rule_structure_incomplete", {item["reason_code"] for item in queue["items"]})
        self.assertEqual(
            next(item for item in queue["items"] if item["reason_code"] == "rule_structure_incomplete")["status"],
            "REVIEW_REQUIRED",
        )

    def test_rule_phase_retries_once_after_invalid_response(self):
        text = "PTR = 受检血浆。"
        chunk = EvidenceChunk("synthetic:invalid-rule-response", text, hashlib.sha256(text.encode()).hexdigest())
        schema = graph_builder.load_candidate_graph_schema()

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"nodes": [{
                        "label": "LabIndicator", "properties": {
                            "mention": "PTR", "canonical_name_candidate": "PTR", "exact_quote": text,
                        }
                    }], "relationships": []}, ensure_ascii=False))
                if self.calls == 2:
                    raise graph_builder.LLMGenerationError("invalid graph response")
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": []}))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk, schema=schema, output_dir=output, source_manifest_sha256="g" * 64,
            ))
            graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
            queue = json.loads((output / "review-queue.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["node_count"], 1)
        self.assertFalse(any(item["entity_type"] == "RuleDefinition" for item in graph["nodes"]))
        self.assertFalse(any(item["reason_code"] == "rule_phase_model_response_invalid" for item in queue["items"]))

    def test_rule_phase_two_failures_record_rejected_stage_failure(self):
        text = "PTR = 受检血浆。"
        chunk = EvidenceChunk("synthetic:two-rule-failures", text, hashlib.sha256(text.encode()).hexdigest())
        schema = graph_builder.load_candidate_graph_schema()

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"nodes": [{
                        "label": "LabIndicator", "properties": {
                            "mention": "PTR", "canonical_name_candidate": "PTR", "exact_quote": text,
                        }
                    }], "relationships": []}, ensure_ascii=False))
                if self.calls in {2, 3}:
                    raise graph_builder.LLMGenerationError("invalid graph response")
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": []}))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk, schema=schema, output_dir=output, source_manifest_sha256="z" * 64,
            ))
            queue = json.loads((output / "review-queue.json").read_text(encoding="utf-8"))

        hold = next(item for item in queue["items"] if item["reason_code"] == "rule_phase_model_response_invalid")
        self.assertEqual(hold["status"], "REJECTED")
        self.assertEqual(hold["candidate_summary"]["attempts"], 2)

    def test_judge_draft_is_bounded_and_never_enters_graph(self):
        text = "血清铁降低。"
        chunk = EvidenceChunk("synthetic:judge-draft", text, hashlib.sha256(text.encode()).hexdigest())
        graph = Neo4jGraph(nodes=[{
            "id": "unknown", "label": "UnknownType", "properties": {
                "mention": "血清铁降低", "canonical_name_candidate": "血清铁降低",
                "exact_quote": "x" * 2_100,
            },
        }])

        result = graph_builder.normalize_candidate_nodes(
            graph, chunk=chunk, schema=graph_builder.load_candidate_graph_schema()
        )

        self.assertEqual(result.accepted, [])
        self.assertEqual(len(result.judge_drafts), 1)
        draft = result.judge_drafts[0]
        self.assertEqual(draft["judge_status"], "PENDING")
        self.assertLessEqual(len(draft["candidate_draft"]["properties"]["exact_quote"]), 2_000)
        self.assertNotIn("source_ref", draft["candidate_draft"])

    def test_partial_nodes_are_not_exposed_in_frozen_catalog(self):
        nodes = [
            {"candidate_key": "candidate:valid", "entity_type": "LabIndicator", "mention": "血清铁",
             "canonical_name_candidate": "血清铁", "extraction_status": "VALID"},
            {"candidate_key": "candidate:partial", "entity_type": "IndicatorState", "mention": "血清铁降低",
             "canonical_name_candidate": "血清铁降低", "extraction_status": "PARTIAL"},
        ]
        catalog = json.loads(graph_builder._catalog_for_prompt(nodes))["frozen_candidate_catalog"]
        self.assertEqual([item["candidate_key"] for item in catalog], ["candidate:valid"])

    def test_rule_response_diagnostic_records_shape_without_raw_model_text(self):
        raw = json.dumps({"nodes": [], "relationships": [], "node_types": []})
        diagnostic = graph_builder._response_shape_diagnostic(raw)

        self.assertEqual(diagnostic["parse_phase"], "adapter_json")
        self.assertEqual(diagnostic["json_top_level_fields"], ["node_types", "nodes", "relationships"])
        self.assertEqual(diagnostic["json_top_level_field_types"]["node_types"], "list")
        self.assertEqual(diagnostic["missing_fields"], [])
        self.assertNotIn(raw, json.dumps(diagnostic, ensure_ascii=False))

    def test_response_adapter_moves_declared_top_level_node_fields_into_properties(self):
        class FakeLLM:
            async def ainvoke(self, prompt):
                return SimpleNamespace(content=json.dumps({"nodes": [{
                    "label": "Disease", "mention": "消化性溃疡出血",
                    "extraction_reason": "原文明示的疾病性原因。",
                }], "relationships": []}, ensure_ascii=False))

        result = asyncio.run(graph_builder._GraphRagIdCompletingLLM(FakeLLM()).ainvoke("prompt"))
        node = json.loads(result.content)["nodes"][0]

        self.assertEqual(node["id"], "transient-node-0")
        self.assertNotIn("mention", node)
        self.assertEqual(node["properties"], {
            "mention": "消化性溃疡出血", "extraction_reason": "原文明示的疾病性原因。",
        })

    def test_semantic_entity_output_gets_code_derived_provenance(self):
        text = "肝硬化使转铁蛋白合成减少。\n肝硬化可影响检验结果。"
        chunk = EvidenceChunk("synthetic:semantic-entity", text, hashlib.sha256(text.encode()).hexdigest())
        graph = Neo4jGraph(nodes=[
            {"id": "disease", "label": "Disease", "properties": {
                "mention": "肝硬化", "extraction_reason": "原文明示的疾病。",
            }},
            {"id": "context", "label": "ClinicalContext", "properties": {
                "mention": "转铁蛋白合成减少", "extraction_reason": "原文明示的机制背景。",
            }},
        ])

        result = graph_builder.normalize_candidate_nodes(
            graph,
            chunk=chunk,
            schema=graph_builder.load_candidate_graph_schema(),
            derive_entity_provenance=True,
        )

        self.assertEqual(result.review_items, [])
        disease = next(item for item in result.accepted if item["entity_type"] == "Disease")
        self.assertEqual(disease["canonical_name_candidate"], "肝硬化")
        self.assertEqual(disease["source_ref"]["exact_quote"], "肝硬化使转铁蛋白合成减少。")
        self.assertEqual(len(disease["source_refs"]), 2)
        self.assertEqual(disease["extraction_reason"], "原文明示的疾病。")

    def test_unanchored_semantic_table_state_enters_judge_queue(self):
        text = "<table><tr><th>血清铁</th></tr><tr><td>↓</td></tr></table>"
        chunk = EvidenceChunk("synthetic:semantic-table", text, hashlib.sha256(text.encode()).hexdigest())
        graph = Neo4jGraph(nodes=[{"id": "state", "label": "IndicatorState", "properties": {
            "mention": "血清铁降低", "extraction_reason": "表格箭头表示指标降低。",
        }}])

        result = graph_builder.normalize_candidate_nodes(
            graph,
            chunk=chunk,
            schema=graph_builder.load_candidate_graph_schema(),
            derive_entity_provenance=True,
        )

        self.assertEqual(result.accepted, [])
        self.assertEqual(result.review_items[0]["reason_code"], "semantic_mention_not_in_source")
        self.assertEqual(result.judge_drafts[0]["candidate_draft"]["properties"]["extraction_reason"],
                         "表格箭头表示指标降低。")

    def test_chunk_without_rules_returns_empty_rule_collection(self):
        text = "血清铁是一个实验室指标。"
        chunk = EvidenceChunk("synthetic:no-rule", text, hashlib.sha256(text.encode()).hexdigest())
        schema = graph_builder.load_candidate_graph_schema()

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"nodes": [{
                        "label": "LabIndicator",
                        "properties": {
                            "mention": "血清铁",
                            "canonical_name_candidate": "血清铁",
                            "exact_quote": text,
                        },
                    }], "relationships": []}, ensure_ascii=False))
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": []}))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk, schema=schema, output_dir=output, source_manifest_sha256="h" * 64,
            ))
            graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["hold_count"], 0)
        self.assertEqual(summary["relationship_count"], 0)
        self.assertEqual([item["entity_type"] for item in graph["nodes"]], ["LabIndicator"])

    def test_candidate_prompts_require_rule_evidence_and_explicit_causality(self):
        self.assertIn("complete sentence", graph_builder.NODE_PROMPT_TEMPLATE)
        self.assertIn("table-row entity", graph_builder.NODE_PROMPT_TEMPLATE)
        self.assertIn("omit a prose state", graph_builder.NODE_PROMPT_TEMPLATE)
        self.assertIn("independently referable source phrase", graph_builder.NODE_PROMPT_TEMPLATE)
        self.assertIn("Enumeration punctuation", graph_builder.NODE_PROMPT_TEMPLATE)
        self.assertIn("mechanism, treatment factor, physiological stage", graph_builder.NODE_PROMPT_TEMPLATE)
        self.assertIn("causal sentence", graph_builder.RELATION_PROMPT_TEMPLATE)
        self.assertNotIn("rule_evidence_json", graph_builder.NODE_PROMPT_TEMPLATE)
        self.assertIn("rule_evidence_json", graph_builder.RULE_NODE_PROMPT_TEMPLATE)
        self.assertIn("GRAPH_COMPOSITE", graph_builder.RULE_NODE_PROMPT_TEMPLATE)
        self.assertIn("decide from its complete context", graph_builder.RULE_NODE_PROMPT_TEMPLATE)
        self.assertIn("table_state_evidence_json", graph_builder.NODE_PROMPT_TEMPLATE)
        self.assertIn("RULE_INPUT", graph_builder.RULE_EDGE_PROMPT_TEMPLATE)
        self.assertNotIn("PTR", graph_builder.RULE_NODE_PROMPT_TEMPLATE)
        self.assertNotIn("ISI", graph_builder.RULE_EDGE_PROMPT_TEMPLATE)

    def test_entity_discovery_uses_diverse_few_shot_examples(self):
        examples = graph_builder.ENTITY_DISCOVERY_EXAMPLES

        self.assertIn("Example 1", examples)
        self.assertNotIn("Example 4", examples)
        self.assertIn("<table>", examples)
        self.assertIn("严重的肝病", examples)
        self.assertIn("转铁蛋白合成减少", examples)
        self.assertIn("extraction_reason", examples)
