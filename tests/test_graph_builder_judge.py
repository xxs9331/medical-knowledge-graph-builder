import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from medical_kg_sourceprep.extraction.graph_builder.judge import (
    build_judge_prompt,
    judge_candidate_graph,
    load_typical_case,
    validate_judge_response,
)
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk


class GraphBuilderJudgeTests(unittest.TestCase):
    _initial_text = "血清铁降低提示缺铁性贫血。"
    chunk = EvidenceChunk(
        "case:chunk", _initial_text, hashlib.sha256(_initial_text.encode()).hexdigest()
    )
    items: list[dict[str, Any]] = []

    def setUp(self):
        text = "血清铁降低提示缺铁性贫血。"
        self.chunk = EvidenceChunk("case:chunk", text, hashlib.sha256(text.encode()).hexdigest())
        self.items = [{
            "judge_item_id": "node:candidate:1", "kind": "node",
            "candidate": {"candidate_key": "candidate:1", "entity_type": "IndicatorState", "mention": "血清铁降低"},
        }]

    def test_prompt_never_contains_gold_answers(self):
        prompt = build_judge_prompt(
            chunks=[self.chunk], schema={"node_types": [], "relationship_types": []},
            candidate_items=self.items,
        )
        self.assertNotIn("must_not_extract", prompt)
        self.assertNotIn("HUMAN_REVIEW_REQUIRED", prompt)
        self.assertIn("explicit mention alone is not sufficient", prompt)
        self.assertIn("Never return REPAIR with repair null", prompt)
        self.assertIn("OUTPUT_TEMPLATE_JSON", prompt)
        self.assertIn('"results"', prompt)

    def test_response_shape_error_reports_only_type_and_fields(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"judge_response_shape_invalid:type=dict:fields=\['judgments'\]",
        ):
            validate_judge_response(
                {"judgments": []}, candidate_items=self.items, chunks=[self.chunk]
            )

    def test_response_requires_complete_ids_and_replays_spans(self):
        result = validate_judge_response({"results": [{
            "judge_item_id": "node:candidate:1", "verdict": "SUPPORTED",
            "reason_code": "SOURCE_SUPPORTS_NODE", "reason": "原文直接支持。",
            "evidence_spans": [{"chunk_id": "case:chunk", "start": 0, "end": 5}],
            "repair": None,
        }]}, candidate_items=self.items, chunks=[self.chunk])
        self.assertEqual(result[0]["evidence_spans"][0]["exact_quote"], "血清铁降低")

    def test_repair_requires_target_and_supported_action(self):
        response = {"results": [{
            "judge_item_id": "node:candidate:1", "verdict": "REPAIR",
            "reason_code": "NODE_TYPE_WRONG", "reason": "节点类型需要修改。",
            "evidence_spans": [{"chunk_id": "case:chunk", "start": 0, "end": 5}],
            "repair": {
                "target_judge_item_id": "node:candidate:1",
                "action": "RETYPE_NODE",
                "proposed_entity_type": "LabIndicator",
            },
        }]}
        result = validate_judge_response(response, candidate_items=self.items, chunks=[self.chunk])
        self.assertEqual(result[0]["repair"]["action"], "RETYPE_NODE")

        response["results"][0]["repair"] = None
        with self.assertRaisesRegex(RuntimeError, "judge_repair_missing"):
            validate_judge_response(response, candidate_items=self.items, chunks=[self.chunk])

    def test_judge_writes_hold_artifact_without_mutating_graph(self):
        graph = {"nodes": [{"candidate_key": "candidate:1", "entity_type": "IndicatorState", "mention": "血清铁降低"}], "relationships": []}
        response = {"results": [{
            "judge_item_id": "node:candidate:1", "verdict": "SUPPORTED",
            "reason_code": "SOURCE_SUPPORTS_NODE", "reason": "原文直接支持。",
            "evidence_spans": [{"chunk_id": "case:chunk", "start": 0, "end": 5}], "repair": None,
        }]}

        class FakeLLM:
            async def ainvoke(self, _prompt):
                return SimpleNamespace(content=json.dumps(response, ensure_ascii=False))

            async def aclose(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            graph_path = Path(temporary) / "graph.json"
            output_path = Path(temporary) / "judge.json"
            graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
            before = graph_path.read_bytes()
            client = SimpleNamespace(llm=FakeLLM())
            document = asyncio.run(judge_candidate_graph(
                client, graph_path=graph_path, chunks=[self.chunk],
                schema={"node_types": [], "relationship_types": []}, output_path=output_path,
                case_id="TC-X",
            ))
            self.assertEqual(graph_path.read_bytes(), before)
            self.assertEqual(document["publication_status"], "HOLD")
            self.assertEqual(document["approved"], 0)
            self.assertFalse(document["configuration"]["gold_answers_exposed"])

    def test_typical_case_gold_has_eight_cases(self):
        path = Path("evaluation/typical-cases/typical-cases-v0.1.json")
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(value["cases"]), 8)
        self.assertEqual(value["status"], "HUMAN_VALIDATED")
        self.assertEqual(load_typical_case(path, "TC-08")["case_id"], "TC-08")
        for case in value["cases"]:
            mentions = {mention for _entity_type, mention in case["entities"]}
            for source, _relation_type, target in case["relationships"]:
                self.assertIn(source, mentions, case["case_id"])
                self.assertIn(target, mentions, case["case_id"])

    def test_specialized_test_sets_match_graph_dataset(self):
        root = Path("evaluation/typical-cases")
        graph = json.loads((root / "typical-cases-v0.1.json").read_text(encoding="utf-8"))
        entities = json.loads((root / "entity-test-set-v0.1.json").read_text(encoding="utf-8"))
        relationships = json.loads(
            (root / "relationship-test-set-v0.1.json").read_text(encoding="utf-8")
        )
        rules = json.loads((root / "rule-test-set-v0.1.json").read_text(encoding="utf-8"))
        graph_by_id = {case["case_id"]: case for case in graph["cases"]}
        self.assertEqual(set(graph_by_id), {case["case_id"] for case in entities["cases"]})
        self.assertEqual(set(graph_by_id), {case["case_id"] for case in relationships["cases"]})
        self.assertEqual(set(graph_by_id), {case["case_id"] for case in rules["cases"]})
        for dataset, field in (
            (entities, "entities"), (relationships, "relationships"), (rules, "rules")
        ):
            for case in dataset["cases"]:
                graph_case = graph_by_id[case["case_id"]]
                self.assertEqual(case["chunk_ids"], graph_case["chunk_ids"])
                self.assertEqual(case["expected"], graph_case[field])
        for case in relationships["cases"]:
            self.assertEqual(case["forbidden"], graph_by_id[case["case_id"]]["must_not_extract"])


if __name__ == "__main__":
    unittest.main()
