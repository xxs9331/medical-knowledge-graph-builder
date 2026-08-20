from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "evaluation" / "chapter-01"
GRAPH_PATH = DATASET_ROOT / "chapter-01-graph-test-set-v0.3.json"
AUDIT_PATH = DATASET_ROOT / "chapter-01-evidence-audit-v0.3.json"
MANIFEST_PATH = REPO_ROOT / "source-packages" / "canonical" / "evidence" / "chapter-01" / "manifest.json"
TYPICAL_CASES_PATH = REPO_ROOT / "evaluation" / "typical-cases" / "typical-cases-v0.1.json"


class Chapter01EvaluationSetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))

    def test_all_manifest_chunks_are_covered_exactly_once(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = {chunk["chunk_id"] for chunk in manifest["chunks"]}
        actual = [chunk_id for case in self.graph["cases"] for chunk_id in case["chunk_ids"]]

        self.assertEqual(expected, set(actual))
        self.assertEqual([], [chunk_id for chunk_id, count in Counter(actual).items() if count != 1])

    def test_relationship_endpoints_exist_in_case_entities(self) -> None:
        for case in self.graph["cases"]:
            with self.subTest(case_id=case["case_id"]):
                entity_mentions = {entity[1] for entity in case["entities"]}
                for start, _relationship_type, end in case["relationships"]:
                    self.assertIn(start, entity_mentions)
                    self.assertIn(end, entity_mentions)

    def test_projected_datasets_match_graph_source(self) -> None:
        views = {
            "entities": "chapter-01-entity-test-set-v0.3.json",
            "relationships": "chapter-01-relationship-test-set-v0.3.json",
            "rules": "chapter-01-rule-test-set-v0.3.json",
        }
        graph_cases = {case["case_id"]: case for case in self.graph["cases"]}
        for field, filename in views.items():
            view = json.loads((DATASET_ROOT / filename).read_text(encoding="utf-8"))
            with self.subTest(filename=filename):
                self.assertEqual(self.graph["status"], view["status"])
                self.assertEqual(set(graph_cases), {case["case_id"] for case in view["cases"]})
                for case in view["cases"]:
                    source = graph_cases[case["case_id"]]
                    self.assertEqual(source["chunk_ids"], case["chunk_ids"])
                    self.assertEqual(source[field], case["expected"])
                    self.assertEqual(
                        source["evaluation_scopes"], case["evaluation_scopes"]
                    )
                    if field == "relationships":
                        self.assertEqual(
                            source["must_not_extract"], case["forbidden"]
                        )
                    if field == "rules":
                        self.assertEqual(source["executor_rules"], case["executor_rules"])
                        self.assertEqual(source["held_rules"], case["held_rules"])

    def test_evaluation_scopes_replay_manifest_boundaries(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        char_counts = {item["chunk_id"]: item["char_count"] for item in manifest["chunks"]}
        for case in self.graph["cases"]:
            with self.subTest(case_id=case["case_id"]):
                scope_ids = [scope["chunk_id"] for scope in case["evaluation_scopes"]]
                self.assertEqual(len(scope_ids), len(set(scope_ids)))
                self.assertTrue(set(case["chunk_ids"]).issubset(scope_ids))
                self.assertEqual(set(case["chunk_ids"]), set(scope_ids))
                for scope in case["evaluation_scopes"]:
                    self.assertEqual(0, scope["start"])
                    self.assertEqual(char_counts[scope["chunk_id"]], scope["end"])

    def test_audit_covers_each_scored_held_and_forbidden_target(self) -> None:
        audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        self.assertEqual("HUMAN_REVIEW_REQUIRED", audit["status"])
        audit_cases = {case["case_id"]: case["items"] for case in audit["cases"]}
        self.assertEqual({case["case_id"] for case in self.graph["cases"]}, set(audit_cases))
        total = 0
        for case in self.graph["cases"]:
            expected = (
                [("entity", item) for item in case["entities"]]
                + [("relationship", item) for item in case["relationships"]]
                + [("graph_rule", item) for item in case["rules"]]
                + [("executor_rule", item) for item in case["executor_rules"]]
                + [("held_rule", item) for item in case["held_rules"]]
                + [
                    ("forbidden_relationship", item)
                    for item in case["must_not_extract"]
                ]
            )
            actual = [(item["kind"], item["target"]) for item in audit_cases[case["case_id"]]]
            self.assertEqual(expected, actual)
            total += len(actual)
        self.assertEqual(total, audit["summary"]["total_items"])

    def test_confirmed_semantic_corrections_are_preserved(self) -> None:
        by_id = {case["case_id"]: case for case in self.graph["cases"]}
        d_dimer = by_id["CH01-08"]
        self.assertNotIn(
            ["深静脉血栓形成", "ASSOCIATED_WITH", "D-二聚体阳性"],
            d_dimer["relationships"],
        )
        self.assertIn(
            ["深静脉血栓形成", "ASSOCIATED_WITH", "D-二聚体阳性"],
            d_dimer["must_not_extract"],
        )
        entity_types: dict[str, set[str]] = {}
        for case in self.graph["cases"]:
            for entity_type, mention in case["entities"]:
                entity_types.setdefault(mention, set()).add(entity_type)
        self.assertEqual({"IndicatorState"}, entity_types["红细胞压积增高"])
        for mention in ("A型血", "B型血", "AB型血", "O型血"):
            self.assertEqual({"IndicatorState"}, entity_types[mention])

    def test_rule_contract_separates_graph_executor_and_held_rules(self) -> None:
        graph_rules = [rule for case in self.graph["cases"] for rule in case["rules"]]
        executor_rules = [
            rule for case in self.graph["cases"] for rule in case["executor_rules"]
        ]
        held_rules = [rule for case in self.graph["cases"] for rule in case["held_rules"]]
        self.assertEqual(36, len(graph_rules))
        self.assertEqual(30, len(executor_rules))
        self.assertEqual(3, len(held_rules))
        self.assertEqual({"GRAPH_COMPOSITE"}, {rule["rule_stage"] for rule in graph_rules})
        self.assertEqual({"PREPROCESS"}, {rule["rule_stage"] for rule in executor_rules})

    def test_added_inference_and_exclusion_rules_are_complete(self) -> None:
        by_id = {case["case_id"]: case for case in self.graph["cases"]}
        anemia_rules = by_id["CH01-02"]["rules"]
        self.assertEqual(10, len(anemia_rules))
        self.assertIn(
            {
                "rule_stage": "GRAPH_COMPOSITE",
                "inputs": ["MCV减小", "MCH显著减小(<23pg)", "MCHC减小"],
                "outputs": ["小细胞低色素性贫血"],
                "logic": "ALL",
            },
            anemia_rules,
        )

        blood_type_rules = by_id["CH01-07"]["rules"]
        self.assertEqual(19, len(blood_type_rules))
        self.assertIn(
            {
                "rule_stage": "GRAPH_COMPOSITE",
                "inputs": ["父母血型组合为O+O"],
                "outputs": [
                    "子女不可能为A型血",
                    "子女不可能为B型血",
                    "子女不可能为AB型血",
                ],
                "logic": "ALL",
            },
            blood_type_rules,
        )
        self.assertEqual(
            10,
            sum(
                any(output.startswith("子女可能为") for output in rule["outputs"])
                for rule in blood_type_rules
            ),
        )
        self.assertEqual(
            9,
            sum(
                any(output.startswith("子女不可能为") for output in rule["outputs"])
                for rule in blood_type_rules
            ),
        )

        d_dimer_rules = by_id["CH01-08"]["rules"]
        self.assertIn(
            {
                "rule_stage": "GRAPH_COMPOSITE",
                "inputs": ["D-二聚体正常"],
                "outputs": ["排除深静脉血栓有重要价值"],
                "logic": "ALL",
            },
            d_dimer_rules,
        )
        self.assertEqual(2, len(by_id["CH01-06"]["rules"]))

    def test_graph_rule_endpoints_exist_in_case_entities(self) -> None:
        for case in self.graph["cases"]:
            mentions = {entity[1] for entity in case["entities"]}
            for rule in case["rules"]:
                with self.subTest(case_id=case["case_id"], rule=rule):
                    self.assertTrue(set(rule["inputs"]).issubset(mentions))
                    self.assertTrue(set(rule["outputs"]).issubset(mentions))

    def test_graph_rules_include_all_typical_case_rules(self) -> None:
        typical = json.loads(TYPICAL_CASES_PATH.read_text(encoding="utf-8"))

        def signature(rule: dict[str, Any]) -> tuple[object, ...]:
            normalize = lambda value: "".join(str(value).split())
            return (
                rule["rule_stage"],
                rule["logic"],
                tuple(normalize(item) for item in rule["inputs"]),
                tuple(normalize(item) for item in rule["outputs"]),
            )

        chapter_rules = {
            signature(rule) for case in self.graph["cases"] for rule in case["rules"]
        }
        typical_rules = {
            signature(rule) for case in typical["cases"] for rule in case["rules"]
        }
        self.assertTrue(typical_rules.issubset(chapter_rules))


if __name__ == "__main__":
    unittest.main()
