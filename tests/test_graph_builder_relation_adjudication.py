"""通用关系候选并集与 Judge 派生测试。"""

from __future__ import annotations

import unittest

from medical_kg_sourceprep.extraction.graph_builder.relation_adjudication import (
    apply_relationship_judgments,
    build_relationship_candidate_union,
    mask_text_outside_ranges,
)


def _relation(key: str, start: int, relation_type: str = "ASSOCIATED_WITH") -> dict[str, object]:
    return {
        "candidate_key": key,
        "source_candidate_key": "candidate:a",
        "target_candidate_key": "candidate:b",
        "relation_type": relation_type,
        "source_ref": {"char_start": start, "char_end": start + 5},
    }


class RelationAdjudicationTests(unittest.TestCase):
    def test等长掩码只公开目标范围并保持坐标(self) -> None:
        text = "范围外甲\n保留正文\n范围外乙"
        start = text.index("保留正文")
        end = start + len("保留正文")

        masked = mask_text_outside_ranges(text, [(start, end)])

        self.assertEqual(len(masked), len(text))
        self.assertEqual(masked[start:end], "保留正文")
        self.assertEqual(masked.count("\n"), text.count("\n"))
        self.assertNotIn("范围外", masked)

    def test候选并集只裁决目标证据范围且保留规则边(self) -> None:
        baseline = {
            "nodes": [{"candidate_key": "candidate:a"}, {"candidate_key": "candidate:b"}],
            "relationships": [
                _relation("relation:inside", 10),
                _relation("relation:outside", 100),
                _relation("relation:rule", 10, "RULE_INPUT"),
            ],
        }
        proposal = {
            "nodes": baseline["nodes"],
            "relationships": [_relation("relation:proposal", 20, "CAUSES")],
        }

        judge_graph, preserved_graph = build_relationship_candidate_union(
            baseline=baseline,
            proposal_graphs=[proposal],
            evidence_ranges=[(0, 30)],
        )

        self.assertEqual(
            {item["candidate_key"] for item in judge_graph["relationships"]},
            {"relation:inside", "relation:proposal"},
        )
        self.assertEqual(
            {item["candidate_key"] for item in preserved_graph["relationships"]},
            {"relation:outside", "relation:rule"},
        )

    def test候选并集按端点和类型消除不同候选键的重复关系(self) -> None:
        baseline = {
            "nodes": [{"candidate_key": "candidate:a"}, {"candidate_key": "candidate:b"}],
            "relationships": [_relation("relation:first", 10)],
        }
        proposal = {
            "nodes": baseline["nodes"],
            "relationships": [_relation("relation:duplicate", 20)],
        }

        judge_graph, _preserved = build_relationship_candidate_union(
            baseline=baseline,
            proposal_graphs=[proposal],
            evidence_ranges=[(0, 30)],
        )

        self.assertEqual(
            [item["candidate_key"] for item in judge_graph["relationships"]],
            ["relation:first"],
        )

    def test派生图只采纳明确支持且不执行repair(self) -> None:
        judge_graph = {
            "nodes": [{"candidate_key": "candidate:a"}, {"candidate_key": "candidate:b"}],
            "relationships": [
                _relation("relation:supported", 10),
                _relation("relation:unsupported", 11),
                _relation("relation:repair", 12),
            ],
        }
        preserved_graph = {"nodes": judge_graph["nodes"], "relationships": []}
        judge_document = {"results": [
            {"judge_item_id": "relationship:relation:supported", "verdict": "SUPPORTED"},
            {"judge_item_id": "relationship:relation:unsupported", "verdict": "UNSUPPORTED"},
            {"judge_item_id": "relationship:relation:repair", "verdict": "REPAIR"},
        ]}

        graph, counts = apply_relationship_judgments(
            judge_graph=judge_graph,
            preserved_graph=preserved_graph,
            judge_document=judge_document,
        )

        self.assertEqual(
            [item["candidate_key"] for item in graph["relationships"]],
            ["relation:supported"],
        )
        self.assertEqual(counts["supported"], 1)
        self.assertEqual(counts["repair"], 1)
        self.assertEqual(graph["publication_status"], "HOLD")


if __name__ == "__main__":
    unittest.main()
