"""无 Judge 结构闭集关系实验的确定性组合测试。"""

from __future__ import annotations

import unittest

from medical_kg_sourceprep.extraction.graph_builder.joint_extraction import RoutedEvidenceGroup
from medical_kg_sourceprep.extraction.graph_builder.runner.structure_closed_relation_evaluation import (
    _compose_groups,
)


def _relation(key: str, start: int, end: int, relation_type: str = "CAUSES") -> dict[str, object]:
    return {
        "candidate_key": key,
        "source_candidate_key": f"{key}-source",
        "target_candidate_key": f"{key}-target",
        "relation_type": relation_type,
        "source_ref": {"char_start": start, "char_end": end},
    }


class StructureClosedCompositionTests(unittest.TestCase):
    def test主实验组保留结构基线并替换范围外关系(self) -> None:
        baseline = {
            "nodes": [{"candidate_key": "n1"}],
            "relationships": [
                _relation("structured-base", 10, 20),
                _relation("narrative-base", 40, 50),
                _relation("rule", 0, 5, "RULE_INPUT"),
            ],
        }
        classified = [
            _relation("structured-new", 12, 18),
            _relation("narrative-new", 60, 70),
        ]
        groups = [RoutedEvidenceGroup(
            group_id="g1",
            route="table",
            start=8,
            end=25,
            units=(),
            instructions="",
        )]

        result = _compose_groups(
            baseline=baseline,
            classified_relationships=classified,
            structured_groups=groups,
        )

        d_keys = {item["candidate_key"] for item in result["D"]["relationships"]}
        self.assertEqual(d_keys, {"structured-base", "narrative-new", "rule"})
        b_keys = {item["candidate_key"] for item in result["B"]["relationships"]}
        self.assertEqual(b_keys, {"structured-new", "narrative-new", "rule"})
        c_keys = {item["candidate_key"] for item in result["C"]["relationships"]}
        self.assertEqual(
            c_keys,
            {"structured-base", "narrative-base", "structured-new", "narrative-new", "rule"},
        )


if __name__ == "__main__":
    unittest.main()
