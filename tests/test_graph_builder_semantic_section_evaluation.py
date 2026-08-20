"""语义 section 实验入口的目标盲映射测试。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from medical_kg_sourceprep.extraction.graph_builder.contract import PROJECT_ROOT
from medical_kg_sourceprep.extraction.graph_builder.runner.semantic_section_evaluation import (
    map_cases_to_semantic_sections,
)


def test_typical_cases_map_to_expected_semantic_sections_without_changing_targets() -> None:
    """8 个冻结案例应唯一映射，且目标字段和分母不得被适配器修改。"""
    gold_path = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
    dataset, mapped_cases, _sections, mappings = map_cases_to_semantic_sections(
        gold_path=gold_path,
        canonical_manifest_path=(
            PROJECT_ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json"
        ),
        semantic_root=(
            PROJECT_ROOT / "source-packages/derived/semantic-sections/full-book-v0.2"
        ),
    )

    assert {
        item["case_id"]: item["semantic_section_id"] for item in mappings
    } == {
        "TC-01": "health-check-interpretation-v2:section:000015",
        "TC-02": "health-check-interpretation-v2:section:000007",
        "TC-03": "health-check-interpretation-v2:section:000018",
        "TC-04": "health-check-interpretation-v2:section:000034",
        "TC-05": "health-check-interpretation-v2:section:000004",
        "TC-06": "health-check-interpretation-v2:section:000004",
        "TC-07": "health-check-interpretation-v2:section:000011",
        "TC-08": "health-check-interpretation-v2:section:000033",
    }
    raw_cases = dataset["cases"]
    assert isinstance(raw_cases, list)
    original_by_id: dict[str, Mapping[str, object]] = {}
    for raw_case in cast(list[object], raw_cases):
        assert isinstance(raw_case, Mapping)
        original = cast(Mapping[str, object], raw_case)
        original_by_id[str(original["case_id"])] = original
    for mapped in mapped_cases:
        original = original_by_id[str(mapped["case_id"])]
        for field in ("entities", "relationships", "rules", "must_not_extract"):
            assert mapped[field] == original[field]
        assert mapped["chunk_ids"] != original["chunk_ids"]


def test_semantic_mapping_can_select_one_case() -> None:
    """case 过滤只影响实验范围，不改变映射算法。"""
    _dataset, mapped_cases, _sections, mappings = map_cases_to_semantic_sections(
        gold_path=PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json",
        canonical_manifest_path=(
            PROJECT_ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json"
        ),
        semantic_root=(
            PROJECT_ROOT / "source-packages/derived/semantic-sections/full-book-v0.2"
        ),
        case_ids={"TC-04"},
    )

    assert [item["case_id"] for item in mapped_cases] == ["TC-04"]
    assert mappings[0]["semantic_scope"] == {"start": 120, "end": 285}
