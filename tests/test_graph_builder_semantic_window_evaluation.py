"""semantic section 功能切窗与目标盲映射测试。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from medical_kg_sourceprep.extraction.graph_builder.contract import PROJECT_ROOT
from medical_kg_sourceprep.extraction.graph_builder.runner.semantic_window_evaluation import (
    build_semantic_windows,
    map_cases_to_semantic_windows,
)
from medical_kg_sourceprep.extraction.graph_builder.runner.semantic_section_evaluation import (
    load_semantic_sections,
)


SEMANTIC_ROOT = (
    PROJECT_ROOT / "source-packages/derived/semantic-sections/full-book-v0.2"
)


def test_windows_cover_section_without_splitting_frozen_function_blocks() -> None:
    """窗口必须连续覆盖 section，并在预注册的功能边界处分开。"""
    _manifest, sections = load_semantic_sections(SEMANTIC_ROOT)
    expected = {
        "000004": [(0, 596), (596, 980), (980, 1265)],
        "000007": [(0, 196), (196, 493)],
        "000011": [(0, 94), (94, 218)],
        "000015": [(0, 272), (272, 541)],
        "000018": [(0, 164), (164, 667)],
        "000033": [(0, 317), (317, 535)],
        "000034": [(0, 287)],
    }
    for suffix, spans in expected.items():
        section = sections[f"health-check-interpretation-v2:section:{suffix}"]
        windows = build_semantic_windows(section)
        assert [(window.start, window.end) for window in windows] == spans
        assert "".join(window.text for window in windows) == section.text


def test_eight_cases_select_nine_windows_without_changing_gold_targets() -> None:
    """案例只按原文范围选择窗口，答案字段和分母必须保持不变。"""
    dataset, cases, windows, mappings = map_cases_to_semantic_windows(
        gold_path=PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json",
        canonical_manifest_path=(
            PROJECT_ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json"
        ),
        semantic_root=SEMANTIC_ROOT,
    )
    assert len(cases) == 8
    assert len(windows) == 9
    selected: dict[str, list[str]] = {}
    for item in mappings:
        raw_window_ids = item["window_ids"]
        assert isinstance(raw_window_ids, list)
        selected[str(item["case_id"])] = [
            str(value).rsplit(":", 1)[-1]
            for value in cast(list[object], raw_window_ids)
        ]
    assert selected == {
        "TC-01": ["0001", "0002"],
        "TC-02": ["0002"],
        "TC-03": ["0002"],
        "TC-04": ["0001"],
        "TC-05": ["0002"],
        "TC-06": ["0001"],
        "TC-07": ["0002"],
        "TC-08": ["0001"],
    }
    raw_cases = dataset["cases"]
    assert isinstance(raw_cases, list)
    original_by_id: dict[str, Mapping[str, object]] = {}
    for raw_case in cast(list[object], raw_cases):
        assert isinstance(raw_case, Mapping)
        original = cast(Mapping[str, object], raw_case)
        original_by_id[str(original["case_id"])] = original
    for case in cases:
        original = original_by_id[str(case["case_id"])]
        for field in ("entities", "relationships", "rules", "must_not_extract"):
            assert case[field] == original[field]
