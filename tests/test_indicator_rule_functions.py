from __future__ import annotations

import hashlib
import unittest

from medical_kg_sourceprep.indicator_rule_functions import (
    RuleFunctionError,
    audit_candidates,
    build_rule_prompt,
    build_rule_windows,
    stable_candidates,
    validate_rule_response,
)
from medical_kg_sourceprep.llm_extraction import EvidenceChunk


def chunk(chunk_id: str, text: str, page_index: int, start: int = 0) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        chunk_sha256=hashlib.sha256(text.encode()).hexdigest(),
        chapter_id="chapter-01",
        page_id=f"page:{page_index}",
        printed_page=page_index + 4,
        source_pdf_page=page_index + 21,
        page_index=page_index,
        start_offset=start,
    )


def library(*names: str) -> dict:
    return {
        "indicators": [{
            "candidate_id": f"indicator:{index}",
            "canonical_name": name,
            "aliases": [],
            "index_aliases": [],
            "body_occurrences": [],
            "origins": ["model"],
        } for index, name in enumerate(names)]
    }


class RuleWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            chunk("p0:c0", "前页结尾。", 0),
            chunk("p1:c0", "贫血程度\n", 1),
            chunk("p1:c1", "血红蛋白 90~120g/L 为轻度贫血。", 1, 5),
            chunk("p2:c0", "后页开头。", 2),
        ]
        self.windows = build_rule_windows(self.chunks)
        self.window = self.windows["page:1"]
        self.catalog = library("血红蛋白")

    def valid_rule(self) -> dict:
        source = "贫血程度\n血红蛋白 90~120g/L 为轻度贫血。"
        return {
            "rule_expression": "贫血程度 = 贫血分级判断(血红蛋白)",
            "cases": [{
                "condition": "血红蛋白 BETWEEN 90 AND 120",
                "result": "轻度贫血",
                "evidence": "血红蛋白 90~120g/L 为轻度贫血",
            }],
            "formula": None,
            "default_result": None,
            "evidence_overall": {
                "source_quote": source,
                "condition_quotes": ["血红蛋白 90~120g/L"],
                "conclusion_quote": "轻度贫血",
            },
        }

    def test_window_adds_one_neighbor_chunk_on_each_side(self) -> None:
        self.assertEqual(
            [segment.role for segment in self.window.segments],
            ["left_context", "target", "target", "right_context"],
        )
        self.assertEqual(self.window.text, "".join(chunk.text for chunk in self.chunks))
        self.assertEqual(self.chunks[1].text, "贫血程度\n")

    def test_accepts_rule_whose_source_crosses_adjacent_target_chunks(self) -> None:
        result = validate_rule_response([self.valid_rule()], self.window, self.catalog)
        self.assertEqual(result["counts"], {"accepted": 1, "rejected": 0})
        candidate = result["candidates"][0]
        self.assertEqual(candidate["source"]["source_chunk_ids"], ["p1:c0", "p1:c1"])
        self.assertFalse(candidate["output_catalog_match"])
        self.assertEqual(
            "".join(span["exact_quote"] for span in candidate["source"]["chunk_spans"]),
            candidate["source"]["exact_quote"],
        )

    def test_rejects_rule_owned_by_left_context(self) -> None:
        raw = self.valid_rule()
        raw["rule_expression"] = "前页结尾 = 上下文判断(前页)"
        raw["cases"] = [{"condition": "前页 IN 原文", "result": "结尾", "evidence": "前页结尾"}]
        raw["evidence_overall"] = {
            "source_quote": "前页结尾。",
            "condition_quotes": ["前页结尾"],
            "conclusion_quote": "结尾",
        }
        result = validate_rule_response([raw], self.window, self.catalog)
        self.assertEqual(result["counts"], {"accepted": 0, "rejected": 1})
        self.assertEqual(result["rejections"][0]["reason_code"], "conclusion_outside_target")

    def test_accepts_verbatim_formula(self) -> None:
        source = chunk(
            "formula", "平均红细胞容积=血细胞比容/红细胞计数×10", 3
        )
        window = build_rule_windows([source])["page:3"]
        formula = source.text
        raw = {
            "rule_expression": "平均红细胞容积 = 红细胞指数计算(血细胞比容, 红细胞计数)",
            "cases": [{
                "condition": "血细胞比容 IN 公式输入 且 红细胞计数 IN 公式输入",
                "result": formula,
                "evidence": formula,
            }],
            "formula": formula,
            "default_result": None,
            "evidence_overall": {
                "source_quote": formula,
                "condition_quotes": [formula],
                "conclusion_quote": "平均红细胞容积",
            },
        }
        result = validate_rule_response(
            [raw], window, library("平均红细胞容积", "血细胞比容", "红细胞计数")
        )
        self.assertEqual(result["counts"]["accepted"], 1)
        self.assertEqual(result["candidates"][0]["rule"]["formula"], formula)
        self.assertTrue(result["candidates"][0]["output_catalog_match"])

    def test_rejects_symbolic_condition_operator(self) -> None:
        raw = self.valid_rule()
        raw["cases"][0]["condition"] = "血红蛋白 >= 90"
        result = validate_rule_response([raw], self.window, self.catalog)
        self.assertEqual(result["rejections"][0]["reason_code"], "unsupported_operator")

    def test_rejects_non_verbatim_formula(self) -> None:
        raw = self.valid_rule()
        raw["formula"] = "血红蛋白=90"
        result = validate_rule_response([raw], self.window, self.catalog)
        self.assertEqual(result["rejections"][0]["reason_code"], "component_not_verbatim")

    def test_rejects_condition_threshold_absent_from_source(self) -> None:
        raw = self.valid_rule()
        raw["cases"][0]["condition"] = "血红蛋白 BETWEEN 80 AND 120"
        result = validate_rule_response([raw], self.window, self.catalog)
        self.assertEqual(
            result["rejections"][0]["reason_code"], "condition_value_not_grounded"
        )

    def test_rejects_descriptive_extreme_as_threshold(self) -> None:
        source = chunk("extreme", "血细胞比容增高可高达 0.60 以上", 3)
        window = build_rule_windows([source])["page:3"]
        raw = {
            "rule_expression": "血细胞比容 = 增高判断(血细胞比容)",
            "cases": [{"condition": "血细胞比容 GT 0.60", "result": "增高",
                       "evidence": source.text}],
            "formula": None, "default_result": None,
            "evidence_overall": {"source_quote": source.text,
                                 "condition_quotes": [source.text],
                                 "conclusion_quote": source.text},
        }
        result = validate_rule_response([raw], window, library("血细胞比容"))
        self.assertEqual(
            result["rejections"][0]["reason_code"], "descriptive_extreme_not_threshold"
        )

    def test_catalog_output_can_be_grounded_by_target_section_context(self) -> None:
        source = chunk("range", "血细胞比容\n男性 0.40~0.50L/L\n女性 0.35~0.45L/L", 3)
        window = build_rule_windows([source])["page:3"]
        ranges = "男性 0.40~0.50L/L\n女性 0.35~0.45L/L"
        raw = {
            "rule_expression": "血细胞比容 = 性别参考区间选择(性别)",
            "cases": [
                {"condition": "性别 EQ 男性", "result": "0.40~0.50L/L",
                 "evidence": "男性 0.40~0.50L/L"},
                {"condition": "性别 EQ 女性", "result": "0.35~0.45L/L",
                 "evidence": "女性 0.35~0.45L/L"},
            ],
            "formula": None, "default_result": None,
            "evidence_overall": {"source_quote": ranges,
                                 "condition_quotes": ["男性 0.40~0.50L/L", "女性 0.35~0.45L/L"],
                                 "conclusion_quote": ranges},
        }
        result = validate_rule_response([raw], window, library("血细胞比容"))
        self.assertEqual(result["counts"], {"accepted": 1, "rejected": 0})
        self.assertEqual(result["candidates"][0]["output_grounding"], "target_catalog_context")

    def test_rejects_upper_reference_bound_without_gt_cue(self) -> None:
        source = chunk("range-gt", "血浆黏度 1.12~1.64mPa·s\n血浆黏度升高", 3)
        window = build_rule_windows([source])["page:3"]
        raw = {
            "rule_expression": "血浆黏度 = 升高判断(血浆黏度)",
            "cases": [{"condition": "血浆黏度 GT 1.64", "result": "血浆黏度升高",
                       "evidence": "血浆黏度 1.12~1.64mPa·s"}],
            "formula": None, "default_result": None,
            "evidence_overall": {"source_quote": source.text,
                                 "condition_quotes": ["血浆黏度 1.12~1.64mPa·s"],
                                 "conclusion_quote": "血浆黏度升高"},
        }
        result = validate_rule_response([raw], window, library("血浆黏度"))
        self.assertEqual(result["rejections"][0]["reason_code"], "operator_not_grounded")

    def test_rejects_result_unrelated_to_reference_range_case(self) -> None:
        source = chunk("unlinked", "血小板压积在常规体检中意义不大。\n成人 0.11%~0.28%", 3)
        window = build_rule_windows([source])["page:3"]
        raw = {
            "rule_expression": "血小板压积 = 参考区间判断(血小板压积)",
            "cases": [{"condition": "血小板压积 BETWEEN 0.11% AND 0.28%",
                       "result": "血小板压积在常规体检中意义不大",
                       "evidence": "成人 0.11%~0.28%"}],
            "formula": None, "default_result": None,
            "evidence_overall": {"source_quote": source.text,
                                 "condition_quotes": ["成人 0.11%~0.28%"],
                                 "conclusion_quote": "血小板压积在常规体检中意义不大"},
        }
        result = validate_rule_response([raw], window, library("血小板压积"))
        self.assertEqual(
            result["rejections"][0]["reason_code"], "result_not_linked_to_case"
        )

    def test_rejects_result_found_only_as_same_line_substring(self) -> None:
        source = chunk(
            "same-line-substring",
            "凝血酶原时间比值 = 受检血浆 PT/正常人血浆 PT, 参考区间为 0.86~1.15。",
            3,
        )
        window = build_rule_windows([source])["page:3"]
        raw = {
            "rule_expression": "凝血酶原时间比值 = 参考区间判断(凝血酶原时间比值)",
            "cases": [{"condition": "凝血酶原时间比值 BETWEEN 0.86 AND 1.15",
                       "result": "正常", "evidence": "参考区间为 0.86~1.15"}],
            "formula": None, "default_result": None,
            "evidence_overall": {"source_quote": source.text,
                                 "condition_quotes": ["参考区间为 0.86~1.15"],
                                 "conclusion_quote": "参考区间为 0.86~1.15"},
        }
        result = validate_rule_response([raw], window, library("凝血酶原时间比值"))
        self.assertEqual(
            result["rejections"][0]["reason_code"], "result_not_linked_to_case"
        )

    def test_rejects_non_array_top_level(self) -> None:
        with self.assertRaisesRegex(RuleFunctionError, "JSON array"):
            validate_rule_response({"rules": []}, self.window, self.catalog)

    def test_prompt_contains_target_and_overlap_chunk_index(self) -> None:
        prompt = build_rule_prompt(self.window, self.catalog)
        self.assertIn("TARGET_PAGE_ID=page:1", prompt)
        self.assertIn('"role":"left_context"', prompt)
        self.assertIn('"chunk_id":"p1:c1"', prompt)
        self.assertIn('"formula"', prompt)

    def test_stable_dedupe_and_audit(self) -> None:
        package = validate_rule_response([self.valid_rule()], self.window, self.catalog)
        candidates = stable_candidates([package, package])
        self.assertEqual(len(candidates), 1)
        audit = audit_candidates(candidates, self.chunks)
        self.assertEqual(audit["accepted_rules"], 1)
        self.assertEqual(audit["source_replay_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
