from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from medical_kg_sourceprep.rules.indicator_catalog import (
    IndicatorContractError,
    LABEL_STUDIO_CONFIG,
    aggregate_indicators,
    attach_index_aliases,
    build_indicator_library,
    derive_table_column_indicators,
    label_studio_tasks,
    load_index_entries,
    merge_catalog_indicators,
    validate_indicator_response,
)
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk


def chunk(text: str) -> EvidenceChunk:
    return EvidenceChunk(
        "book:chapter-01:0000:0000", text, hashlib.sha256(text.encode()).hexdigest(),
        chapter_id="chapter-01", page_id="book:chapter-01:0000", printed_page=4,
        source_pdf_page=21, page_index=0,
    )


def ref(source: EvidenceChunk, quote: str) -> dict:
    return {"chunk_id": source.chunk_id, "chunk_sha256": source.chunk_sha256,
            "exact_quote": quote}


class IndicatorValidationTests(unittest.TestCase):
    def test_validates_indicator_attributes_against_exact_quotes(self) -> None:
        source = chunk("血浆凝血酶原时间 (PT)。仪器法: 11~13 秒。")
        raw = {"indicators": [{
            "name": "血浆凝血酶原时间", "item_kind": "atomic_indicator",
            "name_ref": ref(source, "血浆凝血酶原时间 (PT)"),
            "aliases": [{"value": "PT", "source_ref": ref(source, "血浆凝血酶原时间 (PT)")}],
            "value_types": [{"value": "number", "source_ref": ref(source, "仪器法: 11~13 秒")}],
            "units": [{"value": "秒", "source_ref": ref(source, "仪器法: 11~13 秒")}],
            "specimens": [{"value": "血浆", "source_ref": ref(source, "血浆凝血酶原时间 (PT)")}],
            "methods": [{"value": "仪器法", "source_ref": ref(source, "仪器法: 11~13 秒")}],
            "reference_selectors": [{"kind": "method", "value": "仪器法",
                                      "source_ref": ref(source, "仪器法: 11~13 秒")}],
        }]}
        result = validate_indicator_response(raw, [source])
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["aliases"][0]["value"], "PT")
        self.assertEqual(result["rejections"], [])

    def test_rejects_non_replayable_optional_field_without_rejecting_indicator(self) -> None:
        source = chunk("D-二聚体采用乳胶凝集法。")
        raw = {"indicators": [{
            "name": "D-二聚体", "item_kind": "atomic_indicator",
            "name_ref": ref(source, "D-二聚体采用乳胶凝集法"),
            "aliases": [], "value_types": [], "units": [], "specimens": [],
            "methods": [{"value": "免疫法", "source_ref": ref(source, "D-二聚体采用乳胶凝集法")}],
            "reference_selectors": [],
        }]}
        result = validate_indicator_response(raw, [source])
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["methods"], [])
        self.assertEqual(len(result["rejections"]), 1)
        self.assertIn("evidence value", result["rejections"][0]["reason_code"])

    def test_uses_unique_alias_quote_when_name_quote_is_ambiguous(self) -> None:
        source = chunk("血红蛋白(Hb)是指标。血红蛋白参考区间见后文。")
        raw = {"indicators": [{
            "name": "血红蛋白", "item_kind": "atomic_indicator",
            "name_ref": ref(source, "血红蛋白"),
            "aliases": [{"value": "Hb", "source_ref": ref(source, "血红蛋白(Hb)")}],
            "value_types": [], "units": [], "specimens": [], "methods": [],
            "reference_selectors": [],
        }]}
        result = validate_indicator_response(raw, [source])
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["name_source"]["exact_quote"], "血红蛋白(Hb)")
        self.assertEqual(result["rejections"][0]["field_path"], "name_ref_reanchored")

    def test_reanchors_ambiguous_name_without_alias(self) -> None:
        source = chunk("血红蛋白是指标。稍后再次讨论血红蛋白。")
        raw = {"indicators": [{
            "name": "血红蛋白", "item_kind": "atomic_indicator",
            "name_ref": ref(source, "血红蛋白"), "aliases": [], "value_types": [],
            "units": [], "specimens": [], "methods": [], "reference_selectors": [],
        }]}
        result = validate_indicator_response(raw, [source])
        self.assertEqual(len(result["candidates"]), 1)
        anchor = result["candidates"][0]["name_source"]
        self.assertEqual(source.text.count(anchor["exact_quote"]), 1)
        self.assertEqual(anchor["exact_quote"].count("血红蛋白"), 1)

    def test_rejects_unknown_top_level_shape(self) -> None:
        with self.assertRaises(IndicatorContractError):
            validate_indicator_response({"entities": []}, [chunk("text")])


class IndicatorAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = chunk("血浆凝血酶原时间 (PT)。")
        source = {
            "chunk_id": self.source.chunk_id, "chunk_sha256": self.source.chunk_sha256,
            "page_id": self.source.page_id, "printed_page_number": 4,
            "source_pdf_page_number": 21, "char_start": 0, "char_end": 15,
            "exact_quote": "血浆凝血酶原时间 (PT)",
        }
        self.proposals = [{
            "proposal_id": "one", "name": "血浆凝血酶原时间",
            "item_kind": "atomic_indicator", "name_source": source,
            "aliases": [{"value": "PT", "source": source}], "value_types": [],
            "units": [], "specimens": [], "methods": [], "reference_selectors": [],
            "origin": "model",
        }, {
            "proposal_id": "two", "name": "PT", "item_kind": "atomic_indicator",
            "name_source": source, "aliases": [], "value_types": [], "units": [],
            "specimens": [], "methods": [], "reference_selectors": [],
            "origin": "legacy_v02",
        }]

    def test_merges_alias_linked_proposals_and_prefers_chinese_name(self) -> None:
        result = aggregate_indicators(self.proposals)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["canonical_name"], "血浆凝血酶原时间")
        self.assertEqual(result[0]["origins"], ["legacy_v02", "model"])

    def test_index_attaches_alias_but_cannot_create_indicator(self) -> None:
        indicators = aggregate_indicators(self.proposals)
        index_source = {"page_id": "book:index:0209", "printed_page_number": 210,
                        "source_pdf_page_number": 227, "cleaned_sha256": "a" * 64,
                        "char_start": 0, "char_end": 50, "exact_quote": "<tr>PT row</tr>"}
        entries = [{"abbreviation": "PT", "english_full_name": "prothrombin time",
                    "chinese_name": "(血浆)凝血酶原时间", "source": index_source},
                   {"abbreviation": "ALT", "english_full_name": "alanine aminotransferase",
                    "chinese_name": "丙氨酸转氨酶", "source": {**index_source,
                                                             "exact_quote": "<tr>ALT row</tr>"}}]
        attached, unmatched = attach_index_aliases(indicators, entries)
        self.assertEqual(len(attached), 1)
        self.assertEqual([value["value"] for value in attached[0]["index_aliases"]],
                         ["prothrombin time"])
        self.assertEqual([value["abbreviation"] for value in unmatched], ["ALT"])

    def test_index_alias_can_merge_abbreviation_candidate_with_chinese_candidate(self) -> None:
        indicators = aggregate_indicators(self.proposals[:1] + [{
            **self.proposals[1], "name": "INR", "proposal_id": "three",
        }])
        chinese = next(item for item in indicators if item["canonical_name"] == "血浆凝血酶原时间")
        abbreviation = next(item for item in indicators if item["canonical_name"] == "INR")
        chinese["index_aliases"] = [{"value": "INR", "alias_type": "abbreviation",
                                      "origin": "book_index", "source": {
                                          "page_id": "index", "char_start": 0, "char_end": 3,
                                          "exact_quote": "INR"}}]
        merged = merge_catalog_indicators([chinese, abbreviation])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["canonical_name"], "血浆凝血酶原时间")

    def test_label_studio_span_offsets_replay(self) -> None:
        library, _ = build_indicator_library(self.proposals, [], input_hashes={"fixture": "a"})
        tasks = label_studio_tasks(library, {self.source.chunk_id: self.source})
        self.assertEqual(len(tasks), 1)
        for result in tasks[0]["predictions"][0]["result"]:
            value = result["value"]
            self.assertEqual(tasks[0]["data"]["source_text"][value["start"]:value["end"]],
                             value["text"])
            self.assertFalse(result["hidden"])
            self.assertEqual(value["labels"], ["IndicatorName"])
            self.assertEqual(result["from_name"], "indicator_entities")
        self.assertIn('value="IndicatorName" html="指标实体"', LABEL_STUDIO_CONFIG)
        self.assertNotIn("<Choices", LABEL_STUDIO_CONFIG)
        self.assertNotIn("<TextArea", LABEL_STUDIO_CONFIG)
        self.assertEqual(LABEL_STUDIO_CONFIG.count("<Label "), 1)
        self.assertNotIn("candidate_metadata", tasks[0]["data"])
        self.assertNotIn("candidate_id", tasks[0]["data"])
        self.assertNotIn("canonical_name", tasks[0]["data"])
        self.assertEqual(tasks[0]["data"]["task_order"], 1)
        self.assertNotIn("$candidate_metadata", LABEL_STUDIO_CONFIG)

    def test_derives_percentage_and_absolute_indicators_from_table_columns(self) -> None:
        source = chunk(
            '<table><tr><td>细胞类型</td><td>百分数/%</td>'
            '<td>绝对值/\\( \\times 10^{9}/L \\)</td></tr>'
            '<tr><td>中性粒细胞(N)</td><td>40~75</td><td>1.8~6.3</td></tr></table>'
        )
        result = derive_table_column_indicators([source])
        self.assertEqual([item["canonical_name"] for item in result],
                         ["中性粒细胞百分数", "中性粒细胞绝对值"])
        self.assertEqual(result[0]["origins"], ["derived_table_column"])
        self.assertEqual(result[1]["derivation"]["column_header"],
                         "绝对值/\\( \\times 10^{9}/L \\)")

    def test_label_studio_tasks_follow_page_and_source_order(self) -> None:
        early = EvidenceChunk(
            "book:chapter-01:0000:0000", "早指标", hashlib.sha256("早指标".encode()).hexdigest(),
            page_id="book:chapter-01:0000", printed_page=4, page_index=0, start_offset=0,
        )
        late = EvidenceChunk(
            "book:chapter-01:0001:0000", "晚指标", hashlib.sha256("晚指标".encode()).hexdigest(),
            page_id="book:chapter-01:0001", printed_page=5, page_index=1, start_offset=0,
        )

        def indicator(name: str, source: EvidenceChunk) -> dict:
            evidence = {
                "chunk_id": source.chunk_id, "chunk_sha256": source.chunk_sha256,
                "page_id": source.page_id, "printed_page_number": source.printed_page,
                "source_pdf_page_number": None, "char_start": 0, "char_end": len(name),
                "exact_quote": name,
            }
            return {
                "candidate_id": name, "canonical_name": name,
                "item_kind_candidates": ["atomic_indicator"], "status": "candidate",
                "approved": 0, "origins": ["model"],
                "body_occurrences": [{"text": name, "source": evidence, "origin": "model"}],
                "aliases": [], "index_aliases": [], "value_types": [], "units": [],
                "specimens": [], "methods": [], "reference_selectors": [],
            }

        tasks = label_studio_tasks(
            {"indicators": [indicator("晚指标", late), indicator("早指标", early)]},
            {early.chunk_id: early, late.chunk_id: late},
        )
        self.assertEqual([task["data"]["chunk_id"] for task in tasks],
                         [early.chunk_id, late.chunk_id])
        self.assertEqual([task["data"]["task_order"] for task in tasks], [1, 2])

    def test_label_studio_keeps_blank_chunks_and_omits_noncontiguous_derived_names(self) -> None:
        source = chunk("中性粒细胞(N) 40~75 1.8~6.3")
        derived = derive_table_column_indicators([chunk(
            '<table><tr><td>细胞类型</td><td>百分数/%</td>'
            '<td>绝对值/\\( \\times 10^{9}/L \\)</td></tr>'
            '<tr><td>中性粒细胞(N)</td><td>40~75</td><td>1.8~6.3</td></tr></table>'
        )])
        tasks = label_studio_tasks({"indicators": derived}, [source])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["predictions"][0]["result"], [])


class IndexReaderTests(unittest.TestCase):
    def test_reads_only_index_rows_with_source_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            page = root / "pages/cleaned/0204.md"
            page.parent.mkdir(parents=True)
            text = "# 索引\n<table><tr><td>PT (prothrombin time)</td><td>(血浆)凝血酶原时间</td></tr></table>"
            page.write_text(text, encoding="utf-8")
            manifest = {"pages": [{"chapter_page_index": 204, "page_id": "book:index:0204",
                                    "printed_page_number": 205, "source_pdf_page_number": 222,
                                    "cleaned_path": "pages/cleaned/0204.md"}]}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            entries = load_index_entries(manifest_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["abbreviation"], "PT")
        self.assertEqual(entries[0]["english_full_name"], "prothrombin time")
        self.assertEqual(entries[0]["source"]["exact_quote"],
                         "<tr><td>PT (prothrombin time)</td><td>(血浆)凝血酶原时间</td></tr>")

    def test_reads_grouped_two_column_ocr_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            page = root / "pages/cleaned/0213.md"
            page.parent.mkdir(parents=True)
            text = "M\nMCH (mean corpuscular hemoglobin)\nMCV (mean corpuscular volume)\n平均红细胞血红蛋白含量\n平均红细胞容积\n"
            page.write_text(text, encoding="utf-8")
            manifest = {"pages": [{"chapter_page_index": 213, "page_id": "book:index:0213",
                                    "printed_page_number": 214, "source_pdf_page_number": 231,
                                    "cleaned_path": "pages/cleaned/0213.md"}]}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            entries = load_index_entries(manifest_path)
        self.assertEqual([(entry["abbreviation"], entry["chinese_name"]) for entry in entries],
                         [("MCH", "平均红细胞血红蛋白含量"), ("MCV", "平均红细胞容积")])


if __name__ == "__main__":
    unittest.main()
