import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk
from medical_kg_sourceprep.extraction.semantic_v03 import (
    build_entity_catalog, build_relation_prompt, build_rule_prompt, stable_relations,
    validate_relations, validate_rules,
)


def ref(chunk, quote):
    return {"chunk_id": chunk.chunk_id, "chunk_sha256": chunk.chunk_sha256, "exact_quote": quote}


class SemanticV03Tests(unittest.TestCase):
    def setUp(self):
        text = "红细胞计数采用自动血细胞分析法。红细胞计数升高表示异常。"
        self.chunk = EvidenceChunk("chunk-1", text, hashlib.sha256(text.encode()).hexdigest(), page_id="page-1", page_index=0)
        self.other = EvidenceChunk("chunk-2", "红细胞计数采用自动血细胞分析法", hashlib.sha256("红细胞计数采用自动血细胞分析法".encode()).hexdigest(), page_id="page-2", page_index=1)
        extraction = {"candidates": [
            {"candidate_id": "i", "candidate_key": "item", "candidate_type": "entity", "entity_type": "TestItem", "text": "红细胞计数", "text_span": {"chunk_id": "chunk-1", "char_start": 0, "char_end": 5, "exact_quote": "红细胞计数", "chunk_sha256": self.chunk.chunk_sha256}},
            {"candidate_id": "m", "candidate_key": "method", "candidate_type": "entity", "entity_type": "TestMethod", "text": "自动血细胞分析法", "text_span": {"chunk_id": "chunk-1", "char_start": 7, "char_end": 15, "exact_quote": "自动血细胞分析法", "chunk_sha256": self.chunk.chunk_sha256}},
            {"candidate_id": "c", "candidate_key": "abnormal", "candidate_type": "entity", "entity_type": "MedicalConcept", "text": "异常", "text_span": {"chunk_id": "chunk-1", "char_start": 25, "char_end": 27, "exact_quote": "异常", "chunk_sha256": self.chunk.chunk_sha256}},
        ]}
        self.catalog = build_entity_catalog(extraction, (self.chunk, self.other))

    def test_relation_requires_page_local_frozen_endpoints_and_quote(self):
        good = {"source_candidate_key": "item", "target_candidate_key": "method", "relation": "ITEM_MEASURED_BY_METHOD", "source_chunk_id": "chunk-1", "source_chunk_sha256": self.chunk.chunk_sha256, "source_quote": "红细胞计数采用自动血细胞分析法", "relation_cue": "采用"}
        bad = dict(good, source_candidate_key="item", target_candidate_key="method", source_chunk_id="chunk-2", source_chunk_sha256=self.other.chunk_sha256, source_quote=self.other.text)
        no_cue = dict(good, relation_cue="不存在")
        result = validate_relations({"relations": [good, bad, no_cue, dict(good, source_candidate_key="missing")]}, (self.chunk, self.other), self.catalog)
        self.assertEqual(result["counts"], {"accepted": 1, "rejected": 3})
        self.assertTrue(all(item["candidate_id"].startswith("rejected:") for item in result["rejections"]))

    def test_rule_components_and_unique_conclusion(self):
        condition = {"text": "红细胞计数升高", "source_ref": ref(self.chunk, "红细胞计数升高")}
        conclusion = {"text": "表示异常", "source_ref": ref(self.chunk, "表示异常")}
        connector = {"text": "。", "source_ref": ref(self.chunk, "法。红")}
        rule = {"rule_key": "rule-1", "entity_type": "InterpretationRule", "semantic_type": "DEFINES_AS", "subject_logic": "ALL", "subject_candidate_keys": ["item"], "conclusion_candidate_key": "abnormal", "source_ref": ref(self.chunk, self.chunk.text), "components": {"conditions": [condition], "connector": connector, "conclusion": conclusion}}
        result = validate_rules({"rules": [rule]}, (self.chunk,), self.catalog)
        self.assertEqual(result["counts"]["accepted"], 1)
        self.assertEqual(result["candidates"][0]["conclusion_candidate_key"], "abnormal")
        legacy = {"rule_id": "bad", "condition": condition, "conclusion": conclusion}
        self.assertEqual(validate_rules({"rules": [legacy]}, (self.chunk,), self.catalog)["counts"]["rejected"], 1)

    def test_stable_relations_merge_origins_and_evidence(self):
        item = {"page_id": "page-1", "source_candidate_key": "item", "target_candidate_key": "method", "relation": "ITEM_MEASURED_BY_METHOD", "origin": "derived", "source": {"chunk_id": "chunk-1", "exact_quote": "红细胞计数采用自动血细胞分析法"}, "relation_cue": "采用"}
        model = dict(item, origin="model", source=dict(item["source"], exact_quote="红细胞计数采用自动血细胞分析法"))
        result = stable_relations([{"candidates": [item]}, {"candidates": [model]}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["origin"], "model")
        self.assertEqual(len(result[0]["evidence"]), 2)

    def test_prompts_expose_closed_world_contract(self):
        relation_prompt = build_relation_prompt("page-1", (self.chunk,), self.catalog["entries"])
        rule_prompt = build_rule_prompt("page-1", (self.chunk,), self.catalog["entries"])
        self.assertIn("ENTITY_CATALOG", relation_prompt)
        self.assertIn("both endpoint texts", relation_prompt)
        self.assertIn("legacy", rule_prompt)
        self.assertIn("exactly one conclusion object", rule_prompt)


if __name__ == "__main__":
    unittest.main()
