import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk
from medical_kg_sourceprep.extraction.semantic_contract import build_v02_prompt
from medical_kg_sourceprep.extraction.semantic_v04 import (
    augment_catalog, audit_superseded_v02_relations, baseline_model_relations, build_base_catalog,
    recover_structural_relations, validate_endpoints, validate_rules,
)


def chunk(chunk_id, page_id, text, offset=0):
    return EvidenceChunk(chunk_id, text, hashlib.sha256(text.encode()).hexdigest(), page_id=page_id,
                         page_index=int(page_id[-1]), start_offset=offset)


def source_ref(value, quote):
    return {"chunk_id": value.chunk_id, "chunk_sha256": value.chunk_sha256, "exact_quote": quote}


class SemanticV04Tests(unittest.TestCase):
    def catalog(self, entries):
        return {"entries": entries, "catalog_sha256": "x"}

    def test_structural_recovery_does_not_create_page_cartesian_product(self):
        text = "1、项目甲\n【参考区间】\n甲范围 1~2\n【异常结果解读】甲高\n2、项目乙\n【参考区间】\n乙范围 3~4\n"
        value = chunk("c0", "page-0", text)
        def entry(key, kind, exact, origin="derived"):
            start = text.index(exact)
            return {"page_id": "page-0", "candidate_key": key, "candidate_id": key,
                    "entity_type": kind, "text": exact, "origin": origin,
                    "source": {**source_ref(value, exact), "char_start": start, "char_end": start + len(exact)}}
        catalog = self.catalog([entry("a", "TestItem", "项目甲"), entry("b", "TestItem", "项目乙"),
                                entry("ra", "ReferenceRange", "甲范围 1~2"), entry("rb", "ReferenceRange", "乙范围 3~4")])
        relations, review = recover_structural_relations(catalog, (value,))
        self.assertEqual([(x["source_candidate_key"], x["target_candidate_key"]) for x in relations], [("a", "ra"), ("b", "rb")])
        self.assertEqual(review, [])
        self.assertTrue(all(len(item["evidence"]) == 3 for item in relations))

    def test_abnormal_thresholds_are_not_reference_ranges(self):
        text = "1、血小板\n【参考区间】成人 125~350\n【异常结果解读】PLT<100 为减少\n2、平均血小板体积\n【参考区间】成人 7~13fL\n"
        value = chunk("c0", "page-0", text)
        def entry(key, kind, exact):
            start = text.index(exact)
            return {"page_id": "page-0", "candidate_key": key, "candidate_id": key,
                    "entity_type": kind, "text": exact, "origin": "derived",
                    "source": {**source_ref(value, exact), "char_start": start, "char_end": start + len(exact)}}
        catalog = self.catalog([entry("plt", "TestItem", "血小板"), entry("mpv", "TestItem", "平均血小板体积"),
                                entry("normal", "ReferenceRange", "成人 125~350"), entry("bad", "ReferenceRange", "PLT<100"),
                                entry("mpvr", "ReferenceRange", "成人 7~13fL")])
        relations, _ = recover_structural_relations(catalog, (value,))
        pairs = {(x["source_candidate_key"], x["target_candidate_key"]) for x in relations}
        self.assertEqual(pairs, {("plt", "normal"), ("mpv", "mpvr")})

    def test_population_uses_nearest_same_line_label(self):
        text = "男性 4.0~5.0 女性 3.5~4.5"
        value = chunk("c0", "page-0", text)
        def entry(key, kind, exact, occurrence=0):
            start = text.index(exact, occurrence)
            return {"page_id": "page-0", "candidate_key": key, "candidate_id": key,
                    "entity_type": kind, "text": exact, "origin": "model",
                    "source": {**source_ref(value, exact), "char_start": start, "char_end": start + len(exact)}}
        catalog = self.catalog([entry("male", "Population", "男性"), entry("female", "Population", "女性"),
                                entry("mr", "ReferenceRange", "4.0~5.0"), entry("fr", "ReferenceRange", "3.5~4.5")])
        relations, _ = recover_structural_relations(catalog, (value,))
        pairs = {(x["source_candidate_key"], x["target_candidate_key"]) for x in relations}
        self.assertEqual(pairs, {("mr", "male"), ("fr", "female")})

    def test_endpoint_gap_augments_catalog_with_verbatim_candidate(self):
        value = chunk("c0", "page-0", "D-二聚体正常可排除深静脉血栓")
        payload = {"endpoints": [{"candidate_key": "深静脉血栓", "entity_type": "MedicalConcept",
                                  "text": "深静脉血栓", "source_ref": source_ref(value, value.text)}]}
        result = validate_endpoints(payload, (value,), self.catalog([]))
        self.assertEqual(result["counts"], {"accepted": 1, "rejected": 0})
        augmented = augment_catalog(self.catalog([]), [result])
        self.assertEqual(augmented["entries"][0]["origin"], "endpoint-gap")

    def test_extended_rule_semantics_and_endpoints(self):
        text = "D-二聚体正常，对排除深静脉血栓有重要价值。"
        value = chunk("c0", "page-0", text)
        def entry(key, kind):
            start = text.index(key)
            return {"page_id": "page-0", "candidate_key": key, "candidate_id": key,
                    "entity_type": kind, "text": key, "origin": "model",
                    "source": {**source_ref(value, key), "char_start": start, "char_end": start + len(key)}}
        catalog = self.catalog([entry("D-二聚体", "TestItem"), entry("深静脉血栓", "MedicalConcept")])
        payload = {"rules": [{"rule_key": "r1", "entity_type": "InterpretationRule",
            "semantic_type": "DIFFERENTIAL_DIAGNOSIS", "subject_logic": "SINGLE",
            "subject_candidate_keys": ["D-二聚体"], "conclusion_candidate_key": "深静脉血栓",
            "population_candidate_keys": [], "method_candidate_keys": [], "source_ref": source_ref(value, text),
            "components": {"conditions": [{"text": "D-二聚体正常", "source_ref": source_ref(value, "D-二聚体正常")}],
                           "conclusion": {"text": "排除深静脉血栓", "source_ref": source_ref(value, "排除深静脉血栓")}}}]}
        result = validate_rules(payload, (value,), catalog)
        self.assertEqual(result["counts"], {"accepted": 1, "rejected": 0})

        ungrounded = {"rules": [{**payload["rules"][0],
            "subject_candidate_keys": ["深静脉血栓"],
            "conclusion_candidate_key": "深静脉血栓"}]}
        rejected = validate_rules(ungrounded, (value,), catalog)
        self.assertEqual(rejected["counts"], {"accepted": 0, "rejected": 1})

    def test_v02_rule_types_remain_frozen(self):
        prompt = build_v02_prompt(type("Window", (), {"text": "[]"})())
        self.assertIn("DEFINES_AS", prompt)
        self.assertNotIn("DIFFERENTIAL_DIAGNOSIS", prompt)

    def test_v02_cross_page_projection_is_sent_to_review(self):
        value = chunk("c0", "page-0", "项目甲\n【参考区间】1~2")
        def entry(key, kind, exact, candidate_id):
            start = value.text.index(exact)
            return {"page_id": "page-0", "candidate_key": key, "candidate_id": candidate_id,
                    "entity_type": kind, "text": exact, "origin": "derived",
                    "source": {**source_ref(value, exact), "char_start": start, "char_end": start + len(exact)}}
        other = chunk("c1", "page-1", "3~4")
        target = {"page_id": "page-1", "candidate_key": "range", "candidate_id": "range-id",
                  "entity_type": "ReferenceRange", "text": "3~4", "origin": "derived",
                  "source": {**source_ref(other, "3~4"), "char_start": 0, "char_end": 3}}
        catalog = self.catalog([entry("item", "TestItem", "项目甲", "item-id"), target])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "v02.sqlite"
            with sqlite3.connect(database) as db:
                db.execute("CREATE TABLE semantic_edges (edge_id TEXT, source_id TEXT, relation TEXT, target_id TEXT)")
                db.execute("INSERT INTO semantic_edges VALUES ('e', 'item-id', 'ITEM_HAS_REFERENCE_RANGE', 'range-id')")
            result = audit_superseded_v02_relations(database, [], [], catalog)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["reason_code"], "superseded_v02_cross_page_projection")


if __name__ == "__main__":
    unittest.main()
