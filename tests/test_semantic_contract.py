import hashlib
import unittest

from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk
from medical_kg_sourceprep.extraction.semantic_contract import ContractError, build_v02_prompt, validate_v02


def ref(chunk, quote):
    return {"chunk_id": chunk.chunk_id, "chunk_sha256": chunk.chunk_sha256, "exact_quote": quote}


class SemanticContractTests(unittest.TestCase):
    def setUp(self):
        text = "红细胞计数采用自动血细胞分析法，成人参考区间为4.0-5.5×10^12/L；红细胞计数升高表示异常。"
        self.chunk = EvidenceChunk("page-1", text, hashlib.sha256(text.encode()).hexdigest(), page_id="p1")

    def entity(self, key, kind, text):
        quote = "红细胞计数采用自动血细胞分析法" if text in {"红细胞计数", "自动血细胞分析法"} else text
        return {"candidate_key": key, "entity_type": kind, "text": text, "source_ref": ref(self.chunk, quote)}

    def test_candidate_keys_and_relation_evidence_are_replayed(self):
        payload = {
            "entities": [self.entity("item", "TestItem", "红细胞计数"), self.entity("method", "TestMethod", "自动血细胞分析法")],
            "rules": [],
            "relations": [{"source_candidate_key": "item", "target_candidate_key": "method", "relation": "ITEM_MEASURED_BY_METHOD", "source_chunk_id": "page-1", "source_chunk_sha256": self.chunk.chunk_sha256, "source_quote": "红细胞计数采用自动血细胞分析法", "relation_cue": "采用"}],
        }
        result = validate_v02(payload, [self.chunk])
        self.assertEqual(result["counts"], {"accepted": 3, "rejected": 0})
        self.assertTrue(result["candidates"][2]["candidate_id"].startswith("candidate:"))
        self.assertEqual(result["candidates"][0]["text_span"]["exact_quote"], "红细胞计数")
        self.assertGreater(len(result["candidates"][0]["source"]["exact_quote"]),
                           len(result["candidates"][0]["text_span"]["exact_quote"]))

    def test_relation_rejects_missing_cue_cross_chunk_and_wrong_direction(self):
        first = self.chunk
        second_text = "红细胞计数采用自动血细胞分析法"
        second = EvidenceChunk("page-2", second_text, hashlib.sha256(second_text.encode()).hexdigest(), page_id="p2")
        entities = [self.entity("item", "TestItem", "红细胞计数"), self.entity("method", "TestMethod", "自动血细胞分析法")]
        payload = {"entities": entities, "rules": [], "relations": [
            {"source_candidate_key": "item", "target_candidate_key": "method", "relation": "ITEM_MEASURED_BY_METHOD", "source_chunk_id": "page-2", "source_chunk_sha256": second.chunk_sha256, "source_quote": second.text, "relation_cue": "缺失"},
            {"source_candidate_key": "method", "target_candidate_key": "item", "relation": "ITEM_MEASURED_BY_METHOD", "source_chunk_id": "page-1", "source_chunk_sha256": first.chunk_sha256, "source_quote": "红细胞计数采用自动血细胞分析法", "relation_cue": "采用"},
        ]}
        result = validate_v02(payload, [first, second])
        self.assertEqual(result["counts"]["rejected"], 2)
        self.assertTrue(all(item["candidate_id"].startswith("rejected:relation:") for item in result["rejections"]))
        self.assertEqual(len({item["candidate_id"] for item in result["rejections"]}), 2)
        self.assertTrue(all("raw_candidate" in item and "candidate_summary" in item for item in result["rejections"]))
        self.assertEqual({item["page_id"] for item in result["rejections"]}, {"p1", "p2"})

    def test_rule_requires_each_component_and_semantic_trigger(self):
        condition = ref(self.chunk, "红细胞计数升高")
        conclusion = ref(self.chunk, "表示异常")
        connector = ref(self.chunk, "；")
        rule = {"candidate_key": "r1", "entity_type": "InterpretationRule", "text": "红细胞计数升高表示异常", "semantic_type": "DEFINES_AS", "subject_logic": "ALL", "source_ref": ref(self.chunk, "红细胞计数升高表示异常"), "components": {"conditions": {"text": "红细胞计数升高", "source_ref": condition}, "conclusion": {"text": "表示异常", "source_ref": conclusion}, "connector": {"text": "；", "source_ref": connector}}}
        result = validate_v02({"entities": [], "rules": [rule], "relations": []}, [self.chunk])
        self.assertEqual(result["counts"]["accepted"], 1)
        rule["components"]["conclusion"] = {"text": "计数升高", "source_ref": ref(self.chunk, "红细胞计数升高")}
        self.assertEqual(validate_v02({"entities": [], "rules": [rule], "relations": []}, [self.chunk])["counts"]["rejected"], 1)

    def test_reference_range_rejects_labels_and_accepts_structured_numeric(self):
        good = {**self.entity("range", "ReferenceRange", "4.0-5.5×10^12/L"), "low": 4.0, "high": 5.5, "unit": "×10^12/L", "applies_to": "成人"}
        result = validate_v02({"entities": [good], "rules": [], "relations": []}, [self.chunk])
        self.assertEqual(result["counts"]["accepted"], 1)
        bad = {**self.entity("bad", "ReferenceRange", "参考区间"), "low": 1, "high": 2}
        result = validate_v02({"entities": [bad], "rules": [], "relations": []}, [self.chunk])
        self.assertEqual(result["counts"]["rejected"], 1)

    def test_page_limits_are_atomic_and_prompt_has_chinese_contract(self):
        with self.assertRaisesRegex(ContractError, "limit"):
            validate_v02({"entities": [self.entity(str(i), "TestItem", "红细胞计数") for i in range(129)], "rules": [], "relations": []}, [self.chunk])
        prompt = build_v02_prompt(type("Window", (), {"text": "示例"})())
        self.assertIn("candidate_key", prompt)
        self.assertIn("关系引文必须同时含两个端点", prompt)
        self.assertIn("参考区间/参考范围标题", prompt)


if __name__ == "__main__":
    unittest.main()
