"""第一章 v0.8 冻结实体候选图工作流测试。"""

from __future__ import annotations

import hashlib
import unittest

from medical_kg_sourceprep.extraction.graph_builder.runner.chapter_candidate_graph import (
    _canonicalize_relationships,
    build_frozen_nodes_by_chunk,
    run_chapter_candidate_graph,
)
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk


class ChapterCandidateGraphTests(unittest.TestCase):
    def test拒绝未知模型供应商(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "chapter_graph_provider_invalid"):
            import asyncio

            asyncio.run(run_chapter_candidate_graph(provider="unknown", plan_only=True))

    def test映射生成逐mention端点和规范实体证据链接(self) -> None:
        text = "血清铁降低。"
        chunk = EvidenceChunk("chapter:test", text, hashlib.sha256(text.encode()).hexdigest())
        canonical = {
            "mention_to_canonical_links": [{
                "mention_id": "mention:1",
                "canonical_id": "entity:1",
                "derivation": "DIRECT_MENTION",
            }]
        }
        mentions = [{
            "mention_id": "mention:1",
            "chunk_id": chunk.chunk_id,
            "start": 0,
            "end": 3,
            "exact_quote": "血清铁",
        }]
        entities = [{
            "canonical_id": "entity:1",
            "canonical_name": "血清铁",
            "entity_type": "LabIndicator",
            "review_status": "PENDING",
        }]

        by_chunk, links = build_frozen_nodes_by_chunk(
            canonical=canonical,
            mentions=mentions,
            entities=entities,
            chunks_by_id={chunk.chunk_id: chunk},
        )

        node = by_chunk[chunk.chunk_id][0]
        self.assertEqual(node["canonical_id"], "entity:1")
        self.assertEqual(node["mention"], "血清铁")
        self.assertEqual(links[0]["source_ref"]["mention_char_start"], 0)

    def test逐mention关系折叠为规范实体关系(self) -> None:
        nodes = [
            {"candidate_key": "mention:1:entity:1", "canonical_id": "entity:1"},
            {"candidate_key": "mention:2:entity:2", "canonical_id": "entity:2"},
        ]
        relationships = [{
            "candidate_key": "old",
            "relation_type": "HAS_STATE",
            "source_candidate_key": "mention:1:entity:1",
            "target_candidate_key": "mention:2:entity:2",
            "source_ref": {"chunk_id": "chapter:test", "char_start": 0, "char_end": 6},
        }]

        result = _canonicalize_relationships(relationships, nodes)

        self.assertEqual(result[0]["source_candidate_key"], "entity:1")
        self.assertEqual(result[0]["target_candidate_key"], "entity:2")
        self.assertTrue(result[0]["candidate_key"].startswith("relation:"))

    def test嵌套回链不作为关系分类端点(self) -> None:
        text = "平均红细胞血红蛋白浓度(MCHC)。"
        chunk = EvidenceChunk("chapter:test", text, hashlib.sha256(text.encode()).hexdigest())
        canonical = {
            "mention_to_canonical_links": [
                {"mention_id": "mention:outer", "canonical_id": "entity:mchc", "derivation": "PARENTHETICAL_ALIAS"},
                {"mention_id": "mention:outer", "canonical_id": "entity:rbc", "derivation": "NESTED_SOURCE_MENTION"},
            ]
        }
        mentions = [{
            "mention_id": "mention:outer",
            "chunk_id": chunk.chunk_id,
            "start": 0,
            "end": len(text) - 1,
            "exact_quote": text[:-1],
        }]
        entities = [
            {"canonical_id": "entity:mchc", "canonical_name": "平均红细胞血红蛋白浓度", "entity_type": "LabIndicator"},
            {"canonical_id": "entity:rbc", "canonical_name": "红细胞", "entity_type": "LabIndicator"},
        ]

        by_chunk, links = build_frozen_nodes_by_chunk(
            canonical=canonical,
            mentions=mentions,
            entities=entities,
            chunks_by_id={chunk.chunk_id: chunk},
        )

        self.assertEqual([item["canonical_id"] for item in by_chunk[chunk.chunk_id]], ["entity:mchc"])
        self.assertEqual({item["canonical_id"] for item in links}, {"entity:mchc", "entity:rbc"})

    def test缩写不能从更长缩写内部充当关系端点(self) -> None:
        text = "MCHC 正常。"
        chunk = EvidenceChunk("chapter:test", text, hashlib.sha256(text.encode()).hexdigest())
        canonical = {
            "mention_to_canonical_links": [
                {"mention_id": "mention:mch", "canonical_id": "entity:mch", "derivation": "DIRECT_MENTION"},
                {"mention_id": "mention:mchc", "canonical_id": "entity:mchc", "derivation": "DIRECT_MENTION"},
            ]
        }
        mentions = [
            {"mention_id": "mention:mch", "chunk_id": chunk.chunk_id, "start": 0, "end": 3, "exact_quote": "MCH"},
            {"mention_id": "mention:mchc", "chunk_id": chunk.chunk_id, "start": 0, "end": 4, "exact_quote": "MCHC"},
        ]
        entities = [
            {"canonical_id": "entity:mch", "canonical_name": "平均红细胞血红蛋白含量", "entity_type": "LabIndicator"},
            {"canonical_id": "entity:mchc", "canonical_name": "平均红细胞血红蛋白浓度", "entity_type": "LabIndicator"},
        ]

        by_chunk, links = build_frozen_nodes_by_chunk(
            canonical=canonical,
            mentions=mentions,
            entities=entities,
            chunks_by_id={chunk.chunk_id: chunk},
        )

        self.assertEqual([item["canonical_id"] for item in by_chunk[chunk.chunk_id]], ["entity:mchc"])
        self.assertEqual({item["canonical_id"] for item in links}, {"entity:mch", "entity:mchc"})


if __name__ == "__main__":
    unittest.main()
