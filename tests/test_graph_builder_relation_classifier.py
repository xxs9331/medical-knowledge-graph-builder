"""二阶段关系分类实验的确定性候选对测试。"""

from __future__ import annotations

import hashlib
import unittest
from unittest.mock import AsyncMock, patch

from neo4j_graphrag.exceptions import LLMGenerationError

from medical_kg_sourceprep.extraction.graph_builder.relation_classifier import (
    _has_explicit_association_trigger,
    _has_verbatim_trigger,
    _invoke_validated_results_resilient,
    build_relation_pairs,
)
from medical_kg_sourceprep.extraction.graph_builder.schema import load_candidate_graph_schema
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk


class RelationPairTests(unittest.TestCase):
    def test关联关系必须提供证据中的明确触发词(self) -> None:
        evidence = "血小板数量和质量与止血、凝血功能密切相关。"

        self.assertTrue(_has_explicit_association_trigger(evidence, "与止血、凝血功能密切相关"))
        self.assertFalse(_has_explicit_association_trigger(evidence, "同表出现，可能相关"))
        self.assertFalse(_has_explicit_association_trigger("两种贫血并列出现。", "并列出现"))
        self.assertFalse(_has_explicit_association_trigger(evidence, None))

    def test所有正关系必须返回逐字证据(self) -> None:
        evidence = "维生素B12缺乏引起巨幼细胞贫血。"

        self.assertTrue(_has_verbatim_trigger(evidence, "缺乏引起巨幼细胞贫血"))
        self.assertFalse(_has_verbatim_trigger(evidence, "缺乏导致巨幼细胞贫血"))
        self.assertFalse(_has_verbatim_trigger(evidence, ""))

    def test因果候选不得跨表格单元格连接分类名(self) -> None:
        text = (
            "<table><tr><td>大细胞性贫血</td>"
            "<td>维生素B12缺乏引起巨幼细胞贫血</td></tr></table>"
        )
        chunk = EvidenceChunk("test:causal-cell", text, hashlib.sha256(text.encode()).hexdigest())
        nodes = [
            {"candidate_key": "category", "entity_type": "ClinicalContext", "mention": "大细胞性贫血"},
            {"candidate_key": "cause", "entity_type": "ClinicalContext", "mention": "维生素B12缺乏"},
            {"candidate_key": "disease", "entity_type": "Disease", "mention": "巨幼细胞贫血"},
        ]

        pairs = build_relation_pairs(
            chunk=chunk,
            schema=load_candidate_graph_schema(),
            nodes=nodes,
            allowed_relation_types={"CAUSES"},
        )

        self.assertEqual(
            {frozenset((pair.left_key, pair.right_key)) for pair in pairs},
            {frozenset(("cause", "disease"))},
        )

    def test因果候选不连接同侧并列原因(self) -> None:
        text = "慢性感染、炎症引起贫血。"
        chunk = EvidenceChunk("test:causal-coordination", text, hashlib.sha256(text.encode()).hexdigest())
        nodes = [
            {"candidate_key": "infection", "entity_type": "ClinicalContext", "mention": "慢性感染"},
            {"candidate_key": "inflammation", "entity_type": "ClinicalContext", "mention": "炎症"},
            {"candidate_key": "anemia", "entity_type": "ClinicalContext", "mention": "贫血"},
        ]

        pairs = build_relation_pairs(
            chunk=chunk,
            schema=load_candidate_graph_schema(),
            nodes=nodes,
            allowed_relation_types={"CAUSES"},
        )

        identities = {frozenset((pair.left_key, pair.right_key)) for pair in pairs}
        self.assertEqual(
            identities,
            {frozenset(("infection", "anemia")), frozenset(("inflammation", "anemia"))},
        )
        self.assertNotIn(frozenset(("infection", "inflammation")), identities)

    def test复合疾病名可生成词面下位候选(self) -> None:
        text = "贫血。急性失血性贫血。"
        chunk = EvidenceChunk("test:lexical-is-a", text, hashlib.sha256(text.encode()).hexdigest())
        nodes = [
            {
                "candidate_key": "parent",
                "canonical_name": "贫血",
                "entity_type": "Disease",
                "mention": "贫血",
                "source_ref": {"mention_char_start": 0, "mention_char_end": 2},
            },
            {
                "candidate_key": "child",
                "canonical_name": "急性失血性贫血",
                "entity_type": "Disease",
                "mention": "急性失血性贫血",
                "source_ref": {"mention_char_start": 3, "mention_char_end": 10},
            },
        ]

        pairs = build_relation_pairs(
            chunk=chunk,
            schema=load_candidate_graph_schema(),
            nodes=nodes,
            allowed_relation_types={"IS_A"},
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].options, (("IS_A", "RIGHT_TO_LEFT"),))
        self.assertIn("急性失血性贫血", pairs[0].evidence_text)

    def test状态候选必须与同一指标匹配(self) -> None:
        text = "MCV 减小，RDW 正常。"
        chunk = EvidenceChunk("test:state-match", text, hashlib.sha256(text.encode()).hexdigest())
        nodes = [
            {"candidate_key": "mcv", "canonical_name": "平均红细胞容积", "aliases": ["MCV"], "entity_type": "LabIndicator", "mention": "MCV"},
            {"candidate_key": "mcv-low", "canonical_name": "MCV减小", "entity_type": "IndicatorState", "mention": "MCV 减小"},
            {"candidate_key": "rdw-normal", "canonical_name": "RDW正常", "entity_type": "IndicatorState", "mention": "RDW 正常"},
        ]

        pairs = build_relation_pairs(
            chunk=chunk,
            schema=load_candidate_graph_schema(),
            nodes=nodes,
            allowed_relation_types={"HAS_STATE"},
        )

        self.assertEqual(
            {frozenset((pair.left_key, pair.right_key)) for pair in pairs},
            {frozenset(("mcv", "mcv-low"))},
        )

    def test模型超时后自动二分当前批次(self) -> None:
        invoke = AsyncMock(side_effect=[
            LLMGenerationError("timeout"),
            {"left": {}},
            {"right": {}},
        ])
        with patch(
            "medical_kg_sourceprep.extraction.graph_builder.relation_classifier._invoke_validated_results",
            invoke,
        ):
            import asyncio

            result, calls = asyncio.run(_invoke_validated_results_resilient(
                object(),
                lambda pairs: str(len(pairs)),
                ["left", "right"],  # type: ignore[list-item]
            ))

        self.assertEqual(result, {"left": {}, "right": {}})
        self.assertEqual(calls, 3)

    def test只枚举schema允许且共享证据窗口的实体对(self) -> None:
        text = "血清铁降低提示缺铁性贫血。另段文字。"
        chunk = EvidenceChunk("test:relation", text, hashlib.sha256(text.encode()).hexdigest())
        nodes = [
            {"candidate_key": "indicator", "entity_type": "LabIndicator", "mention": "血清铁"},
            {"candidate_key": "state", "entity_type": "IndicatorState", "mention": "血清铁降低"},
            {"candidate_key": "disease", "entity_type": "Disease", "mention": "缺铁性贫血"},
        ]

        pairs = build_relation_pairs(
            chunk=chunk,
            schema=load_candidate_graph_schema(),
            nodes=nodes,
            allowed_relation_types={"HAS_STATE", "INDICATES"},
        )

        options = {
            (pair.left_key, pair.right_key): set(pair.options)
            for pair in pairs
        }
        self.assertIn(("HAS_STATE", "LEFT_TO_RIGHT"), options[("indicator", "state")])
        self.assertIn(("INDICATES", "LEFT_TO_RIGHT"), options[("state", "disease")])
        self.assertTrue(all("血清铁" in pair.evidence_text for pair in pairs if pair.left_key == "indicator"))

    def test不因共享段落而枚举跨行无关实体对(self) -> None:
        text = "贫血分类如下。\n血清铁降低提示缺铁性贫血。\n叶酸降低提示巨幼细胞贫血。"
        chunk = EvidenceChunk("test:local-lines", text, hashlib.sha256(text.encode()).hexdigest())
        nodes = [
            {"candidate_key": "ida", "entity_type": "Disease", "mention": "缺铁性贫血"},
            {"candidate_key": "mega", "entity_type": "Disease", "mention": "巨幼细胞贫血"},
            {"candidate_key": "iron", "entity_type": "IndicatorState", "mention": "血清铁降低"},
            {"candidate_key": "folate", "entity_type": "IndicatorState", "mention": "叶酸降低"},
        ]

        pairs = build_relation_pairs(
            chunk=chunk,
            schema=load_candidate_graph_schema(),
            nodes=nodes,
            allowed_relation_types={"INDICATES"},
        )

        identities = {frozenset((pair.left_key, pair.right_key)) for pair in pairs}
        self.assertEqual(identities, {frozenset(("ida", "iron")), frozenset(("mega", "folate"))})

    def test列表标题只在各子项行内扩展候选(self) -> None:
        text = "(1) 血清铁降低\n1) 摄入不足导致血清铁降低。\n2) 慢性失血导致血清铁降低。"
        chunk = EvidenceChunk("test:list-parent", text, hashlib.sha256(text.encode()).hexdigest())
        nodes = [
            {"candidate_key": "state", "entity_type": "IndicatorState", "mention": "血清铁降低"},
            {"candidate_key": "intake", "entity_type": "ClinicalContext", "mention": "摄入不足"},
            {"candidate_key": "diet", "entity_type": "ClinicalContext", "mention": "缺铁性饮食"},
            {"candidate_key": "loss", "entity_type": "ClinicalContext", "mention": "慢性失血"},
            {"candidate_key": "ulcer", "entity_type": "Disease", "mention": "胃溃疡出血"},
        ]

        pairs = build_relation_pairs(
            chunk=chunk,
            schema=load_candidate_graph_schema(),
            nodes=nodes,
            allowed_relation_types={"CAUSES", "INDICATES"},
        )

        identities = {frozenset((pair.left_key, pair.right_key)) for pair in pairs}
        self.assertIn(frozenset(("state", "intake")), identities)
        self.assertIn(frozenset(("state", "loss")), identities)
        self.assertNotIn(frozenset(("state", "diet")), identities)
        self.assertNotIn(frozenset(("diet", "ulcer")), identities)


if __name__ == "__main__":
    unittest.main()
