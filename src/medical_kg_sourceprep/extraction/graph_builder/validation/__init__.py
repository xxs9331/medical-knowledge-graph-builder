"""候选图本地校验包。

按职责拆分为：
- review：审查队列、拒绝和 PARTIAL 状态；
- provenance：原文回放、证据解析和稳定身份；
- nodes：业务实体与 RuleDefinition 节点接纳；
- relationships：普通关系、规则边和复合规则结构。

本文件保留旧 ``graph_builder.validation`` 导入接口，避免调用方迁移。
"""

from .nodes import _catalog_for_prompt, normalize_candidate_nodes
from .provenance import (
    _candidate_key,
    _relation_key,
    _rule_candidate_key,
    _rule_evidence_ref,
    _source_ref,
    _table_state_candidate_key,
)
from .relationships import deterministic_state_relations, normalize_candidate_relationships
from .result import CandidateNormalization
from .review import _hold

__all__ = [
    "_candidate_key",
    "_catalog_for_prompt",
    "_hold",
    "_relation_key",
    "_rule_candidate_key",
    "_rule_evidence_ref",
    "_source_ref",
    "_table_state_candidate_key",
    "CandidateNormalization",
    "deterministic_state_relations",
    "normalize_candidate_nodes",
    "normalize_candidate_relationships",
]
