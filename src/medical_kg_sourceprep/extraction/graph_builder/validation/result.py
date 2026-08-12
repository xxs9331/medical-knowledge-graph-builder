"""候选准入的统一结果类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(slots=True)
class CandidateNormalization:
    """本地校验的分流结果。

    ``accepted`` 只包含能进入候选图的记录；无法生成可回放身份但仍有足够信息
    交给语义 Judge 修复的单条草稿放在 ``judge_drafts``。保留两个元素的迭代接口，
    使既有 ``records, holds = normalize_*`` 调用仍可工作。
    """

    accepted: list[dict[str, Any]] = field(default_factory=list)
    review_items: list[dict[str, Any]] = field(default_factory=list)
    judge_drafts: list[dict[str, Any]] = field(default_factory=list)
    invalid_rule_keys: set[str] = field(default_factory=set)

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        yield self.accepted
        yield self.review_items
