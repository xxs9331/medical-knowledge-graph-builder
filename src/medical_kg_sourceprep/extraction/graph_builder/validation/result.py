"""候选准入的统一结果类型。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(slots=True)
class CandidateNormalization:
    """本地校验的分流结果。

    一次节点或关系校验可能同时产生三路输出：可写入候选图的记录、供人工审查的
    结构化记录，以及供后续语义 Judge 判断的最小草稿。规则边校验还会额外返回
    结构不完整的 RuleDefinition 候选键，供编排层从最终候选节点中剔除对应规则。

    为兼容既有调用，本类型仍可解包为 ``accepted, review_items``；第三、第四路结果
    必须通过同名属性显式读取，避免旧代码在增加 Judge 流程后改变行为。
    """

    # 已通过本地准入、可以写入 candidate-only 图的节点或关系。这里的“接纳”不等于
    # 人工批准或正式发布，记录仍遵守自身的 publication_status/HOLD 状态。
    accepted: list[dict[str, Any]] = field(default_factory=list)
    # 对拒绝、重复、降级为 PARTIAL 或需要复核的情况生成的可审计记录。
    review_items: list[dict[str, Any]] = field(default_factory=list)
    # 未能进入候选图、但保留了最小身份信息的 Judge 输入；它不是候选图记录。
    judge_drafts: list[dict[str, Any]] = field(default_factory=list)
    # 规则边整体结构校验发现无效时对应的 RuleDefinition candidate_key。普通节点和
    # 普通关系校验不会填充该集合。
    invalid_rule_keys: set[str] = field(default_factory=set)

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        """按旧接口顺序提供候选记录和审查记录，供二元解包使用。"""
        yield self.accepted
        yield self.review_items


def main() -> None:
    """演示统一校验结果的四路分流以及兼容的二元解包。"""
    result = CandidateNormalization(
        accepted=[{
            "candidate_key": "candidate:lab-indicator:serum-iron",
            "entity_type": "LabIndicator",
            "mention": "血清铁",
            "validation_status": "VALID",
            "publication_status": "HOLD",
        }],
        review_items=[{
            "stage": "entity",
            "status": "REVIEW_REQUIRED",
            "reason_code": "duplicate_candidate",
        }],
        judge_drafts=[{
            "stage": "relation",
            "reason_code": "relation_endpoint_not_frozen",
            "candidate": {
                "relation_type": "INDICATES",
                "source": "血清铁降低",
                "target": "缺铁性贫血",
            },
        }],
        invalid_rule_keys={"candidate:rule:incomplete-composite"},
    )

    # 旧调用仍然只能解包出前两路结果；Judge 草稿和无效规则键应显式读取。
    accepted, review_items = result
    output = {
        "二元解包得到的候选记录": accepted,
        "二元解包得到的审查记录": review_items,
        "显式读取的 Judge 草稿": result.judge_drafts,
        "显式读取的无效规则键": sorted(result.invalid_rule_keys),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
