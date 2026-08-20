"""评测流程的工件读取、身份绑定与阶段可用性校验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contract import GraphBuilderConfigurationError
from ..contract import CANDIDATE_RUN_VERSION


def load_json_object(path: Path) -> dict[str, Any]:
    """读取一个 JSON 对象；数组或标量不能作为评测工件。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GraphBuilderConfigurationError(f"expected JSON object: {path}")
    return value


def artifact_matches_graph(
    path: Path, graph_sha256: str, *, hash_path: tuple[str, ...]
) -> dict[str, Any] | None:
    """仅当阶段工件绑定当前候选图哈希时返回该工件。"""
    if not path.is_file():
        return None
    document = load_json_object(path)
    # 不同工件的输入哈希位置不同：Judge 使用 input.graph_sha256，遗漏审查使用
    # input_graph_sha256。逐层读取可以复用同一套身份校验逻辑。
    current: Any = document
    for key in hash_path:
        current = current.get(key) if isinstance(current, dict) else None
    return document if current == graph_sha256 else None


def first_extraction_is_usable(output_dir: Path) -> bool:
    """判断首次抽取是否完整成功，能够作为后续审查的稳定基线。"""
    graph_path = output_dir / "graph.json"
    review_path = output_dir / "review-queue.json"
    if not graph_path.is_file() or not review_path.is_file():
        return False
    manifest_path = output_dir / "run-manifest.json"
    if manifest_path.is_file():
        manifest = load_json_object(manifest_path)
        if manifest.get("schema_version") != CANDIDATE_RUN_VERSION:
            return False
    items = load_json_object(review_path).get("items", [])
    # 首轮任一模型阶段失败都会造成基线不完整，必须重新抽取。
    return isinstance(items, list) and not any(
        isinstance(item, dict)
        and str(item.get("reason_code", "")).endswith("_model_response_invalid")
        for item in items
    )


def second_extraction_is_usable(output_dir: Path) -> bool:
    """判断二次抽取是否至少完成了构建并集所需的实体基础阶段。"""
    graph_path = output_dir / "graph.json"
    review_path = output_dir / "review-queue.json"
    if not graph_path.is_file() or not review_path.is_file():
        return False
    items = load_json_object(review_path).get("items", [])
    # 二次结果最终与首次结果取并集，因此允许后续关系阶段部分失败；实体阶段失败时
    # 后续端点都没有可靠基础，整轮结果不可使用。
    return isinstance(items, list) and not any(
        isinstance(item, dict) and item.get("reason_code") == "entity_phase_model_response_invalid"
        for item in items
    )
