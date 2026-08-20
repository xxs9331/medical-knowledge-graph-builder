"""DeepSeek client setup and response-shape compatibility handling."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import httpx
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM
from openai import AsyncOpenAI

from .contract import (
    DEFAULT_TIMEOUT_SECONDS,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    PROJECT_ROOT,
    GraphBuilderConfigurationError,
)


@contextmanager
def _without_proxy_environment():
    """Prevent GraphRAG's internally-created sync client from reading proxies."""
    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    saved = {name: os.environ.pop(name) for name in names if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


@contextmanager
def _without_all_proxy_environment():
    """让 httpx 使用 HTTPS_PROXY，避免因未安装 socksio 而误选 ALL_PROXY。"""
    names = ("ALL_PROXY", "all_proxy")
    saved = {name: os.environ.pop(name) for name in names if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


@dataclass(slots=True)
class DeepSeekGraphBuilderClient:
    """Own the GraphRAG LLM and its no-proxy asynchronous HTTP client."""

    llm: OpenAILLM
    http_client: httpx.AsyncClient

    async def aclose(self) -> None:
        try:
            await self.llm.aclose()
        finally:
            close = getattr(self.http_client, "aclose", None)
            if close is not None:
                await close()


class _OpenCodeLunaLLM:
    """把 OpenCode Go Responses API 适配为 GraphRAG 的 ``ainvoke`` 接口。"""

    def __init__(self, *, client: Any, model_name: str, reasoning_effort: str) -> None:
        self.client = client
        self.model_name: str = model_name
        self.reasoning_effort: str = reasoning_effort

    async def ainvoke(self, prompt: str, **_kwargs: Any) -> SimpleNamespace:
        response = await self.client.responses.create(
            model=self.model_name,
            input=prompt,
            reasoning={"effort": self.reasoning_effort},
        )
        content = getattr(response, "output_text", "")
        if not isinstance(content, str) or not content.strip():
            raise GraphBuilderConfigurationError("opencode_luna_text_missing")
        response_usage = getattr(response, "usage", None)
        usage = None
        if response_usage is not None:
            usage = SimpleNamespace(
                request_tokens=getattr(response_usage, "input_tokens", 0),
                response_tokens=getattr(response_usage, "output_tokens", 0),
                total_tokens=getattr(response_usage, "total_tokens", 0),
            )
        return SimpleNamespace(
            content=content,
            usage=usage,
        )

    async def aclose(self) -> None:
        await self.client.close()


@dataclass(slots=True)
class OpenCodeGraphBuilderClient:
    """持有 OpenCode Go Luna 适配器，与现有关系分类客户端接口一致。"""

    llm: _OpenCodeLunaLLM

    async def aclose(self) -> None:
        await self.llm.aclose()


class _GraphRagIdCompletingLLM:
    """Supply missing transient GraphRAG node IDs without changing candidate IDs."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.last_response_diagnostic: dict[str, Any] = {}

    async def ainvoke(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        result = await self.delegate.ainvoke(*args, **kwargs)
        self.last_response_diagnostic = _response_shape_diagnostic(getattr(result, "content", None))
        usage = getattr(result, "usage", None)
        self.last_response_diagnostic["usage"] = {
            "input_tokens": getattr(usage, "request_tokens", None),
            "output_tokens": getattr(usage, "response_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        try:
            payload = json.loads(result.content)
            changed = False
            if (
                isinstance(payload, dict)
                and "relationships" not in payload
                and isinstance(payload.get("edges"), list)
            ):
                payload["relationships"] = payload.pop("edges")
                changed = True
            nodes = payload.get("nodes") if isinstance(payload, dict) else None
            if not isinstance(nodes, list):
                return result
            for index, node in enumerate(nodes):
                if isinstance(node, dict) and not isinstance(node.get("id"), str):
                    node["id"] = f"transient-node-{index}"
                    changed = True
                if not isinstance(node, dict):
                    continue
                properties = node.get("properties")
                if not isinstance(properties, dict):
                    properties = {}
                    node["properties"] = properties
                    changed = True
                # 有些模型会忽略 Neo4jGraph 的 properties 包装，直接把业务字段放在节点顶层。
                # 只搬运 Schema 已声明的属性；未知顶层字段仍保留，使下游结构校验能报告问题。
                for field in (
                    "mention", "extraction_reason", "canonical_name_candidate", "exact_quote",
                    "exact_quote_occurrence_index", "mention_occurrence_index", "source_char_start",
                    "source_char_end", "bound_indicator_mention", "rule_stage_candidate", "rule_logic_candidate",
                    "rule_inputs_json", "rule_outputs_json", "rule_excluded_outputs_json",
                    "rule_expression", "rule_name",
                    "rule_evidence_json", "table_state_evidence_json",
                    "derived_entity_evidence_json",
                ):
                    if field in node and field not in properties:
                        properties[field] = node.pop(field)
                        changed = True
                table_evidence = properties.get("table_state_evidence_json")
                if table_evidence == "":
                    properties.pop("table_state_evidence_json")
                    table_evidence = None
                    changed = True
                if isinstance(table_evidence, (dict, list)):
                    # GraphRAG 的节点属性只能是标量；模型常把该字段当嵌套 JSON 返回。
                    properties["table_state_evidence_json"] = json.dumps(
                        table_evidence, ensure_ascii=False, separators=(",", ":")
                    )
                    changed = True
                derived_evidence = properties.get("derived_entity_evidence_json")
                if derived_evidence == "":
                    properties.pop("derived_entity_evidence_json")
                    derived_evidence = None
                    changed = True
                if isinstance(derived_evidence, (dict, list)):
                    # 与表格证据相同，GraphRAG 节点属性只能保存 JSON 编码后的标量字符串。
                    properties["derived_entity_evidence_json"] = json.dumps(
                        derived_evidence, ensure_ascii=False, separators=(",", ":")
                    )
                    changed = True
                for field in (
                    "rule_inputs_json", "rule_outputs_json", "rule_excluded_outputs_json",
                ):
                    endpoint_value = properties.get(field)
                    if isinstance(endpoint_value, list):
                        # GraphRAG 节点属性是标量；结构化端点数组统一编码一次后交给硬校验解析。
                        properties[field] = json.dumps(
                            endpoint_value, ensure_ascii=False, separators=(",", ":")
                        )
                        changed = True
            relationships = payload.get("relationships")
            if isinstance(relationships, list):
                for relationship in relationships:
                    if not isinstance(relationship, dict):
                        continue
                    # Neo4jRelationship 没有关系 id 字段；联合抽取时模型常为边生成
                    # 仅供响应内部展示的 rel_1 等瞬时 ID。该值不参与候选身份，移除后
                    # 仍由端点、类型和证据生成稳定 relation key。
                    if "id" in relationship:
                        relationship.pop("id")
                        changed = True
                    for canonical_field, aliases in {
                        "start_node_id": ("source", "source_node_id", "source_node"),
                        "end_node_id": ("target", "target_node_id", "target_node"),
                        "type": ("label", "relationship_type"),
                    }.items():
                        if canonical_field in relationship:
                            continue
                        for alias in aliases:
                            if alias in relationship:
                                relationship[canonical_field] = relationship.pop(alias)
                                changed = True
                                break
                    properties = relationship.get("properties")
                    if not isinstance(properties, dict):
                        properties = {}
                        relationship["properties"] = properties
                        changed = True
                    # 模型有时会把 Schema 声明的关系属性放在关系顶层。统一搬回
                    # properties 后再交给 Neo4jGraph 解析；这里只修正包装位置，
                    # 字段值是否可回放仍由后续本地关系校验负责。
                    for field in (
                        "exact_quote",
                        "exact_quote_occurrence_index",
                        "source_char_start",
                        "source_char_end",
                        "rule_evidence_role",
                        "relation_evidence_json",
                    ):
                        if properties.get(field) is None and field in properties:
                            properties.pop(field)
                            changed = True
                        if relationship.get(field) is None and field in relationship:
                            relationship.pop(field)
                            changed = True
                        if field in relationship:
                            properties.setdefault(field, relationship.pop(field))
                            changed = True
                    relation_evidence = properties.get("relation_evidence_json")
                    if isinstance(relation_evidence, (dict, list)):
                        properties["relation_evidence_json"] = json.dumps(
                            relation_evidence, ensure_ascii=False, separators=(",", ":")
                        )
                        changed = True
            if changed:
                return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False), usage=usage)
        except (AttributeError, TypeError, json.JSONDecodeError):
            pass
        return result


def _response_shape_diagnostic(content: Any) -> dict[str, Any]:
    """Keep only safe structural facts about a model response for HOLD artifacts."""
    raw = content if isinstance(content, str) else ""
    diagnostic: dict[str, Any] = {
        "parse_phase": "adapter_json",
        "reason_code": "response_not_json_object",
        "response_sha256": hashlib.sha256(raw.encode()).hexdigest() if raw else None,
        "json_top_level_fields": [],
        "json_top_level_field_types": {},
        "missing_fields": ["nodes", "relationships"],
    }
    if not raw:
        diagnostic["reason_code"] = "response_content_missing"
        return diagnostic
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        diagnostic["reason_code"] = "response_json_invalid"
        return diagnostic
    if not isinstance(payload, Mapping):
        diagnostic["reason_code"] = "response_json_not_object"
        return diagnostic
    fields = sorted(str(field) for field in payload)
    diagnostic["json_top_level_fields"] = fields
    diagnostic["json_top_level_field_types"] = {
        str(field): type(value).__name__ for field, value in sorted(payload.items(), key=lambda item: str(item[0]))
    }
    item_shapes: dict[str, dict[str, list[str]]] = {}
    for field in ("nodes", "relationships", "edges"):
        items = payload.get(field)
        if not isinstance(items, list):
            continue
        field_types: dict[str, set[str]] = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            for item_field, item_value in item.items():
                field_types.setdefault(str(item_field), set()).add(type(item_value).__name__)
        item_shapes[field] = {
            item_field: sorted(value_types) for item_field, value_types in sorted(field_types.items())
        }
    diagnostic["json_item_field_types"] = item_shapes
    missing = [field for field in ("nodes", "relationships") if field not in payload]
    nodes = payload.get("nodes")
    if isinstance(nodes, list):
        missing.extend(
            f"nodes[{index}].properties"
            for index, node in enumerate(nodes)
            if isinstance(node, Mapping) and "properties" not in node
        )
    diagnostic["missing_fields"] = missing
    diagnostic["reason_code"] = "response_shape_observed"
    return diagnostic


def load_deepseek_api_key(*, env: Mapping[str, str] | None = None) -> str:
    """Load the key from the root .env without allowing command-line secrets."""
    environment = env
    if environment is None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        environment = os.environ
    key = environment.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise GraphBuilderConfigurationError("DEEPSEEK_API_KEY is required")
    return key


def load_sub2api_api_key(*, env: Mapping[str, str] | None = None) -> str:
    """从环境变量读取 Sub2API Key，不允许通过命令行参数传入密钥。"""
    environment = os.environ if env is None else env
    key = environment.get("SUB2API_API_KEY", "").strip()
    if not key:
        raise GraphBuilderConfigurationError("SUB2API_API_KEY is required")
    return key


def load_dashscope_api_key(*, env: Mapping[str, str] | None = None) -> str:
    """默认从根目录 .env 加载百炼 Key，不允许通过命令行或工件传入。"""
    environment = env
    if environment is None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        environment = os.environ
    key = environment.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise GraphBuilderConfigurationError("DASHSCOPE_API_KEY is required")
    return key


def load_opencode_go_api_key(*, auth_path: Path | None = None) -> str:
    """读取 OpenCode Go 本机认证；密钥不得写入运行工件。"""
    path = auth_path or Path.home() / ".local/share/opencode/auth.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphBuilderConfigurationError(f"invalid_opencode_auth:{path}") from exc
    credential = payload.get("opencode-go") if isinstance(payload, Mapping) else None
    if not isinstance(credential, Mapping):
        raise GraphBuilderConfigurationError(f"opencode_go_credential_missing:{path}")
    key = next(
        (
            credential.get(name)
            for name in ("apiKey", "api_key", "key")
            if isinstance(credential.get(name), str) and credential.get(name)
        ),
        None,
    )
    if not isinstance(key, str):
        raise GraphBuilderConfigurationError(f"opencode_go_key_missing:{path}")
    return key


def create_deepseek_graph_builder(
    *,
    env: Mapping[str, str] | None = None,
    http_client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    llm_factory: Callable[..., OpenAILLM] = OpenAILLM,
    api_key_loader: Callable[..., str] = load_deepseek_api_key,
) -> DeepSeekGraphBuilderClient:
    """Create the official OpenAI-compatible DeepSeek client for GraphRAG."""
    key = api_key_loader(env=env)
    http_client = http_client_factory(
        trust_env=False,
        timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
    )
    with _without_proxy_environment():
        llm = llm_factory(
            model_name=DEEPSEEK_MODEL,
            model_params={
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "extra_body": {"thinking": {"type": "disabled"}},
            },
            api_key=key,
            base_url=DEEPSEEK_BASE_URL,
            http_client=http_client,
        )
    return DeepSeekGraphBuilderClient(llm=llm, http_client=http_client)


def create_luna_graph_builder(
    *,
    env: Mapping[str, str] | None = None,
    base_url: str = "https://api.swlyes.top/v1",
    model_name: str = "gpt-5.6-luna",
    reasoning_effort: str = "high",
    timeout_seconds: float = 120.0,
    http_client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    llm_factory: Callable[..., OpenAILLM] = OpenAILLM,
    api_key_loader: Callable[..., str] = load_sub2api_api_key,
) -> DeepSeekGraphBuilderClient:
    """创建通过 Sub2API 调用 Luna 的 GraphRAG 客户端。"""
    key = api_key_loader(env=env)
    http_client = http_client_factory(
        trust_env=False,
        timeout=httpx.Timeout(timeout_seconds),
    )
    with _without_proxy_environment():
        llm = llm_factory(
            model_name=model_name,
            model_params={"reasoning_effort": reasoning_effort},
            api_key=key,
            base_url=base_url,
            http_client=http_client,
        )
    return DeepSeekGraphBuilderClient(llm=llm, http_client=http_client)


def create_qwen_flash_graph_builder(
    *,
    env: Mapping[str, str] | None = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "qwen-flash",
    timeout_seconds: float = 120.0,
    http_client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    llm_factory: Callable[..., OpenAILLM] = OpenAILLM,
    api_key_loader: Callable[..., str] = load_dashscope_api_key,
) -> DeepSeekGraphBuilderClient:
    """创建百炼 qwen-flash 非思考 JSON Mode 客户端。"""
    key = api_key_loader(env=env)
    http_client = http_client_factory(
        trust_env=False,
        timeout=httpx.Timeout(timeout_seconds),
    )
    with _without_proxy_environment():
        llm = llm_factory(
            model_name=model_name,
            model_params={
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "extra_body": {"enable_thinking": False},
            },
            api_key=key,
            base_url=base_url,
            http_client=http_client,
        )
    return DeepSeekGraphBuilderClient(llm=llm, http_client=http_client)


def create_opencode_luna_graph_builder(
    *,
    model_name: str = "gpt-5.6-luna",
    reasoning_effort: str = "high",
    base_url: str = "https://opencode.ai/zen/go/v1",
    timeout_seconds: float = 180.0,
    auth_path: Path | None = None,
    http_client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    openai_client_factory: Callable[..., Any] = AsyncOpenAI,
    api_key_loader: Callable[..., str] = load_opencode_go_api_key,
) -> OpenCodeGraphBuilderClient:
    """通过 OpenCode Go Responses API 创建无状态 Luna 客户端。"""
    key = api_key_loader(auth_path=auth_path)
    # OpenCode Go 经 HTTPS_PROXY 访问；屏蔽 SOCKS ALL_PROXY，避免要求额外的 socksio。
    with _without_all_proxy_environment():
        http_client = http_client_factory(
            trust_env=True,
            timeout=httpx.Timeout(timeout_seconds),
        )
    client = openai_client_factory(
        api_key=key,
        base_url=base_url,
        default_headers={"User-Agent": "opencode-ai/1.0"},
        http_client=http_client,
        max_retries=0,
    )
    return OpenCodeGraphBuilderClient(
        llm=_OpenCodeLunaLLM(
            client=client,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
    )
