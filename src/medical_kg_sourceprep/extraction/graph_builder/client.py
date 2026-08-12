"""DeepSeek client setup and response-shape compatibility handling."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import httpx
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM

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


class _GraphRagIdCompletingLLM:
    """Supply missing transient GraphRAG node IDs without changing candidate IDs."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.last_response_diagnostic: dict[str, Any] = {}

    async def ainvoke(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        result = await self.delegate.ainvoke(*args, **kwargs)
        self.last_response_diagnostic = _response_shape_diagnostic(getattr(result, "content", None))
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
            relationships = payload.get("relationships")
            if isinstance(relationships, list):
                for relationship in relationships:
                    if not isinstance(relationship, dict):
                        continue
                    for canonical_field, aliases in {
                        "start_node_id": ("source", "source_node_id"),
                        "end_node_id": ("target", "target_node_id"),
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
                    for field in ("exact_quote", "relation_cue"):
                        if properties.get(field) is None and field in properties:
                            properties.pop(field)
                            changed = True
                        if relationship.get(field) is None and field in relationship:
                            relationship.pop(field)
                            changed = True
                        if field in relationship:
                            properties.setdefault(field, relationship.pop(field))
                            changed = True
            if changed:
                return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
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
