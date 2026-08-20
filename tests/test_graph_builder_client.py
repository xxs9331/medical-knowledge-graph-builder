import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from medical_kg_sourceprep.extraction.graph_builder.client import (
    _OpenCodeLunaLLM,
    create_opencode_luna_graph_builder,
    load_dashscope_api_key,
    load_opencode_go_api_key,
)


class _FakeResponses:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text='{"results":[]}',
            usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class OpenCodeLunaClientTests(unittest.TestCase):
    def test_dashscope_key_can_be_loaded_from_explicit_test_environment(self) -> None:
        self.assertEqual(
            load_dashscope_api_key(env={"DASHSCOPE_API_KEY": "dashscope-key"}),
            "dashscope-key",
        )

    def test_loads_local_opencode_go_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            auth_path = Path(temporary_directory) / "auth.json"
            auth_path.write_text(
                json.dumps({"opencode-go": {"key": "subscription-key"}}),
                encoding="utf-8",
            )

            self.assertEqual(
                load_opencode_go_api_key(auth_path=auth_path),
                "subscription-key",
            )

    def test_factory_uses_responses_api_model_name(self) -> None:
        captured: dict[str, object] = {}
        fake_client = _FakeOpenAIClient()

        def http_client_factory(**kwargs: object) -> object:
            captured["http"] = kwargs
            return object()

        def openai_client_factory(**kwargs: object) -> _FakeOpenAIClient:
            captured["openai"] = kwargs
            return fake_client

        client = create_opencode_luna_graph_builder(
            http_client_factory=http_client_factory,
            openai_client_factory=openai_client_factory,
            api_key_loader=lambda **_kwargs: "subscription-key",
        )

        self.assertEqual(client.llm.model_name, "gpt-5.6-luna")
        openai_options = captured["openai"]
        http_options = captured["http"]
        self.assertIsInstance(openai_options, dict)
        self.assertIsInstance(http_options, dict)
        assert isinstance(openai_options, dict)
        assert isinstance(http_options, dict)
        self.assertEqual(openai_options["base_url"], "https://opencode.ai/zen/go/v1")
        self.assertEqual(openai_options["max_retries"], 0)
        self.assertTrue(http_options["trust_env"])

    def test_adapter_returns_existing_graph_builder_shape(self) -> None:
        fake_client = _FakeOpenAIClient()
        adapter = _OpenCodeLunaLLM(
            client=fake_client,
            model_name="gpt-5.6-luna",
            reasoning_effort="high",
        )

        response = asyncio.run(adapter.ainvoke("只返回 JSON"))

        self.assertEqual(response.content, '{"results":[]}')
        self.assertEqual(response.usage.request_tokens, 12)
        self.assertEqual(fake_client.responses.kwargs["model"], "gpt-5.6-luna")
        self.assertEqual(fake_client.responses.kwargs["reasoning"], {"effort": "high"})
        asyncio.run(adapter.aclose())
        self.assertTrue(fake_client.closed)


if __name__ == "__main__":
    unittest.main()
