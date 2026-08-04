import hashlib
import unittest

from medical_kg_sourceprep.llm_extraction import EvidenceChunk
from medical_kg_sourceprep.semantic_projection import (
    LANGEXTRACT_CHAPTER_MAX_OUTPUT_TOKENS,
    LANGEXTRACT_MAX_EXTRACTIONS_PER_PAGE,
    LANGEXTRACT_MAX_RELATIONS_PER_PAGE,
    TransientTransportError,
    _chapter_status,
    _disable_langextract_thinking,
    _enable_strict_schema_when_supported,
    _langextract_prompt,
    _langextract_output_schema,
    _normalize_exact_duplicates,
    _resolve_chapter_provider,
    _run_with_transient_retry,
    _validate_page_extractions,
    adapt_langextract,
)


class SemanticProjectionTests(unittest.TestCase):
    def test_direct_deepseek_provider_uses_fixed_official_contract(self):
        provider = _resolve_chapter_provider(
            "deepseek-direct",
            env={
                "DEEPSEEK_API_KEY": "direct-key",
                "HTTPS_PROXY": "http://proxy.invalid",
                "DEEPSEEK_BASE_URL": "https://attacker.invalid",
            },
        )

        self.assertEqual(provider.provider, "deepseek-direct")
        self.assertEqual(provider.endpoint, "https://api.deepseek.com")
        self.assertEqual(provider.model, "deepseek-v4-flash")
        self.assertEqual(provider.api_key, "direct-key")
        self.assertFalse(provider.supports_json_schema)

    def test_direct_deepseek_provider_requires_explicit_key(self):
        with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
            _resolve_chapter_provider("deepseek-direct", env={})

    def test_strict_schema_matches_langextract_raw_envelope(self):
        schema = _langextract_output_schema()
        self.assertEqual(schema["required"], ["extractions"])
        self.assertFalse(schema["additionalProperties"])
        variants = schema["properties"]["extractions"]["items"]["anyOf"]
        self.assertEqual(len(variants), 3)
        self.assertEqual(
            {next(key for key in variant["properties"] if not key.endswith("_attributes"))
             for variant in variants},
            {"entity", "rule", "relation"},
        )

    def test_schema_probe_applies_strict_response_format(self):
        class Message:
            content = '{"extractions":[]}'

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        class Completions:
            def __init__(self):
                self.request = None

            def create(self, **request):
                self.request = request
                return Response()

        class Model:
            def __init__(self):
                self._client = type("Client", (), {})()
                self._client.chat = type("Chat", (), {})()
                self._client.chat.completions = Completions()
                self.applied = None

            @staticmethod
            def _build_chat_completions_params(prompt, config):
                return {"messages": [{"content": prompt}], "max_tokens": config["max_output_tokens"]}

            def apply_schema(self, schema):
                self.applied = schema

        schema = type("Schema", (), {"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": _langextract_output_schema(), "strict": True},
        }})()
        model = Model()

        self.assertTrue(_enable_strict_schema_when_supported(model, schema))
        self.assertIs(model.applied, schema)
        request = model._client.chat.completions.request
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])

    def test_schema_probe_only_falls_back_on_explicit_unsupported_400(self):
        class Unsupported(RuntimeError):
            status_code = 400

        class Completions:
            @staticmethod
            def create(**request):
                del request
                raise Unsupported("response_format json_schema is unsupported")

        class Model:
            _client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
            applied = None

            @staticmethod
            def _build_chat_completions_params(prompt, config):
                return {"messages": [{"content": prompt}], **config}

            def apply_schema(self, schema):
                self.applied = schema

        schema = type("Schema", (), {"response_format": {"type": "json_schema"}})()
        model = Model()

        self.assertFalse(_enable_strict_schema_when_supported(model, schema))
        self.assertIsNone(model.applied)

    def test_page_normalization_collapses_only_exact_duplicates_before_validation(self):
        prompt = _langextract_prompt()
        self.assertIn('{"extractions":[...]}', prompt)
        self.assertIn("escape every source backslash", prompt)
        self.assertIn("at most 128 extractions", prompt)
        self.assertIn("at most 48 relations", prompt)
        self.assertEqual(LANGEXTRACT_MAX_EXTRACTIONS_PER_PAGE, 128)
        self.assertEqual(LANGEXTRACT_MAX_RELATIONS_PER_PAGE, 48)

        entity = {
            "extraction_class": "entity",
            "extraction_text": "item",
            "attributes": {"source_chunk_id": "chunk", "source_quote": "item"},
        }
        _validate_page_extractions({"extractions": [entity]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _validate_page_extractions({"extractions": [entity, entity]})
        similar = {**entity, "attributes": {**entity["attributes"], "source_quote": "other"}}
        normalized, duplicate_count = _normalize_exact_duplicates(
            {"extractions": [entity, entity, similar]}
        )
        self.assertEqual(duplicate_count, 1)
        self.assertEqual(normalized["extractions"], [entity, similar])
        _validate_page_extractions(normalized)

    def test_page_normalization_enforces_raw_limit_before_duplicate_collapse(self):
        entity = {
            "extraction_class": "entity",
            "extraction_text": "item",
            "attributes": {"source_chunk_id": "chunk", "source_quote": "item"},
        }
        with self.assertRaisesRegex(ValueError, "extraction count"):
            _normalize_exact_duplicates({"extractions": [entity] * 129})

    def test_validator_enforces_page_and_relation_limits(self):
        entity = {
            "extraction_class": "entity",
            "extraction_text": "item",
            "attributes": {"source_chunk_id": "chunk", "source_quote": "item"},
        }
        with self.assertRaisesRegex(ValueError, "extraction count"):
            _validate_page_extractions({"extractions": [
                {**entity, "extraction_text": str(index)} for index in range(129)
            ]})
        with self.assertRaisesRegex(ValueError, "relation count"):
            _validate_page_extractions({"extractions": [
                {
                    "extraction_class": "relation",
                    "extraction_text": str(index),
                    "attributes": {
                        "source_chunk_id": "chunk",
                        "source_quote": str(index),
                    },
                }
                for index in range(49)
            ]})

    def test_chapter_status_uses_canonical_success_windows_not_audit_entries(self):
        windows = {
            **{f"page:{index:04d}": {"status": "success"} for index in range(24)},
            "discarded:page:0007": {"status": "discarded"},
        }

        self.assertEqual(_chapter_status(windows), "all-success")

    def test_transport_retry_is_bounded_and_recovers_from_http_500(self):
        class Http500Error(RuntimeError):
            status_code = 500

        calls = []
        delays = []

        def operation():
            calls.append(None)
            if len(calls) < 3:
                raise Http500Error("HTTP 500 Internal server error")
            return "native-result"

        self.assertEqual(
            _run_with_transient_retry(operation, sleep=delays.append),
            "native-result",
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(delays, [1.0, 2.0])

    def test_router_unavailable_retries_and_exhaustion_redacts_secret(self):
        calls = []

        def operation():
            calls.append(None)
            raise RuntimeError("Router.Unavailable: Bearer secret-token")

        with self.assertRaisesRegex(TransientTransportError, "attempts=3") as caught:
            _run_with_transient_retry(operation, sleep=lambda _: None,
                                      secrets=("secret-token",))
        self.assertEqual(len(calls), 3)
        self.assertNotIn("secret-token", str(caught.exception))
        self.assertIn("[REDACTED]", str(caught.exception))

    def test_wrapped_connection_error_is_retried(self):
        calls = []

        def operation():
            calls.append(None)
            if len(calls) == 1:
                raise RuntimeError("OpenAI API error: Connection error.")
            return "recovered"

        self.assertEqual(
            _run_with_transient_retry(operation, sleep=lambda _: None),
            "recovered",
        )
        self.assertEqual(len(calls), 2)

    def test_validation_error_is_not_retried(self):
        calls = []

        def operation():
            calls.append(None)
            raise ValueError("LangExtract payload must contain an extractions list")

        with self.assertRaisesRegex(ValueError, "payload"):
            _run_with_transient_retry(operation, sleep=lambda _: None)
        self.assertEqual(len(calls), 1)

    def test_langextract_transport_keeps_thinking_disabled(self):
        class Model:
            @staticmethod
            def _build_chat_completions_params(prompt, config):
                return {"messages": [{"content": prompt}], **config}

        model = _disable_langextract_thinking(Model())
        request = model._build_chat_completions_params(
            "source", {"max_tokens": LANGEXTRACT_CHAPTER_MAX_OUTPUT_TOKENS}
        )
        self.assertEqual(request["max_tokens"], 32768)
        self.assertEqual(request["extra_body"]["thinking"], {"type": "disabled"})

    def test_langextract_sdk_request_uses_chapter_output_budget(self):
        from langextract.providers.openai import OpenAILanguageModel

        model = OpenAILanguageModel(
            model_id="synthetic", api_key="synthetic", base_url="https://example.invalid/v1",
            max_output_tokens=LANGEXTRACT_CHAPTER_MAX_OUTPUT_TOKENS,
        )
        request = model._build_chat_completions_params("source", model.merge_kwargs({}))
        self.assertEqual(LANGEXTRACT_CHAPTER_MAX_OUTPUT_TOKENS, 32768)
        self.assertEqual(request["max_tokens"], 32768)

    def test_langextract_envelope_maps_entities_rule_and_fixed_edges(self):
        text = "item method normal condition conclusion"
        chunk = EvidenceChunk("c1", text, hashlib.sha256(text.encode()).hexdigest())
        ref = lambda quote: {"source_chunk_id": "c1", "source_quote": quote}
        payload = {"extractions": [
            {"extraction_class": "entity", "extraction_text": "item", "attributes": {"entity_type": "TestItem", **ref("item")}},
            {"extraction_class": "entity", "extraction_text": "method", "attributes": {"entity_type": "TestMethod", **ref("method")}},
            {"extraction_class": "rule", "extraction_text": "condition conclusion", "attributes": {
                "semantic_type": "DEFINES_AS", "subject_logic": "SINGLE", **ref("condition conclusion"),
                "conditions": [{"text": "condition", "source_ref": ref("condition")}],
                "conclusion": {"text": "conclusion", "source_ref": ref("conclusion")},
            }},
            {"extraction_class": "relation", "extraction_text": "item method", "attributes": {
                "relation_type": "ITEM_MEASURED_BY_METHOD", "source_text": "item", "target_text": "method",
                **ref("item method"),
            }},
        ]}
        result = adapt_langextract(payload, [chunk])
        self.assertEqual(len(result.records), 3)
        self.assertEqual(result.relations[0][1], "ITEM_MEASURED_BY_METHOD")
        rule = next(record for record in result.records if record.entity_type == "InterpretationRule")
        self.assertEqual(rule.rule_payload["conclusion"]["text"], "conclusion")
        self.assertEqual(result.review_queue, ())

    def test_missing_anchor_and_cross_chunk_relation_are_reviewable(self):
        one = EvidenceChunk("c1", "item", hashlib.sha256(b"item").hexdigest())
        two = EvidenceChunk("c2", "method", hashlib.sha256(b"method").hexdigest())
        payload = {"extractions": [
            {"extraction_class": "entity", "extraction_text": "item", "attributes": {"entity_type": "TestItem", "source_chunk_id": "c1", "source_quote": "missing"}},
            {"extraction_class": "entity", "extraction_text": "method", "attributes": {"entity_type": "TestMethod", "source_chunk_id": "c2", "source_quote": "method"}},
        ]}
        result = adapt_langextract(payload, [one, two])
        self.assertEqual(len(result.records), 1)
        self.assertEqual(len(result.review_queue), 1)
        self.assertIn("quote", result.review_queue[0]["reason"])


if __name__ == "__main__":
    unittest.main()
