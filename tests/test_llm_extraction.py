import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from medical_kg_sourceprep.extraction.llm_extraction import (
    EvidenceChunk,
    ExtractionError,
    OpenCodeGoClient,
    SemanticExtractor,
    build_payload,
    build_prompt,
    load_chunk_manifest,
    make_windows,
    validate_candidate_partial,
    validate_candidate,
)


def chunk(chunk_id, text, chapter="chapter-a"):
    return EvidenceChunk(chunk_id, text, hashlib.sha256(text.encode()).hexdigest(), chapter)


class LlmExtractionTests(unittest.TestCase):
    def test_windows_do_not_cross_chapters_or_split_chunks(self):
        windows = make_windows([chunk("a", "alpha"), chunk("b", "bravo", "chapter-b")], max_chars=5)
        self.assertEqual([[item.chunk_id for item in w.chunks] for w in windows], [["a"], ["b"]])

    def test_placeholder_chapter_id_falls_back_to_h1_and_flushes(self):
        first = chunk("a", "lead\n# Chapter One\nalpha", "full-book")
        second = chunk("b", "beta\n# Chapter Two\ngamma", "full-book")
        windows = make_windows([first, second], max_chars=100)
        self.assertEqual([w.chapter_id for w in windows], ["Chapter One", "Chapter Two"])
        self.assertEqual([[c.chunk_id for c in w.chunks] for w in windows], [["a"], ["b"]])

    def test_multiple_h1s_in_one_indivisible_chunk_are_diagnostic(self):
        with self.assertRaisesRegex(ExtractionError, "ambiguous chapter boundary"):
            make_windows([chunk("a", "# One\ntext\n# Two\ntext", "full-book")])

    def test_payload_has_fixed_provider_model_controls(self):
        for effort in ("low", "medium", "high", "max", "xhigh"):
            payload = build_payload("synthetic", reasoning_effort=effort, max_tokens=2048)
            self.assertEqual(payload["model"], "deepseek-v4-flash")
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["reasoning_effort"], effort)
            self.assertEqual(payload["thinking"], {"type": "enabled"})
            self.assertEqual(payload["max_tokens"], 2048)
            self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_payload_omits_reasoning_when_disabled(self):
        payload = build_payload("synthetic", reasoning_effort=None, max_tokens=2048)
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["max_tokens"], 2048)

    def test_payload_rejects_invalid_reasoning_and_token_budget(self):
        for value in ("", "none", True):
            with self.assertRaises(ExtractionError):
                build_payload("synthetic", reasoning_effort=value)
        for value in (0, -1, True, 32769):
            with self.assertRaises(ExtractionError):
                build_payload("synthetic", max_tokens=value)

    def test_exact_unique_quote_is_reconstructed(self):
        source = chunk("a", "Synthetic marker appears.")
        result = validate_candidate({"entities": [{
            "id": "e1", "entity_type": "MedicalConcept", "text": "Synthetic marker",
            "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256,
                           "exact_quote": "Synthetic marker"}}], "rules": [], "relations": []}, [source])
        self.assertEqual(result["entities"][0]["char_start"], 0)
        self.assertEqual(result["entities"][0]["status"], "candidate")

    def test_short_text_is_located_inside_unique_long_quote(self):
        source = chunk("a", "first marker\nsecond marker")
        result = validate_candidate({"entities": [{
            "id": "e1", "entity_type": "MedicalConcept", "text": "marker",
            "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256,
                           "exact_quote": "second marker"}}], "rules": [], "relations": []}, [source])
        self.assertEqual(result["entities"][0]["char_start"], source.text.index("second marker") + 7)

    def test_text_repeated_or_absent_inside_quote_fails(self):
        source = chunk("a", "prefix marker marker suffix")
        ref = {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "prefix marker marker suffix"}
        for text in ("marker", "missing"):
            with self.assertRaisesRegex(ExtractionError, "absent or ambiguous"):
                validate_candidate({"entities": [{"id": "e", "entity_type": "MedicalConcept", "text": text, "source_ref": ref}], "rules": [], "relations": []}, [source])

    def test_repeated_quote_and_hash_drift_fail_closed(self):
        source = chunk("a", "same same")
        ref = {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "same"}
        with self.assertRaisesRegex(ExtractionError, "ambiguous"):
            validate_candidate({"entities": [{"id": "e", "entity_type": "MedicalConcept", "text": "same", "source_ref": ref}], "rules": [], "relations": []}, [source])
        with self.assertRaisesRegex(ExtractionError, "hash"):
            validate_candidate({"entities": [{"id": "e", "entity_type": "MedicalConcept", "text": "same", "source_ref": dict(ref, chunk_sha256="0" * 64)}], "rules": [], "relations": []}, [source])

    def test_invalid_relation_direction_and_rule_contract_are_rejected(self):
        source = chunk("a", "item method")
        entities = [{"id": "i", "entity_type": "TestItem", "text": "item", "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "item"}}, {"id": "m", "entity_type": "TestMethod", "text": "method", "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "method"}}]
        with self.assertRaisesRegex(ExtractionError, "direction"):
            validate_candidate({"entities": entities, "rules": [], "relations": [{"source_id": "m", "relation": "ITEM_MEASURED_BY_METHOD", "target_id": "i"}]}, [source])

    def test_prompt_schema_lists_fixed_contract_and_review_atoms_need_anchors(self):
        source = chunk("a", "condition connector conclusion")
        window = make_windows([source])[0]
        prompt = build_prompt(window)
        self.assertIn("TestItem", prompt)
        self.assertIn("ITEM_MEASURED_BY_METHOD", prompt)
        rule = {"id": "r", "entity_type": "InterpretationRule", "text": "condition", "semantic_type": "DEFINES_AS", "subject_logic": "SINGLE",
                "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "condition"},
                "review_payload": {"conditions": ["condition"], "connector": {"text": "ALL", "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "connector"}}, "conclusion": {"text": "conclusion", "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "conclusion"}}}}
        with self.assertRaisesRegex(ExtractionError, "text plus source_ref"):
            validate_candidate({"entities": [], "rules": [rule], "relations": []}, [source])
        self.assertIn("SINGLE may omit review_payload and has no connector", prompt)
        self.assertIn("operator ALL/ANY/AT_LEAST", prompt)

    def test_prompt_is_compact_but_keeps_fixed_contract_and_limits(self):
        prompt = build_prompt(make_windows([chunk("a", "synthetic")])[0])
        self.assertNotIn('"$schema"', prompt)
        self.assertNotIn("draft/2020-12", prompt)
        for value in ("MedicalConcept", "TestItem", "InterpretationRule", "semantic_type", "subject_logic",
                      "source_ref", "chunk_id", "chunk_sha256", "exact_quote", "RULE_HAS_CONCLUSION",
                      "DEFINES_AS", "SINGLE", "ALL", "ANY", "AT_LEAST", "entities<=24", "rules<=12", "relations<=72"):
            self.assertIn(value, prompt)
        self.assertIn("Select at most", prompt)
        self.assertIn("exact_quote must be verbatim", prompt)
        self.assertIn("only once", prompt)
        self.assertIn("conditions/conclusion={text,source_ref}", prompt)
        self.assertIn("AT_LEAST only", prompt)
        self.assertNotIn("return a schema error", prompt)
        self.assertLess(len(prompt), len(json.dumps(__import__("medical_kg_sourceprep.extraction.llm_extraction", fromlist=["OUTPUT_SCHEMA"]).OUTPUT_SCHEMA)))

    def test_partial_validation_keeps_valid_candidate_and_rejects_bad_quote(self):
        source = chunk("a", "valid marker broken")
        valid = {"id": "good", "entity_type": "MedicalConcept", "text": "valid",
                 "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "valid"}}
        bad = {"id": "bad", "entity_type": "MedicalConcept", "text": "invalid",
               "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "missing"}}
        result = validate_candidate_partial({"entities": [valid, bad], "rules": [], "relations": []}, [source])
        self.assertEqual([item["record_id"] for item in result["entities"]], ["good"])
        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(result["counts"], {"accepted": 1, "rejected": 1})
        self.assertEqual(result["rejections"][0]["kind"], "entity")
        self.assertEqual(result["rejections"][0]["index"], 1)
        self.assertNotIn("missing", json.dumps(result["rejections"]))

    def test_partial_validation_rejects_bad_review_connector_only(self):
        source = chunk("a", "condition conclusion")
        ref = lambda quote: {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": quote}
        valid = {"id": "good", "entity_type": "MedicalConcept", "text": "condition", "source_ref": ref("condition")}
        bad_rule = {"id": "bad-rule", "entity_type": "InterpretationRule", "text": "condition",
                    "semantic_type": "DEFINES_AS", "subject_logic": "ALL", "source_ref": ref("condition"),
                    "review_payload": {"conditions": [{"text": "condition", "source_ref": ref("condition")}],
                                       "connector": {"operator": "ALL", "text": "missing", "source_ref": ref("missing")},
                                       "conclusion": {"text": "conclusion", "source_ref": ref("conclusion")}}}
        result = validate_candidate_partial({"entities": [valid], "rules": [bad_rule], "relations": []}, [source])
        self.assertEqual([item["record_id"] for item in result["entities"]], ["good"])
        self.assertEqual(result["counts"], {"accepted": 1, "rejected": 1})
        self.assertEqual(result["rejections"][0]["kind"], "rule")

    def test_partial_validation_rejects_bad_relations_but_keeps_valid_relation(self):
        source = chunk("a", "item method")
        def entity(record_id, entity_type, text):
            return {"id": record_id, "entity_type": entity_type, "text": text,
                    "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": text}}
        entities = [entity("i", "TestItem", "item"), entity("m", "TestMethod", "method")]
        relations = [
            {"source_id": "i", "relation": "ITEM_MEASURED_BY_METHOD", "target_id": "m"},
            {"source_id": "m", "relation": "ITEM_MEASURED_BY_METHOD", "target_id": "i"},
            {"source_id": "i", "relation": "ITEM_MEASURED_BY_METHOD", "target_id": "missing"},
        ]
        result = validate_candidate_partial({"entities": entities, "rules": [], "relations": relations}, [source])
        self.assertEqual(len(result["relations"]), 1)
        self.assertEqual(result["counts"]["rejected"], 2)
        self.assertEqual(result["counts"]["accepted"], 3)
        self.assertTrue(all("missing" not in json.dumps(item) for item in result["rejections"]))

    def test_partial_validation_keeps_top_level_and_array_limits_atomic(self):
        source = chunk("a", "valid")
        valid = {"id": "good", "entity_type": "MedicalConcept", "text": "valid",
                 "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "valid"}}
        with self.assertRaisesRegex(ExtractionError, "schema_error"):
            validate_candidate_partial({"entities": [valid], "rules": []}, [source])
        with self.assertRaisesRegex(ExtractionError, "limit"):
            validate_candidate_partial({"entities": [valid] * 25, "rules": [], "relations": []}, [source])

    def test_partial_validation_all_rejected_is_no_candidates(self):
        source = chunk("a", "valid marker")
        bad = {"id": "bad", "entity_type": "MedicalConcept", "text": "valid",
               "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "missing"}}
        result = validate_candidate_partial({"entities": [bad], "rules": [], "relations": []}, [source])
        self.assertEqual(result["status"], "no_candidates")
        self.assertEqual(result["counts"], {"accepted": 0, "rejected": 1})

    def test_extractor_checkpoints_partial_success_and_reuses_it(self):
        source = chunk("a", "valid marker broken")
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, prompt):
                self.calls += 1
                good = {"id": "good", "entity_type": "MedicalConcept", "text": "valid",
                        "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "valid"}}
                bad = {"id": "bad", "entity_type": "MedicalConcept", "text": "invalid",
                       "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "missing"}}
                return json.dumps({"entities": [good, bad], "rules": [], "relations": []}), ()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            client = Client()
            first = SemanticExtractor(client).extract([source], checkpoint=checkpoint)
            second = SemanticExtractor(client).extract([source], checkpoint=checkpoint)
        self.assertEqual(first["results"][0]["status"], "partial_success")
        self.assertEqual(first["counts"]["rejected"], 1)
        self.assertEqual(second["counts"]["reused_windows"], 1)
        self.assertEqual(client.calls, 1)

    def test_client_bounded_transport_errors_and_secret_redaction(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{"model":"deepseek-v4-flash","usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5,"completion_tokens_details":{"reasoning_tokens":3}},"choices":[{"finish_reason":"length","message":{"content":"{}","reasoning_content":"do-not-persist"}}]}'
        cases = [lambda req, timeout: HTTPError("u", 429, "secret-key", {}, None), lambda req, timeout: TimeoutError("secret-key")]
        for opener in cases:
            with patch("medical_kg_sourceprep.extraction.llm_extraction.time.sleep"):
                with self.assertRaises(ExtractionError) as caught:
                    OpenCodeGoClient(api_key="secret-key", opener=opener).complete("x")
            self.assertNotIn("secret-key", str(caught.exception))

        with patch("medical_kg_sourceprep.extraction.llm_extraction.time.sleep"):
            client = OpenCodeGoClient(api_key="secret-key", opener=lambda req, timeout: Response())
            with self.assertRaisesRegex(ExtractionError, "length"):
                client.complete("x")
            self.assertNotIn("do-not-persist", json.dumps(client.last_response_meta))
            class ThenTimeout:
                def __init__(self): self.calls = 0
                def __call__(self, req, timeout):
                    self.calls += 1
                    if self.calls == 1: return Response()
                    raise TimeoutError("transport")
            transport = ThenTimeout()
            client = OpenCodeGoClient(api_key="secret-key", opener=transport, max_retries=0)
            with self.assertRaisesRegex(ExtractionError, "length"):
                client.complete("x")
            with self.assertRaisesRegex(ExtractionError, "timeout"):
                client.complete("x")
            self.assertEqual(client.last_response_meta, {})

    def test_extractor_schema_failure_has_one_shared_three_call_budget(self):
        source = chunk("a", "unique")
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, prompt): self.calls += 1; return "{}", ()
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "c.json"
            with self.assertRaisesRegex(ExtractionError, "window_failed"):
                SemanticExtractor(client).extract([source], checkpoint=checkpoint)
            self.assertEqual(json.loads(checkpoint.read_text())["windows"]["window:000000"]["status"], "failed")
        self.assertEqual(client.calls, 3)

    def test_checkpoint_manifest_drift_does_not_reuse(self):
        source = chunk("a", "unique")
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, prompt):
                self.calls += 1
                return json.dumps({"entities": [{"id": "e", "entity_type": "MedicalConcept", "text": "unique", "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "unique"}}], "rules": [], "relations": []}), ()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            client = Client()
            SemanticExtractor(client).extract([source], checkpoint=checkpoint, input_manifest_hash="a" * 64)
            SemanticExtractor(client).extract([source], checkpoint=checkpoint, input_manifest_hash="b" * 64)
            self.assertEqual(client.calls, 2)

    def test_reused_no_candidates_keeps_window_count(self):
        source = chunk("a", "no candidate")
        class Client:
            def complete(self, prompt): return json.dumps({"entities": [], "rules": [], "relations": []}), ()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            first = SemanticExtractor(Client()).extract([source], checkpoint=checkpoint)
            second = SemanticExtractor(Client()).extract([source], checkpoint=checkpoint)
        self.assertEqual(first["counts"]["no_candidates_windows"], 1)
        self.assertEqual(second["counts"]["no_candidates_windows"], 1)
        self.assertEqual(second["counts"]["reused_windows"], 1)

    def test_chinese_connector_is_anchored_separately_from_normalized_operator(self):
        source = chunk("a", "甲且乙 或 丙 至少一项 丁")
        def ref(quote):
            return {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": quote}
        rule = {"id": "r", "entity_type": "InterpretationRule", "text": "甲", "semantic_type": "DEFINES_AS", "subject_logic": "ALL",
                "source_ref": ref("甲"), "review_payload": {"conditions": [{"text": "甲", "source_ref": ref("甲")}, {"text": "乙", "source_ref": ref("乙")}],
                "connector": {"operator": "ALL", "text": "且", "source_ref": ref("甲且乙")}, "conclusion": {"text": "丙", "source_ref": ref("丙")}}}
        self.assertEqual(validate_candidate({"entities": [], "rules": [rule], "relations": []}, [source])["rules"][0]["status"], "candidate")
        rule["review_payload"]["connector"]["operator"] = "AT_LEAST"
        rule["review_payload"]["at_least"] = 1
        rule["review_payload"]["connector"]["text"] = "至少一项"
        rule["review_payload"]["connector"]["source_ref"] = ref("至少一项")
        self.assertEqual(validate_candidate({"entities": [], "rules": [rule], "relations": []}, [source])["rules"][0]["status"], "candidate")

    def test_checkpoint_is_atomic_and_same_hash_skips_window(self):
        source = chunk("a", "unique")
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, prompt):
                self.calls += 1
                return json.dumps({"entities": [{"id": "e", "entity_type": "MedicalConcept", "text": "unique", "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "unique"}}], "rules": [], "relations": []}), ()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            client = Client()
            SemanticExtractor(client, max_chars=100).extract([source], checkpoint=checkpoint)
            SemanticExtractor(client, max_chars=100).extract([source], checkpoint=checkpoint)
            self.assertEqual(client.calls, 1)
            self.assertEqual(json.loads(checkpoint.read_text())["schema_version"], "deepseek-semantic-candidates/v0.3")

    def test_checkpoint_config_change_does_not_reuse_success(self):
        source = chunk("a", "unique")
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, prompt):
                self.calls += 1
                return json.dumps({"entities": [{"id": "e", "entity_type": "MedicalConcept", "text": "unique", "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "unique"}}], "rules": [], "relations": []}), ()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            client = Client()
            SemanticExtractor(client, max_chars=100, reasoning_effort="low").extract([source], checkpoint=checkpoint)
            SemanticExtractor(client, max_chars=100, reasoning_effort="medium").extract([source], checkpoint=checkpoint)
        self.assertEqual(client.calls, 2)

    def test_extractor_passes_and_audits_selected_call_parameters(self):
        source = chunk("a", "unique")
        class Client:
            def __init__(self):
                self.calls = []
            def complete(self, prompt, *, reasoning_effort, max_tokens, retry_budget):
                self.calls.append({"reasoning_effort": reasoning_effort, "max_tokens": max_tokens,
                                   "retry_budget": retry_budget})
                return json.dumps({"entities": [{"id": "e", "entity_type": "MedicalConcept", "text": "unique",
                    "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "unique"}}],
                    "rules": [], "relations": []}), ()
        client = Client()
        result = SemanticExtractor(client, reasoning_effort="medium", max_tokens=4096, max_chars=100).extract([source])
        self.assertEqual(client.calls, [{"reasoning_effort": "medium", "max_tokens": 4096, "retry_budget": 0}])
        self.assertEqual(result["parameters"]["reasoning_effort"], "medium")
        self.assertEqual(result["parameters"]["max_tokens"], 4096)
        self.assertEqual(result["parameters"]["max_chars"], 100)
        self.assertEqual(result["attempts"][0]["parameters"], result["parameters"])

    def test_disabled_reasoning_is_audited_and_changes_checkpoint_identity(self):
        source = chunk("a", "unique")
        class Client:
            def complete(self, prompt):
                return json.dumps({"entities": [], "rules": [], "relations": []}), ()
        disabled = SemanticExtractor(Client(), reasoning_effort=None, max_chars=100).extract([source])
        low = SemanticExtractor(Client(), reasoning_effort="low", max_chars=100).extract([source])
        self.assertEqual(disabled["parameters"]["reasoning_mode"], "disabled")
        self.assertIsNone(disabled["parameters"]["reasoning_effort"])
        self.assertNotEqual(disabled["config_identity"], low["config_identity"])

    def test_checkpoint_max_chars_change_does_not_reuse_success(self):
        source = chunk("a", "unique")
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, prompt):
                self.calls += 1
                return json.dumps({"entities": [{"id": "e", "entity_type": "MedicalConcept", "text": "unique",
                    "source_ref": {"chunk_id": "a", "chunk_sha256": source.chunk_sha256, "exact_quote": "unique"}}],
                    "rules": [], "relations": []}), ()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            client = Client()
            SemanticExtractor(client, max_chars=100).extract([source], checkpoint=checkpoint)
            SemanticExtractor(client, max_chars=101).extract([source], checkpoint=checkpoint)
        self.assertEqual(client.calls, 2)

    def test_manifest_uses_chunk_path_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = "synthetic"
            (root / "chunks").mkdir()
            (root / "chunks" / "a.md").write_text(text, encoding="utf-8")
            manifest = {"schema_version": "evidence-chunk-package/v0.1", "source_manifest_sha256": "a" * 64, "document_id": "d", "chapter_id": "full-book",
                        "page_count": 1, "chunk_count": 1, "pages": [{"page_id": "p", "chapter_page_index": 0, "printed_page_number": 1, "source_pdf_page_number": 1, "review_status": "ok", "cleaned_sha256": "b" * 64}],
                        "chunks": [{"chunk_id": "c", "page_id": "p", "document_id": "d", "chapter_id": "full-book", "chunk_path": "chunks/a.md", "chunk_sha256": hashlib.sha256(text.encode()).hexdigest(), "cleaned_char_start": 0, "cleaned_char_end": len(text)}]}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            _, loaded = load_chunk_manifest(path)
            self.assertEqual(loaded[0].text, text)
            manifest["chunks"][0]["chunk_path"] = "../outside.md"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ExtractionError, "escapes"):
                load_chunk_manifest(path)


if __name__ == "__main__":
    unittest.main()
