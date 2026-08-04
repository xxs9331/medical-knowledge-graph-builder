import hashlib
import json
import unittest
from copy import deepcopy
from dataclasses import replace

from medical_kg_sourceprep.book_sources import build_book_manifest, create_text_anchor

from medical_kg_sourceprep.evidence_policy import (
    AtomicPredicateRef,
    CitationBundle,
    Claim,
    ComputationTraceRef,
    EvidenceGap,
    EvidenceKind,
    RuleMatchRef,
    RuleVersionRef,
    SourceType,
    validate_citation_bundle,
    build_report_source,
)


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_hash(value):
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _provenance():
    text = "condition text\n"
    page = {
        "page_id": "book-fixed-1:page-0", "chapter_page_index": 0,
        "raw_path": "raw/page-0.md", "cleaned_path": "cleaned/page-0.md",
        "raw_sha256": _sha256(text), "cleaned_sha256": _sha256(text),
        "source_line_start": 1, "source_line_end": 1,
        "printed_page_number": 1, "source_pdf_page_number": 1,
        "review_status": "unreviewed",
    }
    manifest = build_book_manifest(
        book={"book_id": "book-fixed-1", "title": "Synthetic source", "edition": "v1"},
        pdf={"pdf_id": "book-fixed-1:pdf", "locator": "synthetic.pdf", "sha256": "a" * 64},
        markdown={"markdown_id": "book-fixed-1:markdown", "locator": "synthetic.md", "sha256": "b" * 64},
        pages=[page], chunks=[{"chunk_id": "book-fixed-1:chunk-0", "page_id": page["page_id"], "cleaned_char_start": 0, "cleaned_char_end": len(text), "chunk_sha256": _sha256(text)}],
    )
    anchor = create_text_anchor(
        anchor_id="book-fixed-1:anchor-0", page_id=page["page_id"], raw_text=text,
        cleaned_text=text, raw_char_start=0, raw_char_end=len("condition text"),
        cleaned_char_start=0, cleaned_char_end=len("condition text"),
        source_line_start=1, source_line_end=1, printed_page_number=1,
        source_pdf_page_number=1, review_status="unreviewed",
    )
    source = {
        "citation_id": "book-1", "source_type": "book", "source_id": "book-fixed-1",
        "version": manifest["content_sha256"], "manifest": manifest,
        "manifest_hash": manifest["content_sha256"], "source_hash": "b" * 64,
        "anchor": anchor, "anchor_hash": _canonical_hash(anchor),
        "raw_text": text, "cleaned_text": text,
    }
    registry = {"book-fixed-1": {"status": "approved", "book_id": "book-fixed-1", "version": manifest["content_sha256"], "manifest_hash": manifest["content_sha256"]}}
    return source, registry


def _validate(bundle):
    _, registry = _provenance()
    return validate_citation_bundle(bundle, registry)


def _bundle(**overrides):
    values = {
        "claim": Claim("claim-1", "synthetic conclusion", "medical_explanation"),
        "rule": RuleVersionRef("rule-1", "1.0.0", "approved"),
        "predicates": (AtomicPredicateRef("predicate-1", "condition text", "book-1"),),
        "matches": (RuleMatchRef("match-1", "claim-1", "rule-1", "predicate-1", "book-1"),),
        "computation": ComputationTraceRef("trace-1", "computation-v1", ("predicate-1",), "claim-1", ("book-1", "report-1"), "report-1"),
        "sources": (_provenance()[0], _report_source()),
    }
    values.update(overrides)
    return CitationBundle(**values)


def _report_source():
    return build_report_source(
        citation_id="report-1",
        source_id="report-fixed-1",
        payload={"schema_version": "report-snapshot/v0.2", "items": [{
            "item_id": "predicate-1", "value": "11", "unit": "U",
            "reference_interval": {"lower": "1", "upper": "20", "lower_inclusive": True, "upper_inclusive": True},
            "report_flag": "normal",
        }]},
    )


class EvidencePolicyTests(unittest.TestCase):
    def test_report_builder_and_validator_reject_unknown_and_sensitive_fields(self):
        report = _report_source()
        sensitive_values = ("Example Patient", "raw clinical text")
        for field in ("patient_name", "source_text", "clinical_background"):
            with self.subTest(field=field):
                payload = deepcopy(report["payload"])
                payload["items"][0][field] = sensitive_values[0]
                with self.assertRaises(ValueError) as builder_error:
                    build_report_source(citation_id="x", source_id="y", payload=payload)
                self.assertNotIn(sensitive_values[0], str(builder_error.exception))
                direct = dict(report, payload=payload)
                with self.assertRaises(ValueError) as validator_error:
                    _validate(_bundle(sources=(_provenance()[0], direct)))
                self.assertNotIn(sensitive_values[0], str(validator_error.exception))

    def test_report_schema_rejects_unknown_fields_at_each_layer(self):
        report = _report_source()
        cases = []
        source = dict(report, unknown_source_field="x")
        cases.append(source)
        payload = deepcopy(report["payload"])
        payload["unknown_payload_field"] = "x"
        cases.append(dict(report, payload=payload))
        item_payload = deepcopy(report["payload"])
        item_payload["items"][0]["unknown_item_field"] = "x"
        cases.append(dict(report, payload=item_payload))
        interval_payload = deepcopy(report["payload"])
        interval_payload["items"][0]["reference_interval"]["unknown_interval_field"] = "x"
        cases.append(dict(report, payload=interval_payload))
        for case in cases:
            with self.assertRaises(ValueError):
                _validate(_bundle(sources=(_provenance()[0], case)))

    def test_medical_bundle_without_report_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "REPORT"):
            _validate(_bundle(sources=(_provenance()[0],), computation=ComputationTraceRef("trace-1", "computation-v1", ("predicate-1",), "claim-1", ("book-1",))))

    def test_report_snapshot_and_explicit_computation_binding_are_required(self):
        report = _report_source()
        valid = _bundle(
            sources=(_provenance()[0], report),
            computation=ComputationTraceRef("trace-1", "computation-v1", ("predicate-1",), "claim-1", ("book-1", "report-1"), "report-1"),
        )
        self.assertEqual(_validate(valid), ())
        with self.assertRaisesRegex(ValueError, "REPORT"):
            _validate(_bundle(sources=(_provenance()[0], report), computation=ComputationTraceRef("trace-1", "computation-v1", ("predicate-1",), "claim-1", ("book-1",))))
        forged = dict(report, payload={"schema_version": "report-snapshot/v0.2", "items": [{
            "item_id": "predicate-1", "value": "12", "unit": "U",
            "reference_interval": {"lower": "1", "upper": "20", "lower_inclusive": True, "upper_inclusive": True},
            "report_flag": "normal",
        }]})
        with self.assertRaisesRegex(ValueError, "hash"):
            _validate(replace(valid, sources=(_provenance()[0], forged)))

    def test_arbitrary_report_source_is_rejected(self):
        report = {"citation_id": "report-1", "source_type": "report", "source_id": "report-1", "version": "1"}
        with self.assertRaisesRegex(ValueError, "unsupported field"):
            _validate(_bundle(sources=(_provenance()[0], report)))

    def test_unapproved_book_source_is_rejected_even_when_source_type_is_book(self):
        forged = {"citation_id": "book-1", "source_type": "book", "source_id": "unapproved-external-content", "version": "1"}
        with self.assertRaises(ValueError):
            _validate(_bundle(sources=(forged,)))

    def test_valid_three_chain_bundle(self):
        self.assertEqual(_validate(_bundle()), ())

    def test_medical_bundle_without_registry_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "approved provenance registry"):
            validate_citation_bundle(_bundle())

    def test_missing_or_unknown_anchor_fails_closed(self):
        source, _ = _provenance()
        missing = dict(source)
        missing.pop("anchor")
        with self.assertRaisesRegex(ValueError, "manifest and TextAnchor"):
            _validate(_bundle(sources=(missing, _report_source())))
        unknown = dict(source, anchor=dict(source["anchor"], page_id="unknown-page"))
        unknown["anchor_hash"] = _canonical_hash(unknown["anchor"])
        with self.assertRaisesRegex(ValueError, "unknown manifest page"):
            _validate(_bundle(sources=(unknown, _report_source())))

    def test_manifest_anchor_content_and_source_hash_drift_fail_closed(self):
        source, _ = _provenance()
        manifest_drift = dict(source, manifest_hash="0" * 64)
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            _validate(_bundle(sources=(manifest_drift, _report_source())))
        anchor_drift = dict(source, anchor_hash="0" * 64)
        with self.assertRaisesRegex(ValueError, "TextAnchor hash"):
            _validate(_bundle(sources=(anchor_drift, _report_source())))
        content_drift = dict(source, cleaned_text="condition text changed\n")
        with self.assertRaisesRegex(ValueError, "invalid approved BOOK provenance"):
            _validate(_bundle(sources=(content_drift, _report_source())))
        source_drift = dict(source, source_hash="c" * 64)
        with self.assertRaisesRegex(ValueError, "source hash"):
            _validate(_bundle(sources=(source_drift, _report_source())))

    def test_missing_book_fails_closed(self):
        bundle = _bundle(sources=(_report_source(),))
        with self.assertRaisesRegex(ValueError, "BOOK"):
            _validate(bundle)

    def test_index_network_and_model_sources_are_rejected(self):
        for source_type in (SourceType.INDEX, SourceType.NETWORK, SourceType.MODEL_TEXT):
            with self.subTest(source_type=source_type):
                bundle = _bundle(sources=({"citation_id": "book-1", "source_type": source_type.value, "source_id": "x", "version": "1"},))
                with self.assertRaises(ValueError):
                    _validate(bundle)

    def test_rule_must_be_approved_and_version_fixed(self):
        for rule in (RuleVersionRef("rule-1", "1.0.0", "candidate"), RuleVersionRef("rule-1", "", "approved")):
            with self.subTest(rule=rule):
                with self.assertRaises(ValueError):
                    _validate(_bundle(rule=rule))

    def test_each_condition_and_conclusion_needs_a_book_anchor(self):
        for predicates, matches in (((), ()), (_bundle().predicates, ())):
            with self.subTest(predicates=predicates, matches=matches):
                with self.assertRaises(ValueError):
                    _validate(_bundle(predicates=predicates, matches=matches))

    def test_unknown_required_condition_creates_gap(self):
        bundle = _bundle(gaps=(EvidenceGap("predicate-1", "unknown", True),))
        with self.assertRaisesRegex(ValueError, "unknown"):
            _validate(bundle)

    def test_report_anomaly_is_allowed_as_fact_but_not_explanation(self):
        report = _report_source()
        fact = _bundle(claim=Claim("claim-1", "synthetic reported finding", "reported_fact"), rule=None, predicates=(), matches=(), sources=(report,), computation=None)
        self.assertEqual(_validate(fact), ())
        explanation = _bundle(claim=Claim("claim-1", "synthetic explanation", "medical_explanation"), sources=(report,))
        with self.assertRaises(ValueError):
            _validate(explanation)

    def test_conflict_requires_explicit_weaker_wording(self):
        bundle = _bundle(conflicts=("rule disagreement",))
        with self.assertRaisesRegex(ValueError, "conflict"):
            _validate(bundle)
        weakened = _bundle(claim=Claim("claim-1", "synthetic conclusion may be associated", "medical_explanation"), conflicts=("rule disagreement",), strength="qualified")
        self.assertEqual(_validate(weakened), ())

    def test_citation_ids_must_map_one_to_one(self):
        duplicate = _bundle(matches=(RuleMatchRef("match-1", "claim-1", "rule-1", "predicate-1", "book-1"), RuleMatchRef("match-1", "claim-1", "rule-1", "predicate-1", "book-1")))
        with self.assertRaisesRegex(ValueError, "unique"):
            _validate(duplicate)


if __name__ == "__main__":
    unittest.main()
