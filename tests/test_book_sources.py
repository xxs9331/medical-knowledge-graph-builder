import hashlib
import unittest

from medical_kg_sourceprep.provenance.book_sources import (
    ANCHOR_SCHEMA_VERSION,
    UNAVAILABLE,
    SourceProvenanceError,
    build_book_manifest,
    build_book_manifest_from_packages,
    create_text_anchor,
    replay_text_anchor,
    validate_book_manifest,
    validate_text_anchor,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TextAnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = "Header\nA \u03b2eta appears here.\nA \u03b2eta appears here.\n"
        self.cleaned = self.raw

    def test_dual_offsets_are_deterministic_unicode_and_replayable(self) -> None:
        start = self.cleaned.index("\u03b2eta")
        first = create_text_anchor(
            anchor_id="book-a:chapter-a:0000:anchor-0000",
            page_id="book-a:chapter-a:0000",
            raw_text=self.raw,
            cleaned_text=self.cleaned,
            raw_char_start=start,
            raw_char_end=start + len("\u03b2eta"),
            cleaned_char_start=start,
            cleaned_char_end=start + len("\u03b2eta"),
            source_line_start=2,
            source_line_end=2,
            printed_page_number=7,
            source_pdf_page_number=19,
            review_status="accepted from upstream page markers",
        )
        second = create_text_anchor(
            anchor_id="book-a:chapter-a:0000:anchor-0000",
            page_id="book-a:chapter-a:0000",
            raw_text=self.raw,
            cleaned_text=self.cleaned,
            raw_char_start=start,
            raw_char_end=start + len("\u03b2eta"),
            cleaned_char_start=start,
            cleaned_char_end=start + len("\u03b2eta"),
            source_line_start=2,
            source_line_end=2,
            printed_page_number=7,
            source_pdf_page_number=19,
            review_status="accepted from upstream page markers",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], ANCHOR_SCHEMA_VERSION)
        self.assertEqual(first["exact_quote"], "\u03b2eta")
        self.assertEqual(replay_text_anchor(first, self.raw, self.cleaned), "\u03b2eta")
        self.assertEqual(first["pdf_bbox"], UNAVAILABLE)
        self.assertEqual(first["review_status"], "accepted from upstream page markers")

    def test_repeated_quote_requires_context_and_fail_closes_on_drift(self) -> None:
        start = self.cleaned.rindex("\u03b2eta")
        anchor = create_text_anchor(
            anchor_id="book-a:chapter-a:0000:anchor-0001",
            page_id="book-a:chapter-a:0000",
            raw_text=self.raw,
            cleaned_text=self.cleaned,
            raw_char_start=start,
            raw_char_end=start + 4,
            cleaned_char_start=start,
            cleaned_char_end=start + 4,
            source_line_start=3,
            source_line_end=3,
            printed_page_number=7,
            source_pdf_page_number=None,
            review_status="unreviewed",
        )
        self.assertNotEqual(anchor["prefix"], "")
        with self.assertRaisesRegex(SourceProvenanceError, "hash mismatch"):
            replay_text_anchor(anchor, self.raw, self.cleaned + "changed")
        ambiguous = dict(anchor, prefix="", suffix="")
        with self.assertRaisesRegex(SourceProvenanceError, "ambiguous"):
            validate_text_anchor(ambiguous, self.raw, self.cleaned)


class BookManifestTests(unittest.TestCase):
    def test_v01_manifest_adapter_preserves_unknown_line_ranges(self) -> None:
        source_manifest = {
            "document_id": "book-a",
            "source_pdf_locator": "book.pdf",
            "source_pdf_sha256": "a" * 64,
            "input_path_locator": "book.md",
            "input_sha256": "b" * 64,
            "pages": [{
                "page_id": "book-a:chapter-a:0000", "chapter_page_index": 0,
                "raw_path": "raw.md", "cleaned_path": "cleaned.md",
                "raw_sha256": "c" * 64, "cleaned_sha256": "d" * 64,
                "printed_page_number": 7, "source_pdf_page_number": 19,
                "review_status": "accepted from upstream page markers",
            }],
        }
        chunk_manifest = {"chunks": [{
            "chunk_id": "book-a:chapter-a:0000:0000", "page_id": "book-a:chapter-a:0000",
            "cleaned_char_start": 0, "cleaned_char_end": 2, "chunk_sha256": "e" * 64,
        }]}
        manifest = build_book_manifest_from_packages(
            book={"book_id": "book-a", "title": "Synthetic source", "edition": "v1"},
            source_manifest=source_manifest,
            chunk_manifest=chunk_manifest,
        )
        self.assertIsNone(manifest["pages"][0]["source_line_start"])
        self.assertEqual(
            manifest["pages"][0]["review_status"], "accepted from upstream page markers"
        )

    def test_manifest_hash_chain_and_page_semantics_are_stable(self) -> None:
        raw = "one\n"
        cleaned = "one\n"
        page = {
            "page_id": "book-a:chapter-a:0000",
            "chapter_page_index": 0,
            "raw_path": "pages/raw/0000.md",
            "cleaned_path": "pages/cleaned/0000.md",
            "raw_sha256": _sha256(raw),
            "cleaned_sha256": _sha256(cleaned),
            "source_line_start": 1,
            "source_line_end": 1,
            "printed_page_number": 7,
            "source_pdf_page_number": 19,
            "review_status": "accepted from upstream page markers",
        }
        chunk = {
            "chunk_id": "book-a:chapter-a:0000:0000",
            "page_id": page["page_id"],
            "cleaned_char_start": 0,
            "cleaned_char_end": 4,
            "chunk_sha256": _sha256(cleaned),
        }
        first = build_book_manifest(
            book={"book_id": "book-a", "title": "Synthetic source", "edition": "v1"},
            pdf={"pdf_id": "book-a:pdf", "locator": "book.pdf", "sha256": "a" * 64},
            markdown={"markdown_id": "book-a:markdown", "locator": "book.md", "sha256": "b" * 64},
            pages=[page],
            chunks=[chunk],
        )
        second = build_book_manifest(
            book={"book_id": "book-a", "title": "Synthetic source", "edition": "v1"},
            pdf={"pdf_id": "book-a:pdf", "locator": "book.pdf", "sha256": "a" * 64},
            markdown={"markdown_id": "book-a:markdown", "locator": "book.md", "sha256": "b" * 64},
            pages=[page],
            chunks=[chunk],
        )
        self.assertEqual(first, second)
        self.assertEqual(first["content_sha256"], second["content_sha256"])
        validate_book_manifest(first)

        changed = dict(first)
        changed["pages"] = [dict(page, source_pdf_page_number=20)]
        with self.assertRaisesRegex(SourceProvenanceError, "content hash mismatch"):
            validate_book_manifest(changed)

    def test_invalid_page_sequences_and_review_status_fail_closed(self) -> None:
        base = {
            "book_id": "book-a",
            "title": "Synthetic source",
            "edition": "v1",
        }
        pdf = {"pdf_id": "book-a:pdf", "locator": "book.pdf", "sha256": "a" * 64}
        markdown = {"markdown_id": "book-a:markdown", "locator": "book.md", "sha256": "b" * 64}
        page = {
            "page_id": "book-a:chapter-a:0001",
            "chapter_page_index": 1,
            "raw_path": "raw.md",
            "cleaned_path": "cleaned.md",
            "raw_sha256": "c" * 64,
            "cleaned_sha256": "d" * 64,
            "source_line_start": 1,
            "source_line_end": 1,
            "printed_page_number": 7,
            "source_pdf_page_number": 19,
            "review_status": "verified against source PDF",
        }
        with self.assertRaisesRegex(SourceProvenanceError, "contiguous"):
            build_book_manifest(book=base, pdf=pdf, markdown=markdown, pages=[page], chunks=[])
        with self.assertRaisesRegex(SourceProvenanceError, "review_status"):
            build_book_manifest(
                book=base,
                pdf=pdf,
                markdown=markdown,
                pages=[dict(page, chapter_page_index=0, review_status="verified")],
                chunks=[],
            )
