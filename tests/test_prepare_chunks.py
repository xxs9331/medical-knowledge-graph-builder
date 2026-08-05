import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medical_kg_sourceprep.provenance.chunks import ChunkingError, prepare_chunks


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_package(root: Path, pages: list[str]) -> Path:
    package = root / "source"
    package.mkdir(parents=True)
    records = []
    for index, content in enumerate(pages):
        relative = Path("pages/cleaned") / f"{index:04d}.md"
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
        records.append(
            {
                "page_id": f"demo-book:chapter-a:{index:04d}",
                "chapter_page_index": index,
                "printed_page_number": index + 4,
                "source_pdf_page_number": index + 21,
                "cleaned_path": relative.as_posix(),
                "cleaned_sha256": _sha256(content.encode("utf-8")),
                "review_status": "verified-boundary",
                "warnings": [],
            }
        )
    manifest = {
        "schema_version": "source-package/v0.1",
        "document_id": "demo-book",
        "chapter_id": "chapter-a",
        "page_count": len(records),
        "pages": records,
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return package


class PrepareChunksTests(unittest.TestCase):
    def test_pages_reassemble_without_crossing_and_preserve_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = _source_package(tmp_path, ["alpha\n\nbeta\ngamma\n", "delta\nepsilon\n"])
            output = tmp_path / "chunks"

            manifest = prepare_chunks(
                source, output, max_chars=8, generation_timestamp="2026-01-01T00:00:00Z"
            )

            self.assertEqual(manifest["schema_version"], "evidence-chunk-package/v0.1")
            self.assertEqual(manifest["page_count"], 2)
            self.assertEqual(
                manifest["source_manifest_sha256"],
                _sha256((source / "manifest.json").read_bytes()),
            )
            for page in manifest["pages"]:
                chunks = [
                    chunk for chunk in manifest["chunks"] if chunk["page_id"] == page["page_id"]
                ]
                rebuilt = "".join(
                    (output / chunk["chunk_path"]).read_text(encoding="utf-8") for chunk in chunks
                )
                source_page = source / page["cleaned_path"]
                self.assertEqual(rebuilt, source_page.read_text(encoding="utf-8"))
                self.assertEqual(chunks[0]["cleaned_char_start"], 0)
                self.assertEqual(chunks[-1]["cleaned_char_end"], len(rebuilt))
                self.assertEqual(chunks[0]["source_page"], page)
                self.assertTrue(all(chunk["document_id"] == "demo-book" for chunk in chunks))
                self.assertTrue(
                    all(chunk["review_status"] == "verified-boundary" for chunk in chunks)
                )
                self.assertTrue(
                    all(
                        chunk["chunk_sha256"]
                        == _sha256((output / chunk["chunk_path"]).read_bytes())
                        for chunk in chunks
                    )
                )

    def test_deterministic_cuts_ids_hashes_and_preferred_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = _source_package(tmp_path, ["one\ntwo\n\nthree\nfour\n"])
            first = prepare_chunks(
                source,
                tmp_path / "first",
                max_chars=10,
                generation_timestamp="2026-01-01T00:00:00Z",
            )
            second = prepare_chunks(
                source,
                tmp_path / "second",
                max_chars=10,
                generation_timestamp="2026-01-01T00:00:00Z",
            )

            self.assertEqual(first, second)
            self.assertEqual(
                [
                    (chunk["cleaned_char_start"], chunk["cleaned_char_end"])
                    for chunk in first["chunks"]
                ],
                [(0, 9), (9, 15), (15, 20)],
            )
            self.assertEqual(
                [chunk["chunk_id"] for chunk in first["chunks"]],
                [
                    "demo-book:chapter-a:0000:0000",
                    "demo-book:chapter-a:0000:0001",
                    "demo-book:chapter-a:0000:0002",
                ],
            )

    def test_protected_blocks_remain_atomic_and_oversize_is_warned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            content = (
                "pre\n\n<table>\ncell-contents\n</table>\n\n```text\ncode-block\n```\n\n"
                "\\[\nx + y\n\\]\n\n$$\na=b\n$$\npost\n"
            )
            source = _source_package(tmp_path, [content])
            manifest = prepare_chunks(
                source,
                tmp_path / "chunks",
                max_chars=10,
                generation_timestamp="2026-01-01T00:00:00Z",
            )

            chunks = [
                (tmp_path / "chunks" / chunk["chunk_path"]).read_text(encoding="utf-8")
                for chunk in manifest["chunks"]
            ]
            self.assertTrue(any("<table>\ncell-contents\n</table>" in chunk for chunk in chunks))
            self.assertTrue(any("```text\ncode-block\n```" in chunk for chunk in chunks))
            self.assertTrue(any("\\[\nx + y\n\\]" in chunk for chunk in chunks))
            self.assertTrue(any("$$\na=b\n$$" in chunk for chunk in chunks))
            self.assertTrue(
                any("oversize_atomic_block" in chunk["warnings"] for chunk in manifest["chunks"])
            )

    def test_invalid_packages_and_outputs_fail_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = _source_package(tmp_path, ["alpha\n", "beta\n"])
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["pages"][1]["page_id"] = manifest["pages"][0]["page_id"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            output = tmp_path / "chunks"
            with self.assertRaisesRegex(ChunkingError, "duplicate"):
                prepare_chunks(source, output, max_chars=8)
            self.assertFalse(output.exists())

            valid_source = _source_package(tmp_path / "second", ["alpha\n"])
            with self.assertRaises(ChunkingError):
                prepare_chunks(valid_source, tmp_path / "invalid", max_chars=0)
            self.assertFalse((tmp_path / "invalid").exists())
            with self.assertRaises(ChunkingError):
                prepare_chunks(valid_source, valid_source / "nested", max_chars=8)
            self.assertFalse((valid_source / "nested").exists())

            collision = tmp_path / "collision"
            collision.mkdir()
            with self.assertRaisesRegex(ChunkingError, "already exists"):
                prepare_chunks(valid_source, collision, max_chars=8)

    def test_missing_or_changed_cleaned_sources_fail_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = _source_package(tmp_path, ["alpha\n"])
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            page = manifest["pages"][0]
            cleaned = source / page["cleaned_path"]

            cleaned.unlink()
            with self.assertRaisesRegex(ChunkingError, "missing"):
                prepare_chunks(source, tmp_path / "missing", max_chars=8)
            self.assertFalse((tmp_path / "missing").exists())

            cleaned.write_text("changed\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(ChunkingError, "hash mismatch"):
                prepare_chunks(source, tmp_path / "mismatch", max_chars=8)
            self.assertFalse((tmp_path / "mismatch").exists())

    def test_source_change_during_run_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = _source_package(tmp_path, ["alpha\nbeta\n"])
            output = tmp_path / "chunks"
            cleaned = source / "pages/cleaned/0000.md"

            from medical_kg_sourceprep.provenance import chunks

            original_slice_page = chunks._slice_page

            def change_source(text: str, max_chars: int) -> list[tuple[int, int, list[str]]]:
                cleaned.write_text("different\n", encoding="utf-8", newline="\n")
                return original_slice_page(text, max_chars)

            with patch("medical_kg_sourceprep.provenance.chunks._slice_page", side_effect=change_source):
                with self.assertRaisesRegex(ChunkingError, "changed during chunking"):
                    prepare_chunks(source, output, max_chars=8)
            self.assertFalse(output.exists())
