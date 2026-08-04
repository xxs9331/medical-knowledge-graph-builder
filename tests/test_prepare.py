import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.cli import main
from medical_kg_sourceprep.prepare import PreparationError, prepare_source


def _config(input_path: Path, output_path: Path, **overrides: object) -> dict[str, object]:
    return {
        "input_path": input_path,
        "output_path": output_path,
        "document_id": "demo-book",
        "chapter_id": "chapter-a",
        "ocr_engine": "baidu/Unlimited-OCR",
        "source_pdf_locator": "/outside/read-only/chapter.pdf",
        "source_pdf_sha256": "a" * 64,
        "printed_page_start": 4,
        "source_pdf_page_start": 21,
        "page_count": 3,
        **overrides,
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _page_map(source: Path, **overrides: object) -> dict[str, object]:
    return {
        "schema_version": "source-page-map/v0.1",
        "input_sha256": _digest(source),
        "page_count": 3,
        "pages": [
            {
                "chapter_page_index": 0,
                "printed_page_number": 4,
                "source_pdf_page_number": 21,
                "start_line": 1,
                "end_line": 3,
                "review_status": "verified against source PDF",
            },
            {
                "chapter_page_index": 1,
                "printed_page_number": 5,
                "source_pdf_page_number": 22,
                "start_line": 4,
                "end_line": 6,
                "review_status": "verified against source PDF",
            },
            {
                "chapter_page_index": 2,
                "printed_page_number": 6,
                "source_pdf_page_number": 23,
                "start_line": 7,
                "end_line": 9,
                "review_status": "verified against source PDF",
            },
        ],
        **overrides,
    }


class PrepareSourceTests(unittest.TestCase):
    def test_page_directory_preserves_provenance_and_cleans_verified_furniture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            pages = tmp_path / "pages"
            pages.mkdir()
            (pages / "page-10.md").write_text("Running Header\n10\nDose: 10 mg\nFooter X\n", encoding="utf-8")
            (pages / "page-2.md").write_text("Running Header\n2\n1. ordinary list item\n| A | 2 |\nFooter X\n", encoding="utf-8")
            (pages / "page-3.md").write_text("Running Header\n3\nArea is 4 cm2\nFooter X\n", encoding="utf-8")

            manifest = prepare_source(**_config(pages, tmp_path / "package"))

            self.assertEqual([record["printed_page_number"] for record in manifest["pages"]], [4, 5, 6])
            self.assertEqual([record["source_pdf_page_number"] for record in manifest["pages"]], [21, 22, 23])
            self.assertEqual([record["chapter_page_index"] for record in manifest["pages"]], [0, 1, 2])
            self.assertEqual(manifest["pages"][0]["page_id"], "demo-book:chapter-a:0000")
            clean = (tmp_path / "package" / manifest["pages"][0]["cleaned_path"]).read_text(encoding="utf-8")
            self.assertEqual(clean, "1. ordinary list item\n| A | 2 |\n")
            self.assertIn("Dose: 10 mg", (tmp_path / "package" / manifest["pages"][2]["cleaned_path"]).read_text(encoding="utf-8"))
            for record in manifest["pages"]:
                self.assertEqual(record["raw_sha256"], _digest(tmp_path / "package" / record["raw_path"]))
                self.assertEqual(record["cleaned_sha256"], _digest(tmp_path / "package" / record["cleaned_path"]))


    def test_combined_markdown_supports_top_and_bottom_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            combined = tmp_path / "combined.md"
            combined.write_text(
                "Header\n4\nalpha Unicode 5 microg\nFooter\nHeader\nbeta formula x=2\n5\nFooter\nHeader\n6\ngamma\nFooter\n",
                encoding="utf-8",
            )

            manifest = prepare_source(**_config(combined, tmp_path / "package"))

            raw = (tmp_path / "package" / manifest["pages"][1]["raw_path"]).read_text(encoding="utf-8")
            self.assertEqual(raw, "Header\nbeta formula x=2\nFooter\n")
            clean = (tmp_path / "package" / manifest["pages"][1]["cleaned_path"]).read_text(encoding="utf-8")
            self.assertEqual(clean, "beta formula x=2\n")


    def test_invalid_combined_markers_fail_without_output(self) -> None:
        for content in ("4\na\n6\nb\n", "4\na\n4\nb\n5\nc\n", "4\na\n5\nb\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temporary_directory:
                tmp_path = Path(temporary_directory)
                source = tmp_path / "combined.md"
                source.write_text(content, encoding="utf-8")
                output = tmp_path / "package"

                with self.assertRaises(PreparationError):
                    prepare_source(**_config(source, output))

                self.assertFalse(output.exists())

    def test_combined_content_before_first_marker_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = tmp_path / "combined.md"
            source.write_text("unassigned opening text\n4\nfirst\n5\nsecond\n6\nthird\n", encoding="utf-8")
            output = tmp_path / "package"

            with self.assertRaisesRegex(PreparationError, "before the first page marker"):
                prepare_source(**_config(source, output))

            self.assertFalse(output.exists())

    def test_page_map_preserves_exact_ranges_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = tmp_path / "combined.md"
            source.write_text("4\nalpha\nfooter\n5\nbeta\nfooter\n6\ngamma\nfooter\n", encoding="utf-8")
            page_map = tmp_path / "page-map.json"
            page_map.write_text(json.dumps(_page_map(source)), encoding="utf-8")

            manifest = prepare_source(**_config(source, tmp_path / "package", page_map_path=page_map))

            self.assertEqual(manifest["page_map_locator"], str(page_map))
            self.assertEqual(manifest["page_map_sha256"], _digest(page_map))
            self.assertEqual(
                [(page["source_line_start"], page["source_line_end"]) for page in manifest["pages"]],
                [(1, 3), (4, 6), (7, 9)],
            )
            self.assertEqual(
                [page["boundary_review_status"] for page in manifest["pages"]],
                ["verified against source PDF"] * 3,
            )
            self.assertEqual(
                [page["review_status"] for page in manifest["pages"]],
                ["verified against source PDF"] * 3,
            )
            raw = (tmp_path / "package" / manifest["pages"][1]["raw_path"]).read_text(encoding="utf-8")
            self.assertEqual(raw, "5\nbeta\nfooter\n")

    def test_page_map_accepts_upstream_marker_review_status_without_upgrading_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = tmp_path / "combined.md"
            source.write_text("4\nalpha\nfooter\n5\nbeta\nfooter\n6\ngamma\nfooter\n", encoding="utf-8")
            page_map = tmp_path / "page-map.json"
            pages = _page_map(source)["pages"]
            assert isinstance(pages, list)
            upstream_status = "accepted from upstream page markers"
            page_map.write_text(
                json.dumps(_page_map(source, pages=[{**page, "review_status": upstream_status} for page in pages])),
                encoding="utf-8",
            )

            manifest = prepare_source(**_config(source, tmp_path / "package", page_map_path=page_map))

            self.assertEqual([page["review_status"] for page in manifest["pages"]], [upstream_status] * 3)
            self.assertEqual([page["boundary_review_status"] for page in manifest["pages"]], [upstream_status] * 3)

    def test_page_map_rejections_are_atomic_and_never_fall_back_to_automatic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = tmp_path / "combined.md"
            source.write_text("opening\n4\nalpha\n5\nbeta\nfooter\n6\ngamma\nfooter\n", encoding="utf-8")
            invalid_maps = {
                "input_hash": {"input_sha256": "0" * 64},
                "schema": {"schema_version": "source-page-map/v0.0"},
                "unknown_top": {"extra": True},
                "unknown_page": {"pages": [{**page, "extra": True} if page["chapter_page_index"] == 1 else page for page in _page_map(source)["pages"]]},
                "wrong_count": {"page_count": 2},
                "duplicate_sequence": {"pages": _page_map(source)["pages"][:2] + [_page_map(source)["pages"][1]]},
                "gap": {"pages": [{**page, "start_line": page["start_line"] + (1 if page["chapter_page_index"] == 1 else 0)} for page in _page_map(source)["pages"]]},
                "overlap": {"pages": [{**page, "start_line": page["start_line"] - (1 if page["chapter_page_index"] == 1 else 0)} for page in _page_map(source)["pages"]]},
                "reversed": {"pages": [{**page, "end_line": 3, "start_line": 4} if page["chapter_page_index"] == 0 else page for page in _page_map(source)["pages"]]},
                "out_of_range": {"pages": [{**page, "end_line": 10} if page["chapter_page_index"] == 2 else page for page in _page_map(source)["pages"]]},
                "unassigned_suffix": {"pages": [{**page, "end_line": 8} if page["chapter_page_index"] == 2 else page for page in _page_map(source)["pages"]]},
                "marker": {"pages": [{**page, "printed_page_number": 7} if page["chapter_page_index"] == 1 else page for page in _page_map(source)["pages"]]},
                "review": {"pages": [{**page, "review_status": "pending"} if page["chapter_page_index"] == 1 else page for page in _page_map(source)["pages"]]},
            }
            for name, overrides in invalid_maps.items():
                with self.subTest(name=name):
                    page_map = tmp_path / f"{name}.json"
                    page_map.write_text(json.dumps(_page_map(source, **overrides)), encoding="utf-8")
                    output = tmp_path / f"package-{name}"

                    with self.assertRaises(PreparationError):
                        prepare_source(**_config(source, output, page_map_path=page_map))

                    self.assertFalse(output.exists())

    def test_page_map_requires_one_edge_printed_marker_per_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = tmp_path / "combined.md"
            source.write_text("4\nalpha\nfooter\n5\nbeta\n5\nfooter\n6\ngamma\nfooter\n", encoding="utf-8")
            pages = _page_map(source)["pages"]
            assert isinstance(pages, list)
            pages[1] = {**pages[1], "start_line": 4, "end_line": 7}
            pages[2] = {**pages[2], "start_line": 8, "end_line": 10}
            page_map = tmp_path / "page-map.json"
            page_map.write_text(json.dumps(_page_map(source, pages=pages)), encoding="utf-8")

            with self.assertRaisesRegex(PreparationError, "one edge printed-page marker"):
                prepare_source(**_config(source, tmp_path / "package", page_map_path=page_map))


    def test_output_collision_and_repeat_runs_are_safe_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            pages = tmp_path / "pages"
            pages.mkdir()
            for number in range(3):
                (pages / f"p{number}.md").write_text(f"{number + 1}\nbody {number}\n", encoding="utf-8")

            with self.assertRaises(PreparationError):
                prepare_source(**_config(pages, pages))

            first = prepare_source(**_config(pages, tmp_path / "one", generation_timestamp="2026-01-01T00:00:00Z"))
            second = prepare_source(**_config(pages, tmp_path / "two", generation_timestamp="2026-01-01T00:00:00Z"))
            self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_input_hash_commits_to_exact_file_bytes_and_directory_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            first = tmp_path / "first"
            second = tmp_path / "second"
            for directory in (first, second):
                directory.mkdir()
            (first / "page-1.md").write_bytes(b"alpha\nbeta\n")
            (first / "page-2.md").write_bytes(b"gamma\n")
            (first / "page-3.md").write_bytes(b"delta\n")
            (second / "page-1.md").write_bytes(b"alpha\n")
            (second / "page-2.md").write_bytes(b"beta\ngamma\n")
            (second / "page-3.md").write_bytes(b"delta\n")

            first_manifest = prepare_source(**_config(first, tmp_path / "first-package"))
            second_manifest = prepare_source(**_config(second, tmp_path / "second-package"))

            self.assertNotEqual(first_manifest["input_sha256"], second_manifest["input_sha256"])
            self.assertEqual(
                first_manifest["input_sha256"],
                prepare_source(**_config(first, tmp_path / "repeat-package"))["input_sha256"],
            )

    def test_file_input_hash_uses_exact_original_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source = tmp_path / "combined.md"
            original = b"4\r\nfirst\r\n5\r\nsecond\r\n6\r\nthird\r\n"
            source.write_bytes(original)

            manifest = prepare_source(**_config(source, tmp_path / "package"))

            self.assertEqual(manifest["input_sha256"], hashlib.sha256(original).hexdigest())

    def test_cli_writes_a_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            pages = tmp_path / "pages"
            pages.mkdir()
            for number in range(3):
                (pages / f"p{number}.md").write_text(f"body {number}\n", encoding="utf-8")
            output = tmp_path / "package"
            main([
                "--input", str(pages), "--output", str(output), "--document-id", "book",
                "--chapter-id", "chapter", "--ocr-engine", "baidu/Unlimited-OCR",
                "--source-pdf-locator", "/outside/chapter.pdf", "--source-pdf-sha256", "b" * 64,
                "--printed-page-start", "4", "--source-pdf-page-start", "21", "--page-count", "3",
                "--generation-timestamp", "2026-01-01T00:00:00Z",
            ])
            self.assertTrue((output / "manifest.json").is_file())
