import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.provenance.chunks import prepare_chunks
from medical_kg_sourceprep.provenance.semantic_sections import (
    SemanticSectionError,
    build_semantic_sections,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_package(root: Path, pages: list[str]) -> Path:
    package = root / "source"
    package.mkdir(parents=True)
    records = []
    for index, text in enumerate(pages):
        relative = Path("pages/cleaned") / f"{index:04d}.md"
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        records.append(
            {
                "page_id": f"book:full:{index:04d}",
                "chapter_page_index": index,
                "printed_page_number": index + 1,
                "source_pdf_page_number": index + 10,
                "cleaned_path": relative.as_posix(),
                "cleaned_sha256": _sha256(text.encode("utf-8")),
                "review_status": "verified",
                "warnings": [],
            }
        )
    manifest = {
        "schema_version": "source-package/v0.1",
        "document_id": "book",
        "chapter_id": "full",
        "page_count": len(records),
        "pages": records,
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8", newline="\n"
    )
    return package


class SemanticSectionTests(unittest.TestCase):
    def test_sections_follow_headings_across_pages_and_replay_all_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pages = [
                "# 第一章 检验\n一、血液\n（一）血红蛋白\n定义。\n【参考区间】\n正常。\n",
                "延续解释。\n（二）红细胞\n定义二。\n",
            ]
            evidence = root / "evidence"
            prepare_chunks(_source_package(root, pages), evidence, max_chars=12)

            output = root / "semantic"
            manifest = build_semantic_sections(
                evidence,
                output,
                target_chars=30,
                max_chars=80,
                generation_timestamp="2026-01-01T00:00:00Z",
            )

            self.assertEqual(manifest["schema_version"], "semantic-section-package/v0.2")
            self.assertEqual(manifest["source_counts"]["characters"], sum(map(len, pages)))
            rebuilt = "".join(
                (output / section["section_file"]).read_text(encoding="utf-8")
                for section in manifest["sections"]
            )
            self.assertEqual(rebuilt, "".join(pages))
            red_cell = next(section for section in manifest["sections"] if section["title"] == "红细胞")
            self.assertEqual(red_cell["section_path"], ["第一章 检验", "血液", "红细胞"])
            self.assertEqual(red_cell["page_ids"], ["book:full:0001"])
            hemoglobin = next(
                section for section in manifest["sections"] if section["title"] == "血红蛋白"
            )
            self.assertEqual(hemoglobin["facets"], ["参考区间"])
            self.assertEqual(hemoglobin["kind"], "semantic")
            self.assertEqual(hemoglobin["content_role"], "clinical_content")
            self.assertTrue(hemoglobin["extraction_eligible"])
            self.assertEqual(hemoglobin["page_ids"], ["book:full:0000", "book:full:0001"])
            self.assertTrue(all(section["source_spans"] for section in manifest["sections"]))

    def test_roles_and_input_view_mask_running_headers_without_shifting_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pages = [
                "# 第一章 检验\n第一章\n检验\n一、项目组\n（一）项目甲\n定义。\n",
                "第一章 检验\n（二）项目乙\n见第二章。\n# 附录三 索引\nA：甲\n",
            ]
            evidence = root / "evidence"
            prepare_chunks(_source_package(root, pages), evidence, max_chars=20)

            output = root / "semantic"
            manifest = build_semantic_sections(
                evidence,
                output,
                target_chars=100,
                max_chars=200,
                generation_timestamp="2026-01-01T00:00:00Z",
            )
            by_title = {section["title"]: section for section in manifest["sections"]}

            self.assertEqual(by_title["第一章 检验"]["content_role"], "introduction")
            self.assertEqual(by_title["项目组"]["content_role"], "structural")
            self.assertFalse(by_title["项目组"]["extraction_eligible"])
            self.assertEqual(by_title["项目甲"]["content_role"], "clinical_content")
            self.assertEqual(by_title["项目乙"]["content_role"], "cross_reference")
            self.assertEqual(by_title["项目乙"]["extraction_route"], "cross_reference_resolution")
            self.assertEqual(by_title["附录三 索引"]["content_role"], "index")

            project_a = by_title["项目甲"]
            original = (output / project_a["section_file"]).read_text(encoding="utf-8")
            input_view = (output / project_a["input_view_file"]).read_text(encoding="utf-8")
            self.assertEqual(len(input_view), len(original))
            self.assertIn("第一章 检验", original)
            self.assertNotIn("第一章 检验", input_view)
            self.assertEqual(len(project_a["noise_spans"]), 1)

            project_b = by_title["项目乙"]
            original = (output / project_b["section_file"]).read_text(encoding="utf-8")
            input_view = (output / project_b["input_view_file"]).read_text(encoding="utf-8")
            self.assertEqual(len(input_view), len(original))
            self.assertEqual(input_view, original)
            self.assertEqual(project_b["noise_spans"], [])

    def test_oversize_sections_split_at_replayable_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            page = "# 章\n一、项目\n" + "甲" * 25 + "\n\n" + "乙" * 25 + "\n"
            evidence = root / "evidence"
            prepare_chunks(_source_package(root, [page]), evidence, max_chars=16)

            first = build_semantic_sections(
                evidence,
                root / "first",
                target_chars=20,
                max_chars=30,
                generation_timestamp="2026-01-01T00:00:00Z",
            )
            second = build_semantic_sections(
                evidence,
                root / "second",
                target_chars=20,
                max_chars=30,
                generation_timestamp="2026-01-01T00:00:00Z",
            )

            self.assertEqual(first, second)
            self.assertTrue(all(section["char_count"] <= 30 for section in first["sections"]))
            self.assertEqual(
                "".join(
                    (root / "first" / section["section_file"]).read_text(encoding="utf-8")
                    for section in first["sections"]
                ),
                page,
            )

    def test_invalid_configuration_and_existing_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = root / "evidence"
            prepare_chunks(_source_package(root, ["text\n"]), evidence, max_chars=10)
            with self.assertRaisesRegex(SemanticSectionError, "positive and ordered"):
                build_semantic_sections(evidence, root / "bad", target_chars=20, max_chars=10)
            collision = root / "collision"
            collision.mkdir()
            with self.assertRaisesRegex(SemanticSectionError, "already exists"):
                build_semantic_sections(evidence, collision)


if __name__ == "__main__":
    unittest.main()
