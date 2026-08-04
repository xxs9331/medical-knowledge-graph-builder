import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.book_sources import (
    build_book_manifest,
    replay_text_anchor,
    validate_text_anchor,
)
from medical_kg_sourceprep.knowledge_graph import (
    GraphBuildError,
    KnowledgeGraphBuilder,
    PageText,
    _unique_raw_offset,
    read_graph,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _inputs(texts: list[str]) -> tuple[dict, tuple[PageText, ...]]:
    return _paired_inputs([(text, text) for text in texts])


def _paired_inputs(texts: list[tuple[str, str]]) -> tuple[dict, tuple[PageText, ...]]:
    pages = []
    chunks = []
    records = []
    for index, (raw_text, cleaned_text) in enumerate(texts):
        page_id = f"synthetic-book:page-{index}"
        pages.append({
            "page_id": page_id,
            "chapter_page_index": index,
            "raw_path": f"raw/{index}.md",
            "cleaned_path": f"cleaned/{index}.md",
            "raw_sha256": _sha256(raw_text),
            "cleaned_sha256": _sha256(cleaned_text),
            "source_line_start": 1,
            "source_line_end": raw_text.count("\n") + 1,
            "printed_page_number": index + 1,
            "source_pdf_page_number": index + 1,
            "review_status": "unreviewed",
        })
        chunks.append({
            "chunk_id": f"synthetic-book:chunk-{index}", "page_id": page_id,
            "cleaned_char_start": 0, "cleaned_char_end": len(cleaned_text),
            "chunk_sha256": _sha256(cleaned_text),
        })
        records.append(PageText(page_id=page_id, raw_text=raw_text, cleaned_text=cleaned_text))
    return build_book_manifest(
        book={"book_id": "synthetic-book", "title": "Synthetic", "edition": "v1"},
        pdf={"pdf_id": "synthetic-book:pdf", "locator": "synthetic.pdf", "sha256": "a" * 64},
        markdown={"markdown_id": "synthetic-book:markdown", "locator": "synthetic.md", "sha256": "b" * 64},
        pages=pages, chunks=chunks,
    ), tuple(records)


class KnowledgeGraphTests(unittest.TestCase):
    def test_raw_anchor_uses_shorter_unique_context_before_removed_page_marker(self) -> None:
        cleaned = "<table><tr><td>O</td><td>O</td></tr></table>\nnext"
        raw = "<table><tr><td>O</td><td>O</td></tr></table>\n26\nnext"
        offset = cleaned.rindex("O")

        self.assertEqual(
            _unique_raw_offset(raw, cleaned, "O", offset),
            raw.rindex("O"),
        )

    def test_html_entity_cells_keep_display_text_and_replayable_source_spans(self) -> None:
        cleaned = (
            "<table><caption>Limits &amp; notes</caption>"
            "<tr><th><strong>Signal &amp; kind</strong></th><th>Threshold</th></tr>"
            "<tr><td>same &amp; same</td><td>&lt;30g/L</td></tr>"
            "<tr><td>same &amp; same</td><td>&#62;10</td></tr></table>\n"
        )
        raw = "Verified furniture\n" + cleaned
        manifest, pages = _paired_inputs([(raw, cleaned)])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            KnowledgeGraphBuilder().build(database, manifest, pages)
            with sqlite3.connect(database) as connection:
                cells = connection.execute(
                    "SELECT table_cells.text, anchors.anchor_json FROM table_cells "
                    "JOIN anchors ON anchors.anchor_id = table_cells.anchor_id "
                    "ORDER BY table_cells.row_index, table_cells.column_index"
                ).fetchall()
                title = json.loads(connection.execute(
                    "SELECT payload FROM nodes WHERE node_type = 'Table'"
                ).fetchone()[0])["title"]
        self.assertEqual(title, "Limits & notes")
        self.assertEqual([text for text, _ in cells], ["Signal & kind", "Threshold", "same & same", "<30g/L", "same & same", ">10"])
        anchors = [json.loads(anchor_json) for _, anchor_json in cells]
        self.assertEqual(
            [anchor["exact_quote"] for anchor in anchors],
            ["<strong>Signal &amp; kind</strong>", "Threshold", "same &amp; same", "&lt;30g/L", "same &amp; same", "&#62;10"],
        )
        self.assertNotEqual(anchors[2]["cleaned_char_start"], anchors[4]["cleaned_char_start"])
        for anchor in anchors:
            self.assertEqual(replay_text_anchor(anchor, raw, cleaned), anchor["exact_quote"])

    def test_table_source_quote_drift_fails_without_database(self) -> None:
        cleaned = "<table><tr><td>&lt;value</td></tr></table>\n"
        raw = "<table><tr><td>&gt;value</td></tr></table>\n"
        manifest, pages = _paired_inputs([(raw, cleaned)])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            with self.assertRaisesRegex(GraphBuildError, "cannot uniquely map"):
                KnowledgeGraphBuilder().build(database, manifest, pages)
            self.assertFalse(database.exists())

    def test_unicode_and_duplicate_heading_ids_are_stable_ascii_and_distinct(self) -> None:
        text = (
            "# \u7ae0\u8282\u4e00\n## \u5c0f\u8282\uff1a\uff21\n"
            "# \u7ae0\u8282\u4e8c\n## \u5c0f\u8282\uff1aA\n"
            "# \u7ae0\u8282\u4e00\n## \u5c0f\u8282\uff1a\uff21\n"
        )
        manifest, pages = _inputs([text])

        def build_heading_nodes(database: Path) -> list[tuple[str, str, str]]:
            KnowledgeGraphBuilder().build(database, manifest, pages)
            with sqlite3.connect(database) as connection:
                return [
                    (node_id, node_type, json.loads(payload)["title"])
                    for node_id, node_type, payload in connection.execute(
                        "SELECT node_id, node_type, payload FROM nodes "
                        "WHERE node_type IN ('Chapter', 'Section') ORDER BY node_id"
                    )
                ]

        with tempfile.TemporaryDirectory() as directory:
            first_database = Path(directory) / "first.sqlite"
            first = build_heading_nodes(first_database)
            second = build_heading_nodes(Path(directory) / "second.sqlite")
            with sqlite3.connect(first_database) as connection:
                title_quotes = [
                    json.loads(anchor_json)["exact_quote"]
                    for anchor_json, in connection.execute(
                        "SELECT anchor_json FROM anchors WHERE role IN ('chapter_title', 'section_title')"
                    )
                ]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(len({node_id for node_id, _, _ in first}), 6)
        self.assertEqual([title for _, _, title in first].count("\u7ae0\u8282\u4e00"), 2)
        self.assertEqual([title for _, _, title in first].count("\u5c0f\u8282\uff1a\uff21"), 2)
        self.assertCountEqual(title_quotes, ["\u7ae0\u8282\u4e00", "\u5c0f\u8282\uff1a\uff21", "\u7ae0\u8282\u4e8c", "\u5c0f\u8282\uff1aA", "\u7ae0\u8282\u4e00", "\u5c0f\u8282\uff1a\uff21"])
        for node_id, _, _ in first:
            self.assertTrue(node_id.isascii())
            self.assertRegex(node_id, r"^[A-Za-z0-9][A-Za-z0-9:._-]*$")

    def test_deletion_only_cleaning_creates_replayable_dual_offset_anchors(self) -> None:
        cleaned = (
            "Test Item: Plain Item\n"
            "Rule: IF first condition AND second condition THEN synthetic conclusion\n"
        )
        raw = "Synthetic page header\n" + cleaned + "Synthetic page footer\n"
        manifest, pages = _paired_inputs([(raw, cleaned)])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            KnowledgeGraphBuilder().build(database, manifest, pages)
            with sqlite3.connect(database) as connection:
                anchors = [json.loads(row[0]) for row in connection.execute(
                    "SELECT anchor_json FROM anchors ORDER BY anchor_id"
                )]
            self.assertTrue(anchors)
            self.assertTrue(all(anchor["raw_char_start"] != anchor["cleaned_char_start"] for anchor in anchors))
            self.assertTrue(any(anchor["raw_char_start"] > anchor["cleaned_char_start"] for anchor in anchors))
            for anchor in anchors:
                validate_text_anchor(anchor, raw, cleaned)
                self.assertEqual(replay_text_anchor(anchor, raw, cleaned), anchor["exact_quote"])

    def test_non_deletion_or_ambiguous_anchor_mapping_fails_without_database(self) -> None:
        cleaned = "Test Item: Plain\nRule: IF first condition THEN synthetic conclusion\n"
        cases = [
            (
                "Test Item: Plain\nRule: IF first condition THEN rewritten conclusion\n",
                cleaned,
                "cannot uniquely map",
            ),
            (
                "Test Item: Pl<removed>ain\nRule: IF first condition THEN synthetic conclusion\n",
                cleaned,
                "cannot uniquely map",
            ),
        ]
        for raw, cleaned_text, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                manifest, pages = _paired_inputs([(raw, cleaned_text)])
                database = Path(directory) / "graph.sqlite"
                with self.assertRaisesRegex(GraphBuildError, message):
                    KnowledgeGraphBuilder().build(database, manifest, pages)
                self.assertFalse(database.exists())

        ambiguous_cleaned = "Test Item: Plain\n" + ("x" * 40) + "\nRule: IF first condition THEN synthetic conclusion\n"
        ambiguous_raw = ambiguous_cleaned[: len("Test Item: Plain\n") + 32] + "raw-only marker\n" + ambiguous_cleaned
        manifest, pages = _paired_inputs([(ambiguous_raw, ambiguous_cleaned)])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            with self.assertRaisesRegex(GraphBuildError, "cannot uniquely map"):
                KnowledgeGraphBuilder().build(database, manifest, pages)
            self.assertFalse(database.exists())

        manifest, pages = _paired_inputs([(cleaned, cleaned)])
        drifted = (PageText(pages[0].page_id, cleaned + "drift", cleaned),)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            with self.assertRaisesRegex(GraphBuildError, "hash drift"):
                KnowledgeGraphBuilder().build(database, manifest, drifted)
            self.assertFalse(database.exists())

    def test_builds_traceable_candidate_rule_and_navigation_edges(self) -> None:
        text = (
            "# Chapter Alpha\n\n## Section Beta\n\n"
            "Test Item: Sample Item\n"
            "Reference Range: 4 to 9\n"
            "Explanation: Short synthetic explanation.\n"
            "Rule: IF first condition AND second condition THEN synthetic conclusion\n"
            "Formula: x = y + z\n\n"
            "<table><caption>Sample table</caption><tr><th>Column A</th><th>Column B</th></tr>"
            "<tr><td>left</td><td>right</td></tr></table>\n"
        )
        manifest, pages = _inputs([text])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            result = KnowledgeGraphBuilder().build(database, manifest, pages)
            self.assertEqual(result.rule_count, 1)
            self.assertEqual(result.candidate_rule_count, 1)
            graph = read_graph(database)
            self.assertEqual(graph.integrity_check(), "ok")
            self.assertEqual(graph.count_nodes("KnowledgeRule"), 1)
            self.assertEqual(graph.count_nodes("AtomicPredicate"), 2)
            self.assertEqual(graph.count_edges("semantic"), 8)
            rule = graph.rules()[0]
            self.assertEqual(rule["status"], "candidate")
            self.assertEqual(len(rule["condition_anchors"]), 2)
            self.assertTrue(rule["conclusion_anchor"])
            self.assertEqual(graph.table_cells()[0]["table_title"], "Sample table")
            self.assertEqual(graph.formulas()[0]["text"], "x = y + z")

    def test_page_chunks_reassemble_and_derived_nodes_do_not_change_text(self) -> None:
        manifest, pages = _inputs(["# Heading\n\nTest Item: Plain\n"])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            KnowledgeGraphBuilder().build(database, manifest, pages)
            graph = read_graph(database)
            self.assertEqual(graph.reconstruct_page("synthetic-book:page-0"), pages[0].cleaned_text)

    def test_ambiguous_or_incomplete_content_keeps_evidence_but_creates_no_rule(self) -> None:
        manifest, pages = _inputs(["Rule: IF a condition THEN a conclusion\n<table><tr><td>open\n"])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            result = KnowledgeGraphBuilder().build(database, manifest, pages)
            self.assertEqual(result.rule_count, 0)
            graph = read_graph(database)
            self.assertEqual(graph.count_nodes("EvidenceChunk"), 1)
            self.assertEqual(graph.count_nodes("KnowledgeRule"), 0)
            self.assertEqual(graph.table_cells(), ())

    def test_missing_anchor_duplicate_ids_hash_drift_and_chunk_gaps_fail_without_database(self) -> None:
        manifest, pages = _inputs(["Test Item: Plain\nRule: IF a THEN b\n"])
        cases = [
            (manifest, (PageText(pages[0].page_id, "changed", "changed"),), "hash drift"),
            (dict(manifest, chunks=[dict(manifest["chunks"][0], chunk_id="duplicate"), dict(manifest["chunks"][0], chunk_id="duplicate")]), pages, "unique"),
            (dict(manifest, chunks=[dict(manifest["chunks"][0], cleaned_char_start=1)]), pages, "content hash"),
        ]
        for broken_manifest, broken_pages, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                database = Path(directory) / "graph.sqlite"
                with self.assertRaisesRegex(GraphBuildError, message):
                    KnowledgeGraphBuilder().build(database, broken_manifest, broken_pages)
                self.assertFalse(database.exists())

    def test_candidate_cannot_be_approved_without_review_record(self) -> None:
        manifest, pages = _inputs(["Test Item: Plain\nRule: IF a THEN b\n"])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            KnowledgeGraphBuilder().build(database, manifest, pages)
            graph = read_graph(database)
            rule_id = graph.rules()[0]["rule_id"]
            with self.assertRaisesRegex(GraphBuildError, "review record"):
                graph.set_rule_status(rule_id, "approved")
            graph.set_rule_status(rule_id, "approved", review_record="review-1")
            self.assertEqual(read_graph(database).rules()[0]["status"], "approved")

    def test_atomic_build_failure_leaves_no_partial_database(self) -> None:
        manifest, pages = _inputs(["Test Item: Plain\nRule: IF  THEN b\n"])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            with self.assertRaisesRegex(GraphBuildError, "empty rule condition"):
                KnowledgeGraphBuilder().build(database, manifest, pages)
            self.assertFalse(database.exists())

    def test_schema_has_no_domain_specific_special_cases(self) -> None:
        manifest, pages = _inputs(["Test Item: Plain\n"])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            KnowledgeGraphBuilder().build(database, manifest, pages)
            with sqlite3.connect(database) as connection:
                schema = "\n".join(row[0] for row in connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table'"
                ))
            self.assertNotIn("disease", schema.lower())
            self.assertNotIn("patient", schema.lower())


if __name__ == "__main__":
    unittest.main()
