import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.provenance.book_sources import build_book_manifest
from medical_kg_sourceprep.graph.graph_retrieval import graph_retrieve
from medical_kg_sourceprep.graph.knowledge_graph import KnowledgeGraphBuilder, PageText
from medical_kg_sourceprep.graph.semantic_graph import (
    SEMANTIC_RELATIONS, ReviewRecord, SemanticGraphBuilder, SemanticGraphError, SemanticRecord, SemanticRelation,
)


def source(texts):
    pages, chunks, values = [], [], []
    for number, text in enumerate(texts):
        page_id = f"book:page-{number}"
        digest = hashlib.sha256(text.encode()).hexdigest()
        pages.append({"page_id": page_id, "chapter_page_index": number, "raw_path": "raw", "cleaned_path": "clean",
                      "raw_sha256": digest, "cleaned_sha256": digest, "source_line_start": 1,
                      "source_line_end": text.count("\n") + 1, "printed_page_number": number + 1,
                      "source_pdf_page_number": number + 1, "review_status": "unreviewed"})
        chunks.append({"chunk_id": f"book:chunk-{number}", "page_id": page_id, "cleaned_char_start": 0,
                       "cleaned_char_end": len(text), "chunk_sha256": digest})
        values.append(PageText(page_id, text, text))
    return build_book_manifest(book={"book_id": "book", "title": "Synthetic", "edition": "1"},
        pdf={"pdf_id": "book:pdf", "locator": "x", "sha256": "a" * 64},
        markdown={"markdown_id": "book:md", "locator": "x", "sha256": "b" * 64}, pages=pages, chunks=chunks), values


def evidence_index(path, texts):
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL, chunk_sha256 TEXT NOT NULL);")
        db.executemany("INSERT INTO chunks VALUES (?, ?, ?)", [(f"book:chunk-{number}", text, hashlib.sha256(text.encode()).hexdigest()) for number, text in enumerate(texts)])
    return path


class SemanticGraphTests(unittest.TestCase):
    def build(self, root, texts, records=(), relations=()):
        manifest, pages = source(texts)
        base = root / "knowledge.sqlite"
        KnowledgeGraphBuilder().build(base, manifest, pages)
        return SemanticGraphBuilder().build(root / "semantic.sqlite", base, manifest, records, relations)

    def test_real_shape_projection_without_test_item_label_mirrors_main_graph(self):
        text = "# Volume\n1. Generic signal\n【参考区间】\nwithin band\n【异常结果解读】condition-like prose\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.build(root, [text])
            self.assertEqual(result.status_counts, {"candidate": 4})
            with sqlite3.connect(result.database_path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM semantic_records WHERE entity_type = 'TestItem'").fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM semantic_records WHERE entity_type = 'ReferenceRange'").fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM nodes WHERE node_type = 'SourceLocator'").fetchone()[0], 2)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM edges WHERE edge_kind = 'semantic'").fetchone()[0], 3)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM edges WHERE relation = 'SOURCE_LOCATOR_TARGETS_CHUNK' AND edge_kind = 'structural'").fetchone()[0], 2)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM anchors WHERE role = 'semantic_source'").fetchone()[0], 4)
            hits = graph_retrieve(result.database_path, evidence_index(root / "evidence.sqlite", [text]), "candidate")
            self.assertEqual(hits[0].chunk_id, "book:chunk-0")
            self.assertIn("SOURCE_LOCATOR_TARGETS_CHUNK", hits[0].path_relations)

    def test_records_with_the_same_source_span_share_one_locator(self):
        text = "shared evidence\n"
        item = SemanticRecord(
            "item", "TestItem", "candidate", "shared", "book:chunk-0", 0, 6
        )
        rule = SemanticRecord(
            "rule", "InterpretationRule", "candidate", "shared", "book:chunk-0", 0, 6,
            semantic_type="DEFINES_AS", subject_logic="SINGLE",
        )
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory), [text], [item, rule])
            with sqlite3.connect(result.database_path) as db:
                self.assertEqual(
                    db.execute(
                        "SELECT COUNT(*) FROM semantic_records WHERE entity_type = 'SourceLocator'"
                    ).fetchone()[0],
                    1,
                )

    def test_records_with_same_start_and_different_end_use_distinct_locators(self):
        text = "shared evidence\n"
        item = SemanticRecord(
            "item", "TestItem", "candidate", "shared", "book:chunk-0", 0, 6
        )
        rule = SemanticRecord(
            "rule", "InterpretationRule", "candidate", "shared evidence",
            "book:chunk-0", 0, 15, semantic_type="DEFINES_AS", subject_logic="SINGLE",
        )
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory), [text], [item, rule])
            with sqlite3.connect(result.database_path) as db:
                self.assertEqual(
                    db.execute(
                        "SELECT COUNT(*) FROM semantic_records WHERE entity_type = 'SourceLocator'"
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    db.execute(
                        "SELECT COUNT(*) FROM semantic_edges WHERE relation IN "
                        "('ITEM_SUPPORTED_BY', 'RULE_SUPPORTED_BY')"
                    ).fetchone()[0],
                    2,
                )

    def test_section_state_handles_tables_cross_chunk_and_resets(self):
        texts = [
            "# First\n1. A\n【参考区间】\n",
            "<table><tr><td>table range</td></tr></table>\n2. B\n【异常结果解读】prose\n【参考区间】next range\n",
            "# Second\n【参考区间】orphan\n3. C\nno label\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory), texts)
            with sqlite3.connect(result.database_path) as db:
                rows = db.execute("""
                    SELECT records.entity_type, records.text, locators.chunk_id
                    FROM semantic_records AS records
                    JOIN semantic_source_locators AS locators ON locators.record_id = records.record_id
                    WHERE records.entity_type IN ('TestItem', 'ReferenceRange')
                    ORDER BY records.entity_type, records.text
                """).fetchall()
            self.assertEqual(rows, [("ReferenceRange", "<table><tr><td>table range</td></tr></table>", "book:chunk-1"), ("ReferenceRange", "next range", "book:chunk-1"), ("TestItem", "A", "book:chunk-0"), ("TestItem", "B", "book:chunk-1")])

    def test_abnormal_prose_is_evidence_only(self):
        text = "# C\n1. Neutral\n【参考区间】value\n【异常结果解读】disease-shaped causal wording\n"
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory), [text])
            with sqlite3.connect(result.database_path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM semantic_records WHERE entity_type IN ('InterpretationRule', 'MedicalConcept')").fetchone()[0], 0)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM semantic_records WHERE status = 'approved'").fetchone()[0], 0)

    def test_fixed_relations_review_gate_and_atomic_failure(self):
        text = "rule item method range concept people locator\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, pages = source([text])
            base = root / "knowledge.sqlite"
            KnowledgeGraphBuilder().build(base, manifest, pages)
            record = SemanticRecord("rule", "InterpretationRule", "approved", "rule", "book:chunk-0", 0, 4,
                                    semantic_type="DEFINES_AS", subject_logic="SINGLE")
            with self.assertRaisesRegex(SemanticGraphError, "review"):
                SemanticGraphBuilder().build(root / "bad.sqlite", base, manifest, [record])
            self.assertFalse((root / "bad.sqlite").exists())
            bad_item = SemanticRecord("item", "TestItem", "candidate", "item", "book:chunk-0", 5, 9)
            bad_range = SemanticRecord("range", "ReferenceRange", "candidate", "range", "book:chunk-0", 17, 22)
            with self.assertRaisesRegex(SemanticGraphError, "relation"):
                SemanticGraphBuilder().build(root / "bad2.sqlite", base, manifest, [bad_item, bad_range], [("range", "ITEM_SUPPORTED_BY", "item")])

    def test_content_hash_binds_records_relations_reviews_and_is_order_stable(self):
        text = "first second\n"
        first = SemanticRecord("first", "TestItem", "candidate", "first", "book:chunk-0", 0, 5)
        second = SemanticRecord("second", "TestMethod", "reviewed", "second", "book:chunk-0", 6, 12)
        reviewed = SemanticRecord("first", "TestItem", "approved", "first", "book:chunk-0", 0, 5, review=ReviewRecord("r", "t", "why", "v"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stable_one = self.build(root / "one", [text], [first, second], [("first", "ITEM_MEASURED_BY_METHOD", "second")])
            stable_two = self.build(root / "two", [text], [second, first], [("first", "ITEM_MEASURED_BY_METHOD", "second")])
            changed_review = self.build(root / "three", [text], [reviewed, second], [("first", "ITEM_MEASURED_BY_METHOD", "second")])
            changed_relation = self.build(root / "four", [text], [first, second])
            self.assertEqual(stable_one.package_hash, stable_two.package_hash)
            self.assertNotEqual(stable_one.package_hash, changed_review.package_hash)
            self.assertNotEqual(stable_one.package_hash, changed_relation.package_hash)

    def test_allowlist_contains_exactly_ten_relations(self):
        self.assertEqual(len(SEMANTIC_RELATIONS), 10)

    def test_rich_relation_persists_independent_replayable_evidence(self):
        text = "item采用method"
        item = SemanticRecord("item", "TestItem", "candidate", "item", "book:chunk-0", 0, 4, candidate_key="item-key")
        method = SemanticRecord("method", "TestMethod", "candidate", "method", "book:chunk-0", 6, 12, candidate_key="method-key")
        relation = SemanticRelation("item", "ITEM_MEASURED_BY_METHOD", "method", "book:chunk-0", text, "采用")
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory), [text], [item, method], [relation])
            with sqlite3.connect(result.database_path) as db:
                row = db.execute("SELECT source_quote, relation_cue, char_start, char_end FROM semantic_relation_evidence").fetchone()
                self.assertEqual(row, (text, "采用", 0, len(text)))
                self.assertEqual(db.execute("SELECT candidate_key FROM semantic_records WHERE record_id = 'item'").fetchone()[0], "item-key")

    def test_duplicate_triple_merges_multiple_evidence_rows_and_preserves_origin(self):
        texts = ["item采用method", "item采用method"]
        item = SemanticRecord("item", "TestItem", "candidate", "item", "book:chunk-0", 0, 4)
        method = SemanticRecord("method", "TestMethod", "candidate", "method", "book:chunk-0", 6, 12)
        first = SemanticRelation("item", "ITEM_MEASURED_BY_METHOD", "method", "book:chunk-0", texts[0], "采用", origin="derived")
        second = SemanticRelation("item", "ITEM_MEASURED_BY_METHOD", "method", "book:chunk-1", texts[1], "采用", origin="model")
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory), texts, [item, method], [first, second])
            with sqlite3.connect(result.database_path) as db:
                rows = db.execute("SELECT evidence_role, source_chunk_id FROM semantic_relation_evidence ORDER BY source_chunk_id").fetchall()
                self.assertEqual(rows, [("relation", "book:chunk-0"), ("relation", "book:chunk-1")])
                self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_all_entity_relation_and_status_contracts_are_accepted(self):
        text = "item method range rule concept people locator text\n"
        names = text.split()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            types = ("TestItem", "TestMethod", "ReferenceRange", "InterpretationRule", "MedicalConcept", "Population", "SourceLocator")
            statuses = ("candidate", "reviewed", "rejected", "candidate", "reviewed", "rejected", "candidate")
            cursor = 0
            for number, (name, entity_type, status) in enumerate(zip(names, types, statuses)):
                cursor = text.index(name, cursor)
                records.append(SemanticRecord(str(number), entity_type, status, name, "book:chunk-0", cursor, cursor + len(name),
                    semantic_type="DEFINES_AS" if entity_type == "InterpretationRule" else None,
                    subject_logic="SINGLE" if entity_type == "InterpretationRule" else None))
                cursor += len(name)
            relations = [("0", "ITEM_MEASURED_BY_METHOD", "1"), ("0", "ITEM_HAS_REFERENCE_RANGE", "2"),
                ("2", "RANGE_APPLIES_TO_POPULATION", "5"), ("0", "ITEM_SUPPORTED_BY", "6"),
                ("2", "RANGE_SUPPORTED_BY", "6"), ("3", "RULE_HAS_SUBJECT", "0"),
                ("3", "RULE_HAS_CONCLUSION", "4"), ("3", "RULE_APPLIES_TO_POPULATION", "5"),
                ("3", "RULE_REQUIRES_METHOD", "1"), ("3", "RULE_SUPPORTED_BY", "6")]
            result = self.build(root, [text], records, relations)
            self.assertEqual(result.edge_count, 13)
            self.assertEqual(result.status_counts, {"candidate": 6, "rejected": 2, "reviewed": 2})
            with sqlite3.connect(result.database_path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM edges WHERE edge_kind = 'semantic'").fetchone()[0], 13)


if __name__ == "__main__":
    unittest.main()
