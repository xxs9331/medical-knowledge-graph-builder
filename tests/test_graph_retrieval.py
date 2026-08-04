import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.graph_retrieval import GraphRetrievalError, graph_retrieve


def evidence_db(path: Path, rows: tuple[tuple[str, str], ...]) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL, chunk_sha256 TEXT NOT NULL);
        """)
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", (("schema_version", "evidence-index/v0.1"), ("chunk_manifest_sha256", "a" * 64)))
        connection.executemany("INSERT INTO chunks VALUES (?, ?, ?)", [(item_id, text, hashlib.sha256(text.encode()).hexdigest()) for item_id, text in rows])
    return path


def graph_db(path: Path, text: str, *, chunk_id: str = "chunk") -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE nodes (node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE edges (edge_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL, relation TEXT NOT NULL, edge_kind TEXT NOT NULL);
            CREATE TABLE anchors (anchor_id TEXT PRIMARY KEY, node_id TEXT NOT NULL, role TEXT NOT NULL, anchor_json TEXT NOT NULL);
            CREATE TABLE chunk_text (chunk_id TEXT PRIMARY KEY, page_id TEXT NOT NULL, start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL, content TEXT NOT NULL);
        """)
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("schema_version", "knowledge-graph/v0.2"))
        connection.executemany("INSERT INTO nodes VALUES (?, ?, ?)", [("term", "Explanation", json.dumps({"text": "alpha route"})), (chunk_id, "EvidenceChunk", "{}")])
        connection.execute("INSERT INTO edges VALUES (?, ?, ?, ?, ?)", ("e1", "term", chunk_id, "SUPPORTS", "semantic"))
        connection.execute("INSERT INTO anchors VALUES (?, ?, ?, ?)", ("anchor-1", "term", "support", "{}"))
        connection.execute("INSERT INTO chunk_text VALUES (?, ?, ?, ?, ?)", (chunk_id, "page", 0, len(text), text))
    return path


class GraphRetrievalTests(unittest.TestCase):
    def test_path_projects_only_to_hash_bound_evidence_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = evidence_db(root / "evidence.sqlite", (("evidence", "linked evidence"),))
            graph = graph_db(root / "graph.sqlite", "linked evidence", chunk_id="evidence")
            hit = graph_retrieve(graph, evidence, "alpha")[0]
            self.assertEqual(hit.chunk_id, "evidence")
            self.assertEqual(hit.anchor_ids, ("anchor-1",))
            self.assertEqual(hit.path_relations, ("SUPPORTS",))

    def test_dangling_or_invalid_graph_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = evidence_db(root / "evidence.sqlite", (("evidence", "different"),))
            graph = graph_db(root / "graph.sqlite", "alpha evidence")
            self.assertEqual(graph_retrieve(graph, evidence, "alpha"), ())
            with sqlite3.connect(graph) as connection:
                connection.execute("UPDATE metadata SET value = 'bad' WHERE key = 'schema_version'")
            with self.assertRaises(GraphRetrievalError):
                graph_retrieve(graph, evidence, "alpha")

    def test_identical_content_keeps_chunk_id_and_hash_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = evidence_db(root / "evidence.sqlite", (("first", "same text"), ("second", "same text")))
            first_graph = graph_db(root / "first.sqlite", "same text", chunk_id="first")
            second_graph = graph_db(root / "second.sqlite", "same text", chunk_id="second")
            self.assertEqual(graph_retrieve(first_graph, evidence, "alpha")[0].chunk_id, "first")
            self.assertEqual(graph_retrieve(second_graph, evidence, "alpha")[0].chunk_id, "second")
            with sqlite3.connect(second_graph) as connection:
                connection.execute("UPDATE chunk_text SET content = 'drifted text'")
            with self.assertRaises(GraphRetrievalError):
                graph_retrieve(second_graph, evidence, "alpha")
