import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.graph_retrieval import (
    GraphRetrievalError,
    graph_query_diagnostic,
    graph_reasoning_paths,
    graph_retrieve,
)


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


def candidate_graph_db(path: Path, quote: str, *, chunk_id: str = "evidence",
                       schema_version: str = "chapter-knowledge-graph/v0.1") -> Path:
    node_evidence = json.dumps([{"chunk_id": chunk_id, "chunk_sha256": hashlib.sha256(quote.encode()).hexdigest(), "exact_quote": quote}])
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE nodes (node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, name TEXT NOT NULL,
                status TEXT NOT NULL, properties_json TEXT NOT NULL, evidence_json TEXT NOT NULL, origins_json TEXT NOT NULL);
            CREATE TABLE edges (triple_id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, predicate TEXT NOT NULL,
                object_id TEXT NOT NULL, layer TEXT NOT NULL, status TEXT NOT NULL, properties_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL, origins_json TEXT NOT NULL);
        """)
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", (
            ("schema_version", schema_version), ("status", "candidate-only"), ("approved", "0"),
        ))
        connection.execute("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", (
            "hb", "TestItem", "血红蛋白", "candidate", json.dumps({"aliases": ["Hb"]}), node_evidence, "[]",
        ))
    return path


def reasoning_graph_db(path: Path, quote: str, *, invalid_subject: bool = False) -> Path:
    evidence = json.dumps([{
        "chunk_id": "evidence",
        "chunk_sha256": hashlib.sha256(quote.encode()).hexdigest(),
        "exact_quote": quote,
    }])
    empty = "[]"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE nodes (node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, name TEXT NOT NULL,
                status TEXT NOT NULL, properties_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
                origins_json TEXT NOT NULL);
            CREATE TABLE edges (triple_id TEXT PRIMARY KEY, subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL, object_id TEXT NOT NULL, layer TEXT NOT NULL,
                status TEXT NOT NULL, properties_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
                origins_json TEXT NOT NULL);
        """)
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", (
            ("schema_version", "chapter-knowledge-graph/v0.2"),
            ("status", "candidate-only"), ("approved", "0"),
        ))
        rule_properties = json.dumps({
            "rule_id": "anemia-index-rule",
            "applicability": {"required_inputs": ["MCV", "MCH"]},
        })
        nodes = [
            ("mcv", "TestItem", "平均红细胞容积", "candidate", json.dumps({"aliases": ["MCV"]}), empty, empty),
            ("mch", "TestItem", "平均红细胞血红蛋白含量", "candidate", json.dumps({"aliases": ["MCH"]}), empty, empty),
            ("hgb", "TestItem", "血红蛋白", "candidate", json.dumps({"aliases": ["HGB"]}), empty, empty),
            ("rule", "InterpretationRule", "贫血红细胞指数形态分类", "candidate", rule_properties, evidence, empty),
            ("result", "MedicalConcept", "贫血红细胞指数形态", "candidate", "{}", empty, empty),
        ]
        connection.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", nodes)
        first_target = "hgb" if invalid_subject else "mcv"
        edges = [
            ("s1", "rule", "RULE_HAS_SUBJECT", first_target, "rule", "candidate", json.dumps({"input_terms": ["MCV"]}), evidence, empty),
            ("s2", "rule", "RULE_HAS_SUBJECT", "mch", "rule", "candidate", json.dumps({"input_terms": ["MCH"]}), evidence, empty),
            ("c1", "rule", "RULE_HAS_CONCLUSION", "result", "rule", "candidate", "{}", evidence, empty),
        ]
        connection.executemany("INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", edges)
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

    def test_candidate_graph_projects_direct_and_rechunked_exact_quote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quote = "血红蛋白参考区间"
            direct = evidence_db(root / "direct.sqlite", (("evidence", quote),))
            rechunked = evidence_db(root / "rechunked.sqlite", (("full-book", "前文 " + quote + " 后文"),))
            graph = candidate_graph_db(root / "candidate.sqlite", quote)
            direct_hit = graph_retrieve(graph, direct, "Hb")[0]
            rechunked_hit = graph_retrieve(graph, rechunked, "血红蛋白")[0]
            self.assertEqual(direct_hit.chunk_id, "evidence")
            self.assertEqual(rechunked_hit.chunk_id, "full-book")
            self.assertEqual(rechunked_hit.graph_status, "candidate-only")
            self.assertEqual(rechunked_hit.matched_node_names, ("血红蛋白",))

    def test_candidate_graph_v02_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quote = "血红蛋白参考区间"
            evidence = evidence_db(root / "evidence.sqlite", (("evidence", quote),))
            graph = candidate_graph_db(root / "candidate.sqlite", quote,
                                       schema_version="chapter-knowledge-graph/v0.2")
            self.assertEqual(graph_retrieve(graph, evidence, "血红蛋白")[0].chunk_id, "evidence")

    def test_candidate_query_normalizes_book_and_report_term_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quote = "平均红细胞容积(MCV)"
            evidence = evidence_db(root / "evidence.sqlite", (("evidence", quote),))
            graph = candidate_graph_db(root / "candidate.sqlite", quote)
            with sqlite3.connect(graph) as connection:
                connection.execute(
                    "UPDATE nodes SET name=?, properties_json=? WHERE node_id='hb'",
                    ("平均红细胞容积", json.dumps({"aliases": ["MCV"]})),
                )
            hit = graph_retrieve(graph, evidence, "平均红细胞体积")[0]
            self.assertEqual(hit.matched_node_names, ("平均红细胞容积",))
            self.assertEqual(hit.match_mode, "normalized_variant")
            diagnostic = graph_query_diagnostic(graph, "NEUT#")
            self.assertEqual(diagnostic["status"], "alias_missing")

            with sqlite3.connect(graph) as connection:
                connection.execute(
                    "UPDATE nodes SET name=?, properties_json=? WHERE node_id='hb'",
                    ("平均红细胞血红蛋白含量", json.dumps({"aliases": ["MCH"]})),
                )
            mch = graph_query_diagnostic(graph, "平均红细胞血红蛋白量")
            self.assertEqual(mch["status"], "matched")
            self.assertEqual(mch["match_mode"], "normalized_variant")

    def test_candidate_hit_preserves_directed_triples_and_path_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quote = "MCV和MCH用于贫血形态分类"
            evidence = evidence_db(root / "evidence.sqlite", (("evidence", quote),))
            graph = reasoning_graph_db(root / "candidate.sqlite", quote)
            hit = next(item for item in graph_retrieve(graph, evidence, "MCV", top_k=10)
                       if item.path_relations == ("RULE_HAS_SUBJECT",))
            self.assertEqual(hit.path_node_names, ("平均红细胞容积", "贫血红细胞指数形态分类"))
            self.assertEqual(hit.path_triples[0]["subject_name"], "贫血红细胞指数形态分类")
            self.assertEqual(hit.path_triples[0]["predicate"], "RULE_HAS_SUBJECT")
            self.assertEqual(hit.path_triples[0]["object_name"], "平均红细胞容积")
            self.assertEqual(hit.path_triples[0]["traversal_direction"], "reverse")

    def test_multi_metric_reasoning_rejects_subject_endpoint_mismatch(self) -> None:
        observations = (
            {"metric_id": "mcv", "terms": ["平均红细胞体积", "MCV"], "computed_flag": "low"},
            {"metric_id": "mch", "terms": ["平均红细胞血红蛋白含量", "MCH"], "computed_flag": "low"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quote = "MCV和MCH用于贫血形态分类"
            evidence = evidence_db(root / "evidence.sqlite", (("evidence", quote),))
            graph = reasoning_graph_db(root / "valid.sqlite", quote)
            valid = graph_reasoning_paths(graph, evidence, observations)
            self.assertEqual(len(valid.paths), 1)
            self.assertEqual(valid.paths[0]["matched_metric_ids"], ["mch", "mcv"])
            self.assertEqual(valid.paths[0]["status"], "candidate-complete")
            self.assertEqual(len(valid.paths[0]["triples"]), 3)

            invalid_graph = reasoning_graph_db(root / "invalid.sqlite", quote, invalid_subject=True)
            invalid = graph_reasoning_paths(invalid_graph, evidence, observations)
            self.assertEqual(invalid.paths, ())
            self.assertEqual(invalid.rejections[0]["reason"], "subject_endpoint_mismatch")
