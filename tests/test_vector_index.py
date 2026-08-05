import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.evidence.vector_index import VectorIndexError, build_vector_index, query_vector_index


def evidence_db(path: Path, rows: tuple[tuple[str, str], ...]) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL, chunk_sha256 TEXT NOT NULL);
        """)
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", (("schema_version", "evidence-index/v0.1"), ("chunk_manifest_sha256", "a" * 64)))
        connection.executemany("INSERT INTO chunks VALUES (?, ?, ?)", [(item_id, text, hashlib.sha256(text.encode()).hexdigest()) for item_id, text in rows])
    return path


class VectorIndexTests(unittest.TestCase):
    def test_build_is_deterministic_and_queries_unicode_ngrams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = evidence_db(root / "evidence.sqlite", (("a", "Alpha signal"), ("b", "乙型信号")))
            first = build_vector_index(source, root / "first.sqlite")
            second = build_vector_index(source, root / "second.sqlite")
            self.assertEqual(first, second)
            self.assertEqual(query_vector_index(root / "first.sqlite", source, "Ａlpha")[0].chunk_id, "a")
            self.assertEqual(query_vector_index(root / "first.sqlite", source, "乙型")[0].chunk_id, "b")

    def test_binding_drift_and_invalid_queries_fail_closed_without_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = evidence_db(root / "evidence.sqlite", (("a", "alpha"),))
            index = root / "vectors.sqlite"
            build_vector_index(source, index)
            with sqlite3.connect(source) as connection:
                connection.execute("UPDATE chunks SET text = 'changed'")
            with self.assertRaises(VectorIndexError):
                query_vector_index(index, source, "alpha")
            with self.assertRaises(VectorIndexError):
                build_vector_index(source, root / "bad.sqlite", ngram_min=4, ngram_max=2)
            self.assertFalse((root / "bad.sqlite").exists())
            self.assertEqual(query_vector_index(index, source, "", top_k=1), ())

    def test_query_uses_the_same_idf_weights_as_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = evidence_db(root / "evidence.sqlite", (("common", "aaaa"), ("rare", "aabb")))
            index = root / "vectors.sqlite"
            build_vector_index(source, index, ngram_min=2, ngram_max=2)
            hits = query_vector_index(index, source, "aaaabb", threshold=0.0)
            self.assertEqual([hit.chunk_id for hit in hits], ["rare", "common"])

    def test_corrupt_vector_payloads_fail_closed(self) -> None:
        cases = (
            "not json", "[]", '{"0": true}', '{"0": "1"}', '{"0": NaN}',
            '{"0": Infinity}', '{"0": -0.2}', '{"9": 1.0}', '{"00": 1.0}',
            '{"0": 0.6, "0": 0.8}', '{"0": 0.0}', '{"0": 0.5}',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = evidence_db(root / "evidence.sqlite", (("a", "alpha"),))
            index = root / "vectors.sqlite"
            build_vector_index(source, index)
            for payload in cases:
                with self.subTest(payload=payload):
                    with sqlite3.connect(index) as connection:
                        connection.execute("UPDATE vectors SET vector_json = ? WHERE chunk_id = 'a'", (payload,))
                        connection.commit()
                    with self.assertRaises(VectorIndexError):
                        query_vector_index(index, source, "alpha")
                    with sqlite3.connect(index) as connection:
                        connection.execute(
                            "UPDATE vectors SET vector_json = ? WHERE chunk_id = 'a'",
                            (json.dumps({"0": 1.0}),),
                        )
                        connection.commit()
