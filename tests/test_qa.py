import hashlib
import json
import socket
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import ProxyHandler, Request, build_opener

from medical_kg_sourceprep.api.qa import MAX_BODY_BYTES, make_server
from medical_kg_sourceprep.evidence.index import QaError, build_evidence_index, query_index


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _chunk_package(
    root: Path,
    page_chunks: tuple[tuple[str, ...], ...] | None = None,
) -> Path:
    package = root / "chunks"
    package.mkdir()
    chunks = []
    pages = []
    chunked_pages = page_chunks or (
        ("样例 alpha 用于索引演示。\n",),
        ("样例 beta 用于第二页演示。\n",),
    )
    for page_index, page_parts in enumerate(chunked_pages):
        text = "".join(page_parts)
        page_id = f"demo:chapter:page:{page_index}"
        page = {
            "page_id": page_id,
            "chapter_page_index": page_index,
            "printed_page_number": page_index + 4,
            "source_pdf_page_number": page_index + 21,
            "cleaned_path": f"pages/cleaned/{page_index:04d}.md",
            "cleaned_sha256": _sha256(text.encode()),
            "source_line_start": page_index * 10 + 1,
            "source_line_end": page_index * 10 + 9,
            "review_status": "verified-boundary",
            "warnings": [],
        }
        pages.append(page)
        offset = 0
        for chunk_index, chunk_text in enumerate(page_parts):
            chunk_id = f"{page_id}:{chunk_index:04d}"
            relative = Path("chunks") / f"{page_index:04d}" / f"{chunk_index:04d}.md"
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(chunk_text, encoding="utf-8", newline="\n")
            chunks.append({
                "chunk_id": chunk_id, "page_id": page_id, "document_id": "demo",
                "chapter_id": "chapter", "chapter_page_index": page_index,
                "printed_page_number": page_index + 4, "source_pdf_page_number": page_index + 21,
                "source_cleaned_path": page["cleaned_path"],
                "source_cleaned_sha256": page["cleaned_sha256"],
                "cleaned_char_start": offset, "cleaned_char_end": offset + len(chunk_text),
                "chunk_path": relative.as_posix(), "chunk_sha256": _sha256(chunk_text.encode()),
                "char_count": len(chunk_text), "warnings": [], "review_status": "verified-boundary",
                "source_page": page,
            })
            offset += len(chunk_text)
    manifest = {
        "schema_version": "evidence-chunk-package/v0.1", "source_manifest_locator": "/read-only/source",
        "source_manifest_sha256": "b" * 64, "document_id": "demo", "chapter_id": "chapter",
        "page_count": len(pages), "chunk_count": len(chunks), "pages": pages, "chunks": chunks,
    }
    (package / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return package


def _candidate_graph(
    path: Path, chunk_id: str, chunk_text: str, node_name: str = "隐含实体"
) -> Path:
    evidence = json.dumps([{
        "chunk_id": chunk_id,
        "chunk_sha256": _sha256(chunk_text.encode()),
        "exact_quote": chunk_text.strip(),
    }])
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
            ("schema_version", "chapter-knowledge-graph/v0.1"),
            ("status", "candidate-only"),
            ("approved", "0"),
        ))
        connection.execute("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", (
            "hidden", "MedicalConcept", node_name, "candidate", "{}", evidence, "[]",
        ))
    return path


class QaTests(unittest.TestCase):
    def test_http_search_uses_candidate_graph_but_returns_index_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _chunk_package(root)
            index = root / "evidence.sqlite"
            build_evidence_index(package, index, "2026-01-01T00:00:00Z")
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            first = manifest["chunks"][0]
            graph = _candidate_graph(
                root / "candidate.sqlite", first["chunk_id"],
                (package / first["chunk_path"]).read_text(encoding="utf-8"),
            )
            server = make_server(index, "127.0.0.1", 0, chunk_package=package, knowledge_graph=graph)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/search",
                    data=json.dumps({"query": "隐含实体"}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
                    result = json.loads(response.read())
                self.assertEqual(result["channels"]["lexical"]["count"], 0)
                self.assertTrue(result["channels"]["graph"]["enabled"])
                self.assertEqual(
                    result["channels"]["graph"]["query_diagnostic"]["status"],
                    "matched",
                )
                self.assertEqual(result["evidence"][0]["chunk_id"], first["chunk_id"])
                self.assertEqual(result["evidence"][0]["retrieval_reason"], "graph_path")
                self.assertEqual(result["evidence"][0]["location_status"], "verified")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

    def test_lan_binding_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            with self.assertRaisesRegex(QaError, "allow_lan"):
                make_server(index, "0.0.0.0", 0)
            server = make_server(index, "0.0.0.0", 0, allow_lan=True)
            server.server_close()

    def test_provenance_binds_sql_offsets_and_reports_distinct_line_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _chunk_package(root)
            index = root / "evidence.sqlite"
            build_evidence_index(package, index, "2026-01-01T00:00:00Z")
            server = make_server(index, "127.0.0.1", 0, chunk_package=package)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/search",
                    data=json.dumps({"query": "alpha"}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
                    evidence = json.loads(response.read())["evidence"]
                first = evidence[0]
                self.assertEqual(first["page_id"], "demo:chapter:page:0")
                self.assertEqual((first["cleaned_char_start"], first["cleaned_char_end"]), (0, len(first["text"])))
                self.assertEqual((first["markdown_line_start"], first["markdown_line_end"]), (1, 1))
                self.assertEqual((first["source_page_line_start"], first["source_page_line_end"]), (1, 9))
                self.assertEqual(first["exact_quote"], first["text"])
                self.assertEqual(first["location_status"], "verified")
                self.assertNotIn(str(package), json.dumps(evidence, ensure_ascii=False))
                for item in evidence:
                    self.assertIn("cleaned_char_start", item)
                    self.assertIn("cleaned_char_end", item)
                    self.assertIn("retrieval_reason", item)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

    def test_cleaned_markdown_lines_follow_multichunk_offsets_and_reset_by_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = "heading\nsecond line\n"
            target = "target third\nfourth"
            package = _chunk_package(root, ((first, target), ("target next page\n",)))
            index = root / "evidence.sqlite"
            build_evidence_index(package, index, "2026-01-01T00:00:00Z")
            server = make_server(index, "127.0.0.1", 0, chunk_package=package)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/search",
                    data=json.dumps({"query": "target third", "top_k": 3}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
                    evidence = json.loads(response.read())["evidence"]
                second_chunk = next(item for item in evidence if item["chunk_id"].endswith(":0001"))
                next_page = next(item for item in evidence if item["page_id"].endswith(":1"))
                self.assertEqual(
                    (second_chunk["cleaned_char_start"], second_chunk["cleaned_char_end"]),
                    (len(first), len(first + target)),
                )
                self.assertEqual(
                    (second_chunk["markdown_line_start"], second_chunk["markdown_line_end"]),
                    (3, 4),
                )
                self.assertEqual(
                    (next_page["markdown_line_start"], next_page["markdown_line_end"]),
                    (1, 1),
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

    def test_provenance_hash_drift_fails_closed_and_unconfigured_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _chunk_package(root)
            index = root / "evidence.sqlite"
            build_evidence_index(package, index, "2026-01-01T00:00:00Z")
            self.assertEqual(query_index(index, "alpha")[0]["location_status"], "unavailable")
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            manifest["chunks"][0]["chunk_sha256"] = "0" * 64
            (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(QaError, "hash"):
                make_server(index, "127.0.0.1", 0, chunk_package=package)

    def test_reconstruction_and_chunk_path_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _chunk_package(root)
            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunk = manifest["chunks"][0]
            replacement = chunk["chunk_path"]
            replacement_path = package / replacement
            changed = replacement_path.read_text(encoding="utf-8").replace("alpha", "gamma")
            replacement_path.write_text(changed, encoding="utf-8", newline="\n")
            chunk["chunk_sha256"] = _sha256(changed.encode())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(QaError, "reconstruction"):
                build_evidence_index(package, root / "bad.sqlite")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _chunk_package(root)
            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["chunks"][0]["chunk_path"] = "../outside.md"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(QaError, "relative path"):
                build_evidence_index(package, root / "bad.sqlite")

    def test_index_location_drift_is_rejected_at_server_start(self) -> None:
        changes = (
            "UPDATE chunks SET cleaned_char_start=1 WHERE chunk_id LIKE '%:0000'",
            "UPDATE pages SET printed_page_number=99 WHERE chapter_page_index=0",
            "UPDATE chunks SET text='tampered' WHERE chunk_id LIKE '%:0000'",
        )
        for statement in changes:
            with self.subTest(statement=statement), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package = _chunk_package(root)
                index = root / "evidence.sqlite"
                build_evidence_index(package, index, "2026-01-01T00:00:00Z")
                with sqlite3.connect(index) as connection:
                    connection.execute(statement)
                with self.assertRaisesRegex(QaError, "location|text"):
                    make_server(index, "127.0.0.1", 0, chunk_package=package)

    def test_build_is_atomic_and_retrieval_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _chunk_package(root)
            index = root / "evidence.sqlite"
            meta = build_evidence_index(package, index, "2026-01-01T00:00:00Z")
            self.assertEqual((meta["document_count"], meta["page_count"], meta["chunk_count"]), (1, 2, 2))
            self.assertEqual(meta["edge_count"], 5)
            results = query_index(index, "样例 alpha", 5)
            self.assertEqual(results[0]["chunk_id"], "demo:chapter:page:0:0000")
            self.assertEqual(results[0]["printed_page_number"], 4)
            self.assertEqual(results[0]["score_components"]["exact_substring_bonus"], 1.0)
            self.assertEqual(results[0]["retrieval_reason"], "term_match+exact_query")
            self.assertEqual(results, query_index(index, "样例 alpha", 5))
            expanded = query_index(index, "alpha", 2)
            self.assertEqual(len(expanded), 2)
            self.assertEqual(expanded[1]["retrieval_reason"], "adjacent_chunk_expansion")
            self.assertEqual(query_index(index, "不存在的词", 5), [])
            with self.assertRaisesRegex(QaError, "already exists"):
                build_evidence_index(package, index)

    def test_invalid_hash_fails_without_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _chunk_package(root)
            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["chunks"][0]["chunk_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(QaError, "hash"):
                build_evidence_index(package, root / "bad.sqlite")
            self.assertFalse((root / "bad.sqlite").exists())

    def test_chunk_page_mapping_drift_fails_without_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _chunk_package(root)
            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["chunks"][0]["printed_page_number"] = 99
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(QaError, "mapping"):
                build_evidence_index(package, root / "bad.sqlite")
            self.assertFalse((root / "bad.sqlite").exists())

    def test_chunk_chapter_and_source_path_must_bind_to_manifest(self) -> None:
        for field, value, expected_error in (
            ("chapter_id", "other-chapter", "chapter"),
            ("source_cleaned_path", "unbound/other.md", "source path"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package = _chunk_package(root)
                manifest_path = package / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["chunks"][0][field] = value
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                index = root / "bad.sqlite"
                with self.assertRaisesRegex(QaError, expected_error):
                    build_evidence_index(package, index)
                self.assertFalse(index.exists())

    def test_optional_provider_fails_closed_without_explicit_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            with patch.dict("os.environ", {"MEDICAL_KG_QA_BASE_URL": "", "MEDICAL_KG_QA_API_KEY": "", "MEDICAL_KG_QA_MODEL": ""}):
                with self.assertRaisesRegex(QaError, "MEDICAL_KG_QA"):
                    make_server(index, "127.0.0.1", 0, "openai-compatible")

    def test_optional_provider_uses_local_stub_and_requires_citation(self) -> None:
        class Provider(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                self.server.authorization = self.headers.get("Authorization")  # type: ignore[attr-defined]
                size = int(self.headers["Content-Length"])
                self.server.request_body = json.loads(self.rfile.read(size))  # type: ignore[attr-defined]
                body = json.dumps({"choices": [{"message": {"content": "来自证据的回答 [1]"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
            provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
            provider_thread.start()
            environment = {
                "MEDICAL_KG_QA_BASE_URL": f"http://127.0.0.1:{provider.server_port}",
                "MEDICAL_KG_QA_API_KEY": "synthetic-test-key",
                "MEDICAL_KG_QA_MODEL": "synthetic-model",
                "MEDICAL_KG_QA_TIMEOUT": "3",
            }
            server = None
            try:
                with patch.dict("os.environ", environment, clear=True):
                    server = make_server(index, "127.0.0.1", 0, "openai-compatible")
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    request = Request(
                        f"http://127.0.0.1:{server.server_port}/api/answer",
                        data=json.dumps({"query": "样例"}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
                        answer = json.loads(response.read())
                self.assertEqual(answer["mode"], "openai-compatible")
                self.assertIn("[1]", answer["answer"])
                self.assertEqual(provider.authorization, "Bearer synthetic-test-key")  # type: ignore[attr-defined]
                self.assertIn("Evidence:", provider.request_body["messages"][0]["content"])  # type: ignore[attr-defined]
            finally:
                if server is not None:
                    server.shutdown()
                    server.server_close()
                provider.shutdown()
                provider.server_close()
                provider_thread.join(timeout=3)

    def test_http_answer_and_static_ui_are_grounded_and_hardened(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            server = make_server(index, "127.0.0.1", 0, "extractive")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            opener = build_opener(ProxyHandler({}))
            try:
                with opener.open(base + "/", timeout=3) as response:
                    html = response.read().decode()
                    self.assertIn('role="main"', html)
                    self.assertIn("/assets/app.js", html)
                    self.assertNotIn("https://", html)
                    self.assertIn("Content-Security-Policy", response.headers)
                request = Request(base + "/api/answer", data=json.dumps({"query": "样例"}).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with opener.open(request, timeout=3) as response:
                    answer = json.loads(response.read())
                self.assertEqual(answer["mode"], "extractive")
                self.assertTrue(answer["citations"])
                self.assertIn("[1]", answer["answer"])
                self.assertEqual(answer["citations"][0]["chunk_id"], answer["evidence"][0]["chunk_id"])
                self.assertIn("score_components", answer["evidence"][0])
                self.assertIn('id="provenance"', html)
                with opener.open(base + "/assets/app.css", timeout=3) as response:
                    css = response.read().decode()
                self.assertIn("*,*::before,*::after{box-sizing:border-box}", css)
                self.assertIn("input{min-width:0;width:100%;max-width:100%", css)
                with self.assertRaises(HTTPError) as bad:
                    opener.open(Request(base + "/api/search", data=b"not-json", method="POST"), timeout=3)
                self.assertEqual(bad.exception.code, 400)
                with socket.create_connection(("127.0.0.1", server.server_port), timeout=3) as raw:
                    raw.sendall(
                        b"POST /api/search HTTP/1.1\r\n"
                        b"Host: 127.0.0.1\r\n"
                        b"Content-Length: nope\r\n\r\n"
                    )
                    raw.shutdown(socket.SHUT_WR)
                    response = raw.makefile("rb").read().decode("utf-8")
                headers, body = response.split("\r\n\r\n", 1)
                self.assertIn(" 400 ", headers)
                self.assertEqual(json.loads(body)["error"], "invalid_request")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
