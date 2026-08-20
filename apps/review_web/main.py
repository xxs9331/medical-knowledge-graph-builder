"""本地候选图人工审查工作台。

默认读取一次实验目录中的候选图、问题报告和运行清单；人工操作只追加写入
review-decisions.jsonl，不修改候选图、参考图或金标。该服务只适合本机审查。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import uuid
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parents[1]
DEFAULT_RUN = PROJECT_ROOT / (
    "runtime/experiments/chapter01-qwen-flash-relation-v0.9/"
    "full-chapter-evidence-gated"
)
DECISIONS_FILENAME = "review-decisions.jsonl"
_write_lock = threading.Lock()


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return fallback


def _load_bundle(run_root: Path) -> dict[str, Any]:
    graph = _read_json(run_root / "graph.json", {"nodes": [], "relationships": []})
    issues = _read_json(run_root / "evaluation-issues.json", {"issues": []})
    manifest = _read_json(run_root / "run-manifest.json", {})
    score = _read_json(run_root / "evidence-v11-score.json", {})
    decisions_path = run_root / DECISIONS_FILENAME
    decisions: list[dict[str, Any]] = []
    if decisions_path.is_file():
        for line in decisions_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                decisions.append(value)
    return {
        "graph": graph if isinstance(graph, dict) else {"nodes": [], "relationships": []},
        "issues": issues if isinstance(issues, dict) else {"issues": []},
        "manifest": manifest if isinstance(manifest, dict) else {},
        "score": score if isinstance(score, dict) else {},
        "decisions": decisions,
    }


def _fact_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    graph = bundle["graph"]
    nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
    by_key = {str(item.get("candidate_key")): item for item in nodes}
    rows: list[dict[str, Any]] = []
    for item in nodes:
        rows.append({
            "fact_id": f"node:{item.get('candidate_key', '')}",
            "kind": "Entity",
            "type": item.get("entity_type"),
            "candidate": item.get("mention") or item.get("canonical_name_candidate"),
            "chunk_id": (item.get("source_ref") or {}).get("chunk_id"),
            "evidence": item.get("source_ref") or {},
            "raw": item,
        })
    for item in graph.get("relationships", []):
        if not isinstance(item, dict):
            continue
        source = by_key.get(str(item.get("source_candidate_key")), {})
        target = by_key.get(str(item.get("target_candidate_key")), {})
        source_name = source.get("mention") or source.get("canonical_name_candidate")
        target_name = target.get("mention") or target.get("canonical_name_candidate")
        rows.append({
            "fact_id": f"relationship:{item.get('candidate_key', '')}",
            "kind": "Relation",
            "type": item.get("relation_type"),
            "candidate": f"{source_name}  --{item.get('relation_type')}-->  {target_name}",
            "chunk_id": (item.get("source_ref") or {}).get("chunk_id"),
            "evidence": item.get("source_ref") or {},
            "raw": item,
        })
    return rows


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "medical-kg-review/0.1"

    @property
    def run_root(self) -> Path:
        return self.server.run_root  # type: ignore[attr-defined]

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file() or APP_ROOT not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        bundle = _load_bundle(self.run_root)
        if parsed.path == "/":
            self._send_file(APP_ROOT / "index.html")
            return
        if parsed.path.startswith("/static/"):
            self._send_file(APP_ROOT / parsed.path.removeprefix("/static/"))
            return
        if parsed.path == "/api/summary":
            score = bundle["score"]
            self._send_json({
                "run": bundle["manifest"],
                "score": score.get("micro", {}),
                "by_relation_type": score.get("by_relation_type", {}),
                "issue_count": bundle["issues"].get("issue_count", 0),
                "issue_counts": bundle["issues"].get("counts_by_severity", {}),
                "decision_count": len(bundle["decisions"]),
            })
            return
        if parsed.path == "/api/facts":
            query = parse_qs(parsed.query)
            rows = _fact_rows(bundle)
            kind = query.get("kind", [""])[0]
            relation_type = query.get("type", [""])[0]
            search = query.get("q", [""])[0].strip().lower()
            if kind:
                rows = [row for row in rows if row["kind"].lower() == kind.lower()]
            if relation_type:
                rows = [row for row in rows if row["type"] == relation_type]
            if search:
                rows = [row for row in rows if search in str(row["candidate"]).lower()]
            self._send_json({"facts": rows, "total": len(rows)})
            return
        if parsed.path == "/api/issues":
            self._send_json(bundle["issues"])
            return
        if parsed.path == "/api/decisions":
            self._send_json({"decisions": bundle["decisions"]})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/decisions":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(size))
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "invalid_json"}, HTTPStatus.BAD_REQUEST)
            return
        if not isinstance(value, dict) or value.get("decision") not in {
            "valid_issue", "invalid_issue", "uncertain",
        }:
            self._send_json({"error": "invalid_review_decision"}, HTTPStatus.BAD_REQUEST)
            return
        decision = {
            "review_decision_id": f"review:{uuid.uuid4().hex}",
            "run_id": _load_bundle(self.run_root)["manifest"].get("run_id", str(self.run_root)),
            "review_item_id": str(value.get("review_item_id", "")),
            "decision": value["decision"],
            "attribution": value.get("attribution"),
            "comment": str(value.get("comment", ""))[:2000],
            "promote_to_gold": bool(value.get("promote_to_gold", False)),
            "reviewer_id": str(value.get("reviewer_id", "local-reviewer")),
            "created_at": datetime.now(UTC).isoformat(),
        }
        path = self.run_root / DECISIONS_FILENAME
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
        self._send_json(decision, HTTPStatus.CREATED)

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本地医学知识图谱人工审查页面")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    server.run_root = args.run_root.resolve()  # type: ignore[attr-defined]
    print(f"审查页面: http://{args.host}:{args.port}/")
    print(f"数据目录: {server.run_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
