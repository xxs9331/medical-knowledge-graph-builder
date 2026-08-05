import json
import io
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_chapter_semantic_v04 as runner


def fake_provider(_key, prompt, _max_tokens=8192, _hard_timeout_seconds=90):
    if "顶层字段只能是 endpoints" in prompt:
        payload = {"endpoints": []}
    elif "顶层字段只能是 relations" in prompt:
        payload = {"relations": []}
    elif "顶层字段只能是 rules" in prompt:
        payload = {"rules": []}
    else:
        raise AssertionError("unknown prompt stage")
    return payload, {"finish_reason": "stop", "reasoning_content": None,
                     "reasoning_tokens": None,
                     "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


class SemanticV04IntegrationTests(unittest.TestCase):
    def test_provider_retries_complete_malformed_json_responses(self):
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class Opener:
            def __init__(self):
                self.calls = 0

            def open(self, *_args, **_kwargs):
                self.calls += 1
                content = '{}{}' if self.calls < 3 else '{"endpoints":[]}'
                envelope = {"choices": [{"finish_reason": "stop", "message": {
                    "content": content, "reasoning_content": None}}],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": None}}}
                return Response(json.dumps(envelope).encode())

        opener = Opener()
        with patch.object(runner, "_pinned_opener", return_value=opener):
            payload, metadata = runner._provider_post("test-secret", "prompt")
        self.assertEqual(payload, {"endpoints": []})
        self.assertEqual(metadata["format_attempts"], 3)

    def test_fixed_fixture_builds_complete_resumable_package(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v04"
            with patch.object(runner, "_provider_post", fake_provider):
                result = runner.run(
                    ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json",
                    ROOT / "source-packages/canonical/source/chapter-01/manifest.json",
                    ROOT / "runtime/archive/semantic/chapter-01-semantic-v0.2",
                    ROOT / "runtime/archive/semantic/chapter-01-semantic-v0.3",
                    output, "test-secret")
            self.assertEqual(result["stages"]["endpoint"]["completed"], 24)
            self.assertEqual(result["stages"]["relation"]["completed"], 24)
            self.assertEqual(result["stages"]["rule"]["completed"], 24)
            self.assertEqual(result["counts"]["catalog_entities"], 227)
            relation = json.loads((output / "relation-extraction.json").read_text(encoding="utf-8"))
            self.assertEqual(len(relation["baseline_model_candidates"]), 9)
            self.assertEqual(len(relation["derived_candidates"]), 6)
            self.assertEqual(len(relation["superseded_v02_review"]), 21)
            self.assertNotIn("test-secret", "".join(
                path.read_text(encoding="utf-8", errors="ignore") for path in output.glob("*.json")))
            with sqlite3.connect(output / "knowledge.sqlite") as db:
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
                self.assertGreater(db.execute("SELECT COUNT(*) FROM semantic_relation_evidence").fetchone()[0], 0)
            checkpoint = json.loads((output / "relation-checkpoint.json").read_text(encoding="utf-8"))
            checkpoint["prompt_version"] = "drift"
            (output / "relation-checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
            with patch.object(runner, "_provider_post", fake_provider):
                with self.assertRaisesRegex(RuntimeError, "checkpoint identity drift"):
                    runner.run(
                        ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json",
                        ROOT / "source-packages/canonical/source/chapter-01/manifest.json",
                        ROOT / "runtime/archive/semantic/chapter-01-semantic-v0.2",
                        ROOT / "runtime/archive/semantic/chapter-01-semantic-v0.3",
                        output, "test-secret")


if __name__ == "__main__":
    unittest.main()
