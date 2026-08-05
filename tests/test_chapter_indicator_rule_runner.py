from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_chapter_indicator_rules_v01",
    ROOT / "scripts/run_chapter_indicator_rules_v01.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class IndicatorRuleRunnerTests(unittest.TestCase):
    chunks = ROOT / "source-packages/chunks/chapter-01/manifest.json"
    library = (
        ROOT / "runtime/chapter-01-indicator-library-deepseek-direct-v0.1/indicator-library.json"
    )

    @staticmethod
    def metadata() -> dict:
        return {
            "finish_reason": "stop", "reasoning_content": None, "reasoning_tokens": None,
            "attempts": 1, "usage": {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
            },
        }

    def test_parallel_run_and_checkpoint_resume(self) -> None:
        lock = threading.Lock()
        calls = 0
        active = 0
        maximum_active = 0

        def post(_key: str, _prompt: str):
            nonlocal calls, active, maximum_active
            with lock:
                calls += 1
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return [], self.metadata()

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            result = RUNNER.run(
                self.chunks, self.library, output, "secret-key",
                workers=4, provider_post=post,
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["counts"]["pages_success"], 24)
            self.assertIn("quality_audit_sha256", result["artifacts"])
            quality = RUNNER._load_json(output / "quality-audit.json")
            self.assertEqual(quality["human_review"]["gold_status"], "unreviewed")
            self.assertFalse(quality["human_review"]["precision_recall_f1_reported"])
            self.assertEqual(calls, 24)
            self.assertGreater(maximum_active, 1)
            calls = 0
            resumed = RUNNER.run(
                self.chunks, self.library, output, "secret-key",
                workers=4, provider_post=post,
            )
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(calls, 0)
            self.assertNotIn("secret-key", (output / "checkpoint.json").read_text())

    def test_provider_errors_are_redacted_and_do_not_abort_other_pages(self) -> None:
        def failing_post(key: str, _prompt: str):
            raise RuntimeError(f"transport failed for {key}")

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            result = RUNNER.run(
                self.chunks, self.library, output, "secret-key",
                workers=4, provider_post=failing_post,
            )
            self.assertEqual(result["status"], "partial_failure")
            self.assertEqual(result["counts"]["pages_failed"], 24)
            checkpoint = (output / "checkpoint.json").read_text()
            self.assertNotIn("secret-key", checkpoint)
            self.assertIn("[REDACTED]", checkpoint)

    def test_local_revalidation_does_not_call_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            RUNNER.run(
                self.chunks, self.library, output, "secret-key",
                workers=4, provider_post=lambda _key, _prompt: ([], self.metadata()),
            )
            checkpoint_path = output / "checkpoint.json"
            checkpoint = RUNNER._load_json(checkpoint_path)
            checkpoint["validator_version"] = "older-validator"
            RUNNER.atomic_write_json(checkpoint_path, checkpoint)

            def forbidden_post(_key: str, _prompt: str):
                raise AssertionError("provider must not be called during revalidation")

            result = RUNNER.run(
                self.chunks, self.library, output, "",
                workers=4, revalidate=True, provider_post=forbidden_post,
            )
            self.assertEqual(result["status"], "complete")
            refreshed = RUNNER._load_json(checkpoint_path)
            self.assertEqual(refreshed["validator_version"], RUNNER.VALIDATOR_VERSION)
            self.assertTrue(all(
                page.get("revalidated_from") == "older-validator"
                for page in refreshed["pages"].values()
            ))

    def test_provider_uses_empty_proxy_handler(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps({
                    "choices": [{"finish_reason": "stop", "message": {"content": "[]"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }).encode()

        opener = mock.Mock()
        opener.open.return_value = Response()
        with mock.patch.object(RUNNER.request, "build_opener", return_value=opener) as build:
            payload, metadata = RUNNER._provider_post("secret-key", "prompt")
        self.assertEqual(payload, [])
        self.assertEqual(metadata["attempts"], 1)
        handler = build.call_args.args[0]
        self.assertEqual(handler.proxies, {})

    def test_worker_count_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "workers"):
                RUNNER.run(
                    self.chunks, self.library, Path(temp), "key",
                    workers=9, provider_post=lambda _key, _prompt: ([], self.metadata()),
                )


if __name__ == "__main__":
    unittest.main()
