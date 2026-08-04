from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import unittest


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

    def test_worker_count_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "workers"):
                RUNNER.run(
                    self.chunks, self.library, Path(temp), "key",
                    workers=9, provider_post=lambda _key, _prompt: ([], self.metadata()),
                )


if __name__ == "__main__":
    unittest.main()
