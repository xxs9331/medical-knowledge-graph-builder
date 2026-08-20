import json
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.extraction.graph_builder.trace import JsonlTrace, TRACE_SCHEMA_VERSION


class GraphBuilderTraceTests(unittest.TestCase):
    def test_records_ordered_jsonl_events(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            trace = JsonlTrace(path, run_id="evaluation-001")

            trace.record("run/start", workflow="single-pass")
            with trace.stage("extraction/entity", chunk_id="chunk-1") as stage:
                _ = stage.update(attempts=1, accepted_count=3)

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([record["seq"] for record in records], [0, 1, 2])
        self.assertEqual(records[0]["schema_version"], TRACE_SCHEMA_VERSION)
        self.assertEqual(records[1]["type"], "extraction/entity/start")
        self.assertEqual(records[2]["type"], "extraction/entity/end")
        self.assertEqual(records[2]["data"]["accepted_count"], 3)
        self.assertEqual(records[2]["data"]["status"], "success")
        self.assertIn("duration_ms", records[2]["data"])

    def test_stage_records_error_without_swallowing_business_exception(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            trace = JsonlTrace(path, run_id="evaluation-002")

            with self.assertRaisesRegex(RuntimeError, "business failure"):
                with trace.stage("judge", chunk_id="chunk-1"):
                    raise RuntimeError("business failure")

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(records[-1]["data"]["status"], "error")
        self.assertEqual(records[-1]["data"]["error_type"], "RuntimeError")
        self.assertNotIn("business failure", json.dumps(records[-1], ensure_ascii=False))

    def test_serialization_failure_is_isolated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            trace = JsonlTrace(path)

            trace.record("invalid", unsupported=object())
            trace.record("valid", count=1)

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([record["type"] for record in records], ["valid"])
        self.assertEqual(records[0]["seq"], 0)
        self.assertEqual(trace.write_errors, ("TypeError",))


if __name__ == "__main__":
    unittest.main()
