import json
import tempfile
import unittest
from pathlib import Path

from medical_kg_sourceprep.extraction.graph_builder.contract import (
    GraphBuilderConfigurationError,
)
from medical_kg_sourceprep.extraction.graph_builder.evaluation.artifacts import (
    artifact_matches_graph,
    load_json_object,
)


class GraphBuilderEvaluationValidationTests(unittest.TestCase):
    def test_load_json_object_rejects_top_level_array(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "artifact.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(GraphBuilderConfigurationError):
                load_json_object(path)

    def test_artifact_is_reused_only_when_graph_hash_matches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "judge-result.json"
            document = {"input": {"graph_sha256": "current-hash"}, "results": []}
            path.write_text(json.dumps(document), encoding="utf-8")

            matched = artifact_matches_graph(
                path, "current-hash", hash_path=("input", "graph_sha256")
            )
            stale = artifact_matches_graph(
                path, "old-hash", hash_path=("input", "graph_sha256")
            )

            self.assertEqual(matched, document)
            self.assertIsNone(stale)


if __name__ == "__main__":
    unittest.main()
