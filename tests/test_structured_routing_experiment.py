"""结构路由评分应在所有组中固定复用跨 chunk 关系。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_structured_routing_experiment import _case_graph_documents


class StructuredRoutingExperimentTests(unittest.TestCase):
    def test跨chunk图加入每个对照组(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "cases" / "CH01-01" / "cross-chunk-graph.json"
            path.parent.mkdir(parents=True)
            cross = {"nodes": [], "relationships": [{"candidate_key": "cross"}]}
            path.write_text(json.dumps(cross), encoding="utf-8")
            graphs = {
                "chunk-a": {"nodes": [], "relationships": []},
                "chunk-b": {"nodes": [], "relationships": []},
            }

            documents, resolved = _case_graph_documents(
                baseline_root=root,
                graphs=graphs,
                case_id="CH01-01",
                chunk_ids=("chunk-a", "chunk-b"),
            )

            self.assertEqual(len(documents), 3)
            self.assertEqual(documents[-1], cross)
            self.assertEqual(resolved, path)


if __name__ == "__main__":
    unittest.main()
