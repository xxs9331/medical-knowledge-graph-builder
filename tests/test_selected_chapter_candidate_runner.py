import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_selected_chapter_candidates", ROOT / "scripts/run_selected_chapter_candidates.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class SelectedChapterCandidateRunnerTests(unittest.TestCase):
    def test_default_selection_covers_requested_chapters_without_header_only_chunks(self):
        chunks = runner.select_chunks(runner.DEFAULT_MANIFEST, ["02", "06", "13", "18"], 20)
        pages = {chunk.page_index for chunk in chunks}
        for start, end in runner.CHAPTER_RANGES.values():
            self.assertTrue(any(start <= page <= end for page in pages))
        self.assertTrue(all(len(chunk.text.strip()) >= 20 for chunk in chunks))
        self.assertTrue(all(
            any(start <= chunk.page_index <= end for start, end in runner.CHAPTER_RANGES.values())
            for chunk in chunks
        ))


if __name__ == "__main__":
    unittest.main()
