"""Produce a bounded v0.1/v0.2 quality comparison without scoring gold labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from medical_kg_sourceprep.extraction.artifacts import directory_sha256, load_json as load



def summarize(path: Path) -> dict:
    data = load(path / "extraction.json")
    if "candidates" in data:
        candidates = data["candidates"]
        rejected = data.get("rejections", [])
        return {"version": data.get("schema_version"), "raw": len(candidates) + len(rejected),
                "accepted": len(candidates), "rejected": len(rejected), "approved": data.get("approved", 0),
                "reasons": dict(Counter(item.get("reason_code", "unknown") for item in rejected)),
                "evidence_replay_rate": 1.0 if all("source" in item for item in candidates if item.get("candidate_type") != "relation") else 0.0}
    extractions = data.get("extractions", [])
    review = load(path / "review-queue.json") if (path / "review-queue.json").exists() else {}
    return {"version": "v0.1-envelope", "raw": len(extractions), "accepted": None,
            "rejected": review.get("counts", {}).get("review_required"), "approved": 0,
            "reasons": dict(Counter(item.get("reason", "unknown") for item in review.get("items", []))),
            "evidence_replay_rate": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema_version": "semantic-quality-comparison/v0.2", "old": summarize(args.old), "new": summarize(args.new),
              "gold_status": "not_generated_from_model", "precision_recall_f1": "HOLD",
              "old_directory_sha256": directory_sha256(args.old, exclude_names={"quality-comparison.json"}),
              "new_directory_sha256": directory_sha256(args.new, exclude_names={"quality-comparison.json"})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
