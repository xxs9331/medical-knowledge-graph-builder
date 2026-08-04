"""Create an empty, auditable human-review template for all chapter pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pages = []
    for page in manifest.get("pages", []):
        pages.append({"page_id": page["page_id"], "printed_page_number": page.get("printed_page_number"),
                      "gold_status": "unreviewed", "candidates": [], "reviewer": None, "reviewed_at": None})
    value = {"schema_version": "semantic-gold-template/v0.2", "status": "HOLD",
             "source_manifest_sha256": __import__("hashlib").sha256(args.manifest.read_bytes()).hexdigest(),
             "generated_from_model": False, "pages": pages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
