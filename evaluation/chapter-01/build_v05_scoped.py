"""把非穷尽的 v0.4 参考集转换为限定评测域的 v0.5 金标。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "evaluation/chapter-01/chapter-01-layered-test-set-v0.4.json"
OUTPUT_PATH = ROOT / "evaluation/chapter-01/chapter-01-scoped-gold-v0.5.json"


if __name__ == "__main__":
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    for case in source["cases"]:
        evidence_by_id = {
            item["evidence_unit_id"]: item for item in case["evidence_units"]
        }
        labels_by_id = {
            item["canonical_id"]: {item["canonical_label"]}
            for item in case["canonical_entities"]
        }
        for link in case["mention_to_canonical_links"]:
            evidence = evidence_by_id[link["evidence_unit_id"]]
            quote = str(evidence["exact_quote"])
            # 只有最小 mention 可以作为等价表面形式；整行表格和上下文不能作为别名。
            if evidence["mention_eligible"] and "<" not in quote and len(quote) <= 64:
                labels_by_id[link["canonical_id"]].add(quote)
        for entity in case["canonical_entities"]:
            entity["accepted_surface_forms"] = sorted(labels_by_id[entity["canonical_id"]])

    source.update({
        "schema_version": "medical-kg-scoped-gold/v0.5",
        "status": "GENERATED_SCOPED_GOLD",
        "source_layered_reference": SOURCE_PATH.name,
        "evaluation_contract": {
            "closed_world": False,
            "mention_domain": "EXACT_ANNOTATED_MENTION_SPANS",
            "canonical_domain": "ANNOTATED_EVIDENCE_CONTEXTS",
            "relationship_domain": "PAIRS_BETWEEN_GOLD_CANONICAL_ENTITIES",
            "outside_scope_predictions": "IGNORED",
            "precision_recall_f1": "TP_FP_FN_WITHIN_DECLARED_DOMAIN",
        },
        "gold_provenance": {
            "method": "V03_REFERENCE_PLUS_SOURCE_SPAN_MAPPING_WITH_SCOPED_DOMAIN",
            "human_approved": False,
            "scoring_eligible": True,
        },
    })
    OUTPUT_PATH.write_text(
        json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "cases": len(source["cases"]),
        "canonical_entities": sum(len(case["canonical_entities"]) for case in source["cases"]),
        "surface_forms": sum(
            len(entity["accepted_surface_forms"])
            for case in source["cases"] for entity in case["canonical_entities"]
        ),
    }, ensure_ascii=False, sort_keys=True))
