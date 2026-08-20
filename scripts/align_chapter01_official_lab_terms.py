#!/usr/bin/env python3
"""Align Chapter 01 laboratory indicators with official Chinese standards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENTITIES = ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json"
DEFAULT_MENTIONS = ROOT / "evaluation/chapter-01/chapter-01-entity-mentions-v0.6.json"
DEFAULT_OUTPUT = ROOT / "knowledge/chapter-01/terminology/official-lab-alignment-v0.1.json"

WST886_URL = (
    "https://wjw.fujian.gov.cn/jggk/csxx/jhsyjtfzczcfgc/gzdt_37901/"
    "202606/P020260618620448875864.pdf"
)
WST779_URL = (
    "https://www.nhc.gov.cn/wjw/s9492/202105/"
    "a85d8b64e0384c98aed8f3157860ee44/files/1739781618961_15816.pdf"
)
EXPECTED_WST886_SHA256 = "4c5c21bd7eb19d42de8ee965e83ccb885db3899d975b50fef158434595d7febc"
EXPECTED_WST779_SHA256 = "19720f9517695d98c0d320da386b3e5ab26ff87cc5a51c05eed989fd0a516b3e"

# The book contains matching sections for rows 1-36. Rows 82, 101, 266 and
# 267 are included because Chapter 01 directly defines the corresponding tests.
WST886_SCOPE_ROWS = frozenset((*range(1, 37), 82, 101, 266, 267))
DIRECT_BOOK_ROWS = frozenset((82, 101, 266, 267))

WST886_ROW_OVERRIDES = {
    35: {
        "name": "纤维蛋白（原）降解产物检测",
        "category": "凝血实验",
        "analyte": "纤维蛋白（原）降解产物",
        "specimen": "血浆",
        "scale": "定量",
    }
}

# These equivalences are stated by Chapter 01 or are a suffix-only difference
# between the standard test name and the book's test heading.
BOOK_EQUIVALENCES = {
    "平均红细胞体积测定": "平均红细胞容积",
    "凝血酶原时间检测": "血浆凝血酶原时间",
    "纤维蛋白（原）降解产物检测": "纤维蛋白降解产物",
    "维生素 B9 测定": "叶酸",
    "中性粒细胞百分数": "中性粒细胞比例",
    "淋巴细胞百分数": "淋巴细胞比例",
}

# Existing extraction duplicates that Chapter 01 itself resolves. Redirects
# preserve source IDs and are applied before relationship publication.
INTERNAL_MERGES = {
    "血清铁蛋白": "铁蛋白",
    "血清转铁蛋白": "转铁蛋白",
    "血红蛋白浓度": "血红蛋白",
    "红细胞压积": "血细胞比容",
    "血细胞容积": "血细胞比容",
    "血浆纤维蛋白原": "纤维蛋白原",
    "血浆纤维蛋白原含量": "纤维蛋白原",
}

WST779_TERMS = (
    ("中性粒细胞绝对值", "Neut#", "x10^9/L"),
    ("淋巴细胞绝对值", "Lymph#", "x10^9/L"),
    ("单核细胞绝对值", "Mono#", "x10^9/L"),
    ("嗜酸性粒细胞绝对值", "Eos#", "x10^9/L"),
    ("嗜碱性粒细胞绝对值", "Baso#", "x10^9/L"),
    ("中性粒细胞百分数", "Neut%", "%"),
    ("淋巴细胞百分数", "Lymph%", "%"),
    ("单核细胞百分数", "Mono%", "%"),
    ("嗜酸性粒细胞百分数", "Eos%", "%"),
    ("嗜碱性粒细胞百分数", "Baso%", "%"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("（", "(").replace("）", ")").replace("％", "%")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"(测定|检测|检查|试验)$", "", value)
    return value.casefold()


def _stable_id(source: str, code: str) -> str:
    digest = hashlib.sha256(f"{source}\0{code}".encode()).hexdigest()[:20]
    return f"official-lab:{digest}"


def _pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_wst886(path: Path) -> list[dict[str, Any]]:
    if _sha256(path) != EXPECTED_WST886_SHA256:
        raise ValueError("WS/T 886-2026 PDF hash does not match the pinned official attachment")
    rows: list[dict[str, Any]] = []
    row_pattern = re.compile(r"^\s*(\d+)\s+(\d{7}[A-D])\s+(.*)$")
    for line in _pdf_text(path).splitlines():
        match = row_pattern.match(line)
        if match is None:
            continue
        sequence = int(match.group(1))
        parts = re.split(r"\s{2,}", line.strip())
        if sequence in WST886_ROW_OVERRIDES:
            fields = WST886_ROW_OVERRIDES[sequence]
        elif len(parts) == 7:
            fields = dict(zip(
                ("sequence", "code", "name", "category", "analyte", "specimen", "scale"),
                parts,
                strict=True,
            ))
        else:
            fields = {}
        rows.append({
            "sequence": sequence,
            "code": match.group(2),
            "name": fields.get("name"),
            "category": fields.get("category"),
            "analyte": fields.get("analyte"),
            "specimen": fields.get("specimen"),
            "scale": fields.get("scale"),
        })
    if len(rows) != 399 or [row["sequence"] for row in rows] != list(range(1, 400)):
        raise ValueError("WS/T 886-2026 table extraction did not yield rows 1 through 399")
    selected = [row for row in rows if row["sequence"] in WST886_SCOPE_ROWS]
    if any(not all(row[key] for key in ("name", "category", "analyte", "specimen", "scale")) for row in selected):
        raise ValueError("a selected WS/T 886-2026 row could not be parsed completely")
    return selected


def build_wst779_terms(path: Path) -> list[dict[str, Any]]:
    if _sha256(path) != EXPECTED_WST779_SHA256:
        raise ValueError("WS/T 779-2021 PDF hash does not match the pinned NHC attachment")
    text = _pdf_text(path)
    terms = []
    for name, abbreviation, unit in WST779_TERMS:
        if name not in text:
            raise ValueError(f"WS/T 779-2021 term not found in PDF: {name}")
        terms.append({
            "sequence": None,
            "code": None,
            "name": name,
            "category": "血细胞分析参考区间",
            "analyte": name,
            "specimen": "静脉血、末梢血",
            "scale": "定量",
            "abbreviations": [abbreviation],
            "unit": unit,
            "scope_reason": "BOOK_TABLE_DIRECT_MATCH",
        })
    return terms


def _load_entities(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    entities = document["canonical_entities"]
    indicators = [item for item in entities if item["entity_type"] == "LabIndicator"]
    return document, indicators


def _mention_evidence(mention_path: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(mention_path.read_text(encoding="utf-8"))
    return {
        mention["mention_id"]: {
            "mention_id": mention["mention_id"],
            "chunk_id": mention["chunk_id"],
            "exact_quote": mention["exact_quote"],
            "start": mention["start"],
            "end": mention["end"],
        }
        for case in document["cases"]
        for mention in case["mentions"]
    }


def _entity_index(indicators: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in indicators:
        for surface in (entity["canonical_name"], *entity.get("aliases", [])):
            if entity not in index[_identity(surface)]:
                index[_identity(surface)].append(entity)
    return index


def _resolve_internal_merges(indicators: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {item["canonical_name"]: item for item in indicators}
    redirects = []
    for source_name, target_name in INTERNAL_MERGES.items():
        source = by_name.get(source_name)
        target = by_name.get(target_name)
        if source is None or target is None:
            continue
        redirects.append({
            "source_canonical_id": source["canonical_id"],
            "source_name": source_name,
            "target_canonical_id": target["canonical_id"],
            "target_name": target_name,
            "decision": "BOOK_VERIFIED_INTERNAL_MERGE",
        })
    return redirects


def _official_term(source: str, row: dict[str, Any]) -> dict[str, Any]:
    code = row.get("code") or _identity(row["name"])
    return {
        "official_term_id": _stable_id(source, code),
        "source_standard": source,
        **row,
    }


def _build_aligned_entities(
    indicators: list[dict[str, Any]],
    terms: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    redirects: list[dict[str, Any]],
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    redirect_ids = {
        item["source_canonical_id"]: item["target_canonical_id"] for item in redirects
    }
    aligned: dict[str, dict[str, Any]] = {
        item["canonical_id"]: {
            **copy.deepcopy(item),
            "aliases": set(item.get("aliases", [])),
            "mention_ids": set(item.get("mention_ids", [])),
            "derivations": set(item.get("derivations", [])),
            "official_term_ids": set(),
            "automation_status": "AUTO_VALIDATED_EXISTING",
        }
        for item in indicators
        if item["canonical_id"] not in redirect_ids
    }
    by_original_id = {item["canonical_id"]: item for item in indicators}
    for source_id, target_id in redirect_ids.items():
        source = by_original_id[source_id]
        target = aligned[target_id]
        target["aliases"].add(source["canonical_name"])
        target["aliases"].update(source.get("aliases", []))
        target["mention_ids"].update(source.get("mention_ids", []))
        target["derivations"].update(source.get("derivations", []))
        target["derivations"].add("BOOK_VERIFIED_INTERNAL_MERGE")

    for addition in additions:
        aligned[addition["canonical_id"]] = {
            **copy.deepcopy(addition),
            "aliases": set(addition.get("aliases", [])),
            "mention_ids": set(addition.get("mention_ids", [])),
            "derivations": set(addition.get("derivations", [])),
            "official_term_ids": set(addition.get("official_term_ids", [])),
        }

    terms_by_id = {item["official_term_id"]: item for item in terms}
    for item in alignments:
        target_id = item.get("target_canonical_id")
        if target_id is None:
            continue
        target_id = redirect_ids.get(target_id, target_id)
        target = aligned[target_id]
        term = terms_by_id[item["official_term_id"]]
        target["official_term_ids"].add(term["official_term_id"])
        target["aliases"].add(term["name"])
        target["aliases"].update(term.get("abbreviations", []))

    result = []
    for item in aligned.values():
        canonical_identity = _identity(item["canonical_name"])
        item["aliases"] = sorted(
            {
                alias.strip() for alias in item["aliases"]
                if alias.strip() and _identity(alias) != canonical_identity
            },
            key=_identity,
        )
        item["mention_ids"] = sorted(item["mention_ids"])
        item["derivations"] = sorted(item["derivations"])
        item["official_term_ids"] = sorted(item["official_term_ids"])
        result.append(item)
    result.sort(key=lambda item: (_identity(item["canonical_name"]), item["canonical_id"]))
    identities = [_identity(item["canonical_name"]) for item in result]
    if len(identities) != len(set(identities)):
        raise ValueError("aligned LabIndicator snapshot contains duplicate canonical names")
    return result


def align(
    entity_path: Path,
    mention_path: Path,
    wst886_path: Path,
    wst779_path: Path,
) -> dict[str, Any]:
    entity_document, indicators = _load_entities(entity_path)
    mention_by_id = _mention_evidence(mention_path)
    index = _entity_index(indicators)
    by_name = {item["canonical_name"]: item for item in indicators}
    redirects = _resolve_internal_merges(indicators)
    redirect_ids = {item["source_canonical_id"]: item["target_canonical_id"] for item in redirects}

    rows886 = parse_wst886(wst886_path)
    for row in rows886:
        row["scope_reason"] = (
            "BOOK_DIRECT_TEST" if row["sequence"] in DIRECT_BOOK_ROWS
            else "BOOK_SECTION_CATEGORY"
        )
        row["abbreviations"] = []
        row["unit"] = None
    terms = [
        *(_official_term("WS/T 886-2026", row) for row in rows886),
        *(_official_term("WS/T 779-2021", row) for row in build_wst779_terms(wst779_path)),
    ]

    alignments: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for term in terms:
        explicit_name = BOOK_EQUIVALENCES.get(term["name"])
        candidates = [by_name[explicit_name]] if explicit_name in by_name else index.get(_identity(term["name"]), [])
        candidates = [
            by_name[next(item["target_name"] for item in redirects if item["source_canonical_id"] == candidate["canonical_id"])]
            if candidate["canonical_id"] in redirect_ids else candidate
            for candidate in candidates
        ]
        candidates = list({item["canonical_id"]: item for item in candidates}.values())
        if len(candidates) > 1:
            conflict = {
                "official_term_id": term["official_term_id"],
                "official_name": term["name"],
                "candidate_ids": sorted(item["canonical_id"] for item in candidates),
                "decision": "QUARANTINED_MULTIPLE_MATCHES",
            }
            conflicts.append(conflict)
            alignments.append(conflict)
            continue
        if not candidates:
            addition = {
                "canonical_id": term["official_term_id"],
                "canonical_name": re.sub(
                    r"(测定|检测|检查|试验)$", "", term["name"]
                ).strip(),
                "entity_type": "LabIndicator",
                "aliases": [term["name"]],
                "mention_ids": [],
                "derivations": ["OFFICIAL_SCOPE_EXTENSION"],
                "automation_status": "AUTO_VALIDATED_OFFICIAL_EXTENSION",
                "official_term_ids": [term["official_term_id"]],
            }
            additions.append(addition)
            alignments.append({
                "official_term_id": term["official_term_id"],
                "official_name": term["name"],
                "target_canonical_id": addition["canonical_id"],
                "target_name": addition["canonical_name"],
                "decision": "OFFICIAL_EXTENSION",
                "book_scope_reason": term["scope_reason"],
                "book_evidence": [],
            })
            continue
        target = candidates[0]
        evidence = [
            mention_by_id[mention_id]
            for mention_id in target.get("mention_ids", [])
            if mention_id in mention_by_id
        ]
        alignments.append({
            "official_term_id": term["official_term_id"],
            "official_name": term["name"],
            "target_canonical_id": target["canonical_id"],
            "target_name": target["canonical_name"],
            "decision": "BOOK_VERIFIED_EQUIVALENCE" if explicit_name else "NORMALIZED_NAME_MATCH",
            "book_scope_reason": term["scope_reason"],
            "book_evidence": evidence,
        })

    counts: dict[str, int] = defaultdict(int)
    for item in alignments:
        counts[item["decision"]] += 1
    aligned_entities = _build_aligned_entities(
        indicators, terms, alignments, redirects, additions
    )
    return {
        "schema_version": "chapter-01-official-lab-alignment/v0.1",
        "status": "AUTOMATED_ALIGNMENT_COMPLETE",
        "publication_status": "READY_FOR_GRAPH_REBUILD" if not conflicts else "BLOCKED_BY_CONFLICTS",
        "contract": {
            "user_validation_required": False,
            "existing_entities_are_primary": True,
            "book_defines_scope_and_equivalence": True,
            "specimen_alone_does_not_define_chapter_scope": True,
            "ambiguous_matches_are_quarantined": True,
            "neo4j_updated": False,
        },
        "sources": [
            {
                "standard": "WS/T 886-2026",
                "url": WST886_URL,
                "sha256": EXPECTED_WST886_SHA256,
                "publication_date": "2026-05-25",
                "effective_date": "2026-11-01",
                "source_status": "PUBLISHED_NOT_YET_EFFECTIVE",
            },
            {
                "standard": "WS/T 779-2021",
                "url": WST779_URL,
                "sha256": EXPECTED_WST779_SHA256,
                "source_status": "EFFECTIVE",
            },
            {
                "dataset": str(entity_path.relative_to(ROOT)),
                "sha256": _sha256(entity_path),
                "source_status": entity_document["status"],
            },
        ],
        "official_terms": terms,
        "alignments": alignments,
        "internal_merge_redirects": redirects,
        "official_extensions": additions,
        "aligned_entities": aligned_entities,
        "conflicts": conflicts,
        "statistics": {
            "existing_lab_indicator_count": len(indicators),
            "official_term_count": len(terms),
            "internal_merge_count": len(redirects),
            "official_extension_count": len(additions),
            "conflict_count": len(conflicts),
            "aligned_lab_indicator_count": len(aligned_entities),
            "decision_counts": dict(sorted(counts.items())),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    parser.add_argument("--mentions", type=Path, default=DEFAULT_MENTIONS)
    parser.add_argument("--wst886-pdf", type=Path, required=True)
    parser.add_argument("--wst779-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = align(args.entities, args.mentions, args.wst886_pdf, args.wst779_pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["statistics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
