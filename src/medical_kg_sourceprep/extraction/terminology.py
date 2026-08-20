"""从规范原文发现候选术语，并用人工审核词典回标实体 mention。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import jieba


ENTITY_TYPES = frozenset(
    {"LabPanel", "LabIndicator", "IndicatorState", "ClinicalContext", "Disease"}
)
_TERM_RE = re.compile(r"^(?:[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9._+\-/()（） ]*|[A-Za-z][A-Za-z0-9._+\-/]*)$")
_LATIN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._+\-/]*$")
_BOUNDARY_CHAR_RE = re.compile(r"[A-Za-z0-9_]$")
_STOP_TERMS = frozenset(
    {
        "一个", "一些", "一般", "主要", "以上", "以下", "其中", "以及", "可以",
        "可能", "同时", "因此", "由于", "进行", "通过", "通常", "需要", "这种",
        "患者", "检查", "结果", "临床", "正常", "异常", "增高", "降低", "减少",
        "增加", "明显", "常见", "其他", "各种", "某些", "有关", "出现", "发生",
    }
)


def load_manifest_pages(manifest_path: Path) -> list[tuple[int, str, str]]:
    """读取规范 source manifest，返回页号、页 ID 和清洗后原文。"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = manifest_path.parent
    pages: list[tuple[int, str, str]] = []
    for page in manifest["pages"]:
        page_index = int(page["chapter_page_index"])
        text = (source_root / page["cleaned_path"]).read_text(encoding="utf-8")
        pages.append((page_index, str(page["page_id"]), text))
    return pages


def load_seed_terms(dataset_path: Path | None) -> set[str]:
    """从已有 mention 数据集读取词面，只用于改善 Jieba 切词，不继承其标签。"""
    if dataset_path is None:
        return set()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    containers = payload.get("pages", payload.get("cases", []))
    return {
        str(mention["exact_quote"])
        for container in containers
        for mention in container.get("mentions", [])
        if str(mention.get("exact_quote", "")).strip()
    }


def _valid_candidate(term: str, token_count: int) -> bool:
    """过滤标点、纯数字、过短词和明显无实体价值的高频通用词。"""
    compact = term.strip()
    if compact in _STOP_TERMS or "\n" in compact or len(compact) > 40:
        return False
    if not _TERM_RE.fullmatch(compact):
        return False
    if _LATIN_RE.fullmatch(compact):
        return len(compact) >= 2
    chinese_count = sum("\u4e00" <= char <= "\u9fff" for char in compact)
    return chinese_count >= 2 and (token_count > 1 or len(compact) >= 2)


def _page_candidates(text: str, max_ngram: int) -> Iterable[tuple[str, int, int]]:
    """产生 Jieba 单词及连续 2..N token 短语，并保留其原文坐标。"""
    tokens = [(word, start, end) for word, start, end in jieba.tokenize(text) if word.strip()]
    for left, (_, start, _) in enumerate(tokens):
        for size in range(1, max_ngram + 1):
            right = left + size - 1
            if right >= len(tokens):
                break
            end = tokens[right][2]
            term = text[start:end]
            if _valid_candidate(term, size):
                yield term.strip(), start, end


def discover_candidates(
    pages: Sequence[tuple[int, str, str]],
    *,
    seed_terms: Iterable[str] = (),
    min_frequency: int = 2,
    max_ngram: int = 4,
    max_examples: int = 3,
) -> list[dict[str, Any]]:
    """统计候选词频、页频与上下文，供人工建立词典。"""
    seed_term_set = set(seed_terms)
    for term in seed_term_set:
        jieba.add_word(term, freq=10_000_000)

    frequencies: Counter[str] = Counter()
    page_hits: dict[str, set[int]] = defaultdict(set)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page_index, page_id, text in pages:
        for term, start, end in _page_candidates(text, max_ngram):
            frequencies[term] += 1
            page_hits[term].add(page_index)
            if len(examples[term]) < max_examples:
                context_start = max(0, start - 30)
                context_end = min(len(text), end + 30)
                examples[term].append(
                    {
                        "page_id": page_id,
                        "page_index": page_index,
                        "start": start,
                        "end": end,
                        "context": text[context_start:context_end].replace("\n", " "),
                    }
                )

    rows = [
        {
            "term": term,
            "frequency": frequency,
            "document_frequency": len(page_hits[term]),
            "examples": examples[term],
            "entity_type": "",
            "review_status": "PENDING",
            "match_policy": "TOKEN_BOUNDARY" if _LATIN_RE.fullmatch(term) else "EXACT",
            "candidate_source": "EXISTING_MENTION" if term in seed_term_set else "JIEBA",
        }
        for term, frequency in frequencies.items()
        # 已有数据中的词面是重要召回线索，即使全书仅出现一次也不能被词频阈值删除。
        if frequency >= min_frequency or term in seed_term_set
    ]
    return sorted(
        rows,
        key=lambda row: (
            row["candidate_source"] != "EXISTING_MENTION",
            -row["document_frequency"],
            -row["frequency"],
            -len(row["term"]),
            row["term"],
        ),
    )


def write_candidate_artifacts(
    candidates: Sequence[dict[str, Any]], output_json: Path, output_tsv: Path
) -> None:
    """同时输出机器可读 JSON 与适合人工筛选的 TSV。"""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"schema_version": "terminology-candidates/v0.1", "entries": candidates},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with output_tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("term", "frequency", "document_frequency", "entity_type",
                        "review_status", "match_policy", "candidate_source", "aliases", "example"),
            delimiter="\t",
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({
                **{key: candidate.get(key, "") for key in writer.fieldnames
                   if key not in {"aliases", "example"}},
                "aliases": "|".join(candidate.get("aliases", [])),
                "example": candidate["examples"][0]["context"] if candidate["examples"] else "",
            })


def load_reviewed_dictionary(dictionary_path: Path) -> list[dict[str, Any]]:
    """读取审核后的 JSON 或 TSV；TSV 的 aliases 使用竖线分隔。"""
    if dictionary_path.suffix.lower() == ".tsv":
        with dictionary_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        return [
            {
                **row,
                "aliases": [alias.strip() for alias in row.get("aliases", "").split("|")
                            if alias.strip()],
            }
            for row in rows
        ]
    payload = json.loads(dictionary_path.read_text(encoding="utf-8"))
    return list(payload["entries"])


def annotate_with_dictionary(
    pages: Sequence[tuple[int, str, str]], entries: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """用已批准词典确定性回标原文；同一跨度和类型只输出一次。"""
    approved: list[tuple[str, str, str]] = []
    for entry in entries:
        if entry.get("review_status") != "APPROVED":
            continue
        entity_type = str(entry.get("entity_type", ""))
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"词典实体类型无效：{entry.get('term')} -> {entity_type}")
        policy = str(entry.get("match_policy", "EXACT"))
        if policy not in {"EXACT", "TOKEN_BOUNDARY"}:
            raise ValueError(f"词典匹配策略无效：{entry.get('term')} -> {policy}")
        for surface in (entry.get("term"), *entry.get("aliases", [])):
            if surface:
                approved.append((str(surface), entity_type, policy))

    documents: list[dict[str, Any]] = []
    for page_index, page_id, text in pages:
        mentions: dict[tuple[int, int, str], dict[str, Any]] = {}
        for surface, entity_type, policy in approved:
            cursor = 0
            while (start := text.find(surface, cursor)) >= 0:
                end = start + len(surface)
                cursor = end
                if policy == "TOKEN_BOUNDARY":
                    left_bad = start > 0 and _BOUNDARY_CHAR_RE.search(text[start - 1]) is not None
                    right_bad = end < len(text) and re.match(r"[A-Za-z0-9_]", text[end]) is not None
                    if left_bad or right_bad:
                        continue
                mentions[(start, end, entity_type)] = {
                    "start": start,
                    "end": end,
                    "exact_quote": surface,
                    "entity_type": entity_type,
                    "annotation_source": "APPROVED_DICTIONARY",
                }
        documents.append({
            "page_index": page_index,
            "page_id": page_id,
            "mentions": sorted(mentions.values(), key=lambda item: (item["start"], item["end"], item["entity_type"])),
        })
    return documents


def main(argv: Sequence[str] | None = None) -> None:
    """运行候选发现或审核词典回标。"""
    parser = argparse.ArgumentParser(description="构建和应用医学实体词典")
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover = subparsers.add_parser("discover", help="从全书原文发现候选词")
    discover.add_argument("--manifest", type=Path, required=True)
    discover.add_argument("--output", type=Path, required=True)
    discover.add_argument("--tsv", type=Path, required=True)
    discover.add_argument("--seed-dataset", type=Path)
    discover.add_argument("--min-frequency", type=int, default=2)
    discover.add_argument("--max-ngram", type=int, choices=range(1, 7), default=4)

    annotate = subparsers.add_parser("annotate", help="用人工批准词典回标原文")
    annotate.add_argument("--manifest", type=Path, required=True)
    annotate.add_argument("--dictionary", type=Path, required=True)
    annotate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    pages = load_manifest_pages(args.manifest)
    if args.command == "discover":
        candidates = discover_candidates(
            pages,
            seed_terms=load_seed_terms(args.seed_dataset),
            min_frequency=args.min_frequency,
            max_ngram=args.max_ngram,
        )
        write_candidate_artifacts(candidates, args.output, args.tsv)
        print(json.dumps({"pages": len(pages), "candidates": len(candidates),
                          "output": str(args.output), "tsv": str(args.tsv)}, ensure_ascii=False))
        return

    documents = annotate_with_dictionary(pages, load_reviewed_dictionary(args.dictionary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema_version": "dictionary-preannotations/v0.1", "documents": documents},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pages": len(documents),
                      "mentions": sum(len(item["mentions"]) for item in documents),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
