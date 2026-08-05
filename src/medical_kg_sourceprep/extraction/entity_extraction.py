"""Evidence-bound, five-category entity extraction for a chapter.

The model proposes page-local entities.  This module validates that every
name and alias is grounded in the supplied page text, then performs a small
deterministic merge without adding terminology from outside the source.
"""

from __future__ import annotations

from collections import Counter
import html
import re
import unicodedata
from typing import Any, Mapping, Sequence


ENTITY_CATEGORIES = (
    "LabTest",
    "Disease",
    "Population",
    "Etiology",
    "MethodOrDrug",
)
PROMPT_VERSION = "deepseek-entity-prompt/v0.3"
_CATEGORY_SET = frozenset(ENTITY_CATEGORIES)
_NON_TEST_COMPONENTS = frozenset({"红细胞", "白细胞", "血小板", "血浆", "骨髓"})
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_ABBREVIATION_RE = re.compile(r"(?:[A-Za-z]{2,}(?:[-_][A-Za-z0-9]+)*|[A-Z][0-9]+)")
_FORMULA_SYMBOL_RE = re.compile(r"^[\u0370-\u03ff\u1f00-\u1fff][A-Za-z]?$")
_GENERIC_NAMES = frozenset({"疾病", "人"})
_NON_METHOD_NAMES = frozenset({"形态学改变", "肌红蛋白", "运铁蛋白", "血常规检验"})


def _match_key(value: str) -> str:
    """Normalize OCR/Markdown decoration while retaining medical wording."""
    value = html.unescape(unicodedata.normalize("NFKC", value)).strip()
    value = value.replace("\\", "")
    value = re.sub(r"\s+", "", value)
    return value.translate(str.maketrans("", "", "(){}[]_$^`~"))


def _grounded(value: str, text: str) -> bool:
    return bool(value) and _match_key(value) in _match_key(text)


def _has_cjk(value: str) -> bool:
    return bool(_CJK_RE.search(value))


def _has_abbreviation(value: str) -> bool:
    return bool(_ABBREVIATION_RE.search(value))


def _is_formula_symbol(value: str) -> bool:
    return bool(_FORMULA_SYMBOL_RE.fullmatch(_match_key(value)))


def _invalid_category_name(category: str, name: str) -> str | None:
    normalized = _match_key(name)
    if normalized in {_match_key(value) for value in _GENERIC_NAMES}:
        return "generic term is not an entity"
    if category == "MethodOrDrug" and (
        normalized in {_match_key(value) for value in _NON_METHOD_NAMES}
        or normalized.endswith("检验")
        or normalized.endswith("检查")
        or normalized.endswith("改变")
    ):
        return "not a method, specimen, device, operation, or drug"
    return None


def build_entity_prompt(page_id: str, text: str) -> str:
    """Build the page-local prompt used by the direct DeepSeek runner."""
    return f"""你是医学知识图谱实体抽取器。只根据下面给出的《临床血液检验》第一章原文页面抽取实体，不使用任何外部知识。

页面标识：{page_id}

只返回一个 JSON 对象，顶层键必须是 entities。每条实体使用：
{{"category":"类别","name":"标准名称","aliases":["别名"],"mentions":["原文出现形式"]}}

允许的 category 只有：LabTest、Disease、Population、Etiology、MethodOrDrug。

类别定义：
- LabTest：检验指标或检验项目。name 使用原文中的中文名称，aliases 必须包含原文明确出现的英文缩写/英文代号；没有同时出现中文名和英文缩写时不要输出该 LabTest。只有“红细胞计数”“白细胞计数”“血小板计数”等测量项目属于 LabTest，单独的“红细胞”“白细胞”“血小板”等细胞成分不是 LabTest。
- Disease：疾病、疾病名称或临床表型/状态，例如贫血、缺铁性贫血、粒细胞减少症。
- Population：人群、生理阶段或适用人群，例如男性、女性、妊娠期、老年人。
- Etiology：原文明确作为病因、诱因、危险因素或致病机制的名词性短语，例如慢性感染、骨髓造血功能受损、维生素缺乏。
- MethodOrDrug：检测方法、标本类型、检验器材/操作或原文出现的相关药物。

严格规则：
1. 只抽取页面原文明确出现、且具有独立医学含义的名词性实体；不要补全、解释、改写或依据常识扩展。
2. name 必须是中文标准名称；英文缩写、英文代号和原文中的其他名称放入 aliases。每个 name 和 alias 都必须能在页面原文中逐字或仅忽略 Markdown/OCR 空白与公式标记后找到。
3. aliases 只放同一实体的原文别名、缩写或另一种名称，不放单位、数值、参考区间、描述性句子、关系词或分类标题。
4. 同一页面已出现多个名称的实体只输出一条，并合并 aliases；不要把上位词和带严重程度/类型限定词的下位实体合并。
5. 不输出章节标题、表格数值、单位、百分比、普通解剖部位、单独的细胞成分或一般动作，除非它们明确属于上述五类之一。若页面同时出现“红细胞(RBC)”和“红细胞计数”，可将 RBC 作为“红细胞计数”的原文缩写，但不要输出“红细胞”这个 LabTest；白细胞/血小板同理。
6. name 和 alias 必须是原文中连续出现的名称，不要把多个原文片段拼接成新名称。例如原文“变异系数(RDW-CV)”的 name 应为“变异系数”，不要写成“红细胞容积分布宽度变异系数”。
7. mentions 列出用于支持该条记录的原文形式；它们也必须来自页面原文。若没有额外形式，至少列出 name。

原文页面开始
<source>
{text}
</source>
原文页面结束
"""


def validate_page_result(
    raw: Any, text: str, page_index: int, *, grounding_text: str | None = None
) -> dict[str, Any]:
    """Validate model records and return accepted candidates plus rejections."""
    if not isinstance(raw, Mapping) or set(raw) != {"entities"} or not isinstance(raw["entities"], list):
        raise ValueError("model output must be {entities: [...]}")

    accepted: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    source_text = grounding_text or text
    for index, raw_item in enumerate(raw["entities"]):
        reason: str | None = None
        if not isinstance(raw_item, Mapping):
            reason = "entity is not an object"
        elif set(raw_item) - {"category", "name", "aliases", "mentions"}:
            reason = "unknown entity fields"
        elif raw_item.get("category") not in _CATEGORY_SET:
            reason = "invalid category"
        elif not isinstance(raw_item.get("name"), str) or not raw_item["name"].strip():
            reason = "name is missing"
        elif not _has_cjk(raw_item["name"]):
            reason = "name must contain Chinese text"
        elif _invalid_category_name(raw_item["category"], raw_item["name"]):
            reason = _invalid_category_name(raw_item["category"], raw_item["name"])
        elif raw_item.get("category") == "LabTest" and _match_key(raw_item["name"]) in {
            _match_key(value) for value in _NON_TEST_COMPONENTS
        }:
            reason = "cell component is not a LabTest"
        elif not _grounded(raw_item["name"], source_text):
            reason = "name is not grounded in source"
        elif not isinstance(raw_item.get("aliases", []), list) or any(
            not isinstance(value, str) or not value.strip() for value in raw_item.get("aliases", [])
        ):
            reason = "aliases must be a list of non-empty strings"

        if reason is None:
            name = raw_item["name"].strip()
            aliases: list[str] = []
            seen = {_match_key(name)}
            for alias in raw_item.get("aliases", []):
                alias = alias.strip()
                key = _match_key(alias)
                if key in seen:
                    continue
                if not _grounded(alias, source_text):
                    reason = "alias is not grounded in source"
                    break
                if _is_formula_symbol(alias):
                    continue
                seen.add(key)
                aliases.append(alias)
            mentions = raw_item.get("mentions", [name])
            if reason is None and (
                not isinstance(mentions, list)
                or any(not isinstance(value, str) or not value.strip() or not _grounded(value, source_text) for value in mentions)
            ):
                reason = "mentions are not grounded in source"

        if reason is not None:
            rejections.append({"page_index": page_index, "index": index, "reason": reason, "raw": raw_item})
            continue
        accepted.append({
            "category": raw_item["category"],
            "name": name,
            "aliases": aliases,
            "mentions": [value.strip() for value in mentions],
            "needs_abbreviation": raw_item["category"] == "LabTest" and not any(_has_abbreviation(alias) for alias in aliases),
            "page_index": page_index,
        })
    return {"entities": accepted, "rejections": rejections}


def _token_set(item: Mapping[str, Any]) -> set[str]:
    return {_match_key(value) for value in [item["name"], *item.get("aliases", [])] if value}


_ETIOLOGY_HINT_RE = re.compile(
    r"缺乏|受损|衰竭|不良|不足|减少|增多|增高|降低|增加|浓缩|稀薄|失血|出血|感染|炎症|缺氧|损伤|坏死|"
    r"过多|过量|暴露|接触|使用|应用|治疗|输血|环境|腹泻|呕吐|出汗|营养不良|吸收不良|功能低下|"
    r"合成|释放|破坏|消耗|聚集性|黏度|浓度|含量|内径|几何形状|弹性|电荷|渗透压|pH"
)
_DISEASE_HINT_RE = re.compile(r"病|症|癌|瘤|梗死|血栓|白血病|淋巴瘤|中毒|溶血|心脏病|脉管炎|哮喘|皮炎|结核")


def _resolve_category(group: Mapping[str, Any]) -> str:
    """Resolve exact-name category collisions without inventing a new type."""
    categories = Counter(group["categories"])
    if len(categories) == 1:
        return next(iter(categories))
    for category in ("LabTest", "MethodOrDrug", "Population"):
        if category in categories:
            return category
    if "Disease" in categories and "Etiology" in categories:
        name = group["name"]
        if _ETIOLOGY_HINT_RE.search(name) and not _DISEASE_HINT_RE.search(name):
            return "Etiology"
        return "Disease"
    return categories.most_common(1)[0][0]


def merge_entities(page_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge exact name/alias overlaps while keeping source order deterministic."""
    groups: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for page_result in page_results:
        for item in page_result.get("entities", []):
            invalid_reason = _invalid_category_name(item["category"], item["name"])
            if invalid_reason:
                conflicts.append({"type": "candidate_filtered", "name": item["name"], "category": item["category"], "reason": invalid_reason})
                continue
            tokens = _token_set(item)
            matches = [
                group for group in groups
                if tokens & group["tokens"]
            ]
            if not matches:
                groups.append({
                    "categories": [item["category"]],
                    "name": item["name"],
                    "aliases": list(item.get("aliases", [])),
                    "mentions": list(item.get("mentions", [])),
                    "tokens": set(tokens),
                    "needs_abbreviation": bool(item.get("needs_abbreviation")),
                    "first_page": item.get("page_index", 0),
                })
                continue
            group = matches[0]
            if len(matches) > 1:
                conflicts.append({"type": "ambiguous_overlap", "name": item["name"], "page_index": item.get("page_index")})
            if item["category"] not in group["categories"]:
                group["categories"].append(item["category"])
            group["needs_abbreviation"] = group["needs_abbreviation"] and bool(item.get("needs_abbreviation"))
            group_tokens = group["tokens"]
            for value in [item["name"], *item.get("aliases", [])]:
                if _is_formula_symbol(value):
                    continue
                if _match_key(value) not in group_tokens:
                    group["aliases"].append(value)
                    group_tokens.add(_match_key(value))
            for value in item.get("mentions", []):
                if _match_key(value) not in {_match_key(existing) for existing in group["mentions"]}:
                    group["mentions"].append(value)

    final: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda value: (ENTITY_CATEGORIES.index(_resolve_category(value)), value["first_page"], value["name"])):
        category = _resolve_category(group)
        if len(group["categories"]) > 1:
            resolutions.append({"name": group["name"], "from": group["categories"], "to": category})
        aliases: list[str] = []
        seen = {_match_key(group["name"])}
        for alias in group["aliases"]:
            if _is_formula_symbol(alias):
                continue
            key = _match_key(alias)
            if key and key not in seen:
                aliases.append(alias)
                seen.add(key)
        if category == "LabTest" and not any(_has_abbreviation(alias) for alias in aliases):
            conflicts.append({"type": "labtest_missing_abbreviation_after_merge", "name": group["name"]})
            continue
        final.append({"category": category, "name": group["name"], "aliases": aliases})

    counts = Counter(item["category"] for item in final)
    return {"entities": final, "conflicts": conflicts, "category_resolutions": resolutions, "counts": dict(counts)}
