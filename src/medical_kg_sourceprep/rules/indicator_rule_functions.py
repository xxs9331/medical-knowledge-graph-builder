"""Evidence-bound Chapter 01 indicator rule-function extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from ..extraction.llm_extraction import EvidenceChunk


SCHEMA_VERSION = "indicator-rule-functions/v0.1"
PROMPT_VERSION = "indicator-rule-functions-prompt/v0.4-overlap"
VALIDATOR_VERSION = "indicator-rule-functions-validator/v0.5-semantic-link"
WINDOW_POLICY_VERSION = "target-page-plus-adjacent-chunk/v0.1"
MAX_RULES_PER_PAGE = 24

OPERATORS = frozenset({
    "EQ", "NE", "GT", "GE", "LT", "LE", "BETWEEN", "IN", "POSITIVE", "NEGATIVE",
})
CONTEXT_INPUTS = frozenset({
    "年龄", "性别", "孕期", "人群", "标本", "样本类型", "检测方法", "方法", "药物",
})


class RuleFunctionError(ValueError):
    """A rule candidate failed the function or evidence contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class WindowSegment:
    chunk: EvidenceChunk
    role: str
    window_start: int
    window_end: int
    chapter_start: int
    chapter_end: int


@dataclass(frozen=True, slots=True)
class RuleWindow:
    target_page_id: str
    target_page_index: int
    segments: tuple[WindowSegment, ...]
    text: str
    chapter_start: int


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *values: Any) -> str:
    payload = "\x1f".join(_canonical(value) for value in values)
    return f"{prefix}:{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def _chunk_order(chunk: EvidenceChunk) -> tuple[int, int, str]:
    return (
        chunk.page_index if chunk.page_index is not None else 10**9,
        chunk.start_offset if chunk.start_offset is not None else 0,
        chunk.chunk_id,
    )


def build_rule_windows(chunks: Sequence[EvidenceChunk]) -> dict[str, RuleWindow]:
    """Build page-owned windows with one neighboring chunk on each side."""
    ordered = sorted(chunks, key=_chunk_order)
    if not ordered:
        return {}
    chapter_offsets: dict[str, tuple[int, int]] = {}
    chapter_offset = 0
    for chunk in ordered:
        chapter_offsets[chunk.chunk_id] = (chapter_offset, chapter_offset + len(chunk.text))
        chapter_offset += len(chunk.text)

    pages: dict[str, list[EvidenceChunk]] = {}
    page_order: list[str] = []
    for chunk in ordered:
        if chunk.page_id not in pages:
            pages[chunk.page_id] = []
            page_order.append(chunk.page_id)
        pages[chunk.page_id].append(chunk)

    windows: dict[str, RuleWindow] = {}
    for page_position, page_id in enumerate(page_order):
        selected: list[tuple[EvidenceChunk, str]] = []
        if page_position > 0:
            selected.append((pages[page_order[page_position - 1]][-1], "left_context"))
        selected.extend((chunk, "target") for chunk in pages[page_id])
        if page_position + 1 < len(page_order):
            selected.append((pages[page_order[page_position + 1]][0], "right_context"))
        segments = []
        window_offset = 0
        for chunk, role in selected:
            chapter_start, chapter_end = chapter_offsets[chunk.chunk_id]
            segments.append(WindowSegment(
                chunk=chunk,
                role=role,
                window_start=window_offset,
                window_end=window_offset + len(chunk.text),
                chapter_start=chapter_start,
                chapter_end=chapter_end,
            ))
            window_offset += len(chunk.text)
        target_index = pages[page_id][0].page_index
        if target_index is None:
            raise RuleFunctionError("missing_page_index", "target page lacks page_index")
        windows[page_id] = RuleWindow(
            target_page_id=page_id,
            target_page_index=target_index,
            segments=tuple(segments),
            text="".join(segment.chunk.text for segment in segments),
            chapter_start=segments[0].chapter_start,
        )
    return windows


def _indicator_catalog(library: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for indicator in library.get("indicators", []):
        aliases = {
            value["value"]
            for field in ("aliases", "index_aliases")
            for value in indicator.get(field, [])
            if isinstance(value, Mapping) and isinstance(value.get("value"), str)
        }
        aliases.update(
            value["text"] for value in indicator.get("body_occurrences", [])
            if isinstance(value, Mapping) and isinstance(value.get("text"), str)
        )
        aliases.discard(indicator["canonical_name"])
        entries.append({
            "indicator_key": indicator["candidate_id"],
            "canonical_name": indicator["canonical_name"],
            "source_names": sorted(aliases),
            "derived_noncontiguous": "derived_table_column" in indicator.get("origins", []),
        })
    return sorted(entries, key=lambda value: value["canonical_name"])


def _catalog_terms(library: Mapping[str, Any]) -> set[str]:
    return {
        term
        for entry in _indicator_catalog(library)
        for term in [entry["canonical_name"], *entry["source_names"]]
        if term
    }


RULE_PROMPT_TEMPLATE = """# 角色
你是一位“医学检验指标规则抽取器”。从第一章当前重叠窗口中提取明确、可执行、可验证的指标规则。

# 规则格式
每条规则必须表示为：输出指标 = 规则名称(输入指标1, 输入指标2, …)
每个对象字段必须且只能是 rule_expression、cases、formula、default_result、evidence_overall。
- cases 必须是非空数组，每项字段只能是 condition、result、evidence。
- formula：计算规则填写原文逐字公式；非计算规则必须为 null。
- default_result：原文没有定义时必须为 null。
- evidence_overall 字段只能是 source_quote、condition_quotes、conclusion_quote。

# 允许抽取
1. 多指标联合判断。
2. 指标计算公式。
3. 阳性、阴性、分级或分类判断。
4. 根据年龄、性别、孕期或人群选择参考区间。
5. 根据标本或检测方法选择阈值。
6. 药物、方法、标本等因素对结果判断的明确影响。
7. 原文明示的 ALL、ANY、SINGLE 逻辑。

# 禁止事项
- 禁止外部医学知识、虚构指标、阈值、默认值和结论。
- 禁止输出只有病因罗列、机制说明或一般相关性而无判断、计算、选择、分级或分类逻辑的内容。
- INDICATOR_CATALOG 只帮助识别指标及别名；输出名称必须使用 WINDOW_TEXT 中实际出现的写法。
- derived_noncontiguous=true 的目录项不能仅凭目录名称当成连续原文。

# 条件运算符
condition 只能使用 EQ、NE、GT、GE、LT、LE、BETWEEN、IN、POSITIVE、NEGATIVE。
不要使用 =、==、!=、>、<、>=、<= 符号。逻辑连接使用中文“且”或“或”。
原文“正常、增大、减小”必须分别写成 EQ 正常、GT 正常范围、LT 正常范围等明确形式；不得省略运算符。
condition 中出现的每个数值必须逐字存在于该规则的 source_quote，禁止根据常识补阈值。
计算公式可用“指标 IN 公式输入”表示输入条件，result 和 formula 均使用原文公式或原文输出描述。

# 重叠窗口与证据
- CHUNKS 中 role=target 是当前目标页；left_context/right_context 仅用于补全跨边界规则。
- 只输出 conclusion_quote 起点位于 target chunk 的规则，避免相邻窗口重复。
- source_quote 必须是 WINDOW_TEXT 中连续且唯一出现的逐字片段，可跨相邻 chunks，但不得跳过文字或改写。
- case.evidence、condition_quotes、conclusion_quote 和非空 formula 都必须是 source_quote 的连续逐字子串。
- rule_expression 等号左侧的输出名称必须逐字出现在 source_quote，不得自行概括名称。
- 唯一例外：输出是 INDICATOR_CATALOG 中的指标且在 target chunk 的标题或正文逐字出现时，参考区间 source_quote 可只覆盖方法、人群和范围行。
- LaTeX、HTML、空格、标点和反斜杠都属于原文，evidence 不得删除或归一化它们。
- “可高达、可低至、最高可达、最低可达”是观察范围，不得改写成 GT/LT 判定阈值。
- 每个 case 的条件和结果都必须由原文明确支持。
- 如果表格定义多个输出，拆成多条规则，每条只有一个输出。
- 任何要求不满足时放弃该候选。

# 输出
只输出合法 JSON 数组，不输出 Markdown、代码围栏、说明文字或顶层对象。没有规则时输出 []。
数组最多 {max_rules} 项。
OUTPUT_SHAPE={output_shape}

TARGET_PAGE_ID={target_page_id}
INDICATOR_CATALOG={indicator_catalog}
CHUNKS={chunks}
WINDOW_TEXT={window_text}
"""


def _prompt_segments(window: RuleWindow) -> list[dict[str, Any]]:
    return [{
        "chunk_id": segment.chunk.chunk_id,
        "chunk_sha256": segment.chunk.chunk_sha256,
        "page_id": segment.chunk.page_id,
        "printed_page_number": segment.chunk.printed_page,
        "source_pdf_page_number": segment.chunk.source_pdf_page,
        "page_char_start": segment.chunk.start_offset,
        "window_char_start": segment.window_start,
        "window_char_end": segment.window_end,
        "chapter_char_start": segment.chapter_start,
        "chapter_char_end": segment.chapter_end,
        "role": segment.role,
        "text": segment.chunk.text,
    } for segment in window.segments]


def build_rule_prompt(window: RuleWindow, library: Mapping[str, Any]) -> str:
    output_shape = [{
        "rule_expression": "输出指标 = 规则名称(输入指标1, 输入指标2)",
        "cases": [{"condition": "输入指标1 BETWEEN 下限 AND 上限",
                   "result": "原文结果", "evidence": "原文连续证据"}],
        "formula": None,
        "default_result": None,
        "evidence_overall": {"source_quote": "覆盖规则的连续原文",
                             "condition_quotes": ["条件原文"],
                             "conclusion_quote": "结论原文"},
    }]
    return RULE_PROMPT_TEMPLATE.format(
        max_rules=MAX_RULES_PER_PAGE,
        output_shape=_canonical(output_shape),
        target_page_id=window.target_page_id,
        indicator_catalog=_canonical(_indicator_catalog(library)),
        chunks=_canonical(_prompt_segments(window)),
        window_text=_canonical(window.text),
    )


_EXPRESSION = re.compile(r"^\s*(?P<output>[^=()]+?)\s*=\s*(?P<name>[^=()]+?)\((?P<inputs>[^()]*)\)\s*$")
_SYMBOLIC_OPERATOR = re.compile(r"==|!=|>=|<=|(?<![A-Za-z])=|>|<")


def _parse_expression(value: Any) -> tuple[str, str, list[str]]:
    if not isinstance(value, str):
        raise RuleFunctionError("invalid_rule_expression", "rule_expression must be a string")
    match = _EXPRESSION.fullmatch(value)
    if match is None:
        raise RuleFunctionError("invalid_rule_expression", "rule_expression has invalid function syntax")
    inputs = [item.strip() for item in re.split(r"[,，]", match.group("inputs")) if item.strip()]
    if not inputs:
        raise RuleFunctionError("missing_inputs", "rule_expression requires at least one input")
    if len(inputs) != len(set(inputs)):
        raise RuleFunctionError("duplicate_inputs", "rule_expression inputs must be unique")
    return match.group("output").strip(), match.group("name").strip(), inputs


def _segment_for_position(window: RuleWindow, position: int) -> WindowSegment:
    for segment in window.segments:
        if segment.window_start <= position < segment.window_end:
            return segment
    if position == len(window.text):
        return window.segments[-1]
    raise RuleFunctionError("span_outside_window", "evidence span is outside the overlap window")


def _source_anchor(value: Any, window: RuleWindow) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise RuleFunctionError("missing_source_quote", "source_quote is required")
    if window.text.count(value) != 1:
        raise RuleFunctionError("source_quote_not_unique", "source_quote must be unique in WINDOW_TEXT")
    start = window.text.index(value)
    end = start + len(value)
    segment_spans = []
    for segment in window.segments:
        overlap_start = max(start, segment.window_start)
        overlap_end = min(end, segment.window_end)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - segment.window_start
        local_end = overlap_end - segment.window_start
        exact_quote = segment.chunk.text[local_start:local_end]
        if exact_quote != window.text[overlap_start:overlap_end]:
            raise RuleFunctionError("source_replay_failed", "window evidence cannot replay in source chunk")
        page_start = (segment.chunk.start_offset or 0) + local_start
        segment_spans.append({
            "chunk_id": segment.chunk.chunk_id,
            "chunk_sha256": segment.chunk.chunk_sha256,
            "page_id": segment.chunk.page_id,
            "printed_page_number": segment.chunk.printed_page,
            "source_pdf_page_number": segment.chunk.source_pdf_page,
            "role": segment.role,
            "chunk_char_start": local_start,
            "chunk_char_end": local_end,
            "page_char_start": page_start,
            "page_char_end": page_start + len(exact_quote),
            "chapter_char_start": segment.chapter_start + local_start,
            "chapter_char_end": segment.chapter_start + local_end,
            "exact_quote": exact_quote,
        })
    if not segment_spans or "".join(value["exact_quote"] for value in segment_spans) != value:
        raise RuleFunctionError("source_replay_failed", "source_quote does not map continuously to chunks")
    return {
        "target_page_id": window.target_page_id,
        "source_chunk_ids": [span["chunk_id"] for span in segment_spans],
        "window_char_start": start,
        "window_char_end": end,
        "chapter_char_start": window.chapter_start + start,
        "chapter_char_end": window.chapter_start + end,
        "exact_quote": value,
        "chunk_spans": segment_spans,
    }


def _subquote(value: Any, source_quote: str, field: str) -> str:
    if not isinstance(value, str) or not value or value not in source_quote:
        raise RuleFunctionError("component_not_verbatim", f"{field} is not verbatim in source_quote")
    return value


def _validate_condition(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise RuleFunctionError("invalid_condition", "case condition is required")
    if _SYMBOLIC_OPERATOR.search(value):
        raise RuleFunctionError("unsupported_operator", "condition uses a symbolic operator")
    if not any(re.search(rf"\b{operator}\b", value) for operator in OPERATORS):
        raise RuleFunctionError("missing_operator", "condition lacks an allowed operator")


def _validate_condition_grounding(condition: str, source_quote: str, evidence: str) -> None:
    for number in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", condition):
        if number not in evidence:
            raise RuleFunctionError(
                "condition_value_not_grounded",
                f"condition numeric value is absent from case evidence: {number}",
            )
    for qualitative in ("正常", "增大", "减小", "阳性", "阴性"):
        if qualitative in condition and qualitative not in evidence:
            raise RuleFunctionError(
                "condition_value_not_grounded",
                f"condition qualitative value is absent from case evidence: {qualitative}",
            )
    if re.search(r"\b(?:GT|GE|LT|LE|BETWEEN)\b", condition) and any(
        cue in evidence for cue in ("可高达", "可低至", "最高可达", "最低可达")
    ):
        raise RuleFunctionError(
            "descriptive_extreme_not_threshold",
            "descriptive extreme value cannot be converted into a decision threshold",
        )
    selector = any(
        re.search(rf"(?:^|且|或)\s*{name}\s+EQ\b", condition)
        for name in CONTEXT_INPUTS
    )
    operator_cues = {
        "BETWEEN": ("~", "至", "到"),
        "GT": (">", "&gt;", "以上", "大于", "高于", "超过", "增大", "升高", "增高"),
        "GE": (">", "&gt;", "以上", "不少于", "不低于"),
        "LT": ("<", "&lt;", "以下", "小于", "低于", "减小", "降低", "减少"),
        "LE": ("<", "&lt;", "以下", "不大于", "不高于"),
        "POSITIVE": ("阳性",),
        "NEGATIVE": ("阴性",),
    }
    for operator, cues in operator_cues.items():
        if re.search(rf"\b{operator}\b", condition) and not any(cue in evidence for cue in cues):
            raise RuleFunctionError(
                "operator_not_grounded",
                f"condition operator lacks a verbatim cue in case evidence: {operator}",
            )
    if re.search(r"\bEQ\b", condition) and not selector:
        qualitative_cues = ("正常", "存在", "增加", "增大", "升高", "降低", "减小", "下降", "使用")
        if not any(cue in evidence for cue in qualitative_cues):
            raise RuleFunctionError(
                "operator_not_grounded",
                "EQ condition lacks a verbatim qualitative cue in case evidence",
            )


def _validate_result_link(result: Any, evidence: str, source_quote: str) -> None:
    text = str(result)
    if text in evidence:
        return
    evidence_start = source_quote.find(evidence)
    prefix = source_quote[:evidence_start] if evidence_start >= 0 else ""
    if re.search(r"\n[ \t]*$", prefix):
        preceding_line = prefix.rstrip().splitlines()[-1].strip()
        heading = re.sub(r"^(?:\(?\d+\)?[、.)]?|[一二三四五六七八九十]+[、.])\s*", "", preceding_line)
        if (
            0 < len(preceding_line) <= 80
            and not preceding_line.endswith(("。", "！", "？", "；", ".", "!", "?", ";"))
            and text in heading
        ):
            return
    raise RuleFunctionError(
        "result_not_linked_to_case",
        "case result is not present in its evidence or an immediately preceding rule heading",
    )


def _target_contains(window: RuleWindow, text: str) -> bool:
    return any(segment.role == "target" and text in segment.chunk.text for segment in window.segments)


def _conclusion_owned_by_target(
    conclusion_quote: str,
    source_quote: str,
    source: Mapping[str, Any],
    window: RuleWindow,
) -> None:
    positions = [match.start() for match in re.finditer(re.escape(conclusion_quote), source_quote)]
    if not positions:
        raise RuleFunctionError("conclusion_not_grounded", "conclusion_quote is absent from source_quote")
    pages = {
        _segment_for_position(window, source["window_char_start"] + position).chunk.page_id
        for position in positions
    }
    if pages != {window.target_page_id}:
        raise RuleFunctionError(
            "conclusion_outside_target",
            "conclusion_quote must be owned only by the target page",
        )


def _reject(index: int, raw: Any, exc: RuleFunctionError, window: RuleWindow) -> dict[str, Any]:
    return {
        "review_id": _stable_id(
            "rule-review", window.target_page_id, index, exc.code, raw,
        ),
        "page_id": window.target_page_id,
        "reason_code": exc.code,
        "reason": str(exc),
        "raw_candidate": raw,
    }


def validate_rule_response(
    payload: Any,
    window: RuleWindow,
    library: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, list):
        raise RuleFunctionError("invalid_top_level", "provider output must be a JSON array")
    if len(payload) > MAX_RULES_PER_PAGE:
        raise RuleFunctionError("too_many_rules", "provider output exceeds the per-page rule limit")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    expected_rule = {"rule_expression", "cases", "formula", "default_result", "evidence_overall"}
    expected_case = {"condition", "result", "evidence"}
    expected_overall = {"source_quote", "condition_quotes", "conclusion_quote"}
    catalog_terms = _catalog_terms(library)

    for index, raw in enumerate(payload):
        try:
            if not isinstance(raw, Mapping) or set(raw) != expected_rule:
                raise RuleFunctionError("invalid_rule_shape", "rule fields do not match the contract")
            output_name, _rule_name, input_names = _parse_expression(raw["rule_expression"])
            cases = raw.get("cases")
            if not isinstance(cases, list) or not cases:
                raise RuleFunctionError("missing_cases", "cases must be a non-empty list")
            overall = raw.get("evidence_overall")
            if not isinstance(overall, Mapping) or set(overall) != expected_overall:
                raise RuleFunctionError("invalid_evidence_shape", "evidence_overall fields are invalid")
            source = _source_anchor(overall["source_quote"], window)
            source_quote = source["exact_quote"]
            condition_quotes = overall["condition_quotes"]
            if not isinstance(condition_quotes, list):
                raise RuleFunctionError("invalid_condition_quotes", "condition_quotes must be a list")
            normalized_condition_quotes = [
                _subquote(value, source_quote, "condition_quote") for value in condition_quotes
            ]
            conclusion_quote = _subquote(overall["conclusion_quote"], source_quote, "conclusion_quote")
            _conclusion_owned_by_target(conclusion_quote, source_quote, source, window)

            output_catalog_match = output_name in catalog_terms
            if output_name in source_quote:
                output_grounding = "source_quote"
            elif output_catalog_match and _target_contains(window, output_name):
                output_grounding = "target_catalog_context"
            else:
                raise RuleFunctionError("output_not_grounded", "rule output lacks source or target catalog grounding")
            for input_name in input_names:
                if input_name not in source_quote and input_name not in CONTEXT_INPUTS:
                    raise RuleFunctionError("input_not_grounded", f"input is absent from source_quote: {input_name}")

            conditions_text = " ".join(
                str(case.get("condition", "")) for case in cases if isinstance(case, Mapping)
            )
            for input_name in input_names:
                if input_name not in conditions_text and input_name not in CONTEXT_INPUTS:
                    raise RuleFunctionError("input_not_used", f"input is absent from all case conditions: {input_name}")
            normalized_cases = []
            for case in cases:
                if not isinstance(case, Mapping) or set(case) != expected_case:
                    raise RuleFunctionError("invalid_case_shape", "case fields do not match the contract")
                _validate_condition(case["condition"])
                evidence = _subquote(case["evidence"], source_quote, "case evidence")
                _validate_condition_grounding(case["condition"], source_quote, evidence)
                result = case["result"]
                if not isinstance(result, (str, int, float, bool)) or result == "":
                    raise RuleFunctionError("invalid_result", "case result must be a non-empty scalar")
                if str(result) not in source_quote and str(result) not in conclusion_quote:
                    raise RuleFunctionError("result_not_grounded", "case result is absent from source evidence")
                _validate_result_link(result, evidence, source_quote)
                normalized_cases.append({
                    "condition": case["condition"], "result": result, "evidence": evidence,
                })

            formula = raw["formula"]
            if formula is not None:
                formula = _subquote(formula, source_quote, "formula")
            default_result = raw["default_result"]
            if default_result is not None and str(default_result) not in source_quote:
                raise RuleFunctionError("default_not_grounded", "default_result is absent from source_quote")
            clean_rule = {
                "rule_expression": raw["rule_expression"],
                "cases": normalized_cases,
                "formula": formula,
                "default_result": default_result,
                "evidence_overall": {
                    "source_quote": source_quote,
                    "condition_quotes": normalized_condition_quotes,
                    "conclusion_quote": conclusion_quote,
                },
            }
            accepted.append({
                "candidate_id": _stable_id("indicator-rule", clean_rule),
                "status": "candidate", "approved": 0, "origin": "model",
                "page_id": window.target_page_id,
                "output_catalog_match": output_catalog_match,
                "output_grounding": output_grounding,
                "source": source,
                "rule": clean_rule,
            })
        except RuleFunctionError as exc:
            rejected.append(_reject(index, raw, exc, window))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate-only", "approved": 0,
        "target_page_id": window.target_page_id,
        "candidates": accepted, "rejections": rejected,
        "counts": {"accepted": len(accepted), "rejected": len(rejected)},
    }


def stable_candidates(packages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for package in packages:
        for candidate in package.get("candidates", []):
            merged.setdefault(candidate["candidate_id"], dict(candidate))
    return [merged[key] for key in sorted(merged)]


def audit_candidates(
    candidates: Sequence[Mapping[str, Any]],
    chunks: Sequence[EvidenceChunk],
) -> dict[str, Any]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    spans = 0
    components = 0
    for candidate in candidates:
        source = candidate["source"]
        replayed = []
        for span in source["chunk_spans"]:
            chunk = by_id.get(span["chunk_id"])
            if chunk is None or chunk.chunk_sha256 != span["chunk_sha256"]:
                raise RuleFunctionError("audit_hash_mismatch", "candidate chunk hash audit failed")
            quote = chunk.text[span["chunk_char_start"]:span["chunk_char_end"]]
            if quote != span["exact_quote"]:
                raise RuleFunctionError("audit_span_mismatch", "candidate chunk span audit failed")
            replayed.append(quote)
            spans += 1
        if "".join(replayed) != source["exact_quote"]:
            raise RuleFunctionError("audit_quote_mismatch", "candidate source quote audit failed")
        rule = candidate["rule"]
        values = [case["evidence"] for case in rule["cases"]]
        values.extend(rule["evidence_overall"]["condition_quotes"])
        values.append(rule["evidence_overall"]["conclusion_quote"])
        if rule["formula"] is not None:
            values.append(rule["formula"])
        if any(value not in source["exact_quote"] for value in values):
            raise RuleFunctionError("audit_component_mismatch", "rule component audit failed")
        components += len(values)
    return {
        "accepted_rules": len(candidates),
        "source_spans_replayed": spans,
        "components_replayed": components,
        "source_replay_rate": 1.0 if candidates else None,
        "component_replay_rate": 1.0 if candidates else None,
    }


def prompt_template_document() -> str:
    return RULE_PROMPT_TEMPLATE.format(
        max_rules=MAX_RULES_PER_PAGE,
        output_shape="{{OUTPUT_JSON_ARRAY_SHAPE}}",
        target_page_id="{{TARGET_PAGE_ID}}",
        indicator_catalog="{{INDICATOR_CATALOG}}",
        chunks="{{CHUNKS_WITH_INDEX_AND_ROLE}}",
        window_text="{{OVERLAP_WINDOW_TEXT}}",
    )
