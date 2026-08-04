"""Generic, provenance-first contracts for composite rule candidates.

This module deliberately contains no medical vocabulary.  It represents candidate
structure and evaluates supplied facts, but never turns a candidate into an
approved rule without an explicit human review record.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Mapping, Sequence


class EvaluationValue(StrEnum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class CandidateStatus(StrEnum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


COMPOSITE_CLASSIFICATION = "COMPOSITE_CLASSIFICATION"
COMPOSITE_INTERPRETATION = "COMPOSITE_INTERPRETATION"
JOINT_TESTING_STATEMENT = "JOINT_TESTING_STATEMENT"


@dataclass(frozen=True, slots=True)
class TextAnchor:
    anchor_id: str
    source_id: str
    page: int | None
    raw_expression: str


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    subject_id: str | None = None


@dataclass(frozen=True, slots=True)
class AtomicPredicate:
    predicate_id: str
    raw_expression: str
    operator: str
    value: Any
    unit: str | None
    anchor: TextAnchor | None
    lower_inclusive: bool = True
    upper_inclusive: bool = True


@dataclass(frozen=True, slots=True)
class LogicNode:
    operator: str
    children: tuple[RuleNode, ...]
    threshold: int | None = None
    anchor: TextAnchor | None = None


RuleNode = AtomicPredicate | LogicNode


def all_of(*children: RuleNode, anchor: TextAnchor | None = None) -> LogicNode:
    return LogicNode("all", tuple(children), anchor=anchor)


def any_of(*children: RuleNode, anchor: TextAnchor | None = None) -> LogicNode:
    return LogicNode("any", tuple(children), anchor=anchor)


def at_least(threshold: int, *children: RuleNode, anchor: TextAnchor | None = None) -> LogicNode:
    return LogicNode("at_least", tuple(children), threshold=threshold, anchor=anchor)


def at_most(threshold: int, *children: RuleNode, anchor: TextAnchor | None = None) -> LogicNode:
    return LogicNode("at_most", tuple(children), threshold=threshold, anchor=anchor)


def not_of(child: RuleNode, *, anchor: TextAnchor | None = None) -> LogicNode:
    return LogicNode("not", (child,), anchor=anchor)


@dataclass(frozen=True, slots=True)
class DecisionRow:
    row_id: str
    conditions: Mapping[str, bool | None]
    output: str
    anchors: tuple[TextAnchor, ...]


@dataclass(frozen=True, slots=True)
class DecisionTable:
    table_id: str
    columns: tuple[str, ...]
    rows: tuple[DecisionRow, ...]
    hit_policy: str
    anchors: tuple[TextAnchor, ...]
    missing_policy: str | None = "unknown"


@dataclass(frozen=True, slots=True)
class CompositeCandidate:
    classification: str
    executable: bool
    status: CandidateStatus
    structure: RuleNode | DecisionTable | None = None
    diagnostics: tuple[str, ...] = ()

    def approve(self, reviewer: str, version: str) -> CompositeCandidate:
        # Candidate extraction cannot manufacture the required human decision.
        raise ValueError("approval requires an explicit review record")


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    reviewer: str
    decision: str
    version: str
    rationale: str


def apply_review(candidate: CompositeCandidate, review: ReviewRecord) -> CompositeCandidate:
    """Apply a separately authored review decision to a candidate."""
    if not review.reviewer.strip() or not review.version.strip() or not review.rationale.strip():
        raise ValueError("reviewer, version, and rationale are required")
    if review.decision not in {CandidateStatus.REVIEWED.value, CandidateStatus.APPROVED.value, CandidateStatus.REJECTED.value}:
        raise ValueError("review decision must be reviewed, approved, or rejected")
    return CompositeCandidate(candidate.classification, candidate.executable, CandidateStatus(review.decision), candidate.structure, candidate.diagnostics)


def evaluate(node: RuleNode, facts: Mapping[str, Any]) -> EvaluationValue:
    if isinstance(node, AtomicPredicate):
        return _evaluate_atomic(node, facts.get(node.predicate_id))
    values = [evaluate(child, facts) for child in node.children]
    if node.operator == "not":
        if len(values) != 1:
            return EvaluationValue.UNKNOWN
        return {EvaluationValue.TRUE: EvaluationValue.FALSE, EvaluationValue.FALSE: EvaluationValue.TRUE}.get(values[0], EvaluationValue.UNKNOWN)
    true_count = sum(value is EvaluationValue.TRUE for value in values)
    false_count = sum(value is EvaluationValue.FALSE for value in values)
    if node.operator == "all":
        return EvaluationValue.FALSE if false_count else (EvaluationValue.UNKNOWN if any(value is EvaluationValue.UNKNOWN for value in values) else EvaluationValue.TRUE)
    if node.operator == "any":
        return EvaluationValue.TRUE if true_count else (EvaluationValue.UNKNOWN if any(value is EvaluationValue.UNKNOWN for value in values) else EvaluationValue.FALSE)
    if node.operator in {"at_least", "at_most"} and node.threshold is not None:
        if node.operator == "at_least":
            return EvaluationValue.TRUE if true_count >= node.threshold else (EvaluationValue.UNKNOWN if true_count + sum(value is EvaluationValue.UNKNOWN for value in values) >= node.threshold else EvaluationValue.FALSE)
        possible_true_count = true_count + sum(value is EvaluationValue.UNKNOWN for value in values)
        if true_count > node.threshold:
            return EvaluationValue.FALSE
        if possible_true_count <= node.threshold:
            return EvaluationValue.TRUE
        return EvaluationValue.UNKNOWN
    return EvaluationValue.UNKNOWN


def _evaluate_atomic(predicate: AtomicPredicate, actual: Any) -> EvaluationValue:
    if actual is None:
        return EvaluationValue.UNKNOWN
    op, expected = predicate.operator, predicate.value
    if op in {"positive", "negative"}:
        if not isinstance(actual, (bool, int, float, Decimal)):
            return EvaluationValue.UNKNOWN
        result = bool(actual) if isinstance(actual, bool) else actual > 0
        return EvaluationValue.TRUE if result == (op == "positive") else EvaluationValue.FALSE
    try:
        left = Decimal(str(actual))
        if op == "in":
            return EvaluationValue.TRUE if left in {Decimal(str(item)) for item in expected} else EvaluationValue.FALSE
        if op == "between":
            low, high = (Decimal(str(item)) for item in expected)
            result = (left >= low if predicate.lower_inclusive else left > low) and (left <= high if predicate.upper_inclusive else left < high)
        else:
            right = Decimal(str(expected))
            result = {"lt": left < right, "le": left <= right, "eq": left == right, "ge": left >= right, "gt": left > right}.get(op)
            if result is None:
                return EvaluationValue.UNKNOWN
        return EvaluationValue.TRUE if result else EvaluationValue.FALSE
    except (InvalidOperation, TypeError, ValueError):
        return EvaluationValue.UNKNOWN


def validate_rule(rule: RuleNode | DecisionTable) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if isinstance(rule, DecisionTable):
        if rule.hit_policy not in {"UNIQUE", "COLLECT"}:
            issues.append(ValidationIssue("unsupported_hit_policy", "Only UNIQUE and COLLECT are explicit hit policies."))
        if rule.missing_policy is None:
            issues.append(ValidationIssue("missing_policy", "Missing-value behavior must be declared."))
        if not rule.anchors:
            issues.append(ValidationIssue("missing_anchor", "Decision table needs a TextAnchor.", rule.table_id))
        issues.extend(_validate_table_rows(rule))
        return tuple(issues)
    _validate_node(rule, issues)
    return tuple(issues)


def _validate_node(node: RuleNode, issues: list[ValidationIssue]) -> None:
    if isinstance(node, AtomicPredicate):
        if node.anchor is None:
            issues.append(ValidationIssue("missing_anchor", "Atomic predicate needs a TextAnchor.", node.predicate_id))
        if node.operator not in {"lt", "le", "eq", "ge", "gt", "between", "in", "positive", "negative"}:
            issues.append(ValidationIssue("unsupported_operator", "Comparison operator is not supported.", node.predicate_id))
        if node.unit is None and node.operator not in {"positive", "negative"}:
            issues.append(ValidationIssue("missing_unit", "Numeric predicate needs a unit.", node.predicate_id))
        return
    if node.anchor is None:
        issues.append(ValidationIssue("missing_anchor", "Logic connection needs a TextAnchor.", node.operator))
    if node.operator not in {"all", "any", "not", "at_least", "at_most"} or not node.children:
        issues.append(ValidationIssue("invalid_logic", "Logic node is not executable.", node.operator))
    if node.operator in {"at_least", "at_most"} and (node.threshold is None or node.threshold < 0):
        issues.append(ValidationIssue("invalid_threshold", "Threshold must be non-negative.", node.operator))
    for child in node.children:
        _validate_node(child, issues)


def _validate_table_rows(table: DecisionTable) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen: dict[tuple[tuple[str, bool | None], ...], DecisionRow] = {}
    for row in table.rows:
        if not row.anchors:
            issues.append(ValidationIssue("missing_anchor", "Decision row needs a TextAnchor.", row.row_id))
        key = tuple((column, row.conditions.get(column)) for column in table.columns)
        if key in seen:
            issues.append(ValidationIssue("overlap", "Decision rows overlap.", row.row_id))
            if seen[key].output != row.output:
                issues.append(ValidationIssue("conflict", "Overlapping rows have conflicting outputs.", row.row_id))
        seen[key] = row
    if table.hit_policy == "UNIQUE" and any(value is None for row in table.rows for value in row.conditions.values()):
        issues.append(ValidationIssue("ambiguous_row", "UNIQUE rows cannot contain wildcard conditions."))
    if table.columns and len(table.rows) < 2 ** len(table.columns):
        issues.append(ValidationIssue("hole", "Decision table does not cover the boolean input space."))
    return issues


def detect_composite_candidates(items: Sequence[Mapping[str, Any]]) -> tuple[CompositeCandidate, ...]:
    candidates: list[CompositeCandidate] = []
    for item in items:
        text = str(item.get("text", ""))
        if item.get("joint_testing") or "joint testing" in text.lower():
            candidates.append(CompositeCandidate(JOINT_TESTING_STATEMENT, False, CandidateStatus.CANDIDATE, diagnostics=("non_executable_statement",)))
        elif item.get("rows"):
            candidates.append(CompositeCandidate(COMPOSITE_INTERPRETATION, True, CandidateStatus.CANDIDATE, diagnostics=("structure_only",)))
        elif item.get("tests") and item.get("operator"):
            candidates.append(CompositeCandidate(COMPOSITE_CLASSIFICATION, True, CandidateStatus.CANDIDATE, diagnostics=("structure_only",)))
    return tuple(candidates)
