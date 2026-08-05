"""Deterministic, fail-closed synthesis of report facts and approved rules.

This module is an adapter over the report, rule, and evidence contracts.  It
does not contain medical vocabulary or interpretation logic; callers provide
the rule wording and approved BOOK citations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, Mapping, Sequence

from .composite_rules import (
    AtomicPredicate,
    CandidateStatus,
    DecisionTable,
    EvaluationValue,
    ReviewRecord,
    RuleNode,
    evaluate,
    validate_rule,
)
from .evidence_policy import (
    AtomicPredicateRef,
    CitationBundle,
    Claim,
    ComputationTraceRef,
    EvidenceGap,
    RuleMatchRef,
    RuleVersionRef,
    build_report_source,
    validate_citation_bundle,
)
from ..report.report_model import AbnormalFlag, EvaluationResult, Observation, evaluate_observation


ANALYSIS_SCHEMA_VERSION = "report-analysis/v0.2"
ALGORITHM_VERSION = "report-analysis-algorithm/v0.2"


@dataclass(frozen=True, slots=True)
class AnalysisRule:
    """A separately reviewed executable rule and its optional BOOK evidence."""

    rule_id: str
    version: str
    status: str
    structure: RuleNode | DecisionTable
    review: ReviewRecord | None
    conclusion: str
    book_sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GroupedAbnormality:
    item_id: str
    raw_name: str
    value: str | None
    unit: str | None
    computed_flag: AbnormalFlag | None
    report_flag: AbnormalFlag | None
    errors: tuple[dict[str, Any], ...] = ()
    report_book_interval_conflict: bool = False


@dataclass(frozen=True, slots=True)
class PredicateTrace:
    predicate_id: str
    input_key: str | None
    input_value: str | None
    input_unit: str | None
    expected_unit: str | None
    value: str
    result: str
    anchor_id: str | None
    citation_id: str | None


@dataclass(frozen=True, slots=True)
class ComputationTrace:
    trace_id: str
    algorithm_version: str
    rule_id: str
    rule_version: str
    predicates: tuple[PredicateTrace, ...]


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    rule_id: str
    rule_version: str
    value: str
    trace: ComputationTrace
    gaps: tuple[EvidenceGap, ...] = ()
    conflicts: tuple[str, ...] = ()
    claim: Claim | None = None
    citation_bundle: CitationBundle | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    schema_version: str
    abnormalities: tuple[GroupedAbnormality, ...]
    rule_evaluations: tuple[RuleEvaluation, ...]
    claims: tuple[Claim, ...] = ()
    citation_bundles: tuple[CitationBundle, ...] = ()
    gaps: tuple[EvidenceGap, ...] = ()


def analyze_report(
    report: Mapping[str, Observation] | Sequence[Observation],
    rules: Sequence[AnalysisRule],
    *,
    approved_book_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> AnalysisResult:
    """Analyze synthetic/structured report data without making a diagnosis.

    Report order and rule order are canonicalized by stable identifiers.  A
    rule may compute a boolean result without being eligible for a claim; any
    missing review, anchor, BOOK evidence, or report input is retained as a
    gap and prevents claim emission.
    """
    report_items = _report_items(report)
    observations: dict[str, EvaluationResult] = {}
    abnormalities: list[GroupedAbnormality] = []
    for item_id, observation in report_items:
        outcome = evaluate_observation(observation)
        observations[item_id] = outcome
        errors = tuple(error.to_dict() for error in outcome.evidence.errors)
        abnormalities.append(
            GroupedAbnormality(
                item_id=item_id,
                raw_name=observation.raw_name,
                value=_text(outcome.normalized.value),
                unit=outcome.normalized.unit,
                computed_flag=outcome.evidence.computed_flag,
                report_flag=observation.report_flag,
                errors=errors,
            )
        )

    evaluations: list[RuleEvaluation] = []
    all_gaps: list[EvidenceGap] = []
    for abnormality in abnormalities:
        for error in abnormality.errors:
            all_gaps.append(EvidenceGap(abnormality.item_id, str(error.get("code", "report error")), True))
    for rule in sorted(rules, key=lambda value: (value.rule_id, value.version)):
        evaluation = _evaluate_rule(rule, report_items, observations, approved_book_registry)
        evaluations.append(evaluation)
        all_gaps.extend(evaluation.gaps)

    claims = tuple(item.claim for item in evaluations if item.claim is not None)
    bundles = tuple(item.citation_bundle for item in evaluations if item.citation_bundle is not None)
    return AnalysisResult(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        abnormalities=tuple(abnormalities),
        rule_evaluations=tuple(evaluations),
        claims=claims,
        citation_bundles=bundles,
        gaps=tuple(_unique_gaps(all_gaps)),
    )


def result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    """Return JSON-friendly output while preserving enum values and decimals."""
    return _jsonable(asdict(result))


def _evaluate_rule(
    rule: AnalysisRule,
    report_items: Sequence[tuple[str, Observation]],
    observations: Mapping[str, EvaluationResult],
    approved_book_registry: Mapping[str, Mapping[str, Any]] | None,
) -> RuleEvaluation:
    atoms = _atoms(rule.structure)
    by_key = _observation_lookup(report_items)
    facts: dict[str, Any] = {}
    traces: list[PredicateTrace] = []
    gaps: list[EvidenceGap] = []
    if isinstance(rule.structure, DecisionTable):
        value = EvaluationValue.UNKNOWN
        gaps.append(EvidenceGap(rule.rule_id, "unsupported decision table execution", True))
    else:
        for issue in validate_rule(rule.structure):
            gaps.append(EvidenceGap(issue.subject_id or rule.rule_id, issue.code, True))
        for atom in atoms:
            key = atom.predicate_id if atom.predicate_id in by_key else atom.raw_expression
            outcome = observations.get(by_key.get(key, "")) if key in by_key else None
            normalized = outcome.normalized if outcome else None
            actual = normalized.value if normalized and not outcome.evidence.errors else None
            if normalized and atom.unit != normalized.unit:
                actual = None
                gaps.append(EvidenceGap(atom.predicate_id, "unit mismatch", True))
            if actual is None:
                gaps.append(EvidenceGap(atom.predicate_id, "unknown", True))
            else:
                facts[atom.predicate_id] = actual
            traces.append(
                PredicateTrace(
                    predicate_id=atom.predicate_id,
                    input_key=by_key.get(key),
                    input_value=_text(normalized.value) if normalized else None,
                    input_unit=normalized.unit if normalized else None,
                    expected_unit=atom.unit,
                    value=_text(actual),
                    result=EvaluationValue.UNKNOWN.value if actual is None else "pending",
                    anchor_id=atom.anchor.anchor_id if atom.anchor else None,
                    citation_id=_citation_id(rule.book_sources.get(atom.predicate_id)),
                )
            )
        value = evaluate(rule.structure, facts)
        traces = [
            PredicateTrace(**{**asdict(trace), "result": _predicate_result(atom, facts)})
            for trace, atom in zip(traces, atoms)
        ]
    if rule.status != CandidateStatus.APPROVED.value:
        gaps.append(EvidenceGap(rule.rule_id, f"rule status is {rule.status}", True))
    if rule.review is None or rule.review.decision != CandidateStatus.APPROVED.value:
        gaps.append(EvidenceGap(rule.rule_id, "missing or non-approved ReviewRecord", True))
    for atom in atoms:
        if atom.anchor is None:
            gaps.append(EvidenceGap(atom.predicate_id, "missing BOOK TextAnchor", True))
        if _citation_id(rule.book_sources.get(atom.predicate_id)) is None:
            gaps.append(EvidenceGap(atom.predicate_id, "missing BOOK evidence", True))

    trace = ComputationTrace(
        trace_id=f"{rule.rule_id}:{rule.version}:trace",
        algorithm_version=ALGORITHM_VERSION,
        rule_id=rule.rule_id,
        rule_version=rule.version,
        predicates=tuple(traces),
    )
    report_source = None
    if value is EvaluationValue.TRUE and not gaps and rule.conclusion.strip():
        try:
            report_source = _report_source(report_items, observations, traces)
        except (ValueError, TypeError):
            gaps.append(EvidenceGap(rule.rule_id, "invalid REPORT snapshot", True))
    claim, bundle = _claim_if_valid(rule, value, trace, gaps, approved_book_registry, report_source)
    if value is EvaluationValue.TRUE and not gaps and claim is None:
        gaps.append(EvidenceGap(rule.rule_id, "invalid or incomplete citation bundle", True))
    return RuleEvaluation(rule.rule_id, rule.version, value.value, trace, tuple(_unique_gaps(gaps)), (), claim, bundle)


def _claim_if_valid(rule, value, trace, gaps, registry, report_source):
    if value is not EvaluationValue.TRUE or gaps or not rule.conclusion.strip() or report_source is None:
        return None, None
    sources = (report_source, *tuple(rule.book_sources.values()))
    predicate_refs = tuple(
        AtomicPredicateRef(atom.predicate_id, atom.raw_expression, _citation_id(rule.book_sources[atom.predicate_id]))
        for atom in _atoms(rule.structure)
    )
    claim = Claim(f"{rule.rule_id}:{rule.version}:claim", rule.conclusion, "medical_explanation", "assertive")
    rule_ref = RuleVersionRef(rule.rule_id, rule.version, rule.status)
    matches = tuple(
        RuleMatchRef(f"{claim.claim_id}:{atom.predicate_id}", claim.claim_id, rule.rule_id, atom.predicate_id, _citation_id(rule.book_sources[atom.predicate_id]), rule.version)
        for atom in _atoms(rule.structure)
    )
    computation = ComputationTraceRef(trace.trace_id, trace.algorithm_version, tuple(item.predicate_id for item in trace.predicates), claim.claim_id, tuple(source["citation_id"] for source in sources), report_source["citation_id"])
    bundle = CitationBundle(claim, rule_ref, predicate_refs, matches, computation, sources)
    try:
        validate_citation_bundle(bundle, registry)
    except (ValueError, TypeError):
        return None, None
    return claim, bundle


def _atoms(structure):
    if isinstance(structure, AtomicPredicate):
        return (structure,)
    atoms = []
    for child in structure.children:
        atoms.extend(_atoms(child))
    return tuple(atoms)


def _report_items(report):
    if isinstance(report, Mapping):
        return tuple(sorted(((str(key), value) for key, value in report.items()), key=lambda item: item[0]))
    return tuple(sorted(((item.standard_name or item.raw_name, item) for item in report), key=lambda item: item[0]))


def _observation_lookup(items):
    lookup = {}
    for item_id, observation in items:
        for key in (item_id, observation.raw_name, observation.standard_name, observation.abbreviation):
            if key:
                lookup[str(key)] = item_id
    return lookup


def _predicate_result(atom, facts):
    if atom.predicate_id not in facts:
        return EvaluationValue.UNKNOWN.value
    return evaluate(atom, facts).value


def _citation_id(source):
    value = source.get("citation_id") if isinstance(source, Mapping) else None
    return value if isinstance(value, str) and value else None


def _report_source(report_items, observations, traces):
    selected = []
    by_id = dict(report_items)
    for trace in traces:
        if trace.input_key is None or trace.input_key not in by_id:
            continue
        item_id = trace.input_key
        observation = by_id[item_id]
        normalized = observations[item_id].normalized
        interval = {
            "lower": _text(normalized.lower),
            "upper": _text(normalized.upper),
            "lower_inclusive": normalized.lower_inclusive,
            "upper_inclusive": normalized.upper_inclusive,
        }
        selected.append({
            "item_id": item_id,
            "value": _text(normalized.value),
            "unit": normalized.unit,
            "reference_interval": interval,
            "report_flag": observation.report_flag.value if observation.report_flag else None,
        })
    payload = {"schema_version": "report-snapshot/v0.2", "items": selected}
    payload_hash_source = build_report_source(citation_id="report-pending", source_id="report-pending", payload=payload)
    payload_hash = payload_hash_source["payload_hash"]
    return build_report_source(
        citation_id=f"report:{payload_hash[:16]}",
        source_id=f"report:{payload_hash}",
        payload=payload,
    )


def _unique_gaps(gaps):
    seen = set()
    output = []
    for gap in gaps:
        key = (gap.subject_id, gap.reason, gap.required)
        if key not in seen:
            seen.add(key)
            output.append(gap)
    return output


def _text(value):
    return str(value) if value is not None else None


def _jsonable(value):
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


__all__ = [
    "ALGORITHM_VERSION", "ANALYSIS_SCHEMA_VERSION", "AnalysisResult", "AnalysisRule",
    "ComputationTrace", "GroupedAbnormality", "PredicateTrace", "RuleEvaluation",
    "analyze_report", "result_to_dict",
]
