"""Versioned, privacy-aware contracts for structured laboratory report facts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Mapping


class AbnormalFlag(StrEnum):
    """A reported or mechanically computed interval comparison result."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class StructuredError:
    code: str
    field: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "field": self.field, "message": self.message}


@dataclass(frozen=True, slots=True)
class ReferenceInterval:
    lower: str | Decimal | None = None
    upper: str | Decimal | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    sex_condition: str | None = None
    age_min_years: int | None = None
    age_max_years: int | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    raw_name: str
    standard_name: str | None
    abbreviation: str | None
    value: str | Decimal | None
    unit: str | None
    reference_interval: ReferenceInterval
    report_flag: AbnormalFlag | None = None
    sample_type: str | None = None
    method: str | None = None


@dataclass(frozen=True, slots=True)
class NormalizedObservation:
    raw_name: str
    standard_name: str | None
    abbreviation: str | None
    value: Decimal | None
    unit: str | None
    lower: Decimal | None
    upper: Decimal | None
    lower_inclusive: bool
    upper_inclusive: bool
    report_flag: AbnormalFlag | None
    sample_type: str | None
    method: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_name": self.raw_name,
            "standard_name": self.standard_name,
            "abbreviation": self.abbreviation,
            "value": _decimal_text(self.value),
            "unit": self.unit,
            "reference_interval": {
                "lower": _decimal_text(self.lower),
                "upper": _decimal_text(self.upper),
                "lower_inclusive": self.lower_inclusive,
                "upper_inclusive": self.upper_inclusive,
            },
            "report_flag": self.report_flag.value if self.report_flag else None,
            "sample_type": self.sample_type,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class ReportComputationEvidence:
    value: Decimal | None
    unit: str | None
    computed_flag: AbnormalFlag | None
    errors: tuple[StructuredError, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "value": _decimal_text(self.value),
            "unit": self.unit,
            "computed_flag": self.computed_flag.value if self.computed_flag else None,
            "errors": [error.to_dict() for error in self.errors],
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    normalized: NormalizedObservation
    evidence: ReportComputationEvidence


@dataclass(frozen=True, slots=True)
class ReportMetadata:
    """Raw metadata; only :meth:`to_safe_log_dict` is suitable for logs."""

    report_id: str | None = None
    patient_name: str | None = None
    patient_identifier: str | None = None
    source_text: str | None = None
    patient_sex: str | None = None
    patient_age_years: int | None = None

    def to_safe_log_dict(self) -> dict[str, object]:
        return {
            "patient_sex": self.patient_sex,
            "patient_age_years": self.patient_age_years,
        }


class UnitConverter:
    """Explicit linear conversions keyed by exact source and target unit names."""

    def __init__(self, conversions: Mapping[tuple[str, str], Decimal]) -> None:
        self._conversions = dict(conversions)

    def convert(self, value: Decimal, source_unit: str, target_unit: str) -> Decimal | None:
        if source_unit == target_unit:
            return value
        factor = self._conversions.get((source_unit, target_unit))
        if factor is None or not factor.is_finite() or factor <= 0:
            return None
        return value * factor


def evaluate_observation(
    observation: Observation,
    *,
    target_unit: str | None = None,
    converter: UnitConverter | None = None,
) -> EvaluationResult:
    """Normalize one observation and produce deterministic comparison evidence.

    Invalid input is represented in ``evidence.errors`` and never receives a
    computed flag. Unit conversion is only available through an exact mapping.
    """

    errors: list[StructuredError] = []
    value = _parse_decimal(observation.value, "value", errors)
    lower = _parse_decimal(
        observation.reference_interval.lower, "reference_interval.lower", errors, required=False
    )
    upper = _parse_decimal(
        observation.reference_interval.upper, "reference_interval.upper", errors, required=False
    )
    if not observation.unit:
        errors.append(_error("missing_unit", "unit", "A unit is required for comparison."))
    if lower is None and upper is None:
        if not any(error.field.startswith("reference_interval") for error in errors):
            errors.append(_error("invalid_interval", "reference_interval", "At least one interval bound is required."))
    elif lower is not None and upper is not None and lower > upper:
        errors.append(_error("reversed_interval", "reference_interval", "Lower bound exceeds upper bound."))

    output_unit = observation.unit
    if target_unit and observation.unit and value is not None:
        if converter is None:
            errors.append(_error("incompatible_unit", "unit", "No explicit converter was supplied."))
        else:
            converted_value = converter.convert(value, observation.unit, target_unit)
            if converted_value is None:
                errors.append(_error("incompatible_unit", "unit", "Units do not have an explicit conversion."))
            else:
                value = converted_value
                output_unit = target_unit
                if lower is not None:
                    lower = converter.convert(lower, observation.unit, target_unit)
                if upper is not None:
                    upper = converter.convert(upper, observation.unit, target_unit)
                if lower is None or upper is None:
                    errors.append(_error("incompatible_unit", "unit", "Interval units cannot be converted."))

    normalized = NormalizedObservation(
        raw_name=observation.raw_name,
        standard_name=observation.standard_name,
        abbreviation=observation.abbreviation,
        value=value,
        unit=output_unit,
        lower=lower,
        upper=upper,
        lower_inclusive=observation.reference_interval.lower_inclusive,
        upper_inclusive=observation.reference_interval.upper_inclusive,
        report_flag=observation.report_flag,
        sample_type=observation.sample_type,
        method=observation.method,
    )
    computed = None if errors else _compare(normalized)
    if computed and observation.report_flag and computed != observation.report_flag:
        errors.append(_error("report_flag_conflict", "report_flag", "Reported flag differs from interval comparison."))
    return EvaluationResult(
        normalized=normalized,
        evidence=ReportComputationEvidence(value, output_unit, computed, tuple(errors)),
    )


def _parse_decimal(
    value: str | Decimal | None, field: str, errors: list[StructuredError], *, required: bool = True
) -> Decimal | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            errors.append(_error("invalid_value", field, "A finite numeric value is required."))
        return None
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        errors.append(_error("invalid_value", field, "A finite numeric value is required."))
        return None
    if not parsed.is_finite():
        errors.append(_error("non_finite_value", field, "NaN and infinity are not allowed."))
        return None
    return parsed


def _compare(observation: NormalizedObservation) -> AbnormalFlag:
    assert observation.value is not None and (observation.lower is not None or observation.upper is not None)
    below_lower = observation.lower is not None and (
        observation.value < observation.lower
        or (observation.value == observation.lower and not observation.lower_inclusive)
    )
    above_upper = observation.upper is not None and (
        observation.value > observation.upper
        or (observation.value == observation.upper and not observation.upper_inclusive)
    )
    if below_lower:
        return AbnormalFlag.LOW
    if above_upper:
        return AbnormalFlag.HIGH
    return AbnormalFlag.NORMAL


def _error(code: str, field: str, message: str) -> StructuredError:
    return StructuredError(code=code, field=field, message=message)


def _decimal_text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None
