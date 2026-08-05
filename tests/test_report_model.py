import unittest
from decimal import Decimal

from medical_kg_sourceprep.report_model import (
    AbnormalFlag,
    Observation,
    ReferenceInterval,
    ReportMetadata,
    UnitConverter,
    evaluate_observation,
)


def _observation(**overrides: object) -> Observation:
    values: dict[str, object] = {
        "raw_name": "Synthetic measure",
        "standard_name": "synthetic_measure",
        "abbreviation": "SM",
        "value": "5.0",
        "unit": "U",
        "reference_interval": ReferenceInterval(lower="2", upper="5", upper_inclusive=True),
        "report_flag": AbnormalFlag.NORMAL,
        "sample_type": "synthetic",
        "method": "synthetic method",
    }
    values.update(overrides)
    return Observation(**values)  # type: ignore[arg-type]


class ReportModelTests(unittest.TestCase):
    def test_deterministic_evaluation_keeps_facts_and_evidence_separate(self) -> None:
        observation = _observation(value="5.00")
        first = evaluate_observation(observation)
        second = evaluate_observation(observation)

        self.assertEqual(first, second)
        self.assertEqual(first.normalized.value, Decimal("5.00"))
        self.assertEqual(first.normalized.unit, "U")
        self.assertEqual(first.evidence.computed_flag, AbnormalFlag.NORMAL)
        self.assertEqual(first.evidence.errors, ())
        self.assertNotIn("computed_flag", first.normalized.to_dict())
        self.assertEqual(first.evidence.to_dict()["value"], "5.00")

    def test_computes_high_low_and_open_closed_boundaries(self) -> None:
        cases = [
            ("6", ReferenceInterval(lower="2", upper="5"), AbnormalFlag.HIGH),
            ("1", ReferenceInterval(lower="2", upper="5"), AbnormalFlag.LOW),
            ("2", ReferenceInterval(lower="2", upper="5"), AbnormalFlag.NORMAL),
            ("2", ReferenceInterval(lower="2", upper="5", lower_inclusive=False), AbnormalFlag.LOW),
            ("5", ReferenceInterval(lower="2", upper="5", upper_inclusive=False), AbnormalFlag.HIGH),
            ("1.25e1", ReferenceInterval(lower="0.1", upper="12.5"), AbnormalFlag.NORMAL),
            ("6", ReferenceInterval(lower=None, upper="5"), AbnormalFlag.HIGH),
            ("1", ReferenceInterval(lower="2", upper=None), AbnormalFlag.LOW),
            ("3", ReferenceInterval(lower="2", upper=None), AbnormalFlag.NORMAL),
        ]
        for value, interval, expected in cases:
            with self.subTest(value=value, interval=interval):
                outcome = evaluate_observation(
                    _observation(value=value, reference_interval=interval, report_flag=None)
                )
                self.assertEqual(outcome.evidence.computed_flag, expected)
                self.assertFalse(outcome.evidence.errors)

    def test_invalid_values_and_intervals_fail_closed_with_structured_errors(self) -> None:
        cases = [
            (_observation(value=""), "invalid_value"),
            (_observation(value="NaN"), "non_finite_value"),
            (_observation(value="Infinity"), "non_finite_value"),
            (_observation(reference_interval=ReferenceInterval(lower="6", upper="2")), "reversed_interval"),
            (_observation(reference_interval=ReferenceInterval(lower=None, upper=None)), "invalid_interval"),
        ]
        for observation, code in cases:
            with self.subTest(code=code):
                outcome = evaluate_observation(observation)
                self.assertIsNone(outcome.evidence.computed_flag)
                self.assertEqual(outcome.evidence.errors[0].code, code)

    def test_missing_unit_allows_only_same_row_interval_comparison(self) -> None:
        outcome = evaluate_observation(_observation(
            value="5.15",
            unit=None,
            reference_interval=ReferenceInterval(lower="3.8", upper="5.1"),
            report_flag=None,
        ))

        self.assertEqual(outcome.evidence.computed_flag, AbnormalFlag.HIGH)
        self.assertEqual([error.code for error in outcome.evidence.errors], ["missing_unit"])
        self.assertIsNone(outcome.normalized.unit)

        conversion = evaluate_observation(
            _observation(value="5.15", unit=None), target_unit="10^12/L"
        )
        self.assertIsNone(conversion.evidence.computed_flag)
        self.assertEqual(conversion.evidence.errors[0].code, "missing_unit")

    def test_report_arrow_conflict_is_preserved(self) -> None:
        outcome = evaluate_observation(_observation(value="6", report_flag=AbnormalFlag.LOW))

        self.assertEqual(outcome.evidence.computed_flag, AbnormalFlag.HIGH)
        self.assertEqual(outcome.evidence.errors[0].code, "report_flag_conflict")
        self.assertEqual(outcome.normalized.report_flag, AbnormalFlag.LOW)

    def test_explicit_converter_can_change_unit_but_incompatible_unit_fails(self) -> None:
        converter = UnitConverter({("U", "mU"): Decimal("1000")})
        converted = evaluate_observation(_observation(value="1.5"), target_unit="mU", converter=converter)
        incompatible = evaluate_observation(_observation(), target_unit="other", converter=converter)

        self.assertEqual(converted.normalized.value, Decimal("1500.0"))
        self.assertEqual(converted.normalized.unit, "mU")
        self.assertEqual(incompatible.evidence.errors[0].code, "incompatible_unit")

    def test_safe_log_projection_excludes_identifiers_and_free_text(self) -> None:
        metadata = ReportMetadata(
            report_id="report-123",
            patient_name="Example Person",
            patient_identifier="patient-456",
            source_text="free text diagnostic context",
            patient_sex="female",
            patient_age_years=42,
        )
        projection = metadata.to_safe_log_dict()
        serialized = str(projection)

        self.assertEqual(projection, {"patient_sex": "female", "patient_age_years": 42})
        for secret in ("report-123", "Example Person", "patient-456", "free text diagnostic context"):
            self.assertNotIn(secret, serialized)
