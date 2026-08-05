# Structured Report Domain v0.2

`medical_kg_sourceprep.report.report_model` defines a small, versionable input contract
for structured measurement reports. It is deliberately not a diagnosis, cause,
treatment, or follow-up recommendation API.

## Input contract

Create an `Observation` with the source item name and optional standard name or
abbreviation, a numeric value, an exact unit, and a `ReferenceInterval`. Numeric
inputs accept decimal strings (including scientific notation) or `Decimal`.
Intervals support optional sex and age conditions as source facts, plus inclusive
or exclusive lower and upper boundaries.

```python
from medical_kg_sourceprep.report.report_model import Observation, ReferenceInterval, evaluate_observation

result = evaluate_observation(
    Observation(
        raw_name="source item",
        standard_name="standard_item",
        abbreviation="SI",
        value="1.25e1",
        unit="U",
        reference_interval=ReferenceInterval(lower="0.1", upper="12.5"),
        sample_type="source sample",
        method="source method",
    )
)
```

`result.normalized` is a report fact. `result.evidence` is a separate,
deterministic comparison record. Decimals are never compared through binary
floating point. `report_flag` remains the source-reported arrow; a disagreement
with `computed_flag` is retained as `report_flag_conflict`.

## Errors and units

Invalid values, `NaN`, infinity, absent interval bounds, reversed bounds, and a
missing unit return `computed_flag=None` with `StructuredError` values in
`evidence.errors`. This is fail-closed: callers must not treat an error result
as a normal comparison.

Units are preserved exactly by default. `UnitConverter` only converts an exact
`(source_unit, target_unit)` pair explicitly supplied by the caller. It performs
linear same-dimension conversions and refuses missing or incompatible mappings;
the module does not infer conversions from similar unit strings.

## Privacy boundary

`ReportMetadata` may receive raw report identifiers, patient identifiers,
patient names, or source text for in-memory processing. Do not serialize that
object to logs. Use `to_safe_log_dict()` instead; it projects only structured,
non-direct patient attributes (`patient_sex` and `patient_age_years`) and omits
identifiers and free text.
