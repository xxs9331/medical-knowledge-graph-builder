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

## Terminology standards and references

The following sources define the external terminology baseline for future report
normalization and graph alignment. Listing a source here does not mean that its
codes have already been imported into Neo4j or approved as project gold data.

1. National Health Commission of the People's Republic of China. *WS/T
   886-2026 Names and codes of common clinical laboratory tests* (published
   2026-05-25; effective 2026-11-01). The table defines 399 common items with a
   code, test name, category, analyte, specimen type, and scale. Use it as the
   primary Chinese baseline for common atomic test names after it takes effect;
   do not treat its category or specimen columns as local order-panel definitions.
   [Official publication and attachment](https://wjw.fujian.gov.cn/jggk/csxx/jhsyjtfzczcfgc/gzdt_37901/202606/t20260618_7165013.htm)

2. National Health Commission of the People's Republic of China. *WS/T
   224-2018 Performance verification of evacuated tubes for blood specimen
   collection*. This standard governs collection tubes and must not be used as
   the terminology authority for `NEUT#`; it is retained only as a specimen
   collection reference.
   [Official PDF](https://www.nhc.gov.cn/ewebeditor/uploadfile/2018/05/20180516135041401.pdf)

3. National Health Commission of the People's Republic of China. *WS/T
   779-2021 Reference intervals for blood cell analysis in Chinese children*.
   It separately lists `中性粒细胞绝对值（Neut#）` in `x10^9/L` and
   `中性粒细胞百分数（Neut%）` in percent. Reference intervals remain
   population- and source-specific facts and must not be copied into a universal
   indicator definition.
   [Official PDF](https://www.nhc.gov.cn/wjw/s9492/202105/a85d8b64e0384c98aed8f3157860ee44/files/1739781618961_15816.pdf)

4. National Health Commission of the People's Republic of China. *Recommended
   adult health examination items (2025 edition)*. Use its blood count,
   urinalysis, stool examination, liver function, kidney function, lipid,
   glucose, uric acid, homocysteine, and thyroid function groupings as a
   health-examination scope reference, not as proof that every institution uses
   an identical panel membership.
   [Official notice](https://www.nhc.gov.cn/yzygj/c100068/202511/4feaeb6de63e44cb9fd4ac8f579ca279.shtml)

5. National Health Commission of the People's Republic of China. *Catalogue of
   clinical laboratory tests for medical institutions* (2007). Retain this as a
   historical breadth reference; it predates WS/T 886-2026 and must not override
   newer names or local verified report mappings.
   [Official notice and catalogue](https://www.nhc.gov.cn/zwgk/wtwj/201304/4eb89ebf61e146f8bf49fa01873a5859.shtml)

6. Regenstrief Institute. *LOINC 2.82* (released 2026-02-24). `Loinc.csv`
   supplies observation and order identifiers; `PanelsAndForms.csv` supplies
   panel membership and required/optional/conditional structure; the Part file
   supplies components including systems/specimen types. Use LOINC as an
   interoperability mapping, while preserving the Chinese source name and local
   code because local panels may not match a standard panel exactly.
   [Download contents](https://loinc.org/downloads/),
   [Panels and Forms file](https://loinc.org/kb/users-guide/additional-content-in-the-loinc-distribution/panels-and-forms-file),
   [panel mapping guidance](https://loinc.org/kb/users-guide/panels/)

7. HL7 International. *FHIR R5 DiagnosticReport, Observation, and Specimen*.
   The model separates the report, grouped or atomic results, and specimen. Use
   this separation as the design reference for keeping `全血`, `血清`, and `血浆`
   as report/indicator specimen facts rather than treating them as fixed
   laboratory panels.
   [DiagnosticReport](https://hl7.org/fhir/diagnosticreport.html),
   [Observation](https://hl7.org/fhir/observation.html),
   [Specimen](https://hl7.org/fhir/specimen.html)

Project adoption boundary:

- canonical indicators must distinguish `中性粒细胞绝对计数` from
  `中性粒细胞百分数`;
- report aliases such as `NEUT` are resolved using the observed unit, specimen,
  result scale, and available panel context, and remain ambiguous when those
  facts are insufficient;
- units and specimen types are structured attributes of indicator definitions
  and report observations; a graph node may be added when relation-based search
  requires it, but the source fact must be retained either way;
- standard and local panels are versioned collections of indicators. No source
  above is treated as a permanent exhaustive list of every institution's panel;
- importing or mapping any external term requires a separate provenance-bearing
  artifact and automated validation gate before Neo4j publication.
