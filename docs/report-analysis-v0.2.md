# Report Analysis v0.2

`medical_kg_sourceprep.analysis` is a pure, deterministic adapter between the
structured report contract, reviewed rule contract, and evidence policy. It
does not contain analyte, disease, patient, book, page, or sample-value
special cases.

## Inputs and ordering

Call `analyze_report(report, rules, approved_book_registry=...)` with a mapping
of stable item keys to `Observation` objects (or a sequence of observations)
and `AnalysisRule` objects. Report items are sorted by item key and rules by
`(rule_id, version)`. Decimal comparison is delegated to
`evaluate_observation`, so invalid values, intervals, units, and reported-arrow
conflicts remain explicit structured errors.

Each rule is evaluated with three-valued semantics (`true`, `false`, or
`unknown`). Missing facts, unit mismatches, invalid report computations,
unsupported decision tables, missing anchors, missing BOOK citations, and
non-approved or missing `ReviewRecord` values produce required `EvidenceGap`
records. They never produce a medical claim.

REPORT snapshot construction is deferred until a rule is true, has no required
gap, and has a non-empty conclusion. Invalid or unknown report inputs therefore
return a serializable result with the original structured report errors,
predicate traces, required gaps, and zero claims/bundles. If strict REPORT
snapshot validation fails on the claim path, only the expected `ValueError` or
`TypeError` contract failure is converted to a stable required gap; unrelated
programming or runtime errors are not swallowed.

## Trace and claim boundary

`ComputationTrace` records the versioned algorithm, rule version, every atomic
predicate, its report input, normalized value/unit, result, anchor, and BOOK
citation. A claim is created only for a true rule with no gaps and a non-empty
conclusion. The generated `CitationBundle` contains an independent minimal
REPORT snapshot with canonical JSON and SHA-256, every required BOOK source,
and a `ComputationTraceRef.report_citation_id` that must also occur in
`citation_ids`. The validator requires approved rule status, REPORT/
COMPUTATION/BOOK-compatible links, every atomic condition's BOOK `TextAnchor`,
and stable report, manifest, anchor, and source hashes. Validator failure is
converted to no claim.

REPORT snapshots contain only the stable item key, normalized value, unit,
reference interval, and reported flag used by the rule. They exclude patient
names, identifiers, source text/images, and unrelated clinical context. Any
payload, version, identity, or hash drift fails closed, so a prior bundle
cannot be replayed against a changed report.

The snapshot schema is exact: source keys are `citation_id`, `source_type`,
`source_id`, `version`, `schema_version`, `payload`, and `payload_hash`;
payload keys are `schema_version` and `items`; item keys are `item_id`,
`value`, `unit`, `reference_interval`, and `report_flag`; interval keys are
`lower`, `upper`, `lower_inclusive`, and `upper_inclusive`. Unknown fields are
rejected by both construction and validation rather than dropped.

`result_to_dict` is JSON-friendly and preserves the distinction between report
facts, computations, rule evaluations, claims, and evidence gaps. It does not
serialize patient metadata because that metadata is not part of the analysis
input contract.

The module emits no diagnosis, treatment, causal assertion, or knowledge not
provided by the caller's approved rule and BOOK evidence. Conflicting sources
must be represented by a caller-supplied qualified rule/claim; otherwise the
strict evidence policy rejects the bundle.
