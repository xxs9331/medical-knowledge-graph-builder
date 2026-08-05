# Composite Rules v0.2

`medical_kg_sourceprep.composite_rules` is a generic, provenance-first contract
for extracted multi-condition candidates. It contains no disease, analyte,
book, page, or sample-value vocabulary.

## Contract

- `TextAnchor` is required for every atomic predicate, logic connection,
  threshold, decision row, table, and output binding represented by the
  contract. Missing anchors create validation issues and prevent execution.
- Atomic operators are `lt`, `le`, `eq`, `ge`, `gt`, `between`, `in`,
  `positive`, and `negative`. Numeric units are explicit; there is no implicit
  conversion.
- Logic is `all`, `any`, `not`, `at_least`, and `at_most`. Evaluation returns
  `true`, `false`, or `unknown`. Missing facts remain `unknown`, including when
  the result is not sufficient to establish a threshold.
- Decision tables require `UNIQUE` or `COLLECT`; first-match behavior is not a
  valid policy. Validation reports missing coverage, overlaps, conflicts,
  ambiguous UNIQUE rows, missing policies, and missing anchors.
- Extraction is always `candidate`. An explicit `ReviewRecord` containing a
  reviewer, decision, version, and rationale is required before a candidate
  can become `reviewed`, `approved`, or `rejected`. The extractor cannot call
  `approve` itself.

## Detection boundary

`detect_composite_candidates` recognizes only supplied structural metadata:
multiple tests with a language operator, decision-table rows, and statements
describing joint testing. The latter is classified as
`JOINT_TESTING_STATEMENT` and is intentionally non-executable. It does not
perform OCR, LLM inference, network retrieval, or automatic medical meaning
selection.

The tests use synthetic predicates and values only. Authorized source checks
must remain bounded read-only operations that report structure counts and
anchor/page coverage without copying source text into this repository.
