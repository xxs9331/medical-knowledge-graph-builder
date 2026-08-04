# Evidence Policy v0.2

This contract governs provenance for generated answers. It is a validation
boundary, not a medical knowledge source, retriever, calculator, or model.

## Three evidence kinds and three chains

- `REPORT` is an observed or imported fact. It may be displayed as a fact.
- `COMPUTATION` is a versioned trace showing how supplied conditions were
  combined. It does not create medical facts.
- `BOOK` is the allow-listed textual source for a medical condition or claim.

`REPORT` citations are immutable, minimal snapshots. They contain a canonical
JSON payload, `payload_hash`, stable `source_id`, and hash-based `version`.
Snapshot items contain only the stable item key, value, unit, reference
interval, and report flag used by computation; patient identifiers, names,
source text/images, and unrelated clinical context are excluded.

The exact REPORT source keys are `citation_id`, `source_type`, `source_id`,
`version`, `schema_version`, `payload`, and `payload_hash`. The exact payload
keys are `schema_version` and `items`; item keys are `item_id`, `value`,
`unit`, `reference_interval`, and `report_flag`; reference interval keys are
`lower`, `upper`, `lower_inclusive`, and `upper_inclusive`. Both the public
builder and direct validator reject missing or unknown keys before hashing.
They do not blacklist selected sensitive names or silently remove fields.

A medical claim must form all three links:

1. the caller supplies an approved BOOK provenance registry; a `source_type`
   declaration or non-empty `source_id` alone is never approval;
2. each BOOK citation resolves to one registry-approved manifest identity,
   version, and `content_sha256`, plus a verified `TextAnchor`;
3. each `AtomicPredicateRef` points to one fixed BOOK citation;
4. each `RuleMatchRef` maps that predicate to one approved, fixed
   `RuleVersionRef` and the claim;
5. a `ComputationTraceRef` binds every predicate and the claim, explicitly
   names `report_citation_id`, and includes that REPORT citation in
   `citation_ids`; matched BOOK citations anchor the conclusion.

Medical claims require at least one independently validated REPORT source,
one COMPUTATION binding to that source, and one or more approved BOOK sources.
Missing, forged, or drifted REPORT payload/hash/version data fails closed even
when the BOOK chain is complete. A changed report cannot reuse an old bundle.

The validator replays the supplied anchor against caller-provided raw and
cleaned synthetic text, checks the anchor hash and page hashes against the
approved manifest, and checks the citation source hash against the manifest's
PDF or Markdown identity. Missing registry entries, unknown anchors, or any
hash/version drift fail closed. The runtime must construct the registry from
approved manifests; this module does not discover or approve sources.

The validator requires one-to-one IDs and rejects missing edges, duplicate
citations, rule-version drift, unsupported claim types, or unknown required
conditions. A rule is usable for a patient-facing explanation only when its
status is exactly `approved` and its version is non-empty and fixed.

## Claim strength and failure closing

Missing BOOK evidence, a missing computation trace, an incomplete predicate
mapping, or an unapproved rule fails closed. Candidate, reviewed, and rejected
rules are not alternatives to approval. A report anomaly can be shown as a
`reported_fact`, but it cannot be upgraded into a medical explanation.

Index records, network responses, directories, task documents, and free-form
model text are not medical evidence. They cannot be used as BOOK citations.

Conflicts must be represented explicitly. A conflicted bundle is accepted only
with qualified or weaker wording (`qualified`, `cautious`, or `uncertain`); the
validator does not silently choose a source or strengthen a conclusion.
