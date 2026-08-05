# Semantic Graph v0.3

`semantic_graph` adds a conservative semantic package to a copied
`knowledge-graph/v0.2` SQLite database. It never opens a PDF, runs OCR, calls a
model, fetches a network resource, or reads medical reference material.

## Input and atomicity

`SemanticGraphBuilder.build(output, base_graph, book_manifest, records, relations)`
requires a v0.2 graph whose Book manifest hash and page-local EvidenceChunk text
match the supplied explicit chunk package. The output is a same-directory staging
copy. Validation, SQLite integrity, and foreign-key checks precede replacement;
the base graph and a failed output path remain unchanged.

Every semantic record has a hash-bound source anchor containing its chunk ID,
chunk SHA-256, printed and source-PDF pages, local offsets, and exact quote. The
same anchor is written to the main graph. Thus schema material is never evidence:
only a quote already present in a bound BOOK chunk may be imported.

## Candidate projection

Projection is deterministic and format-driven. A Markdown chapter heading resets
the context. A Chinese numbered subsection (`1. Title`, `1、Title`, or
`一、Title`) is a tentative TestItem only. It becomes a candidate TestItem only
when that subsection, including a continuation in the next EvidenceChunk, has a
`【参考区间】` label before the next numbered subsection or chapter. The reference
value may be on the label line, on the next non-empty line, or in a complete
table block. Each record remains anchored to the single chunk containing its
quoted title or range.

The legacy explicit `【检验项目】` form remains recognized as the same tentative
format marker. A reference label without a tentative item creates no semantic
fact. `【异常结果解读】` merely marks prose evidence; it does not discard another
valid structure in the chunk, and never creates an InterpretationRule,
MedicalConcept, causal edge, or approval.

## Fixed model and main graph

Entity types are `TestItem`, `TestMethod`, `ReferenceRange`,
`InterpretationRule`, `MedicalConcept`, `Population`, and `SourceLocator`.
The only semantic edge names are the ten fixed relations in the approved
contract. Their source and target types are checked before insertion. Rules
additionally require one of the six fixed semantic types and `SINGLE`, `ALL`, or
`ANY`; a rule may have only one `RULE_HAS_CONCLUSION` edge.

For each TestItem, ReferenceRange, and InterpretationRule, the builder creates a
separate candidate SourceLocator and its required `*_SUPPORTED_BY` relation. All
semantic records and all ten allowed relations are mirrored into the copied main
`nodes` and `edges` tables, with `edge_kind = semantic`. Each SourceLocator has
the structural provenance edge `SOURCE_LOCATOR_TARGETS_CHUNK` to its exact
EvidenceChunk and a main-graph anchor. This structural edge is deliberately not
one of the ten semantic relations. It gives existing `graph_retrieve` a path of
at most three hops from an item, range, or rule to hash-bound evidence.

Statuses are `candidate`, `reviewed`, `approved`, and `rejected`. Import and
projection never promote a record. An `approved` record needs a non-empty
reviewer, time, rationale, and fixed source version in `ReviewRecord`; generated
SourceLocators remain candidates and are non-executable.

## Content binding

`semantic_metadata` stores the semantic package version, base manifest hash,
record and relation counts, and `source_package_hash`. The hash covers the
normalized semantic records, their reviews, normalized relations, semantic schema
version, and BOOK manifest hash. Record order does not change the hash; changing
any record, relation, review, schema version, or manifest binding does.
