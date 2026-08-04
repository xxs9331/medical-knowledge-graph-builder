# Knowledge Graph v0.2

`medical_kg_sourceprep.knowledge_graph` builds a local, deterministic SQLite graph
from a validated `book-source-manifest/v0.2` and caller-supplied `PageText` values.
It does not open PDFs, run OCR, call models, retrieve network content, or infer
domain-specific meaning.

## Input And Safety

`KnowledgeGraphBuilder.build(path, manifest, pages)` requires one `PageText` for
every manifest page. Raw and cleaned text may differ only when each parsed cleaned
span can be mapped to exactly one raw occurrence using its exact quote and retained
surrounding context. The resulting `TextAnchor` stores independent raw and cleaned
offsets, then uses the Foundation replay validator against both texts. This permits
deletion-only cleaning such as verified furniture removal without treating raw
offsets as cleaned offsets.

Page hashes, chunk hashes, contiguous chunk spans, unique IDs, manifest hash chains,
and SQLite foreign keys are validated before the staging database replaces the
requested path. A rewritten quote, a deletion inside an anchored span, repeated or
context-ambiguous raw text, context crossing a removed region, or either hash drift
fails closed. Any failure removes the staging database and leaves no target database.

`EvidenceChunk` text is retained with its page-local offsets, so
`read_graph(path).reconstruct_page(page_id)` returns the original page text exactly.
Derived nodes never mutate this text.

## Graph Contract

Nodes include `Book`, `Chapter`, `Section`, `Page`, `EvidenceChunk`, `TestItem`,
`KnowledgeRule`, `RuleVersion`, `RuleExpression`, and `AtomicPredicate`; table,
table-cell, formula, reference-range, explanation, and conclusion nodes are added
when explicit source syntax permits. `edges.edge_kind` is either `structural` or
`semantic`, so navigation edges cannot be confused with evidence semantics.

The builder writes `BOOK_HAS_CHAPTER`, `CHAPTER_HAS_SECTION`,
`SECTION_HAS_TEST_ITEM`, `TEST_ITEM_HAS_RULE`, `RULE_HAS_VERSION`,
`RULE_HAS_EXPRESSION`, `EXPRESSION_HAS_PREDICATE`, `PREDICATE_SUPPORTED_BY`,
`CONCLUSION_SUPPORTED_BY`, `CHUNK_ON_PAGE`, and `CHUNK_NEXT` where applicable.
Every parsed chapter/section title, item, atomic predicate, conclusion, labeled
evidence, formula, and table cell gets its own Foundation-compatible `TextAnchor`;
tables additionally persist title, row, column, header status and a cell anchor
projection.

## Stable Internal IDs

Chapter, section, and other derived node ID components normalize title text with
Unicode NFKC, retain an ASCII-readable projection where available, and append a
stable SHA-256 digest. Heading nodes also include their deterministic page position
and heading ordinal, so distinct Unicode titles cannot collapse after ASCII
projection and repeated equal titles remain distinct. These identifiers are internal
keys only: the original Unicode title remains in the node payload and its
`TextAnchor`, which are the source-facing representations.

## HTML Table Cells

Table cells retain two distinct values. `TableCell.text` and `table_cells.text`
hold the safely decoded display text, while the cell `TextAnchor.exact_quote` is
the original inner-HTML lexical span, including entity spelling and simple inline
tags. The parser records that span while walking table rows, so equal display values
in separate cells receive their own deterministic offsets instead of a first-match
search. Missing, rewritten, or context-ambiguous source spans fail closed; malformed
tables remain evidence-only and do not produce table-cell anchors.

## Conservative Parsing And Review

The deterministic Markdown labels are `Test Item:`, `Reference Range:`,
`Explanation:`, `Formula:`, and `Rule: IF <condition> AND <condition> THEN
<conclusion>`. HTML tables and `\\[ ... \\]` or `$$ ... $$` formula blocks are also
preserved when structurally complete. Missing a `Test Item:` subject, incomplete
tables, or ambiguous/unanchorable source content retains the page chunk evidence but
does not create a rule.

All generated rules begin as `candidate`. The only status transition API is
`read_graph(path).set_rule_status(rule_id, "approved", review_record="...")`; a
non-empty explicit review record is required. Consumers must treat the graph as
read-only and must continue to require approved rule versions through
`evidence_policy` before any medical explanation is supported.
