# Source Provenance v0.2

`medical_kg_sourceprep.provenance.book_sources` defines a metadata-only contract for a book,
its original PDF identity, an existing OCR Markdown input, cleaned pages, chunks,
and quote anchors. It does not read PDF content, OCR documents, or write source
text into the manifest.

## Records

`SourceBook`, `SourcePdf`, and `SourceMarkdown` use stable identifiers and a
lowercase SHA-256 for every source file. `CleanedPage` retains distinct printed
page and source-PDF page values, source Markdown line boundaries, raw and cleaned
file hashes, and its audit status. `TextAnchor` stores both raw and cleaned
character spans, the exact quote, surrounding prefix/suffix context, both text
hashes, page values, and line boundaries.

`build_book_manifest` accepts only these page fields and a narrow, stable chunk
projection (`chunk_id`, `page_id`, cleaned character range, and chunk hash). Its
content hash is canonical JSON over the complete contract excluding the hash
field itself. It therefore has no timestamp and is stable for identical input.
`build_book_manifest_from_packages` is the v0.1 adapter: it consumes already
loaded source/chunk manifest records and does not open page, chunk, Markdown, or
PDF files. Missing source-line metadata is represented as `null`, not inferred.

## Status Semantics

`verified against source PDF` means a reviewer compared the mapping to the source
PDF. `accepted from upstream page markers` preserves an upstream assertion and
does not become verified. `unreviewed` is explicitly unknown. PDF page numbers
and printed page numbers remain independent. PDF bounding boxes are `unavailable`
in v0.2; callers must not fabricate coordinates.

## Failure Closure

Anchor creation and replay require raw and cleaned spans to produce exactly the
same quote. Replay checks both text hashes, offsets, quote context, and uniqueness;
text changes, offset drift, missing context, or ambiguous repeated quotes raise
`SourceProvenanceError`. Manifest validation reconstructs the deterministic hash
chain and fails on hash drift, invalid review statuses, missing/duplicate page
indexes, unknown chunk pages, or invalid stable IDs.
