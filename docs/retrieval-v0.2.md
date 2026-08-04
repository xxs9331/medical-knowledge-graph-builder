# Offline Retrieval v0.2

`medical_kg_sourceprep.retrieval` is an offline, caller-owned retrieval layer.
It accepts rule-like records through a small read-only protocol: `record_id` (or
`id`), `standard_name`, `title`, `text`, `rule_type`, `conditions`, and
`anchor_ids`. Existing or future `KnowledgeRule` objects can be passed without
changing their schema; dictionaries are also supported for boundary adapters.

Ranking is deterministic: exact standard-name/title phrase, caller-supplied
alias, title phrase, structured rule/condition filters, then SQLite FTS5 BM25.
If the local SQLite build lacks FTS5, a standard-library BM25 calculation is
used. Scores expose the same fixed components in every result. Exact names and
aliases outrank shared-token matches, and ties use `record_id`.

`top_k` limits only primary evidence. Parent/adjacent context is intentionally
not promoted over a primary match. A selected rule returns all of its declared
`anchor_ids`, so a compound rule cannot silently lose a required predicate or
conclusion. Empty/invalid queries, filter mismatches, and no reliable match
return an empty tuple.

Aliases are caller data used for discovery; they are not a medical claim,
approval, or evidence source. Index/directory metadata can improve retrieval
recall but must not be treated as the anchor supporting a claim. This module
does not access the network, a model, vector storage, book text, or an index
owned by another module. A future vector channel is optional and must preserve
the existing result contract, score decomposition, deterministic ordering, and
evidence-first boundary.
