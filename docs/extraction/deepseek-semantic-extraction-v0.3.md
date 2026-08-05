# DeepSeek Semantic Extraction v0.3

`medical_kg_sourceprep.extraction.llm_extraction` is a candidate-only adapter for the
OpenCode Go chat-completions endpoint. Its fixed provider identity is
`opencode-go/deepseek-v4-flash`; the API payload uses `deepseek-v4-flash`,
temperature zero, JSON-object response format, and a bounded token budget.
The default remains low reasoning, 8192 tokens, and 9000 input characters.
Passing `reasoning_effort=None` or the CLI `--disable-reasoning` flag omits the
reasoning effort, sends `thinking={"type":"disabled"}`, records
`reasoning_mode=disabled` and `thinking_mode=disabled`, and gives the
checkpoint a distinct configuration identity. Credentials come from
`OPENCODE_GO_API_KEY` or the local OpenCode auth file. There is no CLI key
argument and secrets are not included in errors, attempts, checkpoints, or
outputs.

The v0.6 prompt uses a compact deterministic key contract rather than embedding
the full Draft schema. It lists all six model entity types, seven endpoint
relations, semantic types, subject logics, and per-window output limits
(`entities<=24`, `rules<=12`, `relations<=36`). The model emits only endpoint
relations; it must not emit `SourceLocator` or the three `*_SUPPORTED_BY` edges,
which are added deterministically by the local builder. Each `exact_quote` must
occur verbatim exactly once in its specified chunk, and each candidate `text`
must occur exactly once inside that quote. Composite review payloads use
`conditions`/`conclusion` objects of `{text,source_ref}` and a separate exact
connector `{operator,text,source_ref}`; `SINGLE` may omit review payload and has
no connector, while `AT_LEAST` is only a connector operator with `at_least`.

The input is a hash-verified `EvidenceChunk` sequence sorted deterministically
by page and offset. Windows never cross chapter boundaries and never split a
chunk. A non-placeholder `chapter_id` is authoritative. When all ids are a
single placeholder (for example `full-book`), the adapter recovers boundaries
from first-level Markdown headings in chunk text; optional exact Chapter title
anchors can normalize those headings. Chunks before the first heading form an
isolated `prologue` group. If a single
indivisible chunk contains leading text followed by its first heading, the
whole chunk is assigned to the new heading because the chunk cannot be split;
this is explicit and deterministic. More than one first-level heading in one
indivisible chunk raises an explicit ambiguity error rather than silently
crossing a chapter boundary. Each
model value must carry `{chunk_id, chunk_sha256, exact_quote}`. The local
validator checks the hash, requires one and only one exact occurrence, computes
the character offsets itself, rejects unknown fields, dangling relations,
invalid fixed entity/relationship types, invalid rule semantics, and every
status other than `candidate`. The strict validator remains an atomic
compatibility interface; v0.6 also offers a partial validator that rejects
individual records with only a stable item hash, index, kind, and bounded reason.
Top-level shape and output-limit failures remain atomic, and rejected endpoints
cannot re-enter through relations. Model output and BOOK text are never treated
as approved knowledge or executable rules.

`SemanticExtractor.extract` stores only validated window results. A checkpoint
is replaced through same-directory temporary-file rename and can resume a
window only when its input hash, top-level manifest hash, and configuration
identity are unchanged. The identity includes model, API model, prompt version,
reasoning effort, max tokens, and max chars. Each
window has one shared initial-plus-two-retry budget across request, parsing,
and local schema validation. HTTP 429, timeout, and 5xx
responses have at most two bounded retries; empty content, malformed JSON,
schema failure, 4xx errors, and length truncation fail closed. Tests inject a
transport/client and use synthetic non-medical text, so the test suite makes no
real model or network request. `--probe-windows N` selects a bounded prefix;
rerunning without the limit continues the same checkpoint to full coverage.
Final metadata records selected/total and reused windows, input and prompt
hashes, configuration identity, attempt classifications, client/API parameters,
version probe, and
record-level counts. An explicit `{entities: [], rules: [], relations: []}` is
recorded as `no_candidates`, while a mixed accepted/rejected window is
`partial_success`. A window with zero accepted records is `no_candidates` even
when rejections are recorded. `{}` or a missing top-level array fails closed
and writes a failed checkpoint. Every attempt retains only bounded
model/finish/token diagnostics, including reasoning token counts; response
content and reasoning text are never persisted.

The `parameters` object is the audit snapshot for the selected request: it
records the fixed model/API model, temperature, JSON-object response format,
reasoning effort, max tokens, and max input characters. The same snapshot is
stored on each window attempt, so a result cannot claim a different budget from
the one passed to the client. Changing any of these result-affecting settings
changes `config_identity` and invalidates prior successful window reuse.

## Chapter 01 relation/rule supplement v0.3

The chapter supplement is implemented in `semantic_v03.py` and
`scripts/run_chapter_semantic_v03.py`. It reads only the frozen v0.2
candidate entities, writes a page-scoped `entity-catalog.json`, and runs
independent relation-only and rule-only requests. Relation endpoints must be
copied from that catalog; rule components and endpoint keys are replayed
verbatim. Model and deterministic relations are merged by a stable page-local
triple while retaining multiple evidence rows and an explicit `origin`.

The runner uses the official DeepSeek endpoint, temperature zero, disabled
thinking, proxy-free HTTPS, and a hidden stdin key. It writes separate stage
checkpoints bound to the chunk manifest, catalog, prompt, validator, model,
and provider configuration. `--catalog-only` is the offline preparation mode;
it does not claim either model stage completed. All artifacts remain
`candidate-only`/`HOLD` with `approved=0`.
