# Offline Retrieval v0.3

`vector_index` builds an atomic SQLite index from an existing evidence index.
It uses only Unicode NFKC character 2-3 gram TF-IDF with L2-normalized sparse
vectors. This is lexical sparse retrieval, not a neural embedding or medical
semantic model. The index stores its vectorizer parameters, dimensions, chunk
manifest hash, and every chunk ID/hash. Query vectors use the same
`count * (log((N + 1) / (df + 1)) + 1)` formula and L2 normalization as
document vectors, where `N` is the strictly bound vector/chunk count. At query
time every persisted vector is rejected with `VectorIndexError` unless it is a
JSON object with unique canonical decimal dimensions in range, finite
non-boolean numeric non-negative weights, and a unit L2 norm within recorded
precision tolerance. Binding or payload drift fails closed.

`graph_retrieval` opens `knowledge.sqlite` read-only, verifies integrity and
the v0.2 schema, traverses existing edges only, then projects graph
`EvidenceChunk` content onto evidence chunks only where the same `chunk_id`
exists in the evidence index and its content hash is an exact match. This keeps
identical text from distinct pages bound to their own evidence chunk; a returned
graph chunk with a missing ID is unmatched, while a same-ID hash drift fails
closed. It returns path relations and anchors encountered on that path. No graph
relation is itself medical evidence, and unmatched or dangling paths produce no
candidate.

`retrieve_hybrid` accepts lexical records plus results from those two channels.
It exposes `vector` and `graph` score components and stable channel reasons.
Exact names and aliases retain their fixed priority, auxiliary hits are bounded
by `top_k`, and a vector-only record requires the common similarity threshold.
All returned anchors remain caller-provided evidence anchors.
