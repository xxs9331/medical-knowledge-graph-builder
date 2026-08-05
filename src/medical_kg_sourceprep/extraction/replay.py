"""Shared replay primitives for hash-bound extraction references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ChunkReplayError(ValueError):
    """A chunk reference cannot be replayed against the supplied input."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ReplayedChunkQuote:
    chunk_id: str
    chunk_sha256: str
    exact_quote: str
    char_start: int
    char_end: int


def replay_chunk_quote(value: Any, chunks: Mapping[str, Any]) -> ReplayedChunkQuote:
    """Validate a chunk id/hash/unique quote and return its exact character span."""
    if not isinstance(value, Mapping):
        raise ChunkReplayError("reference_missing")
    chunk_id = value.get("chunk_id")
    digest = value.get("chunk_sha256")
    quote = value.get("exact_quote")
    if not all(isinstance(item, str) and item for item in (chunk_id, digest, quote)):
        raise ChunkReplayError("reference_fields_missing")
    chunk = chunks.get(chunk_id)
    if chunk is None or digest != chunk.chunk_sha256:
        raise ChunkReplayError("hash_drift")
    start = chunk.text.find(quote)
    if start < 0 or chunk.text.count(quote) != 1:
        raise ChunkReplayError("quote_absent_or_ambiguous")
    return ReplayedChunkQuote(chunk_id, digest, quote, start, start + len(quote))
