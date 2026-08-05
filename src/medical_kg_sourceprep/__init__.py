"""Page-aware source preparation for downstream consumers."""

from .provenance.prepare import PreparationError, prepare_source

__all__ = ["PreparationError", "prepare_source"]
