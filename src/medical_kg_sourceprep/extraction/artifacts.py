"""Shared JSON, hashing, and atomic artifact helpers for pipeline scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    staging.write_text(value, encoding="utf-8")
    staging.replace(path)


def directory_sha256(path: Path, *, exclude_names: set[str] | frozenset[str] = frozenset()) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.name not in exclude_names:
            digest.update(str(item.relative_to(path)).encode())
            digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()
