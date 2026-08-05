"""Shared atomic SQLite storage mechanics for graph packages."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def write_graph_sqlite(
    path: Path,
    *,
    schema_sql: str,
    metadata_rows: Sequence[tuple[str, str]],
    node_rows: Iterable[Sequence[Any]],
    edge_rows: Iterable[Sequence[Any]],
    node_sql: str,
    edge_sql: str,
    integrity_error: type[Exception],
    integrity_message: str,
    foreign_key_message: str,
) -> None:
    """Write a graph SQLite file through a validated same-directory staging file."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".sqlite", delete=False
    ) as handle:
        staging = Path(handle.name)
    try:
        with sqlite3.connect(staging) as db:
            db.execute("PRAGMA foreign_keys = ON")
            db.executescript(schema_sql)
            db.executemany("INSERT INTO metadata VALUES (?, ?)", metadata_rows)
            db.executemany(node_sql, node_rows)
            db.executemany(edge_sql, edge_rows)
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise integrity_error(integrity_message)
            if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise integrity_error(foreign_key_message)
        staging.replace(path)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
