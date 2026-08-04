"""Normalize document-layout table blocks into provenance-addressable grids."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class LayoutBlock:
    page_index: int
    block_id: int
    label: str
    content: str
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True, slots=True)
class GridCell:
    text: str
    source_ref: str


@dataclass(frozen=True, slots=True)
class TableGrid:
    page_index: int
    block_id: int
    rows: tuple[tuple[GridCell, ...], ...]


def layout_blocks(records: Sequence[Mapping[str, Any]]) -> tuple[LayoutBlock, ...]:
    """Read ordered blocks from the official PaddleOCR ``prunedResult`` contract."""
    blocks: list[LayoutBlock] = []
    page_index = 0
    for record in records:
        result = record.get("result")
        pages = result.get("layoutParsingResults") if isinstance(result, Mapping) else None
        if not isinstance(pages, list):
            continue
        for page in pages:
            pruned = page.get("prunedResult") if isinstance(page, Mapping) else None
            raw_blocks = pruned.get("parsing_res_list") if isinstance(pruned, Mapping) else None
            if not isinstance(raw_blocks, list):
                page_index += 1
                continue
            ordered = sorted(
                (item for item in raw_blocks if isinstance(item, Mapping)),
                key=lambda item: _order(item),
            )
            for fallback_id, item in enumerate(ordered):
                content = item.get("block_content")
                label = item.get("block_label")
                if not isinstance(content, str) or not isinstance(label, str):
                    continue
                raw_id = item.get("block_id")
                block_id = (
                    raw_id
                    if isinstance(raw_id, int) and not isinstance(raw_id, bool)
                    else fallback_id
                )
                blocks.append(
                    LayoutBlock(
                        page_index,
                        block_id,
                        label,
                        content,
                        _bbox(item.get("block_bbox")),
                    )
                )
            page_index += 1
    return tuple(blocks)


def table_grids(blocks: Sequence[LayoutBlock]) -> tuple[TableGrid, ...]:
    """Normalize HTML and pipe-Markdown table blocks into one rectangular contract."""
    grids: list[TableGrid] = []
    for block in blocks:
        if block.label != "table" or not block.content.strip():
            continue
        parser = _HtmlGridParser(block.page_index, block.block_id)
        parser.feed(block.content)
        parser.close()
        if parser.grids:
            grids.extend(parser.grids)
            continue
        rows = _markdown_rows(block.content, block.page_index, block.block_id)
        if rows:
            grids.append(TableGrid(block.page_index, block.block_id, rows))
    return tuple(grids)


def block_text(blocks: Sequence[LayoutBlock], *, include_tables: bool = False) -> tuple[str, ...]:
    return tuple(
        block.content.strip()
        for block in blocks
        if block.content.strip() and (include_tables or block.label != "table")
    )


class _HtmlGridParser(HTMLParser):
    def __init__(self, page_index: int, block_id: int) -> None:
        super().__init__(convert_charrefs=True)
        self.page_index = page_index
        self.block_id = block_id
        self.grids: list[TableGrid] = []
        self._depth = 0
        self._rows: list[list[GridCell | None]] | None = None
        self._row_index = -1
        self._column = 0
        self._cell_parts: list[str] | None = None
        self._cell_span = (1, 1)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._rows = []
                self._row_index = -1
            return
        if self._depth != 1 or self._rows is None:
            return
        if tag == "tr":
            self._row_index += 1
            while len(self._rows) <= self._row_index:
                self._rows.append([])
            self._column = 0
        elif tag in {"td", "th"} and self._row_index >= 0:
            values = {name.lower(): value for name, value in attrs if value is not None}
            self._cell_span = (_span(values.get("rowspan")), _span(values.get("colspan")))
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell_parts is not None and self._rows is not None:
            self._place_cell(" ".join("".join(self._cell_parts).split()))
            self._cell_parts = None
            return
        if tag == "table" and self._depth:
            if self._depth == 1 and self._rows is not None:
                rows = _rectangular(self._rows, self.page_index, self.block_id)
                if rows:
                    self.grids.append(TableGrid(self.page_index, self.block_id, rows))
                self._rows = None
            self._depth -= 1

    def _place_cell(self, text: str) -> None:
        assert self._rows is not None
        row = self._rows[self._row_index]
        while self._column < len(row) and row[self._column] is not None:
            self._column += 1
        rowspan, colspan = self._cell_span
        source_ref = f"p{self.page_index}.b{self.block_id}.r{self._row_index}.c{self._column}"
        cell = GridCell(text, source_ref)
        for row_offset in range(rowspan):
            target_index = self._row_index + row_offset
            while len(self._rows) <= target_index:
                self._rows.append([])
            target = self._rows[target_index]
            while len(target) < self._column + colspan:
                target.append(None)
            for column_offset in range(colspan):
                position = self._column + column_offset
                if target[position] is not None:
                    raise ValueError("overlapping HTML table spans")
                target[position] = cell
        self._column += colspan


def _order(item: Mapping[str, Any]) -> tuple[float, float, float]:
    order = item.get("block_order")
    identifier = item.get("block_id")
    bbox = item.get("block_bbox")
    top = (
        float(bbox[1])
        if isinstance(bbox, Sequence)
        and not isinstance(bbox, (str, bytes))
        and len(bbox) == 4
        else float("inf")
    )
    return (
        float(order) if isinstance(order, (int, float)) and not isinstance(order, bool) else float("inf"),
        top,
        float(identifier) if isinstance(identifier, int) and not isinstance(identifier, bool) else float("inf"),
    )


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        return None
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _span(value: str | None) -> int:
    if value is None or not value.isdigit():
        return 1
    return max(1, min(int(value), 100))


def _rectangular(
    rows: Sequence[Sequence[GridCell | None]], page_index: int, block_id: int
) -> tuple[tuple[GridCell, ...], ...]:
    width = max((len(row) for row in rows), default=0)
    output: list[tuple[GridCell, ...]] = []
    for row_index, row in enumerate(rows):
        cells = list(row)
        while len(cells) < width:
            cells.append(None)
        output.append(tuple(
            cell or GridCell("", f"p{page_index}.b{block_id}.r{row_index}.c{column_index}")
            for column_index, cell in enumerate(cells)
        ))
    return tuple(output)


def _markdown_rows(
    content: str, page_index: int, block_id: int
) -> tuple[tuple[GridCell, ...], ...]:
    raw_rows: list[list[str]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        raw_rows.append(cells)
    width = max((len(row) for row in raw_rows), default=0)
    return tuple(
        tuple(
            GridCell(
                row[column] if column < len(row) else "",
                f"p{page_index}.b{block_id}.r{row_index}.c{column}",
            )
            for column in range(width)
        )
        for row_index, row in enumerate(raw_rows)
    )
