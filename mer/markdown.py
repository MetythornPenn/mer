from __future__ import annotations

from typing import Iterable, Mapping, Any

from .types import DocumentResult, LineResult, TableBlockResult
from .document_utils import coerce_document_result


def document_to_markdown(doc: DocumentResult | Mapping[str, Any]) -> str:
    """
    Render DocumentResult into a simple markdown string.
    - Lines are ordered by reading_order.
    - Tables are rendered as markdown tables per block; if no cells present, table bbox is noted.
    Accepts either a DocumentResult or the JSON dict produced by predict(json_result=True).
    """
    doc = coerce_document_result(doc)
    sections: list[str] = []

    # Lines
    lines_by_order = sorted(doc.lines, key=lambda l: l.order)
    if lines_by_order:
        sections.append("\n".join(_escape_md(line.text) for line in lines_by_order if line.text))

    # Tables
    for table in sorted(doc.tables, key=lambda t: t.order):
        rendered = _render_table(table)
        if rendered:
            sections.append(rendered)

    return "\n\n".join([s for s in sections if s]).strip()


def _render_table(table: TableBlockResult) -> str:
    if not table.cells:
        return f"> Table (no cells detected) bbox={table.bbox}"

    # Group by row_id
    rows: dict[int, list[str]] = {}
    for cell in sorted(table.cells, key=lambda c: (c.row_id, c.col_id or 0)):
        rows.setdefault(cell.row_id, []).append(_escape_md(cell.text))

    # Normalize columns
    max_cols = max((len(cols) for cols in rows.values()), default=0)
    normalized_rows = [cols + [""] * (max_cols - len(cols)) for _, cols in sorted(rows.items())]

    header = "| " + " | ".join([f"Col {i+1}" for i in range(max_cols)]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in normalized_rows)
    return "\n".join([header, separator, body])


def _escape_md(text: str) -> str:
    for ch in "|`*_":
        text = text.replace(ch, f"\\{ch}")
    return text.strip()


__all__ = ["document_to_markdown"]
