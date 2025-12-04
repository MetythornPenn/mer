from __future__ import annotations

from typing import Any, Mapping, Sequence

from .types import (
    DetectionBoxResult,
    DocumentResult,
    LayoutBlockResult,
    LineResult,
    TableBlockResult,
    TableCellResult,
)


def coerce_document_result(doc: DocumentResult | Mapping[str, Any]) -> DocumentResult:
    """
    Normalize a JSON-like dict (as returned by Mer.predict with json_result=True)
    into a DocumentResult. If a DocumentResult is provided, it is returned as-is.
    """
    if isinstance(doc, DocumentResult):
        return doc
    if not isinstance(doc, Mapping):
        raise TypeError("doc must be a DocumentResult or mapping produced by predict(json_result=True)")

    lines = [_parse_line(line, idx) for idx, line in enumerate(doc.get("lines", []))]
    tables = [_parse_table(table, idx) for idx, table in enumerate(doc.get("tables", []))]
    layout_blocks = [_parse_layout(block, idx) for idx, block in enumerate(doc.get("layout_blocks", []))]
    detections = [_parse_detection(det) for det in doc.get("detections", [])]

    reading_order = list(doc.get("reading_order", [])) or [line.order for line in lines]
    device = str(doc.get("device", "unknown"))
    timings = doc.get("timings")

    return DocumentResult(
        lines=lines,
        tables=tables,
        layout_blocks=layout_blocks,
        detections=detections,
        reading_order=reading_order,
        device=device,
        timings=timings if isinstance(timings, Mapping) else None,
    )


def document_result_to_json(doc: DocumentResult) -> dict[str, Any]:
    """
    Convert a structured DocumentResult to the JSON-serializable dict format
    returned by Mer.predict(json_result=True).
    """
    return {
        "device": doc.device,
        "timings": doc.timings or {},
        "reading_order": doc.reading_order,
        "lines": [
            {
                "order": line.order,
                "block_index": line.block_index,
                "block_label": line.block_label,
                "text": line.text,
                "polygon": line.polygon,
                "bbox": line.bbox,
            }
            for line in doc.lines
        ],
        "tables": [
            {
                "order": table.order,
                "polygon": table.polygon,
                "bbox": table.bbox,
                "cells": [
                    {
                        "row_id": cell.row_id,
                        "col_id": cell.col_id,
                        "text": cell.text,
                        "is_header": cell.is_header,
                        "polygon": cell.polygon,
                        "bbox": cell.bbox,
                    }
                    for cell in table.cells
                ],
            }
            for table in doc.tables
        ],
        "layout_blocks": [
            {
                "order": block.order,
                "label": block.label,
                "polygon": block.polygon,
                "bbox": block.bbox,
            }
            for block in doc.layout_blocks
        ],
        "detections": [
            {"polygon": det.polygon, "bbox": det.bbox, "confidence": det.confidence} for det in doc.detections
        ],
    }


def _parse_line(line: Mapping[str, Any], fallback_order: int) -> LineResult:
    return LineResult(
        order=int(line.get("order", fallback_order)),
        block_index=int(line.get("block_index", 0)),
        block_label=str(line.get("block_label", line.get("label", "Text"))),
        text=str(line.get("text", "")),
        polygon=_to_polygon(line.get("polygon", [])),
        bbox=_to_bbox(line.get("bbox", [])),
    )


def _parse_table(table: Mapping[str, Any], fallback_order: int) -> TableBlockResult:
    cells = [_parse_cell(cell, idx) for idx, cell in enumerate(table.get("cells", []))]
    return TableBlockResult(
        order=int(table.get("order", fallback_order)),
        polygon=_to_polygon(table.get("polygon", [])),
        bbox=_to_bbox(table.get("bbox", [])),
        cells=cells,
    )


def _parse_cell(cell: Mapping[str, Any], fallback_row: int) -> TableCellResult:
    col_id = cell.get("col_id")
    return TableCellResult(
        row_id=int(cell.get("row_id", fallback_row)),
        col_id=int(col_id) if col_id is not None else None,
        text=str(cell.get("text", "")),
        is_header=bool(cell.get("is_header", False)),
        polygon=_to_polygon(cell.get("polygon", [])),
        bbox=_to_bbox(cell.get("bbox", [])),
    )


def _parse_layout(block: Mapping[str, Any], fallback_order: int) -> LayoutBlockResult:
    return LayoutBlockResult(
        order=int(block.get("order", fallback_order)),
        label=str(block.get("label", "")),
        polygon=_to_polygon(block.get("polygon", [])),
        bbox=_to_bbox(block.get("bbox", [])),
    )


def _parse_detection(det: Mapping[str, Any]) -> DetectionBoxResult:
    confidence = det.get("confidence")
    return DetectionBoxResult(
        polygon=_to_polygon(det.get("polygon", [])),
        bbox=_to_bbox(det.get("bbox", [])),
        confidence=float(confidence) if confidence is not None else None,
    )


def _to_bbox(values: Sequence[Any]) -> list[float]:
    return [float(v) for v in values] if values else []


def _to_polygon(points: Sequence[Sequence[Any]]) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in points] if points else []


__all__ = ["coerce_document_result", "document_result_to_json"]
