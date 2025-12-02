from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, slots=True)
class LineResult:
    order: int
    block_index: int
    block_label: str
    text: str
    polygon: List[List[float]]
    bbox: List[float]


@dataclass(frozen=True, slots=True)
class TableCellResult:
    row_id: int
    col_id: int | None
    text: str
    is_header: bool
    polygon: List[List[float]]
    bbox: List[float]


@dataclass(frozen=True, slots=True)
class TableBlockResult:
    order: int
    polygon: List[List[float]]
    bbox: List[float]
    cells: List[TableCellResult]


@dataclass(frozen=True, slots=True)
class LayoutBlockResult:
    order: int
    label: str
    polygon: List[List[float]]
    bbox: List[float]


@dataclass(frozen=True, slots=True)
class DetectionBoxResult:
    polygon: List[List[float]]
    bbox: List[float]
    confidence: float | None


@dataclass(frozen=True, slots=True)
class DocumentResult:
    lines: List[LineResult]
    tables: List[TableBlockResult]
    layout_blocks: List[LayoutBlockResult]
    detections: List[DetectionBoxResult]
    reading_order: List[int]
    device: str
    timings: dict[str, float] | None = None


__all__ = [
    "LineResult",
    "TableCellResult",
    "TableBlockResult",
    "LayoutBlockResult",
    "DetectionBoxResult",
    "DocumentResult",
]
