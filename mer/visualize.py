from __future__ import annotations

from typing import Tuple, Union
import os
from PIL import Image, ImageDraw

from .types import DocumentResult
from .surya import SuryaDocumentProcessor

PathLike = Union[str, "os.PathLike[str]"]  # noqa: F821


def draw_document_boxes(
    image: Union[bytes, Image.Image, PathLike],
    doc: DocumentResult,
    show_lines: bool = True,
    show_layout: bool = False,
    show_tables: bool = True,
    show_detections: bool = False,
    line_color: Tuple[int, int, int, int] = (255, 0, 0, 180),
    table_color: Tuple[int, int, int, int] = (0, 128, 255, 180),
    layout_color: Tuple[int, int, int, int] = (0, 255, 0, 120),
    detection_color: Tuple[int, int, int, int] = (255, 165, 0, 120),
    width: int = 2,
) -> Image.Image:
    """
    Draw bounding boxes from DocumentResult onto an image and return a new PIL.Image.
    """
    pil_image = SuryaDocumentProcessor._coerce_image(image)  # reuse shared helper
    if pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    if show_lines:
        for line in doc.lines:
            draw.rectangle(line.bbox, outline=line_color, width=width)
    if show_tables:
        for table in doc.tables:
            draw.rectangle(table.bbox, outline=table_color, width=width + 1)
            for cell in table.cells:
                draw.rectangle(cell.bbox, outline=table_color, width=width)
    if show_layout:
        for block in doc.layout_blocks:
            if block.polygon:
                pts = [(int(p[0]), int(p[1])) for p in block.polygon]
                draw.polygon(pts, outline=layout_color)
            draw.rectangle(block.bbox, outline=layout_color, width=max(1, width - 1))
    if show_detections:
        for det in doc.detections:
            draw.rectangle(det.bbox, outline=detection_color, width=max(1, width - 1))

    composed = Image.alpha_composite(pil_image, overlay)
    return composed.convert("RGB")


def gather_bboxes(doc: DocumentResult) -> dict[str, list[list[float]]]:
    """
    Return a plain dict of bounding boxes so callers can plot/overlay however they like.
    Keys: lines, tables, table_cells, layout, layout_polygons, detections.
    """
    return {
        "lines": [line.bbox for line in doc.lines],
        "tables": [table.bbox for table in doc.tables],
        "table_cells": [cell.bbox for table in doc.tables for cell in table.cells],
        "layout": [block.bbox for block in doc.layout_blocks],
        "layout_polygons": [block.polygon for block in doc.layout_blocks],
        "detections": [det.bbox for det in doc.detections],
    }


__all__ = ["draw_document_boxes", "gather_bboxes"]
