from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Callable, List, Sequence, Tuple, Union, TYPE_CHECKING
import time, os

import torch
from PIL import Image

from .types import (
    DetectionBoxResult,
    DocumentResult,
    LayoutBlockResult,
    LineResult,
    TableBlockResult,
    TableCellResult,
)

if TYPE_CHECKING:  # pragma: no cover
    from surya.common.polygon import PolygonBox
    from surya.layout.schema import LayoutResult

PathLike = Union[str, "os.PathLike[str]"]  # noqa: F821


class SuryaDocumentProcessor:
    """
    Surya-powered layout, detection, reading order, and LaTeX OCR hooked into the custom recognizer.
    """

    def __init__(
        self,
        line_predict_fn: Callable[[Image.Image], str],
        device_preference: str | torch.device | None = None,
    ) -> None:
        self._line_predict_fn = line_predict_fn
        self._device_preference = device_preference
        self._foundation = None
        self._layout_predictor = None
        self._detection_predictor = None
        self._table_predictor = None
        self._recognition_predictor = None
        self._load_lock = Lock()
        self._device_label = "unknown"
        self._compute_device = self._resolve_device()
        self._surya_imported = False
        self._PolygonBox: Any = None
        self._TaskNames: Any = None
        self._LayoutBox: Any = None
        self._LayoutResult: Any = None
        self._TableCell: Any = None

    @property
    def device(self) -> str:
        return self._device_label

    def load(self) -> None:
        if self._layout_predictor is not None:
            return

        with self._load_lock:
            if self._layout_predictor is not None:
                return

            self._ensure_surya_imports()
            foundation = self._init_with_device(self._FoundationPredictor)
            self._foundation = foundation
            self._layout_predictor = self._init_with_device(self._LayoutPredictor, foundation)
            self._detection_predictor = self._init_with_device(self._DetectionPredictor)
            try:
                self._table_predictor = self._init_with_device(self._TableRecPredictor)
            except Exception:  # pragma: no cover - optional dependency variance
                self._table_predictor = None
            self._recognition_predictor = self._init_with_device(self._RecognitionPredictor, foundation)
            self._device_label = self._compute_device

    def process_image(self, data: Union[bytes, Image.Image, PathLike]) -> DocumentResult:
        image = self._coerce_image(data)
        if self._layout_predictor is None or self._detection_predictor is None:
            raise RuntimeError("Surya document processor is not loaded")

        timings: dict[str, float] = {}
        overall_start = time.perf_counter()

        layout_start = time.perf_counter()
        layout_results = self._layout_predictor([image])
        timings["layout"] = time.perf_counter() - layout_start

        detect_start = time.perf_counter()
        detection_results = self._detection_predictor([image])
        timings["detection"] = time.perf_counter() - detect_start
        if not layout_results or not detection_results:
            return DocumentResult(
                lines=[],
                tables=[],
                layout_blocks=[],
                detections=[],
                reading_order=[],
                device=self.device,
                timings=timings,
            )

        layout = layout_results[0]
        detection = detection_results[0]

        recognition_time = 0.0

        text_lines, rec_time_lines = self._extract_text_lines(image, detection.bboxes, layout)
        recognition_time += rec_time_lines
        tables, rec_time_tables = self._extract_tables(image, layout)
        recognition_time += rec_time_tables
        layout_blocks = self._convert_layout_blocks(layout)
        detections = self._convert_detections(detection.bboxes)
        timings["recognition"] = recognition_time
        timings["total"] = time.perf_counter() - overall_start

        return DocumentResult(
            lines=text_lines,
            tables=tables,
            layout_blocks=layout_blocks,
            detections=detections,
            reading_order=[line.order for line in text_lines],
            device=self.device,
            timings=timings,
        )

    def recognise_latex(self, data: Union[bytes, Image.Image, PathLike]) -> str:
        if self._recognition_predictor is None:
            raise RuntimeError("Surya document processor is not loaded")

        image = self._coerce_image(data)
        width, height = image.size
        tasks = [self._TaskNames.block_without_boxes]
        bboxes = [[[0, 0, width, height]]]

        results = self._recognition_predictor(
            [image],
            task_names=tasks,
            bboxes=bboxes,
            math_mode=True,
            sort_lines=False,
        )
        if not results or not results[0].text_lines:
            return ""
        return results[0].text_lines[0].text

    def _extract_text_lines(
        self,
        image: Image.Image,
        detections: Sequence["PolygonBox"],
        layout: "LayoutResult",
    ) -> Tuple[List[LineResult], float]:
        text_blocks = [block for block in layout.bboxes if block.label in self._textual_labels]
        sorted_blocks = sorted(text_blocks, key=lambda block: block.position)

        def best_block(poly: "PolygonBox") -> tuple[int, str]:
            best_idx = -1
            best_label = "Text"
            best_score = 0.0
            for idx, block in enumerate(sorted_blocks):
                score = self._polygon_intersection(poly, block)
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_label = block.label
            return (best_idx, best_label) if best_score > 0.05 else (-1, "Text")

        lines: List[LineResult] = []
        rec_time = 0.0
        order = 0
        for poly in self._sort_line_boxes(detections):
            block_index, block_label = best_block(poly)
            text, elapsed = self._run_line_recognition(image, poly.bbox)
            rec_time += elapsed
            lines.append(
                LineResult(
                    order=order,
                    block_index=block_index if block_index >= 0 else 0,
                    block_label=block_label,
                    text=text,
                    polygon=self._clone_polygon(poly.polygon),
                    bbox=[float(v) for v in poly.bbox],
                )
            )
            order += 1
        return lines, rec_time

    def _extract_tables(self, image: Image.Image, layout: "LayoutResult") -> Tuple[List[TableBlockResult], float]:
        table_boxes = [block for block in layout.bboxes if block.label in self._table_labels]
        if not table_boxes or self._table_predictor is None:
            return [], 0.0

        table_images: List[Image.Image] = []
        table_meta: List[Tuple[int, Any, Tuple[int, int, int, int]]] = []
        for block_order, block in enumerate(sorted(table_boxes, key=lambda b: b.position)):
            bbox = self._clamp_bbox(block.bbox, image.size)
            if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4:
                continue
            crop = image.crop(bbox)
            table_images.append(crop)
            table_meta.append((block_order, block, bbox))

        if not table_images:
            return []

        try:
            results = self._table_predictor(table_images)
        except Exception:
            return [], 0.0

        blocks: List[TableBlockResult] = []
        rec_time = 0.0
        for (block_order, block, bbox), table_pred in zip(table_meta, results):
            cells = []
            for cell in table_pred.cells:
                converted, elapsed = self._convert_cell(cell, bbox, image)
                rec_time += elapsed
                if converted is not None:
                    cells.append(converted)
            blocks.append(
                TableBlockResult(
                    order=block_order,
                    polygon=self._clone_polygon(block.polygon),
                    bbox=[float(v) for v in bbox],
                    cells=cells,
                )
            )
        return blocks, rec_time

    def _convert_layout_blocks(self, layout: "LayoutResult") -> List[LayoutBlockResult]:
        blocks: List[LayoutBlockResult] = []
        for block in layout.bboxes:
            blocks.append(
                LayoutBlockResult(
                    order=block.position,
                    label=block.label,
                    polygon=self._clone_polygon(block.polygon),
                    bbox=[float(v) for v in block.bbox],
                )
            )
        return sorted(blocks, key=lambda block: block.order)

    def _convert_detections(self, polygons: Sequence["PolygonBox"]) -> List[DetectionBoxResult]:
        detections: List[DetectionBoxResult] = []
        for poly in polygons:
            detections.append(
                DetectionBoxResult(
                    polygon=self._clone_polygon(poly.polygon),
                    bbox=[float(v) for v in poly.bbox],
                    confidence=poly.confidence,
                )
            )
        return detections

    def _convert_cell(
        self,
        cell: Any,
        offset_bbox: Tuple[int, int, int, int],
        image: Image.Image,
    ) -> Tuple[TableCellResult | None, float]:
        bbox = [
            float(cell.bbox[0] + offset_bbox[0]),
            float(cell.bbox[1] + offset_bbox[1]),
            float(cell.bbox[2] + offset_bbox[0]),
            float(cell.bbox[3] + offset_bbox[1]),
        ]
        if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
            return None, 0.0
        text, elapsed = self._run_line_recognition(image, bbox)
        shifted_polygon = [[point[0] + offset_bbox[0], point[1] + offset_bbox[1]] for point in cell.polygon]
        return (
            TableCellResult(
                row_id=cell.row_id,
                col_id=getattr(cell, "col_id", None),
                text=text,
                is_header=getattr(cell, "is_header", False),
                polygon=self._clone_polygon(shifted_polygon),
                bbox=bbox,
            ),
            elapsed,
        )

    def _run_line_recognition(self, image: Image.Image, bbox: Sequence[float]) -> Tuple[str, float]:
        start = time.perf_counter()
        x1, y1, x2, y2 = self._expand_bbox(bbox, image.size, margin=2)
        if x2 <= x1 or y2 <= y1:
            return "", time.perf_counter() - start
        crop = image.crop((x1, y1, x2, y2))
        try:
            result = self._line_predict_fn(crop)
        except Exception:
            result = ""
        return result, time.perf_counter() - start

    def _ensure_surya_imports(self) -> None:
        if self._surya_imported:
            return
        from surya.common.polygon import PolygonBox  # type: ignore
        from surya.common.surya.schema import TaskNames  # type: ignore
        from surya.detection import DetectionPredictor  # type: ignore
        from surya.foundation import FoundationPredictor  # type: ignore
        from surya.layout import LayoutPredictor  # type: ignore
        from surya.layout.schema import LayoutBox, LayoutResult  # type: ignore
        from surya.recognition import RecognitionPredictor  # type: ignore
        from surya.table_rec import TableRecPredictor  # type: ignore
        from surya.table_rec.schema import TableCell  # type: ignore

        self._PolygonBox = PolygonBox
        self._TaskNames = TaskNames
        self._LayoutBox = LayoutBox
        self._LayoutResult = LayoutResult
        self._DetectionPredictor = DetectionPredictor
        self._FoundationPredictor = FoundationPredictor
        self._LayoutPredictor = LayoutPredictor
        self._RecognitionPredictor = RecognitionPredictor
        self._TableRecPredictor = TableRecPredictor
        self._TableCell = TableCell
        self._surya_imported = True

    @property
    def _textual_labels(self) -> set[str]:
        return {
            "Text",
            "ListItem",
            "SectionHeader",
            "Caption",
            "PageHeader",
            "PageFooter",
            "Footnote",
            "Equation",
            "Code",
            "Form",
        }

    @property
    def _table_labels(self) -> set[str]:
        return {"Table", "TableOfContents"}

    @staticmethod
    def _polygon_intersection(box: "PolygonBox", block: Any) -> float:
        try:
            return float(box.intersection_pct(block, x_margin=0.1, y_margin=0.2))
        except Exception:
            return 0.0

    @staticmethod
    def _sort_line_boxes(polygons: Sequence["PolygonBox"]) -> List["PolygonBox"]:
        groups: List[Tuple[float, List["PolygonBox"]]] = []
        for poly in sorted(polygons, key=lambda p: (p.bbox[1], p.bbox[0])):
            height = max(1.0, poly.height)
            placed = False
            for idx, (center_y, group_polys) in enumerate(groups):
                if abs(poly.bbox[1] - center_y) <= max(10.0, height * 0.6):
                    new_center = (center_y * len(group_polys) + poly.bbox[1]) / (len(group_polys) + 1)
                    groups[idx] = (new_center, group_polys + [poly])
                    placed = True
                    break
            if not placed:
                groups.append((poly.bbox[1], [poly]))

        ordered: List["PolygonBox"] = []
        for _, group_polys in sorted(groups, key=lambda g: g[0]):
            ordered.extend(sorted(group_polys, key=lambda p: p.bbox[0]))
        return ordered

    @staticmethod
    def _clamp_bbox(bbox: Sequence[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        width, height = image_size
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(width, int(bbox[2]))
        y2 = min(height, int(bbox[3]))
        return x1, y1, x2, y2

    @staticmethod
    def _expand_bbox(bbox: Sequence[float], image_size: Tuple[int, int], margin: int = 2) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(image_size[0], int(x2) + margin)
        y2 = min(image_size[1], int(y2) + margin)
        return x1, y1, x2, y2

    @staticmethod
    def _clone_polygon(polygon: Sequence[Sequence[float]]) -> List[List[float]]:
        return [[float(point[0]), float(point[1])] for point in polygon]

    def _resolve_device(self) -> str:
        preferred = (str(self._device_preference) if self._device_preference is not None else "auto").lower()
        if preferred == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        if preferred == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if preferred == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return preferred

    def _init_with_device(self, factory: Any, *args: Any):
        device = self._compute_device
        try:
            import inspect

            sig = inspect.signature(factory)
            if "device" in sig.parameters:
                return factory(*args, device=device)
        except Exception:
            pass

        predictor = factory(*args)
        to_method = getattr(predictor, "to", None)
        if callable(to_method):
            try:
                predictor = to_method(device) or predictor
            except Exception:
                pass
        return predictor

    @staticmethod
    def _coerce_image(data: Union[bytes, Image.Image, PathLike]) -> Image.Image:
        if isinstance(data, Image.Image):
            return data.convert("RGB")
        if isinstance(data, (bytes, bytearray)):
            return Image.open(BytesIO(data)).convert("RGB")
        image_path = Path(data).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        return Image.open(image_path).convert("RGB")


__all__ = ["SuryaDocumentProcessor"]
