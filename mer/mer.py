from __future__ import annotations

from threading import Lock
from typing import Optional, Union, Dict, Any

import torch
from PIL import Image

from .artifacts import ArtifactPaths, ensure_artifacts
from .constants import DEFAULT_CACHE_DIR, REPO_ID
from .markdown import document_to_markdown
from .predictor import Predictor, PathLike
from .surya import SuryaDocumentProcessor
from .postprocess import postprocess_text
from .types import (
    DetectionBoxResult,
    DocumentResult,
    LayoutBlockResult,
    LineResult,
    TableBlockResult,
    TableCellResult,
)


class Mer:
    """
    Public-facing helper that wraps the line recognizer and Surya document pipeline.
    """

    def __init__(
        self,
        cache_dir: PathLike = DEFAULT_CACHE_DIR,
        repo_id: str = REPO_ID,
        device: Optional[Union[str, torch.device]] = "cuda",
        max_length: Optional[int] = None,
        model_path: Optional[PathLike] = None,
        markdown: bool = False,
        postprocess: bool = True,
        json_result: bool = True,
    ) -> None:
        artifacts = ensure_artifacts(
            cache_dir=cache_dir,
            repo_id=repo_id,
            local_dir=model_path,
        )
        self.artifacts = artifacts
        self._return_markdown = markdown
        self._return_json = json_result
        self._apply_postprocess = postprocess
        self._predictor_lock = Lock()
        self._predictor = Predictor(
            model_path=str(artifacts.weights),
            config_path=str(artifacts.config),
            device=device,
            max_length=max_length,
        )
        self._document_processor = SuryaDocumentProcessor(
            line_predict_fn=self._predict_image,
            device_preference=device,
        )

    def _predict_image(self, image: Image.Image) -> str:
        with self._predictor_lock:
            raw = self._predictor.predict(image)
        if not isinstance(raw, str):
            return str(raw)
        return postprocess_text(raw) if self._apply_postprocess else raw

    def recognize_line(self, image: Union[bytes, Image.Image, PathLike]) -> str:
        """Run the custom CNN-Transformer line recognizer directly."""
        pil_image = self._document_processor._coerce_image(image)
        return self._predict_image(pil_image)

    def _document_to_json(self, doc: DocumentResult) -> Dict[str, Any]:
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
                {
                    "polygon": det.polygon,
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                }
                for det in doc.detections
            ],
        }

    def predict(self, image: Union[bytes, Image.Image, PathLike]) -> Union[DocumentResult, str, Dict[str, Any]]:
        """Run layout detection, reading order, tables, and line recognition via Surya."""
        self._document_processor.load()
        doc = self._document_processor.process_image(image)
        if self._return_markdown:
            return document_to_markdown(doc)
        if self._return_json:
            return self._document_to_json(doc)
        return doc

    def recognize_latex(self, image: Union[bytes, Image.Image, PathLike]) -> str:
        """Run Surya's math mode recognizer on the provided image."""
        self._document_processor.load()
        return self._document_processor.recognise_latex(image)

    def load(self, load_surya: bool = True) -> None:
        """
        Eagerly load weights/config and optionally warm up Surya predictors.
        Call this once at application startup to avoid lazy loading during requests.
        """
        # Predictor is already constructed in __init__; nothing else needed here.
        if load_surya:
            self._document_processor.load()


__all__ = [
    "Mer",
    "ArtifactPaths",
    "ensure_artifacts",
    "DocumentResult",
    "LineResult",
    "TableBlockResult",
    "TableCellResult",
    "LayoutBlockResult",
    "DetectionBoxResult",
]
