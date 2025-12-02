from __future__ import annotations

from threading import Lock
from typing import Optional, Union

import torch
from PIL import Image

from .artifacts import ArtifactPaths, ensure_artifacts
from .constants import DEFAULT_CACHE_DIR, REPO_ID
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


class Merl:
    """
    Public-facing helper that wraps the line recognizer and Surya document pipeline.
    """

    def __init__(
        self,
        cache_dir: PathLike = DEFAULT_CACHE_DIR,
        repo_id: str = REPO_ID,
        device: Optional[Union[str, torch.device]] = None,
        max_length: Optional[int] = None,
    ) -> None:
        artifacts = ensure_artifacts(cache_dir=cache_dir, repo_id=repo_id)
        self.artifacts = artifacts
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
        return postprocess_text(raw) if isinstance(raw, str) else str(raw)

    def recognize_line(self, image: Union[bytes, Image.Image, PathLike]) -> str:
        """Run the custom CNN-Transformer line recognizer directly."""
        pil_image = self._document_processor._coerce_image(image)
        return self._predict_image(pil_image)

    def analyze_document(self, image: Union[bytes, Image.Image, PathLike]) -> DocumentResult:
        """Run layout detection, reading order, tables, and line recognition via Surya."""
        self._document_processor.load()
        return self._document_processor.process_image(image)

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
    "Merl",
    "ArtifactPaths",
    "ensure_artifacts",
    "DocumentResult",
    "LineResult",
    "TableBlockResult",
    "TableCellResult",
    "LayoutBlockResult",
    "DetectionBoxResult",
]
