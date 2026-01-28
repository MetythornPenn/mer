from __future__ import annotations

from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Optional, Union, Dict

import torch
from PIL import Image

from .artifacts import ArtifactPaths, ensure_artifacts
from .constants import DEFAULT_CACHE_DIR, REPO_ID
from .predictor import Predictor, PathLike
from .postprocess import postprocess_text


class Mer:
    """
    Public-facing helper around the single-line CNN-Transformer recognizer.
    """

    def __init__(
        self,
        cache_dir: PathLike = DEFAULT_CACHE_DIR,
        repo_id: str = REPO_ID,
        device: Optional[Union[str, torch.device]] = "cuda",
        max_length: Optional[int] = None,
        model_path: Optional[PathLike] = None,
        providers: Optional[list[str]] = None,
        markdown: bool = False,
        postprocess: bool = True,
        json_result: bool = False,
    ) -> None:
        if markdown:
            raise ValueError("Markdown output is no longer supported; Mer now focuses on line recognition only.")
        artifacts = ensure_artifacts(
            cache_dir=cache_dir,
            repo_id=repo_id,
            local_dir=model_path,
        )
        self.artifacts = artifacts
        self._default_json_result = bool(json_result)
        self._apply_postprocess = postprocess
        self._predictor_lock = Lock()
        self._predictor = Predictor(
            model_path=str(artifacts.weights),
            config_path=str(artifacts.config),
            device=device,
            max_length=max_length,
            providers=providers,
        )

    def _predict_image(self, image: Image.Image) -> str:
        with self._predictor_lock:
            raw = self._predictor.predict(image)
        if not isinstance(raw, str):
            return str(raw)
        return postprocess_text(raw) if self._apply_postprocess else raw

    @staticmethod
    def _coerce_image(image: Union[bytes, Image.Image, PathLike]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(BytesIO(image)).convert("RGB")
        image_path = Path(image).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        return Image.open(image_path).convert("RGB")

    def recognize_line(self, image: Union[bytes, Image.Image, PathLike], json_result: bool = False) -> Union[str, Dict[str, str]]:
        """Run the custom CNN-Transformer line recognizer directly."""
        pil_image = self._coerce_image(image)
        text = self._predict_image(pil_image)
        if json_result:
            return {"text": text}
        return text

    def predict(self, image: Union[bytes, Image.Image, PathLike], json_result: Optional[bool] = None) -> Union[str, Dict[str, str]]:
        """
        Backwards-compatible alias for recognize_line.
        json_result defaults to the value provided at initialization.
        """
        effective_json = self._default_json_result if json_result is None else json_result
        return self.recognize_line(image, json_result=effective_json)

    def load(self, load_surya: bool = True) -> None:
        """
        Compatibility hook retained so existing code can continue calling load().
        There is no lazy Surya pipeline anymore, so this method simply returns.
        """
        _ = load_surya  # argument kept for backwards compatibility


__all__ = [
    "Mer",
    "ArtifactPaths",
    "ensure_artifacts",
]
