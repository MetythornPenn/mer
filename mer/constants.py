from pathlib import Path

REPO_ID = "metythorn/ocr-stn-cnn-transformer-base"
MODEL_FILENAME = "khmer_ocr.onnx"
CONFIG_FILENAME = "config.json"
DEFAULT_CACHE_DIR = Path.home() / ".mer" / "ocr-stn-cnn-transformer-base"

__all__ = [
    "REPO_ID",
    "MODEL_FILENAME",
    "CONFIG_FILENAME",
    "DEFAULT_CACHE_DIR",
]
