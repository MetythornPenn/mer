"""
Compatibility shim to preserve older imports.
"""
from .merl import Merl, ensure_artifacts, ArtifactPaths
from .types import (
    DocumentResult,
    LineResult,
    TableBlockResult,
    TableCellResult,
    LayoutBlockResult,
    DetectionBoxResult,
)
from .constants import MODEL_FILENAME, CONFIG_FILENAME, REPO_ID, DEFAULT_CACHE_DIR
from .predictor import Predictor
from .surya import SuryaDocumentProcessor
from .vocab import Vocabulary
from .transformer import KhmerOCRTransformer, PositionalEncoding

__all__ = [
    "Merl",
    "ensure_artifacts",
    "ArtifactPaths",
    "DocumentResult",
    "LineResult",
    "TableBlockResult",
    "TableCellResult",
    "LayoutBlockResult",
    "DetectionBoxResult",
    "MODEL_FILENAME",
    "CONFIG_FILENAME",
    "REPO_ID",
    "DEFAULT_CACHE_DIR",
    "Predictor",
    "SuryaDocumentProcessor",
    "Vocabulary",
    "KhmerOCRTransformer",
    "PositionalEncoding",
]
