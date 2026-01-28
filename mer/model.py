"""
Compatibility shim to preserve older imports.
"""
from .mer import Mer, ensure_artifacts, ArtifactPaths
from .constants import MODEL_FILENAME, CONFIG_FILENAME, REPO_ID, DEFAULT_CACHE_DIR
from .predictor import Predictor
from .vocab import Vocabulary

__all__ = [
    "Mer",
    "ensure_artifacts",
    "ArtifactPaths",
    "MODEL_FILENAME",
    "CONFIG_FILENAME",
    "REPO_ID",
    "DEFAULT_CACHE_DIR",
    "Predictor",
    "Vocabulary",
]
