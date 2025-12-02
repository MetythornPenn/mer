from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from huggingface_hub import hf_hub_download

from .constants import CONFIG_FILENAME, DEFAULT_CACHE_DIR, MODEL_FILENAME, REPO_ID

PathLike = Union[str, "os.PathLike[str]"]  # noqa: F821 - narrow typing without importing os here


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    base_dir: Path
    weights: Path
    config: Path


def ensure_artifacts(
    cache_dir: PathLike = DEFAULT_CACHE_DIR,
    repo_id: str = REPO_ID,
    model_filename: str = MODEL_FILENAME,
    config_filename: str = CONFIG_FILENAME,
) -> ArtifactPaths:
    """
    Make sure model weights and config exist locally, downloading from Hugging Face if missing.
    """
    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    weights_path = cache_root / model_filename
    config_path = cache_root / config_filename

    def _download(filename: str, target: Path) -> Path:
        if target.exists():
            return target
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=cache_root,
                local_dir_use_symlinks=False,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"Failed to download {filename} from {repo_id}") from exc
        return Path(downloaded)

    weights_path = _download(model_filename, weights_path)
    config_path = _download(config_filename, config_path)
    return ArtifactPaths(cache_root, weights_path, config_path)


__all__ = ["ArtifactPaths", "ensure_artifacts"]
