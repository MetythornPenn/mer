from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

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
    show_progress: bool = True,
    local_dir: Optional[PathLike] = None,
) -> ArtifactPaths:
    """
    Make sure model weights and config exist locally, downloading from Hugging Face if missing.
    If `local_dir` is provided, the function will use files from that directory and never attempt
    to download.
    """
    base_dir = Path(local_dir).expanduser() if local_dir else Path(cache_dir).expanduser()
    if not local_dir:
        base_dir.mkdir(parents=True, exist_ok=True)

    weights_path = base_dir / model_filename
    config_path = base_dir / config_filename

    if local_dir:
        if not weights_path.exists() or not config_path.exists():
            missing = [p.name for p in (weights_path, config_path) if not p.exists()]
            raise FileNotFoundError(
                f"Expected local model files in {base_dir}, missing: {', '.join(missing)}"
            )
        return ArtifactPaths(base_dir, weights_path, config_path)

    missing_files = [path for path in (weights_path, config_path) if not path.exists()]
    progress: Optional[tqdm] = None
    if missing_files:
        display_progress = show_progress and sys.stderr.isatty()
        progress = tqdm(
            total=len(missing_files),
            desc="Downloading Mer artifacts",
            unit="file",
            disable=not display_progress,
        )

    def _download(filename: str, target: Path) -> Path:
        if target.exists():
            return target
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=base_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"Failed to download {filename} from {repo_id}") from exc
        if progress:
            progress.update()
        return Path(downloaded)

    try:
        weights_path = _download(model_filename, weights_path)
        config_path = _download(config_filename, config_path)
    finally:
        if progress:
            progress.close()
    return ArtifactPaths(base_dir, weights_path, config_path)


__all__ = ["ArtifactPaths", "ensure_artifacts"]
