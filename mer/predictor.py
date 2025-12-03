from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image
from torchvision import transforms

from .transformer import KhmerOCRTransformer
from .vocab import Vocabulary

PathLike = Union[str, "os.PathLike[str]"]  # noqa: F821


def _coerce_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


class Predictor:
    """
    Loads the trained checkpoint and performs greedy decoding for a single image (or batch).
    """

    def __init__(
        self,
        model_path: PathLike,
        vocab_path: Optional[PathLike] = None,
        config_path: Optional[PathLike] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.vocab_path = Path(vocab_path).expanduser() if vocab_path else None
        self.config_path = Path(config_path).expanduser() if config_path else None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self.config_data = self._load_config()
        self.hparams = self._resolve_hparams()
        self.device = _coerce_device(device or self.hparams.get("device"))
        resolved_max_len = max_length if max_length is not None else self.hparams.get("max_decode_len", 128)
        self.max_length = int(resolved_max_len)
        self.vocab = self._load_vocab()
        self.transform = self._build_transform()
        self.model = self._load_model()

    def _load_config(self) -> dict:
        search_paths: List[Optional[Path]] = []
        if self.config_path:
            search_paths.append(self.config_path)
        if self.model_path:
            search_paths.append(self.model_path.parent / "config.json")

        seen = set()
        for path in search_paths:
            if not path:
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return {}

    def _load_vocab(self) -> Vocabulary:
        if self.config_data.get("vocab"):
            return Vocabulary.from_dict(self.config_data["vocab"])
        if self.vocab_path and self.vocab_path.exists():
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                serialized = json.load(f)
            return Vocabulary.from_dict(serialized)
        raise FileNotFoundError("config.json must include a serialized vocabulary, or provide an explicit vocab_path.")

    def _resolve_hparams(self) -> dict:
        candidates = self.config_data.get("hyperparameters", self.config_data)
        defaults = {
            "img_height": 128,
            "img_width": 320,
            "d_model": 256,
            "nhead": 4,
            "num_layers": 3,
            "backbone": "resnet18",
            "max_decode_len": 128,
            "device": None,
            "dim_feedforward": 2048,
            "dropout": 0.1,
        }
        resolved = {k: candidates.get(k, v) for k, v in defaults.items()}
        resolved["img_height"] = int(resolved["img_height"])
        resolved["img_width"] = int(resolved["img_width"])
        resolved["d_model"] = int(resolved["d_model"])
        resolved["nhead"] = int(resolved["nhead"])
        resolved["num_layers"] = int(resolved["num_layers"])
        resolved["dim_feedforward"] = int(resolved["dim_feedforward"])
        resolved["dropout"] = float(resolved["dropout"])
        resolved["max_decode_len"] = int(resolved["max_decode_len"])
        return resolved

    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.hparams["img_height"], self.hparams["img_width"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self) -> KhmerOCRTransformer:
        model = KhmerOCRTransformer(
            vocab_size=len(self.vocab),
            d_model=self.hparams["d_model"],
            nhead=self.hparams["nhead"],
            num_layers=self.hparams["num_layers"],
            backbone_name=self.hparams["backbone"],
            dim_feedforward=self.hparams["dim_feedforward"],
            dropout=self.hparams["dropout"],
        ).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
        if state_dict is None:
            raise ValueError(f"Unexpected checkpoint format at {self.model_path}")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _prepare_image(self, image: Union[PathLike, Image.Image]) -> torch.Tensor:
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            image_path = Path(image).expanduser()
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            pil_image = Image.open(image_path).convert("RGB")
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return tensor

    def _greedy_decode(self, image_tensor: torch.Tensor) -> List[int]:
        sos_idx = self.vocab.char2idx["<SOS>"]
        eos_idx = self.vocab.char2idx["<EOS>"]
        generated = [sos_idx]

        for _ in range(self.max_length):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)
            with torch.no_grad():
                output = self.model(image_tensor, tgt_tensor)
            next_token = int(output[0, -1, :].argmax(dim=-1).item())
            if next_token == eos_idx:
                break
            generated.append(next_token)

        return generated

    def predict(self, image: Union[PathLike, Image.Image]) -> str:
        image_tensor = self._prepare_image(image)
        tokens = self._greedy_decode(image_tensor)
        return self.vocab.decode(tokens)


__all__ = ["Predictor", "_coerce_device", "PathLike"]
