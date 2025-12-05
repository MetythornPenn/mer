from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from .vocab import Vocabulary

PathLike = Union[str, os.PathLike]


def _providers_from_device(device: Optional[Union[str, os.PathLike]]) -> Optional[List[str]]:
    """
    Map a device hint to ONNX Runtime providers while keeping backwards compatibility
    with the previous torch-style `device` argument.
    """
    if device is None:
        return None
    if isinstance(device, str):
        if device.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.lower() == "cpu":
            return ["CPUExecutionProvider"]
    return None


class Predictor:
    """
    ONNX Runtime inference for the Khmer OCR model.
    """

    def __init__(
        self,
        model_path: PathLike,
        vocab_path: Optional[PathLike] = None,
        config_path: Optional[PathLike] = None,
        device: Optional[Union[str, os.PathLike]] = None,
        max_length: Optional[int] = None,
        providers: Optional[List[str]] = None,
        session: Optional[ort.InferenceSession] = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.vocab_path = Path(vocab_path).expanduser() if vocab_path else None
        self.config_path = Path(config_path).expanduser() if config_path else None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self.config_data = self._load_config()
        self.hparams = self._resolve_hparams()
        resolved_max_len = max_length if max_length is not None else self.hparams.get("max_decode_len", 128)
        self.max_length = int(resolved_max_len)
        self.vocab = self._load_vocab()
        self.transform = self._build_transform()

        self.providers = self._resolve_providers(providers, device)
        self.session = session or ort.InferenceSession(
            str(self.model_path),
            providers=self.providers or ort.get_available_providers(),
        )
        self.output_name = self._select_output_name()
        self.image_input_name, self.tgt_input_name = self._select_input_names()

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
            "max_decode_len": 128,
        }
        resolved = {k: candidates.get(k, v) for k, v in defaults.items()}
        resolved["img_height"] = int(resolved["img_height"])
        resolved["img_width"] = int(resolved["img_width"])
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

    def _resolve_providers(self, providers: Optional[List[str]], device: Optional[Union[str, os.PathLike]]) -> Optional[List[str]]:
        if providers:
            return providers
        hinted = _providers_from_device(device)
        if not hinted:
            return None
        available = set(ort.get_available_providers())
        usable = [p for p in hinted if p in available]
        return usable or None

    def _select_output_name(self) -> str:
        outputs = self.session.get_outputs()
        for candidate in outputs:
            if candidate.name == "logits":
                return candidate.name
        return outputs[0].name

    def _select_input_names(self) -> tuple[str, str]:
        inputs = self.session.get_inputs()
        names = [inp.name for inp in inputs]
        img_name = "images" if "images" in names else names[0]
        tgt_name = "tgt" if "tgt" in names else (names[1] if len(names) > 1 else names[0])
        return img_name, tgt_name

    def _prepare_image(self, image: Union[PathLike, Image.Image]) -> np.ndarray:
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            image_path = Path(image).expanduser()
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            pil_image = Image.open(image_path).convert("RGB")
        tensor = self.transform(pil_image)  # (C, H, W)
        return tensor.unsqueeze(0).cpu().numpy().astype(np.float32)  # (1, C, H, W)

    def _greedy_decode(self, image_array: np.ndarray) -> List[int]:
        sos_idx = self.vocab.char2idx["<SOS>"]
        eos_idx = self.vocab.char2idx["<EOS>"]
        pad_idx = self.vocab.char2idx["<PAD>"]
        generated = [sos_idx]
        max_len = self.max_length

        for _ in range(max_len - 1):  # leave room for EOS
            tgt = np.full((1, max_len), pad_idx, dtype=np.int64)
            tgt[0, : len(generated)] = generated
            outputs = self.session.run(
                [self.output_name],
                {
                    self.image_input_name: image_array,
                    self.tgt_input_name: tgt,
                },
            )
            logits = outputs[0]  # (1, seq, vocab)
            next_pos = len(generated) - 1
            next_token = int(logits[0, next_pos, :].argmax(axis=-1))
            if next_token == eos_idx:
                break
            generated.append(next_token)

        return generated

    def predict(self, image: Union[PathLike, Image.Image]) -> str:
        image_array = self._prepare_image(image)
        tokens = self._greedy_decode(image_array)
        return self.vocab.decode(tokens)


__all__ = ["Predictor", "PathLike"]
