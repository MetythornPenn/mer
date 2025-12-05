from __future__ import annotations
import os
from pathlib import Path
from typing import List, Sequence, Union

import torch

PathLike = Union[str, "os.PathLike[str]"]  # noqa: F821


class Vocabulary:
    """Simple character-level vocabulary helper."""

    def __init__(self, data: Union[dict, PathLike]):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.specials = ["<PAD>", "<SOS>", "<EOS>"]
        self.max_label_len = 0
        if isinstance(data, dict):
            self._init_from_dict(data)
        else:
            self.build_vocab(str(data))

    def build_vocab(self, data_file: PathLike) -> None:
        chars = set()
        data_path = Path(data_file).expanduser()
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    label = parts[1]
                    chars.update(label)
                    self.max_label_len = max(self.max_label_len, len(label))

        all_tokens = self.specials + sorted(chars)
        self.char2idx = {char: idx for idx, char in enumerate(all_tokens)}
        self.idx2char = {idx: char for idx, char in enumerate(all_tokens)}

    def _init_from_dict(self, data: dict) -> None:
        self.specials = data.get("specials", ["<PAD>", "<SOS>", "<EOS>"])
        self.char2idx = data["char2idx"]
        idx2char_raw = data["idx2char"]
        if isinstance(idx2char_raw, dict):
            self.idx2char = {int(k): v for k, v in idx2char_raw.items()}
        else:
            self.idx2char = {int(idx): char for idx, char in enumerate(idx2char_raw)}

    def encode(self, text: str) -> List[int]:
        sos = self.char2idx["<SOS>"]
        eos = self.char2idx["<EOS>"]
        body = [self.char2idx[c] for c in text if c in self.char2idx]
        return [sos, *body, eos]

    def decode(self, tokens: Sequence[int]) -> str:
        pad = self.char2idx["<PAD>"]
        sos = self.char2idx["<SOS>"]
        eos = self.char2idx["<EOS>"]
        result: List[str] = []
        for token in tokens:
            if isinstance(token, torch.Tensor):  # pragma: no cover - convenience
                token = int(token.item())
            if token in (pad, sos):
                continue
            if token == eos:
                break
            result.append(self.idx2char[token])
        return "".join(result)

    def __len__(self) -> int:
        return len(self.char2idx)

    def to_dict(self) -> dict:
        return {
            "specials": self.specials,
            "char2idx": self.char2idx,
            "idx2char": {str(k): v for k, v in self.idx2char.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        return cls(data)


__all__ = ["Vocabulary"]
