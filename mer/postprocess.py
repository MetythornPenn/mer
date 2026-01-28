from __future__ import annotations

import re
from typing import Union

_SPACE_BEFORE_KHMER_PERIOD = re.compile(r"\s+។")


def postprocess_text(text: Union[str, None]) -> Union[str, None]:
    """
    Postprocess recognized text by normalizing whitespace and removing stray spaces
    before Khmer periods.
    """
    if text is None:
        return None
    cleaned = text.replace("\t", " ")
    cleaned = "\n".join(line.lstrip() for line in cleaned.splitlines())
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = _SPACE_BEFORE_KHMER_PERIOD.sub("។", cleaned)
    return cleaned.strip()


__all__ = ["postprocess_text"]
