import re


_SPACE_BEFORE_KHMER_PERIOD = re.compile(r"\s+។")


def postprocess_text(text: str) -> str:
    """
    Postprocess to cleanup recognized text:
    - replace tabs with spaces
    - trim leading spaces on each line
    - collapse multiple spaces into a single space
    - remove spaces before Khmer period '។'
    """
    if not text:
        return text
    cleaned = text.replace("\t", " ")
    cleaned = "\n".join(line.lstrip() for line in cleaned.splitlines())
    # collapse multiple spaces
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = _SPACE_BEFORE_KHMER_PERIOD.sub("។", cleaned)
    return cleaned


__all__ = ["postprocess_text"]
