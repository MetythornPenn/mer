from typing import Any, Mapping, Union
import re

from .document_utils import coerce_document_result, document_result_to_json
from .types import DocumentResult, LineResult, TableBlockResult, TableCellResult


_SPACE_BEFORE_KHMER_PERIOD = re.compile(r"\s+។")


def postprocess_text(text: Union[str, DocumentResult, Mapping[str, Any]]) -> Union[str, DocumentResult, dict[str, Any]]:
    """
    Postprocess to cleanup recognized text. Accepts:
    - raw string
    - DocumentResult
    - JSON dict returned by predict(json_result=True)

    - replace tabs with spaces
    - trim leading spaces on each line
    - collapse multiple spaces into a single space
    - remove spaces before Khmer period '។'
    """
    if isinstance(text, str):
        return _postprocess_str(text)

    doc = coerce_document_result(text)
    processed = _postprocess_document(doc)
    if isinstance(text, DocumentResult):
        return processed
    return document_result_to_json(processed)


def _postprocess_str(text: str) -> str:
    if not text:
        return text
    cleaned = text.replace("\t", " ")
    cleaned = "\n".join(line.lstrip() for line in cleaned.splitlines())
    # collapse multiple spaces
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = _SPACE_BEFORE_KHMER_PERIOD.sub("។", cleaned)
    return cleaned.strip()


def _postprocess_document(doc: DocumentResult) -> DocumentResult:
    lines = [
        LineResult(
            order=line.order,
            block_index=line.block_index,
            block_label=line.block_label,
            text=_postprocess_str(line.text),
            polygon=[list(pt) for pt in line.polygon],
            bbox=list(line.bbox),
        )
        for line in doc.lines
    ]

    tables: list[TableBlockResult] = []
    for table in doc.tables:
        cells = [
            TableCellResult(
                row_id=cell.row_id,
                col_id=cell.col_id,
                text=_postprocess_str(cell.text),
                is_header=cell.is_header,
                polygon=[list(pt) for pt in cell.polygon],
                bbox=list(cell.bbox),
            )
            for cell in table.cells
        ]
        tables.append(
            TableBlockResult(
                order=table.order,
                polygon=[list(pt) for pt in table.polygon],
                bbox=list(table.bbox),
                cells=cells,
            )
        )

    return DocumentResult(
        lines=lines,
        tables=tables,
        layout_blocks=list(doc.layout_blocks),
        detections=list(doc.detections),
        reading_order=list(doc.reading_order),
        device=doc.device,
        timings=doc.timings,
    )


__all__ = ["postprocess_text"]
