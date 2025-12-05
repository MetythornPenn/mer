import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from mer import Mer, postprocess_text, document_to_markdown
from mer.artifacts import ensure_artifacts
from mer.constants import MODEL_FILENAME, CONFIG_FILENAME
from mer import artifacts as artifacts_module
from mer import predictor as predictor_module
import mer.mer as mer_module
from mer.types import LineResult, DocumentResult, TableBlockResult, TableCellResult
from mer.surya import SuryaDocumentProcessor


def _write_dummy_config(path: Path) -> None:
    config = {
        "vocab": {
            "specials": ["<PAD>", "<SOS>", "<EOS>"],
            "char2idx": {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "A": 3},
            "idx2char": {"0": "<PAD>", "1": "<SOS>", "2": "<EOS>", "3": "A"},
        },
        "hyperparameters": {
            "img_height": 32,
            "img_width": 32,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "backbone": "resnet18",
            "max_decode_len": 8,
            "dim_feedforward": 64,
            "dropout": 0.1,
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def _stub_predictor(monkeypatch, return_value: str = "dummy-text") -> None:
    monkeypatch.setattr(predictor_module.Predictor, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(predictor_module.Predictor, "predict", lambda self, image: return_value)


def test_ensure_artifacts_uses_existing_files(tmp_path, monkeypatch):
    weights = tmp_path / MODEL_FILENAME
    cfg = tmp_path / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)

    def fake_download(*args, **kwargs):
        raise AssertionError("Download should not be called when files exist")

    monkeypatch.setattr(artifacts_module, "hf_hub_download", fake_download)

    artifacts = ensure_artifacts(cache_dir=tmp_path)
    assert artifacts.weights == weights
    assert artifacts.config == cfg
    assert weights.exists()
    assert cfg.exists()


def test_ensure_artifacts_downloads_when_missing(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_download(repo_id: str, filename: str, local_dir: Path, local_dir_use_symlinks: bool):
        target = Path(local_dir) / filename
        target.write_text(filename, encoding="utf-8")
        calls.append(filename)
        return str(target)

    monkeypatch.setattr(artifacts_module, "hf_hub_download", fake_download)

    artifacts = ensure_artifacts(cache_dir=tmp_path)
    assert set(calls) == {MODEL_FILENAME, CONFIG_FILENAME}
    assert artifacts.weights.exists()
    assert artifacts.config.exists()


def test_ensure_artifacts_uses_local_dir(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    weights = local_dir / MODEL_FILENAME
    cfg = local_dir / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)

    def fake_download(*args, **kwargs):
        raise AssertionError("Download should not be called when local_dir is provided")

    monkeypatch.setattr(artifacts_module, "hf_hub_download", fake_download)

    artifacts = ensure_artifacts(cache_dir=tmp_path, local_dir=local_dir)
    assert artifacts.base_dir == local_dir
    assert artifacts.weights == weights
    assert artifacts.config == cfg


def test_ensure_artifacts_raises_when_local_missing(tmp_path):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        ensure_artifacts(local_dir=local_dir)


def test_mer_recognize_line_uses_predictor(tmp_path, monkeypatch):
    weights = tmp_path / MODEL_FILENAME
    cfg = tmp_path / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)

    _stub_predictor(monkeypatch, return_value="dummy-text")

    sample_img = tmp_path / "line.png"
    Image.new("RGB", (10, 10), color="white").save(sample_img)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path)
    assert ocr.recognize_line(sample_img) == "dummy-text"


def test_mer_postprocess_flag(tmp_path, monkeypatch):
    weights = tmp_path / MODEL_FILENAME
    cfg = tmp_path / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)

    raw_text = "raw\ttext"
    _stub_predictor(monkeypatch, return_value=raw_text)

    sample_img = tmp_path / "line.png"
    Image.new("RGB", (10, 10), color="white").save(sample_img)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path, postprocess=False)
    assert ocr.recognize_line(sample_img) == raw_text


def test_mer_recognize_line_json_result(tmp_path, monkeypatch):
    weights = tmp_path / MODEL_FILENAME
    cfg = tmp_path / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)

    _stub_predictor(monkeypatch, return_value="dummy-json-text")

    sample_img = tmp_path / "line.png"
    Image.new("RGB", (10, 10), color="white").save(sample_img)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path)
    result = ocr.recognize_line(sample_img, json_result=True)
    assert result == {"text": "dummy-json-text"}


def test_surya_document_processor_flow(tmp_path, monkeypatch):
    class DummyPolygonBox:
        def __init__(self):
            self.bbox = [0, 0, 10, 10]
            self.polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
            self.height = 10.0
            self.confidence = 0.9

        def intersection_pct(self, block, x_margin=0.0, y_margin=0.0):
            return 1.0

    class DummyLayoutBox:
        def __init__(self):
            self.label = "Text"
            self.bbox = [0, 0, 10, 10]
            self.polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
            self.position = 0

    class DummyLayoutResult:
        def __init__(self):
            self.bboxes = [DummyLayoutBox()]

    class DummyDetectionResult:
        def __init__(self):
            self.bboxes = [DummyPolygonBox()]

    class DummyDetectionPredictor:
        def __call__(self, images):
            return [DummyDetectionResult()]

    class DummyLayoutPredictor:
        def __call__(self, images):
            return [DummyLayoutResult()]

    class DummyTableRecPredictor:
        def __call__(self, images):
            return [SimpleNamespace(cells=[])]

    class DummyRecognitionPredictor:
        def __call__(self, images, **kwargs):
            return [SimpleNamespace(text_lines=[SimpleNamespace(text="latex-text")])]

    class DummyFoundationPredictor:
        pass

    def fake_ensure_surya_imports(self):
        self._PolygonBox = DummyPolygonBox
        self._TaskNames = SimpleNamespace(block_without_boxes="block_without_boxes")
        self._LayoutBox = DummyLayoutBox
        self._LayoutResult = DummyLayoutResult
        self._DetectionPredictor = DummyDetectionPredictor
        self._FoundationPredictor = DummyFoundationPredictor
        self._LayoutPredictor = DummyLayoutPredictor
        self._RecognitionPredictor = DummyRecognitionPredictor
        self._TableRecPredictor = DummyTableRecPredictor
        self._surya_imported = True

    def fake_init_with_device(self, factory, *args):
        try:
            return factory(*args)
        except TypeError:
            return factory()

    processor = SuryaDocumentProcessor(line_predict_fn=lambda img: "line-text")
    monkeypatch.setattr(processor, "_ensure_surya_imports", fake_ensure_surya_imports.__get__(processor))
    monkeypatch.setattr(processor, "_init_with_device", fake_init_with_device.__get__(processor))

    processor.load()

    sample_img = tmp_path / "doc.png"
    Image.new("RGB", (10, 10), color="white").save(sample_img)

    doc = processor.process_image(sample_img)
    assert doc.lines and doc.lines[0].text == "line-text"
    assert doc.reading_order == [0]
    assert doc.timings and "total" in doc.timings

    latex = processor.recognise_latex(sample_img)
    assert latex == "latex-text"


def test_recognize_latex_json_result(tmp_path, monkeypatch):
    weights = tmp_path / MODEL_FILENAME
    cfg = tmp_path / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)

    _stub_predictor(monkeypatch)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path)
    monkeypatch.setattr(ocr._document_processor, "load", lambda: None)
    monkeypatch.setattr(ocr._document_processor, "recognise_latex", lambda image: "json-latex")

    result = ocr.recognize_latex("dummy.png", json_result=True)
    assert result == {"latex": "json-latex"}


def test_postprocess_text():
    assert postprocess_text("ទៀតផង ។") == "ទៀតផង។"
    assert postprocess_text("a\tb") == "a b"
    assert postprocess_text("   spaced\n\tIndented") == "spaced\nIndented"
    assert postprocess_text("a   b  c") == "a b c"
    assert postprocess_text("") == ""


def test_postprocess_text_accepts_document_and_json():
    line = LineResult(order=0, block_index=0, block_label="Text", text="  a\t b  ", polygon=[[0, 0]], bbox=[0, 0, 1, 1])
    cell = TableCellResult(row_id=0, col_id=0, text=" c   d ", is_header=False, polygon=[[0, 0]], bbox=[0, 0, 1, 1])
    table = TableBlockResult(order=0, polygon=[[0, 0]], bbox=[0, 0, 1, 1], cells=[cell])
    doc = DocumentResult(
        lines=[line],
        tables=[table],
        layout_blocks=[],
        detections=[],
        reading_order=[0],
        device="cpu",
        timings=None,
    )

    processed_doc = postprocess_text(doc)
    assert isinstance(processed_doc, DocumentResult)
    assert processed_doc.lines[0].text == "a b"
    assert processed_doc.tables[0].cells[0].text == "c d"

    json_doc = {
        "lines": [
            {
                "order": 0,
                "block_index": 0,
                "block_label": "Text",
                "text": "  hello \tworld  ",
                "polygon": [[0, 0]],
                "bbox": [0, 0, 1, 1],
            }
        ],
        "tables": [
            {
                "order": 0,
                "polygon": [[0, 0]],
                "bbox": [0, 0, 1, 1],
                "cells": [
                    {"row_id": 0, "col_id": 0, "text": " cell  text ", "is_header": False, "polygon": [[0, 0]], "bbox": [0, 0, 1, 1]}
                ],
            }
        ],
        "layout_blocks": [],
        "detections": [],
        "reading_order": [0],
        "device": "cpu",
        "timings": {"total": 0.1},
    }
    processed_json = postprocess_text(json_doc)
    assert isinstance(processed_json, dict)
    assert processed_json["lines"][0]["text"] == "hello world"
    assert processed_json["tables"][0]["cells"][0]["text"] == "cell text"


def test_document_to_markdown_accepts_json():
    doc = {
        "lines": [
            {"order": 1, "block_index": 0, "block_label": "Text", "text": "Second line", "polygon": [[0, 0]], "bbox": [0, 0, 1, 1]},
            {"order": 0, "block_index": 0, "block_label": "Text", "text": "First line", "polygon": [[0, 0]], "bbox": [0, 0, 1, 1]},
        ],
        "tables": [
            {
                "order": 0,
                "polygon": [[0, 0]],
                "bbox": [0, 0, 1, 1],
                "cells": [
                    {"row_id": 0, "col_id": 0, "text": "cell-1", "is_header": False, "polygon": [[0, 0]], "bbox": [0, 0, 1, 1]},
                    {"row_id": 0, "col_id": 1, "text": "cell-2", "is_header": False, "polygon": [[0, 0]], "bbox": [0, 0, 1, 1]},
                ],
            }
        ],
        "layout_blocks": [],
        "detections": [],
        "reading_order": [0, 1],
        "device": "cpu",
        "timings": {"total": 0.1},
    }

    md = document_to_markdown(doc)
    assert "First line" in md.splitlines()[0]
    assert "cell-1" in md


def test_analyze_document_markdown_flag(tmp_path, monkeypatch):
    weights = tmp_path / MODEL_FILENAME
    cfg = tmp_path / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)

    _stub_predictor(monkeypatch)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path, markdown=True)
    monkeypatch.setattr(ocr._document_processor, "load", lambda: None)
    monkeypatch.setattr(ocr._document_processor, "process_image", lambda image: "DOC")
    monkeypatch.setattr(mer_module, "document_to_markdown", lambda doc: f"md:{doc}")

    assert ocr.predict("dummy.png") == "md:DOC"


def test_analyze_document_json_flag(tmp_path, monkeypatch):
    weights = tmp_path / MODEL_FILENAME
    cfg = tmp_path / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)

    _stub_predictor(monkeypatch)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path, json_result=True)
    monkeypatch.setattr(ocr._document_processor, "load", lambda: None)

    line = LineResult(order=0, block_index=0, block_label="Text", text="hi", polygon=[[0, 0], [1, 0], [1, 1], [0, 1]], bbox=[0, 0, 1, 1])
    doc = DocumentResult(
        lines=[line],
        tables=[],
        layout_blocks=[],
        detections=[],
        reading_order=[0],
        device="cpu",
        timings={"total": 0.5},
    )
    monkeypatch.setattr(ocr._document_processor, "process_image", lambda image: doc)

    result = ocr.predict("dummy.png")
    assert result["device"] == "cpu"
    assert result["timings"]["total"] == 0.5
    assert result["lines"][0]["text"] == "hi"


def test_predict_returns_json_by_default(tmp_path, monkeypatch):
    weights = tmp_path / MODEL_FILENAME
    cfg = tmp_path / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)

    _stub_predictor(monkeypatch)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path)
    monkeypatch.setattr(ocr._document_processor, "load", lambda: None)

    line = LineResult(order=0, block_index=0, block_label="Text", text="hello", polygon=[[0, 0]], bbox=[0, 0, 1, 1])
    doc = DocumentResult(
        lines=[line],
        tables=[],
        layout_blocks=[],
        detections=[],
        reading_order=[0],
        device="cpu",
        timings=None,
    )
    monkeypatch.setattr(ocr._document_processor, "process_image", lambda image: doc)

    result = ocr.predict("dummy.png")
    assert isinstance(result, dict)
    assert result["device"] == "cpu"
    assert result["lines"][0]["text"] == "hello"
