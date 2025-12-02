import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merl import Merl, postprocess_text
from merl.artifacts import ensure_artifacts
from merl.constants import MODEL_FILENAME, CONFIG_FILENAME
from merl import artifacts as artifacts_module
from merl import predictor as predictor_module
from merl.surya import SuryaDocumentProcessor


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


def test_merl_recognize_line_uses_predictor(tmp_path, monkeypatch):
    weights = tmp_path / MODEL_FILENAME
    cfg = tmp_path / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)

    monkeypatch.setattr(predictor_module.Predictor, "_load_model", lambda self: object())
    monkeypatch.setattr(predictor_module.Predictor, "predict", lambda self, image: "dummy-text")

    sample_img = Path(__file__).resolve().parent.parent / "samples" / "image.png"
    ocr = Merl(cache_dir=tmp_path)
    assert ocr.recognize_line(sample_img) == "dummy-text"


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

    sample_img = Path(__file__).resolve().parent.parent / "samples" / "image.png"
    doc = processor.process_image(sample_img)
    assert doc.lines and doc.lines[0].text == "line-text"
    assert doc.reading_order == [0]
    assert doc.timings and "total" in doc.timings

    latex = processor.recognise_latex(sample_img)
    assert latex == "latex-text"


def test_postprocess_text():
    assert postprocess_text("ទៀតផង ។") == "ទៀតផង។"
    assert postprocess_text("a\tb") == "a b"
    assert postprocess_text("   spaced\n\tIndented") == "spaced\nIndented"
    assert postprocess_text("a   b  c") == "a b c"
    assert postprocess_text("") == ""
