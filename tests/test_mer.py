import json
from pathlib import Path

import pytest
from PIL import Image

from mer import Mer, postprocess_text
from mer.artifacts import ensure_artifacts
from mer.constants import MODEL_FILENAME, CONFIG_FILENAME
from mer import artifacts as artifacts_module
from mer import predictor as predictor_module


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


def _prepare_dummy_artifacts(tmp_path: Path) -> None:
    weights = tmp_path / MODEL_FILENAME
    cfg = tmp_path / CONFIG_FILENAME
    weights.write_text("weights", encoding="utf-8")
    _write_dummy_config(cfg)


def test_ensure_artifacts_uses_existing_files(tmp_path, monkeypatch):
    _prepare_dummy_artifacts(tmp_path)

    def fake_download(*args, **kwargs):
        raise AssertionError("Download should not be called when files exist")

    monkeypatch.setattr(artifacts_module, "hf_hub_download", fake_download)

    artifacts = ensure_artifacts(cache_dir=tmp_path)
    assert artifacts.weights.exists()
    assert artifacts.config.exists()


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
    _prepare_dummy_artifacts(local_dir)

    def fake_download(*args, **kwargs):
        raise AssertionError("Download should not be called when local_dir is provided")

    monkeypatch.setattr(artifacts_module, "hf_hub_download", fake_download)

    artifacts = ensure_artifacts(cache_dir=tmp_path, local_dir=local_dir)
    assert artifacts.base_dir == local_dir
    assert artifacts.weights.exists()
    assert artifacts.config.exists()


def test_ensure_artifacts_raises_when_local_missing(tmp_path):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        ensure_artifacts(local_dir=local_dir)


def test_mer_recognize_line_uses_predictor(tmp_path, monkeypatch):
    _prepare_dummy_artifacts(tmp_path)
    _stub_predictor(monkeypatch, return_value="dummy-text")

    sample_img = tmp_path / "line.png"
    Image.new("RGB", (10, 10), color="white").save(sample_img)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path)
    assert ocr.recognize_line(sample_img) == "dummy-text"


def test_mer_postprocess_flag(tmp_path, monkeypatch):
    _prepare_dummy_artifacts(tmp_path)
    raw_text = "raw\ttext"
    _stub_predictor(monkeypatch, return_value=raw_text)

    sample_img = tmp_path / "line.png"
    Image.new("RGB", (10, 10), color="white").save(sample_img)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path, postprocess=False)
    assert ocr.recognize_line(sample_img) == raw_text


def test_mer_recognize_line_json_result(tmp_path, monkeypatch):
    _prepare_dummy_artifacts(tmp_path)
    _stub_predictor(monkeypatch, return_value="dummy-json-text")

    sample_img = tmp_path / "line.png"
    Image.new("RGB", (10, 10), color="white").save(sample_img)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path)
    result = ocr.recognize_line(sample_img, json_result=True)
    assert result == {"text": "dummy-json-text"}


def test_mer_predict_alias_respects_default_json_flag(tmp_path, monkeypatch):
    _prepare_dummy_artifacts(tmp_path)
    _stub_predictor(monkeypatch, return_value="alias-text")

    sample_img = tmp_path / "line.png"
    Image.new("RGB", (10, 10), color="white").save(sample_img)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path, json_result=True)
    result = ocr.predict(sample_img)
    assert result == {"text": "alias-text"}


def test_mer_predict_overrides_json_flag(tmp_path, monkeypatch):
    _prepare_dummy_artifacts(tmp_path)
    _stub_predictor(monkeypatch, return_value="overridden-text")

    sample_img = tmp_path / "line.png"
    Image.new("RGB", (10, 10), color="white").save(sample_img)

    ocr = Mer(cache_dir=tmp_path, model_path=tmp_path)
    result = ocr.predict(sample_img, json_result=True)
    assert result == {"text": "overridden-text"}
    assert ocr.predict(sample_img, json_result=False) == "overridden-text"


def test_mer_markdown_flag_raises(tmp_path):
    _prepare_dummy_artifacts(tmp_path)
    with pytest.raises(ValueError):
        Mer(cache_dir=tmp_path, model_path=tmp_path, markdown=True)


def test_postprocess_text():
    assert postprocess_text("ទៀតផង ។") == "ទៀតផង។"
    assert postprocess_text("a\tb") == "a b"
    assert postprocess_text("   spaced\n\tIndented") == "spaced\nIndented"
    assert postprocess_text("a   b  c") == "a b c"
    assert postprocess_text("") == ""
