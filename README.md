# Mer (មើល)

Mer (មើល) is a lightweight bilingual (Khmer/English) OCR recognizer built around my custom CNN-Transformer network exported to ONNX.  
This repository now focuses exclusively on line-level OCR — the Surya-powered layout, table, and LaTeX helpers have been removed, so the package is small and easy to embed anywhere you just need text from a single line image.

## Installation

```bash
pip install mer
```

## Getting started

```python
from mer import Mer, postprocess_text

ocr = Mer()
ocr.load()  # optional; kept for backwards compatibility

text = ocr.recognize_line("samples/sample_1.png")
print("Line text:", text)

# predict() is an alias for recognize_line()
json_result = ocr.predict("samples/sample_1.png", json_result=True)
print(json_result["text"])

# Optional extra cleanup
print(postprocess_text("ទៀតផង ។"))  # -> "ទៀតផង។"
```

## Configuration options

All options control the ONNX Runtime predictor:

- `device`: `"cpu"`, `"cuda"`, or specific device strings. Defaults to `"cuda"` with automatic CPU fallback.
- `providers`: optional explicit ONNX Runtime provider list. When omitted, providers are derived from `device`.
- `model_path`: point to a directory containing `khmer_ocr.onnx` and `config.json` to skip Hugging Face downloads.
- `cache_dir` / `repo_id`: control where artifacts are downloaded from Hugging Face Hub (`metythorn/ocr-stn-cnn-transformer-base` by default).
- `max_length`: override the configured maximum decoding length.
- `postprocess`: disable built-in whitespace cleanup if you prefer the raw model output.
- `json_result`: default return type for `predict()`. When `True`, `predict()` returns `{"text": ...}`; otherwise it returns a raw string. You can always override this per-call.

## Using local model files

If you already have the ONNX weights and config on disk, point `Mer` at the folder to skip any Hugging Face download:

```python
from mer import Mer

ocr = Mer(model_path="/path/to/local/model_dir", device="cpu")
ocr.load()
print(ocr.recognize_line("line.png"))
```

## Post-processing helper

`postprocess_text` is exposed as a standalone helper so you can reuse the same Khmer punctuation cleanup on your own strings:

```python
from mer import postprocess_text

assert postprocess_text(" ទៀត\tផង  ។ ") == "ទៀត ផង។"
```

## Sample data

The `samples/` directory contains a few PNGs you can use for quick manual testing. They are untouched and meant purely for experimentation with the line recognizer.
