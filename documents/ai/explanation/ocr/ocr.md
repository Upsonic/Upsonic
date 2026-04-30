---
name: ocr-subsystem
description: Use when working with the upsonic.ocr package to extract text from images and PDFs via the OCR orchestrator and provider-agnostic engines. Use when a user asks to convert PDFs/images to text, configure OCR engines, set per-page timeouts, switch providers via infer_provider, integrate OCR with loaders/knowledge_base, or debug OCR failures. Trigger when the user mentions OCR, OCRProvider, OCRConfig, OCRResult, OCRMetrics, OCRTextBlock, BoundingBox, EasyOCREngine, RapidOCREngine, TesseractOCREngine, DeepSeekOCREngine, DeepSeekOllamaOCREngine, PaddleOCREngine, PPStructureV3, PPChatOCRv4, PaddleOCR-VL, PyMuPDF, pdf2image, Poppler, Tesseract, vLLM, Ollama, layer_0/document_converter, prepare_file_for_ocr, OCRTimeoutError, or OCRUnsupportedFormatError.
---

# `src/upsonic/ocr/` — OCR Subsystem

This document describes the **`upsonic.ocr`** package: a self-contained Optical
Character Recognition layer that turns image and PDF files into raw text and
structured `OCRResult` objects. The module ships with a unified orchestrator
(`OCR`), a provider-agnostic abstract base class (`OCRProvider`), and seven
concrete engines covering classical, ONNX-based, transformer-based, and
vision-language OCR backends.

The package is intentionally split into two **layers**:

| Layer | Responsibility | Input | Output |
|-------|----------------|-------|--------|
| **Layer 0** | Document → images | path to PDF / PNG / JPEG / TIFF / BMP / WEBP | `List[PIL.Image.Image]` |
| **Layer 1** | Images → text | `PIL.Image.Image` | `OCRResult` (text + blocks + bbox) |

The orchestrator (`upsonic.ocr.ocr.OCR`) glues them together and adds
threading, async, timeouts, and metric accumulation around a chosen
Layer 1 engine.

---

## 1. What this folder is — OCR providers

The folder is the **canonical entry point for converting visual documents
into text** inside Upsonic. Every public surface is provider-neutral: a
caller picks an engine (e.g. `EasyOCREngine`, `PaddleOCREngine`,
`DeepSeekOCREngine`), wraps it in `OCR(...)`, and calls
`get_text(...)` / `process_file(...)`. The orchestrator handles:

- **File ingestion**: dispatches PDFs and images through Layer 0
  (`document_converter.convert_document`) which uses **PyMuPDF (fitz)** to
  rasterise PDF pages without requiring the Poppler system binary, and
  Pillow for native image formats.
- **Image normalisation**: EXIF-based rotation transpose, palette
  flattening, RGBA→RGB compositing on white, oversized image
  scale-down (default 3 MB JPEG cap).
- **Engine execution**: invokes `OCRProvider._process_image` per page,
  optionally wrapped in a hard thread-pool timeout.
- **Result aggregation**: stitches per-page text with `\n\n` separators,
  averages confidence scores, propagates page numbers into
  `OCRTextBlock.page_number`, and updates running `OCRMetrics`.
- **Lazy importing**: every heavyweight engine (PaddleOCR, vLLM,
  EasyOCR, RapidOCR, Ollama, Tesseract) is loaded on first attribute
  access via `__getattr__` so importing the package does not pull
  hundreds of MB of optional dependencies.

Supported file formats:

```
.png  .jpg  .jpeg  .bmp  .tiff  .tif  .webp        (images)
.pdf                                                 (multi-page documents)
```

(Layer 0 additionally accepts `.gif`; Layer 1 utils restricts itself to
the seven formats above.)

---

## 2. Folder layout (tree)

```
src/upsonic/ocr/
├── __init__.py                       # Lazy public API surface
├── ocr.py                            # OCR orchestrator + infer_provider()
├── base.py                           # OCRProvider ABC + dataclasses
├── exceptions.py                     # OCRError hierarchy
├── utils.py                          # File validation, pdf2image fallback,
│                                     #   load_image, preprocess_image
├── layer_0/
│   ├── __init__.py                   # (empty placeholder)
│   └── document_converter.py         # PyMuPDF-based PDF->images + optimisation
└── layer_1/
    ├── __init__.py                   # (empty placeholder)
    └── engines/
        ├── __init__.py               # Lazy engine registry
        ├── easyocr.py                # EasyOCREngine (deep-learning, 80+ lang)
        ├── rapidocr.py               # RapidOCREngine (ONNX runtime)
        ├── tesseract.py              # TesseractOCREngine (libtesseract)
        ├── deepseek.py               # DeepSeekOCREngine (vLLM local)
        ├── deepseek_ollama.py        # DeepSeekOllamaOCREngine (Ollama HTTP)
        └── paddleocr.py              # 4× PaddleOCR pipelines (general, structure,
                                      #   chat, vision-language)
```

---

## 3. Top-level files

### 3.1 `__init__.py` — Lazy public surface

The module deliberately avoids eager imports. It defines `__getattr__`
that resolves attribute lookups against four lazy registries:

| Registry helper | Members |
|-----------------|---------|
| `_get_base_classes()` | `OCR`, `infer_provider`, `OCRProvider`, `OCRConfig`, `OCRResult`, `OCRMetrics`, `OCRTextBlock`, `BoundingBox` |
| `_get_exception_classes()` | `OCRError`, `OCRProviderError`, `OCRFileNotFoundError`, `OCRUnsupportedFormatError`, `OCRProcessingError`, `OCRTimeoutError` |
| `_get_engine_classes()` | `EasyOCREngine`, `RapidOCREngine`, `TesseractOCREngine`, `DeepSeekOCREngine`, `DeepSeekOllamaOCREngine` |
| `_get_paddleocr_classes()` | `PaddleOCRConfig`, `PaddleOCREngine`, `PPStructureV3Engine`, `PPChatOCRv4Engine`, `PaddleOCRVLEngine`, plus aliases `PaddleOCR`, `PPStructureV3`, `PPChatOCRv4`, `PaddleOCRVL` |

A `TYPE_CHECKING` block mirrors the registries for static analysers.
The `paddleocr` registry is wrapped in `try/except ImportError` so the
package can be imported on machines without PaddlePaddle.

### 3.2 `ocr.py` — `OCR` orchestrator

The single class `OCR` composes a Layer 1 engine and (optionally) a
per-page hard timeout.

```python
class OCR:
    def __init__(
        self,
        layer_1_ocr_engine: OCRProvider,
        layer_1_timeout: Optional[float] = None,
    ): ...
```

Key methods (all sync wrappers call the async variant via `asyncio.run`):

| Method | Signature | Purpose |
|--------|-----------|---------|
| `get_text(path, **kw)` / `get_text_async` | → `str` | Convenience: extract plain text |
| `process_file(path, **kw)` / `process_file_async` | → `OCRResult` | Full pipeline, returns blocks + metadata |
| `_run_with_timeout(image, page_num)` | internal | Runs `_process_image` on a `ThreadPoolExecutor(max_workers=1)` and `asyncio.wait_for`; raises `OCRTimeoutError` on timeout |
| `get_metrics()` / `reset_metrics()` | → `OCRMetrics` / `None` | Forwards to engine's `_metrics` |
| `get_info()` | → `dict` | Forwards to engine |
| `name`, `supported_languages`, `config` | properties | Forwards to engine |

The orchestrator's pipeline (in `process_file_async`):

```python
images  = await asyncio.to_thread(convert_document, file_path, pdf_dpi=cfg.pdf_dpi)
for page_num, image in enumerate(images, 1):
    page_result = await self._run_with_timeout(image, page_num)  # or _process_image_async
    for block in page_result.blocks:
        block.page_number = page_num   # mutate in place
        all_blocks.append(block)
```

It then concatenates page texts (`"\n\n".join(...)`), averages
confidences, builds an `OCRResult`, and **mutates the engine's
`_metrics`** (total pages, characters, processing time, files
processed, average confidence).

### 3.3 `infer_provider(provider_name, **kwargs)` — string-keyed factory

A helper that maps friendly names to fully-qualified class paths:

| String key(s) | Resolves to |
|---------------|-------------|
| `easyocr` | `EasyOCREngine` |
| `rapidocr` | `RapidOCREngine` |
| `tesseract` | `TesseractOCREngine` |
| `deepseek`, `deepseek_ocr` | `DeepSeekOCREngine` |
| `deepseek_ollama` | `DeepSeekOllamaOCREngine` |
| `paddleocr`, `paddle` | `PaddleOCREngine` |
| `ppstructurev3`, `pp_structure_v3` | `PPStructureV3Engine` |
| `ppchatocrv4`, `pp_chat_ocr_v4` | `PPChatOCRv4Engine` |
| `paddleocr_vl`, `paddleocrvl` | `PaddleOCRVLEngine` |

Unknown keys raise `OCRError(error_code="UNKNOWN_PROVIDER")`.

### 3.4 `base.py` — types and ABC

Public dataclasses and Pydantic config:

```python
@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

@dataclass
class OCRTextBlock:
    text: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    page_number: int = 0
    language: Optional[str] = None

@dataclass
class OCRResult:
    text: str
    blocks: List[OCRTextBlock] = field(default_factory=list)
    confidence: float = 0.0
    page_count: int = 1
    processing_time_ms: float = 0.0
    provider: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]: ...

@dataclass
class OCRMetrics:
    total_pages: int = 0
    total_characters: int = 0
    average_confidence: float = 0.0
    processing_time_ms: float = 0.0
    provider: str = ""
    files_processed: int = 0

class OCRConfig(BaseModel):
    languages: List[str]              = ['en']
    confidence_threshold: float       = 0.0
    rotation_fix: bool                = False
    enhance_contrast: bool            = False
    remove_noise: bool                = False
    pdf_dpi: int                      = 300
    preserve_formatting: bool         = True
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
```

`OCRProvider` is the contract every engine implements:

| Member | Kind | Required | Notes |
|--------|------|----------|-------|
| `name` | abstract `@property` | ✅ | Lower-case identifier; used in `OCRResult.provider` |
| `supported_languages` | abstract `@property` | ✅ | List of language codes the engine knows about |
| `_validate_dependencies()` | abstract | ✅ | Print install instructions / raise `OCRProviderError` |
| `_get_reader()` | abstract | ✅ | Lazy, **thread-safe** (must use `self._reader_lock`) |
| `_process_image(image, **kw)` | abstract | ✅ | Sync core inference, returns `OCRResult` |
| `_process_image_async` | concrete | ❌ | Defaults to `asyncio.to_thread(self._process_image, ...)` |
| `process_file_async` / `process_file` | concrete | ❌ | High-level pipeline using `prepare_file_for_ocr` |
| `get_text_async` / `get_text` | concrete | ❌ | Convenience wrapper |
| `get_metrics` / `reset_metrics` / `get_info` | concrete | ❌ | Inspection helpers |

The base class **owns** `self._metrics` and `self._reader_lock`
(a `threading.Lock`) so subclasses can rely on the double-checked
locking idiom for engine creation.

### 3.5 `exceptions.py` — error hierarchy

```
OCRError(Exception)
├── OCRProviderError              (e.g. binary missing, lang pack missing)
├── OCRFileNotFoundError          (file path / not-a-file)
├── OCRUnsupportedFormatError     (extension not in whitelist)
├── OCRProcessingError            (runtime failures during conversion / inference)
└── OCRTimeoutError               (per-page hard timeout exceeded)
```

Every `OCRError` carries optional `error_code: str` and `original_error:
Exception` so callers can filter (`if isinstance(e, OCRTimeoutError):`)
or surface diagnostic chains.

### 3.6 `utils.py` — pdf2image-based path

Used **only by `OCRProvider.process_file_async`** in the base class
(not by the orchestrator's Layer 0). Provides a parallel
implementation of file → images using **pdf2image + Poppler** (rather
than PyMuPDF), preserving compatibility with pre-existing engine code:

| Function | Purpose |
|----------|---------|
| `check_dependencies()` | Verify Pillow available |
| `check_pdf_dependencies()` | Verify pdf2image available |
| `check_poppler_installed()` | Print OS-specific Poppler install help |
| `validate_file_path(path)` | Existence + is-file checks |
| `get_file_format(path)` | Whitelist extension; raise `OCRUnsupportedFormatError` |
| `is_pdf(path)` / `is_image(path)` | Boolean helpers |
| `load_image(path)` | EXIF-transpose + mode normalisation (P, RGBA, L, RGB) |
| `pdf_to_images(path, dpi=300)` | Wraps `pdf2image.convert_from_path`; on `PDFInfoNotInstalledError` invokes `check_poppler_installed()` |
| `detect_rotation(image)` | Variance-of-projection heuristic; returns one of `{0, 90, 180, 270}` |
| `rotate_image(image, angle)` | `image.rotate(-angle, expand=True)` |
| `preprocess_image(image, rotation_fix, enhance_contrast, remove_noise, target_size)` | Composite preprocessor (uses `PIL.ImageEnhance.Contrast(1.5)` and `PIL.ImageFilter.MedianFilter(size=3)`) |
| `prepare_file_for_ocr(path, rotation_fix, enhance_contrast, remove_noise, pdf_dpi)` | End-to-end: validate → branch on PDF/image → preprocess → list of images |

This module also exposes the format whitelists `SUPPORTED_IMAGE_FORMATS`,
`SUPPORTED_PDF_FORMATS`, and `ALL_SUPPORTED_FORMATS`.

---

## 4. Per-provider files

### 4.1 `layer_0/document_converter.py` — Layer 0 reference implementation

A **leaner, Poppler-free** alternative to `utils.prepare_file_for_ocr`,
used by the `OCR` orchestrator's `process_file_async`. Differences vs
`utils.py`:

- Uses **PyMuPDF (`fitz`)** for PDF rasterisation:
  ```python
  doc = fitz.open(str(pdf_path))
  for page_num in range(doc.page_count):
      mat   = fitz.Matrix(dpi/72, dpi/72)
      pix   = doc[page_num].get_pixmap(matrix=mat)
      img   = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
  ```
- **Optimises oversized images** in-memory: encodes to JPEG-quality-90,
  and if the encoded byte-length exceeds `MAX_FILE_SIZE` (3 MB),
  iteratively resizes by 10 % until it fits or hits a 200×200 floor.
  This protects downstream engines from OOM / latency spikes.
- Same EXIF transpose + mode normalisation (`P`, `RGBA`, others → `RGB`,
  keep `L` and `RGB`).
- Exposes both `convert_document(...)` (sync) and
  `convert_document_async(...)` (`asyncio.to_thread` wrapper).

### 4.2 `layer_1/engines/easyocr.py` — `EasyOCREngine`

| Attribute | Value |
|-----------|-------|
| `name` | `"easyocr"` |
| `supported_languages` | 50 codes including `en, zh, ja, ko, ar, ru, de, fr, es, pt, it, nl, pl, tr, hi, …` |
| Reader class | `easyocr.Reader(...)` |
| Init kwargs | `gpu`, `model_storage_directory`, `download_enabled` |

`_get_reader` patches `ssl._create_default_https_context =
ssl._create_unverified_context` while constructing the reader to allow
first-run model downloads behind corporate proxies, and restores the
original context in a `finally:`. It calls `reader.readtext(img,
detail=1, paragraph=False, min_size=10, text_threshold=0.7,
low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1.0)`
(all overridable via `**kwargs`). Bounding boxes are flattened from
4-point quads to axis-aligned `BoundingBox(x, y, width, height)` by
`(min/max)(x_coords)`.

### 4.3 `layer_1/engines/rapidocr.py` — `RapidOCREngine`

| Attribute | Value |
|-----------|-------|
| `name` | `"rapidocr"` |
| `supported_languages` | `en, ch, chinese_cht, japan, korean, ta, te, ka, latin, arabic, cyrillic, devanagari` |
| Reader class | `rapidocr_onnxruntime.RapidOCR` (falls back to `rapidocr_openvino.RapidOCR`) |
| Custom model paths | `det_model_path`, `rec_model_path`, `cls_model_path` |

Returns `(dt_boxes, rec_res, time_dict)` from RapidOCR. The engine
parses items as `[box_coords, text, confidence_str]` and converts the
confidence string to float defensively. `metadata={'processing_time':
time_dict}` is preserved on the result.

> **Note**: there is a mismatch in `_get_reader` where the keyword
> `'CLs.model_path'` (with a capital C-L-s) is used for `rec_model_path`.
> This is in the source as written.

### 4.4 `layer_1/engines/tesseract.py` — `TesseractOCREngine`

| Attribute | Value |
|-----------|-------|
| `name` | `"tesseract"` |
| `supported_languages` | 60+ ISO-639-3 codes (`eng`, `fra`, `deu`, `chi_sim`, …) |
| Reader class | `pytesseract` module (stateless) |
| Init kwargs | `tesseract_cmd`, `tessdata_dir` |

Build configuration string in `_get_tesseract_config(**kwargs)`:

| Tesseract flag | Default | kwarg |
|----------------|---------|-------|
| `--psm` | `3` (auto page seg) | `psm` |
| `--oem` | `3` (legacy + LSTM) | `oem` |
| `--tessdata-dir "<path>"` | only if `tessdata_dir` set | (constructor) |
| custom suffix | — | `custom_config` |

Two parallel calls per image: `pytesseract.image_to_data(...,
output_type=DICT)` for blocks/bboxes/confidence (divided by 100 to
0–1), and a second `pytesseract.image_to_string(...)` only when
`config.preserve_formatting=True`. Confidence below
`config.confidence_threshold` is filtered out.

`get_info()` is overridden to also return `tesseract_version` and
`available_languages` (queried at runtime via
`pytesseract.get_languages('')`).

### 4.5 `layer_1/engines/deepseek.py` — `DeepSeekOCREngine`

| Attribute | Value |
|-----------|-------|
| `name` | `"deepseek_ocr"` |
| `supported_languages` | 20 codes (`en, zh, ja, ko, es, fr, de, …`) |
| Reader class | `vllm.LLM(model="deepseek-ai/DeepSeek-OCR")` |
| Init kwargs | `model_name`, `prompt`, `temperature`, `max_tokens`, `ngram_size`, `window_size` |

Uses **vLLM** (locally) with the DeepSeek-OCR multimodal model.

- Defaults: `prompt="<image>\nFree OCR."`, `temperature=0.0`,
  `max_tokens=8192`.
- If `vllm.model_executor.models.deepseek_ocr.NGramPerReqLogitsProcessor`
  is importable, it wires `logits_processors=[NGramPerReqLogitsProcessor]`
  on the LLM and `extra_args={ngram_size, window_size,
  whitelist_token_ids={128821, 128822}}` (HTML `<td>`/`</td>` tokens) on
  the `SamplingParams`.
- If vLLM does not recognise `DeepseekOCRForCausalLM`, it raises
  `OCRProviderError(error_code="UNSUPPORTED_MODEL_ARCHITECTURE")` with a
  fallback recommendation list.
- Override of `process_file_async` adds **batched multi-page PDF
  inference** via `process_images_batch_async` (single
  `llm.generate([model_input_1, model_input_2, ...])` call, then
  per-page `OCRResult`s with `metadata['batch_index']`).
- DeepSeek-OCR does not return per-token confidence; the engine pins
  every confidence to `1.0`.

### 4.6 `layer_1/engines/deepseek_ollama.py` — `DeepSeekOllamaOCREngine`

| Attribute | Value |
|-----------|-------|
| `name` | `"deepseek_ollama_ocr"` |
| `supported_languages` | same 20-code set as `DeepSeekOCREngine` |
| Reader class | `ollama.Client(host=...)` |
| Init kwargs | `host="http://localhost:11434"`, `model="deepseek-ocr:3b"`, `prompt`, `timeout=60.0` |

This is the **lightweight** DeepSeek path: it shells out to a running
Ollama daemon over HTTP rather than loading vLLM in-process. Per
image, it:

1. Saves the PIL image to a `tempfile.NamedTemporaryFile(suffix='.png')`.
2. Calls `client.chat(model=..., messages=[{'role': 'user', 'content':
   prompt, 'images': [tmp_path]}], stream=True)`.
3. Streams chunks in a **separate thread** so a `threading.Event`
   stop-flag can interrupt at the configured `timeout`.
4. On timeout sets `metadata['timeout_reached']=True`, otherwise
   returns the accumulated streamed content.
5. Removes the temp file in `finally:`.

`process_file_async` is overridden to preserve streaming metadata
(`chunk_count`, `processing_time`, `timeout_duration`,
`timeout_reached`) — the base implementation would otherwise overwrite
it.

### 4.7 `layer_1/engines/paddleocr.py` — Four PaddleOCR pipelines

This file is the largest in the package (~1480 lines) and contains
five classes (one base + four concrete engines):

| Class | `name` | Underlying PaddleOCR class | Use-case |
|-------|--------|----------------------------|----------|
| `BasePaddleOCREngine` | (abstract) | — | Shared init, params builder, predict-result extractor |
| `PaddleOCREngine` | `"PaddleOCR-{ocr_version}"` | `paddleocr.PaddleOCR` | General-purpose text OCR (PP-OCR v3/v4/v5) |
| `PPStructureV3Engine` | `"PP-StructureV3"` | `paddleocr.PPStructureV3` | Layout + table + chart + formula + seal recognition |
| `PPChatOCRv4Engine` | `"PP-ChatOCRv4"` | `paddleocr.PPChatOCRv4Doc` | RAG-style chat over docs, key-value extraction |
| `PaddleOCRVLEngine` | `"PaddleOCR-VL"` | `paddleocr.PaddleOCRVL` | Vision-language; outputs Markdown |

Aliases at module bottom:

```python
PaddleOCR        = PaddleOCREngine
PPStructureV3    = PPStructureV3Engine
PPChatOCRv4      = PPChatOCRv4Engine
PaddleOCRVL      = PaddleOCRVLEngine
```

`PaddleOCRConfig(OCRConfig)` extends the base config with **30+ Paddle-
specific** Optional fields covering model paths, batch sizes, feature
toggles (`use_doc_orientation_classify`, `use_doc_unwarping`,
`use_textline_orientation`), text-detection thresholds (`text_det_thresh`,
`text_det_box_thresh`, `text_det_unclip_ratio`), recognition settings
(`text_rec_score_thresh`, `return_word_box`), and the `lang` /
`ocr_version` selectors.

`BasePaddleOCREngine._build_paddle_params(config, **kwargs)` strips
the seven base `OCRConfig` fields (`languages`, `confidence_threshold`,
`rotation_fix`, `enhance_contrast`, `remove_noise`, `pdf_dpi`,
`preserve_formatting`) and pydantic internals before forwarding to
PaddleOCR.

`_extract_paddle_predict_result(paddle_result)` walks the dict-like
result objects produced by `paddle.predict()`:

```python
res.get('rec_texts')   # list[str]
res.get('rec_scores')  # list[float]
res.get('dt_polys')    # list[list[(x, y)]]
res.get('rec_boxes')   # list (unused for bbox)
```

For each text it builds an `OCRTextBlock(text, confidence, bbox,
page_number)`. `dt_polys` polygons are reduced to axis-aligned bboxes
via `min/max` of the four corner coordinates.

**`PaddleOCREngine.process_file_async`** is overridden to bypass
Layer 0's PyMuPDF path and use Paddle's native PDF support: it
`pdf_to_images(...)` (via `utils.py`) at `min(pdf_dpi, 200)`, saves each
page to a temp PNG, and concatenates all pages into a single `predict`
call sequence.

**`PPChatOCRv4Engine`** exposes a richer surface for RAG-style
workflows:

| Method | Purpose |
|--------|---------|
| `visual_predict(input, **flags)` | Returns `[{visual_info, layout_parsing_result}, …]` with table/seal-aware layout |
| `visual_predict_iter(input, ...)` | Streaming/iterator version |
| `build_vector(visual_info, min_characters=3500, block_size=300, ...)` | Embed visual chunks; returns vector dict |
| `mllm_pred(input, key_list, mllm_chat_bot_config)` | Multi-modal LLM pass over selected keys |
| `chat(key_list, visual_info, use_vector_retrieval=True, vector_info, ...)` | Final K-V extraction with optional retrieval |
| `save_vector` / `load_vector` | Persist FAISS-style vectors |
| `save_visual_info_list` / `load_visual_info_list` | Persist intermediate visual info |

**`PaddleOCRVLEngine`** wraps the vision-language pipeline; it adds
VLM sampling controls (`temperature`, `top_p`, `repetition_penalty`,
`min_pixels`, `max_pixels`, `vl_rec_backend`, `vl_rec_server_url`,
`vl_rec_max_concurrency`) and exposes
`concatenate_markdown_pages(markdown_list)` to glue per-page Markdown.

---

## 5. Cross-file relationships

```
                ┌──────────────────────────────────────────────────┐
                │                                                  │
                │        upsonic.ocr (lazy __getattr__)            │
                │                                                  │
                └──────────────┬───────────────────────────────────┘
                               │
               ┌───────────────┼─────────────────────────┐
               │               │                         │
               ▼               ▼                         ▼
         ocr.py           base.py                  exceptions.py
        ┌──────┐        ┌─────────────┐          ┌──────────────┐
        │ OCR  │──uses─▶│ OCRProvider │          │ OCRError ... │
        │      │        │ (ABC)       │◀─raises──│              │
        │      │        │ OCRConfig   │          └──────────────┘
        │      │        │ OCRResult   │
        │      │        │ OCRMetrics  │
        │      │        │ OCRTextBlock│
        │      │        │ BoundingBox │
        │      │        └──────┬──────┘
        │      │               │ inherited by
        │      │               │
        │      │               ▼
        │      │     ┌────────────────────────────────────────────┐
        │      │     │ layer_1/engines/                           │
        │      │     │   easyocr.py    EasyOCREngine             │
        │      │     │   rapidocr.py   RapidOCREngine            │
        │      │     │   tesseract.py  TesseractOCREngine        │
        │      │     │   deepseek.py   DeepSeekOCREngine         │
        │      │     │   deepseek_ollama.py DeepSeekOllamaOCR... │
        │      │     │   paddleocr.py  PaddleOCREngine,          │
        │      │     │                 PPStructureV3Engine,       │
        │      │     │                 PPChatOCRv4Engine,         │
        │      │     │                 PaddleOCRVLEngine          │
        │      │     └─────────────┬──────────────────────────────┘
        │      │                   │ each provider's process_file
        │      │                   │ may call:
        │      │ pipeline calls    ▼
        │      │             utils.py
        │      │             ┌────────────────┐
        │      │             │ pdf2image      │
        │      │             │ + Pillow       │
        │      │             │ preprocess     │
        │      │             └────────────────┘
        │      │
        │      └─pipeline─▶ layer_0/document_converter.py
        │                  ┌─────────────────────────────┐
        │                  │ PyMuPDF (fitz)              │
        │                  │ + Pillow                    │
        │                  │ + 3 MB JPEG cap optimiser   │
        │                  └─────────────────────────────┘
        ▼
    infer_provider()
    (string → OCR)
```

Important nuances:

1. **Two parallel Layer-0 implementations** exist:
   - `layer_0/document_converter.convert_document` (PyMuPDF, no Poppler)
     used by the `OCR` orchestrator.
   - `utils.prepare_file_for_ocr` (pdf2image + Poppler) used by every
     `OCRProvider.process_file_async` call when invoked **directly** on
     an engine (i.e. without the orchestrator).
   Engines that override `process_file_async` (`PaddleOCREngine`,
   `DeepSeekOCREngine`, `DeepSeekOllamaOCREngine`) usually pick the
   `utils.py` path explicitly.
2. **Metrics are mutated by both** the orchestrator (`OCR.process_file_async`)
   and the engine itself (`OCRProvider.process_file_async`). Pick one
   call site per workflow to avoid double-counting.
3. **Thread safety** is centralised on `OCRProvider._reader_lock`.
   Subclasses must use the double-checked-locking pattern or risk
   creating multiple heavyweight reader instances under concurrent load.
4. The `OCR` orchestrator validates the engine via
   `isinstance(layer_1_ocr_engine, OCRProvider)` and raises
   `OCRError(error_code="INVALID_PROVIDER")` otherwise.

---

## 6. Public API

The package exports the following symbols (from `__init__.__all__`):

```python
from upsonic.ocr import (
    # Orchestrator
    OCR,
    infer_provider,

    # Base types
    OCRProvider,
    OCRConfig,
    OCRResult,
    OCRMetrics,
    OCRTextBlock,
    BoundingBox,

    # Exceptions
    OCRError,
    OCRProviderError,
    OCRFileNotFoundError,
    OCRUnsupportedFormatError,
    OCRProcessingError,
    OCRTimeoutError,

    # Engines (always available)
    EasyOCREngine,
    RapidOCREngine,
    TesseractOCREngine,
    DeepSeekOCREngine,
    DeepSeekOllamaOCREngine,

    # Engines (only if `paddleocr` is installed)
    PaddleOCRConfig,
    PaddleOCREngine,
    PPStructureV3Engine,
    PPChatOCRv4Engine,
    PaddleOCRVLEngine,
    PaddleOCR,        # alias of PaddleOCREngine
    PPStructureV3,    # alias of PPStructureV3Engine
    PPChatOCRv4,      # alias of PPChatOCRv4Engine
    PaddleOCRVL,      # alias of PaddleOCRVLEngine
)
```

### 6.1 Minimal usage

```python
from upsonic.ocr import OCR, EasyOCREngine

engine = EasyOCREngine(languages=['en', 'tr'], gpu=False)
ocr    = OCR(layer_1_ocr_engine=engine, layer_1_timeout=30.0)

text   = ocr.get_text("invoice.pdf")               # str
print(text)
```

### 6.2 Detailed result with bounding boxes

```python
from upsonic.ocr import OCR, TesseractOCREngine, OCRConfig

cfg    = OCRConfig(languages=['eng'], confidence_threshold=0.6,
                   rotation_fix=True, enhance_contrast=True)
engine = TesseractOCREngine(config=cfg, psm=6)     # PSM forwarded via **kwargs
ocr    = OCR(layer_1_ocr_engine=engine)

result = ocr.process_file("scan.png")
print(result.confidence, result.page_count, len(result.blocks))
for b in result.blocks:
    print(f"[p{b.page_number}] {b.text!r} @ {b.bbox} conf={b.confidence:.2f}")
```

### 6.3 String-keyed factory

```python
from upsonic.ocr import infer_provider

ocr = infer_provider("paddleocr", lang='en', ocr_version='PP-OCRv5')
md  = ocr.get_text("research_paper.pdf")
```

### 6.4 PaddleOCR-VL → Markdown

```python
from upsonic.ocr import OCR, PaddleOCRVLEngine

vl  = PaddleOCRVLEngine(use_layout_detection=True,
                         use_chart_recognition=True,
                         format_block_content=True,
                         vl_rec_backend='local')
ocr = OCR(layer_1_ocr_engine=vl)
md  = ocr.get_text("paper.pdf")   # markdown-formatted output
```

### 6.5 PP-ChatOCRv4 — chat over a document

```python
from upsonic.ocr.layer_1.engines.paddleocr import PPChatOCRv4Engine

eng = PPChatOCRv4Engine(use_table_recognition=True,
                         use_seal_recognition=True,
                         mllm_chat_bot_config={'api_key': 'sk-...'})

visual = eng.visual_predict("invoice.pdf")
vector = eng.build_vector(visual, min_characters=3500, block_size=300)
answer = eng.chat(key_list=['vendor', 'invoice_number', 'total'],
                   visual_info=visual,
                   vector_info=vector,
                   use_vector_retrieval=True)
print(answer)
```

### 6.6 Hard timeout

```python
from upsonic.ocr import OCR, EasyOCREngine, OCRTimeoutError

ocr = OCR(EasyOCREngine(), layer_1_timeout=10.0)   # seconds per page
try:
    text = ocr.get_text("huge_scan.pdf")
except OCRTimeoutError as e:
    print("Page timed out:", e)
```

### 6.7 Inspecting metrics

```python
m = ocr.get_metrics()
print(m.files_processed, m.total_pages, m.total_characters,
      m.average_confidence, m.processing_time_ms)
ocr.reset_metrics()
```

---

## 7. Integration with rest of Upsonic (loaders / knowledge_base)

The `upsonic.ocr` package is **not currently imported** by
`upsonic.loaders` or `upsonic.knowledge_base`. PDF loaders that need
OCR call **RapidOCR directly** rather than going through
`upsonic.ocr.OCR`. Here is what each loader does:

| Loader | OCR backend | When OCR runs |
|--------|------------|---------------|
| `loaders/pdf.py` (PyPDF) | `from rapidocr_onnxruntime import RapidOCR; OCR_ENGINE = RapidOCR()` | `extraction_mode in ("ocr_only", "hybrid")` → `_perform_ocr(page)` extracts page images and runs OCR per image in a thread pool |
| `loaders/pymupdf.py` | same module-level `OCR_ENGINE = RapidOCR()` | `if "ocr" in self.config.extraction_mode:` → `_perform_ocr` uses PyMuPDF's image extraction then RapidOCR |
| `loaders/pdfplumber.py` | same `OCR_ENGINE = RapidOCR()` | `extraction_mode in ("ocr_only", "hybrid")` → renders page to image, then OCR |
| `loaders/docling.py` | Configures Docling's own OCR pipeline (`RapidOcrOptions` or `TesseractCliOcrOptions`) | Driven by `config.ocr_enabled`, `config.ocr_force_full_page`, `config.ocr_backend`, `config.ocr_lang`, `config.ocr_backend_engine`, `config.ocr_text_score` |

`loaders/config.py` defines the `extraction_mode` field on the
`PdfLoaderConfig` (and PyMuPDF/pdfplumber siblings):

```python
extraction_mode: Literal["hybrid", "text_only", "ocr_only"] = "hybrid"
```

— and additional Docling-specific OCR knobs (`ocr_enabled`,
`ocr_force_full_page`, `ocr_backend`, `ocr_lang`,
`ocr_backend_engine`, `ocr_text_score`).

**Recommended integration pattern** (when wiring `upsonic.ocr` into a
custom loader):

```python
from upsonic.ocr import OCR, RapidOCREngine
from upsonic.knowledge_base import KnowledgeBase
from upsonic.schemas.data_models import Document  # hypothetical

ocr = OCR(RapidOCREngine(languages=['en', 'ch']))

def load_image_as_document(path: str) -> Document:
    result = ocr.process_file(path)
    return Document(
        content=result.text,
        metadata={
            "source": path,
            "ocr_provider": result.provider,
            "ocr_confidence": result.confidence,
            "ocr_page_count": result.page_count,
            "ocr_processing_time_ms": result.processing_time_ms,
        },
    )

docs = [load_image_as_document(p) for p in image_paths]
kb   = KnowledgeBase(documents=docs)
```

Because `OCRResult.metadata` already carries `file_path` and the
serialised `OCRConfig`, downstream RAG steps can inspect provenance
without re-running inference.

---

## 8. End-to-end flow

The complete flow when `OCR(EasyOCREngine()).get_text("scan.pdf")` is
called:

```
┌──────────────────────────────────────────────────────────────────────┐
│ user code:                                                           │
│     ocr.get_text("scan.pdf")                                         │
└──────────────┬───────────────────────────────────────────────────────┘
               │  asyncio.run
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OCR.get_text_async                                                   │
│   └─▶ OCR.process_file_async("scan.pdf")                             │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 0: convert_document("scan.pdf", pdf_dpi=cfg.pdf_dpi)           │
│   1. _validate(path) — extension whitelist, file exists              │
│   2. branch:                                                         │
│        is PDF → _pdf_to_images(path, dpi)                            │
│           - fitz.open(path)                                          │
│           - for each page: pixmap → PIL.Image                        │
│        is image → _load_image(path)                                  │
│           - PIL.Image.open + ImageOps.exif_transpose                 │
│           - mode P/RGBA/other → RGB                                  │
│   3. for each image: _optimize_image(img, max_bytes=3 MB)            │
│        - JPEG-90 encode, iteratively scale 90 % until ≤ 3 MB         │
│   4. return List[PIL.Image]                                          │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 1: for each (page_num, image) in enumerate(images, 1):         │
│                                                                      │
│   if layer_1_timeout is set:                                         │
│       executor = ThreadPoolExecutor(max_workers=1)                   │
│       future   = loop.run_in_executor(executor,                      │
│                       engine._process_image, image)                  │
│       page_result = await asyncio.wait_for(future, timeout)          │
│         └─ on TimeoutError → raise OCRTimeoutError                   │
│   else:                                                              │
│       page_result = await engine._process_image_async(image)         │
│                                                                      │
│   engine._process_image(image):                                      │
│     ┌─ EasyOCREngine ────────────────────────────────────────────┐  │
│     │  reader = self._get_reader()  # double-checked-locking      │  │
│     │  arr    = np.array(image)                                   │  │
│     │  results = reader.readtext(arr, detail=1, ...)              │  │
│     │  for (bbox_quad, text, conf) in results:                    │  │
│     │      if conf < cfg.confidence_threshold: continue           │  │
│     │      bbox  = BoundingBox(min/max of quad, conf)             │  │
│     │      blocks.append(OCRTextBlock(text, conf, bbox))          │  │
│     │  return OCRResult(text=" ".join, blocks, avg_conf, ...)     │  │
│     └────────────────────────────────────────────────────────────┘  │
│                                                                      │
│   for block in page_result.blocks:                                   │
│       block.page_number = page_num    # mutation                     │
│       all_blocks.append(block)                                       │
│   all_text_parts.append(page_result.text)                            │
│   total_confidence += page_result.confidence                         │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Combine:                                                             │
│   combined_text = "\n\n".join(all_text_parts)                        │
│   avg_confidence = total_confidence / len(images)                    │
│   processing_time = (time.time() - start) * 1000                     │
│                                                                      │
│ result = OCRResult(                                                  │
│   text=combined_text,                                                │
│   blocks=all_blocks,                                                 │
│   confidence=avg_confidence,                                         │
│   page_count=len(images),                                            │
│   processing_time_ms=processing_time,                                │
│   provider=engine.name,                                              │
│   metadata={"file_path": ..., "config": cfg.model_dump()},           │
│ )                                                                    │
│                                                                      │
│ Update engine._metrics:                                              │
│   total_pages       += len(images)                                   │
│   total_characters  += len(combined_text)                            │
│   processing_time_ms += processing_time                              │
│   files_processed   += 1                                             │
│   average_confidence = mean(b.confidence for b in all_blocks)        │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OCR.get_text_async returns result.text                               │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.1 Variant: engine called directly (no orchestrator)

When a caller invokes `engine.process_file("scan.pdf")` directly
instead of wrapping in `OCR(...)`, the flow shifts to
`OCRProvider.process_file_async`:

```
engine.process_file("scan.pdf")
  └─▶ asyncio.run( engine.process_file_async(...) )
       └─▶ utils.prepare_file_for_ocr(path,
                                      rotation_fix=cfg.rotation_fix,
                                      enhance_contrast=cfg.enhance_contrast,
                                      remove_noise=cfg.remove_noise,
                                      pdf_dpi=cfg.pdf_dpi)
            └─▶ pdf2image.convert_from_path  (requires Poppler)
                └─▶ preprocess_image  (rotation / contrast / median filter)
       └─▶ for each image: engine._process_image_async(image)
       └─▶ same combine + mutate-metrics step as above
```

This is the path used by override-implementations in
`PaddleOCREngine._process_file_sync`, `DeepSeekOCREngine.process_file_async`,
and `DeepSeekOllamaOCREngine.process_file_async`. The two paths are
**not interchangeable** — orchestrator path uses PyMuPDF (no Poppler),
direct path uses pdf2image (Poppler required).

### 8.2 Failure modes per stage

| Stage | Exception | Where raised |
|-------|-----------|--------------|
| Bad engine type | `OCRError("INVALID_PROVIDER")` | `OCR.__init__` |
| Missing file | `OCRFileNotFoundError("FILE_NOT_FOUND")` | `_validate` / `validate_file_path` |
| Wrong extension | `OCRUnsupportedFormatError("UNSUPPORTED_FORMAT")` | `_validate` / `get_file_format` |
| PyMuPDF missing | `OCRError("MISSING_DEPENDENCY")` | `_pdf_to_images` |
| Poppler missing | `OCRProcessingError("POPPLER_NOT_INSTALLED")` (with help banner) | `pdf_to_images` |
| Engine binary missing | `OCRProviderError("TESSERACT_NOT_INSTALLED")`, `("VLLM_NOT_AVAILABLE")`, `("OLLAMA_NOT_AVAILABLE")`, `("PYTESSERACT_NOT_AVAILABLE")` | each engine's `_validate_dependencies` / `_get_reader` |
| Lang pack missing | `OCRProviderError("UNSUPPORTED_LANGUAGE")` | engine `_get_reader` (Tesseract / EasyOCR / RapidOCR) |
| Model arch unsupported | `OCRProviderError("UNSUPPORTED_MODEL_ARCHITECTURE")` | DeepSeek-OCR vLLM init |
| Layer-1 inference error | `OCRProcessingError("<engine>_PROCESSING_FAILED")` | each engine's `_process_image` |
| Per-page hard timeout | `OCRTimeoutError("LAYER1_TIMEOUT")` | `OCR._run_with_timeout` |

All exceptions carry `error_code` and (when wrapping a third-party
exception) `original_error` so callers can build deterministic
diagnostic plumbing.

---

## Appendix: file → key symbols cheat-sheet

| File | Top-level symbols |
|------|-------------------|
| `__init__.py` | `__getattr__`, `__all__` (lazy registries) |
| `ocr.py` | `OCR`, `infer_provider` |
| `base.py` | `BoundingBox`, `OCRTextBlock`, `OCRResult`, `OCRMetrics`, `OCRConfig`, `OCRProvider` |
| `exceptions.py` | `OCRError`, `OCRProviderError`, `OCRFileNotFoundError`, `OCRUnsupportedFormatError`, `OCRProcessingError`, `OCRTimeoutError` |
| `utils.py` | `SUPPORTED_IMAGE_FORMATS`, `SUPPORTED_PDF_FORMATS`, `ALL_SUPPORTED_FORMATS`, `validate_file_path`, `get_file_format`, `is_pdf`, `is_image`, `load_image`, `pdf_to_images`, `detect_rotation`, `rotate_image`, `preprocess_image`, `prepare_file_for_ocr`, `check_dependencies`, `check_pdf_dependencies`, `check_poppler_installed` |
| `layer_0/document_converter.py` | `IMAGE_EXTENSIONS`, `PDF_EXTENSIONS`, `ALL_EXTENSIONS`, `MAX_FILE_SIZE`, `convert_document`, `convert_document_async` |
| `layer_1/engines/easyocr.py` | `EasyOCREngine` |
| `layer_1/engines/rapidocr.py` | `RapidOCREngine` |
| `layer_1/engines/tesseract.py` | `TesseractOCREngine` |
| `layer_1/engines/deepseek.py` | `DeepSeekOCREngine` |
| `layer_1/engines/deepseek_ollama.py` | `DeepSeekOllamaOCREngine` |
| `layer_1/engines/paddleocr.py` | `PaddleOCRConfig`, `BasePaddleOCREngine`, `PaddleOCREngine`, `PPStructureV3Engine`, `PPChatOCRv4Engine`, `PaddleOCRVLEngine`, plus aliases `PaddleOCR`, `PPStructureV3`, `PPChatOCRv4`, `PaddleOCRVL` |
