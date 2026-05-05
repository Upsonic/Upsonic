---
name: document-loaders
description: Use when ingesting documents into Upsonic's RAG pipeline, picking or configuring a loader for PDFs, DOCX, CSV, JSON, XML, YAML, Markdown, HTML, or URLs, or extending the loader registry. Use when a user asks to load files into a KnowledgeBase, configure OCR/table extraction, split CSVs by row, parse JSON via JQ, walk XML XPath, chunk PDFs with Docling, or troubleshoot LoaderFactory behavior. Trigger when the user mentions BaseLoader, LoaderConfig, LoaderFactory, create_loader, load_document, TextLoader, CSVLoader, PdfLoader, PyMuPDFLoader, PdfPlumberLoader, DOCXLoader, JSONLoader, XMLLoader, YAMLLoader, MarkdownLoader, HTMLLoader, DoclingLoader, Document, document ingestion, RAG loaders, extraction_mode, split_by_xpath, split_by_jq_query, content_synthesis_mode, rapidocr, or docling.
---

# `src/upsonic/loaders/` — Document Loaders for RAG

## 1. What this folder is

The `loaders` package is the **document ingestion layer** of Upsonic's RAG
pipeline. It is the first stage that turns "a thing on disk or on the web"
(a PDF, a DOCX, a CSV, a URL, a YAML config, …) into an in‑memory
`upsonic.schemas.data_models.Document` object that the rest of the RAG stack
(`text_splitter` → embedding → vector DB) can consume.

Each concrete loader knows:

- **Which file extensions it handles** (`get_supported_extensions()`).
- **How to extract textual content** from that format (digital text, OCR,
  table extraction, XPath / JQ queries, etc.).
- **How to build per‑document metadata** (file size, page count, headings,
  PDF info dictionary, HTML `<meta>` tags, …).
- **How to deduplicate** sources within one loader instance (via an MD5 of the
  absolute path).
- **How to honor a unified configuration model** (`LoaderConfig` subclasses
  in `config.py`) for cleaning, error handling, max file size, custom metadata.

A `LoaderFactory` then sits on top, intelligently picking the right loader
for a given source (path, directory, URL, raw string content) and applying
size‑aware optimizations (e.g. drop OCR + lower DPI on huge PDFs, switch CSV
to concatenated mode on huge CSVs).

The output is always a `List[Document]`, where every `Document` has:

```python
class Document(BaseModel):
    content: str               # raw extracted text
    metadata: Dict[str, Any]   # source path, content_type, loader_type, …
    document_id: str           # md5 of absolute path (deterministic)
    doc_content_hash: str      # filled in later, for change detection
```

This document is then fed to a `text_splitter` to become `Chunk`s.

---

## 2. Folder layout

```
src/upsonic/loaders/
├── __init__.py            # lazy public API surface (loaders + factory + configs)
├── base.py                # BaseLoader ABC, path resolution, metadata helpers
├── config.py              # Pydantic LoaderConfig + 12 per‑loader subclasses
├── factory.py             # LoaderFactory + module‑level helpers
│
├── text.py                # TextLoader        (.txt, .py, .js, .log, .rst, …)
├── csv.py                 # CSVLoader         (.csv)
├── json.py                # JSONLoader        (.json, .jsonl) — JQ‑driven
├── xml.py                 # XMLLoader         (.xml)         — XPath‑driven
├── yaml.py                # YAMLLoader        (.yaml, .yml)  — JQ‑driven
├── markdown.py            # MarkdownLoader    (.md, .markdown) — heading split
├── html.py                # HTMLLoader        (.html, .htm, .xhtml + URLs)
│
├── pdf.py                 # PdfLoader         (.pdf, pypdf-based)
├── pymupdf.py             # PyMuPDFLoader     (.pdf, fitz/PyMuPDF)
├── pdfplumber.py          # PdfPlumberLoader  (.pdf, pdfplumber, table-rich)
│
├── docx.py                # DOCXLoader        (.docx)
└── docling.py             # DoclingLoader     (universal: pdf/docx/xlsx/pptx/
                           #                    html/md/asciidoc/csv/images/URLs)
```

Three things are conspicuously *not* in this folder:

- **`Document` / `Chunk` schemas** live in `src/upsonic/schemas/data_models.py`.
- **OCR engines** live in `src/upsonic/ocr/`. Loaders only use `rapidocr` /
  `tesseract` directly via optional imports (PDF loaders) or via Docling.
- **Chunking / splitting** lives in `src/upsonic/text_splitter/`. Loaders never
  chunk — they emit one `Document` per logical file or, for some loaders, one
  per row / record / chunk‑of‑rows / heading section.

---

## 3. Top‑level files

### 3.1 `base.py` — `BaseLoader` ABC

`BaseLoader` is a small but opinionated abstract base class that:

- Stores the `LoaderConfig` and an internal `_processed_document_ids: set`
  for instance‑level deduplication.
- Declares the four abstract entry points every loader must implement:
  `load`, `aload`, `batch`, `abatch`.
- Declares the abstract classmethod `get_supported_extensions()`.
- Provides several concrete helpers that subclasses reuse heavily:

| Helper                          | Purpose                                                                                                               |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `can_load(source)` (classmethod)| Default: file exists and `suffix.lower()` is in `get_supported_extensions()`. Overridden by `HTMLLoader` / `DoclingLoader` to also accept URLs. |
| `_generate_document_id(path)`   | `md5(absolute_path)` → deterministic `document_id` for `Document`.                                                    |
| `_handle_loading_error(src, e)` | Honors `config.error_handling` (`"raise"` / `"warn"` / `"ignore"`), returning `[]` for `warn`/`ignore`.               |
| `_create_metadata(path)`        | Builds `source`, `document_name`, `file_path`, `file_size`, ctime/mtime, `file_extension`, `content_type`, `loader_type`, plus `config.custom_metadata`. |
| `_check_file_size(path)`        | Skips files larger than `config.max_file_size` and warns.                                                             |
| `_get_supported_files_from_directory(dir)` | Recursive `rglob` filtered by `can_load`.                                                                  |
| `_resolve_sources(source)`      | Turns `str | Path | List[...]` (file *or* directory) into a deduplicated `List[Path]`, honoring `error_handling`.    |
| `_detect_content_type(path)`    | Maps extension → semantic type (`document`, `markdown`, `web_content`, `structured_data`, `tabular_data`, `configuration`, `code`, `plain_text`, `unknown`). Stamped into metadata. |
| `reset()`                       | Clears `_processed_document_ids` so a loader can re‑ingest the same path.                                              |

The base class never opens a file directly; it only orchestrates path
resolution, metadata, and error handling.

### 3.2 `config.py` — Pydantic configurations

All loader configurations inherit from `LoaderConfig(BaseModel, ABC)`, which
defines the fields shared by every loader:

| Field                | Type                                  | Purpose                                                  |
|----------------------|---------------------------------------|----------------------------------------------------------|
| `encoding`           | `Optional[str]`                       | File encoding (auto‑detected if `None`).                 |
| `error_handling`     | `Literal["ignore","warn","raise"]`    | What `_handle_loading_error` does.                       |
| `include_metadata`   | `bool` (default `True`)               | If `False`, `Document.metadata` is set to `{}`.          |
| `custom_metadata`    | `Dict[str, Any]`                      | Merged into every document's metadata.                   |
| `max_file_size`      | `Optional[int]` (bytes)               | Per‑file size limit.                                     |
| `skip_empty_content` | `bool` (default `True`)               | Drop documents with empty content.                       |

Subclasses then add format‑specific fields. The full set of subclasses:

| Config class                | Loader               | Notable fields                                                                                                                                                  |
|-----------------------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `TextLoaderConfig`          | `TextLoader`         | `strip_whitespace`, `min_chunk_length`                                                                                                                          |
| `CSVLoaderConfig`           | `CSVLoader`          | `content_synthesis_mode` (`"concatenated"` / `"json"`), `split_mode` (`"single_document"` / `"per_row"` / `"per_chunk"`), `rows_per_chunk`, `include/exclude_columns`, `delimiter`, `quotechar`, `has_header` |
| `PdfLoaderConfig`           | `PdfLoader`          | `extraction_mode` (`"hybrid"` / `"text_only"` / `"ocr_only"`), `start_page`/`end_page`, `clean_page_numbers`, `page_num_start/end_format`, `extra_whitespace_removal`, `pdf_password` |
| `PyMuPDFLoaderConfig`       | `PyMuPDFLoader`      | All fields of `PdfLoaderConfig` + `text_extraction_method` (`text`/`dict`/`html`/`xml`), `include_images`, `image_dpi`, `preserve_layout`, `extract_annotations`, `annotation_format` |
| `PdfPlumberLoaderConfig`    | `PdfPlumberLoader`   | All fields of `PdfLoaderConfig` + `extract_tables`, `table_format` (`text`/`markdown`/`csv`/`grid`), `table_settings`, `extract_images`, `layout_mode`, `use_text_flow`, `char/line/word_margin`, `extract_page_dimensions`, `crop_box`, `extract_annotations`, `keep_blank_chars` |
| `DOCXLoaderConfig`          | `DOCXLoader`         | `include_tables`, `include_headers`, `include_footers`, `table_format` (`text`/`markdown`/`html`)                                                                |
| `JSONLoaderConfig`          | `JSONLoader`         | `mode` (`"single"`/`"multi"`), `record_selector` (JQ), `content_mapper` (JQ), `metadata_mapper` (`Dict[str, jq_query]`), `content_synthesis_mode` (`"json"`/`"text"`), `json_lines` |
| `XMLLoaderConfig`           | `XMLLoader`          | `split_by_xpath`, `content_xpath`, `content_synthesis_mode` (`"smart_text"`/`"xml_snippet"`), `include_attributes`, `metadata_xpaths`, `strip_namespaces`, `recover_mode` |
| `YAMLLoaderConfig`          | `YAMLLoader`         | `split_by_jq_query`, `handle_multiple_docs`, `content_synthesis_mode` (`"canonical_yaml"`/`"json"`/`"smart_text"`), `yaml_indent`, `json_indent`, `flatten_metadata`, `metadata_jq_queries` |
| `MarkdownLoaderConfig`      | `MarkdownLoader`     | `parse_front_matter`, `include_code_blocks`, `code_block_language_metadata`, `heading_metadata`, `split_by_heading` (`h1`/`h2`/`h3`)                              |
| `HTMLLoaderConfig`          | `HTMLLoader`         | `extract_text`, `preserve_structure`, `include_links`, `include_images`, `remove_scripts`, `remove_styles`, `extract_metadata`, `clean_whitespace`, `extract_headers/paragraphs/lists/tables`, `table_format`, `user_agent` |
| `DoclingLoaderConfig`       | `DoclingLoader`      | `extraction_mode` (`"markdown"`/`"chunks"`), `chunker_type` (`"hybrid"`/`"hierarchical"`), `allowed_formats`, OCR config (`ocr_enabled`, `ocr_force_full_page`, `ocr_backend`, `ocr_lang`, `ocr_backend_engine`, `ocr_text_score`), table config (`enable_table_structure`, `table_structure_cell_matching`), `max_pages`, `page_range`, `parallel_processing`, `batch_size`, `extract_document_metadata`, `confidence_threshold`, `support_urls`, `url_timeout` |

There are also two convenience helpers:

```python
LoaderConfigFactory.create_config("pdf", extraction_mode="text_only")
simple_config("markdown")             # defaults
advanced_config("html", extract_text=True, include_links=False)
```

### 3.3 `factory.py` — `LoaderFactory` + module helpers

`LoaderFactory` is the single brain that picks loaders for arbitrary sources.
It is constructed once (via `get_factory()` global singleton) and:

1. **Lazily imports** each loader module inside `try/except ImportError` so a
   missing optional dep (e.g. no `pymupdf`) doesn't break the whole package.
2. **Registers** loaders → extensions in `self._extensions`. Later
   registrations win, which is how `DoclingLoader` is `insert(0, …)` to take
   priority for shared extensions when installed.
3. **Detects** the loader type for a source via `_detect_loader_type`:
   - `http://` / `https://` / `ftp://` → `html`.
   - File extension → registered loader.
   - Falls back to **content sniffing** (`_is_json_content`, `_is_xml_content`,
     `_is_yaml_content`, `_is_markdown_content`, `_is_html_content`) for raw
     strings.
4. **Optimizes** configuration per file via `_create_optimized_config`:

| Loader        | Optimization on `file_size > 100 MB` (or PDF > 50 MB)                                                          |
|---------------|----------------------------------------------------------------------------------------------------------------|
| `pdf`         | `extraction_mode="text_only"` (skip OCR)                                                                       |
| `pymupdf`     | `extraction_mode="text_only"`, `include_images=False`, `extract_annotations=False`, `image_dpi=72`            |
| `docling`     | `chunker_type="hybrid"`, `max_pages=100`, `ocr_enabled=False`, `table_structure_cell_matching=False`           |
| `csv`         | `content_synthesis_mode="concatenated"`, `include_metadata=False`                                              |
| `json`        | `mode="single"`, `include_metadata=False`                                                                       |
| `xml`         | `content_synthesis_mode="smart_text"`, `strip_namespaces=False`, `include_metadata=False`                      |
| `yaml`        | `content_synthesis_mode="smart_text"`, `flatten_metadata=False`, `include_metadata=False`                      |
| `markdown`    | turns off heading + front‑matter metadata                                                                       |
| `docx`        | `table_format="text"`, drop headers/footers/metadata                                                            |
| `text`        | `min_chunk_length=100`, `include_metadata=False`                                                                 |

5. Provides high‑level helpers:

```python
factory = get_factory()                       # singleton
loader  = factory.get_loader("foo.pdf")       # explicit one-off
loader  = create_loader("foo.pdf")            # module-level shortcut
loader  = create_loader_for_file("./x.csv")
loader  = create_loader_for_content("{...}", "json")
loaders = create_intelligent_loaders([Path("a.pdf"), Path("b.docx")])
docs    = load_document("foo.pdf")            # 1-shot: build loader + load
batch   = load_documents_batch(["a.pdf","b.csv"])
info    = validate_source("https://x.com")
stats   = get_loader_statistics()
loaders = list_available_loaders()
conflicts = check_extension_conflicts()
```

`LoaderFactory` is also a context manager — `__exit__` clears the
`_detect_loader_type` LRU cache.

### 3.4 `__init__.py` — Public surface

The package uses `__getattr__`‑based **lazy attribute access** to keep import
time low (no `pymupdf`, `docling`, `lxml`, `aiohttp`, … pulled until you
actually touch a class). The lazy resolver walks four dictionaries in order:

1. Base classes (`BaseLoader`, `LoaderConfig`, `LoaderConfigFactory`).
2. Loader classes (`TextLoader`, `PdfLoader`, …, optionally `DoclingLoader`).
3. Config classes (`TextLoaderConfig`, …, `DoclingLoaderConfig`).
4. Factory functions (`get_factory`, `create_loader`, …).

`__all__` lists everything so static type checkers and `from upsonic.loaders
import *` see the same surface.

---

## 4. Per‑loader file‑by‑file

### 4.1 `text.py` — `TextLoader`

| Property            | Value                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| Extensions          | `.txt .rst .log .py .js .ts .java .c .cpp .h .cs .go .rs .php .rb .css .ini`                                |
| Required dep        | `aiofiles` (`pip install "upsonic[text-loader]"`)                                                            |
| Output granularity  | One `Document` per file (no internal splitting).                                                            |
| Cleaning            | Optional `strip_whitespace`, `min_chunk_length` filter, `skip_empty_content`.                                |
| Async strategy      | `aload` uses `asyncio.gather` over `_process_single_file`, each reading via `aiofiles.open`.                |

Plus the universal "if running inside an existing event loop, run `aload`
in a `ThreadPoolExecutor`" trick for `load`.

### 4.2 `csv.py` — `CSVLoader`

| Property            | Value                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| Extensions          | `.csv`                                                                                                      |
| Required dep        | `aiofiles`, stdlib `csv`                                                                                    |
| Output granularity  | Driven by `split_mode`: `single_document` (one doc per file) / `per_row` (one doc per row) / `per_chunk` (`rows_per_chunk` rows per doc) |
| Content modes       | `"concatenated"` (`key: value\nkey: value`) or `"json"` (`json.dumps(filtered_row)`).                       |
| Column control      | `include_columns` / `exclude_columns`.                                                                       |
| Header              | `has_header=True` → `csv.DictReader`; otherwise auto‑names columns `column_0`, `column_1`, …                |
| Metadata            | Adds `row_count` / `row_index` / `total_rows` / `chunk_index` / `rows_in_chunk` depending on `split_mode`.   |

### 4.3 `json.py` — `JSONLoader`

| Property            | Value                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| Extensions          | `.json`, `.jsonl`                                                                                            |
| Required dep        | `jq` (`pip install "upsonic[json-loader]"`)                                                                  |
| Output granularity  | `mode="single"` (one doc per file) or `mode="multi"` (one doc per JQ‑selected record). For `.jsonl` (`json_lines=True`), each line yields its own block. |
| Mappers             | `record_selector` (JQ), `content_mapper` (JQ), `metadata_mapper` (`Dict[str, jq_query]`).                    |
| Content modes       | `"json"` (default; `json.dumps`) or `"text"` (raw value).                                                     |
| Validation          | Raises `ValueError` if `mode="multi"` without a `record_selector`.                                            |

### 4.4 `xml.py` — `XMLLoader`

| Property            | Value                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| Extensions          | `.xml`                                                                                                       |
| Required dep        | `lxml` (`pip install "upsonic[xml-loader]"`)                                                                 |
| Output granularity  | One **combined** `Document` per file, but the `split_by_xpath` selects elements whose contents are joined with `\n\n`. |
| Default XPath       | `"//*[not(*)] | //item | //product | //book"` — a flexible, leaf‑element pattern.                            |
| Content modes       | `"smart_text"` (concat all text nodes) or `"xml_snippet"` (`etree.tostring`).                                 |
| Metadata            | `include_attributes` (merge each element's `attrib`), `metadata_xpaths` (custom kv pairs).                    |
| Robustness          | `strip_namespaces=True` makes XPath simpler; `recover_mode=True` parses malformed XML.                        |

### 4.5 `yaml.py` — `YAMLLoader`

| Property            | Value                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| Extensions          | `.yaml`, `.yml`                                                                                              |
| Required dep        | `pyyaml` + `jq` (`pip install "upsonic[yaml-loader]"`)                                                       |
| Multi‑doc support   | `handle_multiple_docs=True` → uses `yaml.safe_load_all` for files with `---` separators.                     |
| Splitting           | `split_by_jq_query` defaults to `"."` (entire file). E.g. `".articles[]"` for one `Document` per article.    |
| Content modes       | `"canonical_yaml"` / `"json"` / `"smart_text"` (recursive string harvesting).                                |
| Metadata            | `flatten_metadata` (`{"a":{"b":1}} → {"a.b":1}`), plus optional `metadata_jq_queries`.                       |

### 4.6 `markdown.py` — `MarkdownLoader`

| Property            | Value                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| Extensions          | `.md`, `.markdown`                                                                                           |
| Required dep        | `python-frontmatter`, `markdown-it-py` (`pip install "upsonic[markdown-loader]"`)                             |
| Output granularity  | One `Document` per file, **or** one per heading block when `split_by_heading="h1"|"h2"|"h3"`. Each chunk gets `chunk_index`. |
| Metadata            | YAML front matter (if `parse_front_matter`), extracted `headings` per level, `code_languages` set.            |
| Cleanups            | `include_code_blocks=False` skips `fence` / `code_block` tokens.                                               |

The token‑based approach uses `markdown_it.MarkdownIt` and walks tokens
linearly, splitting whenever a `heading_open` of the configured tag is hit.

### 4.7 `html.py` — `HTMLLoader`

| Property            | Value                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| Extensions          | `.html`, `.htm`, `.xhtml` **and** any `http://` / `https://` URL                                              |
| Required dep        | `aiohttp`, `requests`, `beautifulsoup4` (`pip install "upsonic[html-loader]"`)                                |
| `can_load` override | Returns `True` for URL strings even though they are not files.                                                |
| Network             | Sync path uses `requests.get`; async path uses one shared `aiohttp.ClientSession`.                           |
| Cleaning            | Strips `<script>` / `<style>`, `clean_whitespace` collapses `\n\n+`.                                         |
| Structure           | Selects `<main>` / `<article>` / `<body>` / fallback whole‑doc, then iterates over configured selectors (`h1‑h6`, `p`, `ul`/`ol`, `table`). Headings rendered as `## Title`, lists as `- item`, tables via `_format_table` (`text` / `markdown` / `html`). |
| Metadata            | `<title>`, `<meta name|property=...>` keys, plus `final_url` / `status_code` for URL fetches.                |
| Document ID         | Overridden `_generate_document_id` accepts arbitrary strings (so a URL can become an ID).                     |

### 4.8 `pdf.py` — `PdfLoader` (pypdf)

| Property            | Value                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| Extensions          | `.pdf`                                                                                                       |
| Required deps       | `pypdf`, optional `rapidocr-onnxruntime` for OCR (`pip install "upsonic[pdf-loader]"`)                       |
| Extraction modes    | `"text_only"` (digital text via `page.extract_text()`), `"ocr_only"` (RapidOCR over each embedded image), `"hybrid"` (both, joined with `\n\n`). |
| Encryption          | If `reader.is_encrypted` and `pdf_password` is missing → `PermissionError`.                                   |
| Page range          | `start_page` / `end_page` (1‑indexed, inclusive).                                                            |
| Cleaning            | `_normalize_whitespace` (regex collapse) and `_clean_page_numbers` (sequence‑match heuristic with 40% threshold), plus optional `page_num_start_format` / `page_num_end_format` markers. |
| Concurrency         | `aload` runs one `_process_single_pdf` per file with `asyncio.gather`. Inside a file, OCR images are dispatched onto a `ThreadPoolExecutor`. |
| Output              | Always **one** `Document` per PDF; full text is the joined cleaned pages. Adds `page_count`.                  |

### 4.9 `pymupdf.py` — `PyMuPDFLoader`

Superset of `PdfLoader` for users who can install `pymupdf` (a.k.a. `fitz`):

| Extra capability                  | Detail                                                                                                          |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Multiple text extraction methods  | `text_extraction_method` ∈ `text` / `dict` (structured spans) / `html` / `xml`.                                  |
| Image metadata                    | `include_images=True` adds an `images` array with `xref/width/height/colorspace/...` entries to metadata.       |
| Annotations                       | `extract_annotations=True` returns `type/content/rect` per annot, with optional JSON dump.                       |
| OCR via raster                    | Renders the entire page to a PNG at `image_dpi`, then runs RapidOCR — better for scanned pages than pypdf's per‑image OCR. |
| Layout                            | `preserve_layout` flag is exposed in metadata.                                                                   |
| PDF info                          | Pulls `doc.metadata` (Title/Author/...) into `pdf_*` keys, plus `doc.get_pdf_metadata()` if available.          |

### 4.10 `pdfplumber.py` — `PdfPlumberLoader`

The "structured PDF" loader, designed for tables and forms:

| Capability                | Detail                                                                                                                  |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Tables (the headline)     | `extract_tables=True` and `table_settings` (vertical/horizontal strategies, snap/join/edge tolerances). Tables are emitted in `markdown` / `text` / `csv` / `grid` style with a `[Table N]` header. |
| Layout text               | `layout_mode` (`layout`/`default`/`simple`), `char_margin`, `line_margin`, `word_margin`, `keep_blank_chars`, `use_text_flow`. |
| Images                    | `extract_images=True` emits an `[Images: N found on page]` block with size + position.                                  |
| Annotations / hyperlinks  | `extract_annotations=True` lists notes (`Note N:`) and links (`Link N:`) — capped at 10 each per page.                   |
| Crop                      | `crop_box=(x0, y0, x1, y1)` to extract only a region.                                                                    |
| OCR                       | Uses `page.to_image(resolution=150)` → PIL → PNG bytes → RapidOCR.                                                       |
| Page dimensions           | `extract_page_dimensions=True` adds `page_width` / `page_height`.                                                         |
| Cleaning                  | Same `_normalize_whitespace` and `_clean_page_numbers` as the other PDF loaders.                                         |

### 4.11 `docx.py` — `DOCXLoader`

| Property            | Value                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| Extensions          | `.docx`                                                                                                      |
| Required dep        | `python-docx` (`pip install "upsonic[docx-loader]"`)                                                          |
| Output granularity  | One `Document` per file.                                                                                     |
| Sections collected  | Headers (default + first‑page + even‑page) → paragraphs → tables → footers (configurable via `include_*` flags). |
| Tables              | Rendered in `text` (tab‑separated), `markdown`, or `html`.                                                    |
| Metadata            | Adds Word `core_properties`: `author`, `category`, `comments`, `title`, `subject`, `created`, `modified`.    |
| Async strategy      | `aload` thread‑offloads `_load_single_file` (python‑docx is synchronous).                                    |

### 4.12 `docling.py` — `DoclingLoader`

The "do‑everything" loader powered by IBM's
[`docling`](https://github.com/DS4SD/docling).

| Property            | Value                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| Extensions          | `.pdf .docx .xlsx .pptx .html .htm .md .markdown .adoc .asciidoc .csv .png .jpg .jpeg .tiff .bmp .webp` + URLs |
| Required dep        | `docling` (`pip install "upsonic[docling-loader]"`); chunkers come from `docling.chunking` or `docling_core.transforms.chunker.*`. |
| Extraction modes    | `"markdown"` (one `Document` per file, full markdown export) or `"chunks"` (one `Document` per semantic chunk via `HybridChunker` or `HierarchicalChunker`). |
| OCR                 | Backend is `rapidocr` or `tesseract`, with `ocr_lang`, `ocr_backend_engine` (`onnxruntime`/`openvino`/`paddle`/`torch`), `ocr_text_score`, `ocr_force_full_page`. |
| Tables              | `enable_table_structure`, `table_structure_cell_matching`.                                                    |
| URLs                | `support_urls=True`, `url_timeout` (seconds).                                                                 |
| Page control        | `max_pages` or `page_range=(start, end)` (1‑indexed, inclusive).                                              |
| Parallelism         | `parallel_processing=True` + `batch_size` for `abatch`.                                                       |
| Confidence filter   | Drops chunks with `chunk.confidence < confidence_threshold`.                                                   |
| `can_load` override | Accepts URLs.                                                                                                  |
| Metadata            | `extract_document_metadata` injects `docling_metadata` (export JSON), `document_name`, `page_count`. Each chunk also stores `chunker_type`, `chunk_index`, `chunk_meta`, optional `confidence`. |

When registered, `DoclingLoader` is `insert(0, ...)` into the factory's
default loader list, which means **if Docling is installed, it claims the
canonical extension first** for `.pdf`, `.docx`, `.html`, `.md`, `.csv`, etc.

---

## 5. Cross‑file relationships

```
            ┌───────────────────────────────────────────────────────┐
            │                  loaders/__init__.py                  │
            │   (lazy __getattr__ → BaseLoader, configs, factory)   │
            └────────────────────────────┬──────────────────────────┘
                                         │
                       ┌─────────────────┼────────────────────────────┐
                       │                 │                            │
                       ▼                 ▼                            ▼
              ┌────────────────┐ ┌───────────────┐         ┌────────────────────┐
              │   base.py      │ │   config.py   │         │     factory.py     │
              │  BaseLoader    │ │ LoaderConfig  │         │  LoaderFactory     │
              │  (ABC)         │ │  + 12 subs    │         │  + create_loader / │
              │  metadata,     │ │  +Config-     │         │    load_document / │
              │  resolve src,  │ │   Factory     │         │    intelligent ... │
              │  error policy  │ └───────┬───────┘         └─────────┬──────────┘
              └───────┬────────┘         │                           │
                      │                  │ (per-loader Config used   │ (registers each
                      │                  │  by __init__ of loader)   │  loader, picks
                      ▼                  ▼                           │  by extension)
       ┌──────┬───────┬─────────┬──────┬──────┬──────┬──────┬───────┬──────────┐
       │text  │ csv   │ pdf     │ docx │ json │ xml  │ yaml │ markdn│ html /   │
       │.py   │ .py   │ .py     │ .py  │ .py  │ .py  │ .py  │ .py   │ docling  │
       └──┬───┴──┬────┴────┬────┴──┬───┴──┬───┴───┬──┴──┬───┴──┬────┴──┬───────┘
          │      │         │       │      │       │     │      │       │
          ▼      ▼         ▼       ▼      ▼       ▼     ▼      ▼       ▼
      Document Document Document Document …                       (URL/file)
                                  │
                                  ▼
                          schemas.data_models.Document
                                  │
                                  ▼
                  text_splitter.* → Chunk → embedder → vector DB
```

Key contracts:

- **Every concrete loader inherits from `BaseLoader`** and implements the four
  required methods + `get_supported_extensions`.
- **Every concrete loader's `__init__` accepts an `Optional[<X>LoaderConfig]`**
  and falls back to the default config when none is provided.
- **All loaders import `Document` from `upsonic.schemas.data_models`** — no
  loader has its own document type.
- **Factory talks only to `BaseLoader` + `LoaderConfig`**; it never imports
  concrete loaders eagerly. That is what makes optional dependencies optional.

---

## 6. Public API

The package re‑exports everything in `__all__`:

```python
# Bases
from upsonic.loaders import BaseLoader

# Configs
from upsonic.loaders import (
    LoaderConfig, LoaderConfigFactory,
    TextLoaderConfig, CSVLoaderConfig, PdfLoaderConfig, PyMuPDFLoaderConfig,
    PdfPlumberLoaderConfig, DOCXLoaderConfig, JSONLoaderConfig, XMLLoaderConfig,
    YAMLLoaderConfig, MarkdownLoaderConfig, HTMLLoaderConfig, DoclingLoaderConfig,
    simple_config, advanced_config,
)

# Concrete loaders
from upsonic.loaders import (
    TextLoader, CSVLoader, PdfLoader, PyMuPDFLoader, PdfPlumberLoader,
    DOCXLoader, JSONLoader, XMLLoader, YAMLLoader, MarkdownLoader,
    HTMLLoader, DoclingLoader,
)

# Factory
from upsonic.loaders import (
    LoaderFactory, get_factory, create_loader, create_loader_for_file,
    create_loader_for_content, can_handle_file, get_supported_extensions,
    get_supported_loaders, load_document, load_documents_batch,
    create_intelligent_loaders, validate_source, get_loader_statistics,
    list_available_loaders, check_extension_conflicts, create_factory,
    with_factory,
)
```

### 6.1 Three idiomatic ways to load

**(A) Explicit loader + explicit config — full control:**

```python
from upsonic.loaders import PdfPlumberLoader, PdfPlumberLoaderConfig

cfg = PdfPlumberLoaderConfig(
    extraction_mode="hybrid",
    extract_tables=True,
    table_format="markdown",
    pdf_password="hunter2",
    custom_metadata={"team": "research"},
)
docs = PdfPlumberLoader(cfg).load("./reports/q1.pdf")
```

**(B) Factory shortcut — auto loader, custom config:**

```python
from upsonic.loaders import create_loader

loader = create_loader(
    "./data.csv",
    split_mode="per_chunk",
    rows_per_chunk=50,
    content_synthesis_mode="json",
)
docs = loader.load("./data.csv")
```

**(C) One‑liner — best for scripts:**

```python
from upsonic.loaders import load_document, load_documents_batch

docs   = load_document("https://example.com/blog")
batch  = load_documents_batch(["a.pdf", "b.docx", "c.csv"])
```

### 6.2 Inspecting the registry

```python
from upsonic.loaders import (
    get_supported_extensions, list_available_loaders,
    validate_source, check_extension_conflicts, get_loader_statistics,
)

get_supported_extensions()
# ['.csv', '.pdf', '.docx', '.json', '.jsonl', '.xml', '.yaml', '.yml',
#  '.md', '.markdown', '.html', '.htm', '.xhtml', '.txt', '.rst', '.log', …]

validate_source("https://example.com/x")
# {'source': '...', 'is_url': True, 'is_file': False,
#  'detected_type': 'html', 'can_handle': True, 'recommended_loader': 'html'}
```

---

## 7. Integration with `knowledge_base` + `text_splitter` + `ocr`

### 7.1 `knowledge_base/`

`KnowledgeBase` is the consumer. From
`src/upsonic/knowledge_base/knowledge_base.py`:

```python
from ..loaders.base import BaseLoader
from ..loaders.factory import create_intelligent_loaders
```

The relevant flow inside `KnowledgeBase`:

1. **Construction** accepts an optional `loaders=` argument (single
   `BaseLoader`, list, or `None`).
2. **`_setup_loaders`** decides:
   - If `loaders is None` → calls `create_intelligent_loaders(self.sources)`,
     which delegates to `LoaderFactory.create_intelligent_loaders` and
     returns one optimized loader per file source (skipping raw‑string
     sources, which are treated as direct content).
   - Otherwise normalizes to a list via `_normalize_loaders`.
3. **`_load_documents`** (Step 1/4 of the ingestion pipeline) iterates
   `self.sources`, picks a loader via `_get_component_for_source`, validates
   `loader.can_load(source)`, and calls `loader.load(source)`. Each source's
   resulting `List[Document]` is stored under
   `processing_metadata['source_to_documents'][source_index]`.
4. The downstream stages then run change detection (using
   `Document.doc_content_hash`, computed later), text splitting, embedding,
   and vector‑DB insertion.

So the loaders folder is exclusively **stage 1** of the four‑stage
KnowledgeBase pipeline.

### 7.2 `text_splitter/`

Loaders never call into `text_splitter`. The contract is:

```
loaders.* → List[Document]   (per-file, per-row, per-record, per-heading, …)
text_splitter.* → List[Chunk]  (semantically sized fragments)
```

Every loader emits `Document` objects that carry **already‑structured** text
where structure is cheap (CSV row, JSON record, Markdown heading, PDF page)
and **whole‑file** text where structure is the splitter's job (TXT, DOCX,
HTML, PDF). The splitter can rely on:

- `Document.content` — the canonical string to chunk.
- `Document.metadata` — passed through to each `Chunk` and augmented with
  chunk‑level keys (`page_number`, `chunk_index`, …).
- `Document.document_id` — preserved on every `Chunk` via `parent_doc_id`.

The loaders' `_create_metadata` writes `content_type` (`document`,
`tabular_data`, `web_content`, …) which lets a splitter pick a specialized
strategy if it wants to.

### 7.3 `ocr/`

The loaders folder does **not** depend on `src/upsonic/ocr/`. PDF loaders
talk directly to `rapidocr_onnxruntime` (a hard‑wired OCR engine for
`PdfLoader`, `PyMuPDFLoader`, `PdfPlumberLoader`):

```python
try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_ENGINE = RapidOCR()
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    ...
```

`DoclingLoader` is the only loader that exposes a *choice* of OCR backend
(`rapidocr` or `tesseract`) via `DoclingLoaderConfig.ocr_backend`. Even there,
the configuration is consumed by Docling's `PdfPipelineOptions`, not by
Upsonic's `ocr/` package.

The `src/upsonic/ocr/` module is a **separate** subsystem (multi‑layer OCR
with engines under `ocr/layer_1/engines/`) used elsewhere in the framework;
the RAG ingestion path bypasses it.

---

## 8. End‑to‑end flow: file → chunks

Here is what happens when you write:

```python
from upsonic.knowledge_base import KnowledgeBase

kb = KnowledgeBase(sources=["./reports/q1.pdf", "./data.csv", "https://x.com"])
await kb.run()
```

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Step 0  KnowledgeBase.__init__                                              │
│   loaders is None  →  loaders.factory.create_intelligent_loaders(sources)   │
│      For each source:                                                       │
│        - .pdf      → LoaderFactory._detect_loader_type → "docling" or "pdf" │
│                       (whichever is registered first; with docling          │
│                        installed it wins)                                   │
│                      _create_optimized_config tunes extraction_mode,        │
│                       OCR, page limits based on file_size                   │
│        - .csv      → "csv" → optimized CSV config                           │
│        - http(s):  → "html"                                                 │
│      Returns: List[BaseLoader] (one per file source).                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Step 1  KnowledgeBase._load_documents                                       │
│   For each (source_index, source):                                          │
│     loader = self._get_component_for_source(source_index, self.loaders, …)  │
│     if not loader.can_load(source): warn + skip                              │
│     source_documents = loader.load(source)                                   │
│                                                                              │
│   Inside `loader.load(source)`:                                             │
│     1. _resolve_sources(source) → List[Path]                                │
│        - normalizes to list, expands directories, dedupes paths             │
│        - obeys config.error_handling for missing paths                      │
│     2. For each path:                                                        │
│        a. _generate_document_id(path) (md5 of absolute path)                │
│        b. _check_file_size(path) (compares to config.max_file_size)         │
│        c. Loader-specific extraction:                                        │
│             pypdf / pymupdf / pdfplumber: open, decrypt if needed,          │
│               iterate pages, optional OCR, _normalize_whitespace,           │
│               _clean_page_numbers                                            │
│             docx: paragraphs + tables + headers + footers                    │
│             csv:  csv.DictReader, _filter_row_columns,                       │
│                   _synthesize_content, _create_documents_from_rows          │
│             json: jq.compile(record_selector), per-record content_mapper    │
│             xml:  lxml.parse, tree.xpath(split_by_xpath)                    │
│             yaml: yaml.safe_load_all + jq.all(split_by_jq_query)            │
│             md:   markdown_it tokens → split by heading                      │
│             html: requests/aiohttp → BeautifulSoup → structural extract     │
│             docling: DocumentConverter.convert → DoclingDocument →           │
│                      export_to_markdown OR HybridChunker.chunk               │
│        d. _create_metadata(path) (size, ctime/mtime, content_type, …)        │
│        e. yield Document(content=..., metadata=..., document_id=...)         │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼  List[Document]
┌──────────────────────────────────────────────────────────────────────────────┐
│ Step 2  KnowledgeBase._filter_changed_documents                              │
│   - Computes Document.doc_content_hash                                       │
│   - Skips unchanged docs, deletes stale chunks for edited docs               │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Step 3  KnowledgeBase._split_documents → text_splitter                       │
│   For each Document → splitter → List[Chunk]                                 │
│   Each Chunk inherits Document.metadata + adds chunk_id, chunk_content_hash, │
│   start_idx / end_idx, parent_doc_id.                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Step 4  Embed + persist into vector DB                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

A concrete trace for `./reports/q1.pdf` (12 MB digital PDF, with Docling
installed):

```python
# 1. factory picks DoclingLoader because it was inserted first in the registry
loader = DoclingLoader(DoclingLoaderConfig(
    extraction_mode="chunks",
    chunker_type="hierarchical",      # small file → quality settings
    ocr_enabled=True,
    ocr_backend="rapidocr",
    ocr_text_score=0.6,
    confidence_threshold=0.7,
    extract_document_metadata=True,
))

# 2. _convert_document → DoclingDocument
# 3. _extract_chunks iterates HybridChunker.chunk(dl_doc)
#    → for each chunk:
#         metadata = _create_metadata(Path("./reports/q1.pdf"))
#         metadata["extraction_mode"] = "chunks"
#         metadata["chunker_type"]    = "hierarchical"
#         metadata["chunk_index"]     = i
#         metadata["confidence"]      = 0.83
#         metadata["docling_metadata"]= {...}
#         doc_id = "<md5>_chunk_<i>"
#         yield Document(content=chunk_text, document_id=doc_id, metadata=metadata)

# 4. KnowledgeBase receives ~50 Document objects, runs them through
#    text_splitter (which may further split long chunks),
#    embeds, and persists.
```

For the same `q1.pdf` *without* Docling installed, the factory falls back to
`PdfPlumberLoader` (also fits if installed) or finally `PdfLoader` — yielding
a single `Document` per PDF whose content is the joined cleaned page text.
The downstream pipeline is unchanged because the contract is the same:
**one or more `Document` objects with `content`, `metadata`, `document_id`.**

That contract is the entire point of this folder.
