---
name: text-splitter-chunking
description: Use when working with the `upsonic.text_splitter` package to split Documents into Chunks for RAG, embedding, and indexing pipelines. Use when a user asks to choose, configure, or extend a chunking strategy, build a custom chunker, register a strategy, troubleshoot chunk size/overlap or fallback behavior, or wire splitters into KnowledgeBase. Trigger when the user mentions text splitter, chunking, chunker, BaseChunker, BaseChunkingConfig, CharacterChunker, RecursiveChunker, MarkdownChunker, HTMLChunker, JSONChunker, PythonChunker, SemanticChunker, AgenticChunker, create_chunking_strategy, create_adaptive_strategy, create_intelligent_splitters, create_rag_strategy, ContentType, ChunkingUseCase, Language, breakpoint_threshold_type, propositions, topic grouping, chunk_size, chunk_overlap, min_chunk_size, separators, header_stack, AST chunking, semantic chunking, agentic chunking, or RAG document splitting.
---

# `upsonic.text_splitter` — Text Chunking for RAG

This document explains the `src/upsonic/text_splitter/` package: a complete, pluggable
chunking subsystem that converts `Document` objects into `Chunk` objects suitable for
embedding, indexing, and retrieval-augmented generation (RAG). It documents every
strategy, the shared base class, the factory/registry layer, and how the rest of
Upsonic (notably `KnowledgeBase`) wires chunkers into the document pipeline.

---

## 1. What this folder is — text chunking for RAG

`upsonic.text_splitter` is the chunking layer of Upsonic's RAG stack. Its job is to take
a list of `Document` instances (loaded from PDFs, web pages, code, JSON files, etc.) and
slice them into smaller `Chunk` objects, each:

- short enough to fit inside an embedding model's context window,
- semantically coherent enough that retrieval still returns useful neighborhoods, and
- annotated with metadata (start/end offsets, parent doc id, headers, JSON paths, etc.)
  so retrieval and citations can map back to the source.

The package ships with eight different chunking strategies, ranging from purely
mechanical (fixed-size character splits) to AI-driven (an Agent extracts propositions
and groups them into thematic chunks). All strategies implement the same `BaseChunker`
interface, so they are interchangeable inside `KnowledgeBase` and any other Upsonic
component that calls `chunker.chunk(documents)` or `await chunker.achunk(documents)`.

A factory layer on top of the strategies lets callers either:

- ask for a specific strategy by name (`create_chunking_strategy("markdown")`), or
- describe the content/use-case and let Upsonic pick (`create_adaptive_strategy(...)`,
  `create_rag_strategy(...)`, `create_intelligent_splitters([...])`).

The package only depends on `Document`/`Chunk` data models from
`upsonic.schemas.data_models` and a few optional libraries (`numpy` for semantic,
`bs4`/`lxml` for HTML). The agentic chunker depends on `upsonic.agent.Agent`.

### Mental model

```
Document(s)  ─►  BaseChunker.chunk / achunk  ─►  List[Chunk]
                  (selects/uses one strategy)
```

The chunker is a stateless transformer: configuration in, documents in, chunks out.
It does not own embeddings, vector stores, or retrieval — those live in
`upsonic.embeddings`, `upsonic.vectordb`, and `upsonic.rag`.

---

## 2. Folder layout (tree)

```
src/upsonic/text_splitter/
├── __init__.py              # Lazy re-exports of all chunkers, configs, factory helpers
├── base.py                  # BaseChunker (ABC), BaseChunkingConfig, _create_chunk helper
├── factory.py               # Registry, ContentType/ChunkingUseCase enums, factories
├── character.py             # CharacterChunker — single-separator split & merge
├── recursive.py             # RecursiveChunker — prioritized-separator recursion + Language
├── markdown.py              # MarkdownChunker — header/code/table-aware segmentation
├── html_chunker.py          # HTMLChunker — BeautifulSoup DOM-aware chunking
├── json_chunker.py          # JSONChunker — path-aware JSON graph splitting
├── python.py                # PythonChunker — AST-based code chunking
├── semantic.py              # SemanticChunker — embedding-distance topic boundaries
└── agentic.py               # AgenticChunker — propositions + AI topic grouping
```

Every strategy file follows the same shape: a Pydantic `*ChunkingConfig` extending
`BaseChunkingConfig`, plus a `*Chunker` extending `BaseChunker[ConfigType]`.

---

## 3. Top-level files (`base.py`, `factory.py`, `__init__.py`)

### 3.1 `base.py`

Defines the contract every chunker must implement.

#### `BaseChunkingConfig`

Pydantic model with the parameters every strategy consumes.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `chunk_size` | `int` (>0) | `1024` | Target size of each chunk (units defined by `length_function`). |
| `chunk_overlap` | `int` (>=0) | `200` | How many trailing units of the previous chunk to repeat at the start of the next. |
| `min_chunk_size` | `Optional[int]` | `None` | Threshold below which a final small chunk is merged into its predecessor. If `None`, derived as `max(chunk_size * 0.8, 50)`. |
| `length_function` | `Callable[[str], int]` | `len` | Function used to measure size — swap to a token counter for token-aware chunking. |
| `strip_whitespace` | `bool` | `False` | Whether `_create_chunk` should `.strip()` the text before storing it. |

The `__init__` of `BaseChunker` warns when `chunk_overlap >= chunk_size` because that
configuration can spin into an infinite loop in the merge phase.

#### `BaseChunker` (Generic[ConfigType])

```python
class BaseChunker(ABC, Generic[ConfigType]):
    def __init__(self, config: Optional[ConfigType] = None): ...
    def chunk(self, documents: List[Document]) -> List[Chunk]: ...
    async def achunk(self, documents: List[Document]) -> List[Chunk]: ...
    def batch(self, documents, **kwargs) -> List[Chunk]: ...                    # alias for chunk
    async def abatch(self, documents, **kwargs) -> List[Chunk]: ...              # asyncio.gather

    @abstractmethod
    def _chunk_document(self, document: Document) -> List[Chunk]: ...
    async def _achunk_document(self, document: Document) -> List[Chunk]:         # default: wraps sync

    def _create_chunk(self, parent_document, text_content, start_index,
                      end_index, extra_metadata=None) -> Chunk: ...
    def _get_effective_min_chunk_size(self) -> int: ...
    @classmethod
    def _get_default_config(cls) -> ConfigType: ...
```

Key behaviours:

- `chunk` and `achunk` iterate documents, skip empty/whitespace-only content, and
  catch errors per-document, emitting an `[CHUNKING ERROR: ...]` chunk with
  `chunking_error: True` metadata so the caller still receives a record of failure.
- `abatch` runs `_achunk_document` for every document concurrently via `asyncio.gather`
  and flattens the result.
- `_create_chunk` is the canonical `Chunk` factory: it deep-copies parent metadata,
  merges in `extra_metadata`, computes an MD5 `chunk_content_hash` over the final text,
  and stamps `document_id`/`doc_content_hash` from the parent document. This is how
  the rest of Upsonic detects duplicate chunks across runs.
- `_get_effective_min_chunk_size` returns either the user-set `min_chunk_size` or
  `max(chunk_size * 0.8, 50)`. Most strategies use it to merge a small trailing chunk
  back into its predecessor so RAG indexes don't get a long tail of fragments.

#### `Document` and `Chunk`

The `Document` and `Chunk` types come from `upsonic.schemas.data_models` (see that
folder's docs). Each `Chunk` carries:

```
text_content, metadata, document_id, doc_content_hash,
chunk_content_hash, start_index, end_index
```

Per-strategy metadata (e.g. `header_text`, `path`, `chunk_json_paths`,
`agentic_title`, `coherence_score`) is added on top of the parent document metadata.

### 3.2 `factory.py`

The factory module owns three concerns:

1. **Registry of strategies and configs** (`_STRATEGY_REGISTRY`, `_CONFIG_REGISTRY`),
   populated lazily by `_lazy_import_strategies()` so optional dependencies (e.g.
   `bs4` for HTML) only fail at registration time.
2. **Content classification & strategy recommendation** (`ContentType`,
   `ChunkingUseCase`, `detect_content_type`, `recommend_strategy_for_content`).
3. **Public factory functions** that build a chunker from a name, a content sample,
   or a list of source paths.

#### Enums

| Enum | Values |
|------|--------|
| `ContentType` | `PLAIN_TEXT`, `MARKDOWN`, `HTML`, `CODE`, `JSON`, `CSV`, `XML`, `PYTHON`, `JAVASCRIPT`, `TECHNICAL_DOC`, `NARRATIVE` |
| `ChunkingUseCase` | `RAG_RETRIEVAL`, `SEMANTIC_SEARCH`, `SUMMARIZATION`, `QUESTION_ANSWERING`, `CLASSIFICATION`, `GENERAL` |

#### Registry API

| Function | Purpose |
|----------|---------|
| `register_chunking_strategy(name, strategy_class, config_class=None)` | Register a custom chunker. Validates name uniqueness and that classes inherit from the right base. |
| `unregister_chunking_strategy(name)` | Remove a registration. Returns `bool`. |
| `clear_strategy_registry()` | Wipe both registries (testing). |
| `_lazy_import_strategies()` | Internal — import every built-in chunker inside `try/except ImportError` and register it. Also registers aliases: `recursive_character`→recursive, `markdown_header`/`markdown_recursive`→markdown, `code`/`py`/`python_code`→python, `ai`→agentic. |

#### Inspection

| Function | Returns |
|----------|---------|
| `list_available_strategies() -> List[str]` | All registered strategy names (after lazy import). |
| `get_strategy_info() -> Dict[str, Dict]` | Detailed dictionary per strategy: `description`, `best_for`, `features`, `use_cases`, `config_params`. |
| `detect_content_type(content, metadata=None) -> ContentType` | Heuristic detection by file extension (`metadata['source']`, `metadata['document_name']`) and content scanning (HTML tags, Markdown patterns, JSON parse, XML, Python/JS keywords, narrative sentence-length). |
| `recommend_strategy_for_content(content_type, use_case, content_length, quality_preference) -> str` | Selects the best registered strategy, see rules below. |

`recommend_strategy_for_content` decision rules (in order):

1. Structured content overrides quality preference: `MARKDOWN→markdown`, `HTML→html`,
   `JSON→json`, `PYTHON`/`JAVASCRIPT`→`python`.
2. `quality_preference == "fast"` or `content_length > 100k` → `character`, else
   `recursive`.
3. `quality_preference == "quality"`:
   - `SEMANTIC_SEARCH` → `semantic`.
   - `RAG_RETRIEVAL` / `QUESTION_ANSWERING` → `agentic` if content < 50k, else
     `semantic`.
4. `quality_preference == "balanced"`:
   - `SEMANTIC_SEARCH` and `< 75k` → `semantic`.
   - RAG/QA and `< 75k` → `semantic`.
   - Otherwise `recursive`.
5. Final fallback: `recursive` → `character` → first registered.

#### Factory functions

| Function | Purpose |
|----------|---------|
| `create_chunking_strategy(strategy, config=None, **kwargs) -> BaseChunker` | Build any registered chunker. Handles two special cases: `semantic` (requires `embedding_provider`, raises `ConfigurationError(MISSING_EMBEDDING_PROVIDER)` if absent) and `agentic`/`ai` (creates a default `Agent("openai/gpt-4o")` if no `agent` is passed). Internally calls `_create_final_config` to merge dict/object config with kwargs. |
| `create_adaptive_strategy(content, metadata=None, use_case=GENERAL, quality_preference="balanced", embedding_provider=None, agent=None, **kwargs)` | Detect type → recommend strategy → build optimized config → create chunker. Optimized config: chunk_size 1500 (code) / 1200 (technical) / 800 (narrative) / 1000 (default); chunk_overlap 25% (narrative/technical) or 15% otherwise; injects language-specific `separators` for recursive on Markdown/code. |
| `create_intelligent_splitters(sources, content_samples=None, use_case=RAG_RETRIEVAL, quality_preference="balanced", embedding_provider=None, agent=None, **global_config_kwargs)` | One-splitter-per-source helper used by `KnowledgeBase`. For each source it reads up to 5000 bytes of content, builds source-optimized config (chunk size by file size, overlap by extension), then calls `create_adaptive_strategy`. Falls back to `create_chunking_strategy("recursive")` on per-source failure. |
| `create_rag_strategy(content="", embedding_provider=None, agent=None, **kwargs)` | Convenience wrapper: adaptive with `RAG_RETRIEVAL`/`balanced` if content given, else `recursive`. |
| `create_semantic_search_strategy(content="", embedding_provider=None, **kwargs)` | Adaptive with `SEMANTIC_SEARCH`/`quality` when content present; otherwise creates `semantic` if a provider was given, else warns and returns `recursive`. |
| `create_fast_strategy(**kwargs)` | Hardcoded `character` chunker for very large or speed-critical inputs. |
| `create_quality_strategy(content="", embedding_provider=None, agent=None, **kwargs)` | Tries `agentic` → `semantic` (with provider) → `recursive`. |

### 3.3 `__init__.py`

A pure re-export module using `__getattr__` for **lazy loading** of every chunker,
config, and factory function. Heavy imports (numpy, bs4, the agent system) are
deferred until the symbol is accessed. `__all__` is the canonical list of public
names. `TYPE_CHECKING` guards keep static analysis correct without paying the
import cost at runtime.

---

## 4. Per-splitter files

The following sections describe each chunker. Every chunker's config inherits the
five `BaseChunkingConfig` fields (`chunk_size`, `chunk_overlap`, `min_chunk_size`,
`length_function`, `strip_whitespace`) — only the additional fields are listed.

### 4.1 `CharacterChunker` (`character.py`)

A foundational "split-and-merge" chunker. Splits content using a single regex or
literal separator, then greedily merges adjacent atomic pieces up to `chunk_size`
with a tail-overlap window.

#### Config — `CharacterChunkingConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `separator` | `"\n\n"` | The split boundary. |
| `is_separator_regex` | `False` | If True, treat as regex, else literal (`re.escape`). |
| `keep_separator` | `True` | Emit the separator as its own atomic split alongside the content (preserves layout). |

#### Algorithm

1. **Tokenize** content into `(text, start_idx, end_idx)` triples by walking
   `re.finditer(pattern)` over the source. If `keep_separator=True`, the matched
   separator becomes its own triple. If `separator == ""`, every character becomes a
   triple (true character-level splitting).
2. **Merge loop**: walk atomic splits, append to `current_chunk_parts` while
   `current_length + part_length <= chunk_size`.
3. When the next part would overflow, finalize the current chunk using
   `content[first_start:last_end]` (slicing the original string preserves whitespace
   and original indices), then carry a tail of parts whose lengths sum up to
   `chunk_overlap`.
4. A single atomic split larger than `chunk_size` is emitted as an oversized
   chunk and a warning is logged — content integrity wins over size.
5. After the loop, if the trailing accumulator is shorter than the effective
   `min_chunk_size`, it is merged into the previous chunk by re-slicing the
   original content (so positional indices stay correct).

### 4.2 `RecursiveChunker` (`recursive.py`)

The default general-purpose chunker. Tries a *prioritized* list of separators; for
any segment still larger than `chunk_size`, it recurses with the remaining,
lower-priority separators. This preserves the strongest semantic boundaries first.

#### Config — `RecursiveChunkingConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `separators` | `["\n\n", "\n", ". ", "? ", "! ", " ", ""]` | Prioritized separator list. The empty string acts as the final character-level fallback. |
| `keep_separator` | `True` | When True, the separator is attached to the preceding segment (`text[cursor:match.end()]`). |
| `is_separator_regex` | `False` | Treat separators as regex when True. |

#### `Language` enum and `RECURSIVE_SEPARATORS`

`recursive.py` exports a `Language` enum (`PYTHON`, `MARKDOWN`, `HTML`, `LATEX`,
`JAVA`, `JS`) and a `RECURSIVE_SEPARATORS` mapping with curated regex separators
for each. Use `RecursiveChunker.from_language(Language.MARKDOWN)` to get a
preconfigured instance:

```python
chunker = RecursiveChunker.from_language(Language.PYTHON)
# separators: [r"\nclass ...:", r"\ndef ...:", r"\n\tdef ...:", "\n\n", "\n", " ", ""]
# is_separator_regex=True, keep_separator=True
```

#### Algorithm

1. `_recursive_split(text, separators, offset)` finds the first separator that
   actually matches in `text`. If none matches, the whole `text` is returned as a
   single atomic span (with absolute offsets via `offset + ...`).
2. For each match, the segment is either accepted as-is, or — if it's still longer
   than `chunk_size` *and* there are more separators — recursed with the remaining
   separators. This yields a flat list of `(text, start_idx, end_idx)` triples that
   already respect `chunk_size` whenever possible.
3. The merge phase mirrors `CharacterChunker`: greedy accumulate, finalize at
   overflow with a tail-overlap window, and merge the trailing under-sized chunk.

### 4.3 `MarkdownChunker` (`markdown.py`)

Syntax-aware Markdown chunker. Segments the document into semantic blocks
(headers, code blocks, lists, blockquotes, tables, horizontal rules, paragraphs)
and assembles chunks at structural boundaries.

#### Config — `MarkdownChunkingConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `split_on_elements` | `["h1", "h2", "h3", "code_block", "table", "horizontal_rule"]` | Element types that force a chunk boundary. |
| `preserve_whole_elements` | `["code_block", "table"]` | Elements emitted as a single indivisible chunk, even if oversized. |
| `strip_elements` | `True` | Remove `^#{1,6}\s` syntax from emitted text. |
| `preserve_original_content` | `False` | If True, emit raw markdown and keep absolute char indices accurate. |
| `text_chunker_to_use` | `RecursiveChunker(512/50)` (`get_default_text_chunker`) | Sub-chunker used to split a single block whose text exceeds `chunk_size`. |

#### Internal model — `_SemanticBlock`

`NamedTuple(type, raw_content, start_index, end_index, metadata)`. Types include
`h1`–`h6`, `code_block`, `list_item`, `blockquote`, `table`, `horizontal_rule`,
and `paragraph` for text between matches.

#### Algorithm

1. `_segment_markdown(text)` runs a single regex with named alternation
   (`h1_h6`, `code_block`, `list_item`, `blockquote`, `table`, `horizontal_rule`)
   over the full text and emits `_SemanticBlock`s. Gaps between matches become
   `paragraph` blocks. For headers it parses the level and stores `header_text`.
2. The main loop maintains a `header_stack` (each entry like `{"h2": "Setup"}`)
   that is pruned whenever a same-or-higher level header arrives — the resulting
   stack is attached to every chunk as metadata so RAG can keep section context.
3. Whenever a boundary or preserved element is encountered, the accumulated
   blocks are finalized via `_finalize_chunk(blocks, header_stack, document)`.
4. `_finalize_chunk` joins `raw_content`, optionally strips header syntax, and:
   - if the joined text fits, emits one chunk;
   - otherwise builds a temporary `Document` and runs
     `text_chunker_to_use.chunk([temp_doc])`, translating sub-chunk indices back
     into absolute coordinates (`start_index + sub_chunk.start_index`).
5. Final pass: merge a too-small trailing chunk into its predecessor with
   `"\n\n"` join.

### 4.4 `HTMLChunker` (`html_chunker.py`)

DOM-aware chunker built on BeautifulSoup4 (with `lxml`). Optional dependency: the
import path raises `ImportError("BeautifulSoup4 is required ...")` on construction
if `bs4` isn't installed. Registration in the factory is wrapped in try/except so
HTMLChunker simply doesn't appear in `list_available_strategies()` when bs4 is
missing.

#### Config — `HTMLChunkingConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `split_on_tags` | `["h1"–"h6", "p", "li", "table"]` | Tags that mark semantic boundaries. |
| `tags_to_ignore` | `["script", "style", "nav", "footer", "aside", "header", "form", "head", "meta", "link"]` | Decomposed before any processing. |
| `tags_to_extract` | `None` | Allowlist; if set, only these tags survive sanitization. (split_on_tags are auto-added so segmentation still works.) |
| `preserve_whole_tags` | `["table", "pre", "code", "ul", "ol"]` | Indivisible — emitted as a single chunk even if oversized. |
| `extract_link_info` | `True` | Rewrites `<a href>` text to `"Link Text (https://...)"`. |
| `preserve_html_content` | `False` | If True, store raw HTML in chunks (preserves indices); else extract clean text via `get_text(strip=True)`. |
| `text_chunker_to_use` | `RecursiveChunker(512/50)` | Sub-chunker for oversized blocks. |
| `merge_small_chunks` | `True` | Run the post-pass merge step. |
| `min_chunk_size_ratio` | `0.3` (0–1) | Merge threshold = `chunk_size * ratio`. |

#### Pipeline

1. **Parse & sanitize** (`_parse_and_sanitize`): build the soup, decompose
   ignored tags, optionally rebuild a new soup containing only `tags_to_extract`.
2. **Segment DOM** (`_segment_dom`): walk every match of `split_on_tags` in
   document order. For each tag, the block content is the tag plus all next
   siblings up to (but not including) the next split-tag. For each block, capture
   the running header stack (`h1`–`h6` ancestors via `find_previous`) into
   metadata. Compute absolute char offsets in the raw HTML by regex-matching
   `<tag[^>]*>...</tag>` (`_calculate_tag_indices`).
3. **Chunk per block**: a preserved tag → one chunk. A small block → one chunk
   with raw HTML content. An oversized block → fed through `text_chunker_to_use`
   with absolute offsets recomputed.
4. **Merge small chunks** (`_merge_small_chunks`): if `merge_small_chunks` is on,
   walk chunks; while a chunk's size is below `chunk_size * min_chunk_size_ratio`,
   greedily merge with the next chunk (separator is `""` if HTML preserved else
   `" "`) up to `chunk_size * 1.5`.

### 4.5 `JSONChunker` (`json_chunker.py`)

Path-aware JSON splitter. Operates on parsed JSON, not the raw string, so chunks
are always **valid JSON** and each chunk's metadata records which JSON paths it
contains. Falls back to `RecursiveChunker` if the document fails to parse.

#### Config — `JSONChunkingConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `convert_lists_to_dicts` | `True` | Recursively converts arrays to dicts keyed by stringified index, enabling long arrays to be split. |
| `max_depth` | `50` | Recursion safeguard. Beyond this depth a value is treated as atomic. |
| `json_encoder_options` | `{}` | Forwarded to `json.dumps` (e.g. `{"indent": 2, "ensure_ascii": False}`). |

#### Algorithm

1. `json.loads(document.content)` — on `JSONDecodeError`, call
   `_fallback_to_text_chunking` which spins up a `RecursiveChunker` with the same
   `chunk_size`/`chunk_overlap` and tags chunks with
   `chunking_fallback="json_to_text"`, `original_strategy="json"`.
2. `_preprocess_lists` turns lists into `{"0": ..., "1": ...}` so list items can be
   distributed across chunks.
3. `_recursive_walk` traverses the dict graph depth-first, calling `_add_to_chunk`
   for each leaf value. `_add_to_chunk`:
   - builds a one-key nested dict mirroring the path,
   - if adding the leaf would push the current chunk past `chunk_size` *and* the
     current chunk is at least `min_chunk_size`, opens a new chunk builder,
   - inserts the value via `_set_nested_dict` and appends the dotted path to
     `chunk_builders[-1]["paths"]`.
4. Each chunk builder is serialized with `json.dumps(content, **options)`, and
   the chunk metadata gets `chunk_json_paths: [...]`. Note: `start_index` and
   `end_index` are intentionally `None` because indices into the parsed/serialized
   JSON don't map back cleanly to the original string.

### 4.6 `PythonChunker` (`python.py`)

AST-driven Python source chunker. Extracts class/function/method definitions as
their own semantic blocks with full source slices and rich metadata.

#### Config — `PythonChunkingConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `split_on_nodes` | `["ClassDef", "FunctionDef", "AsyncFunctionDef"]` | AST node types that become chunk boundaries. |
| `min_chunk_lines` | `1` | Skip blocks shorter than N lines. |
| `include_docstrings` | `True` | If False, strip docstrings from the chunk body. |
| `strip_decorators` | `False` | If True, remove `@decorator` lines. |
| `text_chunker_to_use` | `RecursiveChunker(512/50, separators=["\n\n","\n"," ",""])` | Sub-chunker for oversized methods/classes. |

#### Algorithm

1. `_segment_python_code`: `ast.parse(code)` then run `_CodeVisitor` (an
   `ast.NodeVisitor`). The visitor tracks a `context_stack` so a method's
   `full_name_path` is e.g. `"OuterClass.method"`. Each emitted `_SemanticBlock`
   carries `type` (`class`, `function`, or `method`), `name`, `full_name_path`,
   `raw_content` (sliced from `source_lines`), `start_line`, `end_line`,
   `docstring`. On `SyntaxError`, an empty list is returned.
2. Per block, build `metadata = {path, type, start_line, end_line}`. Translate
   line numbers to character offsets using a precomputed
   `line_start_indices = [0] + [m.end() for m in re.finditer(r'\n', content)]`.
3. If the block fits, emit one chunk. Otherwise feed the block content through
   `text_chunker_to_use` and remap sub-chunk offsets back into the document's
   absolute coordinate system.
4. Optional decorator stripping (`_strip_decorators_from_string`) drops lines
   starting with `@`. Optional docstring stripping uses `str.replace` once for
   each quote style.
5. Final pass merges a tiny trailing chunk into its predecessor by re-slicing
   `document.content[prev.start_index:last.end_index]`.

### 4.7 `SemanticChunker` (`semantic.py`)

Embedding-driven topic-shift chunker. Embeds each sentence and finds points where
adjacent-sentence cosine distance exceeds a statistical threshold.

#### Config — `SemanticChunkingConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `embedding_provider` | **required** | An `EmbeddingProvider` from `upsonic.embeddings.base`. |
| `breakpoint_threshold_type` | `PERCENTILE` | One of `PERCENTILE`, `STD_DEV`, `INTERQUARTILE`, `GRADIENT`. |
| `breakpoint_threshold_amount` | `95.0` | For PERCENTILE, value 0–100 (validated). |
| `sentence_splitter` | regex splitter | Function `str → List[str]`. Defaults to a regex that respects abbreviations (`Mr.`, etc.). Replaceable with NLTK/spaCy. |

#### Algorithm

1. `_segment_into_sentences` splits the document and re-finds each sentence in the
   original text to compute exact `start_index`/`end_index`. Sentences that can't be
   located emit a warning and are skipped.
2. If fewer than 2 sentences, return the whole document as one chunk.
3. `await embedding_provider.embed_texts(sentence_texts, show_progress=False)` —
   this is why `_chunk_document` runs the async path under
   `concurrent.futures.ThreadPoolExecutor → asyncio.run` when called inside an
   already-running event loop.
4. `_calculate_distances` computes `1 - cosine_similarity` between every adjacent
   pair using NumPy.
5. `_calculate_breakpoint_threshold` picks a cutoff via the chosen statistical
   method:
   - `PERCENTILE`: `np.percentile(distances, amount)`,
   - `STD_DEV`: `mean + amount * std`,
   - `INTERQUARTILE`: `mean + amount * IQR`,
   - `GRADIENT`: percentile of `np.gradient(distances)`.
6. Indices where `distance > threshold` are breakpoints. Sentences between
   consecutive breakpoints are joined with `" "` into one chunk; the absolute
   `start_index`/`end_index` come from the first/last sentence of the group.
7. A small trailing chunk is merged into its predecessor.

### 4.8 `AgenticChunker` (`agentic.py`)

The premium strategy. Uses an Upsonic `Agent` to extract atomic propositions,
group them into thematic topics, and emit chunks with AI-generated titles and
summaries. Fully cached and resilient — falls back to `RecursiveChunker` on
failure.

#### Config — `AgenticChunkingConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `max_agent_retries` | `3` | Retries per agent call (proposition extraction, topic grouping, refinement). |
| `min_proposition_length` | `20` | Min characters for a valid proposition. |
| `max_propositions_per_chunk` | `15` | Cap per chunk. |
| `min_propositions_per_chunk` | `3` | Floor per chunk. |
| `enable_proposition_caching` | `True` | Cache by MD5 of `document.content`. |
| `enable_topic_caching` | `True` | Cache by MD5 of proposition list. |
| `enable_refinement_caching` | `True` | Cache by MD5 of chunk text. |
| `enable_proposition_validation` | `True` | Strip and dedupe propositions. |
| `enable_topic_optimization` | `True` | Merge under-sized topics. |
| `enable_coherence_scoring` | `True` | Add `coherence_score` and `quality_assessment` to metadata. |
| `fallback_to_recursive` | `True` | Use recursive chunking on failure. |
| `include_proposition_metadata` | `True` | Add first 5 propositions and total count to metadata. |
| `include_topic_scores` | `True` | Add coherence score/quality assessment. |
| `include_agent_metadata` | `True` | Add `agent_calls`, `cache_hits`, `processing_time_ms`. |

#### Pipeline

1. **Extract propositions** (`_extract_propositions`): submit a `Task` with
   `response_format=PropositionList` to `self.agent.do_async(task)`. Cached by
   document hash. Wrapped with `@upsonic_error_handler(max_retries=1)`.
2. **Validate propositions** (`_validate_propositions`): trim, drop too-short or
   duplicate strings.
3. **Group into topics** (`_group_propositions_into_topics`): another agent call
   with `response_format=TopicAssignmentList`. Followed by `_validate_topic_sizes`
   which splits any topic exceeding `max_propositions_per_chunk` into multiple
   `Topic` objects.
4. **Optimize topics** (`_optimize_topics`): collect propositions from
   below-floor topics, then re-cluster them into right-sized topics.
5. **Build chunks** (`_create_chunks_from_topics`): greedy fill up to
   `chunk_size`, finalize via `_finalize_topic_chunk` which calls
   `_refine_topic_metadata` (third agent call) for an `agentic_title` and
   `agentic_summary`. Chunk text is `" ".join(propositions)`. The chunk's
   `start_index`/`end_index` are recovered from the original document via
   `_find_chunk_indices_in_document` — exact substring match, then prefix-N-words
   fallback, then any sufficiently-long word, then `(0, len(text))`.
6. **Score coherence** (`_score_chunk_coherence`,
   `_calculate_coherence_score`): heuristic
   `0.4*length_score + 0.6*proposition_score`, both clamped to 1.0.
   `_assess_chunk_quality` maps to `excellent`/`good`/`fair`/`poor`.
7. **Add processing metadata** (`_add_processing_metadata`): `agent_calls`,
   `cache_hits`, `processing_time_ms`, `processing_stage`.
8. **Fallback** (`_fallback_chunking`): on any exception, run
   `RecursiveChunker(chunk_size, chunk_overlap)._chunk_document(document)` and
   tag chunks with `agentic_fallback=True`, `chunking_method="recursive_fallback"`.

#### Public stats / cache control

```python
chunker.get_agentic_stats() -> Dict       # call counts, cache sizes, feature flags
chunker.clear_agentic_caches() -> None    # reset propositions/topics/refinements
```

#### Schemas used

`AgenticChunker` consumes three Pydantic models from
`upsonic.schemas.agentic`: `PropositionList`, `TopicAssignmentList`, `Topic`,
`RefinedTopic`. The agent returns these as structured outputs so the chunker can
work without parsing free-form text.

---

## Splitter cheat sheet (strategy → strategy/parameters)

| Strategy | Best for | Splits on | Key params (beyond base) | Async-native | Optional deps |
|----------|----------|-----------|--------------------------|--------------|---------------|
| `character` | Logs, simple text, fixed delimiters | Single regex/literal separator | `separator`, `is_separator_regex`, `keep_separator` | No | — |
| `recursive` | General text, code, mixed content | Prioritized separators with recursion | `separators`, `keep_separator`, `is_separator_regex`; `from_language(...)` for PYTHON/MARKDOWN/HTML/LATEX/JAVA/JS | No | — |
| `markdown` | `.md`/`.markdown` docs | Headers, code blocks, tables, lists | `split_on_elements`, `preserve_whole_elements`, `strip_elements`, `preserve_original_content`, `text_chunker_to_use` | No | — |
| `html` | `.html`/`.htm` web pages | DOM tags via BeautifulSoup | `split_on_tags`, `tags_to_ignore`, `tags_to_extract`, `preserve_whole_tags`, `extract_link_info`, `preserve_html_content`, `text_chunker_to_use`, `merge_small_chunks`, `min_chunk_size_ratio` | No | `beautifulsoup4`, `lxml` |
| `json` | API payloads, configs | JSON graph paths | `convert_lists_to_dicts`, `max_depth`, `json_encoder_options` | No | — |
| `python` | `.py` source | AST: ClassDef/FunctionDef/AsyncFunctionDef | `split_on_nodes`, `min_chunk_lines`, `include_docstrings`, `strip_decorators`, `text_chunker_to_use` | No | — |
| `semantic` | Narrative, research docs | Sentence-embedding cosine distances | `embedding_provider` (req.), `breakpoint_threshold_type`, `breakpoint_threshold_amount`, `sentence_splitter` | Yes (true async) | `numpy`, an `EmbeddingProvider` |
| `agentic` (alias `ai`) | Premium RAG, QA over complex docs | Agent-extracted propositions and topics | `max_agent_retries`, proposition/topic min/max, caching flags, `fallback_to_recursive`, metadata flags | Yes (true async) | `Agent` instance |

### Metadata each strategy adds to `Chunk.metadata`

| Strategy | Added metadata keys |
|----------|---------------------|
| `character`/`recursive` | none beyond base + parent metadata |
| `markdown` | one entry per active header in stack, e.g. `{"h1": "...", "h2": "..."}` |
| `html` | per-block headers (h1..h6), via DOM ancestor scan |
| `json` | `chunk_json_paths: List[str]` |
| `python` | `path`, `type`, `start_line`, `end_line` |
| `semantic` | none beyond base + parent metadata |
| `agentic` | `agentic_title`, `agentic_summary`, `topic_ids`, `proposition_count`, `chunking_method`, `agent_processed`, `merged_topics`, `propositions` (truncated), `total_propositions`, `coherence_score`, `quality_assessment`, `agent_calls`, `cache_hits`, `processing_time_ms`, `processing_stage` |
| Any (on error) | `chunking_error: True`, `error_message: str` |
| Any (fallback) | `agentic_fallback`, `chunking_method`, or `chunking_fallback`, `original_strategy` |

---

## 5. Cross-file relationships

- `base.py` is the only required dependency for every strategy file. It provides
  the abstract `BaseChunker` class, the shared `BaseChunkingConfig`, the
  `_create_chunk` factory, and the `_get_effective_min_chunk_size` helper used by
  almost every strategy's tail-merge step.
- `recursive.py` is the *gravity well* of the package: `markdown.py`, `html_chunker.py`,
  `python.py`, `json_chunker.py`, and `agentic.py` all import either
  `RecursiveChunker` or `RecursiveChunkingConfig`. The first three use it as
  `text_chunker_to_use` (sub-chunker for oversized blocks); the last two use it
  as a fallback when their primary strategy fails.

```
                ┌──────────────────────────┐
                │ base.py                  │
                │   BaseChunker            │
                │   BaseChunkingConfig     │
                └──────────┬───────────────┘
                           │ inherits
   ┌─────────┬─────────┬───┴─────┬──────────┬──────────┬──────────┐
   ▼         ▼         ▼         ▼          ▼          ▼          ▼
character recursive markdown  html_chunker  python   semantic   agentic
              ▲           │       │           │                    │
              │           │       │           │                    │
              └───────────┴───────┴───────────┴────────── used as ─┘
                       text_chunker_to_use / fallback
```

- `factory.py` imports every strategy lazily (try/except) and registers them by
  string key. It is the only file the wider codebase needs to import to get a
  chunker; nothing outside the package directly imports a `*Chunker` class
  unless it needs the type for `isinstance` or static typing.
- `__init__.py` provides lazy attribute access so importing the package doesn't
  pay for `numpy`, `bs4`, or the agent system unless those names are actually
  referenced.

---

## 6. Public API

These are the names exported via `__all__` in `__init__.py`:

```python
from upsonic.text_splitter import (
    # Base contract
    BaseChunker, BaseChunkingConfig,

    # Strategies and their configs
    CharacterChunker, CharacterChunkingConfig,
    RecursiveChunker, RecursiveChunkingConfig,
    HTMLChunker,      HTMLChunkingConfig,
    JSONChunker,      JSONChunkingConfig,
    MarkdownChunker,  MarkdownChunkingConfig,
    PythonChunker,    PythonChunkingConfig,
    SemanticChunker,  SemanticChunkingConfig,
    AgenticChunker,   AgenticChunkingConfig,

    # Factory functions
    create_chunking_strategy,
    create_adaptive_strategy,
    create_rag_strategy,
    create_semantic_search_strategy,
    create_fast_strategy,
    create_quality_strategy,
    create_intelligent_splitters,
    list_available_strategies,
    get_strategy_info,
    detect_content_type,
    recommend_strategy_for_content,

    # Enums
    ContentType, ChunkingUseCase,
)
```

`Language` and `BreakpointThresholdType` are exported by their respective
sub-modules (`upsonic.text_splitter.recursive`,
`upsonic.text_splitter.semantic`).

### Minimal usage

```python
from upsonic.schemas.data_models import Document
from upsonic.text_splitter import create_chunking_strategy

splitter = create_chunking_strategy("recursive", chunk_size=800, chunk_overlap=120)
chunks = splitter.chunk([Document(content=open("readme.md").read())])
```

### Adaptive selection

```python
from upsonic.text_splitter import create_adaptive_strategy, ChunkingUseCase

splitter = create_adaptive_strategy(
    content=text_sample,
    metadata={"source": "spec.md"},
    use_case=ChunkingUseCase.RAG_RETRIEVAL,
    quality_preference="balanced",
)
```

### Per-source intelligent selection (used by `KnowledgeBase`)

```python
from pathlib import Path
from upsonic.text_splitter import create_intelligent_splitters, ChunkingUseCase

splitters = create_intelligent_splitters(
    sources=[Path("a.md"), Path("b.py"), Path("c.json")],
    use_case=ChunkingUseCase.RAG_RETRIEVAL,
    quality_preference="balanced",
    embedding_provider=my_embedder,  # only used if 'semantic' is chosen
)
```

### Semantic chunking (true async)

```python
import asyncio
from upsonic.text_splitter import SemanticChunker, SemanticChunkingConfig

cfg = SemanticChunkingConfig(
    embedding_provider=my_embedder,
    chunk_size=1024,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95.0,
)
chunks = asyncio.run(SemanticChunker(cfg).achunk([doc]))
```

### Agentic chunking

```python
from upsonic.agent.agent import Agent
from upsonic.text_splitter import AgenticChunker, AgenticChunkingConfig

agent = Agent("openai/gpt-4o")
chunker = AgenticChunker(agent, AgenticChunkingConfig(chunk_size=1500))
chunks = await chunker.achunk([doc])
print(chunker.get_agentic_stats())
```

### Registering a custom strategy

```python
from upsonic.text_splitter.factory import register_chunking_strategy
from upsonic.text_splitter.base import BaseChunker, BaseChunkingConfig

class MyChunker(BaseChunker[BaseChunkingConfig]):
    def _chunk_document(self, document):
        ...

register_chunking_strategy("mine", MyChunker, BaseChunkingConfig)
```

---

## 7. Integration with the rest of Upsonic

The text_splitter package is consumed primarily by `upsonic.knowledge_base.KnowledgeBase`.
References (from `src/upsonic/knowledge_base/knowledge_base.py`):

- L15: `from ..text_splitter.base import BaseChunker`
- L22: `from ..text_splitter.factory import create_intelligent_splitters`
- L63 / L143: `KnowledgeBase` accepts
  `splitters: Optional[Union[BaseChunker, List[BaseChunker]]] = None` and stores
  them on `self.splitters: List[BaseChunker]`.
- L468–L532: `_setup_splitters` and `_normalize_splitters`. If the caller passes
  no splitters, `KnowledgeBase` calls
  `create_intelligent_splitters(sources, use_case=RAG_RETRIEVAL, ...)`. On
  failure it falls back to `create_chunking_strategy("recursive")`.
- L1106 / L1130 / L1604: per-document fallback paths — a `RecursiveChunker` is
  constructed if a configured splitter chokes on a particular document.
- L1438 / L1469 / L1539 / L1562 / L1630: explicit `add_*` and `_load_*` entry
  points accept an optional `BaseChunker`; when omitted, `KnowledgeBase` again
  calls `create_intelligent_splitters` or `create_chunking_strategy("recursive")`.

In effect, every Upsonic code path that turns documents into vector-DB rows
funnels through `BaseChunker.chunk(...)`, regardless of whether the caller
supplied an explicit splitter or relied on auto-detection.

Other consumers:

- `upsonic.agent.Agent` is consumed *by* `AgenticChunker` (not the reverse).
  `factory.create_chunking_strategy("agentic", ...)` will instantiate
  `Agent("openai/gpt-4o")` if no agent is provided.
- `upsonic.embeddings.base.EmbeddingProvider` is consumed *by* `SemanticChunker`.
  Any provider in `upsonic/embeddings/` (OpenAI, HuggingFace, Bedrock, etc.) can
  be passed in via `SemanticChunkingConfig.embedding_provider`.
- `upsonic.schemas.data_models.Document` / `Chunk` is the data interchange
  format. Loaders in `upsonic.loaders.*` produce `Document` objects; chunkers
  produce `Chunk` objects; vector stores in `upsonic.vectordb.*` consume
  `Chunk` objects.

---

## 8. End-to-end flow

The following walkthrough traces a `KnowledgeBase` call from raw files to chunks.

```text
┌────────────────────────────────────────────────────────────────────┐
│ 1. Caller                                                          │
│    KnowledgeBase(sources=[Path("docs/spec.md"), Path("a.py")])     │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ 2. KnowledgeBase._setup_splitters                                  │
│    - splitter argument is None → call factory                      │
│    - create_intelligent_splitters(sources, use_case=RAG_RETRIEVAL) │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ 3. factory.create_intelligent_splitters                            │
│    For each source:                                                │
│      a. read up to 5 KB sample                                     │
│      b. _create_source_optimized_config (size→chunk_size,          │
│         extension→chunk_overlap, use_case adjustments)             │
│      c. create_adaptive_strategy(                                  │
│           content=sample, metadata={'source': str(source)},        │
│           use_case=RAG_RETRIEVAL,                                  │
│           quality_preference='balanced', ...)                      │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ 4. factory.create_adaptive_strategy                                │
│      a. detect_content_type(sample, metadata)                      │
│         spec.md → ContentType.MARKDOWN                             │
│         a.py   → ContentType.PYTHON                                │
│      b. recommend_strategy_for_content(...)                        │
│         MARKDOWN → "markdown"                                      │
│         PYTHON   → "python"                                        │
│      c. _create_optimized_config(...)                              │
│         (chunk_size=1500 for code, 1000 default; overlap 25/15%)   │
│      d. create_chunking_strategy(strategy_name, config=...)        │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ 5. factory.create_chunking_strategy                                │
│      - lookup _STRATEGY_REGISTRY[strategy_name]                    │
│      - special handling for 'semantic' / 'agentic'                 │
│      - _create_final_config(config_dict, ConfigClass, kwargs)      │
│      - return ChunkerClass(config=config)                          │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ 6. KnowledgeBase processes documents                               │
│      for doc, splitter in zip(documents, splitters):               │
│          chunks = splitter.chunk([doc])                            │
│      OR (async path):                                              │
│          chunks = await splitter.achunk([doc])                     │
│      OR (parallel async):                                          │
│          chunks = await splitter.abatch(documents)                 │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ 7. BaseChunker.chunk loop                                          │
│      - skip empty/whitespace docs                                  │
│      - delegate to _chunk_document(doc) (per-strategy logic)       │
│      - on exception: emit error chunk with chunking_error=True     │
│      - flatten and return List[Chunk]                              │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ 8. Strategy-specific work (illustrated for MarkdownChunker)        │
│      a. _segment_markdown(text) → List[_SemanticBlock]             │
│      b. iterate blocks, maintain header_stack                      │
│      c. on boundary/preserved → _finalize_chunk                    │
│         - if fits, emit one chunk                                  │
│         - else recurse into text_chunker_to_use (RecursiveChunker) │
│      d. final tail-merge if last chunk < min_chunk_size            │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ 9. Chunk objects flow downstream                                   │
│      - KnowledgeBase persists chunks                               │
│      - Embedding provider embeds chunk.text_content                │
│      - Vector DB indexes (chunk_content_hash, embedding, metadata) │
│      - Retrieval returns Chunks with metadata for citation         │
└────────────────────────────────────────────────────────────────────┘
```

### Notes on async behaviour

- `BaseChunker._achunk_document` defaults to wrapping the synchronous
  `_chunk_document`, so every chunker is async-compatible out of the box.
- `SemanticChunker` and `AgenticChunker` truly run async (network/AI calls).
  They detect a running event loop and, if present, run inside a
  `concurrent.futures.ThreadPoolExecutor` using `asyncio.run` to avoid
  re-entering the same loop.
- `BaseChunker.abatch` parallelizes per-document chunking with
  `asyncio.gather` — useful for `SemanticChunker`/`AgenticChunker`, neutral for
  the synchronous strategies.

### Notes on positional integrity

Every strategy returns chunks with `start_index`/`end_index` *into the original
document content*, except `JSONChunker` (which sets both to `None` because the
chunks come from a re-serialized object graph). This matters for highlights and
citations: callers can do `original.content[chunk.start_index:chunk.end_index]`
to get the exact source span.

### Notes on error handling

- Per-document errors do not abort the run; they emit an error chunk so the
  caller still has a record (`metadata.chunking_error=True`).
- Strategy-level fallbacks: `AgenticChunker` and `JSONChunker` fall back to
  `RecursiveChunker` on agent/parse failure. `KnowledgeBase` adds a second
  layer of fallback: if its assigned splitter raises, it constructs a fresh
  `RecursiveChunker` for that document.
- `factory.create_chunking_strategy` raises
  `ConfigurationError(error_code="UNKNOWN_STRATEGY"|"MISSING_EMBEDDING_PROVIDER"|"MISSING_AGENT")`
  for invalid configurations — callers should catch `ConfigurationError` and
  inspect `error_code` to react.

---

## Quick reference — choosing a strategy

| Situation | Recommended strategy | Factory call |
|-----------|----------------------|--------------|
| Plain text, speed > quality | `character` | `create_fast_strategy()` |
| Generic mixed text | `recursive` | `create_chunking_strategy("recursive")` |
| Markdown docs | `markdown` | `create_chunking_strategy("markdown")` |
| HTML pages | `html` | `create_chunking_strategy("html")` |
| JSON payloads | `json` | `create_chunking_strategy("json")` |
| Python source | `python` | `create_chunking_strategy("python")` |
| Long narrative, semantic search | `semantic` | `create_semantic_search_strategy(content, embedding_provider=...)` |
| Premium RAG/QA, complex docs | `agentic` | `create_quality_strategy(content, embedding_provider=..., agent=...)` |
| Don't know — let Upsonic decide | adaptive | `create_adaptive_strategy(content, use_case=...)` |
| Many heterogeneous sources at once | per-source adaptive | `create_intelligent_splitters([...])` |
