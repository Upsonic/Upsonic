---
name: knowledge-base-rag-orchestrator
description: Use when working with Upsonic's RAG knowledge base orchestrator that ties together loaders, text splitters, embeddings, and vector DBs into a single class. Use when a user asks to ingest documents, build a RAG pipeline, register a knowledge base as an agent tool, configure idempotent indexing, isolate searches across multiple KBs, or debug ingestion/query flows. Trigger when the user mentions KnowledgeBase, KBState, knowledge_base.py, setup_async, query_async, aadd_source, aadd_text, arefresh, search_<name> tool, build_context, knowledge_id, isolate_search, doc_content_hash, BaseVectorDBProvider, EmbeddingProvider, BaseLoader, BaseChunker, RAGSearchResult, RecursiveChunker, create_intelligent_loaders, create_intelligent_splitters, KnowledgeRow, RAG ingestion, hash-based change detection, or hybrid/dense/FTS search via a KB.
---

# `src/upsonic/knowledge_base/` — RAG Knowledge Base Abstraction

This document explains the `knowledge_base` package: its public API, internal
machinery, the way it composes with the rest of Upsonic's RAG stack
(`vectordb`, `embeddings`, `loaders`, `text_splitter`), and the end-to-end flow
of ingestion and querying.

The package is intentionally small in surface area — there is essentially one
class — but it sits at the *center* of Upsonic's RAG architecture and pulls
together five subsystems into a single, framework-agnostic, idempotent
orchestrator.

---

## 1. What this folder is — RAG/knowledge base abstraction

The `knowledge_base` package is the **central orchestrator** for
Retrieval-Augmented Generation in Upsonic. A `KnowledgeBase` instance:

- Takes raw input **sources** (file paths, directory paths, or direct content
  strings).
- Picks **loaders** (from `upsonic.loaders`) to parse those sources into
  `Document` objects.
- Picks **splitters/chunkers** (from `upsonic.text_splitter`) to break those
  documents into `Chunk` objects.
- Generates embeddings via an **`EmbeddingProvider`** (from
  `upsonic.embeddings`) — or skips this step entirely for vector-DB providers
  that handle embedding internally (e.g. SuperMemory).
- Persists chunks + vectors in a **`BaseVectorDBProvider`** (from
  `upsonic.vectordb`).
- Exposes the resulting collection through a single high-level `query_async()`
  method *and* through the **Tool Provider Protocol** (`get_tools()`) so that
  any `Agent`/`Direct` instance can register the KB as a callable RAG tool.
- Optionally writes a row per ingested document into a `Storage` backend so
  ingestion history survives across processes.

In short: it converts the question "*how do I let an LLM consult these
documents?*" into a single class, with idempotency, change-detection,
auto-detection of components, and a per-instance search tool baked in.

### Design properties

| Property              | What it means in this folder                                                                                                                                                |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Idempotent**        | `setup_async()` is guarded by an `asyncio.Lock` and a `KBState` enum. Re-running on identical sources is a no-op; re-running on changed sources only re-indexes the deltas. |
| **Async-first**       | All real work is `async`; sync wrappers (`setup`, `add_source`, …) defer to a fresh `asyncio.new_event_loop()` or a `ThreadPoolExecutor` when called inside a running loop. |
| **Auto-detecting**    | If `loaders=` / `splitters=` is omitted, intelligent factories scan source extensions and pick reasonable defaults.                                                          |
| **Provider-agnostic** | Talks to vector DBs through `BaseVectorDBProvider`, embeddings through `EmbeddingProvider`. No vendor lock-in.                                                              |
| **Tool provider**     | Implements `get_tools()` / `build_context()` so a `KnowledgeBase` can be passed straight into `Agent(tools=[kb])` or `Task(context=[kb])`.                                  |
| **Deterministic ID**  | A SHA-256 of `{sources, loader classes, splitter classes, embedding provider class}` becomes the `knowledge_id`. Changing components forces a fresh collection.             |

---

## 2. Folder layout

```
src/upsonic/knowledge_base/
├── __init__.py          # Lazy re-exports: KnowledgeBase, KBState
└── knowledge_base.py    # The full implementation (~2,166 lines)
```

That is the entire package — two files. Everything below documents
`knowledge_base.py`.

---

## 3. Top-level files

### 3.1 `__init__.py`

A 22-line lazy loader. It avoids importing the heavy `knowledge_base.py`
(which transitively pulls in vectordb, loaders, embeddings, etc.) until the
caller actually accesses `KnowledgeBase` or `KBState`.

```python
# src/upsonic/knowledge_base/__init__.py
_LAZY_MAP = {
    "KnowledgeBase": "KnowledgeBase",
    "KBState":       "KBState",
}

def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        from . import knowledge_base as _mod
        return getattr(_mod, _LAZY_MAP[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["KnowledgeBase", "KBState"]
```

This is a common pattern in Upsonic for keeping import time low when only a
fraction of the framework is used in a given run.

### 3.2 `knowledge_base.py`

The whole RAG orchestrator lives here. It defines two public symbols:

| Symbol          | Kind  | Purpose                                                                                          |
| --------------- | ----- | ------------------------------------------------------------------------------------------------ |
| `KBState`       | Enum  | Lifecycle states: `UNINITIALIZED → CONNECTED → INDEXED → CLOSED`.                                |
| `KnowledgeBase` | Class | The orchestrator. Manages sources, loaders, splitters, embeddings, vector DB, and the search tool. |

Everything else in the file is private (leading underscore) and supports those
two symbols.

#### 3.2.1 `KBState` (Enum)

```python
class KBState(str, Enum):
    UNINITIALIZED = "uninitialized"   # Constructed but no DB connection
    CONNECTED     = "connected"        # Vector DB is reachable
    INDEXED       = "indexed"          # All chunks have been upserted
    CLOSED        = "closed"           # Connections released; instance is dead
```

Inheriting from `str` lets the values be JSON-serialized and compared to
plain strings; the framework uses this when emitting health-check payloads.

State transitions are linear:

```
UNINITIALIZED ──aconnect()──▶ CONNECTED ──aupsert()──▶ INDEXED ──close()──▶ CLOSED
                                                            │
                                                  refresh()/setup(force=True)
                                                            ▼
                                                       CONNECTED → INDEXED
```

`CLOSED` is terminal — every mutating method raises `RuntimeError("Cannot
… a closed KnowledgeBase.")` once it is reached.

#### 3.2.2 `KnowledgeBase` — constructor

```python
KnowledgeBase(
    sources:             Union[str, Path, List[Union[str, Path]]],
    vectordb:            BaseVectorDBProvider,
    embedding_provider:  Optional[EmbeddingProvider] = None,
    splitters:           Optional[Union[BaseChunker, List[BaseChunker]]] = None,
    loaders:             Optional[Union[BaseLoader, List[BaseLoader]]]   = None,
    name:                Optional[str] = None,
    description:         Optional[str] = None,
    topics:              Optional[List[str]] = None,
    use_case:            str  = "rag_retrieval",
    quality_preference:  str  = "balanced",
    loader_config:       Optional[Dict[str, Any]] = None,
    splitter_config:     Optional[Dict[str, Any]] = None,
    isolate_search:      bool = True,
    storage:             Optional["Storage"] = None,
    **config_kwargs,
)
```

Construction is **lightweight** by design — no network I/O, no document
loading, no chunking. It only:

1. Validates that every file/directory source exists
   (`_validate_sources_exist`).
2. Resolves directory sources by walking them recursively for known
   extensions (`_get_supported_files_from_directory` — see allowed extensions
   list below).
3. Auto-creates loaders via `create_intelligent_loaders` if none were passed.
4. Auto-creates splitters via `create_intelligent_splitters` if none were
   passed (passing `use_case`, `quality_preference`, and the embedding
   provider through).
5. Computes `self.knowledge_id` (`_generate_knowledge_id`).
6. Auto-derives a `collection_name` of the form
   `kb_<sanitized_name>_<id_prefix>` if the vector DB config is still using
   `"default_collection"`.
7. Computes the **per-instance tool name** `search_<sanitized_name>` so
   multiple KBs can coexist on one agent.

Source extensions auto-walked from a directory:

```
.txt .md .rst .log .py .js .ts .java .c .cpp .h .cs .go .rs .php .rb
.html .css .xml .json .yaml .yml .ini .csv .pdf .docx .jsonl .markdown
.htm .xhtml
```

Anything else inside a directory is silently skipped.

#### 3.2.3 Tool Provider Protocol

`KnowledgeBase` is a **first-class tool provider**. The agent system does not
need to know it is a "knowledge base"; it just calls `get_tools()` /
`build_context()`.

```python
def get_tools(self) -> List["Tool"]:
    from upsonic.tools.wrappers import FunctionTool

    async def search_knowledge_base(query: str) -> str:
        return await self._search_impl(query)

    return [FunctionTool.from_callable(
        search_knowledge_base,
        name=self._search_tool_name,            # e.g. "search_technical_docs"
        description=f"Search the '{self.name}' knowledge base ..."
    )]
```

Two important consequences:

- **Per-instance tool naming.** A `KnowledgeBase(name="legal_docs")` becomes
  `search_legal_docs`; a `KnowledgeBase(name="product")` becomes
  `search_product`. An agent can register many KBs without name collisions.
- **`build_context()` returns a `<knowledge_base>…</knowledge_base>` block**
  that is injected into the agent's system prompt, telling the model the KB
  exists, what it is for, and which tool to call.

```text
<knowledge_base>
You have access to a knowledge base called 'technical_docs' that you can
search using the search_technical_docs tool.
Knowledge base description: Knowledge base for technical_docs
Topics covered: python, kubernetes
Always search this knowledge base before answering questions related to its
topics — do not assume you already know the answer. For ambiguous questions,
search first rather than asking for clarification.
</knowledge_base>
```

`aget_tools()` and `abuild_context()` are simple async re-exports for
symmetry.

#### 3.2.4 The just-in-time engine: `setup_async()`

`setup_async(force: bool = False)` is the **main pipeline**. It is wrapped in
an `asyncio.Lock` so concurrent callers serialize, and it short-circuits if
the KB is already `INDEXED` and `force=False`.

The pipeline runs in clearly numbered steps:

| Step | Method                       | What happens                                                                                                                                |
| ---: | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
|    0 | `_ensure_connection`         | Opens (or reuses) the vector-DB connection.                                                                                                  |
|    1 | `_load_documents`            | For each source: direct content → `Document` synthesized in-memory; file → loader.load(). Tracks `source_to_documents` / `source_to_loader`. |
|  1.5 | `_filter_changed_documents`  | Per document: if `doc_content_hash` exists → skip; else if `document_id` exists → delete old chunks; else → new.                            |
|    2 | `_chunk_documents`           | Picks splitter per source, with a fallback to `RecursiveChunker` if the primary splitter yields zero chunks or raises.                      |
|    3 | `_generate_embeddings`       | `embed_documents(chunks)` — or zero-vector placeholders if `embedding_provider is None`.                                                    |
|    4 | `_store_in_vectordb`         | Creates the collection if missing, then `aupsert(...)` chunks + vectors + metadata, with `knowledge_base_id` injected when isolation is on. |
|    5 | (storage hook)               | If `storage` was supplied, build a `KnowledgeRow` per indexed document and `upsert_knowledge_content`.                                       |

If anything before step 4 fails *and* the collection did not pre-exist, the
partially-created collection is dropped (`adelete_collection()`) so the next
`setup_async()` can start clean.

#### 3.2.5 Hash-based change detection

`_filter_changed_documents` is the function that makes ingestion cheap on
re-runs. For every loaded `Document`:

```python
if await self.vectordb.adoc_content_hash_exists(doc.doc_content_hash):
    continue                                      # unchanged → skip
if await self.vectordb.adocument_id_exists(doc.document_id):
    await self.vectordb.adelete_by_document_id(doc.document_id)   # edited → replace
changed_docs.append(doc)                          # new or replaced
```

`doc_content_hash` is an MD5 of the raw file content; `document_id` is set by
the loader (typically a stable hash of the file path or the file content).
This dual-key check supports both *unchanged* (skip) and *edited* (replace)
cases without re-embedding everything.

#### 3.2.6 Search & retrieval

```python
async def query_async(
    query: str,
    top_k: Optional[int] = None,
    filter: Optional[Dict[str, Any]] = None,
    task: Optional[Any] = None,
    alpha: Optional[float] = None,
    fusion_method: Optional[Literal["rrf", "weighted"]] = None,
    similarity_threshold: Optional[float] = None,
    apply_reranking: bool = True,
    sparse_query_vector: Optional[Dict[str, Any]] = None,
) -> List[RAGSearchResult]
```

Behavior:

1. Calls `setup_async()` (lazy idempotent indexing).
2. Embeds the query (`embed_query`) — or makes a zero-vector if no
   embedding provider is configured.
3. Pulls overrides from a `Task` if one is passed (`vector_search_top_k`,
   `vector_search_alpha`, `vector_search_fusion_method`,
   `vector_search_similarity_threshold`, `vector_search_filter`).
4. If `isolate_search=True` (default), AND-merges
   `{"knowledge_base_id": self.knowledge_id}` into the filter so that two
   KBs sharing one collection do not bleed into each other's results.
5. Calls `vectordb.asearch(...)` — the dense / full-text / hybrid choice is
   made *inside* the vector DB provider, not here.
6. Converts `VectorSearchResult` → `RAGSearchResult` via
   `_convert_to_rag_results`. If `result.text` is empty, it falls back to
   `payload["content"]`, `payload["chunk"]`, or `payload["text"]` — which
   means RAG works even with vector DBs that store text in different payload
   keys.

`search(query: str) -> str` is a thin async wrapper around `_search_impl`
that returns a string already formatted for an LLM prompt:

```text
Result 1:
<chunk text>

Result 2:
<chunk text>
```

If the search yields nothing, it returns
`"No relevant information found in the knowledge base."`.

#### 3.2.7 Dynamic content management

Beyond the bulk pipeline, the class supports incremental edits:

| Method                      | What it does                                                                                                                                  |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `aadd_source(...)`          | Add a new file/dir/string after setup. Auto-detects loader+splitter unless overrides are passed. Skips already-ingested content via hash.     |
| `aadd_text(...)`            | Insert raw text (no file). Uses `RecursiveChunker` by default. Returns the document_id (deduplicated if the same text was already present).   |
| `aremove_document(doc_id)`  | `vectordb.adelete_by_document_id(...)` and removes the matching `self.sources` entry.                                                        |
| `arefresh()`                | `loader.reset()` for all loaders, then `setup_async(force=True)` — a full rescan of existing sources.                                        |
| `adelete_by_filter(...)`    | Deletes any chunks whose metadata matches a filter.                                                                                           |
| `aupdate_document_metadata` | Pulls all chunks for `document_id` and re-writes their payload.                                                                              |

Every async method has a sync sibling (`add_source`, `add_text`, …) that uses
either a brand-new event loop or a `ThreadPoolExecutor` when an event loop is
already running. This keeps the API usable from notebooks, scripts, FastAPI,
and inside other async frameworks alike.

#### 3.2.8 Diagnostics

| Method                       | Returns                                                                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `get_collection_info_async`  | Provider-specific collection metadata (or a generic dict with `collection_name`, `exists`, `provider`, `processing_stats`).         |
| `get_config_summary`         | Big self-describing dict: knowledge_id, sources, loader/splitter/embedding/vectordb classes, processing stats.                       |
| `health_check_async`         | Per-component health: embeddings (calls `validate_connection` if available), splitters, loaders, vectordb (`ais_ready`).             |
| `optimize_vectordb`          | Forwards to `vectordb.aoptimize()` (e.g. compact / reindex on Qdrant or pgvector).                                                  |
| `markdown()`                 | One-line markdown digest: `# Knowledge Base: <name>\n\nSources: …`.                                                                  |

#### 3.2.9 Lifecycle

```python
async with KnowledgeBase(sources=["docs/"], vectordb=...) as kb:
    results = await kb.query_async("How do I deploy?")
# kb.close() runs automatically here
```

`close()`:
- Calls `embedding_provider.close()` if provided (sync or async).
- Calls `vectordb.adisconnect()`.
- Sets state to `CLOSED` (idempotent — calling `close()` twice is fine).

`__del__` is a best-effort fallback that emits a warning if the KB was not
explicitly closed and tries to call `vectordb.disconnect()`. Always prefer
`async with` or an explicit `await kb.close()`.

---

## 4. Subfolders

There are no subfolders. The package is a single Python module plus its
lazy-loading `__init__.py`. All complexity lives in `knowledge_base.py`.

---

## 5. Cross-file relationships

`KnowledgeBase` is glue — almost every line delegates to a sibling package.

```
                ┌──────────────────────┐
                │   KnowledgeBase      │
                └──────────┬───────────┘
                           │
   ┌───────────────────────┼─────────────────────────────────────────┐
   │                       │                                         │
   ▼                       ▼                                         ▼
loaders/               text_splitter/                          embeddings/
  BaseLoader             BaseChunker                             EmbeddingProvider
  factory.create_         factory.create_                          .embed_documents()
   intelligent_            intelligent_                            .embed_query()
   loaders                 splitters
                           factory.create_
                            chunking_strategy
                                                                   ▲
                                                                   │ used by both
                                                                   │ ingest + query
                           ┌─────────────┐                         │
                           │  vectordb/   │◀────────────────────────┘
                           │  BaseVectorDBProvider
                           │  .acreate_collection / .aupsert
                           │  .asearch / .adelete_by_document_id
                           │  .adoc_content_hash_exists
                           └─────────────┘
                                  ▲
                                  │
                  schemas/data_models.py: Document, Chunk, RAGSearchResult
                  schemas/vector_schemas.py: VectorSearchResult

                                  ▲
                                  │ optional
                                  │
                          storage/ (Storage + KnowledgeRow)
                          → upsert_knowledge_content / get_knowledge_content
                            / delete_knowledge_content
```

### Concrete imports performed by `knowledge_base.py`

```python
from ..text_splitter.base       import BaseChunker
from ..embeddings.base          import EmbeddingProvider
from ..vectordb.base            import BaseVectorDBProvider
from ..loaders.base             import BaseLoader
from ..schemas.data_models      import Document, RAGSearchResult, Chunk
from ..schemas.vector_schemas   import VectorSearchResult
from ..loaders.factory          import create_intelligent_loaders
from ..text_splitter.factory    import create_intelligent_splitters
from ..utils.printing           import info_log, debug_log, warning_log, error_log, success_log
from upsonic.utils.package.exception import VectorDBConnectionError, UpsertError

# Lazy / type-only:
from upsonic.storage.base        import Storage           # (TYPE_CHECKING)
from upsonic.storage.schemas     import KnowledgeRow      # (lazy)
from upsonic.tools.base          import Tool              # (TYPE_CHECKING)
from upsonic.tools.wrappers      import FunctionTool      # (lazy, in get_tools)
```

This list is essentially the contract: any new vector DB / loader / splitter /
embedding provider only has to satisfy its respective base interface, and
`KnowledgeBase` will pick it up unchanged.

### How each dependency is used

| Dependency                      | Used in                                                                 | Method calls                                                            |
| ------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `BaseLoader`                    | `_setup_loaders`, `_load_documents`, `aadd_source`                      | `can_load(source)`, `load(source)`, `reset()`                            |
| `BaseChunker`                   | `_setup_splitters`, `_chunk_documents`, `aadd_source`, `aadd_text`      | `chunk(documents)`                                                      |
| `EmbeddingProvider`             | `_generate_embeddings`, `query_async`                                   | `embed_documents(chunks)`, `embed_query(query)`, `validate_connection`  |
| `BaseVectorDBProvider`          | every step of `setup_async`, query, mutate                              | see table below                                                         |
| `Document` / `Chunk`            | data flow                                                               | constructed via loaders/splitters; chunks carry `chunk_id`, `metadata` |
| `RAGSearchResult` / `VectorSearchResult` | search return types                                              | `_convert_to_rag_results`                                                |
| `Storage` / `KnowledgeRow`      | optional ingestion log                                                  | `upsert_knowledge_content`, `get_knowledge_content`, `delete_knowledge_content` |
| `FunctionTool`                  | `get_tools()`                                                           | `from_callable(fn, name=, description=)`                                 |
| `info_log`, `error_log`, …      | telemetry                                                               | structured prefixed logging                                              |
| `VectorDBConnectionError`, `UpsertError` | error mapping                                                  | raised on connection / upsert failure                                    |

#### Vector DB methods called by `KnowledgeBase`

```text
aconnect / adisconnect                  – connection lifecycle
ais_ready                               – health check
acollection_exists / acreate_collection – collection bootstrap
adelete_collection                      – cleanup on partial failure
aupsert(vectors, payloads, ids,
        chunks, document_ids,
        doc_content_hashes,
        chunk_content_hashes,
        knowledge_base_ids)             – ingestion
adoc_content_hash_exists                – "is this document still the same?"
adocument_id_exists                     – "do we have an older version?"
adelete_by_document_id                  – stale-chunk cleanup
adelete_by_metadata                     – delete-by-filter
asearch(query_vector, query_text,
        top_k, filter, alpha,
        fusion_method,
        similarity_threshold,
        apply_reranking,
        sparse_query_vector)            – retrieval (dense / FTS / hybrid)
aupdate_metadata                        – per-chunk metadata edit
aoptimize                               – optional optimization
get_collection_info                     – optional sync-or-async info getter
disconnect                              – sync fallback used in __del__
```

Any class implementing `BaseVectorDBProvider` with this method set will plug
straight into `KnowledgeBase` without code changes.

---

## 6. Public API

The two re-exported names are everything user code needs to import:

```python
from upsonic.knowledge_base import KnowledgeBase, KBState
```

### 6.1 `KnowledgeBase` — instance methods

| Category    | Method                              | Notes                                                                                                           |
| ----------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Lifecycle   | `setup_async(force=False)`          | Idempotent JIT pipeline (load → filter → chunk → embed → upsert). Safe under concurrency (asyncio.Lock).        |
| Lifecycle   | `setup(force=False)`                | Sync wrapper.                                                                                                   |
| Lifecycle   | `close()` (async)                   | Disconnects vector DB & embedding provider. Idempotent.                                                         |
| Lifecycle   | `__aenter__` / `__aexit__`          | `async with kb:` calls setup on enter, close on exit.                                                           |
| Tool        | `get_tools()` / `aget_tools()`      | Returns `[FunctionTool]` named `search_<name>` for agent registration.                                          |
| Tool        | `build_context()` / `abuild_context`| Returns `<knowledge_base>…</knowledge_base>` system-prompt block.                                               |
| Tool        | `search(query)` (async)             | Convenience: returns formatted result string. Used by agent tool wrapper.                                       |
| Query       | `query_async(query, …)`             | Returns `List[RAGSearchResult]`. Honors `task.vector_search_*`, isolation, alpha/fusion/threshold/reranking.     |
| Mutate      | `aadd_source(source, …)`            | Add and ingest a new file/dir/string post-setup.                                                                |
| Mutate      | `aadd_text(text, …)`                | Add raw text. Returns `document_id`.                                                                            |
| Mutate      | `aremove_document(doc_id)`          | Remove by id; also removes from `self.sources` and `Storage`.                                                   |
| Mutate      | `arefresh()`                        | Reset loaders + `setup_async(force=True)`.                                                                      |
| Mutate      | `adelete_by_filter(meta)`           | Delete every chunk whose payload matches the filter.                                                            |
| Mutate      | `aupdate_document_metadata(id, m)`  | Patch metadata on every chunk of a document.                                                                    |
| Diagnostics | `get_collection_info_async()`       | Provider-specific collection info.                                                                              |
| Diagnostics | `get_config_summary()`              | Sources + class names + stats.                                                                                  |
| Diagnostics | `health_check_async()`              | Per-component health dict (embeddings, splitters, loaders, vectordb).                                           |
| Diagnostics | `optimize_vectordb()`               | Forwards to `vectordb.aoptimize()`.                                                                             |
| Diagnostics | `markdown()`                        | Short markdown summary.                                                                                         |

Every async mutate/diagnostic method has a sync sibling (`add_source`,
`add_text`, `remove_document`, `refresh`, `delete_by_filter`,
`update_document_metadata`).

### 6.2 `KnowledgeBase` — important attributes

| Attribute              | Type                              | Purpose                                                                          |
| ---------------------- | --------------------------------- | -------------------------------------------------------------------------------- |
| `name`                 | `str`                             | Human-readable name; first 16 chars of `knowledge_id` if not given.              |
| `description`          | `str`                             | Free-text description (used by `build_context`).                                 |
| `topics`               | `List[str]`                       | Topic tags (used by `build_context`).                                            |
| `sources`              | `List[Union[str, Path]]`          | Resolved source list. Updated by `aadd_source` / `aremove_document`.             |
| `loaders`              | `List[BaseLoader]`                | Either user-supplied or auto-detected.                                           |
| `splitters`            | `List[BaseChunker]`               | Either user-supplied or auto-detected.                                           |
| `embedding_provider`   | `Optional[EmbeddingProvider]`     | `None` is legal (vectordb-internal embedding mode).                              |
| `vectordb`             | `BaseVectorDBProvider`            | Required.                                                                        |
| `storage`              | `Optional[Storage]`               | If set, every ingested document is logged via `KnowledgeRow`.                    |
| `isolate_search`       | `bool`                            | When True, every upsert/search adds `knowledge_base_id` filter.                  |
| `knowledge_id`         | `str`                             | SHA-256 of `(sources, loader classes, splitter classes, embedding class)`.       |
| `_state`               | `KBState`                         | Lifecycle state — read it via `get_config_summary()["knowledge_base"]["state"]`. |
| `_processing_stats`    | `Dict[str, Any]`                  | Stats from the most recent setup (counts + ISO timestamp).                       |
| `_search_tool_name`    | `str`                             | Pre-computed `search_<sanitized_name>` for `get_tools` and `build_context`.      |

---

## 7. Integration with the rest of Upsonic

### 7.1 As a **tool** on an agent

`KnowledgeBase` quacks like a `ToolProvider`: any object with `get_tools()`
can be registered with a `Direct`/`Agent`. The agent registry picks it up
along with regular tools and MCP tools.

```python
from upsonic import Agent, KnowledgeBase
from upsonic.vectordb.providers.chroma import ChromaProvider, ChromaConfig
from upsonic.embeddings.openai_provider import OpenAIEmbedding

kb = KnowledgeBase(
    name="technical_docs",
    description="Internal engineering wiki",
    topics=["python", "kubernetes"],
    sources=["./docs/", "./README.md"],
    vectordb=ChromaProvider(config=ChromaConfig(path="./chroma_db")),
    embedding_provider=OpenAIEmbedding(),
)

agent = Agent(
    "openai/gpt-4o",
    tools=[kb],            # KnowledgeBase exposes get_tools()
)

await agent.print_do("What does our k8s deployment guide say?")
# Internally the model calls search_technical_docs(query="...") and the KB
# answers with a formatted results string.
```

The agent gets:
- A function tool `search_technical_docs(query: str) -> str`.
- A `<knowledge_base>` block injected into its system prompt by
  `build_context()`.

### 7.2 As **task context**

Tasks accept context items, including `KnowledgeBase` instances. When
`query_async` receives a `task=` argument it copies `vector_search_*`
attributes from it:

```python
from upsonic import Task

t = Task(
    description="Summarize the deployment guide.",
    context=[kb],
    vector_search_top_k=8,
    vector_search_alpha=0.6,
    vector_search_fusion_method="rrf",
    vector_search_similarity_threshold=0.25,
    vector_search_filter={"section": "deployment"},
)
```

The agent layer iterates the `context` list, calls `setup_async()` (lazily),
runs `query_async(query, task=t)`, and inserts the formatted results into the
prompt.

### 7.3 With **Storage** (optional)

If a `Storage` instance is passed, every successfully indexed `Document` is
mirrored as a `KnowledgeRow` (`upsert_knowledge_content`). This survives
process restarts and lets you list / inspect / delete knowledge entries from
outside the live `KnowledgeBase` instance.

`_remove_source_for_document(doc_id)` consults storage first to find the
file path, then removes that path from `self.sources`.

### 7.4 With **vector DB providers**

Any provider implementing `BaseVectorDBProvider` works. The framework already
ships providers (Chroma, Qdrant, pgvector, FAISS, Milvus, Pinecone, Weaviate,
SuperMemory, …); see `src/upsonic/vectordb/providers/`. The
`embedding_provider=None` path exists specifically so providers like
SuperMemory — which embed text on their side — can plug in without forcing
the user to supply a useless embedder.

### 7.5 With **embeddings**

`EmbeddingProvider` only has to implement `embed_documents(List[Chunk]) →
List[List[float]]` and `embed_query(str) → List[float]`. Optional
`validate_connection()` lights up the health-check.

### 7.6 With **loaders & text_splitter factories**

Auto-detection delegates to:

- `loaders.factory.create_intelligent_loaders(sources, **loader_config)`
- `text_splitter.factory.create_intelligent_splitters(sources, use_case=…,
   quality_preference=…, embedding_provider=…, **splitter_config)`
- `text_splitter.factory.create_chunking_strategy("recursive")` — used both
  as a default fallback and as the splitter for `aadd_text`.

If detection fails for any reason it falls back to `RecursiveChunker`.

---

## 8. End-to-end flow of ingestion + query

### 8.1 Ingestion (`setup_async`)

```text
sources                            ┌────────────────────────────────────────────┐
  │                                │              KnowledgeBase                  │
  │  ["docs/", "README.md", "raw   │                                            │
  │   text..."]                    │                                            │
  ▼                                │                                            │
_resolve_sources                   │                                            │
  ├─ direct content?  ──── yes ──▶ _create_document_from_content                │
  └─ Path → exists? → file/dir → walk dir for known extensions                  │
                                                                                │
_validate_component_counts (loaders == file_sources, splitters == sources)      │
                                                                                │
_generate_knowledge_id  ──▶  hash({sources, loaders, splitters, embedding})    │
auto-derive collection_name                                                     │
                                                                                │
─────────────────  await setup_async()  ─────────────────                       │
                                                                                │
[0] _ensure_connection                                                          │
       → vectordb.aconnect()                                                    │
[1] _load_documents                                                             │
       for each source i:                                                       │
         loader = loaders[i] (or shared loaders[0])                             │
         loader.can_load(source) ; loader.load(source) → List[Document]         │
       compute md5 on doc.content if doc_content_hash missing                   │
[1.5] if collection_existed:                                                    │
        _filter_changed_documents:                                              │
          for each doc:                                                         │
            adoc_content_hash_exists?  → skip                                   │
            adocument_id_exists?       → adelete_by_document_id                 │
            else                       → mark as "to process"                   │
[2] _chunk_documents                                                            │
       for each source i:                                                       │
         splitter = splitters[i] (or splitters[0])                              │
         try splitter.chunk(docs)                                               │
         if 0 chunks or exception → fallback to RecursiveChunker                │
[3] _generate_embeddings                                                        │
       if embedding_provider is None: vectors = [[0.0]*vector_size for chunks]  │
       else: vectors = await embedding_provider.embed_documents(chunks)         │
[4] _store_in_vectordb                                                          │
       acollection_exists / acreate_collection                                  │
       aupsert(vectors, payloads, ids, chunks, document_ids,                    │
               doc_content_hashes, chunk_content_hashes,                        │
               knowledge_base_ids = [knowledge_id]*N if isolate_search)         │
[5] storage.upsert_knowledge_content(KnowledgeRow per doc)  (if storage)        │
                                                                                │
state ─▶ INDEXED                                                                │
processing_stats = {sources_count, documents_count, chunks_count,               │
                    vectors_count, indexed_at}                                  │
                                                                                │
on exception: if collection didn't pre-exist → adelete_collection (cleanup)     │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Query (`query_async` / agent tool call)

```text
agent.run(...)
  │
  ├─ agent system prompt injected with kb.build_context()
  │     → "<knowledge_base>You have access to a KB called …</knowledge_base>"
  │
  ├─ model decides to call    search_technical_docs(query="…")
  │     │
  │     ▼
  │   _search_impl(query)
  │     │
  │     └─▶ query_async(query, task=current_task)
  │            │
  │            ├─ await setup_async()   (idempotent — usually a no-op)
  │            │
  │            ├─ if embedding_provider:
  │            │      query_vector = await embedding_provider.embed_query(query)
  │            │  else:
  │            │      query_vector = [0.0] * vector_size
  │            │
  │            ├─ _perform_search:
  │            │     pull top_k / alpha / fusion_method / similarity_threshold /
  │            │       filter from task if not explicitly passed
  │            │     if isolate_search:
  │            │         filter = {**filter, "knowledge_base_id": knowledge_id}
  │            │     vectordb.asearch(query_vector, query_text=query,
  │            │                      top_k, filter, alpha,
  │            │                      fusion_method, similarity_threshold,
  │            │                      apply_reranking, sparse_query_vector)
  │            │
  │            └─ _convert_to_rag_results:
  │                   for each VectorSearchResult:
  │                       text = result.text
  │                            or payload["content"]
  │                            or payload["chunk"]
  │                            or payload["text"]
  │                       RAGSearchResult(text, metadata=payload, score, chunk_id)
  │
  └─ tool returns:
        "Result 1:\n<chunk1>\n\nResult 2:\n<chunk2>\n\n..."
        (or "No relevant information found in the knowledge base.")
```

### 8.3 Re-running on changed sources

A typical second-run scenario, all happening inside `setup_async()`:

| File state                          | What `_filter_changed_documents` does                                             |
| ----------------------------------- | --------------------------------------------------------------------------------- |
| Same path, same content (hash)      | Skip — `adoc_content_hash_exists` is True.                                        |
| Same path, edited content           | `adocument_id_exists` True but hash differs → `adelete_by_document_id`, re-ingest. |
| New path/source                     | Both checks False → ingest fresh.                                                 |
| Path removed entirely               | Not handled here — call `aremove_document(doc_id)` or `adelete_by_filter(...)`.    |

The combination of "skip on identical hash" and "replace on changed hash"
makes `setup_async()` cheap enough to call before *every* `query_async()`
without thinking about it.

### 8.4 Search isolation: how multiple KBs share one collection

When two `KnowledgeBase` instances point at the same collection (say, you
deliberately want one Chroma collection per environment), `isolate_search`
keeps them apart:

```text
Upsert  : payload["knowledge_base_id"] = self.knowledge_id          (per chunk)
Search  : filter += {"knowledge_base_id": self.knowledge_id}         (every query)
```

You can disable this (`isolate_search=False`) when you *want* cross-KB
retrieval, e.g. a global "search everything" surface.

---

## 9. Cheat sheet — putting it all together

```python
import asyncio
from upsonic import Agent, Task, KnowledgeBase
from upsonic.vectordb.providers.chroma import ChromaProvider, ChromaConfig
from upsonic.embeddings.openai_provider import OpenAIEmbedding
from upsonic.storage.providers.json_storage import JSONStorage

async def main():
    kb = KnowledgeBase(
        name="engineering_handbook",
        description="Internal engineering practices and runbooks.",
        topics=["python", "deployment", "incident_response"],
        sources=["./docs/", "./RUNBOOK.md", "Some inline FAQ content..."],
        vectordb=ChromaProvider(config=ChromaConfig(path="./chroma_db")),
        embedding_provider=OpenAIEmbedding(),
        storage=JSONStorage(file_path="./kb_state.json"),
        use_case="rag_retrieval",
        quality_preference="balanced",
        isolate_search=True,
    )

    async with kb:
        # explicit setup — also done lazily inside query_async()
        await kb.setup_async()

        # one-off direct query
        results = await kb.query_async("How do we handle on-call rotations?", top_k=5)
        for r in results:
            print(r.score, r.text[:120], r.metadata.get("source"))

        # use as an agent tool
        agent = Agent("openai/gpt-4o", tools=[kb])
        task = Task(
            "Summarize on-call expectations.",
            context=[kb],
            vector_search_top_k=8,
            vector_search_alpha=0.7,
            vector_search_fusion_method="rrf",
        )
        result = await agent.do_async(task)
        print(result)

        # incremental ingestion
        await kb.aadd_text(
            "Severity definitions: SEV1 = customer impact ...",
            metadata={"section": "incidents"},
        )
        await kb.aadd_source("./new_runbook.pdf")

        # diagnostics
        print(await kb.health_check_async())
        print(kb.get_config_summary())

asyncio.run(main())
```

---

## 10. Key files referenced

- `src/upsonic/knowledge_base/__init__.py` — lazy re-exports of
  `KnowledgeBase`, `KBState`.
- `src/upsonic/knowledge_base/knowledge_base.py` — full implementation
  (≈2,166 lines).
- `src/upsonic/loaders/base.py` (`BaseLoader`) and
  `src/upsonic/loaders/factory.py` (`create_intelligent_loaders`).
- `src/upsonic/text_splitter/base.py` (`BaseChunker`) and
  `src/upsonic/text_splitter/factory.py` (`create_intelligent_splitters`,
  `create_chunking_strategy`).
- `src/upsonic/embeddings/base.py` (`EmbeddingProvider`).
- `src/upsonic/vectordb/base.py` (`BaseVectorDBProvider`).
- `src/upsonic/schemas/data_models.py` (`Document`, `Chunk`,
  `RAGSearchResult`).
- `src/upsonic/schemas/vector_schemas.py` (`VectorSearchResult`).
- `src/upsonic/storage/base.py` (`Storage`) and
  `src/upsonic/storage/schemas.py` (`KnowledgeRow`).
- `src/upsonic/tools/wrappers.py` (`FunctionTool.from_callable`).
- `src/upsonic/utils/printing.py` (`info_log`, `debug_log`, `warning_log`,
  `error_log`, `success_log`).
- `src/upsonic/utils/package/exception.py` (`VectorDBConnectionError`,
  `UpsertError`).
