---
name: shared-schemas
description: Use when working with the central type-vocabulary in `src/upsonic/schemas/` that flows between loaders, splitters, embedders, vector providers, knowledge bases, agents, and memory. Use when a user asks to add, modify, or understand cross-cutting Pydantic models or dataclasses used across RAG, vector-DB, agentic chunking, KB query parameters, or user-profile memory. Trigger when the user mentions Document, Chunk, RAGSearchResult, VectorSearchResult, KBFilterExpr, UserTraits, PropositionList, Topic, TopicAssignmentList, RefinedTopic, schemas package, lazy `__getattr__` re-exports, `document_id`, `doc_content_hash`, `chunk_id`, `chunk_content_hash`, hybrid-search alpha, fusion_method, similarity_threshold, `from_task`, agentic chunker structured outputs, or default profile_schema_model.
---

# `src/upsonic/schemas/` — Shared Pydantic / Dataclass Schemas

## 1. What this folder is — shared Pydantic schemas

`src/upsonic/schemas/` is the **central type-vocabulary** of Upsonic. Every cross-cutting data
shape that needs to flow between subsystems lives here:

- The **knowledge-base / RAG pipeline** uses `Document`, `Chunk` and `RAGSearchResult` to move
  text from loaders → splitters → embedders → vector stores → agents.
- The **vector-database layer** uses `VectorSearchResult` as the *only* return type that
  every provider (Pinecone, Qdrant, Weaviate, Milvus, Chroma, FAISS, pgvector, Supermemory)
  must produce.
- The **knowledge-base query layer** uses `KBFilterExpr` to encapsulate `top_k`, hybrid-search
  `alpha`, fusion strategy, similarity threshold and metadata filter into one validated
  pydantic object passed end-to-end from `Task` through `Agent` to the vector store.
- The **agentic chunker** (`text_splitter/agentic.py`) uses `PropositionList`,
  `Topic`, `TopicAssignmentList` and `RefinedTopic` as the **structured `response_format`**
  for LLM calls in its proposition-extraction → clustering → refinement pipeline.
- The **user-memory layer** (`storage/memory/user/`) uses `UserTraits` as the default
  long-term profile schema when no custom schema is provided.

Two cardinal design rules:

1. **No subsystem should invent its own copy of these models.** Loaders, splitters,
   embedders and vector providers all import from `upsonic.schemas.*`. This is what makes
   the chunking pipeline pluggable and the vector-DB providers swappable.
2. **Every model is provider-agnostic.** None of them depend on a specific LLM, embedder
   or vector backend — they describe *data*, not *behavior*.

The package uses a `__getattr__`-based **lazy loader** in `__init__.py` so that importing
`upsonic.schemas` does not eagerly pull in every Pydantic model. Names are resolved on
first access.

---

## 2. Folder layout (tree)

```
src/upsonic/schemas/
├── __init__.py          # Lazy-loading re-exports of every public schema
├── user.py              # UserTraits — long-term user profile model
├── agentic.py           # Proposition-clustering models for AgenticChunker
├── data_models.py       # Document / Chunk / RAGSearchResult — RAG core
├── vector_schemas.py    # VectorSearchResult dataclass — vector-DB return type
└── kb_filter.py         # KBFilterExpr — typed query parameters for KB lookup
```

There are **no subfolders**.

---

## 3. Top-level files file-by-file (every model with its fields)

### 3.1 `__init__.py` — Lazy public surface

This file deliberately avoids importing any heavy modules at package-import time. It uses
PEP 562 module-level `__getattr__` to materialise schema classes only on first attribute
access:

```python
def __getattr__(name: str) -> Any:
    schema_classes = _get_schema_classes()
    if name in schema_classes:
        return schema_classes[name]
    raise AttributeError(...)
```

`__all__` enumerates the public surface:

| Re-exported name        | Defined in            |
| ----------------------- | --------------------- |
| `UserTraits`            | `user.py`             |
| `PropositionList`       | `agentic.py`          |
| `Topic`                 | `agentic.py`          |
| `TopicAssignmentList`   | `agentic.py`          |
| `RefinedTopic`          | `agentic.py`          |
| `Document`              | `data_models.py`      |
| `Chunk`                 | `data_models.py`      |
| `RAGSearchResult`       | `data_models.py`      |
| `VectorSearchResult`    | `vector_schemas.py`   |
| `KBFilterExpr`          | `kb_filter.py`        |

Static analysis still works because the same names are listed under a `TYPE_CHECKING`
import block.

---

### 3.2 `user.py` — `UserTraits`

A long-running, cross-session user profile inferred by the AI to personalise responses.
Every field is optional so the model can be partially populated and incrementally
updated.

`class UserTraits(BaseModel)`

| Field                          | Type                       | Default | Purpose                                                                                                          |
| ------------------------------ | -------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `detected_expertise`           | `Optional[str]`            | `None`  | Expertise level on the main topic — e.g. `'beginner'`, `'intermediate'`, `'expert'`.                             |
| `detected_tone`                | `Optional[str]`            | `None`  | Preferred tone — e.g. `'formal'`, `'casual'`, `'technical'`.                                                     |
| `inferred_interests`           | `Optional[List[str]]`      | `None`  | Topics or keywords the user seems interested in.                                                                 |
| `current_goal`                 | `Optional[str]`            | `None`  | Concise summary of the user's immediate objective in *this* conversation.                                        |
| `session_sentiment`            | `Optional[str]`            | `None`  | Dominant emotional state — e.g. `'curious'`, `'frustrated'`, `'pleased'`.                                        |
| `communication_style`          | `Optional[List[str]]`      | `None`  | Inferred presentation preferences (concise, bullet points, code examples, ...).                                  |
| `key_entities`                 | `Optional[Dict[str, str]]` | `None`  | Named entities (people, projects, tools) the user mentioned, mapped to a contextual description.                 |
| `long_term_objective_summary`  | `Optional[str]`            | `None`  | Running summary of overarching goals across multiple sessions — meant to be **updated**, not replaced.           |

Used by `storage/memory/user/user.py` as the default `profile_schema_model` when no custom
profile schema is supplied.

---

### 3.3 `agentic.py` — Proposition clustering models

These four models define the structured-output contracts for the three LLM stages in
`text_splitter/agentic.py`:

1. Extract atomic propositions from raw text.
2. Cluster propositions into emergent topics (batch).
3. Refine each topic into a human-readable title + summary.

`class PropositionList(BaseModel)`

| Field          | Type        | Default      | Purpose                                                              |
| -------------- | ----------- | ------------ | -------------------------------------------------------------------- |
| `propositions` | `List[str]` | required     | Simple, self-contained factual statements extracted from source text. |

`class Topic(BaseModel)`

| Field          | Type        | Default  | Purpose                                                            |
| -------------- | ----------- | -------- | ------------------------------------------------------------------ |
| `topic_id`     | `int`       | required | Unique integer ID for this topic.                                  |
| `propositions` | `List[str]` | required | The propositions the LLM has assigned to this topic cluster.       |

`class TopicAssignmentList(BaseModel)`

| Field    | Type          | Default  | Purpose                                                            |
| -------- | ------------- | -------- | ------------------------------------------------------------------ |
| `topics` | `List[Topic]` | required | Full list of topics the LLM has identified during batch clustering.|

`class RefinedTopic(BaseModel)`

| Field      | Type   | Default  | Purpose                                                          |
| ---------- | ------ | -------- | ---------------------------------------------------------------- |
| `title`    | `str`  | required | Concise, human-readable title for the topic cluster.             |
| `summary`  | `str`  | required | One-sentence summary covering the propositions in the cluster.   |

---

### 3.4 `data_models.py` — RAG-core data shapes

This is the **most-imported file** in the package. It defines the three data types that
form the spine of every RAG pipeline:

```
loaders → Document → splitter → Chunk → embedder → vectordb → RAGSearchResult
```

`class Document(BaseModel)`

| Field               | Type             | Default                | Purpose                                                                                                |
| ------------------- | ---------------- | ---------------------- | ------------------------------------------------------------------------------------------------------ |
| `content`           | `str`            | required               | Full raw text extracted from the source.                                                               |
| `metadata`          | `Dict[str, Any]` | `{}` (factory)         | Arbitrary source metadata — e.g. `{"source": "resume1.pdf", "author": "John Doe"}`.                    |
| `document_id`       | `str`            | required               | Unique deterministic identifier (typically MD5 of absolute path or URL).                               |
| `doc_content_hash`  | `str`            | `""`                   | MD5 of full content — used for change detection / deduplication by knowledge-base writers.             |

`class Chunk(BaseModel)`

| Field                | Type             | Default                                  | Purpose                                                                                       |
| -------------------- | ---------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| `text_content`       | `str`            | required                                 | Actual text of this chunk.                                                                    |
| `metadata`           | `Dict[str, Any]` | required                                 | Inherited from parent `Document`, optionally augmented with `page_number`, etc.               |
| `document_id`        | `str`            | required                                 | Parent document ID — back-reference to `Document.document_id`.                                |
| `doc_content_hash`   | `str`            | `""`                                     | Inherited parent doc hash — propagates change detection.                                      |
| `chunk_id`           | `str`            | `default_factory=lambda: str(uuid4())`   | Unique chunk identifier (UUID4).                                                              |
| `chunk_content_hash` | `str`            | `""`                                     | MD5 of `text_content` — used for chunk-level dedup and integrity checks.                      |
| `start_index`        | `Optional[int]`  | `None`                                   | Character offset of this chunk in the parent document (when known).                           |
| `end_index`          | `Optional[int]`  | `None`                                   | Character end-offset of this chunk in the parent document (when known).                       |

`class RAGSearchResult(BaseModel)`

| Field      | Type             | Default        | Purpose                                                         |
| ---------- | ---------------- | -------------- | --------------------------------------------------------------- |
| `text`     | `str`            | required       | Retrieved text content.                                         |
| `metadata` | `Dict[str, Any]` | `{}` (factory) | Source metadata, scores, citations, etc.                        |
| `score`    | `Optional[float]`| `None`         | Similarity score for this result.                               |
| `chunk_id` | `Optional[str]`  | `None`         | Originating `Chunk.chunk_id` — useful for citation / dedup.     |

`Document` and `Chunk` are linked through `document_id` (and `doc_content_hash`).
`RAGSearchResult` is what an `Agent` actually sees after a knowledge-base lookup —
it is the **flattened, agent-friendly** form of `VectorSearchResult`.

---

### 3.5 `vector_schemas.py` — `VectorSearchResult`

A frozen dataclass (not a Pydantic model) because vector providers create thousands of
these per query and immutability + slot-friendliness are useful here.

`@dataclass(frozen=True) class VectorSearchResult`

| Field      | Type                       | Default | Purpose                                                                |
| ---------- | -------------------------- | ------- | ---------------------------------------------------------------------- |
| `id`       | `Union[str, int]`          | required| Provider-native primary key for the matched record.                    |
| `score`    | `float`                    | required| Similarity / relevance score from the vector backend.                  |
| `payload`  | `Optional[Dict[str, Any]]` | `None`  | Provider-stored metadata attached to the vector.                       |
| `vector`   | `Optional[List[float]]`    | `None`  | Raw embedding (only when the caller asked for `with_vectors=True`).    |
| `text`     | `Optional[str]`            | `None`  | Original text content (when the provider stored it alongside vector).  |

Every concrete vector provider in `src/upsonic/vectordb/providers/` (Pinecone, Qdrant,
Weaviate, Milvus, Chroma, FAISS, pgvector, Supermemory) **must** return `List[VectorSearchResult]`
from its `search` / `dense_search` / `hybrid_search` methods, defined on the abstract base
class in `vectordb/base.py`.

---

### 3.6 `kb_filter.py` — `KBFilterExpr`

Typed bag-of-parameters for knowledge-base / vector-DB queries. Designed so that
`Task` can declaratively encode vector-search hyper-parameters and pass them through
`Agent` to the underlying `KnowledgeBase`.

`class KBFilterExpr(BaseModel)`

| Field                  | Type                                  | Default | Constraints      | Purpose                                                                |
| ---------------------- | ------------------------------------- | ------- | ---------------- | ---------------------------------------------------------------------- |
| `top_k`                | `Optional[int]`                       | `None`  | —                | Maximum number of results to return.                                   |
| `alpha`                | `Optional[float]`                     | `None`  | `0.0 ≤ α ≤ 1.0`  | Hybrid-search blend: 0.0 = pure keyword, 1.0 = pure vector.            |
| `fusion_method`        | `Optional[Literal['rrf','weighted']]` | `None`  | —                | Fusion strategy for combining vector and keyword results.              |
| `similarity_threshold` | `Optional[float]`                     | `None`  | `0.0 ≤ t ≤ 1.0`  | Minimum similarity score; results below this are dropped.              |
| `filter`               | `Optional[Dict[str, Any]]`            | `None`  | —                | Metadata filter applied at the vector-DB level.                        |

It also exposes three convenience methods:

| Method                          | Behaviour                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| `to_dict() -> Dict[str, Any]`   | `model_dump()` with all `None` values stripped — safe to splat into provider kwargs.       |
| `from_dict(data) -> KBFilterExpr` | `model_validate(data)` round-trip.                                                       |
| `from_task(task) -> Optional[KBFilterExpr]` | Reads `vector_search_top_k`, `vector_search_alpha`, `vector_search_fusion_method`, `vector_search_similarity_threshold`, `vector_search_filter` attributes off a `Task`. Returns `None` when *all* are unset. |

`from_task` is the bridge that lets users set vector-search options on a `Task` and have
them flow through `Agent.do(task)` into the knowledge-base call site.

---

## 4. Subfolders

There are no subfolders inside `src/upsonic/schemas/`. The package is intentionally flat:
each domain gets exactly one file.

---

## 5. Cross-file relationships

```
                        ┌──────────────┐
                        │   Document   │  data_models.py
                        └──────┬───────┘
                               │ document_id, doc_content_hash, metadata
                               ▼
                        ┌──────────────┐
                        │    Chunk     │  data_models.py
                        └──────┬───────┘
                               │ embedded → stored
                               ▼
                  ┌────────────────────────┐
                  │  VectorSearchResult    │  vector_schemas.py  (provider-side)
                  └────────────┬───────────┘
                               │ flattened
                               ▼
                  ┌────────────────────────┐
                  │    RAGSearchResult     │  data_models.py    (agent-side)
                  └────────────────────────┘

                        ┌──────────────┐
                        │ KBFilterExpr │  kb_filter.py
                        └──────┬───────┘
                               │ from_task(task) →
                  Task (vector_search_*) ────► Agent ────► KnowledgeBase ────► VectorDB

   AgenticChunker stages (text_splitter/agentic.py):
       PropositionList ──► TopicAssignmentList(List[Topic]) ──► RefinedTopic
       (extract)            (cluster)                            (refine)

   UserTraits (user.py) ──► storage/memory/user/user.py (default profile schema)
```

Key invariants to remember:

- `Chunk.document_id` always equals the `Document.document_id` it was produced from.
- `Chunk.doc_content_hash` is **propagated** from the parent — splitters never recompute it.
- `chunk_content_hash` is a hash of `text_content` only, not metadata.
- `RAGSearchResult.chunk_id` is the same UUID as the originating `Chunk.chunk_id`,
  enabling citation and back-reference into the knowledge base.

---

## 6. Public API

Importing from the package root resolves through the lazy `__getattr__`:

```python
from upsonic.schemas import (
    UserTraits,
    PropositionList,
    Topic,
    TopicAssignmentList,
    RefinedTopic,
    Document,
    Chunk,
    RAGSearchResult,
    VectorSearchResult,
    KBFilterExpr,
)
```

Or directly from sub-modules (recommended in framework code, avoids the lazy hop):

```python
from upsonic.schemas.data_models   import Document, Chunk, RAGSearchResult
from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.schemas.kb_filter      import KBFilterExpr
from upsonic.schemas.agentic        import PropositionList, Topic, TopicAssignmentList, RefinedTopic
from upsonic.schemas.user           import UserTraits
```

| Symbol                | Kind            | One-line role                                                              |
| --------------------- | --------------- | -------------------------------------------------------------------------- |
| `UserTraits`          | Pydantic model  | Long-term user profile.                                                    |
| `PropositionList`     | Pydantic model  | LLM `response_format` for the proposition-extraction stage.                |
| `Topic`               | Pydantic model  | Single topic cluster from the agentic chunker.                             |
| `TopicAssignmentList` | Pydantic model  | LLM `response_format` for batch clustering.                                |
| `RefinedTopic`        | Pydantic model  | Final per-chunk title/summary metadata.                                    |
| `Document`            | Pydantic model  | Pre-chunking source unit.                                                  |
| `Chunk`               | Pydantic model  | Atomic embeddable unit of a `Document`.                                    |
| `RAGSearchResult`     | Pydantic model  | Agent-facing search hit with text + metadata + score.                      |
| `VectorSearchResult`  | Frozen dataclass| Provider-facing search hit (id/score/payload/vector/text).                 |
| `KBFilterExpr`        | Pydantic model  | Typed knowledge-base query parameter bundle.                               |

---

## 7. Integration with rest of Upsonic (which modules consume these schemas)

A non-exhaustive consumer map (verified against the source tree):

### 7.1 `Document` — used by every loader
```
src/upsonic/loaders/text.py
src/upsonic/loaders/markdown.py
src/upsonic/loaders/html.py
src/upsonic/loaders/json.py
src/upsonic/loaders/yaml.py
src/upsonic/loaders/xml.py
src/upsonic/loaders/csv.py
src/upsonic/loaders/pdf.py
src/upsonic/loaders/pdfplumber.py
src/upsonic/loaders/pymupdf.py
src/upsonic/loaders/docx.py
src/upsonic/loaders/docling.py
src/upsonic/loaders/base.py
src/upsonic/loaders/factory.py
src/upsonic/text_splitter/factory.py
```
Each loader's `load()` / `aload()` returns `List[Document]`.

### 7.2 `Document` + `Chunk` — used by every text splitter
```
src/upsonic/text_splitter/base.py
src/upsonic/text_splitter/character.py
src/upsonic/text_splitter/recursive.py
src/upsonic/text_splitter/semantic.py
src/upsonic/text_splitter/markdown.py
src/upsonic/text_splitter/html_chunker.py
src/upsonic/text_splitter/json_chunker.py
src/upsonic/text_splitter/python.py
src/upsonic/text_splitter/agentic.py
```

### 7.3 `PropositionList` / `Topic` / `TopicAssignmentList` / `RefinedTopic`
```
src/upsonic/text_splitter/agentic.py    # uses all four as Task.response_format
```

### 7.4 `Chunk` — used by embedders
```
src/upsonic/embeddings/base.py
```

### 7.5 `VectorSearchResult` — used by the entire vector-DB layer
```
src/upsonic/vectordb/base.py
src/upsonic/vectordb/store.py
src/upsonic/vectordb/providers/pinecone.py
src/upsonic/vectordb/providers/qdrant.py
src/upsonic/vectordb/providers/weaviate.py
src/upsonic/vectordb/providers/milvus.py
src/upsonic/vectordb/providers/chroma.py
src/upsonic/vectordb/providers/faiss.py
src/upsonic/vectordb/providers/pgvector.py
src/upsonic/vectordb/providers/supermemory.py
```

### 7.6 `Document` / `RAGSearchResult` / `VectorSearchResult` — knowledge base
```
src/upsonic/knowledge_base/knowledge_base.py
```

### 7.7 `RAGSearchResult` — agent context manager
```
src/upsonic/agent/context_managers/context_manager.py
```

### 7.8 `KBFilterExpr` — agent + run output
```
src/upsonic/agent/agent.py            # builds KBFilterExpr from a Task before KB calls
src/upsonic/run/agent/output.py       # consumes KBFilterExpr in run-loop output handling
```

### 7.9 `UserTraits` — memory storage
```
src/upsonic/storage/memory/user/user.py   # default profile schema model
```

---

## 8. End-to-end usage example

The following snippet shows every schema in this folder participating in one realistic
knowledge-base lifecycle.

```python
from __future__ import annotations
import hashlib
from upsonic.schemas.data_models    import Document, Chunk, RAGSearchResult
from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.schemas.kb_filter      import KBFilterExpr
from upsonic.schemas.agentic        import (
    PropositionList, Topic, TopicAssignmentList, RefinedTopic,
)
from upsonic.schemas.user           import UserTraits


# ───────────────────────────────────────────────────────────────
# 1. Loader stage — produce a Document from raw text
# ───────────────────────────────────────────────────────────────
raw_text = "Upsonic is a reliability-focused AI agent framework. It supports MCP."
doc_id   = hashlib.md5(b"/tmp/about.txt").hexdigest()
doc_hash = hashlib.md5(raw_text.encode()).hexdigest()

document = Document(
    content=raw_text,
    metadata={"source": "/tmp/about.txt", "author": "docs"},
    document_id=doc_id,
    doc_content_hash=doc_hash,
)

# ───────────────────────────────────────────────────────────────
# 2. Splitter stage — emit one or more Chunks
# ───────────────────────────────────────────────────────────────
chunk = Chunk(
    text_content=document.content,
    metadata={**document.metadata, "page_number": 1},
    document_id=document.document_id,
    doc_content_hash=document.doc_content_hash,
    chunk_content_hash=hashlib.md5(document.content.encode()).hexdigest(),
    start_index=0,
    end_index=len(document.content),
)

# ───────────────────────────────────────────────────────────────
# 3. Agentic chunker — structured LLM outputs
#    (these are normally produced by Task(response_format=...))
# ───────────────────────────────────────────────────────────────
props = PropositionList(propositions=[
    "Upsonic is an AI agent framework.",
    "Upsonic emphasizes reliability.",
    "Upsonic supports MCP.",
])

clusters = TopicAssignmentList(topics=[
    Topic(topic_id=1, propositions=[
        "Upsonic is an AI agent framework.",
        "Upsonic emphasizes reliability.",
    ]),
    Topic(topic_id=2, propositions=["Upsonic supports MCP."]),
])

refined = RefinedTopic(
    title="Upsonic Framework Overview",
    summary="Upsonic is a reliability-focused AI agent framework that supports MCP.",
)

# ───────────────────────────────────────────────────────────────
# 4. Vector-DB stage — provider returns VectorSearchResult
# ───────────────────────────────────────────────────────────────
provider_hit = VectorSearchResult(
    id=chunk.chunk_id,
    score=0.91,
    payload=chunk.metadata,
    text=chunk.text_content,
)

# ───────────────────────────────────────────────────────────────
# 5. KnowledgeBase stage — flatten provider hits into RAGSearchResult
# ───────────────────────────────────────────────────────────────
kb_hit = RAGSearchResult(
    text=provider_hit.text or "",
    metadata=provider_hit.payload or {},
    score=provider_hit.score,
    chunk_id=str(provider_hit.id),
)

# ───────────────────────────────────────────────────────────────
# 6. Task → Agent: build a typed query filter
# ───────────────────────────────────────────────────────────────
class FakeTask:
    vector_search_top_k = 5
    vector_search_alpha = 0.7
    vector_search_fusion_method = "rrf"
    vector_search_similarity_threshold = 0.4
    vector_search_filter = {"source": "/tmp/about.txt"}

filter_expr = KBFilterExpr.from_task(FakeTask())
assert filter_expr is not None
assert filter_expr.to_dict() == {
    "top_k": 5,
    "alpha": 0.7,
    "fusion_method": "rrf",
    "similarity_threshold": 0.4,
    "filter": {"source": "/tmp/about.txt"},
}

# Round-trip:
restored = KBFilterExpr.from_dict(filter_expr.to_dict())
assert restored == filter_expr

# ───────────────────────────────────────────────────────────────
# 7. User-memory: long-term profile
# ───────────────────────────────────────────────────────────────
profile = UserTraits(
    detected_expertise="intermediate",
    detected_tone="technical",
    inferred_interests=["agent frameworks", "RAG", "reliability"],
    current_goal="evaluate Upsonic for production use",
    session_sentiment="curious",
    communication_style=["prefers concise answers", "requests code examples"],
    key_entities={"Upsonic": "framework under evaluation"},
    long_term_objective_summary="Building a production AI agent stack.",
)

print(document.model_dump())
print(chunk.model_dump())
print(kb_hit.model_dump())
print(filter_expr.to_dict())
print(profile.model_dump())
```

What this example demonstrates end-to-end:

- A loader produces a `Document` with a stable `document_id` and content hash.
- A splitter produces a `Chunk` that **inherits** `document_id` and `doc_content_hash` from
  its parent, adds a fresh `chunk_id` UUID, and records its own `chunk_content_hash`.
- The agentic chunker is fed `PropositionList` / `TopicAssignmentList` / `RefinedTopic` as
  structured outputs to be returned by the LLM stages.
- A vector provider returns a `VectorSearchResult` (with `id == chunk_id`).
- The knowledge base flattens that into a `RAGSearchResult` that the agent will see.
- A `Task` carrying `vector_search_*` attributes is converted into a `KBFilterExpr` which
  is round-trippable and serialises cleanly via `to_dict()` (dropping `None`s).
- The same package also hosts `UserTraits`, the long-term profile model used by the
  memory layer.

That single example exercises every public symbol in `upsonic.schemas`.
