---
name: vectordb-providers
description: Use when working with the vector database abstraction layer in `src/upsonic/vectordb/` — implementing or configuring vector stores for RAG, hybrid/dense/full-text search, sparse vectors, or multi-provider support. Use when a user asks to add, configure, or debug Chroma, FAISS, Qdrant, Pinecone, Milvus, Weaviate, PgVector, or SuperMemory backends, build a `VectorStore`, write a `BaseVectorDBProvider` subclass, tune HNSW/IVF_FLAT/FLAT indexes, set up sparse vectors with BM25, perform RRF or weighted fusion, or wire a vector DB into `KnowledgeBase`. Trigger when the user mentions vectordb, vector database, BaseVectorDBProvider, VectorStore, ChromaProvider, FaissProvider, QdrantProvider, PineconeProvider, MilvusProvider, WeaviateProvider, PgVectorProvider, SuperMemoryProvider, ChromaConfig, QdrantConfig, BaseVectorDBConfig, ConnectionConfig, HNSWIndexConfig, IVFIndexConfig, FlatIndexConfig, DistanceMetric, IndexType, Mode, create_config, aupsert, asearch, adense_search, afull_text_search, ahybrid_search, sparse vectors, hybrid search, RRF, reciprocal rank fusion, weighted fusion, BM25, pgvector, Milvus Lite, Qdrant cloud, Pinecone serverless, Weaviate multi-tenancy, chunk_content_hash, doc_content_hash, knowledge_base_id, or two-layer dedup.
---

# `src/upsonic/vectordb/` — Vector Database Providers for RAG

## 1. What this folder is

`src/upsonic/vectordb/` is the **vector database abstraction layer** of Upsonic. It exposes one
async-first contract — `BaseVectorDBProvider` — implemented by eight concrete providers (Chroma,
FAISS, Qdrant, Pinecone, Milvus, Weaviate, PgVector, SuperMemory) and a small `VectorStore`
convenience wrapper for standalone use without `KnowledgeBase`.

The folder solves four distinct problems:

1. **Common interface.** Every provider exposes the same `aconnect / acreate_collection / aupsert /
   asearch / adelete / aupdate_metadata` surface, with both `async` (`a*`) and synchronous
   (`*`) versions. The base class auto-generates sync wrappers via a `_run_async_from_sync` helper
   that handles both "no running loop" and "already-in-a-loop" cases.
2. **Provider-specific configuration.** A single `BaseVectorDBConfig` (Pydantic, frozen,
   `extra='forbid'`) defines the cross-provider knobs (`collection_name`, `vector_size`,
   `distance_metric`, search-mode toggles). Each provider has a Pydantic subclass
   (`ChromaConfig`, `QdrantConfig`, ...) carrying its own connection / index / sparse-vector knobs,
   plus a `create_config(provider="qdrant", ...)` factory that resolves the string name.
3. **Standardised payload contract.** Every provider stores the same seven "standard fields" per
   record: `chunk_id`, `chunk_content_hash`, `document_id`, `doc_content_hash`, `document_name`,
   `content`, `knowledge_base_id`, plus a free-form `metadata` JSON blob. Existence checks
   (`achunk_id_exists`, `adoc_content_hash_exists`, ...) and field-scoped delete helpers
   (`adelete_by_chunk_id`, `adelete_by_document_id`, ...) are derived from this contract — this is
   what powers `KnowledgeBase`'s two-layer dedup architecture (document-level + chunk-level).
4. **Search-mode dispatch.** `asearch(...)` is a master method that routes to `adense_search`,
   `afull_text_search`, or `ahybrid_search` based on which of `query_vector` / `query_text` /
   `sparse_query_vector` are supplied — and each call respects the provider's
   `dense_search_enabled / full_text_search_enabled / hybrid_search_enabled` flags.

Every provider gates its third-party SDK behind a `try / except ImportError` block and raises a
nicely formatted `import_error(...)` if the user tries to instantiate the provider without the
optional extra installed (e.g. `pip install "upsonic[qdrant]"`).

## 2. Folder layout

```text
src/upsonic/vectordb/
├── __init__.py                # Lazy-import public API surface
├── base.py                    # BaseVectorDBProvider — async-first ABC
├── config.py                  # All Pydantic config classes + create_config()
├── factory.py                 # Empty file (legacy; logic now in __init__/create_config)
├── store.py                   # VectorStore high-level wrapper (embed + add/search/delete)
└── providers/
    ├── chroma.py              # ChromaProvider — embedded/local/cloud/in-memory
    ├── faiss.py               # FaissProvider — single-process file-backed
    ├── qdrant.py              # QdrantProvider — async client, sparse + dense
    ├── pinecone.py            # PineconeProvider — cloud-only, hybrid-in-one-index
    ├── milvus.py              # MilvusProvider — schema-based, hybrid via AnnSearchRequest
    ├── weaviate.py            # WeaviateProvider — schema + multi-tenancy + BM25
    ├── pgvector.py            # PgVectorProvider — Postgres + pgvector + SQLAlchemy
    └── supermemory.py         # SuperMemoryProvider — managed-memory text API wrapper
```

Sizes (lines of code):

| File                      | LOC   |
|---------------------------|-------|
| `base.py`                 | 442   |
| `config.py`               | 604   |
| `store.py`                | 326   |
| `providers/chroma.py`     | 1736  |
| `providers/faiss.py`      | 1203  |
| `providers/qdrant.py`     | 1902  |
| `providers/pinecone.py`   | 1520  |
| `providers/milvus.py`     | 1271  |
| `providers/weaviate.py`   | 2093  |
| `providers/pgvector.py`   | 1754  |
| `providers/supermemory.py`| 960   |
| `factory.py`              | 0     |
| `__init__.py`              | 190   |

## 3. Top-level files

### 3.1 `base.py` — `BaseVectorDBProvider` (the contract)

`BaseVectorDBProvider` is an `abc.ABC` whose abstract methods are **all `async`** (named with the
`a` prefix). The class also provides synchronous wrappers automatically — provider authors only
need to implement the `a*` methods.

#### Sync ↔ async bridge

```python
def _run_async_from_sync(self, awaitable: Awaitable[T]) -> T:
    try:
        asyncio.get_running_loop()
        # Already inside an event loop — run on a worker thread with its own loop
        with ThreadPoolExecutor(max_workers=1) as executor:
            def run_in_thread() -> T:
                thread_loop = self._get_sync_loop()
                return thread_loop.run_until_complete(awaitable)
            return executor.submit(run_in_thread).result()
    except RuntimeError:
        # No running loop — reuse the persistent instance loop
        loop = self._get_sync_loop()
        return loop.run_until_complete(awaitable)
```

This is what allows e.g. `provider.search(...)` to be called from sync code that may or may not
already be inside an async context (e.g. Jupyter, FastAPI handlers, plain scripts).

#### Public surface

| Lifecycle               | Collection             | Data ops                            | Search                                                  |
|-------------------------|------------------------|-------------------------------------|---------------------------------------------------------|
| `aconnect`              | `acreate_collection`   | `aupsert`                           | `asearch`  (dispatcher)                                  |
| `adisconnect`           | `adelete_collection`   | `adelete` / `adelete_by_id`         | `adense_search`                                          |
| `ais_ready`             | `acollection_exists`   | `afetch` / `afetch_by_id`           | `afull_text_search`                                      |
|                         |                        | `aupdate_metadata`                  | `ahybrid_search`                                         |
|                         |                        | `aoptimize`                         | `aget_supported_search_types`                            |

| Existence checks (one per standard field)              | Delete-by-field helpers                                |
|--------------------------------------------------------|--------------------------------------------------------|
| `adocument_id_exists(document_id)`                     | `adelete_by_document_id(document_id)`                  |
| `adocument_name_exists(document_name)`                 | `adelete_by_document_name(document_name)`              |
| `achunk_id_exists(chunk_id)`                           | `adelete_by_chunk_id(chunk_id)`                        |
| `adoc_content_hash_exists(doc_content_hash)`           | `adelete_by_doc_content_hash(doc_content_hash)`        |
| `achunk_content_hash_exists(chunk_content_hash)`       | `adelete_by_chunk_content_hash(chunk_content_hash)`    |
|                                                        | `adelete_by_metadata(metadata: Dict)`                  |

#### `aupsert` signature (contract)

Every provider implements:

```python
async def aupsert(
    self,
    vectors: Optional[List[List[float]]] = None,
    payloads: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[Union[str, int]]] = None,
    chunks: Optional[List[str]] = None,
    document_ids: Optional[List[str]] = None,
    document_names: Optional[List[str]] = None,
    doc_content_hashes: Optional[List[str]] = None,
    chunk_content_hashes: Optional[List[str]] = None,
    sparse_vectors: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    knowledge_base_ids: Optional[List[str]] = None,
) -> None:
```

All per-item arrays MUST be the same length; providers validate this and raise `UpsertError` /
`ValueError` on mismatch. Standard fields are taken **only** from these dedicated parameters —
duplicating them inside the `payloads` dicts triggers a one-time `debug_log` warning and the
duplicates are ignored. Non-standard keys in `payloads[i]` are folded into the JSON `metadata`
field of the stored record.

#### Context-manager support

```python
async with QdrantProvider(config=qdrant_config) as provider:
    await provider.acreate_collection()
    await provider.aupsert(...)
# adisconnect() called automatically
```

`__enter__` / `__exit__` exist too, dispatching through `_run_async_from_sync`.

### 3.2 `config.py` — Pydantic config tree

All config classes are `pydantic.BaseModel` with `model_config = ConfigDict(frozen=True,
extra='forbid')`. Frozenness means after construction you cannot mutate the config; this is what
makes configs safe to share across coroutines.

#### Enums

```python
class Mode(str, Enum):
    CLOUD = 'cloud'
    LOCAL = 'local'
    EMBEDDED = 'embedded'
    IN_MEMORY = 'in_memory'

class DistanceMetric(str, Enum):
    COSINE = 'Cosine'
    EUCLIDEAN = 'Euclidean'
    DOT_PRODUCT = 'DotProduct'

class IndexType(str, Enum):
    HNSW = 'HNSW'
    IVF_FLAT = 'IVF_FLAT'
    FLAT = 'FLAT'
```

#### Index configs (discriminated union via `type` literal)

```python
class HNSWIndexConfig(BaseModel):
    type: Literal[IndexType.HNSW] = IndexType.HNSW
    m: int = 16
    ef_construction: int = 200
    ef_search: Optional[int] = None

class IVFIndexConfig(BaseModel):
    type: Literal[IndexType.IVF_FLAT] = IndexType.IVF_FLAT
    nlist: int = 100
    nprobe: Optional[int] = None

class FlatIndexConfig(BaseModel):
    type: Literal[IndexType.FLAT] = IndexType.FLAT

IndexConfig = Union[HNSWIndexConfig, IVFIndexConfig, FlatIndexConfig]
```

Each provider config restricts which index types it accepts: Chroma rejects IVF, Qdrant rejects
IVF, Weaviate rejects IVF, FAISS / Milvus / PgVector accept all three.

#### `BaseVectorDBConfig` (cross-provider essentials)

```python
collection_name: str = "default_collection"
vector_size: int                                 # required
distance_metric: DistanceMetric = COSINE
recreate_if_exists: bool = False
default_top_k: int = 10
default_similarity_threshold: Optional[float] = None  # validated 0.0-1.0
dense_search_enabled: bool = True
full_text_search_enabled: bool = True
hybrid_search_enabled: bool = True
default_hybrid_alpha: float = 0.5
default_fusion_method: Literal['rrf', 'weighted'] = 'weighted'
provider_name / provider_description / provider_id: Optional[str] = None
default_metadata: Optional[Dict[str, Any]] = None
indexed_fields: Optional[List[Union[str, Dict[str, Any]]]] = None
```

`indexed_fields` accepts two formats:

```python
# Simple — type defaults to "keyword"
indexed_fields=["document_name", "document_id"]

# Advanced — explicit per-field type
indexed_fields=[
    {"field": "document_name", "type": "keyword"},
    {"field": "age",           "type": "integer"},
]
```

#### `ConnectionConfig` (cloud / local / embedded / in-memory)

```python
mode: Mode  # required
host / port / api_key / use_tls / grpc_port / prefer_grpc / https / prefix /
timeout / url / location / db_path
```

The `model_validator` enforces:
- `CLOUD` ⇒ `api_key` or full `url`
- `LOCAL` ⇒ `host`+`port` or full `url`
- `EMBEDDED` ⇒ `db_path`

#### `create_config(provider, **kwargs)` factory

```python
config = create_config("qdrant",
    collection_name="my_kb",
    vector_size=1536,
    connection=ConnectionConfig(mode=Mode.CLOUD,
                                api_key=SecretStr("..."),
                                url="https://cluster.qdrant.io"),
    use_sparse_vectors=True,
)
```

Provider name resolution:

| `provider` (string)  | Class returned       |
|----------------------|----------------------|
| `"chroma"`           | `ChromaConfig`       |
| `"faiss"`            | `FaissConfig`        |
| `"qdrant"`           | `QdrantConfig`       |
| `"pinecone"`         | `PineconeConfig`     |
| `"milvus"`           | `MilvusConfig`       |
| `"weaviate"`         | `WeaviateConfig`     |
| `"pgvector"`         | `PgVectorConfig`     |
| `"supermemory"`      | `SuperMemoryConfig`  |

### 3.3 `__init__.py` — Lazy-import public surface

The module relies on `__getattr__(name)` (PEP 562) to lazily import provider classes and configs
only when first accessed. This keeps `import upsonic` cheap even if the user has many vectordb
extras installed:

```python
_PROVIDER_MAP = {
    'ChromaProvider':    '.providers.chroma',
    'FaissProvider':     '.providers.faiss',
    'PineconeProvider':  '.providers.pinecone',
    'QdrantProvider':    '.providers.qdrant',
    'MilvusProvider':    '.providers.milvus',
    'WeaviateProvider':  '.providers.weaviate',
    'PgVectorProvider':  '.providers.pgvector',
    'SuperMemoryProvider': '.providers.supermemory',
}
```

Cached after first lookup in `_provider_cache: dict[str, Any]`.

### 3.4 `factory.py`

Currently empty (0 lines). Provider/config resolution lives in `config.create_config()` and the
lazy `__init__.py` import surface. The file is kept as a placeholder for future factories.

### 3.5 `store.py` — `VectorStore` high-level wrapper

`VectorStore` is a thin convenience layer wrapping a `BaseVectorDBProvider` plus an
`EmbeddingProvider`. It is **not** the standard `KnowledgeBase` path; it is for users who want a
quick LangChain-style `add(text) / search(query)` API without document loaders, splitters, or
source management.

```python
from upsonic.vectordb import VectorStore, QdrantProvider
from upsonic.embeddings import OpenAIEmbedding

store = VectorStore(
    vectordb=QdrantProvider(config=qdrant_config),
    embedding_provider=OpenAIEmbedding(),
)
async with store:
    doc_id = await store.aadd("The quick brown fox...")
    results = await store.asearch("brown fox")
    await store.adelete(doc_id)
```

Public methods: `aadd / aadd_many / asearch / adelete / adelete_by_filter` (plus sync wrappers).

Internal logic:

- **Connection caching**: `_is_connected: bool` plus `_ensure_connected` / `_ensure_collection`
  hooks; collection auto-created on first add.
- **Deduplication**: `chunk_content_hash = md5(text)` is computed and used as both the chunk_id
  AND looked up via `achunk_content_hash_exists` before re-embedding. Adds are skipped if the
  hash is already stored.
- **No-embedder fallback**: if `embedding_provider is None`, zero-vectors are produced sized to
  `vectordb._config.vector_size`. This is important for providers like SuperMemory that handle
  embedding internally.

## 4. Per-provider files

### 4.1 At-a-glance provider matrix

| Provider     | Driver SDK            | Pip extra                       | Modes supported                         | Indexes                  | Sparse vectors          | Hybrid search                    |
|--------------|-----------------------|---------------------------------|-----------------------------------------|--------------------------|-------------------------|----------------------------------|
| Chroma       | `chromadb`            | `upsonic[chroma]`               | IN_MEMORY, EMBEDDED, LOCAL, CLOUD       | HNSW, FLAT               | No (BM25-ish in Python) | dense + Python BM25 fusion       |
| FAISS        | `faiss-cpu`, `numpy`  | `upsonic[faiss]`                | IN_MEMORY, EMBEDDED (file)              | HNSW, IVF_FLAT, FLAT     | No                       | No (degrades to dense)           |
| Qdrant       | `qdrant-client`       | `upsonic[qdrant]`               | IN_MEMORY, EMBEDDED, LOCAL, CLOUD       | HNSW, FLAT               | Yes (`use_sparse_vectors`) | native (RRF/DBSF) or manual fusion |
| Pinecone     | `pinecone`, `pinecone-text` | `upsonic[pinecone]`        | CLOUD only (Serverless or Pod)          | provider-managed         | Yes (`use_sparse_vectors`) | single-index alpha-weighted     |
| Milvus       | `pymilvus` (AsyncMilvusClient) | `upsonic[milvus]`      | EMBEDDED (Lite), LOCAL, CLOUD           | HNSW, IVF_FLAT, FLAT     | Yes                      | `AnnSearchRequest` + RRFRanker / WeightedRanker |
| Weaviate     | `weaviate-client` v4  | `upsonic[weaviate]`             | IN_MEMORY, EMBEDDED, LOCAL, CLOUD       | HNSW, FLAT               | No (uses BM25)           | native `query.hybrid()` (RANKED / RELATIVE_SCORE) |
| PgVector     | `sqlalchemy`, `psycopg`, `pgvector` | `upsonic[pgvector]` | LOCAL (any reachable Postgres)          | HNSW, IVF_FLAT           | No (uses tsvector GIN)   | manual: `<=>` + `tsvector` ranking, RRF fusion |
| SuperMemory  | `supermemory`         | `upsonic[supermemory]`          | CLOUD only                              | provider-managed         | No (managed)             | API search_mode='hybrid'         |

### 4.2 `chroma.py` — `ChromaProvider`

ChromaDB is the most accessible provider — it works in-memory by default, persists to disk in
EMBEDDED mode, and supports HTTP/Cloud clients. The provider:

- Builds a `chromadb.Client()` (in-memory), `PersistentClient(path=...)` (embedded),
  `HttpClient(host, port)` (local), or `CloudClient(api_key, tenant?, database?)` (cloud).
- Maps `DistanceMetric` to Chroma's `hnsw:space` metadata: `cosine`, `l2`, `ip`.
- Stores **all standard fields as flat top-level metadata keys** (Chroma indexes all metadata
  automatically) and the user metadata blob as a `metadata: <json>` string. `_split_filter()`
  separates native standard-field filters (handed to Chroma's `where=`) from
  user-metadata filters (applied in Python after the candidate set is fetched).
- Full-text search is **not native** — `afull_text_search` uses `where_document={"$contains": q}`
  and then runs a Python BM25-like ranker (term density × match-ratio).
- Hybrid search runs `adense_search` + `afull_text_search` in parallel via `asyncio.gather` and
  fuses with `_reciprocal_rank_fusion` or `_weighted_fusion`.
- Connection is auto-recovered: every `aget_client()` call sends a `heartbeat` and recreates the
  client if it raises.

### 4.3 `faiss.py` — `FaissProvider`

FAISS is a pure Python/C++ similarity library — no server. The provider is **single-process,
file-backed** and explicitly **not thread-safe**.

- Persists four files into `db_path/`: `index.faiss`, `metadata.json`, `id_map.json`,
  `field_indexes.json`. `aconnect()` hydrates them; `adisconnect()` writes them back.
- Builds the FAISS index via `faiss.index_factory(dim, factory_string, metric)` — the factory
  string is composed from `IndexConfig` plus optional quantization (`PQ{m}` for product,
  `SQ{bits}` for scalar).
- For COSINE distance, vectors are L2-normalised and the index uses `METRIC_INNER_PRODUCT`
  (`normalize_vectors` config field forces this; if False with COSINE, validation rejects the
  config).
- Maintains in-Python field indexes (`_field_indexes: Dict[str, Dict[Any, Set[str]]]`) for fast
  metadata lookups when `indexed_fields` is set. `_get_candidate_ids_from_filter` first prunes by
  these indexes, then post-filters with `_build_filter_function` lambdas.
- `afull_text_search` raises `NotImplementedError`; `ahybrid_search` warns then degrades to
  `adense_search`.

### 4.4 `qdrant.py` — `QdrantProvider`

The most feature-complete async provider. Uses `qdrant_client.AsyncQdrantClient`.

- Connection modes: `:memory:`, embedded (`path=...`), local (`host`+`port`+`grpc_port`), cloud
  (`url`+`api_key`). gRPC is auto-enabled when `prefer_grpc=True`.
- IDs are normalised to UUID strings or `int` (Qdrant only accepts those); arbitrary string IDs
  get hashed to a stable int via MD5.
- Field indexing: `payload_field_configs` (advanced — `PayloadFieldConfig` with explicit
  `field_type` of `text|keyword|integer|float|boolean|geo` plus custom `params`) or simpler
  `indexed_fields: ["document_name", ...]` — auto-mapped to `KeywordIndexParams` /
  `TextIndexParams` etc.
- Sparse-vector support is opt-in via `use_sparse_vectors=True`. When enabled, the collection is
  created with **named vectors** — `dense_vector_name` (`"dense"`) and `sparse_vector_name`
  (`"sparse"`) — both populated per point.
- Hybrid search has two paths:
  - **Native Qdrant** (`_native_hybrid_search`): if both sparse vectors are configured AND a
    `sparse_query_vector` is supplied, uses `query_points(prefetch=[Prefetch(sparse), Prefetch(dense)],
    query=FusionQuery(fusion=RRF|DBSF))`.
  - **Manual fusion**: otherwise, runs `adense_search` and `afull_text_search` separately and
    fuses via `_fuse_weighted` (alpha) or `_fuse_rrf` (`k=60`).
- Full-text search uses Qdrant's text index (`MatchText`) on the `text_search_field` (default
  `content`). For `IN_MEMORY` mode the text index isn't supported, so it falls back to a
  client-side scroll + Python BM25 score.
- Filter translation: `_build_qdrant_filter` maps a Mongo-style dict (`{"$gte"/"$lte"/"$gt"/
  "$lt"/"$in"/"$eq"}`) to `models.Filter(must=[FieldCondition(key, range/match)])`.
- Optional `reranker` parameter at construction time — applied at the end of search if it has a
  `rerank(query, documents)` method.

### 4.5 `pinecone.py` — `PineconeProvider`

Cloud-only. Uses the official `pinecone` Python SDK and `pinecone-text` for BM25 sparse encoding.

- Connection: `Pinecone(api_key=..., host?, additional_headers?, pool_threads?)`. `aconnect()`
  attaches to an existing index (sanitising the collection name to lowercase + dashes).
- Spec building: `_build_spec()` accepts `ServerlessSpec`/`PodSpec` instances, raw dicts, or
  derives a `ServerlessSpec(cloud=..., region=...)` from `environment` strings like
  `"aws-us-east-1"`.
- **Single-index hybrid**: dense and sparse vectors share one index. When
  `hybrid_search_enabled=True`, the metric is **forced** to `dotproduct` (Pinecone requirement)
  even if the user passed `cosine`.
- Sparse encoder: a `BM25Encoder().default()` is initialised per provider when sparse mode is
  active. `aupsert` automatically generates sparse values from the chunk text if the user did not
  supply `sparse_vectors`.
- Hybrid weighting: `_hybrid_scale(dense, sparse, alpha)` multiplies dense values by `alpha` and
  sparse values by `1-alpha` before issuing a single `index.query(vector=hdense,
  sparse_vector=hsparse)`.
- Score normalisation per metric: `dotproduct`/`cosine` clipped to [0,1]; `euclidean` mapped via
  `1 / (1 + d)`. Implemented in `_normalize_score`.
- Async-first wrapper: every Pinecone call goes through `loop.run_in_executor(None, _fn)`
  because the SDK is sync.
- Standard fields are stored at flat top level for native filtering; user metadata lives in a
  `metadata: <json string>` Pinecone-side key. Filter splitting is identical to Chroma's pattern
  (`_split_filter` + `_matches_post_filter`).
- Batch upsert with `_generate_batches(records, batch_size)` (default 100) and configurable
  `show_progress`.

### 4.6 `milvus.py` — `MilvusProvider`

Uses the **`pymilvus.AsyncMilvusClient`** (Milvus 2.6+ API).

- Modes: EMBEDDED (Milvus Lite — `uri="./milvus.db"`), LOCAL (`http(s)://host:port`),
  CLOUD/Zilliz (`uri=https://...`, `token=api_key`).
- Schemas are explicit: `chunk_id` (VARCHAR primary key), `chunk_content_hash`,
  `doc_content_hash`, `knowledge_base_id`, `document_name`, `document_id`, `content` (VARCHAR
  65535), `metadata` (VARCHAR 65535 — JSON string), plus the dense vector
  (`FLOAT_VECTOR`). `enable_dynamic_field=True` lets unknown user keys live in a JSON sidecar.
- **Hybrid collection**: when `use_sparse_vectors=True`, an additional
  `SPARSE_FLOAT_VECTOR` field is added with a `SPARSE_INVERTED_INDEX` (`metric_type="IP"`,
  `drop_ratio_build=0.2`).
- **Milvus Lite quirk**: HNSW is not supported in EMBEDDED mode. The provider
  auto-falls-back to `IVF_FLAT` with `nlist = min(1024, max(64, m*4))` and logs a warning.
  Likewise scalar indexes (e.g. on `knowledge_base_id`) are skipped on EMBEDDED because
  MilvusLite rejects nullable+scalar-index fields.
- Hybrid search: builds two `AnnSearchRequest`s (dense + sparse), then ranks with
  `RRFRanker(rrf_k=60)` or `WeightedRanker(alpha, 1-alpha)` and submits via
  `client.hybrid_search(reqs=..., ranker=...)`.
- Filter expressions: `_build_filter_expression` synthesises Milvus expressions like
  `'chunk_id == "abc" and metadata like "%\"category\": \"docs\"%"'`. Standard fields are
  rendered as native column comparisons; user metadata uses LIKE on the JSON-serialised string
  (Milvus Lite/Zilliz `json_contains` is inconsistent on string-serialised JSON).
- Two-phase upsert: a query-by-`chunk_content_hash` for stale duplicates, then a real
  `client.upsert(...)` (Milvus PK upsert handles row replacement natively).

### 4.7 `weaviate.py` — `WeaviateProvider`

Uses the v4 Weaviate Python client (async clients via `weaviate.use_async_with_*`).

- Connection helpers per mode:
  - CLOUD: `use_async_with_weaviate_cloud(cluster_url, auth_credentials, headers,
    additional_config)`.
  - LOCAL: `use_async_with_local(host, port)`.
  - EMBEDDED / IN_MEMORY: `use_async_with_embedded(persistence_data_path?)`.
- **Multi-tenancy**: when `namespace` is set, the provider auto-enables
  `multi_tenancy_enabled=True` and creates the tenant on `acreate_collection`. Reads/writes are
  performed through `collection.with_tenant(...)`.
- **Generative & reranker modules**: the `generative_config` and `reranker_config` dicts (e.g.
  `{'provider': 'openai', 'model': 'gpt-4'}`) are passed to `collections.create()`. The provider
  collects API keys from `config.api_keys` or environment variables (full mapping in
  `_build_api_headers` — supports OpenAI, Anthropic, Cohere, Google, Mistral, Voyage, xAI, and
  more) and forwards them as `X-OpenAI-Api-Key`-style headers.
- Search APIs:
  - Dense: `collection.query.near_vector(near_vector, limit, filters, certainty=...,
    rerank=Rerank(prop=..., query=None))`.
  - Full-text: `collection.query.bm25(query, query_properties=["content"], limit, filters,
    rerank=Rerank(...))`.
  - Hybrid: `collection.query.hybrid(query=text, vector=vec, alpha, fusion_type=HybridFusion.RANKED
    | HybridFusion.RELATIVE_SCORE)`.
- IDs: chunk_id is `uuid.UUID(chunk_id_str)` if it parses as a UUID, else
  `generate_uuid5(identifier=chunk_id_str, namespace=collection_name)` for stable hashing.
- Upsert path: replace-or-insert by checking `await self.afield_exists("chunk_content_hash",
  hash)`; if a duplicate exists, `data.replace(uuid, properties, vector)` overwrites it,
  otherwise `data.insert(...)`.

### 4.8 `pgvector.py` — `PgVectorProvider`

Uses **synchronous** SQLAlchemy with `pgvector.sqlalchemy.Vector`, wrapped in
`asyncio.to_thread` for the async API.

- Engine: `create_engine(connection_string, pool_size, max_overflow, pool_timeout,
  pool_recycle)`. Session factory is a `scoped_session(sessionmaker(bind=engine,
  expire_on_commit=False))`.
- Schema v1 (`_get_table_schema_v1`): one table per collection with columns for `id` (PK),
  `chunk_id`, `chunk_content_hash`, `content`, `embedding` (`Vector(vector_size)`), `metadata`
  (JSONB default `'{}'::jsonb`), `document_name`, `document_id`, `doc_content_hash`,
  `knowledge_base_id`, `created_at`, `updated_at`.
- `acreate_collection()` does:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE SCHEMA IF NOT EXISTS "{schema_name}";  -- if not "public"
  CREATE TABLE ...
  -- Vector index (HNSW or IVFFlat) via _create_vector_index()
  -- GIN index on tsvector(content) via _create_gin_index()
  -- Optional metadata GIN indexes
  ```
- Identifier validation: `schema_name` and `table_name` are validated against
  `^[a-zA-Z_][a-zA-Z0-9_]*$` to prevent SQL injection in DDL.
- Upsert uses Postgres native `INSERT ... ON CONFLICT (chunk_content_hash) DO UPDATE` for
  content-hash dedup.
- Hybrid search: runs a vector-distance query (`embedding <=> :query_vector`) and a tsvector
  query (`to_tsvector('{content_language}', content) @@ plainto_tsquery(:q)`) separately, then
  fuses with RRF (`rrf_k=60`).
- Pool sizing: `pool_size=5`, `max_overflow=10`, `pool_timeout=30s`, `pool_recycle=3600s` —
  tunable via the config.

### 4.9 `supermemory.py` — `SuperMemoryProvider`

A **special case**: SuperMemory is a managed memory API that handles its own embedding,
chunking, and indexing. The provider adapts its API to the `BaseVectorDBProvider` contract.

- `vector_size = 0` and `dense_search_enabled = False` by default — the provider literally
  cannot do raw dense vector search. `adense_search` warns and returns `[]`.
- Ingestion is **text-based**: `aupsert` requires `chunks` and ignores `vectors` (sparse and
  dense are no-ops). Dedup by `chunk_content_hash` is done via SuperMemory search filters before
  insert.
- `acreate_collection` is a **no-op** — SuperMemory groups records by `container_tag` (defaults
  to `collection_name`). `adelete_collection` calls `documents.delete_bulk(container_tags=[...])`.
- Search modes: `'hybrid' | 'memories' | 'documents'` (passed straight through to the
  SuperMemory `search.memories()` API). `afull_text_search` always uses `mode='memories'`;
  `ahybrid_search` uses the configured `search_mode`.
- Indexing latency: SuperMemory indexes asynchronously, so after upsert the provider sleeps
  `index_delay` seconds (default `7.0`) so subsequent searches see the new data.
- Filter format: SuperMemory uses `{"AND": [{"key": ..., "value": ...}], "OR": [...]}`. The
  provider's `_convert_filters` translates `{"document_id": "x", ...}`-style dicts to that
  shape.

## 5. Cross-file relationships

```
                                 ┌─────────────────────┐
                                 │   knowledge_base/   │
                                 │  knowledge_base.py  │
                                 └──────────┬──────────┘
                                            │ holds vectordb: BaseVectorDBProvider
                                            ▼
   ┌──────────────────┐         ┌──────────────────────────┐
   │   embeddings/    │ embeds  │   vectordb.base          │
   │ EmbeddingProvider│ ──────► │   BaseVectorDBProvider   │ ◄──── store.VectorStore
   └──────────────────┘         │   (abstract; all `a*`)   │       (convenience wrapper)
                                └──────────┬───────────────┘
                                           │
        ┌────────────┬────────────┬────────┴────────┬────────────┬────────────┬────────────┐
        ▼            ▼            ▼                 ▼            ▼            ▼            ▼
  ChromaProvider  FaissProvider QdrantProvider PineconeProvider MilvusProvider
                                                                              WeaviateProvider PgVectorProvider SuperMemoryProvider

   each provider ──▶ uses its own *Config from   vectordb.config
   each provider ──▶ returns                     schemas.vector_schemas.VectorSearchResult
```

Key relationships:

- **`base.py` ⟷ `config.py`**: `BaseVectorDBProvider.__init__` accepts either a
  `BaseVectorDBConfig` instance OR a raw dict (which it instantiates as `BaseVectorDBConfig(**dict)`).
  Subclass providers narrow this — `ChromaProvider.__init__` upgrades the dict to a `ChromaConfig`
  via `ChromaConfig.from_dict`.
- **All providers ⟷ `schemas.vector_schemas.VectorSearchResult`**: every search method returns a
  `List[VectorSearchResult]` (frozen dataclass with `id, score, payload?, vector?, text?`).
- **All providers ⟷ `utils.package.exception`**: providers raise typed exceptions —
  `VectorDBConnectionError`, `ConfigurationError`, `CollectionDoesNotExistError`, `VectorDBError`,
  `SearchError`, `UpsertError`.
- **All providers ⟷ `utils.printing`**: structured logging via `info_log`, `debug_log`,
  `warning_log`, `error_log`, plus `import_error(...)` for missing-extra messages.
- **`store.py` ⟷ `embeddings/`**: `VectorStore` accepts an optional `EmbeddingProvider` and calls
  `embed_query(text)` / `embed_texts([text, ...])` to vectorize before delegating to the
  underlying provider's `aupsert` / `asearch`.
- **`__init__.py`**: lazy-imports keep startup fast — only the provider you actually reference
  pays the import cost (which is significant for, e.g., `weaviate-client` or `pinecone`).

## 6. Public API

Importable from `upsonic.vectordb`:

```python
# Base contract
from upsonic.vectordb import BaseVectorDBProvider

# High-level convenience wrapper
from upsonic.vectordb import VectorStore

# Concrete providers (lazy-imported)
from upsonic.vectordb import (
    ChromaProvider, FaissProvider, QdrantProvider, PineconeProvider,
    MilvusProvider, WeaviateProvider, PgVectorProvider, SuperMemoryProvider,
)

# Config classes + factory
from upsonic.vectordb import (
    BaseVectorDBConfig,
    DistanceMetric, IndexType, Mode,
    ConnectionConfig,
    HNSWIndexConfig, IVFIndexConfig, FlatIndexConfig,
    PayloadFieldConfig,
    ChromaConfig, FaissConfig, QdrantConfig, PineconeConfig,
    MilvusConfig, WeaviateConfig, PgVectorConfig, SuperMemoryConfig,
    create_config,
)
```

The provider authoring contract (every provider implements):

| Method                                        | Async   | Sync   |
|-----------------------------------------------|---------|--------|
| Connect / disconnect                          | `aconnect`, `adisconnect`, `ais_ready` | `connect`, `disconnect`, `is_ready` |
| Collection lifecycle                          | `acreate_collection`, `adelete_collection`, `acollection_exists` | `create_collection`, `delete_collection`, `collection_exists` |
| Data ops                                      | `aupsert`, `adelete`, `afetch`, `aupdate_metadata`, `aoptimize` | `upsert`, `delete`, `fetch`, `update_metadata`, `optimize` |
| Search                                        | `asearch`, `adense_search`, `afull_text_search`, `ahybrid_search` | `search`, `dense_search`, `full_text_search`, `hybrid_search` |
| Existence checks                              | `adocument_id_exists`, `adocument_name_exists`, `achunk_id_exists`, `adoc_content_hash_exists`, `achunk_content_hash_exists` | sync versions |
| Delete by field                               | `adelete_by_document_id`, `adelete_by_document_name`, `adelete_by_chunk_id`, `adelete_by_doc_content_hash`, `adelete_by_chunk_content_hash`, `adelete_by_metadata` | sync versions |
| Capabilities                                  | `aget_supported_search_types` | `get_supported_search_types` |

## 7. Integration with the rest of Upsonic

### 7.1 `KnowledgeBase` is the primary consumer

`src/upsonic/knowledge_base/knowledge_base.py` accepts a `BaseVectorDBProvider` at construction:

```python
from upsonic.knowledge_base import KnowledgeBase
from upsonic.vectordb import ChromaProvider, ChromaConfig, ConnectionConfig, Mode

kb = KnowledgeBase(
    sources=["./docs/intro.md", "./docs/api.md"],
    vectordb=ChromaProvider(config=ChromaConfig(
        collection_name="my_kb",
        vector_size=1536,
        connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path="./chroma_db"),
    )),
    embedding_provider=OpenAIEmbedding(),
    text_splitter=RecursiveCharacterTextSplitter(...),
)
```

The KB does not bypass the contract — every operation it needs is on
`BaseVectorDBProvider`:

- **Setup**: `kb.async_setup()` calls `vectordb.aconnect()` then
  `vectordb.acollection_exists()` / `vectordb.acreate_collection()`.
- **Two-layer dedup**: before processing each source it checks
  `vectordb.adoc_content_hash_exists(...)` to skip unchanged documents and
  `vectordb.adelete_by_document_id(...)` to clean up old versions before re-ingesting.
- **Storage**: `_store_in_vectordb(chunks, vectors)` calls `vectordb.aupsert(vectors=...,
  chunks=..., document_ids=..., document_names=..., doc_content_hashes=...,
  chunk_content_hashes=..., knowledge_base_ids=...)`.
- **Search**: `kb.aretrieve(query, ...)` (or `kb.asearch(...)`) → `vectordb.asearch(query_vector,
  query_text, top_k, filter, ...)`. The KB delegates the dense / full-text / hybrid choice to
  the provider's `asearch` master dispatcher.
- **Embedder zero-fallback**: the KB has provider-agnostic logic that calls
  `vectordb._config.vector_size` and produces zero vectors when the provider does its own
  embedding (e.g. SuperMemory) — see `_get_query_vector` and `_embed_chunks` in
  `knowledge_base.py`.

### 7.2 `VectorStore` (standalone use)

Already covered in §3.5 — for users who want add/search/delete without the KB machinery.

### 7.3 `Agent`/`Task` indirection

Agents access vector stores through `KnowledgeBase`, not directly. There is no `Agent.vectordb`
attribute; instead, an agent receives a `KnowledgeBase` (via `agent.knowledge` or the
`knowledge=...` task kwarg), and the agent's RAG tool calls `kb.aretrieve(...)`.

## 8. End-to-end flow of an upsert + similarity search

This walkthrough uses **Qdrant** as the concrete example since it implements the full surface
(dense, full-text, hybrid native, sparse vectors).

### 8.1 Construct config and provider

```python
from pydantic import SecretStr
from upsonic.vectordb import (
    QdrantProvider, QdrantConfig, ConnectionConfig, Mode,
    DistanceMetric, HNSWIndexConfig,
)

config = QdrantConfig(
    collection_name="research_papers",
    vector_size=1536,
    distance_metric=DistanceMetric.COSINE,
    connection=ConnectionConfig(
        mode=Mode.CLOUD,
        url="https://my-cluster.qdrant.io",
        api_key=SecretStr("..."),
    ),
    index=HNSWIndexConfig(m=16, ef_construction=200),
    use_sparse_vectors=True,
    indexed_fields=["document_name", {"field": "year", "type": "integer"}],
    default_top_k=10,
    default_hybrid_alpha=0.5,
)
provider = QdrantProvider(config=config)
```

What happens at construction:

1. `QdrantProvider.__init__` validates that the package is installed (`_QDRANT_AVAILABLE`).
2. `super().__init__(config)` (in `BaseVectorDBProvider.__init__`) assigns `self._config`,
   defaults `self.client = None`, generates a deterministic `self.id` via MD5 of
   class name + collection name (overridden by Qdrant to MD5 of host/url/port + collection).
3. Logging: `info_log("Initializing QdrantProvider for collection 'research_papers'.", ...)`.

### 8.2 Connect and create collection

```python
async with provider:
    await provider.acreate_collection()
    ...
```

What happens:

1. `__aenter__` → `aconnect()`.
2. `aconnect()` calls `aget_client()` which:
   - `_create_async_client()` → builds an `AsyncQdrantClient(url=..., api_key=..., prefer_grpc=...)`.
   - `await client.get_collections()` (health check). On failure, the client is recreated once.
3. `acreate_collection()` builds:
   ```python
   vectors_config = {
       "dense":  models.VectorParams(size=1536, distance=models.Distance.COSINE),
   }
   sparse_vectors_config = {
       "sparse": models.SparseVectorParams(),
   }
   hnsw_config = models.HnswConfigDiff(m=16, ef_construct=200)
   ```
   Then calls `client.create_collection(..., vectors_config=..., sparse_vectors_config=...,
   hnsw_config=...)` and finally `_create_field_indexes(...)` which creates one
   `KeywordIndexParams` for `document_name` and one `IntegerIndexParams` for `year`.

### 8.3 Upsert chunks

```python
import hashlib
texts = [
    "Attention is all you need.",
    "BERT is bidirectional.",
]
vectors = await openai_embedder.embed_texts(texts)         # via embeddings/
sparse  = bm25.encode_documents(texts)                     # provider-supplied
chunks_meta = [
    {"year": 2017},
    {"year": 2018},
]

await provider.aupsert(
    vectors=vectors,
    sparse_vectors=sparse,
    chunks=texts,
    payloads=chunks_meta,
    ids=["paper-1-c1", "paper-2-c1"],
    document_ids=["paper-1", "paper-2"],
    document_names=["Vaswani et al.", "Devlin et al."],
    chunk_content_hashes=[hashlib.md5(t.encode()).hexdigest() for t in texts],
)
```

Inside `QdrantProvider.aupsert`:

1. **Length validation** — checks every per-item array has length `n=2`.
2. **Hash computation** — uses the supplied `chunk_content_hashes` (or computes MD5 from chunks).
3. **Stale-duplicate detection** — `_batch_find_existing_by_hashes(computed_hashes)` issues a
   single `client.scroll(scroll_filter=Filter(must=[FieldCondition(key="chunk_content_hash",
   match=MatchAny(any=hashes))]))` to find any existing point with the same content hash but a
   different point ID. Stale IDs collected for deletion.
4. **Build `PointStruct` per chunk**:
   - `point_id` = normalised user-supplied ID (UUID string or int).
   - `vector_data` = `{"dense": vec, "sparse": SparseVector(indices, values)}` (named-vector
     mode because `use_sparse_vectors=True`).
   - `structured_payload` = standard fields (`chunk_id`, `chunk_content_hash`, `document_id`,
     `doc_content_hash`, `document_name`, `content`, `knowledge_base_id`) **flat at the top
     level**, plus a `metadata` dict containing `{"year": 2017}` etc.
5. **Stale delete + upsert**:
   ```python
   if stale_point_ids:
       await client.delete(..., points_selector=PointIdsList(points=stale_point_ids), wait=True)
   await client.upsert(collection_name=..., points=points, wait=False)
   ```
   `wait` is `True` only if `write_consistency_factor > 1`.

### 8.4 Run a hybrid similarity search

```python
query_text = "transformer attention mechanism"
query_vector = await openai_embedder.embed_query(query_text)
sparse_query = bm25.encode_queries(query_text)

results = await provider.asearch(
    top_k=5,
    query_vector=query_vector,
    query_text=query_text,
    sparse_query_vector=sparse_query,
    filter={"year": {"$gte": 2017}},
    alpha=0.6,                 # 60 % dense, 40 % sparse
    fusion_method="rrf",
    similarity_threshold=0.3,
    apply_reranking=True,
)
```

Dispatch in `QdrantProvider.asearch`:

1. `is_hybrid = (query_vector is not None) and (query_text is not None)` ⇒ True.
2. `hybrid_search_enabled` is True (default), so it calls `ahybrid_search(...)`.

Inside `ahybrid_search`:

1. **Native path** taken because `use_sparse_vectors` AND `sparse_query_vector` are both set.
2. `_native_hybrid_search(...)`:
   ```python
   await client.query_points(
       collection_name="research_papers",
       prefetch=[
           Prefetch(query=SparseVector(indices=..., values=...), using="sparse", limit=5),
           Prefetch(query=query_vector,                          using="dense",  limit=5),
       ],
       query=FusionQuery(fusion=Fusion.RRF),
       query_filter=Filter(must=[
           FieldCondition(key="year", range=Range(gte=2017)),
       ]),
       limit=5, with_payload=True, with_vectors=True,
   )
   ```
3. Each returned `point` is converted into a `VectorSearchResult(id=..., score=...,
   payload=self._flatten_payload(point.payload), vector=..., text=point.payload["content"])`.
4. If `self.reranker is not None and apply_reranking`, results are re-sorted via
   `self.reranker.rerank(query=query_text, documents=results)`.

Returned to caller: a `List[VectorSearchResult]` of length ≤ 5, sorted by fused score.

### 8.5 Disconnect

```python
# leaving the `async with provider:` block
```

`__aexit__` → `adisconnect()` → `aclose()` → `await client.close()` (sets `self.client = None`,
`self._is_connected = False`).

---

## Quick reference

### Choosing a provider

| Need                                              | Recommended provider |
|---------------------------------------------------|----------------------|
| "Just works" local prototyping                    | `ChromaProvider` (`Mode.IN_MEMORY` or `Mode.EMBEDDED`) |
| Single-process, no server, file-backed            | `FaissProvider`     |
| Production-grade async server with sparse hybrid  | `QdrantProvider` (`use_sparse_vectors=True`) |
| Fully managed cloud, hybrid in one index          | `PineconeProvider` (`hybrid_search_enabled=True`) |
| OSS server with hybrid, generative & rerank modules | `WeaviateProvider`  |
| Already on Postgres                               | `PgVectorProvider`  |
| Massive-scale, advanced ranking strategies         | `MilvusProvider` (with `RRFRanker` / `WeightedRanker`) |
| Don't want to manage embeddings yourself          | `SuperMemoryProvider` |

### Pip extras

```
pip install "upsonic[chroma]"      # chromadb
pip install "upsonic[faiss]"       # faiss-cpu, numpy
pip install "upsonic[qdrant]"      # qdrant-client
pip install "upsonic[pinecone]"    # pinecone, pinecone-text
pip install "upsonic[milvus]"      # pymilvus
pip install "upsonic[weaviate]"    # weaviate-client
pip install "upsonic[pgvector]"    # sqlalchemy, psycopg, pgvector
pip install "upsonic[supermemory]" # supermemory
```
