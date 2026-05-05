---
name: storage-persistence-layer
description: Use when working with Upsonic's persistence layer for sessions, user memory, cultural knowledge, or knowledge-base registry rows across multiple backends. Use when a user asks to choose, configure, or implement a storage backend (in-memory, JSON, SQLite, PostgreSQL, Redis, MongoDB, Mem0), wire a Memory orchestrator into an Agent/Team/Workflow, persist HITL checkpoints, or debug session save/load flows. Trigger when the user mentions Storage, AsyncStorage, InMemoryStorage, JSONStorage, SqliteStorage, AsyncSqliteStorage, PostgresStorage, AsyncPostgresStorage, RedisStorage, MongoStorage, AsyncMongoStorage, Mem0Storage, AsyncMem0Storage, Memory, SessionMemoryFactory, AgentSessionMemory, UserMemory, CultureMemory, KnowledgeRow, AgentSession, upsert_session, aupsert_session, prepare_inputs_for_task, save_session_async, run_awaitable_sync, is_async_storage_backend, HITL checkpoints, session memory, user analysis memory, summary memory, cultural knowledge, generic-model storage, JSONB, SQLAlchemy storage, gzip metadata chunking, or backend-portable persistence.
---

# `src/upsonic/storage/` — Multi-Provider Persistence Layer

This document describes every file under `src/upsonic/storage/` in the Upsonic
codebase. Storage is the persistence substrate that backs three orthogonal
concerns:

1. **Sessions** — chat history, runs, summaries, usage, metadata for an
   `Agent`/`Team`/`Workflow` execution.
2. **User memory** — extracted long-lived user profile/traits.
3. **Knowledge** — knowledge-base document registry rows + cultural-knowledge
   entries.

A unified `Storage` (sync) and `AsyncStorage` (async) abstract base defines
the API; concrete subclasses implement it for a specific backend
(in-memory, JSON files, SQLite, PostgreSQL, Redis, MongoDB, Mem0). On top of
the storage primitives sits a higher-level **`Memory`** orchestrator that
agents actually use; it composes session-memory and user-memory strategies
which call the storage primitives under the hood.

---

## 1. What this folder is

The folder defines a clean contract — *"persist these dataclasses, give them
back, optionally filter and sort"* — and provides **eight interchangeable
backends** behind that contract:

| Backend           | Sync class           | Async class             | Persistent | External deps                        |
|-------------------|----------------------|-------------------------|------------|--------------------------------------|
| In-memory         | `InMemoryStorage`    | —                       | No         | none (stdlib only)                   |
| JSON files        | `JSONStorage`        | —                       | Yes        | none (stdlib `json`)                 |
| SQLite            | `SqliteStorage`      | `AsyncSqliteStorage`    | Yes        | `sqlalchemy`, `aiosqlite`            |
| PostgreSQL        | `PostgresStorage`    | `AsyncPostgresStorage`  | Yes        | `sqlalchemy`, `psycopg`/`asyncpg`    |
| Redis             | `RedisStorage`       | —                       | Yes        | `redis`                              |
| MongoDB           | `MongoStorage`       | `AsyncMongoStorage`     | Yes        | `pymongo` (+ `motor` for async)      |
| Mem0 / Mem0 Cloud | `Mem0Storage`        | `AsyncMem0Storage`      | Yes        | `mem0`                               |

Backends are loaded lazily via `__getattr__` hooks at every package level so
that an environment without `psycopg` can still `import upsonic.storage` and
use `InMemoryStorage` or `JSONStorage`.

The **same abstract API** (`upsert_session` / `aget_session` /
`upsert_user_memory` / etc.) is honoured by every backend, so code that
takes a `Storage` argument is fully backend-portable. The `Memory`
orchestrator detects async-vs-sync backends at runtime via
`is_async_storage_backend()` and routes calls accordingly.

---

## 2. Folder layout (tree)

```
src/upsonic/storage/
├── __init__.py                 # Lazy package entry, re-exports
├── base.py                     # Storage / AsyncStorage abstract bases
├── schemas.py                  # UserMemory, KnowledgeRow dataclasses
├── utils.py                    # JSON-field (de)serialization, deserialize_session
│
├── in_memory/
│   ├── __init__.py
│   ├── in_memory.py            # InMemoryStorage
│   └── utils.py                # apply_sorting / apply_pagination / deep_copy_*
│
├── json/
│   ├── __init__.py
│   ├── json.py                 # JSONStorage
│   └── utils.py                # filter_sessions / filter_user_memories / sort / paginate
│
├── sqlite/
│   ├── __init__.py
│   ├── schemas.py              # SQLAlchemy column defs (String/JSON/BigInt)
│   ├── sqlite.py               # SqliteStorage  (sync, scoped_session)
│   ├── async_sqlite.py         # AsyncSqliteStorage (aiosqlite)
│   └── utils.py                # is_table_available_*, apply_sorting
│
├── postgres/
│   ├── __init__.py
│   ├── schemas.py              # JSONB column defs
│   ├── postgres.py             # PostgresStorage  (sync, psycopg)
│   ├── async_postgres.py       # AsyncPostgresStorage (asyncpg)
│   └── utils.py                # create_schema, ais_*_, sanitize_postgres_*
│
├── redis/
│   ├── __init__.py
│   ├── schemas.py              # SESSION_SCHEMA, USER_MEMORY_SCHEMA, *_INDEX_FIELDS
│   ├── redis.py                # RedisStorage (single + cluster)
│   └── utils.py                # key gen, indexing, filter/sort/paginate
│
├── mongo/
│   ├── __init__.py
│   ├── schemas.py              # *_COLLECTION_INDEXES, document structures
│   ├── mongo.py                # MongoStorage (PyMongo sync)
│   ├── async_mongo.py          # AsyncMongoStorage (Motor / async PyMongo)
│   └── utils.py                # remove_mongo_id, create_collection_indexes_*
│
├── mem0/
│   ├── __init__.py
│   ├── mem0.py                 # Mem0Storage (sync; self-hosted + Platform)
│   ├── async_mem0.py           # AsyncMem0Storage
│   └── utils.py                # serialize_*_to_mem0, gzip+base64 chunking
│
└── memory/                     # NOT a storage backend — orchestrator
    ├── __init__.py
    ├── memory.py               # Memory (top-level orchestrator)
    ├── factory.py              # SessionMemoryFactory (registry)
    ├── storage_dispatch.py     # is_async_storage_backend, run_awaitable_sync
    │
    ├── strategy/
    │   ├── __init__.py
    │   └── base.py             # BaseMemoryStrategy
    │
    ├── session/
    │   ├── __init__.py
    │   ├── base.py             # BaseSessionMemory + PreparedSessionInputs
    │   └── agent.py            # AgentSessionMemory (the only registered impl)
    │
    ├── user/
    │   ├── __init__.py
    │   ├── base.py             # BaseUserMemory
    │   └── user.py             # UserMemory (profile extraction + injection)
    │
    └── culture/
        ├── __init__.py
        ├── base.py             # BaseCultureMemory
        └── culture.py          # CultureMemory (cultural-knowledge wrapper)
```

---

## 3. Top-level files

### `__init__.py`

The package entry. **Everything is lazy.** Importing
`upsonic.storage` does not import SQLAlchemy, Redis, PyMongo or Mem0. Names
are resolved on attribute access through `__getattr__`, which iterates an
ordered `_LOADERS` list:

```python
_LOADERS: list[tuple[bool, Any]] = [
    (False, _get_base_classes),       # base.py — always available
    (False, _get_schema_classes),     # schemas.py — always available
    (False, _get_json_classes),       # JSONStorage — stdlib only
    (False, _get_in_memory_classes),  # InMemoryStorage — stdlib only
    (False, _get_memory_classes),     # Memory orchestrator
    (True,  _get_sqlite_classes),     # optional: sqlalchemy + aiosqlite
    (True,  _get_postgres_classes),   # optional
    (True,  _get_redis_classes),      # optional
    (True,  _get_mem0_classes),       # optional
    (True,  _get_mongo_classes),      # optional
]
```

Optional loaders are wrapped in `_safe_get` which swallows `ImportError /
ModuleNotFoundError`, so a missing optional dep never blocks unrelated
imports. The `__all__` list gives the public surface — `Storage`,
`AsyncStorage`, `KnowledgeRow`, `Memory`, every `*Storage` class, helper
schemas, and in-memory utility functions.

### `base.py` — `Storage` and `AsyncStorage`

The abstract contracts. Both share an `__init__` that records four
configurable table names (`session_table`, `user_memory_table`,
`cultural_knowledge_table`, `knowledge_table`) plus an `id` (auto-`uuid4` if
not given). Defaults are `upsonic_sessions`, `upsonic_user_memories`,
`upsonic_cultural_knowledge`, `upsonic_knowledge`.

The class defines six abstract groups of methods. Sync names use the bare
verb, async names are prefixed with `a` (e.g. `upsert_session` /
`aupsert_session`).

| Group              | Methods                                                                                      |
|--------------------|----------------------------------------------------------------------------------------------|
| Session            | `upsert_session`, `upsert_sessions`, `get_session`, `get_sessions`, `delete_session`, `delete_sessions` |
| User memory        | `upsert_user_memory`, `upsert_user_memories`, `get_user_memory`, `get_user_memories`, `delete_user_memory`, `delete_user_memories` |
| Cultural knowledge | `upsert_cultural_knowledge`, `get_cultural_knowledge`, `get_all_cultural_knowledge`, `delete_cultural_knowledge`, `delete_cultural_knowledges` |
| Knowledge content  | `upsert_knowledge_content`, `get_knowledge_content`, `get_knowledge_contents`, `delete_knowledge_content`, `delete_knowledge_contents` |
| Generic models     | `upsert_model`, `get_model`, `delete_model`, `list_models` (default no-ops; backends override) |
| Utility            | `table_exists`, `_create_all_tables`, `close`, `clear_all`, `get_usage`                      |

Two non-abstract conveniences:

* `delete_cultural_knowledges(ids)` has a default fallback that loops
  `delete_cultural_knowledge` per id and counts successes — backends may
  override for bulk efficiency.
* `get_usage(session_id)` defaults to `getattr(session, 'usage', None)` after
  loading the session.

Every method consistently accepts a `deserialize: bool` flag. When `True`
(the default) the backend returns proper dataclass instances
(`AgentSession`, `UserMemory`, `CulturalKnowledge`, `KnowledgeRow`); when
`False` it returns raw `Dict[str, Any]` (and for paginated reads a tuple
`(rows, total_count)`).

### `schemas.py` — generic data shapes

Two `@dataclass`es, both backend-agnostic:

```python
@dataclass
class UserMemory:
    user_memory: Dict[str, Any]
    user_id: Optional[str] = None
    created_at: Optional[int] = None       # epoch seconds
    updated_at: Optional[int] = None
    agent_id: Optional[str] = None
    team_id: Optional[str] = None
```

`__post_init__` coerces `created_at` to `now_epoch_s()` if missing and
normalises `updated_at` via `to_epoch_s()`. `to_dict` emits ISO-8601 strings
(stripping `None` values). `from_dict` is the inverse.

```python
@dataclass
class KnowledgeRow:
    id: str
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    size: Optional[int] = None
    knowledge_base_id: Optional[str] = None
    content_hash: Optional[str] = None
    chunk_count: Optional[int] = None
    source: Optional[str] = None
    status: Optional[str] = None
    status_message: Optional[str] = None
    access_count: Optional[int] = field(default=0)
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
```

This is the relational *registry* counterpart to chunks stored in the
`VectorDB`. It tracks document existence, processing status, content hashes,
chunk counts and access patterns.

### `utils.py` — shared helpers

A backend-neutral helper module. It defines:

* `SESSION_JSON_FIELDS` — the seven fields whose values are nested
  dict/list and must be JSON-serialised for SQL/JSON-file backends:
  `session_data`, `agent_data`, `team_data`, `workflow_data`, `metadata`,
  `runs`, `messages`.
* `serialize_session_json_fields(d)` and
  `deserialize_session_json_fields(d)` — convert each of those fields
  to/from JSON strings, idempotently.
* `deserialize_session(session_dict, session_type=None)` — central session
  factory. Reads `session_type` from the dict (string or nested dict),
  defaults to `SessionType.AGENT`, and dispatches to
  `AgentSession.from_dict(...)`. `TEAM` and `WORKFLOW` types are recognised
  but currently fall back to `AgentSession` (TeamSession / WorkflowSession
  not yet implemented).

Every backend that stores session JSON-as-text uses these helpers, which
guarantees cross-backend round-trip equivalence.

---

## 4. Per-backend file walkthrough

### 4.1 In-memory — `in_memory/`

A pure-Python, no-deps backend designed for tests, examples, and
ephemeral runs. Data is held in module-level lists and dicts on the
instance:

```python
self._sessions: List[Dict[str, Any]] = []
self._user_memories: List[Dict[str, Any]] = []
self._cultural_knowledge: List[Dict[str, Any]] = []
self._knowledge: List[Dict[str, Any]] = []
self._generic_models: Dict[str, Dict[str, Any]] = {}
```

`close()` clears all four lists (it is destructive, unlike SQL backends).
`table_exists()` always returns `True` — there is nothing to create.

Every read returns a `deepcopy` of the stored record (`deep_copy_record`)
so callers cannot accidentally mutate internal state. Sorting and
pagination are delegated to the `utils.py` siblings:

* `apply_sorting(data, sort_by, sort_order)` — uses `(value is None,
  value)` as key so Nones go to the end; defaults to `desc`. Falls back
  from `updated_at` to `created_at` when `updated_at` is `None`.
* `apply_pagination(data, limit, offset)` — slicing with `offset or 0`
  and optional `limit`.
* `get_sort_value` / `deep_copy_record` / `deep_copy_records` — helpers
  used inside `apply_sorting`.

`upsert_session` builds a session-record dict from `session.to_dict
(serialize_flag=True)`, writes/refreshes `created_at` and `updated_at`
(epoch seconds), and either appends or replaces the matching entry. For
`AgentSession` it picks a canonical column subset; for unknown session
types it stores the whole dict and stamps a `session_type` value via
`_get_session_type_value`. The same in-memory pattern repeats for
`upsert_user_memory`, `upsert_cultural_knowledge`, `upsert_knowledge_content`.

`upsert_model` / `get_model` / `delete_model` / `list_models` use the
extra `_generic_models[collection][key]` dict and `model.model_dump()` to
support arbitrary Pydantic models — this is how
`agent/deepagent/backends/memory_backend.py` persists `FilesystemEntry`
objects.

### 4.2 JSON files — `json/`

A drop-in persistent variant of the in-memory backend that writes one
JSON file per logical table under `db_path` (default
`./upsonic_json_db/`). Files are created on demand by
`_read_json_file(table_name, create_if_not_found=True)` which always
ensures `db_path.mkdir(parents=True, exist_ok=True)`. `_write_json_file`
dumps with `indent=2, default=str` for human readability.

`upsert_session` mirrors the in-memory implementation but writes the full
list back to disk after every call. Unlike the in-memory backend it must
deal with re-hydrating JSON blobs: every read goes through
`deserialize_session_json_fields` from `storage/utils.py` to convert
stringified JSON back to Python objects before handing it to
`deserialize_session`.

`json/utils.py` provides `filter_sessions(...)` (handles
`session_type` as either a string or `{"session_type": ...}` nested dict),
`filter_user_memories(...)`, `apply_sorting(...)`, `apply_pagination(...)`.

Generic-model storage is implemented by writing to a separate
`models_<collection>.json` file (`_get_model_file_name`).

### 4.3 SQLite — `sqlite/`

Two implementations sharing one schema definition.

**`schemas.py`** declares four SQLAlchemy column maps:

| Schema constant                    | Columns (highlights)                                                                                |
|------------------------------------|------------------------------------------------------------------------------------------------------|
| `SESSION_TABLE_SCHEMA`             | `session_id (PK)`, `session_type`, `agent_id`, `team_id`, `workflow_id`, `user_id`, `session_data`, `agent_data`, `team_data`, `workflow_data`, `metadata`, `runs`, `messages`, `usage`, `summary`, `created_at`, `updated_at` |
| `USER_MEMORY_TABLE_SCHEMA`         | `user_id (PK)`, `user_memory`, `agent_id`, `team_id`, `created_at`, `updated_at`                       |
| `CULTURAL_KNOWLEDGE_TABLE_SCHEMA`  | `id (PK)`, `name`, `summary`, `content`, `metadata`, `notes`, `categories`, `input`, `agent_id`, `team_id`, timestamps |
| `KNOWLEDGE_TABLE_SCHEMA`           | `id (PK)`, `name`, `description`, `metadata`, `type`, `size`, `knowledge_base_id`, `content_hash`, `chunk_count`, `source`, `status`, `status_message`, `access_count`, timestamps |

JSON columns are SQLAlchemy `JSON`, integer timestamps are `BigInteger`.
Each `index: True` flag becomes an `idx_<table>_<column>` index.
`get_table_schema_definition(table_type)` returns a copy by name.

**`utils.py`** provides three small helpers:

* `is_table_available_sync(session, name)` and the async cousin —
  `SELECT name FROM sqlite_master WHERE type='table' AND name=:name`.
* `apply_sorting(stmt, table, sort_by, sort_order)` — defaults to
  `created_at DESC` when `sort_by` is missing.

**`sqlite.py` (`SqliteStorage`)** uses a SQLAlchemy `Engine` + `scoped_session`.
Engine selection follows the priority: explicit `db_engine` → `db_url` →
`db_file` → `./upsonic.db`. The `_create_all_tables` method creates each
configured table by calling `_get_or_create_table(table_name, table_type,
create_if_not_found=True)`, which either reflects an existing table or
builds a new one from the schema definition (constructing
`Column` instances and per-column `Index`es).

`upsert_session` uses dialect-specific `sqlite.insert(...)
.on_conflict_do_update(...).returning(*table.columns)` to do a true
upsert in one round-trip; reads use `select(table)` with `where()` clauses
and `func.count()` for total counts. JSON columns are pre-serialised via
`serialize_session_json_fields` and post-deserialised via
`deserialize_session_json_fields` (keeping JSON structure even when
SQLite's `JSON` type already does it natively, for cross-backend
consistency).

Generic-model support uses dynamically created tables named
`upsonic_models_<collection>` with columns
`(key, collection, model_data, created_at, updated_at)`. The
`_get_generic_model_table` method caches `Table` objects in `self._tables`.

**`async_sqlite.py` (`AsyncSqliteStorage`)** mirrors the sync version using
SQLAlchemy's `AsyncEngine` + `async_sessionmaker(expire_on_commit=False)`
and the `aiosqlite` driver (`sqlite+aiosqlite:///...`). All public method
names are prefixed with `a` (`aupsert_session`, `aget_session`, etc.).

### 4.4 PostgreSQL — `postgres/`

Same dual sync/async pattern, with three Postgres-specific differences:

* **JSONB columns.** `schemas.py` imports
  `from sqlalchemy.dialects.postgresql import JSONB` and uses it for every
  JSON column, enabling indexed JSON queries.
* **Schemas (PG schemas, not table schemas).** `db_schema` (default
  `"public"`) is honoured throughout. `_create_all_tables` creates the
  Postgres schema first if `create_schema_flag=True` and `db_schema !=
  "public"`, via `CREATE SCHEMA IF NOT EXISTS "<schema>"`. `MetaData` is
  bound to the schema.
* **String sanitisation.** `utils.py` adds
  `sanitize_postgres_string`, `sanitize_postgres_dict`,
  `sanitize_postgres_list` which recursively strip `\x00` null bytes that
  Postgres rejects in text columns.
* **Connection pool tuning.** When creating an engine from `db_url`,
  `PostgresStorage` sets `pool_pre_ping=True` and `pool_recycle=3600` to
  weather "terminating connection due to administrator command" failures
  and stale connections.
* **Sorting fallback.** `apply_sorting` uses
  `func.coalesce(updated_at, created_at)` for SQL-side
  `updated_at`-with-fallback ordering.

Async variant uses `asyncpg` (`postgresql+asyncpg://...`).

### 4.5 Redis — `redis/`

Redis is a key-value store, so this backend simulates tables and indexes.

**Key layout.** Records live at
`{db_prefix}:{table_type}:{record_id}`. Index entries are Redis SETs at
`{db_prefix}:{table_type}:index:{field}:{value}` whose members are record
IDs. `db_prefix` defaults to `upsonic`.

**Schemas (`schemas.py`).** Plain `Final[Dict]`s describing field types
plus `*_INDEX_FIELDS` lists that drive index creation:

```python
SESSION_INDEX_FIELDS:           ["user_id", "agent_id", "team_id", "workflow_id", "session_type"]
USER_MEMORY_INDEX_FIELDS:       ["user_id", "agent_id", "team_id"]
CULTURAL_KNOWLEDGE_INDEX_FIELDS:["name", "agent_id", "team_id"]
KNOWLEDGE_INDEX_FIELDS:         ["knowledge_base_id", "type", "content_hash", "status"]
```

**Utils (`utils.py`).**

* `serialize_data` / `deserialize_data` — JSON dumps with a
  `CustomEncoder` that handles `UUID`, `date`, `datetime`.
* `generate_redis_key`, `generate_index_key` — string formatters above.
* `get_all_keys_for_table` — uses `scan_iter` (memory-efficient SCAN) and
  filters out `:index:` keys.
* `create_index_entries`, `remove_index_entries`, `get_records_by_index` —
  SET-based index maintenance via `sadd`, `srem`, `smembers`.
* `apply_filters`, `apply_sorting`, `apply_pagination` — Python-side
  post-fetch filtering.

**`redis.py` (`RedisStorage`).** Accepts a `redis_client` (Redis or
RedisCluster) or constructs one from `db_url`. An optional `expire`
parameter is applied as a TTL to every `SET` (and re-applied to indexes).

`_store_record(table_type, record_id, data, index_fields)` writes the
record key, then maintains the indexes if `index_fields` is given.
Reads first try direct ID lookup, then fall back to scanning the
table-type prefix and applying filter/sort/paginate in Python.

Generic-model storage and async support: this backend does not have an
`AsyncRedisStorage` class — only sync.

### 4.6 MongoDB — `mongo/`

MongoDB collections are created on first access. `_create_all_tables`
calls `_get_collection(<type>, create_collection_if_not_found=True)` for
each of the four logical collections (`sessions`, `user_memories`,
`cultural_knowledge`, `knowledge`). On first access, indexes are
installed via `create_collection_indexes(...)` which iterates the
matching `*_COLLECTION_INDEXES` from `schemas.py`:

| Collection         | Indexes (key, unique?)                                                                                   |
|--------------------|----------------------------------------------------------------------------------------------------------|
| sessions           | `session_id (unique)`, `session_type`, `agent_id`, `team_id`, `workflow_id`, `user_id`, `created_at desc`, `updated_at desc` |
| user_memories      | `user_id (unique)`, `agent_id`, `team_id`, `created_at desc`, `updated_at desc`                            |
| cultural_knowledge | `id (unique)`, `name`, `agent_id`, `team_id`, `created_at desc`, `updated_at desc`                          |
| knowledge          | `id (unique)`, `knowledge_base_id`, `type`, `content_hash`, `status`, `created_at desc`, `updated_at desc`  |

`upsert_session` uses
`collection.find_one_and_replace({"session_id": id}, doc, upsert=True,
return_document=ReturnDocument.AFTER)` for atomic upserts. `remove_mongo_id`
strips Mongo's auto-injected `_id` before returning. `apply_sorting`
returns a `[(field, ±1)]` list suitable for `cursor.sort(...)` and
`apply_pagination` chains `.skip()` and `.limit()` on the cursor.

Async variant (`async_mongo.py` / `AsyncMongoStorage`) uses the same
collection schema and index spec but with awaited PyMongo async (or Motor)
calls.

### 4.7 Mem0 — `mem0/`

Mem0 isn't a traditional database; it's a memory-as-a-service product
where each "memory" has a string `content` (which Mem0 may run through an
LLM) and a `metadata` dict. This backend abuses metadata to simulate
tables.

Key tricks live in `mem0/utils.py`:

* **Memory IDs.** `f"session__{table}__{session_id}"`,
  `f"user_memory__{table}__{user_id}"`,
  `f"cultural_knowledge__{table}__{id}"`,
  `f"knowledge__{table}__{document_id}"`. Underscores instead of colons
  for URL safety on Mem0 Platform.
* **Real data goes in `metadata._data`.** The `content` field is a short
  human-readable label (`"Upsonic session abc123"`). The actual session
  payload — which can be megabytes when tools are pickled — is JSON-
  serialised, **gzip-compressed** (`gzip.compress` + base64 encode), and
  stored in `metadata._data`.
* **2 KB metadata cap → chunking.** Mem0 caps metadata at ~2000 chars
  (`MEM0_METADATA_SIZE_LIMIT = 2000`,
  `MEM0_COMPRESSION_THRESHOLD = 1500`). If the compressed payload still
  doesn't fit, `serialize_session_to_mem0` splits it into chunks
  (`metadata._chunked = True`, `_chunk_count`, `_chunk_ids`) which are
  stored as separate Mem0 memories and reassembled on read.
* **Filter dicts.** `build_session_filters`, `build_user_memory_filters`,
  `build_cultural_knowledge_filters`, `build_knowledge_filters` build
  metadata-search dicts of the form
  `{"_type": "session", "_table": ..., "session_id": ..., ...}`.
* **Sort/paginate happen in Python** (`sort_records_by_field`,
  `apply_pagination`) since Mem0 does not provide server-side ordering
  of its results.

`mem0.py` (`Mem0Storage`) constructs either a `Memory` (self-hosted) or
`MemoryClient` (Platform) depending on whether you pass `config=...` or
`api_key=...`. Sync class. The async sibling lives in `async_mem0.py`.

---

## 5. The `memory/` subfolder — orchestrator, NOT a backend

This subpackage is what `Agent`/`Team`/`Workflow` actually instantiate.
It composes a backend-agnostic storage with three feature strategies
(session, user, culture) and exposes a clean save/load API.

### 5.1 `memory/__init__.py`

Re-exports `Memory`, `SessionMemoryFactory`, `BaseMemoryStrategy`,
`BaseSessionMemory`, `PreparedSessionInputs`, `AgentSessionMemory`,
`BaseUserMemory`, `UserMemory`, `BaseCultureMemory`, `CultureMemory` lazily.

### 5.2 `memory/memory.py` — `Memory` class

The top-level facade. It takes a `Storage`/`AsyncStorage` instance and a
set of *save/load* flags:

| Flag (save)              | Flag (load)                  | Effect                                                              |
|--------------------------|------------------------------|---------------------------------------------------------------------|
| `full_session_memory`    | `load_full_session_memory`   | Persist / inject chat history                                       |
| `summary_memory`         | `load_summary_memory`        | Generate / inject a session summary                                 |
| `user_analysis_memory`   | `load_user_analysis_memory`  | Run the user-trait analyzer / inject the resulting profile          |

Load flags default to their save siblings, so existing single-flag code
keeps working. The dual-flag design lets you, e.g., persist everything
but inject only summaries to save tokens.

Constructor responsibilities:

```python
self.session_id = session_id or uuid.uuid4()
self.user_id    = user_id    or uuid.uuid4()
self._session_memory_cache: Dict[SessionType, BaseSessionMemory] = {}
self._user_memory: Optional[BaseUserMemory] = (
    UserMemory(...) if user_analysis_memory or load_user_analysis_memory else None
)
```

Key methods:

* `get_session_memory(session_type)` — *runtime* selection. Calls
  `SessionMemoryFactory.create(session_type, ...)`, caches the result.
  Always creates a session memory if storage is available, even when
  every save flag is `False`, because **HITL checkpoints** for paused /
  errored / cancelled runs **must be persistable** to enable cross-process
  resumption.
* `prepare_inputs_for_task(session_type, agent_metadata)` — gathers
  `message_history`, `context_injection` (summary), `metadata_injection`
  (session + caller-supplied agent metadata) and `system_prompt_injection`
  (user profile). Returns a dict the agent injects into its run.
* `save_session_async(output, ...)` — single entry point. Internally calls
  `run_memory_agents_async(output)` (runs summary + trait sub-agents
  *before* `task.task_end()` so their `model_execution_time` rolls into
  the parent task's `RunUsage`) and then
  `persist_session_async(output)` (writes to storage *after*
  `task.task_end()` so `output.usage.duration` is finalised).
* `get_session_async`, `find_session_async`, `list_sessions_async`,
  `set_metadata_async`, `get_metadata_async`, `delete_session_async` —
  thin convenience wrappers around the underlying storage primitives.
* `load_resumable_run` / `load_run` — delegate to the per-type session
  memory's loaders.

Every async method has a sync mirror that calls
`run_awaitable_sync(...)` so callers can use either style regardless of
whether the underlying storage is sync or async.

### 5.3 `memory/factory.py` — `SessionMemoryFactory`

A small registry that maps `SessionType` → `BaseSessionMemory` subclass:

```python
class SessionMemoryFactory:
    _registry: Dict[SessionType, Type[BaseSessionMemory]] = {}
    @classmethod
    def register(cls, session_type, memory_class): ...
    @classmethod
    def create(cls, session_type, storage, session_id, ...) -> BaseSessionMemory: ...
```

`_ensure_initialized()` registers `AgentSessionMemory` for
`SessionType.AGENT` lazily. `TEAM` and `WORKFLOW` are not yet registered
(team/workflow session memory implementations are pending), so
`Memory.get_session_memory(SessionType.TEAM)` will raise
`ValueError("No session memory registered for SessionType.TEAM")` until
those classes ship.

### 5.4 `memory/storage_dispatch.py`

Two tiny helpers that make sync ↔ async storage transparent to strategy
code:

```python
def is_async_storage_backend(storage) -> bool:
    return isinstance(storage, AsyncStorage)

def run_awaitable_sync(awaitable):
    try:
        asyncio.get_running_loop()                # already in a loop?
    except RuntimeError:
        return asyncio.run(awaitable)             # nope → just run
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, awaitable).result()  # already in a loop → side-loop
```

Strategy classes use this pair everywhere: in their sync entry points
they detect an async backend and pump the corresponding `a*` call; in
async entry points they do the inverse and call sync storage directly.

### 5.5 `memory/strategy/base.py` — `BaseMemoryStrategy`

ABC inherited by every strategy (session, user, culture). It standardises
shared state (`storage`, `enabled`, `model`, `debug`, `debug_level`) and
forces subclasses to provide four entrypoints:

```python
@abstractmethod async def aget(...) -> Any: ...
@abstractmethod async def asave(...) -> Any: ...
@abstractmethod      def  get(...) -> Any: ...
@abstractmethod      def  save(...) -> Any: ...
```

### 5.6 `memory/session/` — session memory strategies

`base.py`:

```python
@dataclass
class PreparedSessionInputs:
    message_history: List[Any] = []
    context_injection: str = ""        # session summary
    metadata_injection: str = ""       # session/agent metadata
    session: Optional[Any] = None
```

`BaseSessionMemory(BaseMemoryStrategy, ABC)` adds:

* class-level `session_type: SessionType` (subclasses must define),
* save flags `enabled` / `summary_enabled`,
* load flags `load_enabled` / `load_summary_enabled`,
* knobs `num_last_messages`, `feed_tool_call_results`,
* abstract HITL loaders: `aload_resumable_run`, `aload_run` and their sync
  mirrors.

`agent.py` — `AgentSessionMemory` (the only registered impl). Highlights:

* `_build_prepared_inputs_from_session(session)` — applies the **load**
  flags only (save flags are irrelevant for reading) to construct a
  `PreparedSessionInputs`. Wraps summary in `<SessionSummary>...
  </SessionSummary>` and metadata in `<SessionMetadata>...
  </SessionMetadata>` for prompt injection.
* `arun_agents(output, is_completed)` — loads/creates the AgentSession,
  decides what to persist (always saves checkpoints for incomplete runs),
  optionally runs the summarizer sub-agent, and stashes the result in
  `self._pending_session` for a subsequent `apersist` call.
* `apersist(output, is_completed)` — writes `_pending_session` to storage
  via `storage.aupsert_session` / `storage.upsert_session` (chosen by
  `is_async_storage_backend`).
* `asave` — backwards-compatible shorthand: `arun_agents` + `apersist`.
* `_save_completed_run` — populates session from output, optionally
  generates a session summary by spinning up a `Summarizer` `Agent` whose
  task description bakes in the previous summary, the new turn's messages
  (via `output.new_messages()` + `ModelMessagesTypeAdapter`), and recent
  history (via `AgentSession.get_all_messages_for_session_id_async`). The
  summarizer's `RunUsage` is captured in `self._last_llm_usage` so the
  parent `output.usage.incr(...)` aggregates it.
* `_save_incomplete_run` — saves a **HITL checkpoint** for paused / error /
  cancelled runs regardless of save flags.
* `aload_resumable_run(run_id, agent_id)` — first checks the current
  session, then sweeps all sessions for the agent_id; only returns runs
  whose `output.status ∈ {paused, error, cancelled}`.
* `aload_run(run_id, agent_id)` — same search, no status filter.
* `_limit_message_history(messages)` — keeps only the last
  `num_last_messages` request/response pairs, but always re-injects the
  original `SystemPromptPart` so the agent's instructions never get lost.
* `_filter_tool_messages(messages)` — drops `tool-call` /
  `tool-return` parts when `feed_tool_call_results=False`.

### 5.7 `memory/user/` — user memory strategy

`BaseUserMemory` adds `user_id`, `load_enabled`, `profile_schema`,
`dynamic_profile`, `update_mode` (`"update" | "replace"`).

`UserMemory.aget` reads via `storage.aget_user_memory(user_id, agent_id,
team_id)`, then formats the dict into a `<UserProfile>...
</UserProfile>` block.

`UserMemory.asave(output, agent_id, team_id)`:

1. Loads the current profile dict.
2. Calls `_analyze_interaction_for_traits(output, current_profile)`,
   which:
   * Pulls historical user prompts via
     `AgentSession.get_all_user_prompt_messages_for_user_id_async(...,
     exclude_run_id=current_run_id)`.
   * Pulls new prompts from `output.new_messages()` via
     `AgentSession._extract_user_prompts_from_messages`.
   * Spawns a `User Trait Analyzer` `Agent`. If `dynamic_profile=True`,
     it first generates a 2-5 field schema (`ProposedSchema`) and uses
     `pydantic.create_model` to build a fly-by-night
     `DynamicUserTraitModel`; otherwise it uses
     `self._profile_schema_model` (defaults to `upsonic.schemas.UserTraits`).
   * Records the analyzer's `RunUsage` in `self._last_llm_usage` for
     parent aggregation.
3. Merges (`update`) or overwrites (`replace`) into `final_profile`.
4. Wraps in a `UserMemory(user_id, user_memory=final_profile, agent_id,
   team_id)` schema dataclass and calls `storage.aupsert_user_memory`.

### 5.8 `memory/culture/` — culture memory strategy

`BaseCultureMemory` exposes `aget_all`, `adelete`, `get_all`, `delete` plus
the inherited `aget` / `asave` etc.

`CultureMemory` is a thin wrapper around the storage's
`get_all_cultural_knowledge`, `get_cultural_knowledge`,
`upsert_cultural_knowledge`, `delete_cultural_knowledge` calls. It
maintains the same async/sync dispatch via `is_async_storage_backend +
run_awaitable_sync`. Unlike user memory it is shared across users — it
represents general cultural principles/guidelines accessible to any agent
or team.

---

## 6. Cross-file relationships

```
                         ┌───────────────┐
                         │   __init__.py │  (lazy public re-exports)
                         └───────┬───────┘
                                 │ exposes
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              base.py                                       │
│   class Storage(ABC)                       class AsyncStorage(ABC)         │
└─────────────┬─────────────────────────────────────────┬───────────────────┘
              │ inherits                                 │ inherits
              │                                          │
┌─────────────┴───────────────────────────────────┐ ┌────┴───────────────────┐
│ InMemoryStorage   JSONStorage                   │ │ AsyncSqliteStorage     │
│ SqliteStorage     PostgresStorage               │ │ AsyncPostgresStorage   │
│ RedisStorage      MongoStorage   Mem0Storage    │ │ AsyncMongoStorage      │
│                                                 │ │ AsyncMem0Storage       │
└─────────────────────────────────────────────────┘ └────────────────────────┘
                          │ all use
                          ▼
              ┌───────────────────────┐
              │ schemas.py            │  UserMemory, KnowledgeRow
              │ utils.py              │  serialize/deserialize_session_json_fields,
              │                       │  deserialize_session
              └───────────────────────┘

                                           ┌─────────────────────────────────┐
                                           │   memory/                       │
                                           │     Memory  (orchestrator)      │
                                           │       │ uses                    │
                                           │       ▼                         │
                                           │   SessionMemoryFactory ──┐      │
                                           │       │                  │      │
                                           │       ▼                  │      │
                                           │   AgentSessionMemory ──┐ │      │
                                           │   UserMemory       ────┤─┤──── storage_dispatch.py │
                                           │   CultureMemory    ────┘ │   is_async_storage_backend
                                           │                          │   run_awaitable_sync     │
                                           │   strategy/base.py ──────┘                          │
                                           └─────────────────────────────────────────────────────┘
```

* Every backend depends on `base.py`, `schemas.py` and `utils.py`.
* Every backend's `_create_all_tables` knows about all four logical tables
  (sessions, user_memories, cultural_knowledge, knowledge).
* Each backend has its own `schemas.py` describing **how** to lay out
  those four tables (SQL columns, JSONB columns, Redis index fields,
  Mongo indexes, Mem0 metadata filters, etc).
* `memory/` *consumes* a `Storage`/`AsyncStorage` instance — it never
  defines one. The dispatch helpers in `storage_dispatch.py` ensure
  strategy code can be written sync-style and still work over async
  backends, and vice versa.

---

## 7. Public API

The names available as `from upsonic.storage import X`:

```python
# Bases
Storage, AsyncStorage

# Schemas
KnowledgeRow                                     # also: UserMemory via storage.schemas

# Backends (sync)
InMemoryStorage, JSONStorage, SqliteStorage,
PostgresStorage, RedisStorage, MongoStorage, Mem0Storage

# Backends (async)
AsyncSqliteStorage, AsyncPostgresStorage,
AsyncMongoStorage, AsyncMem0Storage

# Postgres / SQLite schema helpers
SESSION_TABLE_SCHEMA, USER_MEMORY_TABLE_SCHEMA, get_table_schema_definition

# Redis schema helpers
SESSION_SCHEMA, USER_MEMORY_SCHEMA

# Memory orchestrator
Memory, SessionMemoryFactory, BaseMemoryStrategy,
BaseSessionMemory, PreparedSessionInputs, AgentSessionMemory,
BaseUserMemory, UserMemory

# In-memory utilities
apply_pagination, apply_sorting,
deep_copy_record, deep_copy_records, get_sort_value
```

`upsonic.storage.memory` additionally exposes `BaseCultureMemory` and
`CultureMemory`.

The contract every `*Storage` class fulfils is given in `base.py` and is
identical across backends (modulo the `a` prefix on async classes). Code
that takes `storage: Storage` therefore works with any sync backend; code
that takes `storage: Storage | AsyncStorage` and relies on
`is_async_storage_backend` + `run_awaitable_sync` (or its async
equivalent) works with both.

---

## 8. Integration with the rest of Upsonic

* **`Agent` / `Team` / `Workflow`** receive an optional `memory: Memory`
  parameter, not raw storage. Internally `Memory.storage` is the
  `Storage`/`AsyncStorage` instance; the agent never touches storage
  directly during a run — instead it calls
  `memory.prepare_inputs_for_task(...)` before LLM execution and
  `memory.save_session_async(output, ...)` (or the
  `run_memory_agents_async` + `persist_session_async` pair) after.
  See `src/upsonic/agent/context_managers/memory_manager.py` and the
  `Agent` class's run pipeline in `src/upsonic/agent/agent.py`.
* **DeepAgent / FilesystemEntry**. The agentic file system in
  `src/upsonic/agent/deepagent/backends/memory_backend.py` uses the
  generic-model API (`upsert_model` / `get_model` / `delete_model` /
  `list_models`) of `Storage` to persist `FilesystemEntry` Pydantic
  models in a `deepagent_filesystem` collection, transparently using
  `aupsert_model`/`alist_models` when the storage is async. Default
  backend, when none is provided, is `SqliteStorage` (see
  `composite_backend.py`).
* **`Session` types** live in `upsonic.session.*`. `Storage.upsert_session`
  takes a `Session` (today always `AgentSession`), serialises via
  `session.to_dict(serialize_flag=True)`, runs JSON fields through
  `serialize_session_json_fields`, writes, and returns either the
  dataclass or a raw dict depending on `deserialize`.
* **`KnowledgeBase` / `RAG`**. The `KnowledgeRow` registry is the
  relational companion to vector storage; chunks live in the VectorDB,
  metadata + content hashes + processing status live in
  `Storage.knowledge_table`. See `src/upsonic/knowledge_base/` for the
  consumer side.
* **`CulturalKnowledge`**. Defined in `src/upsonic/culture/`,
  persisted via `upsert_cultural_knowledge` and queried via
  `get_all_cultural_knowledge`. The `CultureMemory` strategy is the
  orchestrator that agents/teams plug into.
* **HITL (Human-in-the-Loop) checkpointing**. Incomplete runs (paused,
  error, cancelled) are *always* persisted (regardless of
  `full_session_memory`/`summary_memory` flags) so they can be resumed
  cross-process via `Memory.load_resumable_run(run_id, agent_id)`.

---

## 9. End-to-end flow of save / load

Below is what happens in a typical agent run with
`Memory(storage=SqliteStorage(...), full_session_memory=True,
summary_memory=True, user_analysis_memory=True, model="openai/gpt-4o")`.

### Save (after each agent run)

```text
Agent.run(...) finishes
        │
        ▼
Agent calls memory.save_session_async(output)
        │
        ▼
Memory.save_session_async  →  run_memory_agents_async
                              │
                              ▼
                     UserMemory.asave(output) — runs the analyzer Agent
                                                 with historical+new prompts,
                                                 builds final_profile,
                                                 storage.aupsert_user_memory(UserMemory(...))
                              │
                              ▼
                     AgentSessionMemory.arun_agents(output, is_completed)
                              │
                              ▼
                     storage.aget_session(session_id)  →  AgentSession
                              │
                              ▼
                     session.populate_from_run_output(output)
                     session.upsert_run(output)
                     summarizer Agent (if summary_enabled) → session.summary
                     session.append_new_messages_from_run_output(output)
                     self._pending_session = session                  ← NOT persisted yet
        │
        ▼
Memory.save_session_async  →  persist_session_async   (after task.task_end())
                              │
                              ▼
                     AgentSessionMemory.apersist(output, is_completed)
                              │
                              ▼
                     session.update_usage_from_run(output)             ← finalized usage
                     session.updated_at = now
                              │
                              ▼
                     storage.aupsert_session(session, deserialize=True)
                              │
                              ▼
                     SqliteStorage  →  sqlite.insert(table)
                                          .values(... session_data=JSON,
                                                  agent_data=JSON,
                                                  runs=JSON, messages=JSON,
                                                  usage=JSON, ...)
                                          .on_conflict_do_update(
                                              index_elements=["session_id"],
                                              set_={...})
                                          .returning(*table.columns)
                              │
                              ▼
                     Row written, JSON fields stay as JSON in SQLite.
```

The two-step `run_memory_agents_async` → `persist_session_async` split is
what lets sub-agent LLM calls (summarizer, trait analyzer) contribute
`model_execution_time` to `output.usage` *before* `task.task_end()` stops
the timer, while still ensuring the actual database write happens with
the finalised `output.usage.duration`.

### Load (before the next agent run)

```text
Agent starts a new task
        │
        ▼
Agent calls memory.prepare_inputs_for_task(SessionType.AGENT, agent_metadata)
        │
        ▼
Memory.get_session_memory(SessionType.AGENT)
   → SessionMemoryFactory.create(...)
   → AgentSessionMemory(storage, session_id, ...)         (cached)
        │
        ▼
session_memory.aget()  →  storage.aget_session(session_id, SessionType.AGENT)
        │
        ▼
SqliteStorage  →  select(table).where(session_id == ...)
                  row._mapping → dict
                  deserialize_session_json_fields(d)        ← JSON-strings → dicts/lists
                  deserialize_session(d, SessionType.AGENT) ← AgentSession.from_dict(...)
        │
        ▼
_build_prepared_inputs_from_session(session):
   if load_summary_enabled and session.summary:
       result.context_injection = "<SessionSummary>...</SessionSummary>"
   if load_enabled and session.messages:
       msgs = self._limit_message_history(session.messages)         ← last N runs
       msgs = self._filter_tool_messages(msgs)                      ← optional
       result.message_history = msgs
   if session.metadata:
       result.metadata_injection = "<SessionMetadata>...</SessionMetadata>"
        │
        ▼
UserMemory.aget()  →  storage.aget_user_memory(user_id)
                       → "<UserProfile>...</UserProfile>"
        │
        ▼
prepared_data = {
    "message_history":         [...],
    "context_injection":       "<SessionSummary>...</SessionSummary>",
    "system_prompt_injection": "<UserProfile>...</UserProfile>",
    "metadata_injection":      "<SessionMetadata>...</SessionMetadata><AgentMetadata>...</AgentMetadata>",
}
        │
        ▼
Agent merges into its system prompt + message history → LLM call.
```

The exact same flow works against `AsyncPostgresStorage`,
`AsyncMongoStorage`, etc. — `run_awaitable_sync` and
`is_async_storage_backend` route the calls correctly without the
strategy code knowing or caring about the backend kind. Swapping
backends is a one-line change at the call site that constructs the
storage instance:

```python
storage = SqliteStorage(db_file="./agent_data.db")
# or:
storage = PostgresStorage(db_url="postgresql://user:pass@host/db")
# or:
storage = RedisStorage(db_url="redis://localhost:6379/0", expire=86400)
# or:
storage = MongoStorage(db_url="mongodb://localhost:27017", db_name="upsonic")
# or:
storage = Mem0Storage(api_key=os.environ["MEM0_API_KEY"])

memory = Memory(storage=storage, ..., model="openai/gpt-4o")
agent  = Agent(memory=memory, ...)
```

That single substitution covers everything described above — sessions,
user memory, cultural knowledge, knowledge-base registry, generic
Pydantic-model storage, HITL checkpointing.
