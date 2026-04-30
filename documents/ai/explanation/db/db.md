---
name: db-storage-memory-facades
description: Use when working with the `src/upsonic/db/` package that bundles a Storage backend with a Memory runtime into a single `*Database` object passed to `Agent(database=...)`. Use when a user asks to pick or configure a session/memory backend, switch between SQLite, Postgres, Mongo, Redis, JSON, in-memory, or Mem0 persistence, enable full-session memory, summary memory, or user-analysis profiles, share an engine across databases, or debug optional-dependency import errors for storage extras. Trigger when the user mentions DatabaseBase, SqliteDatabase, PostgresDatabase, MongoDatabase, RedisDatabase, InMemoryDatabase, JSONDatabase, Mem0Database, upsonic.db, full_session_memory, summary_memory, user_analysis_memory, user_profile_schema, num_last_messages, session_id, user_id, SqliteStorage, PostgresStorage, Memory, Mem0Storage, sqlite-storage / postgres-storage / mongo-storage / redis-storage / mem0-storage extras, or PEP 562 lazy loading in upsonic.db.
---

# `src/upsonic/db/` — Database Facades for Storage + Memory

## 1. What this folder is — its role in Upsonic

The `db` package is a thin **composition layer** that ties together two
otherwise-independent subsystems of Upsonic:

1. The **Storage** subsystem (`upsonic.storage`) — pluggable backends that
   persist `Session` rows and `UserMemory` records to SQLite, Postgres, Mongo,
   Redis, JSON files, in-process dicts, or to an external Mem0 service.
2. The **Memory** runtime (`upsonic.storage.memory.memory.Memory`) — the
   stateful object responsible for full-session conversation memory, automatic
   summarization, and structured "user analysis" profiles fed back into prompts.

A Storage on its own cannot remember anything across an `Agent` invocation; a
`Memory` on its own has nowhere to durably write. The classes in this folder —
`SqliteDatabase`, `PostgresDatabase`, `MongoDatabase`, `RedisDatabase`,
`InMemoryDatabase`, `JSONDatabase`, `Mem0Database` — bundle a backend
constructor with a `Memory` constructor under a single `__init__` so that user
code can write:

```python
from upsonic.db import SqliteDatabase
from upsonic import Agent

db = SqliteDatabase(
    db_file="agent.db",
    session_id="chat-001",
    user_id="alice",
    full_session_memory=True,
    summary_memory=True,
    model="openai/gpt-4o",
)

agent = Agent(database=db, model="openai/gpt-4o")
```

…instead of hand-wiring `SqliteStorage(...)` then `Memory(storage=...)`.

The package is therefore a **facade / aggregate root**, not a database driver.
All actual SQL, BSON, JSON, and Redis I/O lives in `upsonic.storage`; all
prompt-shaping, summarization, and profile-update logic lives in
`upsonic.storage.memory.memory.Memory`. The `db` package just glues them and
publishes a single object the rest of Upsonic (Agent, Team, Task) accepts.

A secondary role is **optional-dependency isolation**: each backend lives behind
a `try/except AttributeError → ImportError` shim, so `import upsonic.db` works
even when `psycopg`, `pymongo`, `redis`, or `mem0ai` are not installed. Users
only pay the install cost for the storage they actually instantiate.

## 2. Folder layout — tree diagram

```
src/upsonic/db/
├── __init__.py        # Lazy-loading public API (PEP 562 __getattr__)
└── database.py        # All concrete Database* classes + DatabaseBase generic
```

That is the entire surface — two files, one of which is a re-export shim. The
package is intentionally flat: the real complexity is in `upsonic.storage.*` and
`upsonic.storage.memory.*`, and `db` only adds a typed wrapper around them.

| File | Lines (approx.) | Purpose |
| --- | --- | --- |
| `database.py` | ~505 | Defines `DatabaseBase` and seven concrete subclasses |
| `__init__.py` | ~61 | PEP-562 lazy loader; re-exports the seven `*Database` classes |

## 3. Top-level files — file-by-file walkthrough

### 3.1 `database.py`

#### Imports and `TYPE_CHECKING` block

```python
from __future__ import annotations
from typing import Optional, Type, Union, Dict, Any, List, Literal, Generic, TypeVar, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from pymongo import MongoClient
    from redis import Redis
    from mem0 import Memory as Mem0Memory, MemoryClient as Mem0MemoryClient
    from mem0.configs.base import MemoryConfig
    from ..storage.mem0 import Mem0Storage
    from ..storage.postgres import PostgresStorage
    from ..storage.redis import RedisStorage
    from ..storage.sqlite import SqliteStorage
    from ..storage.mongo import MongoStorage
    from ..storage.in_memory import InMemoryStorage
    from ..storage.json import JSONStorage
```

Everything heavy (`sqlalchemy`, `pymongo`, `redis`, `mem0`) is hidden inside a
`TYPE_CHECKING` block so static analyzers see the types but the runtime never
imports those packages from this module. This matters because the storage
backends are guarded by extras (`postgres-storage`, `mongo-storage`,
`redis-storage`, `mem0-storage`, `sqlite-storage`) — pulling the symbols at
runtime would defeat that guard.

The runtime imports kept at module load are deliberately minimal:

```python
from ..storage.base import Storage
from ..storage.memory.memory import Memory
from ..models import Model
from ..storage import InMemoryStorage, JSONStorage
```

`InMemoryStorage` and `JSONStorage` are pulled eagerly because they have no
optional dependencies — they ship with Upsonic core. The other backends are
lazily imported inside each `__init__`.

#### `StorageType` TypeVar

```python
StorageType = TypeVar('StorageType', bound=Storage)
```

A bounded type variable used to parameterize `DatabaseBase[StorageType]`. This
lets `SqliteDatabase` advertise itself as `DatabaseBase["SqliteStorage"]` so
that `db.storage` is correctly typed as `SqliteStorage` in IDEs.

#### `class DatabaseBase(Generic[StorageType])`

The abstract-ish base that holds a `(storage, memory)` pair.

| Attribute / Method | Type | Purpose |
| --- | --- | --- |
| `storage` | `StorageType` | The concrete `Storage` instance (Sqlite, Postgres, …) |
| `memory` | `Memory` | The `upsonic.storage.memory.memory.Memory` instance bound to `storage` |
| `__init__(storage, memory)` | constructor | Stores both fields verbatim |
| `session_id` (property) | `Optional[str]` | Proxy to `self.memory.session_id` |
| `user_id` (property) | `Optional[str]` | Proxy to `self.memory.user_id` |
| `__repr__()` | `str` | `"<ClassName>(storage=<StorageClass>, memory=<MemoryClass>)"` |

```python
@property
def session_id(self) -> Optional[str]:
    """Get session_id from memory."""
    return self.memory.session_id if self.memory else None
```

The `if self.memory else None` guard is defensive — in normal use `memory` is
always set by subclasses, but the guard prevents an `AttributeError` if a
caller subclasses `DatabaseBase` and forgets to call `super().__init__`.

`DatabaseBase` is **generic over `StorageType`** but not over `MemoryType` —
`Memory` is the only memory implementation in the framework today. Subclasses
specialize `StorageType` via the class header (e.g. `class SqliteDatabase(
DatabaseBase["SqliteStorage"])`) so `db.storage` is statically typed.

#### Common constructor parameters

Every concrete subclass exposes the same set of *memory-side* parameters,
forwarded verbatim into `Memory(...)`. Documenting them once here:

| Parameter | Type | Default | Forwarded to |
| --- | --- | --- | --- |
| `session_id` | `Optional[str]` | `None` | `Memory.session_id` (auto-generated UUID if `None`) |
| `user_id` | `Optional[str]` | `None` | `Memory.user_id` (auto-generated UUID if `None`) |
| `full_session_memory` | `bool` | `False` | Enables raw message-history persistence per session |
| `summary_memory` | `bool` | `False` | Enables LLM-generated rolling summaries |
| `user_analysis_memory` | `bool` | `False` | Enables structured-profile extraction (Pydantic schema) |
| `load_full_session_memory` | `Optional[bool]` | `None` | Whether to *load* on read (defaults to write flag) |
| `load_summary_memory` | `Optional[bool]` | `None` | Same, for summary |
| `load_user_analysis_memory` | `Optional[bool]` | `None` | Same, for user-analysis |
| `user_profile_schema` | `Optional[Type[BaseModel]]` | `None` | Pydantic schema for user-analysis extraction |
| `dynamic_user_profile` | `bool` | `False` | If `True`, schema is inferred dynamically rather than fixed |
| `num_last_messages` | `Optional[int]` | `None` | Sliding-window cap on injected message history |
| `model` | `Optional[Union[Model, str]]` | `None` | Model used by `Memory` for summarization / extraction |
| `debug` | `bool` | `False` | Verbose logging from `Memory` |
| `debug_level` | `int` | `1` | Verbosity level (1, 2, 3) |
| `feed_tool_call_results` | `bool` | `False` | If `True`, tool results enter the memory stream |
| `user_memory_mode` | `Literal['update', 'replace']` | `'update'` | How `Memory` merges new analysis (patch vs. overwrite) |

Each subclass adds **storage-specific** parameters on top of these. The split
is intentional: storage params describe *where* bytes go, memory params
describe *what* the bytes mean.

#### `class SqliteDatabase(DatabaseBase["SqliteStorage"])`

SQLite-backed database. Backed by `upsonic.storage.sqlite.SqliteStorage`.

Storage-specific parameters:

| Parameter | Type | Purpose |
| --- | --- | --- |
| `db_file` | `Optional[str]` | Filesystem path to a SQLite file (creates if missing) |
| `db_engine` | `Optional[Any]` | A pre-built SQLAlchemy `Engine` (overrides `db_file` / `db_url`) |
| `db_url` | `Optional[str]` | A SQLAlchemy URL (`sqlite:///...`) |
| `session_table` | `Optional[str]` | Custom table name for session rows |
| `user_memory_table` | `Optional[str]` | Custom table name for user-memory rows |

Behavior:

```python
try:
    from ..storage import SqliteStorage
except AttributeError as e:
    raise ImportError(
        "SqliteStorage is not available. Install optional dependency: uv sync --extra sqlite-storage"
    ) from e
```

The lazy-import pattern uses `AttributeError` (not `ImportError`) because
`upsonic.storage.__init__` exposes optional storages as module-level attributes
guarded by their own try/except — when an extra is missing the symbol is simply
absent, surfacing as `AttributeError`. This module re-frames that into the
familiar `ImportError("Install optional dependency …")`.

The constructor then:

1. Builds `SqliteStorage(db_file, db_engine, db_url, session_table, user_memory_table)`.
2. Builds `Memory(storage=storage, ... all the memory params ...)`.
3. Calls `super().__init__(storage=storage, memory=memory)`.

#### `class PostgresDatabase(DatabaseBase["PostgresStorage"])`

Postgres-backed database. Backed by `upsonic.storage.postgres.PostgresStorage`.

Storage-specific parameters:

| Parameter | Type | Purpose |
| --- | --- | --- |
| `db_url` | `Optional[str]` | SQLAlchemy URL (`postgresql+psycopg://...`) |
| `db_engine` | `Optional[Any]` | Pre-built SQLAlchemy `Engine` |
| `db_schema` | `Optional[str]` | Postgres schema name (e.g. `"upsonic"`) |
| `session_table` | `Optional[str]` | Custom session table name |
| `user_memory_table` | `Optional[str]` | Custom user-memory table name |
| `create_schema` | `bool` (default `True`) | Whether to `CREATE SCHEMA IF NOT EXISTS` on init |

Same import-guard / construct / forward pattern as `SqliteDatabase`. The
`create_schema=True` default mirrors the Postgres backend's expectation of
namespace isolation in shared databases.

#### `class MongoDatabase(DatabaseBase["MongoStorage"])`

MongoDB-backed database. Backed by `upsonic.storage.mongo.MongoStorage`.

Storage-specific parameters:

| Parameter | Type | Purpose |
| --- | --- | --- |
| `db_client` | `Optional[Any]` | Pre-built `pymongo.MongoClient` |
| `db_name` | `Optional[str]` | Mongo database name |
| `db_url` | `Optional[str]` | Connection URI |
| `session_collection` | `Optional[str]` | Collection name for sessions |
| `user_memory_collection` | `Optional[str]` | Collection name for user-memory |

Note the parameter naming: Mongo uses `*_collection` instead of `*_table`,
matching domain vocabulary. The memory params are still `session_table` /
`user_memory_table` — that asymmetry is preserved here because those names are
purely client-side identifiers in `Memory` (it does not know it's running on
Mongo).

#### `class RedisDatabase(DatabaseBase["RedisStorage"])`

Redis-backed database. Backed by `upsonic.storage.redis.RedisStorage`.

Storage-specific parameters:

| Parameter | Type | Purpose |
| --- | --- | --- |
| `redis_client` | `Optional[Any]` | Pre-built `redis.Redis` instance |
| `db_url` | `Optional[str]` | `redis://host:port/db` URL |
| `db_prefix` | `str` (default `"upsonic"`) | Key prefix to namespace within a Redis DB |
| `expire` | `Optional[int]` | TTL (seconds) applied to keys |
| `session_table` | `Optional[str]` | Logical session "table" name (used as part of the Redis key) |
| `user_memory_table` | `Optional[str]` | Same for user-memory |

`expire` is the only storage-level parameter that affects retention semantics —
useful for ephemeral chat sessions where you want auto-eviction without
implementing GC logic.

#### `class InMemoryDatabase(DatabaseBase[InMemoryStorage])`

Process-local dict-backed database. Backed by
`upsonic.storage.in_memory.InMemoryStorage`. **Has no optional-dependency
guard** because `InMemoryStorage` is in core.

Storage-specific parameters:

| Parameter | Type | Purpose |
| --- | --- | --- |
| `session_table` | `Optional[str]` | Logical "table" key |
| `user_memory_table` | `Optional[str]` | Logical "table" key |

Use cases: tests, REPL exploration, short-lived scripts, anywhere persistence
is undesirable. Notice the class header is `DatabaseBase[InMemoryStorage]`
(the type itself, not a string forward ref) — possible because the symbol is
imported eagerly at module top.

#### `class JSONDatabase(DatabaseBase[JSONStorage])`

JSON-file-backed database. Backed by `upsonic.storage.json.JSONStorage`. Also
in core, no extras needed.

Storage-specific parameters:

| Parameter | Type | Purpose |
| --- | --- | --- |
| `db_path` | `Optional[str]` | Directory or file path for the JSON store |
| `session_table` | `Optional[str]` | Logical filename / key |
| `user_memory_table` | `Optional[str]` | Logical filename / key |

A common dev workflow: `JSONDatabase(db_path="./.upsonic")` for human-readable
session inspection.

#### `class Mem0Database(DatabaseBase["Mem0Storage"])`

Bridge to the external [mem0](https://mem0.ai) memory service. Backed by
`upsonic.storage.mem0.Mem0Storage`.

Storage-specific parameters:

| Parameter | Type | Purpose |
| --- | --- | --- |
| `memory_client` | `Optional[Any]` | Pre-built `mem0.Memory` / `mem0.MemoryClient` |
| `api_key` | `Optional[str]` | Mem0 API key |
| `host` | `Optional[str]` | Self-hosted endpoint |
| `org_id` | `Optional[str]` | Mem0 organization id |
| `project_id` | `Optional[str]` | Mem0 project id |
| `config` | `Optional[Any]` | A `mem0.configs.base.MemoryConfig` for self-hosted setups |
| `session_table` | `Optional[str]` | Logical session table |
| `user_memory_table` | `Optional[str]` | Logical user-memory table |
| `default_user_id` | `str` (default `"upsonic_default"`) | Fallback user id when none provided |

Mem0 is the only backend where the memory layer (`Memory`) and the storage
layer (`Mem0Storage`) potentially overlap conceptually — Mem0 itself is a
memory service. Upsonic still wraps it with its own `Memory` because the
`Memory` class is responsible for prompt assembly and summarization regardless
of where the bytes live.

### 3.2 `__init__.py`

The package init implements **PEP 562** (`__getattr__` on a module) to defer
heavy imports until a class is actually requested.

#### `TYPE_CHECKING` re-exports

```python
if TYPE_CHECKING:
    from .database import (
        DatabaseBase,
        SqliteDatabase,
        PostgresDatabase,
        MongoDatabase,
        RedisDatabase,
        InMemoryDatabase,
        JSONDatabase,
        Mem0Database,
    )
```

These imports run only under static analysis (mypy, pyright). They give IDEs
full type information for `from upsonic.db import SqliteDatabase` without
forcing the runtime to import `database.py` (which transitively touches `Model`
and the `Memory` class).

#### `_get_database_classes() -> dict`

```python
def _get_database_classes():
    """Lazy import of database classes."""
    from .database import (
        DatabaseBase,
        SqliteDatabase,
        ...
        Mem0Database,
    )
    return {
        'DatabaseBase': DatabaseBase,
        'SqliteDatabase': SqliteDatabase,
        ...
        'Mem0Database': Mem0Database,
    }
```

A private helper that performs the actual `from .database import ...` and
returns a `name -> class` dict. Called once per `__getattr__` lookup (not
cached at the module level — the import system itself caches `database` after
first hit, so the cost is a dictionary build).

#### `__getattr__(name: str) -> Any`

```python
def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    database_classes = _get_database_classes()
    if name in database_classes:
        return database_classes[name]

    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )
```

This is the PEP-562 hook. Any `from upsonic.db import X` resolves through here
on first access, returning the matching class or raising a clean
`AttributeError`. The error message hints that unrecognized names should come
from a sub-module — which currently means `upsonic.storage` directly.

#### `__all__`

```python
__all__ = [
    "DatabaseBase",
    "SqliteDatabase",
    "PostgresDatabase",
    "MongoDatabase",
    "RedisDatabase",
    "InMemoryDatabase",
    "JSONDatabase",
    "Mem0Database",
]
```

Defines the public surface — exactly the seven `*Database` classes plus the
generic base. Anything else is internal.

## 4. Subfolders

There are no subfolders. The package is intentionally flat — a single
`database.py` plus the lazy-loading `__init__.py`. All variation is in
**class instances**, not directories.

## 5. Cross-file relationships

```
upsonic.db.__init__          ← public entry; lazy-imports from .database
        │
        ▼
upsonic.db.database
        │   uses
        ├─→ upsonic.storage.base.Storage              (TypeVar bound)
        ├─→ upsonic.storage.memory.memory.Memory      (composed in every subclass)
        ├─→ upsonic.models.Model                      (forwarded into Memory)
        ├─→ upsonic.storage.InMemoryStorage           (eager — core)
        ├─→ upsonic.storage.JSONStorage               (eager — core)
        └─→ upsonic.storage.{Sqlite,Postgres,Mongo,Redis,Mem0}Storage
                                                       (lazy — extras)
```

| Direction | Module | Why |
| --- | --- | --- |
| Outbound | `upsonic.storage.base.Storage` | Bound for the `StorageType` TypeVar; all concrete storages subclass it |
| Outbound | `upsonic.storage.memory.memory.Memory` | Every `*Database` instantiates a `Memory(storage=...)` |
| Outbound | `upsonic.models.Model` | Type hint for the `model` constructor arg passed to `Memory` |
| Outbound | `upsonic.storage` (package) | Provides eager `InMemoryStorage`/`JSONStorage` and lazy `*Storage` |
| Inbound | `upsonic.agent.agent` | Agents accept a `database=...` argument — the `*Database` object |
| Inbound | `upsonic.team.team` | Teams may forward a `database` to coordinated agents |
| Inbound | User code | `from upsonic.db import SqliteDatabase` etc. |

Note that **no code in `upsonic.db` imports `upsonic.agent` or `upsonic.team`**.
The dependency direction is one-way: agents and teams depend on `db`, not the
reverse. This keeps `db` reusable in scripts that don't construct an `Agent`.

## 6. Public API

After lazy resolution, the package exposes exactly:

| Symbol | Kind | Description |
| --- | --- | --- |
| `DatabaseBase` | class (generic) | Abstract base bundling `(storage, memory)` |
| `SqliteDatabase` | class | SQLite + Memory |
| `PostgresDatabase` | class | Postgres + Memory |
| `MongoDatabase` | class | Mongo + Memory |
| `RedisDatabase` | class | Redis + Memory |
| `InMemoryDatabase` | class | Dict + Memory (no extras) |
| `JSONDatabase` | class | JSON file + Memory (no extras) |
| `Mem0Database` | class | Mem0 service + Memory |

All seven `*Database` classes share **the same memory-side parameter set** (see
table in §3.1) and differ only in their storage-side parameters.

### Quick reference — choose your storage

| Need | Use | Required extra |
| --- | --- | --- |
| Throwaway tests / REPL | `InMemoryDatabase` | none |
| Single-machine dev with file inspection | `JSONDatabase` | none |
| Single-machine prod, embedded | `SqliteDatabase` | `sqlite-storage` |
| Multi-process / horizontally scaled | `PostgresDatabase` | `postgres-storage` |
| Document-oriented persistence | `MongoDatabase` | `mongo-storage` |
| Ephemeral sessions with TTL | `RedisDatabase` (`expire=...`) | `redis-storage` |
| Externalized memory service | `Mem0Database` | `mem0-storage` |

### Common usage patterns

#### Direct construction

```python
from upsonic.db import PostgresDatabase
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str | None = None
    preferences: list[str] = []

db = PostgresDatabase(
    db_url="postgresql+psycopg://upsonic:secret@localhost/upsonic",
    db_schema="upsonic",
    session_id="conv-42",
    user_id="bob",
    full_session_memory=True,
    summary_memory=True,
    user_analysis_memory=True,
    user_profile_schema=UserProfile,
    num_last_messages=20,
    model="openai/gpt-4o",
)
```

#### Reading proxied identifiers

```python
db.session_id   # → "conv-42"        (from db.memory.session_id)
db.user_id      # → "bob"            (from db.memory.user_id)
db.storage      # → PostgresStorage instance
db.memory       # → Memory instance
```

`storage` and `memory` are deliberately public — power users can drop down to
the storage layer for raw row queries without going through `Memory`.

#### Reusing a pre-built engine / client

Each backend that takes a `db_engine` / `db_client` / `redis_client` /
`memory_client` accepts a *pre-built* instance, so a single connection pool can
be shared across multiple `*Database` constructions:

```python
from sqlalchemy import create_engine
from upsonic.db import PostgresDatabase

engine = create_engine("postgresql+psycopg://...")
db_a = PostgresDatabase(db_engine=engine, session_id="a", user_id="alice")
db_b = PostgresDatabase(db_engine=engine, session_id="b", user_id="alice")
```

Because the `Memory` is per-`*Database`, two databases sharing a backend are
two independent memory streams — a common pattern for multi-conversation apps.

## 7. Integration with the rest of Upsonic

### 7.1 `Agent` integration

`upsonic.agent.agent.Agent` accepts a `database=...` keyword. Internally the
agent reads:

| Field | Used for |
| --- | --- |
| `database.storage` | Persisting/retrieving session rows and user-memory rows |
| `database.memory` | Pre-call: assembling system + history; post-call: writing the new turn |
| `database.session_id` / `database.user_id` | Defaulting per-call identifiers |

Because `*Database` is just a holder, the agent does not depend on which
backend is in use. Swapping `SqliteDatabase` → `RedisDatabase` is a one-line
change with no agent-side modifications.

### 7.2 `Team` integration

`upsonic.team.team.Team` can forward a single `database=...` to constituent
agents so all members share session state, or each agent can hold its own. The
generic `DatabaseBase` is the type expected — concrete backends are
interchangeable.

### 7.3 `Memory` lifecycle

Inside each `*Database.__init__` the construction order is fixed:

1. **Storage first** — because `Memory` requires a `storage` argument and
   reads/writes through it for every operation.
2. **Memory second** — wired with the freshly built storage and the user's
   memory-config flags.
3. **`super().__init__(storage, memory)`** — registers both on `DatabaseBase`.

Once `super().__init__` completes, the `*Database` is *ready*: it has not
issued any I/O yet — the underlying `Storage` is connected (e.g., engine
created, schema ensured) but no session has been read or written. The first
real I/O happens on the first `Agent.run(...)`.

### 7.4 Optional-dependency story

The seven backends correspond to seven `pyproject.toml` extras:

| Backend | Extra | Underlying packages |
| --- | --- | --- |
| `SqliteDatabase` | `sqlite-storage` | `sqlalchemy`, `aiosqlite` |
| `PostgresDatabase` | `postgres-storage` | `sqlalchemy`, `psycopg` |
| `MongoDatabase` | `mongo-storage` | `pymongo` |
| `RedisDatabase` | `redis-storage` | `redis` |
| `Mem0Database` | `mem0-storage` | `mem0ai` |
| `InMemoryDatabase` | (core) | none |
| `JSONDatabase` | (core) | none |

If a user instantiates `PostgresDatabase` without `--extra postgres-storage`,
the inner `from ..storage import PostgresStorage` raises `AttributeError`,
which the `try/except` in `database.py` re-raises as:

```
ImportError: PostgresStorage is not available. Install optional dependency: uv sync --extra postgres-storage
```

This is the canonical Upsonic pattern for guarding extras — see the same idiom
in `upsonic.knowledge_base` and `upsonic.rag`.

## 8. End-to-end flow

A typical agent turn touches `db` at three points. Below is the full lifecycle
for a single `Agent.run("hello")` call configured with `SqliteDatabase` and
`full_session_memory + summary_memory + user_analysis_memory` all enabled.

```
                  ┌────────────────────────────────────────────────┐
                  │   User code constructs the database            │
                  └────────────────────────────────────────────────┘
                                     │
                                     ▼
       SqliteDatabase(__init__):
          1. from ..storage import SqliteStorage  (lazy)
          2. SqliteStorage(db_file=...) ─── opens engine, creates tables
          3. Memory(storage=..., flags) ─── prepares prompt-assembly logic
          4. DatabaseBase.__init__(storage, memory)
                                     │
                                     ▼
                  ┌────────────────────────────────────────────────┐
                  │   Agent.run("hello")  (first call)             │
                  └────────────────────────────────────────────────┘
                                     │
                                     ▼
       Pre-call (READ):
          • db.memory.load_session(session_id, user_id)
              └── db.storage.read(session_table, …)        ─── SQL SELECT
          • Memory builds:
              - prior turns (full_session_memory)
              - rolling summary (summary_memory)
              - user-analysis profile (user_analysis_memory)
          • Memory injects them as system / context messages
                                     │
                                     ▼
       Model call:
          • Agent runs the configured Model with merged messages
                                     │
                                     ▼
       Post-call (WRITE):
          • db.memory.append_turn(user_msg, assistant_msg, tool_calls)
              ├── if full_session_memory: db.storage.upsert(session_table, ...)
              ├── if summary_memory:      Memory asks `model` to summarize, writes back
              └── if user_analysis_memory: Memory asks `model` to update profile
                                           via user_profile_schema (mode=update|replace)
                                     │
                                     ▼
                  ┌────────────────────────────────────────────────┐
                  │   Result returned to the user                  │
                  └────────────────────────────────────────────────┘
```

Key invariants visible in this flow:

1. **`db.storage` is the only I/O surface**. `Memory` never opens its own
   connection — it always goes through `storage`.
2. **`db.memory` owns the prompt shape**. The agent never re-implements
   summarization or profile extraction; that is the `Memory`'s job.
3. **`*Database.__init__` is the only place storage params and memory params
   meet**. Once construction is done, the two halves are independent and the
   agent talks to each through its own interface.

### Concrete example — full lifecycle

```python
from upsonic import Agent
from upsonic.db import SqliteDatabase
from pydantic import BaseModel

class Profile(BaseModel):
    name: str | None = None
    favorite_topic: str | None = None

# 1. Build the database (no I/O yet on Memory; SqliteStorage opens the file)
db = SqliteDatabase(
    db_file="chat.db",
    session_id="s-001",
    user_id="u-alice",
    full_session_memory=True,
    summary_memory=True,
    user_analysis_memory=True,
    user_profile_schema=Profile,
    num_last_messages=10,
    model="openai/gpt-4o",
)

# 2. Plug it into an agent
agent = Agent(database=db, model="openai/gpt-4o")

# 3. Each call goes through the load → model → save loop above
result_1 = agent.do("My name is Alice and I love astrophysics.")
result_2 = agent.do("What did I tell you about myself?")

# 4. Inspect what was persisted (drop down to storage for raw access)
sessions = db.storage.read(session_id="s-001")
profile  = db.memory.user_analysis  # <-- pydantic Profile or dict
```

By the time `result_2` is computed, `db.memory` will have:

* loaded the prior turn from `chat.db` via `SqliteStorage`,
* injected a summary line and the rolling profile (`Profile(name="Alice",
  favorite_topic="astrophysics")`) into the system prompt,
* asked the model to answer, and
* written the new turn back, updating both the session row and the user
  profile.

All of this is enabled by **two files and seven classes** in `upsonic/db/`,
which is the entire API surface a user needs to understand to pick a backend
and turn on memory.
