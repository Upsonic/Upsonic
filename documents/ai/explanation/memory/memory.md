---
name: agent-memory-orchestration
description: Use when working with Upsonic's agent memory layer, configuring session history, summaries, or user-profile extraction, or wiring a Memory orchestrator to a Storage backend. Use when a user asks to enable full_session_memory, summary_memory, user_analysis_memory, set load flags, resume paused/error/cancelled runs (HITL), generate session summaries, extract user traits, use dynamic_user_profile, or use the legacy save_agent_memory/get_agent_memory/reset_agent_memory scratchpad. Trigger when the user mentions Memory, MemoryManager, AgentSessionMemory, UserMemory, CultureMemory, BaseMemoryStrategy, SessionMemoryFactory, PreparedSessionInputs, prepare_inputs_for_task, run_memory_agents_async, persist_session_async, save_session_async, load_resumable_run, SessionType.AGENT, num_last_messages, feed_tool_call_results, user_profile_schema, UserTraits, update vs replace mode, summarizer sub-agent, or src/upsonic/memory and src/upsonic/storage/memory.
---

# `src/upsonic/memory/` — Agent Memory Folder

This document walks through the **top-level memory package** at
`src/upsonic/memory/` and the **memory orchestration sub-tree** at
`src/upsonic/storage/memory/` that the top-level package re-exports as
`Memory`. The two pieces are intentionally split: the small folder at
`src/upsonic/memory/` is a **legacy, file-based "scratchpad" memory** for
agents and a thin **public re-export shim** for the heavyweight `Memory`
orchestrator that actually drives session, summary and user-profile
features.

> **Important:** Throughout the codebase the *behavioural* memory layer
> lives under `src/upsonic/storage/memory/` (orchestration). The folder
> `src/upsonic/storage/` itself is the *backend abstraction* layer
> (SQLite, Postgres, Redis, Mongo, JSON, in-memory, mem0). They are
> different concerns. This doc focuses on the orchestration side and
> calls out the storage interaction explicitly.

---

## 1. What this folder is — Memory orchestration (NOT to be confused with `/storage`)

The Upsonic framework draws a hard line between two responsibilities:

| Concern | Folder | Purpose |
| --- | --- | --- |
| **Memory orchestration** | `src/upsonic/memory/` (legacy file scratchpad) + `src/upsonic/storage/memory/` (modern `Memory` class) | Decides *what* to load before a run, *what* to save after a run, runs sub-agents that synthesize summaries / user profiles. |
| **Storage backends** | `src/upsonic/storage/` (excluding `memory/`) | Pure CRUD against a backend (SQLite, Postgres, Redis, Mongo, JSON, in-memory, mem0). Implements the `Storage` / `AsyncStorage` ABC. |
| **Session schema** | `src/upsonic/session/` | Domain types (`AgentSession`, `RunData`, `SessionType`) that move between the two layers above. |

The orchestration layer **uses** the storage layer; the storage layer
knows nothing about agents or LLMs. The `Memory` class is what ties
them together for an `Agent`/`Team`/`Workflow` instance.

**The `src/upsonic/memory/` folder itself plays two roles:**

1. **Legacy scratchpad memory** (`memory.py`) — a tiny SHA256-keyed JSON
   file persistence for raw `pydantic` message histories, used by older
   agent code paths. Files are written **next to the source file**
   (`MEMORY_DIR = Path(__file__).parent`).
2. **Public lazy-import shim** (`__init__.py`) — re-exports
   `save_agent_memory`, `get_agent_memory`, `reset_agent_memory` from
   the local `memory.py`, plus the heavyweight `Memory` orchestrator
   from `..storage.memory.memory`. Users importing
   `from upsonic.memory import Memory` get the orchestrator, not the
   scratchpad.

```python
# Public surface exposed by src/upsonic/memory/__init__.py
from upsonic.memory import (
    Memory,             # → upsonic.storage.memory.memory.Memory (orchestrator)
    save_agent_memory,  # → local file-based scratchpad
    get_agent_memory,
    reset_agent_memory,
)
```

The `__getattr__` hook makes the import lazy so importing `upsonic`
does not pull in the entire storage stack.

---

## 2. Folder layout (tree)

The literal contents of `src/upsonic/memory/`:

```
src/upsonic/memory/
├── __init__.py           # Lazy public re-exports (Memory + scratchpad fns)
└── memory.py             # File-based per-agent JSON message-history scratchpad
```

The orchestration sub-tree at `src/upsonic/storage/memory/` that this
folder re-exports `Memory` from:

```
src/upsonic/storage/memory/
├── __init__.py                    # Lazy re-exports: Memory, factory, bases
├── memory.py                      # Memory class (orchestrator)
├── factory.py                     # SessionMemoryFactory (per-SessionType)
├── storage_dispatch.py            # is_async_storage_backend / run_awaitable_sync
├── strategy/
│   └── base.py                    # BaseMemoryStrategy ABC (aget/asave/get/save)
├── session/
│   ├── base.py                    # BaseSessionMemory + PreparedSessionInputs
│   └── agent.py                   # AgentSessionMemory (HITL, summaries, history)
├── user/
│   ├── base.py                    # BaseUserMemory ABC
│   └── user.py                    # UserMemory (profile extraction sub-agent)
└── culture/
    ├── base.py                    # BaseCultureMemory ABC
    └── culture.py                 # CultureMemory (cross-agent shared knowledge)
```

The remainder of this document treats both trees as a unit because
that is how the public API behaves.

---

## 3. Top-level files

### 3.1 `src/upsonic/memory/__init__.py` — public lazy shim

A single-purpose module with two responsibilities:

| Responsibility | Implementation |
| --- | --- |
| Type-checker visibility | `if TYPE_CHECKING:` branch importing the real symbols so editors / `mypy` see them. |
| Runtime lazy resolution | `__getattr__(name)` calls `_get_memory_functions()` which imports `save_agent_memory`, `get_agent_memory`, `reset_agent_memory` from the local `.memory` module and `Memory` from `..storage.memory.memory`. |

```python
# src/upsonic/memory/__init__.py (key parts)
def _get_memory_functions():
    from .memory import save_agent_memory, get_agent_memory, reset_agent_memory
    from ..storage.memory.memory import Memory
    return {
        'save_agent_memory': save_agent_memory,
        'get_agent_memory': get_agent_memory,
        'reset_agent_memory': reset_agent_memory,
        'Memory': Memory,
    }

def __getattr__(name: str) -> Any:
    memory_functions = _get_memory_functions()
    if name in memory_functions:
        return memory_functions[name]
    raise AttributeError(...)

__all__ = ['save_agent_memory', 'get_agent_memory', 'reset_agent_memory', 'Memory']
```

Effects of this design:

- `import upsonic.memory` is cheap.
- `from upsonic.memory import Memory` triggers the heavy import of the
  orchestrator only on first access.
- Anyone misspelling a name gets a helpful `AttributeError` directing
  them to "import from the appropriate sub-module".

### 3.2 `src/upsonic/memory/memory.py` — file-based scratchpad memory

The local `memory.py` is **independent** of the `Memory` orchestrator.
It implements three top-level functions that read/write a single JSON
file per agent identity:

| Function | Signature | Behaviour |
| --- | --- | --- |
| `save_agent_memory` | `(agent, answer)` | Calls `answer.all_messages()`, serialises via `ModelMessagesTypeAdapter.dump_python(history, mode='json')`, writes to `{sha256(agent_id)}.json` next to this file. Silently swallows `OSError` and `UnicodeEncodeError`. |
| `get_agent_memory` | `(agent)` | Reads the file, validates with `ModelMessagesTypeAdapter.validate_python(json_data)`. Returns `[]` on missing file, malformed JSON, or list-shape mismatch. |
| `reset_agent_memory` | `(agent)` | `agent_file.unlink()` if exists, swallows `OSError`. |
| `_get_agent_file_path` | `(agent_id) -> Path` | `MEMORY_DIR / f"{sha256(agent_id).hexdigest()}.json"`. |

```python
# src/upsonic/memory/memory.py
import json, hashlib
from pathlib import Path
from upsonic.messages.messages import ModelMessagesTypeAdapter

MEMORY_DIR = Path(__file__).parent

def _get_agent_file_path(agent_id: str) -> Path:
    agent_hash = hashlib.sha256(agent_id.encode('utf-8')).hexdigest()
    return MEMORY_DIR / f"{agent_hash}.json"

def save_agent_memory(agent, answer):
    history = answer.all_messages()
    json_data = ModelMessagesTypeAdapter.dump_python(history, mode='json')
    agent_file = _get_agent_file_path(agent.get_agent_id())
    try:
        with open(agent_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    except (OSError, UnicodeEncodeError):
        pass
```

#### Notes / caveats

- The functions take an *agent-like* object that exposes
  `get_agent_id()` — they make no other assumptions about the agent.
- **Bytes are encoded as base64** by the pydantic type adapter
  (`mode='json'`), which keeps binary content (e.g. images, audio, raw
  PDFs in `BinaryContent` / `BinaryImage`) round-trippable.
- Files are co-located with the source. If `src/upsonic/memory/` lives
  inside an installed wheel, write attempts will silently fail in
  read-only environments — the `try/except` is intentional.
- This module is **not** wired into the modern `Agent` pipeline. Modern
  agents use the `Memory` orchestrator + a `Storage` backend. The
  scratchpad remains for legacy/test code paths and direct user calls.

### 3.3 The `Memory` class (re-exported from `storage/memory/memory.py`)

`Memory` is the **central coordinator** for an agent's memory features.
Located at `src/upsonic/storage/memory/memory.py` and re-exported by
both `src/upsonic/memory/__init__.py` and
`src/upsonic/storage/memory/__init__.py`.

#### 3.3.1 Constructor parameters

| Parameter | Type | Role |
| --- | --- | --- |
| `storage` | `Storage` (sync) or `AsyncStorage` | Required backend. The orchestrator detects async via `is_async_storage_backend`. |
| `session_id` | `Optional[str]` | Auto-`uuid4()` if `None`. |
| `user_id` | `Optional[str]` | Auto-`uuid4()` if `None`. |
| `full_session_memory` | `bool=False` | **Save flag** — persist chat history. |
| `summary_memory` | `bool=False` | **Save flag** — generate & persist summaries. |
| `user_analysis_memory` | `bool=False` | **Save flag** — extract & persist user profile. |
| `load_full_session_memory` | `Optional[bool]` | **Load flag** — inject history. Defaults to save flag. |
| `load_summary_memory` | `Optional[bool]` | **Load flag** — inject summary. Defaults to save flag. |
| `load_user_analysis_memory` | `Optional[bool]` | **Load flag** — inject profile. Defaults to save flag. |
| `user_profile_schema` | `Optional[Type[BaseModel]]` | Pydantic schema for the profile (defaults to `upsonic.schemas.UserTraits` when not dynamic). |
| `dynamic_user_profile` | `bool=False` | Build profile schema on-the-fly from the conversation. |
| `num_last_messages` | `Optional[int]` | Cap on retained turns when injecting history. |
| `model` | `Optional[Union[Model, str]]` | LLM used for summary + profile sub-agents. **Required** if either is enabled. |
| `debug` | `bool=False` | Verbose info logs. |
| `debug_level` | `int=1` | 1–3 verbosity. |
| `feed_tool_call_results` | `bool=False` | Include tool-call/return parts in injected history. |
| `user_memory_mode` | `Literal['update','replace']='update'` | Profile merge strategy. |

#### 3.3.2 Save vs Load flag separation

This is the most important design point of the orchestrator. Save and
load flags are independent so callers can configure asymmetric
behaviour:

| Use case | Save flags | Load flags |
| --- | --- | --- |
| Full memory (default) | all `True` | mirror save (default) |
| Save everything, inject only summary | all `True` | only `load_summary_memory=True` |
| Stateless run, load only profile | all `False` | `load_user_analysis_memory=True` |
| Read-only audit | all `False` | all `True` |

When a load flag is `None`, it inherits the corresponding save flag —
preserving prior behaviour for users that never set load flags.

#### 3.3.3 Caching & lazy creation

```python
self._session_memory_cache: Dict[SessionType, BaseSessionMemory] = {}
self._user_memory: Optional[BaseUserMemory] = None
if user_analysis_memory or self.load_user_analysis_memory_enabled:
    self._user_memory = self._create_user_memory()
```

- **Session memory** is created **on demand** per `SessionType` via the
  `SessionMemoryFactory` and cached for the lifetime of the `Memory`
  instance (see §4.1).
- **User memory** is created eagerly **iff** save *or* load is enabled
  (it's tied to `user_id`, not `SessionType`).

#### 3.3.4 Public method index

| Method | Async | Purpose |
| --- | --- | --- |
| `prepare_inputs_for_task` | yes | Gather data **before** a run. |
| `run_memory_agents_async` | yes | Run summary + user-profile sub-agents (no persistence). |
| `persist_session_async` | yes | Flush prepared session to storage. |
| `save_session_async` | yes | Backwards-compatible wrapper for the two above. |
| `get_session_async` / `get_session` | both | Fetch the current `Session` from storage. |
| `get_messages_async` / `get_messages` | both | Convenience: `session.messages or []`. |
| `set_metadata_async` / `set_metadata` | both | Patch `session.metadata` then upsert. |
| `get_metadata_async` / `get_metadata` | both | Read-only metadata access. |
| `list_sessions_async` / `list_sessions` | both | List sessions for `user_id`. |
| `find_session_async` / `find_session` | both | Lookup by `session_id`. |
| `delete_session_async` / `delete_session` | both | Hard delete via storage. |
| `load_resumable_run_async` / `load_resumable_run` | both | HITL: paused/error/cancelled runs only. |
| `load_run_async` / `load_run` | both | Load any run regardless of status. |
| `get_session_memory(session_type)` | sync | Lazily build per-`SessionType` `BaseSessionMemory`. |

#### 3.3.5 `prepare_inputs_for_task`

Returns a single dict shaped exactly like what the agent pipeline
expects to inject into the next LLM call:

```python
{
    "message_history":         List[Any],   # past messages (load-flag-gated)
    "context_injection":       str,         # "<SessionSummary>...</SessionSummary>"
    "system_prompt_injection": str,         # "<UserProfile>...</UserProfile>"
    "metadata_injection":      str,         # "<AgentMetadata>...</AgentMetadata>" + "<SessionMetadata>...</SessionMetadata>"
}
```

Logic flow:

1. Default `session_type = SessionType.AGENT`.
2. `get_session_memory(session_type)` → `aget()` returns
   `PreparedSessionInputs` (history + summary + session metadata).
3. If `_user_memory` exists → `aget()` formats `<UserProfile>...</UserProfile>`.
4. If `agent_metadata` was passed in → it is wrapped as
   `<AgentMetadata>...</AgentMetadata>` and **prepended** to any session
   metadata.
5. Each step is wrapped in `try/except` and degrades gracefully via
   `warning_log`.
6. Optional debug-level-2 logging samples the last three messages.

#### 3.3.6 `save_session_async` (and the split helpers)

```python
async def save_session_async(self, output, session_type=None, agent_id=None):
    await self.run_memory_agents_async(output, session_type, agent_id)
    await self.persist_session_async(output, session_type, agent_id)
```

The split exists because **memory sub-agents make LLM calls that should
be billed against the parent task's `model_execution_time`**. The
pipeline calls `run_memory_agents_async` *before* `task.task_end()`
and `persist_session_async` *after*, so storage receives the finalised
duration.

| Run status | What happens in `run_memory_agents_async` | What happens in `persist_session_async` |
| --- | --- | --- |
| `completed` | (1) load/create session, (2) `_save_completed_run` populates session, optionally generates summary via sub-agent, optionally appends new messages, (3) user-memory `asave` runs trait extraction. | Updates `session.usage`, sets `updated_at`, calls `storage.aupsert_session` / `upsert_session`. |
| `paused` / `error` / `cancelled` | `_save_incomplete_run` always runs — checkpoint state is **always** preserved for HITL resumption regardless of save flags. | Same as above; persists checkpoint. |

The stash variable `self._pending_session` carries the prepared
`AgentSession` between the two calls.

---

## 4. Subfolders walked through

### 4.1 `storage/memory/strategy/` — `BaseMemoryStrategy`

Common ABC for every strategy (session, user, culture). Stores the
shared config and forces four entry points:

```python
class BaseMemoryStrategy(ABC):
    def __init__(self, storage, enabled=True, model=None, debug=False, debug_level=1): ...
    @abstractmethod
    async def aget(self, *args, **kwargs): ...
    @abstractmethod
    async def asave(self, *args, **kwargs): ...
    @abstractmethod
    def get(self, *args, **kwargs): ...
    @abstractmethod
    def save(self, *args, **kwargs): ...
```

Subclasses are expected to detect async vs sync storage backends with
`is_async_storage_backend` and bridge through `run_awaitable_sync` from
sync entry points when the backend is async.

### 4.2 `storage/memory/session/` — session memory strategies

#### `PreparedSessionInputs` (dataclass, `session/base.py`)

| Field | Type | Description |
| --- | --- | --- |
| `message_history` | `List[Any]` | Limited / filtered chat history. |
| `context_injection` | `str` | `"<SessionSummary>…</SessionSummary>"` if summary loaded. |
| `metadata_injection` | `str` | `"<SessionMetadata>…</SessionMetadata>"` if any. |
| `session` | `Optional[Any]` | Raw `AgentSession` for downstream use. |

#### `BaseSessionMemory` (`session/base.py`)

Adds session-specific config and abstract loaders:

```python
class BaseSessionMemory(BaseMemoryStrategy, ABC):
    session_type: SessionType  # set on subclass

    def __init__(self, storage, session_id, enabled=True, summary_enabled=False,
                 load_enabled=None, load_summary_enabled=None,
                 num_last_messages=None, feed_tool_call_results=False,
                 model=None, debug=False, debug_level=1): ...

    @abstractmethod async def aload_resumable_run(self, run_id, agent_id=None): ...
    @abstractmethod async def aload_run(self, run_id, agent_id=None): ...
    @abstractmethod def load_resumable_run(self, run_id, agent_id=None): ...
    @abstractmethod def load_run(self, run_id, agent_id=None): ...
```

#### `AgentSessionMemory` (`session/agent.py`)

The only registered session memory implementation. Responsibilities:

- **Build `PreparedSessionInputs`** from a fetched `AgentSession`
  (`_build_prepared_inputs_from_session`):
  - inject `<SessionSummary>` if `load_summary_enabled` and present;
  - load + limit + filter messages if `load_enabled`;
  - emit `<SessionMetadata>` if `session.metadata` non-empty.
- **Save runs** with full async/sync parity:
  - `arun_agents` populates the session, runs the summary sub-agent
    (only on completed runs, only if `summary_enabled`), and stashes
    the prepared session in `self._pending_session`.
  - `apersist` finalises usage, sets `updated_at`, and calls
    `aupsert_session` / `upsert_session`.
  - `asave` is the back-compat wrapper that calls the two in sequence.
  - `save` (sync) inlines the same logic against a sync storage; if
    given an async storage it bridges through `run_awaitable_sync`.
- **Load resumable runs** via `aload_resumable_run` / `load_resumable_run`
  filtered to `RunStatus.{paused, error, cancelled}`. Falls back to
  scanning *all* sessions for the agent when the current `session_id`
  doesn't contain the `run_id`.
- **Load any run** via `aload_run` / `load_run` (status-agnostic).
- **History limiting** in `_limit_message_history`:
  - groups raw messages into request/response pairs;
  - keeps the last `num_last_messages` runs;
  - **always preserves the original `SystemPromptPart`** by copying it
    onto the first kept request, so the system prompt is never lost
    when the conversation is truncated.
- **Tool filtering** in `_filter_tool_messages` removes
  `part_kind == 'tool-return'` from `ModelRequest` and
  `part_kind == 'tool-call'` from `ModelResponse` when
  `feed_tool_call_results` is `False`.
- **Summary generation** in `_generate_summary`:
  - instantiates a tiny `Agent(name="Summarizer", model=self.model)`;
  - prompt is built from the previous summary, the new turn (extracted
    via `output.new_messages()` and serialised by
    `ModelMessagesTypeAdapter.dump_python(..., mode='json')`), and the
    full session message history (fetched with
    `AgentSession.get_all_messages_for_session_id_async`);
  - the response format is `str` via a `Task`;
  - sub-agent usage is captured in `self._last_llm_usage` and folded
    back into the parent `output.usage` via `usage.incr(...)` before
    persistence.

Failure handling: every save path catches `Exception`, logs a
truncated traceback (`error_trace[-500:]`), and ensures the
`_pending_session` is reset.

#### HITL invariant

Comments in `agent.py` are explicit:

> **CRITICAL:** For HITL to work, incomplete runs (paused, error,
> cancelled) MUST ALWAYS be saved, regardless of `enabled` setting.

Cross-process resumption depends on this: a `paused` run from a tool
call must be retrievable even when `full_session_memory=False`.

### 4.3 `storage/memory/user/` — user-profile memory

#### `BaseUserMemory` (`user/base.py`)

Adds user-specific config to the strategy ABC:

| Field | Default | Meaning |
| --- | --- | --- |
| `user_id` | required | Storage key. |
| `enabled` | `True` | Save flag. |
| `load_enabled` | mirrors `enabled` | Load flag. |
| `profile_schema` | `None` | Pydantic schema for traits. |
| `dynamic_profile` | `False` | Generate schema from the conversation itself. |
| `update_mode` | `'update'` | `'update'` merges, `'replace'` overwrites. |

#### `UserMemory` (`user/user.py`)

The only built-in user memory. Notable behaviour:

- **Schema selection**:

  ```python
  if self.dynamic_profile:
      if profile_schema:
          warning_log("`dynamic_user_profile` is True, so the provided "
                      "`user_profile_schema` will be ignored.", "UserMemory")
      self._profile_schema_model = None
  else:
      self._profile_schema_model = profile_schema or UserTraits  # from upsonic.schemas
  ```

- **`aget` / `get`** call `storage.aget_user_memory` /
  `get_user_memory`, format the dict via `_format_profile_data`, wrap
  in `<UserProfile>…</UserProfile>` tags, and skip entirely when
  `load_enabled=False`.

- **`_format_profile_data`** is defensive: drops `None`, empty strings,
  empty lists, and empty dicts. Lists become comma-joined strings,
  dicts become JSON.

- **`asave` flow**:

  1. Fetch the current profile from storage (`get_user_memory`).
  2. Call `_analyze_interaction_for_traits(output, current_profile)`:
     - pulls **all historical user prompts across all sessions for
       this `user_id`** via
       `AgentSession.get_all_user_prompt_messages_for_user_id_async`,
       excluding the current `run_id`;
     - extracts new prompts from `output.new_messages()` via
       `AgentSession._extract_user_prompts_from_messages`;
     - if `dynamic_profile`: runs **two sub-agent passes** —
       (a) propose 2-5 fields with `ProposedSchema(fields=...)`,
       (b) populate a `create_model('DynamicUserTraitModel', ...)` with
       `Optional[str]` fields described by the schema;
     - else: a single pass using `self._profile_schema_model`.
  3. Aggregate sub-agent `usage` into the parent's `output.usage` via
     `output._ensure_usage().incr(self._last_llm_usage)`.
  4. Merge or replace based on `update_mode`.
  5. `storage.aupsert_user_memory(UserMemory(user_id=..., user_memory=final_profile, agent_id=..., team_id=...), deserialize=True)`.

- **`save` (sync)** delegates everything through `run_awaitable_sync`
  because trait analysis is inherently async (it talks to an LLM).

### 4.4 `storage/memory/culture/` — shared cross-agent culture

#### `BaseCultureMemory` (`culture/base.py`)

A different shape from session/user: culture is **not tied to any
single user or session**. It exposes list and delete instead of a
strict get/save pair:

```python
class BaseCultureMemory(BaseMemoryStrategy, ABC):
    @abstractmethod async def aget_all(self, agent_id=None, team_id=None,
                                       categories=None, limit=None) -> List[CulturalKnowledge]: ...
    @abstractmethod async def adelete(self, culture_id: str) -> bool: ...
    @abstractmethod def get_all(...): ...
    @abstractmethod def delete(...): ...
```

`CultureMemory` (in `culture/culture.py`) implements the contract
against `Storage.get_cultural_knowledge` etc. — culture memory is not
created automatically by the `Memory` orchestrator constructor; it
must be wired in deliberately by callers that want shared cultural
knowledge injected into agents.

### 4.5 `storage/memory/factory.py` — `SessionMemoryFactory`

A **registry-based factory** keyed by `SessionType`:

```python
class SessionMemoryFactory:
    _registry: Dict[SessionType, Type[BaseSessionMemory]] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, session_type, memory_class): ...
    @classmethod
    def create(cls, session_type, storage, session_id, ...) -> BaseSessionMemory: ...
    @classmethod
    def get_supported_types(cls) -> List[SessionType]: ...
    @classmethod
    def is_supported(cls, session_type) -> bool: ...
    @classmethod
    def clear_registry(cls) -> None:  # for tests
        cls._registry.clear()
        cls._initialized = False
```

`_ensure_initialized` lazily registers `AgentSessionMemory` for
`SessionType.AGENT`. Adding TEAM/WORKFLOW support is purely additive —
write the subclass and call `SessionMemoryFactory.register(...)`.

### 4.6 `storage/memory/storage_dispatch.py` — sync↔async bridge

Two tiny helpers that the entire memory subsystem leans on:

```python
def is_async_storage_backend(storage) -> bool:
    from upsonic.storage.base import AsyncStorage
    return isinstance(storage, AsyncStorage)

def run_awaitable_sync(awaitable):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, awaitable).result()
```

The pattern across every strategy is:

```python
def get(self):
    if is_async_storage_backend(self.storage):
        return run_awaitable_sync(self._aget_prepared_async_storage())
    # ... pure sync path against Storage
```

This keeps the **public sync API** (`get`, `save`, `load_run`, …)
usable even when the backend is `AsyncStorage` (Postgres async, Mongo
async, async SQLite, async mem0).

---

## 5. Cross-file relationships (interaction with `/storage` and `/session`)

```
            ┌──────────────────────────────────┐
            │   src/upsonic/memory/            │
            │   (lazy public shim)             │
            │   - save_agent_memory            │
            │   - get_agent_memory             │
            │   - reset_agent_memory           │
            │   - Memory  (re-export)          │
            └──────────────┬───────────────────┘
                           │ re-exports
                           ▼
            ┌──────────────────────────────────┐
            │  src/upsonic/storage/memory/     │   orchestration layer
            │  ┌────────────────────────────┐  │
            │  │ Memory (orchestrator)      │  │
            │  │  ├─ SessionMemoryFactory   │  │
            │  │  │   └─ AgentSessionMemory │  │
            │  │  └─ UserMemory             │  │
            │  └────────────────────────────┘  │
            └──────────────┬───────────────────┘
            uses Storage   │   uses Session schema
                           ▼
            ┌──────────────────────────────────┐
            │  src/upsonic/storage/  (backend) │
            │  - Storage / AsyncStorage ABC    │
            │  - SQLite / Postgres / Redis /   │
            │    Mongo / Mem0 / JSON / InMem   │
            │  - schemas.UserMemory            │
            └──────────────────────────────────┘
                           ▲
            ┌──────────────┴───────────────────┐
            │   src/upsonic/session/           │
            │   - SessionType (AGENT/TEAM/WF)  │
            │   - AgentSession                 │
            │   - RunData                      │
            └──────────────────────────────────┘
```

Concrete touchpoints:

| Memory module | Calls into | Methods used |
| --- | --- | --- |
| `Memory.get_session_async` | `Storage` | `aget_session(session_id, session_type, deserialize=True)` |
| `Memory.set_metadata_async` | `Storage` | `aupsert_session(session, deserialize=True)` |
| `Memory.list_sessions_async` | `Storage` | `aget_sessions(user_id=..., session_type=...)` |
| `Memory.delete_session_async` | `Storage` | `adelete_session(session_id)` |
| `AgentSessionMemory._load_or_create_session` | `Storage` + `session/agent.py` | `aget_session` / `get_session` then `AgentSession(...)` |
| `AgentSessionMemory.apersist` | `Storage` | `aupsert_session` / `upsert_session` |
| `AgentSessionMemory._generate_summary` | `session/agent.py` | `AgentSession.get_all_messages_for_session_id_async` |
| `UserMemory.aget` | `Storage` | `aget_user_memory(user_id, agent_id, team_id, deserialize=True)` |
| `UserMemory.asave` | `Storage` + `storage/schemas.py` | `aupsert_user_memory(UserMemory(user_id, user_memory, agent_id, team_id))` |
| `UserMemory._analyze_interaction_for_traits` | `session/agent.py` | `AgentSession.get_all_user_prompt_messages_for_user_id_async`, `_extract_user_prompts_from_messages` |
| `_limit_message_history` / `_filter_tool_messages` | `messages/messages.py` | `ModelRequest`, `ModelResponse`, `SystemPromptPart`, `UserPromptPart`, `part_kind` |

Key decoupling rules:

- **Memory never imports a concrete backend.** It only programs
  against `Storage` / `AsyncStorage`.
- **Memory never imports an Agent.** It imports `Agent` and `Task`
  *inside* `_generate_summary` / `_analyze_interaction_for_traits` to
  avoid circular imports — these are runtime imports.
- **Session module is a pure data layer.** It defines `SessionType`,
  `AgentSession`, `RunData` and helpers like
  `get_all_messages_for_session_id_async`. Memory consumes those
  helpers but never modifies the schema directly.

---

## 6. Public API

The full surface re-exported by `src/upsonic/memory/__init__.py`:

```python
from upsonic.memory import (
    save_agent_memory,    # legacy file scratchpad
    get_agent_memory,     # legacy file scratchpad
    reset_agent_memory,   # legacy file scratchpad
    Memory,               # orchestrator (preferred)
)
```

The full surface re-exported by `src/upsonic/storage/memory/__init__.py`:

```python
from upsonic.storage.memory import (
    Memory,
    SessionMemoryFactory,
    BaseMemoryStrategy,
    BaseSessionMemory,
    PreparedSessionInputs,
    AgentSessionMemory,
    BaseUserMemory,
    UserMemory,
    BaseCultureMemory,
    CultureMemory,
)
```

### 6.1 Cookbook

#### a) Construct a memory with full session + summary + profile

```python
from upsonic.memory import Memory
from upsonic.storage.sqlite.sqlite import SqliteStorage

storage = SqliteStorage(db_file="agents.db")

memory = Memory(
    storage=storage,
    session_id="session_001",
    user_id="user_123",
    full_session_memory=True,
    summary_memory=True,
    user_analysis_memory=True,
    model="openai/gpt-4o",
    num_last_messages=10,
    feed_tool_call_results=False,
)
```

#### b) Save everything but inject only the summary (token-budget mode)

```python
memory = Memory(
    storage=storage,
    full_session_memory=True,
    summary_memory=True,
    user_analysis_memory=True,
    load_full_session_memory=False,
    load_summary_memory=True,
    load_user_analysis_memory=True,
    model="openai/gpt-4o",
)
```

#### c) Manually drive prepare → save in tests

```python
prepared = await memory.prepare_inputs_for_task(
    session_type=SessionType.AGENT,
    agent_metadata={"agent_name": "Alice"},
)
# ... feed prepared into the model call ...
await memory.save_session_async(output)  # output is the AgentRunOutput
```

#### d) HITL resumption

```python
run_data = await memory.load_resumable_run_async(
    run_id="run_xyz",
    session_type=SessionType.AGENT,
    agent_id=agent.agent_id,
)
if run_data:
    # Resumable run found (status in {paused, error, cancelled}).
    ...
```

#### e) Legacy scratchpad

```python
from upsonic.memory import save_agent_memory, get_agent_memory, reset_agent_memory

save_agent_memory(agent, run_output)
history = get_agent_memory(agent)
reset_agent_memory(agent)
```

The scratchpad does **not** participate in `Memory`. Use it only when
you want a self-contained, dependency-free file artifact for a single
agent identity.

---

## 7. Integration with the rest of Upsonic (`Agent.memory`)

The `Memory` orchestrator is wired into `Agent` (and through it, into
`Team` / `Workflow`) via two seams:

### 7.1 Construction in `src/upsonic/agent/agent.py`

```python
class Agent:
    def __init__(self, ..., memory: Optional["Memory"] = None, db: Optional[Storage] = None, ...):
        ...
        if db is not None:
            self.memory = db.memory   # db carries a Memory bound to itself
        else:
            self.memory = memory
        ...
        if self.memory and feed_tool_call_results is not None:
            self.memory.feed_tool_call_results = feed_tool_call_results
```

The agent exposes convenience accessors that fall back to memory:

```python
def _get_session_id(self):
    if self.memory and hasattr(self.memory, 'session_id'):
        return self.memory.session_id
    if self.db and hasattr(self.db, 'memory'):
        return self.db.memory.session_id
    ...

def _get_user_id(self):
    # symmetric
```

`Agent.cost_in_session` / `cost_in_session_async` similarly call
`self.memory.get_session()` and aggregate per-task usage.

### 7.2 The `MemoryManager` context manager

`src/upsonic/agent/context_managers/memory_manager.py` is the
**pipeline-side wrapper** that the agent's run pipeline uses:

```python
class MemoryManager:
    async def aprepare(self) -> None:
        if self.memory:
            self._prepared_inputs = await self.memory.prepare_inputs_for_task(
                agent_metadata=self.agent_metadata
            )
            # ... agent_metadata override path ...

    async def afinalize(self) -> None:
        if self.memory and self._agent_run_output:
            await self.memory.save_session_async(output=self._agent_run_output)
```

It exposes accessors used by other context managers:

| Accessor | Consumed by |
| --- | --- |
| `get_message_history()` | the agent's run loop (history feed) |
| `get_context_injection()` | `ContextManager` (summary into user prompt) |
| `get_system_prompt_injection()` | `SystemPromptManager` (profile into system prompt) |
| `get_metadata_injection()` | `ContextManager` (metadata into user prompt) |

### 7.3 Pipeline steps

The modern pipeline (`src/upsonic/agent/pipeline/steps.py` and
`pipeline/manager.py`) explicitly splits the save into two calls so
that LLM-based sub-agent time is billed correctly:

```python
# pipeline/steps.py (simplified)
await agent.memory.run_memory_agents_async(output, ...)   # before task.task_end()
task.task_end()                                            # finalises usage.duration
await agent.memory.persist_session_async(output, ...)      # after task.task_end()
```

For pipelines that don't need the split (e.g. error/cancel paths,
`pipeline/manager.py`), the legacy single-call wrapper
`save_session_async` is used. Streaming runs follow the same pattern
in `pipeline/steps.py`.

### 7.4 Resumable / completed lookups in `Agent`

```python
async def load_paused_run(self, run_id):
    if not self.memory:
        raise ValueError("No memory configured. ...")
    return await self.memory.load_resumable_run_async(
        run_id=run_id,
        session_type=self.session_type,
        agent_id=self.agent_id,
    )
```

The agent's pipeline also calls `memory.load_run_async(...)` when it
needs to discriminate between a completed and an in-memory output.

---

## 8. End-to-end flow: `prepare_inputs` → `save_session`

The diagram below traces a single agent invocation that uses memory
(annotations `[storage]` mean an actual storage call, `[llm]` means an
LLM call originating in a memory sub-agent).

```
Agent.do_async(task)
│
├─ pipeline step: MemoryPrepareStep
│   └─ MemoryManager.aprepare()
│       └─ Memory.prepare_inputs_for_task()
│           ├─ session_memory = get_session_memory(SessionType.AGENT)
│           │   └─ SessionMemoryFactory.create(...)  → AgentSessionMemory (cached)
│           ├─ session_inputs = await session_memory.aget()
│           │   └─ [storage] aget_session(session_id, SessionType.AGENT, deserialize=True)
│           │   └─ _build_prepared_inputs_from_session(session)
│           │       ├─ load_summary_enabled → "<SessionSummary>...</SessionSummary>"
│           │       ├─ load_enabled → _limit_message_history → _filter_tool_messages
│           │       └─ session.metadata → "<SessionMetadata>...</SessionMetadata>"
│           ├─ profile_str = await _user_memory.aget()
│           │   └─ [storage] aget_user_memory(user_id, agent_id, team_id, deserialize=True)
│           │   └─ _format_profile_data → "<UserProfile>...</UserProfile>"
│           └─ if agent_metadata → prepend "<AgentMetadata>...</AgentMetadata>"
│       returns: { message_history, context_injection,
│                  system_prompt_injection, metadata_injection }
│
├─ pipeline injects:
│   ├─ message_history          → model request history
│   ├─ system_prompt_injection  → SystemPromptManager
│   ├─ context_injection        → ContextManager (user prompt)
│   └─ metadata_injection       → ContextManager (user prompt)
│
├─ LLM call produces AgentRunOutput
│
├─ pipeline step: MemoryRunAgentsStep (BEFORE task.task_end())
│   └─ Memory.run_memory_agents_async(output)
│       ├─ user_memory.asave(output, agent_id) if completed
│       │   ├─ [storage] aget_user_memory(...)
│       │   ├─ _analyze_interaction_for_traits(output, current_profile)
│       │   │   ├─ [storage] AgentSession.get_all_user_prompt_messages_for_user_id_async(
│       │   │   │           storage, user_id, exclude_run_id=current_run_id)
│       │   │   ├─ output.new_messages() → _extract_user_prompts_from_messages
│       │   │   ├─ if dynamic_profile:
│       │   │   │     [llm] sub-agent #1 → ProposedSchema(fields=[...])
│       │   │   │     [llm] sub-agent #2 → DynamicUserTraitModel(...)
│       │   │   │   else:
│       │   │   │     [llm] sub-agent → self._profile_schema_model
│       │   │   └─ output.usage.incr(_last_llm_usage)
│       │   ├─ merge or replace based on update_mode
│       │   └─ [storage] aupsert_user_memory(UserMemory(user_id, user_memory, ...))
│       └─ session_memory.arun_agents(output, is_completed)
│           ├─ _load_or_create_session
│           │   └─ [storage] aget_session(...)
│           ├─ if completed:
│           │     _save_completed_run(session, output)
│           │       ├─ session.populate_from_run_output(output)
│           │       ├─ session.upsert_run(output)
│           │       ├─ if summary_enabled and model:
│           │       │     [llm] _generate_summary(session, output)
│           │       │       ├─ output.new_messages() → ModelMessagesTypeAdapter
│           │       │       ├─ [storage] AgentSession.get_all_messages_for_session_id_async
│           │       │       ├─ Agent("Summarizer", model).do_async(Task(prompt, str))
│           │       │       └─ self._last_llm_usage.incr(summary_output.usage)
│           │       ├─ output.usage.incr(self._last_llm_usage)  # bills parent
│           │       └─ if enabled: session.append_new_messages_from_run_output(output)
│           │   else (paused/error/cancelled):
│           │     _save_incomplete_run(session, output)  # ALWAYS saved (HITL)
│           └─ self._pending_session = session
│
├─ task.task_end()   ← finalises output.usage.duration
│
└─ pipeline step: MemoryPersistStep (AFTER task.task_end())
    └─ Memory.persist_session_async(output)
        └─ session_memory.apersist(output, is_completed)
            ├─ session.update_usage_from_run(output)
            ├─ session.updated_at = int(time.time())
            └─ [storage] aupsert_session(session, deserialize=True)
```

### 8.1 Why split `run_memory_agents_async` from `persist_session_async`?

Memory sub-agents (summarizer, user trait analyzer) make real LLM
calls. Their wall-clock time is part of the user-visible "task time".
The pipeline therefore:

1. Runs sub-agents while the parent task timer is **still ticking**,
   so `usage.model_execution_time` keeps accruing.
2. Calls `task.task_end()` to freeze `usage.duration`.
3. Persists with the **finalised** usage.

If the persistence happened *first*, storage would record an
under-counted duration and the dashboards would lie. The split is the
fix.

### 8.2 Why `save_session_async` still exists

`save_session_async` is the **legacy single-call wrapper**:

```python
async def save_session_async(self, output, ...):
    await self.run_memory_agents_async(output, ...)
    await self.persist_session_async(output, ...)
```

It is still used by:

- `MemoryManager.afinalize` (top-level context-manager finalisation),
- `pipeline/manager.py` (error/cancel paths),

where the slight over-counting of duration is acceptable or where
`task_end()` is invoked outside this path.

### 8.3 HITL invariant in the flow

The `_save_incomplete_run` path is the **always-on checkpoint
mechanism**. Its commitments:

- Runs in **both async and sync** save paths.
- Ignores `enabled` and `summary_enabled` — it is unconditional.
- Calls `session.upsert_run(output)` so `RunStatus` reflects pause/error/cancel.
- Calls `session.append_new_messages_from_run_output(output)` so a
  resumed run has the messages it needs.
- Logs a `Checkpoint saved for run {run_id} at step {n} ({name})`
  message when paused at a specific step (extracted via
  `output.get_paused_step()`).

This is what makes cross-process resume (e.g. Lambda → second Lambda
via a SQL/Mongo backend) work even when the user chose
`full_session_memory=False`.

---

## 9. Quick reference

### 9.1 Files in `src/upsonic/memory/`

| File | LOC | Role |
| --- | --- | --- |
| `__init__.py` | 32 | Lazy `__getattr__` shim re-exporting `Memory` + scratchpad fns. |
| `memory.py` | 62 | File-based per-agent SHA256-keyed JSON scratchpad: `save_agent_memory`, `get_agent_memory`, `reset_agent_memory`. |

### 9.2 Files in `src/upsonic/storage/memory/`

| File | Role |
| --- | --- |
| `__init__.py` | Lazy re-exports of all memory classes. |
| `memory.py` | `Memory` orchestrator. |
| `factory.py` | `SessionMemoryFactory` registry. |
| `storage_dispatch.py` | `is_async_storage_backend`, `run_awaitable_sync`. |
| `strategy/base.py` | `BaseMemoryStrategy` ABC. |
| `session/base.py` | `BaseSessionMemory` ABC, `PreparedSessionInputs` dataclass. |
| `session/agent.py` | `AgentSessionMemory` (HITL, summaries, history limit + tool filter). |
| `user/base.py` | `BaseUserMemory` ABC. |
| `user/user.py` | `UserMemory` (trait extraction, dynamic schema, update vs replace). |
| `culture/base.py` | `BaseCultureMemory` ABC (cross-agent shared knowledge). |
| `culture/culture.py` | `CultureMemory` impl. |

### 9.3 Save/Load flag combinations cheatsheet

| `full_session_memory` | `summary_memory` | `user_analysis_memory` | `load_*` | Behaviour |
| --- | --- | --- | --- | --- |
| `False` | `False` | `False` | (all) `False` | Stateless agent. HITL still works (incomplete runs always saved). |
| `True` | `False` | `False` | mirror | Classic chat history. |
| `False` | `True` | `False` | mirror | Summary-only memory (token-cheap). |
| `True` | `True` | `True` | mirror | Full memory (history + summary + profile). |
| `True` | `True` | `True` | only `load_summary=True`, `load_user=True` | Save everything; inject only summary + profile (token-budget). |
| `False` | `False` | `True` | mirror | User profile only — across all sessions. |

### 9.4 Storage methods consumed by `Memory`

| Operation | Sync method | Async method |
| --- | --- | --- |
| Read session | `get_session(session_id, session_type, deserialize=True)` | `aget_session(...)` |
| List sessions | `get_sessions(user_id=..., agent_id=..., session_type=...)` | `aget_sessions(...)` |
| Upsert session | `upsert_session(session, deserialize=True)` | `aupsert_session(...)` |
| Delete session | `delete_session(session_id)` | `adelete_session(...)` |
| Read user memory | `get_user_memory(user_id, agent_id, team_id, deserialize=True)` | `aget_user_memory(...)` |
| Upsert user memory | `upsert_user_memory(UserMemory(...), deserialize=True)` | `aupsert_user_memory(...)` |

The orchestrator chooses sync vs async based on
`is_async_storage_backend(self.storage)` and bridges through
`run_awaitable_sync` whenever a sync entry point gets an
`AsyncStorage`.

---

## 10. Common pitfalls

1. **Importing the wrong `Memory`.** There is only one `Memory` class
   you should ever instantiate; it lives at
   `upsonic.storage.memory.memory.Memory` and is re-exported from
   `upsonic.memory`. Do **not** confuse it with
   `upsonic.storage.schemas.UserMemory` (a row schema) or with the
   legacy `save_agent_memory` family in `upsonic.memory.memory`.
2. **Forgetting `model=` when enabling summary or user analysis.** The
   sub-agents will silently skip with a `warning_log`. Set
   `model="openai/gpt-4o"` (or your provider) on the `Memory` instance.
3. **Setting `feed_tool_call_results=True` after construction without
   re-creating session memory.** The `Agent.__init__` patches
   `self.memory.feed_tool_call_results` directly, but the cached
   `AgentSessionMemory` may already have been created. The flag is
   read on each `aget`, so this works in practice — but it means
   *currently cached strategies do honour the new value* because
   `self.feed_tool_call_results` is read off the `Memory` object's
   parameter at construction time of each `BaseSessionMemory`. Verify
   ordering when you change this dynamically.
4. **HITL works without `full_session_memory`.** Even if you turned
   off save flags, paused / error / cancelled runs are persisted. The
   `enabled` flag only gates **completed-run** message persistence and
   summary generation.
5. **Dynamic profile + provided schema.** Setting both
   `dynamic_user_profile=True` and a `user_profile_schema=` will log a
   warning and **ignore the schema**.
6. **Sync API on async storage.** Calling `memory.get_session()` (sync)
   when the backend is `AsyncStorage` is supported via
   `run_awaitable_sync` but spawns a worker thread. Prefer
   `await memory.get_session_async()` from coroutine code.
7. **Scratchpad files in installed wheels.** Because
   `MEMORY_DIR = Path(__file__).parent`, the legacy scratchpad writes
   into the installed package directory. In read-only deployments
   (e.g. inside a container layer) the writes silently fail. Use the
   `Memory` orchestrator with a real `Storage` for production.
