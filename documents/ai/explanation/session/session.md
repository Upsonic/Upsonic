---
name: session-and-run-persistence
description: Use when working with conversational session state, run persistence, or the data layer that storage backends serialize. Use when a user asks to load, save, mutate, or query AgentSession objects, manage RunData entries, aggregate per-session usage, filter messages by run status, or understand how Memory and Storage interact with sessions. Trigger when the user mentions AgentSession, RunData, SessionType, session_id, run_id, upsert_run, populate_from_run_output, append_new_messages_from_run_output, get_messages, get_chat_history, get_resumable_runs, get_paused_runs, session_data, agent_data, RunUsage on a session, summary, src/upsonic/session/, session/agent.py, session/base.py, serialize_flag, deserialize_flag, HITL resumable runs, or cross-session user prompt queries.
---

# `src/upsonic/session/` — Session and Run Persistence

This document describes the `session` package in Upsonic. The package owns the
in-memory data model that represents a *conversational session* between a user
and an agent (or team / workflow), plus the per-run output payloads that get
persisted by the storage layer.

The package is intentionally tiny — only **two `.py` files** — because it is a
*pure data layer*. All side effects (database writes, serialization formats,
backend selection) live in `src/upsonic/storage/`. All business logic that
mutates a session (memory composition, message tracking, agent state) lives in
`src/upsonic/memory/` and `src/upsonic/agent/`.

The session module sits at the bottom of the stack and is consumed by
**Storage**, **Memory**, **Agent pipeline**, **Team**, **Workflow** and
**Chat**.

---

## 1. What this folder is — Session / Run Persistence

`session/` defines two things and *only* these two things:

1. **`SessionType`** — an `Enum` discriminator (`agent`, `team`, `workflow`)
   used by storage backends to know which kind of conversation they are
   loading.
2. **`AgentSession` + `RunData`** — the dataclasses that hold the full state of
   a session and each of its runs.

A *session* is the database row (or JSON blob, or Redis hash) that survives
across multiple agent invocations, so that the next call can recover prior
chat history, accumulated cost, paused human-in-the-loop runs, attached
documents and so on. A *run* is one execution of `Agent.do()` /
`Agent.do_async()` / `Agent.stream()` inside a session.

Conceptually:

```
User
 └── user_id
      └── AgentSession (session_id)
            ├── messages          : List[ModelMessage]    # flattened chat history
            ├── usage             : RunUsage              # aggregated tokens/$
            ├── session_data      : Dict                  # user_prompts, image/doc refs
            ├── agent_data        : Dict                  # agent_name, model_name
            ├── metadata          : Dict                  # arbitrary key/value
            ├── summary           : str                   # rolling summary
            └── runs              : Dict[run_id -> RunData]
                                          └── output : AgentRunOutput
                                                      ├── messages        (per-run msgs)
                                                      ├── status          (RunStatus)
                                                      ├── usage           (RunUsage)
                                                      ├── input           (AgentRunInput)
                                                      └── ...
```

The `AgentSession` dataclass is **storage-agnostic**: the same instance is
serialized to SQLite, Postgres, Mongo, Redis, JSON or held in memory. Backends
call `AgentSession.to_dict(serialize_flag=True/False)` /
`AgentSession.from_dict(...)`.

### Design highlights

| Concern                | Where it lives                               |
| ---------------------- | -------------------------------------------- |
| Session shape          | `session/agent.py::AgentSession`             |
| Per-run payload        | `session/agent.py::RunData`                  |
| Session kind selector  | `session/base.py::SessionType`               |
| Serialization          | `to_dict()` / `from_dict()` on each class    |
| Persistence I/O        | `src/upsonic/storage/` (Sync + Async)        |
| Memory composition     | `src/upsonic/memory/memory.py`               |
| New-message tracking   | `AgentRunOutput.new_messages()` (run module) |

`AgentRunOutput` is the *single source of truth* for everything that happened
during one run. `RunData` is just a thin wrapper that lets the session evolve
its serialization format without touching all callers.

---

## 2. Folder layout (tree)

```text
src/upsonic/session/
├── __pycache__/                # bytecode cache (ignored)
├── agent.py                    # AgentSession + RunData dataclasses
└── base.py                     # SessionType enum + Session type alias
```

That's the whole package — there are **no subfolders** beyond `__pycache__`
and **no `__init__.py`**. Consumers import directly:

```python
from upsonic.session.base  import SessionType, Session
from upsonic.session.agent import AgentSession, RunData
```

---

## 3. Top-level files

### 3.1 `base.py` — `SessionType` and the `Session` alias

`base.py` is small (≈80 lines) but load-bearing. It defines a string-backed
enum that every storage backend uses to filter rows.

| Symbol        | Kind         | Purpose                                                         |
| ------------- | ------------ | --------------------------------------------------------------- |
| `SessionType` | `str`-`Enum` | Discriminates `agent` / `team` / `workflow` rows.               |
| `Session`     | `Union`      | Currently `Union["AgentSession"]` — placeholder for future kinds. |

```python
class SessionType(str, Enum):
    AGENT    = "agent"
    TEAM     = "team"
    WORKFLOW = "workflow"
```

Important behaviors:

* `SessionType.to_dict()` returns `{"session_type": self.value}`. This is the
  canonical wire format embedded inside `AgentSession.to_dict()`.
* `SessionType.from_dict(data)` is **lenient**. It accepts:
    * `None`                        → falls back to `AGENT`
    * a bare string (`"team"`)      → parses
    * a dict (`{"session_type": x}`)→ parses
    * an unknown value              → falls back to `AGENT` (does **not** raise)
* `SessionType.from_string(value)` is the **strict** counterpart and raises
  `ValueError` listing valid types if the input is wrong.

The leniency on `from_dict` is intentional: storage rows that pre-date a
schema change should still load as agent sessions rather than crash.

### 3.2 `agent.py` — `AgentSession` + `RunData`

`agent.py` is the heart of the package (~1045 lines, but most of it is helper
methods). It defines two dataclasses.

#### 3.2.1 `RunData`

```python
@dataclass
class RunData:
    output: AgentRunOutput
    deserialize_flag: bool = False
```

`RunData` is intentionally a one-field wrapper. Historically it carried more
context (paused-execution snapshots, etc.); today `AgentRunOutput` holds all
of that, and `RunData` is kept for forward compatibility and storage symmetry.

| Method                                              | Purpose                                                                                       |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `to_dict(serialize_flag=False)`                     | Returns `{"output": <AgentRunOutput.to_dict>}`. `serialize_flag` toggles cloudpickle for `Task`. |
| `from_dict(data, deserialize_flag=False)`           | Reconstructs a `RunData`. Accepts either a serialized dict or an already-built `AgentRunOutput`. |

#### 3.2.2 `AgentSession`

```python
@dataclass
class AgentSession:
    session_id:    str
    agent_id:      Optional[str]      = None
    team_id:       Optional[str]      = None
    workflow_id:   Optional[str]      = None
    user_id:       Optional[str]      = None

    session_type:  SessionType        = SessionType.AGENT

    session_data:  Optional[Dict]     = None   # user_prompts, image/doc identifiers
    metadata:      Optional[Dict]     = None   # arbitrary user-supplied k/v
    agent_data:    Optional[Dict]     = None   # agent_name, model_name

    runs:          Optional[Dict[str, RunData]] = field(default_factory=dict)
    summary:       Optional[str]      = None
    messages:      Optional[List[ModelMessage]] = field(default_factory=list)
    usage:         Optional[RunUsage] = None

    created_at:    Optional[int]      = None
    updated_at:    Optional[int]      = None
```

##### Field semantics

| Field           | Filled by                           | Notes                                                                                                                |
| --------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `session_id`    | caller / `Memory`                   | Required. Primary key.                                                                                               |
| `agent_id`      | `populate_from_run_output()`-style  | The owning `Agent.id`.                                                                                               |
| `team_id`       | team coordinator                    | Set when this session belongs to a Team.                                                                              |
| `workflow_id`   | workflow runtime                    | Set when invoked from a Workflow.                                                                                     |
| `user_id`       | caller                              | Used by `get_all_user_prompt_messages_for_user_id_async()`.                                                           |
| `session_type`  | constructor                         | Default `AGENT`.                                                                                                      |
| `session_data`  | `add_input_to_session_data()`       | Bag with `user_prompts`, `image_identifiers`, `document_identifiers` (deduped lists).                                 |
| `metadata`      | `update_metadata()`                 | Free-form k/v. Useful for tagging.                                                                                    |
| `agent_data`    | `update_agent_data()`               | `{agent_name, model_name}`.                                                                                          |
| `runs`          | `upsert_run()`                      | `Dict[run_id, RunData]` preserving insertion order (Python ≥3.7).                                                    |
| `summary`       | memory's summarizer                 | Rolling natural-language summary, used to compress old turns.                                                         |
| `messages`      | `append_new_messages_from_run_output()` | **Flat** chat history across runs. Only NEW messages from each run are appended (filtered by memory may differ).      |
| `usage`         | `update_usage_from_run()`           | Session-level `RunUsage` (sum of all runs).                                                                          |
| `created_at`    | storage backend                     | Unix epoch seconds.                                                                                                  |
| `updated_at`    | storage backend                     | Unix epoch seconds, refreshed on every write.                                                                        |

##### Method groups

The class has many methods. They fit into seven groups.

###### a) Serialization

| Method                                                | Returns                                | Notes                                                                                          |
| ----------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `to_dict(serialize_flag=False)`                       | `Dict[str, Any]`                       | Uses `ModelMessagesTypeAdapter.dump_python(..., mode="json")` for messages and `RunUsage.to_dict()` for usage. |
| `from_dict(data, deserialize_flag=False)` *(classmethod)* | `Optional[AgentSession]`           | Returns `None` if `session_id` is missing (and logs a warning).                                |

`serialize_flag` is propagated all the way down to `RunData.to_dict` →
`AgentRunOutput.to_dict` → cloudpickle of `Task` instances. Backends that need
to round-trip arbitrary Python objects (e.g. JSON files) pass `True`; backends
that store native Python objects (e.g. in-memory) pass `False`.

###### b) Run management

| Method                          | Behavior                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------- |
| `upsert_run(output)`            | Insert or replace the run keyed by `output.run_id`. Lazily initializes `self.runs` if `None`.    |
| `flatten_messages_from_completed_runs()` | Builds a flat message list from runs whose `status == RunStatus.completed`.                  |
| `get_run(run_id)`               | Returns the `AgentRunOutput` or `None`.                                                          |
| `get_run_data(run_id)`          | Returns the `RunData` wrapper or `None`.                                                         |
| `get_last_run()` / `get_last_run_data()` | Last inserted (dict insertion order).                                                       |
| `get_run_count()`               | `len(runs)` or 0.                                                                                |
| `clear_runs()`                  | Resets to `{}`.                                                                                  |
| `get_paused_runs()`             | Filter: `status == paused`. Used by HITL.                                                         |
| `get_error_runs()`              | Filter: `status == error`. Used by durable execution.                                             |
| `get_cancelled_runs()`          | Filter: `status == cancelled`.                                                                    |
| `get_resumable_runs()`          | Filter: `status in {paused, error, cancelled}`. Convenience for resumption flows.                 |

###### c) Message accessors

| Method                                                                  | Behavior                                                                                                                                                     |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `get_messages_by_run_id(run_id)`                                        | Copy of `output.messages` for one run.                                                                                                                       |
| `get_messages(agent_id?, last_n_runs?, limit?, skip_roles?, skip_statuses?, include_paused=False)` | Powerful multi-filter accessor. Skips child runs (`parent_run_id is not None`), de-duplicates the system message (keeps only the first). |
| `get_chat_history(last_n_runs=None)`                                    | Convenience: `get_messages(skip_roles=["system","tool"], last_n_runs=...)`.                                                                                  |
| `get_tool_calls(num_calls=None)`                                        | Walks runs in reverse insertion order, collects `message.tool_calls`.                                                                                        |

The default behavior of `get_messages` is to **skip cancelled and error runs**;
paused runs are also skipped unless `include_paused=True`. This matters for
memory composition — see Section 5.

###### d) Static helpers

| Method                                                                | Behavior                                                                                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `_extract_messages_from_runs(runs, exclude_run_id=None)`              | Concat all run-level messages, optionally skipping one `run_id`.                                                    |
| `_extract_user_prompts_from_messages(messages)`                       | Pull out string content from `ModelRequest` → `UserPromptPart`.                                                    |
| `_extract_messages_from_session(session, exclude_run_id=None)`        | Smart: if exclusion is needed it walks `runs` (boundaries preserved); else uses the flat `session.messages`.        |

###### e) Cross-storage convenience (async + sync)

These reach into `Storage` so callers don't have to. Sync versions delegate to
the async ones via a thread pool when an event loop is running.

| Method                                                                  | Returns                  |
| ----------------------------------------------------------------------- | ------------------------ |
| `_get_sessions_by_user_id_async(storage, user_id)`                      | `List[AgentSession]`     |
| `_get_session_by_session_id_async(storage, session_id)`                 | `Optional[AgentSession]` |
| `get_all_messages_for_session_id[_async](storage, session_id, exclude_run_id?)` | `List[ModelMessage]` |
| `get_all_user_prompt_messages_for_user_id[_async](storage, user_id, exclude_run_id?)` | `List[str]`     |
| `get_messages_for_run_id[_async](storage, session_id, run_id)`          | `List[ModelMessage]`     |

These methods automatically branch on `isinstance(storage, AsyncStorage)` to
call `aget_session(s)` (async) or `get_session(s)` (sync) with
`session_type=SessionType.AGENT` and `deserialize=True`.

###### f) Usage aggregation

| Method                              | Behavior                                                              |
| ----------------------------------- | --------------------------------------------------------------------- |
| `get_session_usage()`               | Returns `self.usage` if set; otherwise `_compute_usage_from_runs()`.  |
| `_compute_usage_from_runs()`        | Sums `run_data.output.usage` for every run via `RunUsage + RunUsage`. |
| `update_usage_from_run(run_output)` | Adds `run_output.usage` to the session's stored `usage`.              |
| `reset_usage()`                     | `self.usage = RunUsage()` (zero).                                     |

###### g) Metadata / agent-data / input mirroring

| Method                                                                 | Behavior                                                                                                  |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `update_metadata(key, value)` / `get_metadata(key, default=None)`      | Free-form bag.                                                                                            |
| `update_agent_data(agent_name?, model_name?)`                          | Stores in `self.agent_data["agent_name"]` / `["model_name"]`.                                             |
| `add_input_to_session_data(user_prompt?, image_identifiers?, document_identifiers?)` | Initializes lists if missing, appends user prompts, dedupes image/doc identifiers.                    |
| `get_user_prompts()` / `get_image_identifiers()` / `get_document_identifiers()` | Read accessors over `session_data`.                                                                |
| `get_agent_name()` / `get_model_name()`                                | Read accessors over `agent_data`.                                                                         |
| `populate_from_run_output(run_output)`                                 | Populates `agent_data` and `session_data` from a finished `AgentRunOutput`.                                |
| `append_new_messages_from_run_output(run_output)`                      | Appends *only the new* messages from this run via `run_output.new_messages()`. Returns the count appended. |
| `_populate_agent_data_from_output(run_output)`                         | Pulls `agent_name` and `model_name` (str-coerced).                                                        |
| `_populate_session_data_from_output(run_output)`                       | Pulls `user_prompt`, `images[*].identifier`, `documents[*].identifier`.                                   |
| `_extract_image_identifiers(run_input)` / `_extract_document_identifiers(run_input)` / `_extract_user_prompt(run_input)` | Static input parsers.                                                          |

The split between `populate_from_run_output` (metadata only) and
`append_new_messages_from_run_output` (messages only) is deliberate. Memory
filtering / limiting can shrink the model-visible history, but the *session
record* should still capture every new message produced by a run.

---

## 4. Subfolders

There are **no subfolders** under `session/` (only `__pycache__/`). The
package was intentionally kept flat:

* All persistence backends live in `src/upsonic/storage/<backend>/`.
* All memory composition lives in `src/upsonic/memory/`.
* The session module is a *value object* package and never imports either.

The lack of subfolders should be preserved unless a new session kind
(`TeamSession`, `WorkflowSession`) is introduced — in which case the natural
split would be:

```text
src/upsonic/session/
├── base.py          # SessionType, Session = Union[AgentSession, TeamSession, ...]
├── agent.py         # AgentSession
├── team.py          # TeamSession (future)
└── workflow.py      # WorkflowSession (future)
```

---

## 5. Cross-file relationships (with `/memory` and `/storage`)

The session package is consumed by three layers:

```text
                ┌──────────────────────────────┐
                │ upsonic/agent/agent.py       │  Direct.do/.do_async/.stream
                └──────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │ upsonic/memory/memory.py      │  AgentSessionMemory
               │  - asave(session)             │
               │  - aget_messages(session, ...)│
               │  - asummarize(session, ...)   │
               └──────────────┬────────────────┘
                              │ uses AgentSession.upsert_run(),
                              │       populate_from_run_output(),
                              │       append_new_messages_from_run_output(),
                              │       update_usage_from_run(),
                              │       get_messages(...).
                              ▼
            ┌──────────────────────────────────────────┐
            │ upsonic/storage/base.py::Storage         │
            │   get_session / aget_session             │
            │   upsert_session / aupsert_session       │
            │   get_sessions / aget_sessions           │
            │   delete_session / adelete_session       │
            │   (filtered by session_type=SessionType) │
            └─────────────────┬────────────────────────┘
                              │ uses AgentSession.to_dict / from_dict
                              ▼
   ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
   │ in_memory  │ json       │ sqlite     │ postgres   │ mongo      │ redis/mem0 │
   └────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```

### `storage/base.py` integration

The abstract `Storage` (sync) and `AsyncStorage` classes accept
`session_type: Optional[SessionType] = None` on `get_session`,
`get_sessions`, `delete_session` and their `a*` async siblings (`base.py`
lines 122, 147, 770, 795). Filtering by `SessionType.AGENT` is what isolates
agent rows from team / workflow rows in shared tables.

### `memory/memory.py` integration

`memory/memory.py` (the `AgentSessionMemory` orchestrator) is the *only*
caller of the mutating methods on `AgentSession`. After the agent pipeline
finishes a run it does, roughly:

```python
session = await storage.aget_session(session_id, session_type=SessionType.AGENT)
            or AgentSession(session_id=session_id, ...)

session.upsert_run(run_output)                      # add RunData
session.populate_from_run_output(run_output)        # mirror agent/session_data
session.append_new_messages_from_run_output(run_output)  # extend flat history
session.update_usage_from_run(run_output)           # accumulate tokens/$

await storage.aupsert_session(session)
```

When composing the next-turn context, memory calls back into the session for
filtered views: `session.get_messages(...)`,
`session.get_chat_history(last_n_runs=N)`, `session.get_session_summary()`,
`session.get_user_prompts()`.

### `agent/agent.py` and `agent/pipeline/steps.py`

* `Direct.do/.do_async/.stream` create / load a session via memory.
* `agent/pipeline/steps.py` ends with a *Save AgentSession* step
  (`StreamMemoryMessageTrackingStep` in streaming, the synchronous
  equivalent in non-streaming). That step is the one that ultimately calls
  the four mutating methods above.
* `Direct.cost` reads `session.get_session_usage()` to expose
  cumulative token counts and cost.

### `chat/session_manager.py`

The chat layer wraps an `AgentSession` to provide the high-level
"conversation" abstraction surfaced in user-facing APIs.

### Why no circular imports

`session/agent.py` uses `TYPE_CHECKING` guards for
`upsonic.storage.base.Storage`, `upsonic.run.agent.input.AgentRunInput` and
`upsonic.usage.RunUsage`. Only runtime-cheap modules
(`upsonic.messages.messages`, `upsonic.run.agent.output`, `upsonic.run.base`,
`upsonic.utils.logging_config`, `upsonic.session.base`) are imported eagerly.
Heavy imports (`RunUsage`, `Storage`, `ModelRequest`/`ModelResponse`) happen
inside method bodies.

---

## 6. Public API

The `session` package does **not** ship an `__init__.py`. Consumers import
from the submodules directly.

```python
from upsonic.session.base  import SessionType, Session
from upsonic.session.agent import AgentSession, RunData
```

### `SessionType`

```python
class SessionType(str, Enum):
    AGENT    = "agent"
    TEAM     = "team"
    WORKFLOW = "workflow"

    def to_dict(self) -> Dict[str, str]: ...
    @classmethod
    def from_dict(cls, data: str | dict | None) -> "SessionType": ...
    @classmethod
    def from_string(cls, value: str) -> "SessionType": ...   # strict
```

### `RunData`

```python
@dataclass
class RunData:
    output: AgentRunOutput
    deserialize_flag: bool = False

    def to_dict(self, serialize_flag: bool = False) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any], deserialize_flag: bool = False) -> "RunData": ...
```

### `AgentSession`

```python
@dataclass
class AgentSession:
    session_id: str
    agent_id: Optional[str]    = None
    team_id: Optional[str]     = None
    workflow_id: Optional[str] = None
    user_id: Optional[str]     = None
    session_type: SessionType  = SessionType.AGENT
    session_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]]     = None
    agent_data: Optional[Dict[str, Any]]   = None
    runs: Optional[Dict[str, RunData]]     = field(default_factory=dict)
    summary: Optional[str]                 = None
    messages: Optional[List[ModelMessage]] = field(default_factory=list)
    usage: Optional[RunUsage]              = None
    created_at: Optional[int]              = None
    updated_at: Optional[int]              = None
```

Methods (grouped — see Section 3.2 for full descriptions):

| Group                     | Methods                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Serialization             | `to_dict`, `from_dict`                                                                                         |
| Run management            | `upsert_run`, `flatten_messages_from_completed_runs`, `get_run`, `get_run_data`, `get_last_run`, `get_last_run_data`, `get_run_count`, `clear_runs`, `get_paused_runs`, `get_error_runs`, `get_cancelled_runs`, `get_resumable_runs` |
| Message accessors         | `get_messages_by_run_id`, `get_messages`, `get_chat_history`, `get_tool_calls`                                |
| Cross-storage helpers     | `get_all_messages_for_session_id[_async]`, `get_all_user_prompt_messages_for_user_id[_async]`, `get_messages_for_run_id[_async]` |
| Usage                     | `get_session_usage`, `update_usage_from_run`, `reset_usage`                                                   |
| Metadata / data mirroring | `update_metadata`, `get_metadata`, `update_agent_data`, `add_input_to_session_data`, `get_user_prompts`, `get_image_identifiers`, `get_document_identifiers`, `get_agent_name`, `get_model_name`, `populate_from_run_output`, `append_new_messages_from_run_output` |

### Minimal usage

```python
from upsonic.session.agent import AgentSession
from upsonic.session.base  import SessionType
from upsonic.storage.in_memory.in_memory import InMemoryStorage

storage = InMemoryStorage()

# Either load an existing session or create a fresh one
session = (
    storage.get_session("session-123", session_type=SessionType.AGENT)
    or AgentSession(session_id="session-123", user_id="alice")
)

# After the agent finishes a run...
session.upsert_run(run_output)
session.populate_from_run_output(run_output)
session.append_new_messages_from_run_output(run_output)
session.update_usage_from_run(run_output)

storage.upsert_session(session)

# Inspect later
print(session.get_session_usage().total_tokens)
for m in session.get_chat_history(last_n_runs=5):
    print(m)
```

---

## 7. Integration with the rest of Upsonic

### Agent pipeline

The agent pipeline (`src/upsonic/agent/pipeline/steps.py`) treats the session
as the authoritative log of a conversation:

* **Pre-run.** Memory loads the session, then synthesizes a context window for
  the model using `session.get_messages(...)` and `session.get_session_summary()`.
* **Mid-run.** Streaming and HITL pauses use `RunStatus.paused` flagged on the
  `AgentRunOutput` so `session.get_paused_runs()` can find them later.
* **Post-run.** The pipeline saves the session (see flow below). For streaming
  this happens in `StreamMemoryMessageTrackingStep` (`steps.py` line 4794);
  for non-streaming the equivalent step at `steps.py` line 2334 is
  *"Save AgentSession to storage. This is the LAST step for completed runs."*
* **Cost.** `Direct.cost` reads `session.get_session_usage()` (`agent.py`
  lines 848 and 879).

### Team and Workflow

Teams and workflows share the same `AgentSession` shape but tag their rows
with `SessionType.TEAM` / `SessionType.WORKFLOW`. The `team_id` /
`workflow_id` fields point back to the owning entity. Because `Session` in
`base.py` is currently `Union["AgentSession"]`, those subsystems reuse
`AgentSession` directly until specialized classes are introduced.

### Memory backends

`upsonic/memory/memory.py` is the orchestrator that wires the agent pipeline
to storage. It is the only file that mutates an `AgentSession` outside of
tests.

### Storage backends

Each backend in `src/upsonic/storage/<provider>/` implements:

| Method                                       | Calls                                                              |
| -------------------------------------------- | ------------------------------------------------------------------ |
| `get_session(...)` / `aget_session(...)`     | `AgentSession.from_dict(row, deserialize_flag=True)`               |
| `get_sessions(...)` / `aget_sessions(...)`   | `[AgentSession.from_dict(r, deserialize_flag=True) for r in rows]` |
| `upsert_session(s)` / `aupsert_session(s)`   | `s.to_dict(serialize_flag=True)`                                   |
| `delete_session(session_id, session_type)`   | filtered DELETE                                                    |

`SessionType` is always passed as `session_type=SessionType.AGENT` in the
agent path; this lets a single backing table hold rows for agents, teams and
workflows.

### Chat

`src/upsonic/chat/session_manager.py` builds a higher-level "conversation"
object on top of `AgentSession` and exposes it to user-facing APIs.

---

## 8. End-to-end flow of a session save / load

Below is the canonical save / load lifecycle for one agent invocation.

### 8.1 Load

```text
[Caller]           Direct.do_async(task, session_id="s1", user_id="alice")
   │
   ▼
[Agent.do_async]   asks Memory to hydrate the session.
   │
   ▼
[Memory]           storage.aget_session(
                       session_id="s1",
                       session_type=SessionType.AGENT,
                       deserialize=True,
                   )
   │
   ▼
[Storage backend]  reads row → calls AgentSession.from_dict(row, deserialize_flag=True)
                       ├── SessionType.from_dict(row["session_type"])
                       ├── RunData.from_dict(...) for each entry in row["runs"]
                       │     └── AgentRunOutput.from_dict(..., deserialize_flag=True)
                       ├── ModelMessagesTypeAdapter.validate_python(row["messages"])
                       └── RunUsage.from_dict(row["usage"])
   │
   ▼
[Memory]           builds context via:
                       - session.get_messages(skip_statuses=[error, cancelled],
                                              include_paused=...)
                       - session.get_session_summary()
                       - session.get_user_prompts() / get_image_identifiers() / ...
   │
   ▼
[Pipeline]         runs steps with that context, producing an AgentRunOutput.
```

If the storage row doesn't exist, `Memory` constructs a fresh
`AgentSession(session_id=..., user_id=..., session_type=SessionType.AGENT)`.

### 8.2 Save

```text
[Pipeline finishes] → AgentRunOutput(run_id="r-7", status=completed,
                                     messages=[...], usage=RunUsage(...),
                                     input=AgentRunInput(...), ...)
   │
   ▼
[Memory.asave(session, run_output)]
   ├── session.upsert_run(run_output)
   │      └── self.runs["r-7"] = RunData(output=run_output)
   ├── session.populate_from_run_output(run_output)
   │      ├── _populate_agent_data_from_output → update_agent_data(name, model)
   │      └── _populate_session_data_from_output
   │              ├── _extract_image_identifiers
   │              ├── _extract_document_identifiers
   │              ├── _extract_user_prompt
   │              └── add_input_to_session_data(...)  # dedupes lists
   ├── session.append_new_messages_from_run_output(run_output)
   │      └── self.messages.extend(run_output.new_messages())
   ├── session.update_usage_from_run(run_output)
   │      └── self.usage = (self.usage or RunUsage()) + run_output.usage
   └── storage.aupsert_session(session)
           └── row = session.to_dict(serialize_flag=True)
                       ├── runs: {run_id: RunData.to_dict(serialize_flag=True)}
                       ├── messages: ModelMessagesTypeAdapter.dump_python(..., mode="json")
                       ├── usage: RunUsage.to_dict()
                       └── session_type: {"session_type": "agent"}
```

After this point the session row contains everything needed to fully
reconstruct the conversation, including paused / errored / cancelled runs that
HITL or durable execution can later resume via:

```python
for run_data in session.get_resumable_runs():
    output = run_data.output
    # output.input + output.status + output.messages → resume here
```

### 8.3 Streaming variant

For `Direct.stream(...)` the same flow applies, but the *Save* step is named
`StreamMemoryMessageTrackingStep` (`steps.py` line 4794) and runs **after**
the stream is fully consumed. Until that step runs, intermediate messages
live in the in-flight `AgentRunOutput` only; the persisted `AgentSession` is
not updated mid-stream. Session-level `usage` is updated inside
`AgentSessionMemory.asave()` (note in `steps.py` line 3488).

### 8.4 Cross-session queries

Some flows need data across many sessions for the same user:

```python
# All user-typed prompts ever sent by alice (excluding the current run)
prompts: List[str] = AgentSession.get_all_user_prompt_messages_for_user_id(
    storage,
    user_id="alice",
    exclude_run_id=current_run_id,
)
```

Internally this:

1. Calls `_get_sessions_by_user_id_async(storage, "alice")` which picks
   `aget_sessions` or `get_sessions` based on `isinstance(storage, AsyncStorage)`.
2. For each session, runs `_extract_messages_from_session(s, exclude_run_id)`
   — which prefers `session.messages` when no exclusion is needed, and walks
   `session.runs` (preserving run boundaries) when it is.
3. Pipes the result through `_extract_user_prompts_from_messages` to surface
   only `UserPromptPart` content.

The sync entry points (`get_all_messages_for_session_id`,
`get_all_user_prompt_messages_for_user_id`, `get_messages_for_run_id`)
all delegate to their async siblings. If they are called from inside a
running event loop they spawn a `concurrent.futures.ThreadPoolExecutor` and
run `asyncio.run` inside the worker thread — otherwise they call
`asyncio.run` directly. This is what lets the same helpers be invoked from
notebooks, sync agent code and async agent code without ceremony.

---

## Appendix — Quick reference

### File map

| File                       | Lines | Symbols                          |
| -------------------------- | ----- | -------------------------------- |
| `src/upsonic/session/base.py`  | ~80   | `SessionType`, `Session`         |
| `src/upsonic/session/agent.py` | ~1045 | `RunData`, `AgentSession`        |

### Imports cheat-sheet

```python
from upsonic.session.base   import SessionType, Session
from upsonic.session.agent  import AgentSession, RunData

# Often used together
from upsonic.run.agent.output import AgentRunOutput
from upsonic.run.base         import RunStatus
from upsonic.usage            import RunUsage
from upsonic.messages.messages import ModelMessage
from upsonic.storage.base     import Storage, AsyncStorage
```

### Conventions

| Topic                | Convention                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| Run identity         | `run_id` is the dict key in `AgentSession.runs`. Insertion order is preserved (Python ≥3.7).            |
| Child runs           | `output.parent_run_id is not None` — these are skipped by `get_messages` to avoid duplication.          |
| System messages      | Only the **first** system message survives `get_messages`, regardless of how many runs contain one.    |
| Status filtering     | `get_messages` skips `cancelled` + `error` by default; skips `paused` unless `include_paused=True`.    |
| Message mirroring    | Use `append_new_messages_from_run_output` (NOT raw `extend`) — it consults `output.new_messages()`.    |
| Serialization toggle | Pass `serialize_flag=True` to `to_dict` from backends that need cloudpickled `Task` objects (e.g. JSON). |
| Defaulting           | `SessionType.from_dict` is lenient (defaults to AGENT); `SessionType.from_string` is strict.            |

