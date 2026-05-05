---
name: run-lifecycle-data-model
description: Use when working with the data spine of agent execution in `src/upsonic/run/` — input/output dataclasses, run status, events, HITL requirements, and cancellation. Use when a user asks to inspect or extend `AgentRunInput`, `AgentRunOutput`, `RunStatus`, run events, `RunRequirement`, `ToolExecution`, the cancel registry, or `PipelineExecutionStats`; serialize/rehydrate a run; add a new event type; debug HITL pause/resume; or trace what flows through `Agent.do` / `Agent.stream` / `continue_run_async`. Trigger when the user mentions AgentRunInput, AgentRunOutput, RunStatus, RunRequirement, ToolExecution, RunCancellationManager, raise_if_cancelled, register_run, cleanup_run, AgentEvent, AgentRunEvent, PipelineExecutionStats, run lifecycle, run state, run events, HITL, confirmation, user_input, external tool, pause, cancel, streaming events, to_dict/from_dict on a run, run_id, or `upsonic.run`.
---

# `src/upsonic/run/` — Run lifecycle data model

This folder is the **data spine** of every agent execution in Upsonic. It does not
implement the pipeline itself (that lives in `src/upsonic/agent/pipeline/`); instead
it defines the **types** that flow through the pipeline:

- The **input** (`AgentRunInput`) — the user prompt, plus optional images, documents,
  and pre-built `ModelRequest` history.
- The **output / state container** (`AgentRunOutput`) — the single source of truth
  for everything that happens during a run: messages, tool calls, requirements,
  step results, usage, status, events.
- The **status enum** (`RunStatus`) — `running` / `completed` / `paused` /
  `cancelled` / `error`.
- The **events** (`events/events.py`) — ~40 dataclass event types emitted during
  pipeline execution and streamed to the user.
- The **HITL requirements model** (`requirements.py` + `tools/tools.py`) —
  `RunRequirement` and `ToolExecution` for confirmation, user-input, and external
  tool patterns.
- The **cancel registry** (`cancel.py`) — a thread-safe, process-global manager
  that maps `run_id` → cancelled flag, with a `raise_if_cancelled()` checkpoint
  that the pipeline polls between steps.
- The **execution stats** (`pipeline/stats.py`) — per-step timing and status
  rollup attached to the output.

Every `Agent.do(...)`, `Agent.do_async(...)`, `Agent.stream(...)`, and
`Agent.continue_run_async(...)` call constructs an `AgentRunInput`, threads it
through the pipeline, and returns an `AgentRunOutput`. Every event the user sees
in `async for event in agent.stream(...)` is one of the dataclasses in
`events/events.py`. Every `run.cancel()` call routes through the registry in
`cancel.py`.

Read this folder first when you want to understand *what* an agent run actually
*is* in Upsonic — the rest of the framework is just transformation of these
types.

---

## 1. What this folder is

The run module models the **lifecycle of a single agent run** as plain Python
dataclasses with `to_dict` / `from_dict` round-trip serialization. It is
deliberately **storage-agnostic** and **pipeline-agnostic**: the dataclasses
contain no pipeline logic and no I/O — they are pure state.

The lifecycle looks like:

```text
                ┌──────────────────────────────────────────────────┐
                │  Agent.do(input=...) / Agent.stream(...)         │
                └──────────────────────┬───────────────────────────┘
                                       │
                       AgentRunInput ──┤
                       (user_prompt,   │      register_run(run_id)
                        images,        ├──►   in cancel.py
                        documents)     │
                                       ▼
                       ┌──────────────────────────────┐
                       │   PipelineManager runs Steps │
                       │   (initialization → cache →  │
                       │    policy → memory →         │
                       │    LLM call → tools → ...)   │
                       └──────────────┬───────────────┘
                                      │ each Step:
                                      │   - emit events into state.events
                                      │   - mutate AgentRunOutput
                                      │   - raise_if_cancelled()
                                      ▼
                       AgentRunOutput is mutated in place:
                         • status      (RunStatus.running → completed/paused/...)
                         • chat_history, messages, response
                         • tools, tool_call_count
                         • requirements (HITL pauses)
                         • step_results, execution_stats
                         • events (audit trail)
                         • usage (TaskUsage)
                                      │
                                      ▼
                            cleanup_run(run_id)
                                      │
                                      ▼
                            return AgentRunOutput
```

The whole file `run/agent/output.py` describes itself as the *"SINGLE SOURCE OF
TRUTH for agent run state"* — because the same dataclass is consumed by
streaming, by HITL continuation, by storage, by the pricing layer, and by the
chat UI.

---

## 2. Folder layout

```text
src/upsonic/run/
├── __init__.py              # Lazy-loaded public re-exports
├── base.py                  # RunStatus enum (running/completed/paused/cancelled/error)
├── cancel.py                # Process-global RunCancellationManager + module helpers
├── requirements.py          # RunRequirement (HITL: confirmation/user_input/external)
│
├── agent/
│   ├── __init__.py          # Lazy re-exports of AgentRunInput / AgentRunOutput
│   ├── input.py             # AgentRunInput dataclass + URL→BinaryContent conversion
│   └── output.py            # AgentRunOutput — the run-state container (~1100 lines)
│
├── events/
│   ├── __init__.py          # Lazy re-exports of all ~40 event classes
│   └── events.py            # AgentEvent base + all concrete event dataclasses + AgentRunEvent enum
│
├── pipeline/
│   ├── __init__.py          # Re-exports PipelineExecutionStats
│   └── stats.py             # PipelineExecutionStats: per-step timing + status rollup
│
└── tools/
    └── tools.py             # ToolExecution dataclass (HITL flags + result + metrics)
```

Note that `run/tools/` does *not* contain an `__init__.py` — it is imported as
`upsonic.run.tools.tools` (namespace package). Only one module lives there.

---

## 3. Top-level files

### 3.1 `run/__init__.py` — lazy public API

This module is intentionally an **import-on-demand façade**. It builds a
`_lazy_imports` dict and exposes a `__getattr__` so importing
`upsonic.run.AgentRunOutput` does not pull in pydantic, httpx, or the
serialization stack until you actually touch the symbol.

```python
__all__ = [
    "RunCancellationManager",
    "register_run", "cancel_run", "cleanup_run",
    "raise_if_cancelled", "is_cancelled",
    "RunStatus",
    "AgentRunInput", "AgentRunOutput",
    "RunRequirement",
    "ToolExecution",
    "EventEmitter",   # exposed for forward compatibility (no concrete impl in this folder)
]
```

Each `_get_*()` helper calls `_lazy_import("upsonic.run.<module>", "<Class>")`
which `importlib.import_module`s the submodule the first time and caches the
module reference. The tradeoff is simple:

| Feature                           | Behavior                                                |
| --------------------------------- | ------------------------------------------------------- |
| `import upsonic.run`              | No transitive import of pydantic / cloudpickle / httpx. |
| `upsonic.run.AgentRunOutput`      | First access triggers import; subsequent uses are free. |
| `from upsonic.run import RunStatus` | Same — handled via `__getattr__`.                     |

> Note: `EventEmitter` is listed in `__all__` and lazily routed to
> `upsonic.run.events.emitter`, but **no `emitter.py` module currently exists**
> in this folder. Accessing `upsonic.run.EventEmitter` will raise
> `ModuleNotFoundError`. The lazy import is a placeholder for a future helper.

### 3.2 `run/base.py` — `RunStatus`

```python
class RunStatus(str, Enum):
    running   = "RUNNING"
    completed = "COMPLETED"
    paused    = "PAUSED"
    cancelled = "CANCELLED"
    error     = "ERROR"
```

Inheriting from `str` means `RunStatus.running == "RUNNING"` is true and the
enum is JSON-serializable for free. Three helpers on top:

| Method                                | Purpose                                                                                  |
| ------------------------------------- | ---------------------------------------------------------------------------------------- |
| `to_dict() -> str`                    | Returns `self.value`. Used by `AgentRunOutput.to_dict()` to persist status.              |
| `RunStatus.from_dict(data: str)`      | `cls(data)` — the reverse direction.                                                     |
| `RunStatus.from_step_status(step_status)` | Maps `agent.pipeline.step.StepStatus` → `RunStatus`. `SKIPPED` → `completed`.           |

The `from_step_status` mapping is the bridge between **step**-level state
(within the pipeline) and **run**-level state (visible to the caller):

| StepStatus  | RunStatus    |
| ----------- | ------------ |
| `RUNNING`   | `running`    |
| `COMPLETED` | `completed`  |
| `PAUSED`    | `paused`     |
| `CANCELLED` | `cancelled`  |
| `ERROR`     | `error`      |
| `SKIPPED`   | `completed`  |

The full status semantic table:

| Status      | Meaning                                                             | Set by                                  |
| ----------- | ------------------------------------------------------------------- | --------------------------------------- |
| `running`   | The pipeline is currently executing this run.                       | Default when `AgentRunOutput()` is constructed. |
| `completed` | The pipeline finished and produced final output.                    | `output.mark_completed()`               |
| `paused`    | A HITL requirement (external tool / confirmation / user input) needs to be resolved before the pipeline can proceed. The run is *resumable* via `Agent.continue_run_async(run_output, ...)`. | `output.mark_paused(reason=...)`        |
| `cancelled` | A `cancel_run(run_id)` call was observed at a `raise_if_cancelled()` checkpoint and `RunCancelledException` was raised. | `output.mark_cancelled()`               |
| `error`     | An exception bubbled up out of a Step.                              | `output.mark_error(error=...)`          |

The `is_problematic` property on `AgentRunOutput` is the union `paused ∪ cancelled ∪ error` — the runs that need user/system action before they progress.

### 3.3 `run/cancel.py` — cancellation registry

The cancel module is **deliberately tiny**. It owns one global instance
`_cancellation_manager = RunCancellationManager()` and exposes module-level
shorthands.

```python
class RunCancellationManager:
    _cancelled_runs: Dict[str, bool]   # run_id -> True if cancelled
    _lock: threading.Lock              # protects the dict
```

| Method                          | Behavior                                                             |
| ------------------------------- | -------------------------------------------------------------------- |
| `register_run(run_id)`          | Records `run_id → False`. Called by the agent when starting a run.   |
| `cancel_run(run_id) -> bool`    | Sets `run_id → True`. Returns `False` if `run_id` was never registered. Logs a warning in that case. |
| `is_cancelled(run_id) -> bool`  | Reads the flag with `False` default.                                 |
| `cleanup_run(run_id)`           | Removes the entry. Called on completion/error/cancel finalization.   |
| `raise_if_cancelled(run_id)`    | If cancelled, raise `RunCancelledException(f"Run {run_id} was cancelled")`. |
| `get_active_runs()`             | Returns a copy of the dict for diagnostics / debugging.              |

Module-level `register_run`, `cancel_run`, `is_cancelled`, `cleanup_run`,
`raise_if_cancelled` are thin wrappers over `_cancellation_manager`. They are
the entry points the rest of the codebase actually imports — see
`grep -rn "from upsonic.run.cancel"`:

- `agent/agent.py` calls `register_run` at the start of `do_async` and
  `cleanup_run` at the end (in `finally`).
- `agent/pipeline/steps.py` calls `raise_if_cancelled(state.run_id)` at the top
  of **every** step's `execute` method (~25 call sites). This is the
  cooperative-cancellation pattern: cancellation cannot interrupt blocking I/O
  inside a step, but it *will* be honored at the next step boundary.

```python
# Example pattern from steps.py
async def execute(self, state):
    from upsonic.run.cancel import raise_if_cancelled
    raise_if_cancelled(state.run_id)
    # ... do step work ...
```

### 3.4 `run/requirements.py` — `RunRequirement`

`RunRequirement` is the **HITL pause primitive**. It wraps a single
`ToolExecution` and exposes properties that classify *what kind* of human
interaction is needed.

```python
@dataclass
class RunRequirement:
    id: str                                # uuid4
    tool_execution: Optional[ToolExecution]
    created_at: datetime
    confirmation: Optional[bool]           # None=unset, True=confirmed, False=rejected
    confirmation_note: Optional[str]
    user_input_schema: Optional[List[Dict[str, Any]]]
```

There are **three HITL patterns** modeled here, mutually exclusive but
combined into one type to keep the `requirements: List[RunRequirement]` on
`AgentRunOutput` simple.

| Property                  | Meaning                                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------------------- |
| `needs_confirmation`      | `tool_execution.requires_confirmation` is True and `self.confirmation is None`.                      |
| `needs_user_input`        | `tool_execution.requires_user_input` is True, `answered` is not True, and at least one schema field still has `value=None`. |
| `needs_external_execution`| `tool_execution.external_execution_required` is True and `tool_execution.result is None`.            |
| `is_external_tool_execution` | Alias for `needs_external_execution`.                                                             |
| `has_result`              | `tool_execution.result is not None`.                                                                 |
| `is_resolved`             | `not needs_confirmation and not needs_user_input and not needs_external_execution`.                  |

The mutation methods are equally narrow:

| Method                                | Purpose                                                                                                  |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `confirm()`                           | Sets `confirmation=True` and `tool_execution.confirmed=True`. Raises `ValueError` if not pending confirm. |
| `reject(note=None)`                   | Sets `confirmation=False`, `confirmation_note=note`, mirrors to `tool_execution`.                        |
| `set_external_execution_result(result)`| Writes `tool_execution.result = result`. Caller code typically follows up with `Agent.continue_run_async`. |
| `mark_for_external_execution()`       | Flips `tool_execution.external_execution_required = True`.                                               |

`to_dict()` / `from_dict()` round-trip through plain JSON — `tool_execution`
delegates to `ToolExecution.to_dict()` and `created_at` is ISO-formatted.

### 3.5 `run/tools/tools.py` — `ToolExecution`

The data record the agent fills in for **every tool call** (whether a regular
function tool, an MCP tool, or an external/HITL tool).

```python
@dataclass
class ToolExecution:
    tool_call_id: Optional[str]
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    tool_call_error: Optional[bool]
    result: Optional[str]
    metrics: Optional[ToolMetrics]
    child_run_id: Optional[str]            # if the tool spawned a sub-run (Agent-as-Tool / Team)
    stop_after_tool_call: bool             # if True, end the agent loop after this tool
    created_at: int                        # unix-time

    # HITL flags
    requires_confirmation: Optional[bool]
    confirmed: Optional[bool]
    confirmation_note: Optional[str]
    requires_user_input: Optional[bool]
    user_input_schema: Optional[List[Dict[str, Any]]]
    answered: Optional[bool]
    external_execution_required: Optional[bool]

    result_injected: bool                  # internal: has the result been pushed back into chat_history?
```

| Field                          | Used by                                                                          |
| ------------------------------ | -------------------------------------------------------------------------------- |
| `tool_call_id`, `tool_name`, `tool_args`, `result` | The pipeline's tool-call step + the LLM model adapter.        |
| `metrics`                      | `upsonic.tools.metrics.ToolMetrics` — duration / token / cost telemetry.         |
| `child_run_id`                 | Set when an Agent-as-Tool or Team sub-call writes a nested run.                  |
| `stop_after_tool_call`         | Honored by the pipeline's `ProcessToolCallsStep` to short-circuit the model loop.|
| `requires_*` / `confirmed` / `answered` / `external_execution_required` | The HITL fields that drive `RunRequirement.needs_*`. |
| `result_injected`              | Bookkeeping so the same result is not appended to `chat_history` twice during a continuation. |

The single derived property is:

```python
@property
def is_paused(self) -> bool:
    return bool(self.requires_confirmation
                or self.requires_user_input
                or self.external_execution_required)
```

`metrics` round-trips through `ToolMetrics.to_dict()` / `from_dict()`.

---

## 4. Subfolders

### 4.1 `agent/` — input and output

#### 4.1.1 `agent/input.py` — `AgentRunInput`

This is what the **caller** constructs. Conceptually:

> A user prompt + zero-or-more images + zero-or-more documents.

```python
@dataclass
class AgentRunInput:
    user_prompt: Union[str, ModelRequest, BaseModel, List[ModelRequest]]
    images: Optional[Union[List[BinaryContent], List[str]]] = None
    documents: Optional[Union[List[BinaryContent], List[str]]] = None
    input: Optional[Union[str, List[Union[str, BinaryContent]]]] = None
```

| `user_prompt` type        | Interpretation                                                              |
| ------------------------- | --------------------------------------------------------------------------- |
| `str`                     | Plain text prompt.                                                          |
| `ModelRequest`            | A pre-built model request (advanced: replay, fixtures).                     |
| `BaseModel`               | A pydantic model — serialized via `model_dump()` and tagged with `__pydantic__` for round-trip. |
| `List[ModelRequest]`      | Multiple pre-built requests (used when continuing a paused run).            |

| `images` / `documents` element type | Behavior on `__post_init__` / `build_input` |
| ----------------------------------- | ------------------------------------------- |
| `BinaryContent`                     | Used as-is.                                 |
| `ImageUrl` / `DocumentUrl`          | **Synchronously downloaded with `httpx.get`** in `__post_init__` and converted to `BinaryContent`. |
| `str` (file path)                   | Read from disk in `build_input()` / `abuild_input()` with `mimetypes.guess_type`. |

The two helpers `build_input(context_formatted=None)` and its async twin
`abuild_input(...)` produce the final `self.input` field that the LLM message
builder will consume:

- If no media: `self.input` is the (possibly context-augmented) user prompt
  string.
- If media: `self.input` is a list of `[user_prompt, *images, *documents]`,
  each media item being a `BinaryContent`.

`to_dict()` / `from_dict()` round-trip is **format-aware**:

- `user_prompt`: `str` → as-is; `ModelRequest` → wrapped in `[req]` and serialized via `ModelMessagesTypeAdapter`; `BaseModel` → `{"__pydantic__": True, "class": ..., "module": ..., "data": model.model_dump()}`.
- `images` / `documents`: `BinaryContentTypeAdapter.dump_python(..., mode="json")`. String paths are **filtered out** during serialization (paths are local-machine artifacts that should not survive cross-machine restore).

#### 4.1.2 `agent/output.py` — `AgentRunOutput`

The biggest file in this folder (~1100 lines). The class docstring states:

> *This is the SINGLE SOURCE OF TRUTH for agent run state.*

Every step in the pipeline reads and writes attributes on this object, the
streaming layer reads `events` and `accumulated_text` from it, and the storage
layer pickles its `to_dict()` form to persist runs.

##### 4.1.2.1 Identity / context fields

| Field             | Type             | Purpose                                                  |
| ----------------- | ---------------- | -------------------------------------------------------- |
| `run_id`          | `Optional[str]`  | Unique identifier for *this* run. Matches `cancel.register_run`. |
| `agent_id`        | `Optional[str]`  | Which agent produced this run.                           |
| `agent_name`      | `Optional[str]`  | Human-readable agent name.                               |
| `session_id`      | `Optional[str]`  | Session this run belongs to (multi-run conversation).    |
| `parent_run_id`   | `Optional[str]`  | If this run was spawned by another (Team / Agent-as-Tool). |
| `user_id`         | `Optional[str]`  | End-user identity (for memory partitioning).             |
| `trace_id`        | `Optional[str]`  | OpenTelemetry trace correlation id.                      |
| `task`            | `Optional[Task]` | Embedded task object — single source of truth for HITL.  |
| `input`           | `Optional[AgentRunInput]` | Reference back to the caller-supplied input.    |

##### 4.1.2.2 Output / model-info fields

| Field                       | Type                                | Purpose                                                       |
| --------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| `output`                    | `Optional[Union[str, bytes]]`       | Final processed output (text or binary, e.g. an image).       |
| `output_schema`             | `Optional[Union[str, Type[BaseModel]]]` | Output type constraint (set when caller passed `output_type=...`). |
| `thinking_content`          | `Optional[str]`                     | Concatenated reasoning text (Claude / GPT-o1 / DeepSeek).      |
| `thinking_parts`            | `Optional[List[ThinkingPart]]`      | Structured reasoning parts.                                    |
| `model_name`                | `Optional[str]`                     | The model that ran (e.g. `gpt-4o`).                            |
| `model_provider`            | `Optional[str]`                     | The provider (e.g. `openai`).                                  |
| `model_provider_profile`    | `Optional[ModelProfile]`            | Capability/limit profile in effect during the run.             |

##### 4.1.2.3 Message tracking

| Field                          | Purpose                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `chat_history`                 | **Full** conversation across the whole session: historical messages from memory + everything new this run. Built by `ChatHistoryStep`. |
| `messages`                     | **Only the new** messages from THIS run (extracted at run end).                                          |
| `response`                     | The current `ModelResponse` (single, not a list).                                                        |
| `usage`                        | `TaskUsage` accumulator (tokens, cost, latency).                                                         |
| `additional_input_message`     | Continuation prompt(s) injected when resuming a paused run.                                              |
| `memory_message_count`         | How many of the messages in `chat_history` came from memory (vs were generated this run).                |
| `_run_boundaries`              | List of `len(chat_history)` snapshots — one per `start_new_run()` call. Used by `finalize_run_messages()` to slice out the new tail. |

The `_run_boundaries` mechanism is what makes message tracking robust across
nested calls:

```python
def start_new_run(self) -> None:
    """Called by ChatHistoryStep AFTER loading historical messages."""
    self._run_boundaries.append(len(self.chat_history))

def finalize_run_messages(self) -> None:
    """Called by MemorySaveStep at run end."""
    run_start = self._run_boundaries[-1] if self._run_boundaries else 0
    self.messages = list(self.chat_history[run_start:])
```

The methods `all_messages()`, `new_messages()`, `add_message(...)`,
`add_messages(...)`, `get_last_model_response()`, and `has_new_messages()` all
work off this slice.

##### 4.1.2.4 Tool / media fields

| Field                          | Purpose                                                                                                     |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `tools`                        | `List[ToolExecution]` — every tool call attempted in this run, including HITL-gated ones.                    |
| `tool_call_count`              | Counter; compared against `Agent.tool_call_limit`.                                                          |
| `tool_limit_reached`           | `True` once the limit was hit (used by the pipeline to short-circuit).                                      |
| `images`                       | `List[BinaryContent]` — images returned by tool calls or model output.                                      |
| `files`                        | `List[BinaryContent]` — files returned by tool calls.                                                       |

##### 4.1.2.5 Status / HITL fields

| Field                          | Purpose                                                                                                     |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `status`                       | `RunStatus` — current overall state.                                                                        |
| `requirements`                 | `List[RunRequirement]` — open HITL pauses (external execution, confirmation, user input).                   |
| `step_results`                 | `List[StepResult]` — one entry per executed pipeline step.                                                  |
| `current_step_result`          | The step currently executing (set by `Step.execute`, read by `Step.run` for outer instrumentation).          |
| `pause_reason`                 | `Literal["external_tool", "confirmation", "user_input"]` — set by `mark_paused(reason=...)`.                |
| `error_details`                | Error message string (populated by `mark_error`).                                                           |

The HITL property accessors deduplicate across both `tools` and
`requirements` (they may overlap because the pipeline records the same tool
call in both lists for different reasons):

| Property                              | Returns                                                                                  |
| ------------------------------------- | ---------------------------------------------------------------------------------------- |
| `is_paused`                           | `status == RunStatus.paused`                                                             |
| `is_cancelled`                        | `status == RunStatus.cancelled`                                                          |
| `is_complete`                         | `status == RunStatus.completed`                                                          |
| `is_error`                            | `status == RunStatus.error`                                                              |
| `is_problematic`                      | `status in {paused, cancelled, error}`                                                   |
| `active_requirements`                 | requirements where `not is_resolved`                                                     |
| `tools_requiring_confirmation`        | dedup'd `ToolExecution`s with `requires_confirmation=True`, `confirmed is None`          |
| `tools_requiring_user_input`          | dedup'd `ToolExecution`s with `requires_user_input=True`, `answered != True`             |
| `tools_awaiting_external_execution`   | dedup'd `ToolExecution`s with `external_execution_required=True`, `result is None`       |

`add_requirement(req)`, `get_external_tool_requirements()`,
`get_external_tool_requirements_with_results()`, and
`has_pending_external_tools()` are the requirement-management methods used by
the pipeline.

##### 4.1.2.6 Step-result inspection

These methods are how durable execution / continuation logic locates the
relevant step for resumption:

| Method                          | Returns                                                                  |
| ------------------------------- | ------------------------------------------------------------------------ |
| `get_step_results()`            | `self.step_results` (or `[]`).                                          |
| `get_execution_stats()`         | `self.execution_stats` (`PipelineExecutionStats`).                       |
| `get_last_successful_step()`    | Most recent `StepResult` with `status=COMPLETED`. Walks in reverse.      |
| `get_error_step()`              | First `StepResult` with `status=ERROR`. Used for durable retries.        |
| `get_cancelled_step()`          | First `StepResult` with `status=CANCELLED`. Used for cancel resumption.  |
| `get_paused_step()`             | First `StepResult` with `status=PAUSED`. Used for HITL continuation.     |
| `get_problematic_step()`        | `error or cancelled or paused` (in that priority order).                 |

##### 4.1.2.7 Status-transition methods

```python
def mark_paused(self, reason: Literal["external_tool", "confirmation", "user_input"] = "external_tool")
def mark_cancelled(self)
def mark_completed(self)
def mark_error(self, error: Optional[str] = None)
```

All four:

1. set `self.status` to the corresponding `RunStatus`,
2. update `self.updated_at` to the current epoch,
3. call `_sync_status_to_task()` so the embedded `Task` object reflects the
   same status.

`mark_completed()` additionally promotes `accumulated_text` into `output` if
streaming finished without anybody else setting `output`. `mark_error()`
populates both `error_details` and `metadata["error"]`.

##### 4.1.2.8 Usage-tracking methods

```python
start_usage_timer()
stop_usage_timer(set_duration=True)
set_usage_time_to_first_token()
increment_tool_calls(count=1)
add_model_execution_time(elapsed)
add_tool_execution_time(elapsed)
set_usage_cost(cost)             # Adds to existing cost if already set
```

All delegate to `self.usage` (`TaskUsage` from `upsonic.usage`), which is
lazily created and dict-compatible (a stored run that comes back as a `dict`
under `usage` is rehydrated through `TaskUsage.from_dict`).

##### 4.1.2.9 Serialization (`to_dict` / `from_dict`)

`to_dict(serialize_flag: bool = False) -> Dict[str, Any]` produces a dict
ready for JSON/SQL/Mongo storage. The serialization is **delegating**:

| Field type                                                          | Adapter / method used                                |
| ------------------------------------------------------------------- | ---------------------------------------------------- |
| `chat_history`, `messages`, `additional_input_message`              | `ModelMessagesTypeAdapter.dump_python(..., mode="json")` |
| `response` (single `ModelResponse`)                                 | Same adapter, wrapped in a `[]` then unwrapped.      |
| `thinking_parts`                                                    | `ModelResponsePartTypeAdapter`                       |
| `images`, `files`                                                   | `BinaryContentTypeAdapter`                           |
| `usage`                                                             | `TaskUsage.to_dict()`                                |
| `tools`                                                             | `[t.to_dict() for t in self.tools]`                  |
| `status`                                                            | `RunStatus.to_dict()`                                |
| `requirements`                                                      | `[r.to_dict() for r in self.requirements]`           |
| `step_results`                                                      | `[sr.to_dict() for sr in self.step_results]`         |
| `execution_stats`                                                   | `PipelineExecutionStats.to_dict()`                   |
| `events`                                                            | `[e.to_dict() for e in self.events]`                 |
| `agent_knowledge_base_filter`                                       | `KBFilterExpr.to_dict()`                             |
| `current_step_result`                                               | `StepResult.to_dict()`                               |
| `model_provider_profile`                                            | `ModelProfile.to_dict()`                             |
| `task`                                                              | `Task.to_dict(serialize_flag=serialize_flag)` — propagates the cloudpickle flag |
| `input`                                                             | `AgentRunInput.to_dict()`                            |
| `output_schema`                                                     | Tagged `{"__builtin_type__": True, "name": "str"}` for `str`, `{"__pydantic_type__": True, "name", "module"}` for pydantic models, `{"__type__": True, ...}` for arbitrary types. |

`from_dict(data, deserialize_flag: bool = False)` reverses the same mapping:
each tagged dict is dispatched to the corresponding `from_dict`, each adapter
is called with `validate_python(...)`, and the `cls(...)` constructor is
called with the rehydrated objects.

`to_json(indent=2, serialize_flag=False)` is just `json.dumps(self.to_dict(...))`.

`__str__` returns the textual `output`. `__repr__` returns a compact
`AgentRunOutput(run_id=..., status=..., output_length=...)` line.

---

### 4.2 `events/` — pipeline event types

Almost all of the streaming and observability surface area of Upsonic is
defined in `events/events.py`. There are roughly forty event classes, all
inheriting from a common `AgentEvent` base.

#### 4.2.1 `AgentEvent` base class

```python
@dataclass(repr=False, kw_only=True)
class AgentEvent(ABC):
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: Optional[str] = None
    timestamp: datetime = field(default_factory=_now_utc)
    event_kind: Literal['agent_event'] = 'agent_event'
```

Two notable bits of machinery on the base class:

- `to_dict()` walks `dataclasses.fields(self)`, special-cases `datetime`
  (ISO-format), `Enum` (`.value`), and any nested object that has its own
  `to_dict`. It tags the output with `__event_class__` = class name so
  deserialization knows which subclass to instantiate.
- `AgentEvent.from_dict(data)` looks up `data["__event_class__"]` in
  `_EVENT_CLASS_REGISTRY` (defined at the bottom of the file) and constructs
  the right subclass with the right kwargs.

`event_kind` is a `Literal` field that is overridden in every subclass — this
is what pydantic uses as the discriminator when constructing `AgentStreamEvent`
union values.

#### 4.2.2 The full event taxonomy

##### Pipeline-level events

| Class                | `event_kind`        | Key fields                                                                                |
| -------------------- | ------------------- | ----------------------------------------------------------------------------------------- |
| `PipelineStartEvent` | `pipeline_start`    | `total_steps`, `is_streaming`, `task_description`                                         |
| `PipelineEndEvent`   | `pipeline_end`      | `total_steps`, `executed_steps`, `total_duration`, `status: 'success'\|'error'\|'paused'`, `error_message` |

##### Step-level events (one pair per Step in the pipeline)

| Class            | `event_kind`  | Key fields                                                                                  |
| ---------------- | ------------- | ------------------------------------------------------------------------------------------- |
| `StepStartEvent` | `step_start`  | `step_name`, `step_description`, `step_index`, `total_steps`                                |
| `StepEndEvent`   | `step_end`    | `step_name`, `step_index`, `status: 'success'\|'error'\|'paused'\|'skipped'`, `message`, `execution_time` |

##### Step-specific events (emitted from inside individual Steps)

| Class                       | `event_kind`           | Emitted by step                                                  |
| --------------------------- | ---------------------- | ---------------------------------------------------------------- |
| `AgentInitializedEvent`     | `agent_initialized`    | InitializationStep                                               |
| `StorageConnectionEvent`    | `storage_connection`   | StorageStep                                                      |
| `MemoryPreparedEvent`       | `memory_prepared`      | MemoryStep / ChatHistoryStep                                     |
| `SystemPromptBuiltEvent`    | `system_prompt_built`  | SystemPromptStep (`prompt_length`, `has_culture`, `has_skills`)  |
| `ContextBuiltEvent`         | `context_built`        | ContextStep (`context_length`, `has_knowledge_base`, `has_prior_outputs`) |
| `UserInputBuiltEvent`       | `user_input_built`     | UserInputStep (`input_type`, `has_images`, `has_documents`, `input_length`) |
| `ChatHistoryLoadedEvent`    | `chat_history_loaded`  | ChatHistoryStep (`history_count`)                                |
| `CacheCheckEvent`           | `cache_check`          | CacheStep (`cache_enabled`, `cache_method`, `cache_hit`, `similarity`, `input_preview`) |
| `CacheHitEvent`             | `cache_hit`            | CacheStep on hit                                                 |
| `CacheMissEvent`            | `cache_miss`           | CacheStep on miss                                                |
| `PolicyCheckEvent`          | `policy_check`         | UserPolicyStep / AgentPolicyStep (`action: ALLOW\|BLOCK\|REPLACE\|ANONYMIZE\|RAISE ERROR`) |
| `PolicyFeedbackEvent`       | `policy_feedback`      | When feedback-loop retry kicks in (`retry_count`, `max_retries`, `feedback_message`, `violated_policy`) |
| `LLMPreparedEvent`          | `llm_prepared`         | ModelStep (`default_model`, `requested_model`, `model_changed`)  |
| `ModelSelectedEvent`        | `model_selected`       | ModelStep (`model_name`, `provider`, `is_override`)              |
| `ToolsConfiguredEvent`      | `tools_configured`     | ToolsStep (`tool_count`, `tool_names`, `has_mcp_handlers`)       |
| `MessagesBuiltEvent`        | `messages_built`       | MessageStep (`message_count`, `has_system_prompt`, `has_memory_messages`, `is_continuation`) |
| `ModelRequestStartEvent`    | `model_request_start`  | ModelCallStep (`model_name`, `is_streaming`, `has_tools`, `tool_call_count`, `tool_call_limit`) |
| `ModelResponseEvent`        | `model_response`       | ModelCallStep (`has_text`, `has_tool_calls`, `tool_call_count`, `finish_reason`) |
| `ToolCallEvent`             | `tool_call`            | ProcessToolCallsStep (`tool_name`, `tool_call_id`, `tool_args`, `tool_index`) |
| `ToolResultEvent`           | `tool_result`          | ProcessToolCallsStep (`tool_name`, `tool_call_id`, `result`, `result_preview`, `execution_time`, `is_error`, `error_message`) |
| `ExternalToolPauseEvent`    | `external_tool_pause`  | ProcessToolCallsStep when external tool needs HITL               |
| `ReflectionEvent`           | `reflection`           | ReflectionStep (`reflection_applied`, `improvement_made`, `original_preview`, `improved_preview`) |
| `MemoryUpdateEvent`         | `memory_update`        | MemorySaveStep (`messages_added`, `memory_type`)                 |
| `CultureUpdateEvent`        | `culture_update`       | CultureStep (`culture_enabled`, `extraction_triggered`, `knowledge_updated`) |
| `ReliabilityEvent`          | `reliability`          | ReliabilityStep (`reliability_applied`, `modifications_made`)    |
| `CacheStoredEvent`          | `cache_stored`         | CacheStep when persisting (`cache_method`, `duration_minutes`)   |
| `ExecutionCompleteEvent`    | `execution_complete`   | FinalStep (`output_type`, `has_output`, `output_preview`, `total_tool_calls`, `total_duration`) |

##### Run-level events

| Class                | `event_kind`     | Emitted when                                              |
| -------------------- | ---------------- | --------------------------------------------------------- |
| `RunStartedEvent`    | `run_started`    | Pipeline start (`agent_id`, `task_description`)           |
| `RunCompletedEvent`  | `run_completed`  | Pipeline finished cleanly (`agent_id`, `output_preview`)  |
| `RunPausedEvent`     | `run_paused`     | HITL pause (`reason`, `requirements`, `step_name`)        |
| `RunCancelledEvent`  | `run_cancelled`  | `RunCancelledException` raised (`message`, `step_name`)   |

##### Streaming text/tool-call events (LLM-level deltas wrapped)

| Class                  | `event_kind`        | Key fields                                                            |
| ---------------------- | ------------------- | --------------------------------------------------------------------- |
| `TextDeltaEvent`       | `text_delta`        | `content`, `accumulated_content`, `part_index`                        |
| `TextCompleteEvent`    | `text_complete`     | `content`, `part_index`                                               |
| `ThinkingDeltaEvent`   | `thinking_delta`    | `content`, `part_index`                                               |
| `ToolCallDeltaEvent`   | `tool_call_delta`   | `tool_name`, `tool_call_id`, `args_delta`, `part_index` (all may be partial during streaming) |
| `FinalOutputEvent`     | `final_output`      | `output`, `output_type: 'text'\|'structured'\|'cached'\|'blocked'`    |

#### 4.2.3 `AgentRunEvent` enum and pydantic discriminators

`AgentRunEvent` is a string enum that mirrors every `event_kind` literal — it
exists so user code can do membership checks without importing each event
class:

```python
async for event in agent.stream(...):
    if event.event_kind == AgentRunEvent.TEXT_DELTA:
        print(event.content, end="")
```

Three pydantic-discriminated unions are also exported:

| Type alias            | Members                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| `PipelineEvent`       | `PipelineStartEvent \| PipelineEndEvent` (discriminated by `event_kind`) |
| `StepEvent`           | `StepStartEvent \| StepEndEvent`                                         |
| `StepSpecificEvent`   | All step-specific events                                                 |
| `LLMStreamEvent`      | All five streaming-delta events                                          |
| `AgentStreamEvent`    | The **union of everything** — this is the type the public streaming API yields |

A pydantic `TypeAdapter` named `AgentStreamEventTypeAdapter` is built at module
import (with `defer_build=True`) so external callers can validate streamed
events against the union without re-deriving the adapter each time.

#### 4.2.4 LLM-event → AgentEvent conversion

The bridge `convert_llm_event_to_agent_event(llm_event, accumulated_text="")`
takes a raw `ModelResponseStreamEvent` from `upsonic.messages` and wraps it
into one of the higher-level streaming events:

| `llm_event` type        | Inner part / delta                | Returned AgentEvent          |
| ----------------------- | --------------------------------- | ---------------------------- |
| `PartStartEvent`        | `TextPart`                        | `TextDeltaEvent` with the full part content |
| `PartStartEvent`        | `ThinkingPart`                    | `ThinkingDeltaEvent`         |
| `PartStartEvent`        | `ToolCallPart`                    | `ToolCallDeltaEvent` (full args) |
| `PartDeltaEvent`        | `TextPartDelta`                   | `TextDeltaEvent`             |
| `PartDeltaEvent`        | `ThinkingPartDelta`               | `ThinkingDeltaEvent`         |
| `PartDeltaEvent`        | `ToolCallPartDelta`               | `ToolCallDeltaEvent`         |
| `PartEndEvent`          | `TextPart`                        | `TextCompleteEvent`          |
| `FinalResultEvent`      | —                                 | **`None`** — `FinalOutputEvent` is emitted separately by `execute_stream` so this avoids duplicates |

Three convenience predicates are provided:

| Predicate                       | Returns True for                                              |
| ------------------------------- | ------------------------------------------------------------- |
| `is_text_event(event)`          | `TextDeltaEvent` or `TextCompleteEvent`                       |
| `is_tool_event(event)`          | `ToolCallEvent`, `ToolResultEvent`, `ToolCallDeltaEvent`      |
| `is_pipeline_event(event)`      | `PipelineStartEvent`, `PipelineEndEvent`, `StepStartEvent`, `StepEndEvent` |
| `extract_text_from_event(event)`| Returns `event.content` if event is text-related, else `None`. |

#### 4.2.5 Event registry for deserialization

`_EVENT_CLASS_REGISTRY` is a dict mapping every event class name to the class
itself. This is what `AgentEvent.from_dict()` uses to decode an event from
storage:

```python
event_class_name = data.get("__event_class__")
event_class = _EVENT_CLASS_REGISTRY.get(event_class_name)
return event_class(**filtered_kwargs)
```

It includes all ~38 concrete classes plus `AgentEvent` itself.

#### 4.2.6 `events/__init__.py` — lazy re-export

The `__init__.py` builds a `_EVENT_CLASSES` list of class names (also
`convert_llm_event_to_agent_event`) and lazily resolves them via `__getattr__`.
The same lazy mechanism reserves a slot for `EventEmitter` at
`upsonic.run.events.emitter`, but again, **`emitter.py` does not exist** in the
folder today — the slot is forward-compatible scaffolding.

---

### 4.3 `pipeline/` — execution stats

`pipeline/stats.py` defines a single dataclass:

```python
@dataclass
class PipelineExecutionStats:
    total_steps: int
    executed_steps: int = 0
    resumed_from: Optional[int] = None
    step_timing: Dict[str, float] = field(default_factory=dict)
    step_statuses: Dict[str, str] = field(default_factory=dict)
```

| Field             | Meaning                                                                         |
| ----------------- | ------------------------------------------------------------------------------- |
| `total_steps`     | How many steps the pipeline was configured to run.                              |
| `executed_steps`  | How many actually ran (≤ `total_steps`).                                        |
| `resumed_from`    | If this is a continuation of a paused/cancelled/errored run, the step index we resumed at. `None` for fresh runs. |
| `step_timing`     | `{step_name: seconds}` measured by `Step.run`.                                  |
| `step_statuses`   | `{step_name: "COMPLETED"\|"PAUSED"\|...}` — the StepStatus value as a string.   |

Plain `to_dict` / `from_dict` round-trip. The pipeline manager attaches an
instance to `AgentRunOutput.execution_stats` and updates it as steps run; user
code reads it to build per-step latency reports.

`pipeline/__init__.py` simply re-exports `PipelineExecutionStats`.

---

### 4.4 `tools/tools.py`

Already covered in §3.5 above. The folder has no `__init__.py` so it is a
*namespace package*; the canonical import path is
`upsonic.run.tools.tools.ToolExecution`.

---

## 5. Cross-file relationships

```text
                       ┌─────────────────────────────────┐
                       │          AgentRunOutput         │
                       │   (run/agent/output.py)         │
                       └─────────────────────────────────┘
                                │           │
        composes status         │           │ composes events
                                ▼           ▼
                    ┌─────────────┐    ┌─────────────────┐
                    │ RunStatus   │    │ AgentEvent (≈40 │
                    │ (run/base)  │    │ subclasses)     │
                    │             │    │ (run/events/    │
                    │ str-Enum    │    │  events.py)     │
                    └─────────────┘    └─────────────────┘
        composes step results            │
                                │        │
                                ▼        ▼
                    ┌──────────────────────────┐
                    │  PipelineExecutionStats  │
                    │  (run/pipeline/stats.py) │
                    └──────────────────────────┘

                                │
                                │ composes requirements
                                ▼
                    ┌─────────────────────┐
                    │ RunRequirement      │
                    │ (run/requirements   │
                    │  .py)               │
                    └─────────┬───────────┘
                              │ wraps a
                              ▼
                    ┌──────────────────────┐
                    │ ToolExecution        │
                    │ (run/tools/tools.py) │
                    └──────────────────────┘

                                │
                                │ composes input
                                ▼
                    ┌─────────────────────┐
                    │ AgentRunInput       │
                    │ (run/agent/input.py)│
                    └─────────────────────┘

  cancel.py is OUTSIDE this graph — it is a global
  registry keyed by run_id, not a field on the dataclasses.
```

| Relationship                                         | Where it shows up                                                    |
| ---------------------------------------------------- | -------------------------------------------------------------------- |
| `AgentRunOutput.status: RunStatus`                   | Default `RunStatus.running`. Mutated by `mark_*` methods.            |
| `AgentRunOutput.requirements: List[RunRequirement]`  | Populated when an `ExternalToolPauseEvent` / confirmation / user-input pause happens. |
| `RunRequirement.tool_execution: Optional[ToolExecution]` | The tool that needs HITL. The same `ToolExecution` may also live in `output.tools`. |
| `AgentRunOutput.events: List[AgentEvent]`            | Populated by every Step via the pipeline manager.                    |
| `AgentRunOutput.execution_stats: PipelineExecutionStats` | Built by the pipeline manager.                                    |
| `AgentRunOutput.input: AgentRunInput`                | Stored after `Agent.do(input=...)` constructs the input.             |
| `AgentRunOutput.task: Task`                          | The `Task` whose `status` is mirrored from `output.status` via `_sync_status_to_task`. |
| `cancel.register_run(run_id)` / `raise_if_cancelled(run_id)` | Run-id is the same string stored in `AgentRunOutput.run_id`. |
| `RunStatus.from_step_status(StepStatus)`             | Converts `agent.pipeline.step.StepStatus` (which is the per-step status) into a run-level status. |

---

## 6. Public API

Importable from `upsonic.run` (lazy, see `run/__init__.py`):

```python
from upsonic.run import (
    # Identity / state
    AgentRunInput,
    AgentRunOutput,
    RunStatus,

    # HITL
    RunRequirement,
    ToolExecution,

    # Cancellation
    RunCancellationManager,
    register_run,
    cancel_run,
    cleanup_run,
    raise_if_cancelled,
    is_cancelled,

    # Reserved (no concrete implementation in this folder yet)
    EventEmitter,
)
```

Importable from `upsonic.run.events`:

```python
from upsonic.run.events import (
    AgentEvent,                     # base class
    AgentRunEvent,                  # str-Enum of all event_kinds
    AgentStreamEvent,               # discriminated union
    convert_llm_event_to_agent_event,

    # Pipeline events
    PipelineStartEvent, PipelineEndEvent,
    StepStartEvent,    StepEndEvent,

    # Step-specific events
    AgentInitializedEvent, MemoryPreparedEvent, SystemPromptBuiltEvent,
    ContextBuiltEvent, UserInputBuiltEvent, ChatHistoryLoadedEvent,
    StorageConnectionEvent, CacheCheckEvent, CacheHitEvent, CacheMissEvent,
    PolicyCheckEvent, PolicyFeedbackEvent,
    LLMPreparedEvent, ModelSelectedEvent, ToolsConfiguredEvent,
    MessagesBuiltEvent, ModelRequestStartEvent, ModelResponseEvent,
    ToolCallEvent, ToolResultEvent, ExternalToolPauseEvent,
    ReflectionEvent, MemoryUpdateEvent, CultureUpdateEvent, ReliabilityEvent,
    CacheStoredEvent, ExecutionCompleteEvent,

    # Streaming deltas
    TextDeltaEvent, TextCompleteEvent, ThinkingDeltaEvent,
    ToolCallDeltaEvent, FinalOutputEvent,

    # Run-level events
    RunStartedEvent, RunCompletedEvent, RunPausedEvent, RunCancelledEvent,

    # Forwarded from upsonic.messages — exposed for stream-event interop
    PartStartEvent, PartDeltaEvent, PartEndEvent, FinalResultEvent,
)
```

Importable from `upsonic.run.pipeline`:

```python
from upsonic.run.pipeline import PipelineExecutionStats
```

Importable from `upsonic.run.tools.tools`:

```python
from upsonic.run.tools.tools import ToolExecution
```

Importable from `upsonic.run.agent.input` / `upsonic.run.agent.output`:

```python
from upsonic.run.agent.input  import AgentRunInput
from upsonic.run.agent.output import AgentRunOutput
```

### Method cheat-sheet — `AgentRunInput`

| Method                                            | Purpose                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------ |
| `__post_init__`                                   | Auto-converts `ImageUrl` / `DocumentUrl` → `BinaryContent` via httpx.    |
| `build_input(context_formatted=None)`             | Sync: read file paths, build the final `self.input` (str or list).       |
| `abuild_input(context_formatted=None)`            | Async equivalent (uses `aiofiles`).                                      |
| `to_dict()` / `from_dict(data)`                   | Round-trip through `ModelMessagesTypeAdapter` / `BinaryContentTypeAdapter`. |
| (static) `_is_url_type` / `_convert_url_to_binary` / `_aconvert_url_to_binary` | Internal helpers. |

### Method cheat-sheet — `AgentRunOutput`

(Truncated — see §4.1.2 for the full set.)

| Method                                | Purpose                                                                |
| ------------------------------------- | ---------------------------------------------------------------------- |
| `mark_paused(reason)`                 | `status → paused`, sets `pause_reason`, syncs to `task`.               |
| `mark_cancelled()`                    | `status → cancelled`, syncs to `task`.                                 |
| `mark_completed()`                    | `status → completed`, copies `accumulated_text → output` if streaming. |
| `mark_error(error)`                   | `status → error`, populates `error_details` and `metadata["error"]`.   |
| `start_new_run()`                     | Records the current `len(chat_history)` as a run boundary.             |
| `finalize_run_messages()`             | Slices the new-this-run messages out of `chat_history`.                |
| `add_message(msg)` / `add_messages(...)` | Append a single / many messages to the run.                         |
| `all_messages()` / `new_messages()`   | Return the run's message list (defensive copy).                        |
| `add_requirement(req)`                | Append a `RunRequirement` to `requirements`.                           |
| `has_pending_external_tools()`        | `len(active_requirements) > 0`.                                        |
| `start_usage_timer()` / `stop_usage_timer()` / `set_usage_time_to_first_token()` | Delegate to `TaskUsage`. |
| `increment_tool_calls(count=1)`       | Bumps `usage.tool_calls`.                                              |
| `add_model_execution_time(elapsed)` / `add_tool_execution_time(elapsed)` | Latency accounting. |
| `set_usage_cost(cost)`                | Adds (not sets) cost into `usage.cost`.                                |
| `to_dict(serialize_flag=False)` / `from_dict(data, deserialize_flag=False)` | Full round-trip. |
| `to_json(indent=2)`                   | `json.dumps(self.to_dict())`.                                          |

---

## 7. Integration with the rest of Upsonic

The run module is the **lingua franca** of the framework. Below is who imports
what (verified by `grep -rn "from upsonic.run"` over `src/upsonic`):

| Caller                                                  | What it imports                                              | Why                                              |
| ------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `upsonic/agent/agent.py`                                | `AgentRunInput`, `AgentRunOutput`, `RunStatus`, `register_run`, `cleanup_run`, `raise_if_cancelled`, `cancel_run`, `is_cancelled`, `RunEvent`, `TextDeltaEvent`, `ToolExecution`, `RunRequirement`, `AgentStreamEvent` | The `Agent` class is the **primary producer** of `AgentRunOutput`. |
| `upsonic/agent/pipeline/manager.py`                     | `AgentRunOutput`, `AgentEvent`, `PipelineExecutionStats`, `PipelineStartEvent`, `PipelineEndEvent`, `RunStartedEvent`, `RunCompletedEvent`, `RunCancelledEvent`, `StepStartEvent`, `StepEndEvent`, `RunRequirement`, `ToolExecution` | The `PipelineManager` **drives** the pipeline and emits the events. |
| `upsonic/agent/pipeline/step.py`                        | `AgentRunOutput`, `RunStatus`, `AgentEvent`, `StepStartEvent`, `StepEndEvent` | Base class `Step` instruments every step with start/end events and writes `current_step_result`. |
| `upsonic/agent/pipeline/steps.py`                       | `raise_if_cancelled` (~25 sites), `AgentRunInput`, `AgentEvent`, plus several specific event classes | Each concrete step calls `raise_if_cancelled` and emits its event. |
| `upsonic/agent/events.py` and `upsonic/agent/__init__.py`| All event classes and `AgentRunEvent` enum                  | Re-export of the streaming event surface.        |
| `upsonic/agent/context_managers/memory_manager.py`      | `AgentRunOutput`                                             | Reads `chat_history`, `memory_message_count`.    |
| `upsonic/agent/context_managers/call_manager.py`        | `AgentRunOutput`                                             | Reads/writes `tools`, `tool_call_count`.         |
| `upsonic/tasks/tasks.py`                                | `RunStatus`                                                  | `Task.status: RunStatus` is mirrored from `AgentRunOutput`. |
| `upsonic/chat/chat.py`                                  | `AgentRunOutput`, `AgentEvent`                               | The chat front-end consumes runs and replays events. |
| `upsonic/chat/cost_calculator.py` and `chat/schemas.py` | `AgentRunOutput`                                             | Pricing layer reads `usage` and `tools`.         |
| `upsonic/utils/agent/events.py`                         | All event classes                                            | Pretty-printing and formatting utilities.        |

A practical example — every public agent-execution method on `Agent`:

```python
async def do_async(self, input, **kwargs) -> AgentRunOutput:
    run_id = str(uuid4())
    register_run(run_id)                              # ← cancel.py
    try:
        run_output = AgentRunOutput(run_id=run_id, ...)  # ← run/agent/output.py
        run_output.input = AgentRunInput(user_prompt=input, ...)  # ← run/agent/input.py
        await pipeline_manager.execute(run_output)        # mutates run_output in place
        return run_output                                  # status now COMPLETED/PAUSED/...
    finally:
        cleanup_run(run_id)                                # ← cancel.py
```

…and the cancel side, called from any thread:

```python
def cancel(self, run_id: str) -> bool:
    return cancel_run(run_id)                              # ← cancel.py
```

---

## 8. End-to-end flow

Here is the complete trace of a single run, threading through every file in
this folder.

### 8.1 Caller side — fresh run

```python
from upsonic import Agent

agent = Agent(model="openai/gpt-4o")
out = await agent.do_async(
    input="Summarize this document.",
    documents=["./report.pdf"],
)
print(out.output)
```

1. **Agent constructs `AgentRunInput`** with `user_prompt="Summarize..."` and
   `documents=["./report.pdf"]`. `__post_init__` does nothing (no URL types
   yet — the path is a `str`).
2. **Agent generates `run_id`** (uuid4) and calls
   `register_run(run_id)` → `cancel.py` records `run_id → False`.
3. **Agent constructs `AgentRunOutput(run_id=run_id, ...)`** with default
   `status=RunStatus.running`, empty `chat_history`, empty `events`, empty
   `step_results`, `usage=None`.
4. **Agent attaches the input**: `run_output.input = run_input`.

### 8.2 Pipeline side — step execution

5. The `PipelineManager` starts. It builds a `PipelineExecutionStats(total_steps=N)`
   and stashes it on `run_output.execution_stats`.
6. It emits `PipelineStartEvent(total_steps=N, is_streaming=False)` and
   `RunStartedEvent(agent_id=...)` — both are appended to `run_output.events`.
7. It calls `run_output.start_usage_timer()` (creates a fresh `TaskUsage`).
8. For each `Step` in the pipeline:

   a. **Step.execute** calls `raise_if_cancelled(state.run_id)` first thing.
      If a separate thread called `cancel_run(run_id)`, this raises
      `RunCancelledException`, which the manager catches → emits
      `RunCancelledEvent` and `PipelineEndEvent(status='paused')` →
      `run_output.mark_cancelled()` → returns.

   b. The step emits `StepStartEvent`. It runs its body:
      - The `UserInputStep` calls `run_input.abuild_input()` to materialize
        `./report.pdf` into a `BinaryContent` and packs the `input` field as
        `["Summarize...", BinaryContent(report.pdf)]`. It emits
        `UserInputBuiltEvent(input_type="multipart", has_documents=True, ...)`.
      - The `ChatHistoryStep` loads memory messages into
        `run_output.chat_history`, emits `ChatHistoryLoadedEvent(history_count=K)`,
        then calls `run_output.start_new_run()` — recording
        `_run_boundaries.append(K)`.
      - The `CacheStep` checks the cache, emits `CacheCheckEvent` and either
        `CacheHitEvent` (in which case it short-circuits the run by setting
        `run_output.output` and emitting `ExecutionCompleteEvent`) or
        `CacheMissEvent`.
      - The `SystemPromptStep` builds the system prompt and emits
        `SystemPromptBuiltEvent(prompt_length=...)`.
      - The `ContextStep` builds RAG context (if any) and emits
        `ContextBuiltEvent(has_knowledge_base=True)`.
      - The `ModelStep` selects a model, emits `LLMPreparedEvent` and
        `ModelSelectedEvent`. The `ToolsStep` configures tools and emits
        `ToolsConfiguredEvent`.
      - The `MessagesStep` builds the model request payload and emits
        `MessagesBuiltEvent(message_count=M, is_continuation=False)`.
      - The `ModelCallStep` emits `ModelRequestStartEvent`, calls the LLM
        adapter (timed via `run_output.add_model_execution_time(elapsed)`),
        emits `ModelResponseEvent(has_text=True, has_tool_calls=False)`. The
        response is appended to `chat_history`.
      - The `ProcessToolCallsStep` (if there are tool calls) loops over each:
        emits `ToolCallEvent`, executes the tool (timing via
        `run_output.add_tool_execution_time(...)`), emits `ToolResultEvent`,
        appends a `ToolExecution` to `run_output.tools`, increments
        `run_output.tool_call_count`. If the tool has
        `external_execution_required=True`, it emits `ExternalToolPauseEvent`
        and creates a `RunRequirement(tool_execution=tool_exec)` which it
        appends via `run_output.add_requirement(req)`. Then the manager
        calls `run_output.mark_paused("external_tool")` and short-circuits
        the rest of the pipeline.

   c. **Step.run** wraps execute: on success it builds a `StepResult(status=COMPLETED)`,
      records it via `run_output.step_results.append(...)`,
      updates `run_output.execution_stats.step_timing[step_name]`, and emits
      `StepEndEvent(status='success', execution_time=t)`.

   d. On exception other than `RunCancelledException`, the step builds a
      `StepResult(status=ERROR, error=...)`, sets
      `run_output.mark_error(str(error))`, emits
      `StepEndEvent(status='error', message=...)`, and the manager re-raises.

### 8.3 Wrap-up

9. After the last step, the manager:
   - Calls `run_output.finalize_run_messages()` → slices
     `chat_history[run_boundary:]` into `run_output.messages`.
   - Calls `run_output.stop_usage_timer()`.
   - Emits `RunCompletedEvent(output_preview=...)` and
     `PipelineEndEvent(executed_steps=N, total_duration=...)`.
   - Calls `run_output.mark_completed()` → `status=RunStatus.completed`,
     `output` is finalized, `_sync_status_to_task()` mirrors to `task.status`.
10. The agent's `finally:` calls `cleanup_run(run_id)` → `cancel.py` removes
    the entry.
11. Returns `run_output`.

### 8.4 What the caller now sees

```python
out.status                       # RunStatus.completed
out.output                       # str — the summary
out.usage.total_tokens           # int
out.usage.cost                   # float (if pricing was on)
out.events                       # list of ~30 AgentEvent instances
out.tools                        # list of ToolExecution
out.tool_call_count              # int
out.step_results                 # list of StepResult, one per executed step
out.execution_stats              # PipelineExecutionStats with timing
out.messages                     # only the new messages from this run
out.chat_history                 # full session conversation
```

### 8.5 Continuation flow — paused run

If the run paused on an external tool:

```python
out.status                                       # RunStatus.paused
out.pause_reason                                 # "external_tool"
out.tools_awaiting_external_execution            # [ToolExecution(...)]
out.requirements                                 # [RunRequirement(...)]
```

The user resolves the requirement:

```python
req = out.active_requirements[0]
req.set_external_execution_result("Tool returned: ...")
# Or for confirmation:
#   req.confirm()  /  req.reject(note="not allowed")
# Or for user input: fill in req.user_input_schema fields with values.
```

Then continues:

```python
out = await agent.continue_run_async(out)
```

The manager re-enters the pipeline at the *paused step* (located via
`out.get_paused_step()`), `PipelineExecutionStats.resumed_from` is set, the
tool result is folded into `chat_history` (with `result_injected=True`
guarding against double-injection), and execution proceeds.

### 8.6 Cancellation flow

From any thread (e.g. a UI cancel button):

```python
agent.cancel(run_id)             # → cancel_run(run_id) → flag flips to True
```

The next `raise_if_cancelled(run_id)` checkpoint inside any step raises
`RunCancelledException`. The pipeline manager catches it and:

1. Emits `RunCancelledEvent(step_name=...)`.
2. Calls `out.mark_cancelled()`.
3. Returns the (partially populated) `AgentRunOutput`.

The caller then sees `out.status == RunStatus.cancelled`,
`out.is_cancelled is True`, `out.is_problematic is True`, and
`out.get_cancelled_step()` points at the step where cancellation took effect.

---

## 9. Quick-reference tables

### 9.1 Status × property

| `status`    | `is_paused` | `is_cancelled` | `is_complete` | `is_error` | `is_problematic` |
| ----------- | :---------: | :------------: | :-----------: | :--------: | :--------------: |
| `running`   |             |                |               |            |                  |
| `completed` |             |                |       ✓       |            |                  |
| `paused`    |      ✓      |                |               |            |         ✓        |
| `cancelled` |             |        ✓       |               |            |         ✓        |
| `error`     |             |                |               |      ✓     |         ✓        |

### 9.2 HITL pattern × `RunRequirement` properties

| Pattern              | `tool_execution` flags set                 | `req.needs_*` true                | Resolution                                         |
| -------------------- | ------------------------------------------ | --------------------------------- | -------------------------------------------------- |
| External tool        | `external_execution_required=True`         | `needs_external_execution`        | `req.set_external_execution_result(result)`       |
| User confirmation    | `requires_confirmation=True`               | `needs_confirmation`              | `req.confirm()` or `req.reject(note=...)`         |
| User input           | `requires_user_input=True`, `user_input_schema=[...]` | `needs_user_input`     | Fill `value` in each schema dict, set `tool_execution.answered=True` |

### 9.3 Event categories

| Category              | Count | Examples                                                     |
| --------------------- | :---: | ------------------------------------------------------------ |
| Pipeline-level        |   2   | `PipelineStartEvent`, `PipelineEndEvent`                     |
| Step-level            |   2   | `StepStartEvent`, `StepEndEvent`                             |
| Step-specific         |   ~22 | `AgentInitializedEvent`, `CacheCheckEvent`, `ModelResponseEvent`, ... |
| Run-level             |   4   | `RunStartedEvent`, `RunCompletedEvent`, `RunPausedEvent`, `RunCancelledEvent` |
| Streaming deltas      |   5   | `TextDeltaEvent`, `TextCompleteEvent`, `ThinkingDeltaEvent`, `ToolCallDeltaEvent`, `FinalOutputEvent` |

### 9.4 Cancel-registry API at a glance

| Function                          | Returns | Behavior                                                              |
| --------------------------------- | :-----: | --------------------------------------------------------------------- |
| `register_run(run_id)`            | `None`  | Records `run_id → False`. Idempotent overwrite.                       |
| `cancel_run(run_id)`              | `bool`  | Sets `run_id → True`; returns `False` if `run_id` was unknown.        |
| `is_cancelled(run_id)`            | `bool`  | `False` for unknown ids.                                              |
| `cleanup_run(run_id)`             | `None`  | Removes the entry. Safe on unknown ids.                               |
| `raise_if_cancelled(run_id)`      | `None`  | Raises `RunCancelledException` if cancelled. No-op otherwise.         |
| `RunCancellationManager.get_active_runs()` | `Dict[str, bool]` | Snapshot copy of the dict, for diagnostics. |

---

## 10. Notes and gotchas

- **`run/__init__.py` lazy load is real.** Don't `from upsonic.run import *`
  — `*` doesn't trigger the `__getattr__` path the way explicit name access
  does. Use `from upsonic.run import AgentRunOutput`.
- **`EventEmitter` is reserved but not implemented** in this folder. Both
  `run/__init__.py` and `run/events/__init__.py` route it to a non-existent
  `upsonic.run.events.emitter` module; accessing it raises
  `ModuleNotFoundError`. Treat it as forward scaffolding.
- **`tools/` has no `__init__.py`**. It is imported as a namespace package via
  `upsonic.run.tools.tools`. Do not `from upsonic.run.tools import ToolExecution`
  — it will fail.
- **`AgentRunInput.__post_init__` is synchronous** and will issue
  `httpx.get(...)` if `images` or `documents` contain `ImageUrl` /
  `DocumentUrl`. In an async context that *might* block briefly; prefer
  passing `BinaryContent` directly or using `abuild_input()` for file paths.
- **`build_input` filters string paths during `to_dict()`.** Local file paths
  are deliberately not serialized — only the materialized `BinaryContent` is.
  This keeps stored runs portable across machines.
- **`_run_boundaries` is critical** for the new-vs-historical message split.
  A bug where `start_new_run()` is not called means `messages` will silently
  include the entire memory window. The pipeline calls it from
  `ChatHistoryStep` after loading history.
- **`mark_completed()` only copies `accumulated_text` → `output` when streaming**.
  In non-streaming runs `output` must be set explicitly by the pipeline.
- **`TaskUsage` may arrive as a dict** (e.g. when an `AgentRunOutput` is
  rehydrated from storage and a usage method is called before deserialization
  is finalized). `_ensure_usage()` handles both cases by coercing dicts via
  `TaskUsage.from_dict`.
- **Event registry is import-time only.** New event classes must be added to
  `_EVENT_CLASS_REGISTRY` at the bottom of `events/events.py` *and* listed in
  the `_EVENT_CLASSES` of `events/__init__.py` for both round-trip
  deserialization and lazy import to work.
- **Cancellation is cooperative.** `raise_if_cancelled` only raises *between*
  steps. A long-running tool or LLM call inside a step will run to completion
  before the cancel takes effect. If you need eager cancellation, you must
  poll `is_cancelled(run_id)` from inside the tool/model call.
