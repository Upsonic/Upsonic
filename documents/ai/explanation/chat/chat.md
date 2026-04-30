---
name: chat-session-layer
description: Use when working on the conversational session orchestrator that wraps an Agent with persisted multi-turn state under `src/upsonic/chat/`. Use when a user asks to build a Chat session, manage session_id/user_id with Storage, stream responses, retry failed invocations, edit chat history, remove attachments, compute session metrics/cost, or handle HITL paused runs. Trigger when the user mentions Chat, ChatMessage, ChatAttachment, SessionManager, SessionState, SessionMetrics, InvokeResult, CostTracker, format_cost, format_tokens, invoke, astream, blocking vs streaming, retry_attempts, max_concurrent_invocations, reopen_session, clear_history, remove_attachment, AgentRunOutput, RunUsage, or AgentSession in the chat package.
---

# `src/upsonic/chat/` — Conversational Session Layer

## 1. What this folder is

The `upsonic.chat` package is the high-level conversational orchestrator of the Upsonic framework. While `upsonic.agent.Agent` is a stateless executor that turns one `Task` into one model run, the `Chat` class wraps an `Agent` with a *named, persisted, multi-turn session*: it owns a `session_id` / `user_id`, binds a `Storage` backend, drives a `Memory` instance, tracks cumulative usage and cost from the storage layer, exposes both blocking and streaming interfaces, and gives developers a clean `ChatMessage` view of history without leaking the internal `ModelMessage` representation.

Architecturally, this folder enforces a strict "storage is the single source of truth" rule. Token counts, cost, request counters, tool-call counts, run duration and time-to-first-token are *never* recomputed in the chat layer — they are read out of `AgentSession.usage` (a `RunUsage`) every time the developer touches a property. Local state in this folder is restricted to runtime concerns: the current `SessionState` enum value, the concurrent-invocation counter, the response-time list, and the wall-clock `start_time` / `end_time` / `last_activity_time` that are intentionally *not* persisted because they describe one Python process, not the durable session.

The package also owns the developer-facing message model. `ChatMessage` and `ChatAttachment` are dataclasses that flatten the multi-part `ModelRequest` / `ModelResponse` structures from `upsonic.messages.messages` into something a UI or notebook user can iterate over: a `role` ("user" or "assistant"), a `content` string, an optional list of `ChatAttachment` (image / audio / document / video / binary), an optional list of tool-call dicts, and a `message_index` that lets you point back at the raw record in storage for surgical editing (delete a turn, strip a PDF, swap the message list wholesale).

Finally, `chat/` is the layer that adds production polish on top of `Agent.do_async` / `Agent.astream`: input normalization (`str` or `Task` → `Task`), state-machine guards, max-concurrent-invocation enforcement, exponential-backoff retry on classified retryable errors, robust streaming generator cleanup, async context-manager support (`async with chat:`), reopen-after-close semantics for cumulative duration accounting, and a HITL escape hatch via `InvokeResult.run_output` for paused runs.

## 2. Folder layout

```
src/upsonic/chat/
├── __init__.py            # Lazy re-exports of the public surface (Chat, ChatMessage, etc.)
├── chat.py                # The Chat class — top-level session orchestrator
├── session_manager.py     # SessionManager + SessionState + SessionMetrics (storage-bound)
├── message.py             # ChatMessage + ChatAttachment (developer view of ModelMessage)
├── cost_calculator.py     # CostTracker static helpers + format_cost / format_tokens
└── schemas.py             # InvokeResult dataclass for HITL-aware blocking returns
```

| File | Lines | Primary export(s) | Role |
| --- | --- | --- | --- |
| `__init__.py` | 69 | `__getattr__`, `__all__` | PEP-562 lazy import facade |
| `chat.py` | 1045 | `Chat` | Stateful session orchestrator |
| `session_manager.py` | 1202 | `SessionState`, `SessionMetrics`, `SessionManager` | Storage-bound session bridge |
| `message.py` | 377 | `ChatAttachment`, `ChatMessage` | Developer-friendly message view |
| `cost_calculator.py` | 209 | `CostTracker`, `format_cost`, `format_tokens` | Cost formatting helpers |
| `schemas.py` | 17 | `InvokeResult` | HITL-aware blocking return type |

## 3. Top-level files — file-by-file walkthrough

### 3.1 `__init__.py`

The package init never imports its own modules at top level. Instead it defines `_get_chat_classes()` which performs the deferred imports of `Chat`, `InvokeResult`, `ChatMessage`, `ChatAttachment`, `SessionManager`, `SessionState`, `SessionMetrics`, `CostTracker`, `format_cost` and `format_tokens`, and a module-level `__getattr__(name: str) -> Any` (PEP-562) that calls that helper on demand and raises `AttributeError` listing the available keys when the name is missing. `TYPE_CHECKING`-guarded `from .chat import …` blocks satisfy static type checkers without paying any import cost at runtime. `__all__` mirrors the same ten names for `from upsonic.chat import *` consumers.

This pattern means `import upsonic.chat` is essentially free — heavy modules like `upsonic.storage.memory.memory` and `upsonic.messages.messages` are only loaded when a developer first reads `upsonic.chat.Chat`.

### 3.2 `schemas.py`

A single 17-line dataclass:

| Symbol | Kind | Description |
| --- | --- | --- |
| `InvokeResult` | `@dataclass` | Return value of `Chat.invoke` when `return_run_output=True`. Holds `text: str` and `run_output: Optional[AgentRunOutput]`. When the underlying agent run is paused (e.g. waiting on a confirmation tool), `run_output` is the `AgentRunOutput` so the caller can render HITL UI and later call `continue_run_async`. When the run completed normally, `run_output` is `None`. |

The `AgentRunOutput` import is `TYPE_CHECKING`-guarded to avoid the heavy run module unless the field is actually annotated.

### 3.3 `cost_calculator.py`

Pure cost utility module — *no local state*. Cost numbers in this codebase always flow through `upsonic.utils.usage`; this file is just the chat-flavored adapter on top of those primitives.

#### Class `CostTracker`

All methods are `@staticmethod`. The class exists purely as a namespace.

| Method | Signature | Behaviour |
| --- | --- | --- |
| `calculate_cost` | `(input_tokens: int, output_tokens: int, model: Model \| str \| None) -> str` | Delegates to `get_estimated_cost`. Returns the canonical `"~$0.0123"`-style string. |
| `calculate_cost_from_usage` | `(usage: RequestUsage, model=None) -> str` | Wraps `get_estimated_cost_from_usage`. |
| `calculate_cost_from_agent_run_output` | `(agent_run_output: AgentRunOutput, model=None) -> str` | Wraps `get_estimated_cost_from_run_output`. |
| `calculate_cost_from_run_usage` | `(run_usage: RunUsage, model=None) -> str` | Pulls `input_tokens` / `output_tokens` off the `RunUsage` (defaulting to `0`) and delegates to `get_estimated_cost`. Returns `"~$0.0000"` when `run_usage is None`. |
| `extract_cost_from_string` | `(cost_string: str) -> float` | Parses `"~$0.0123"` / `"$0.0123"` / `"0.0123"` into a `float`, returning `0.0` on failure. |
| `get_name` | `(model: Model \| str \| None) -> str` | Wraps `get_model_name`. |
| `get_provider_name` | `(model: Model \| None) -> str` | Inspects `model.provider_name`, falls back to splitting `model.model_name` on `/`, then to prefix heuristics: `gpt-` → `"openai"`, `claude-` → `"anthropic"`, `gemini-` → `"google"`, otherwise `"unknown"`. |

#### Module-level helpers

| Function | Signature | Behaviour |
| --- | --- | --- |
| `format_cost` | `(cost: float, currency: str = "USD") -> str` | Pure display formatter. `cost < 0.0001` → 6 decimals, `< 0.01` → 5 decimals, otherwise 4 decimals. Currency arg is accepted for API symmetry but ignored — the prefix is always `$`. |
| `format_tokens` | `(tokens: int) -> str` | `< 1_000` → raw integer string, `< 1_000_000` → `"X.YK"`, otherwise `"X.YM"`. |

### 3.4 `message.py`

This file owns the developer-facing message model. The crucial design rule is in the docstring: *"ChatMessage is ONLY for developer-facing display. Internal logic should use `ModelMessage` from `session.messages` in storage. ChatMessage is created on-demand when users access chat history."*

#### `ChatAttachment` (dataclass)

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `type` | `str` | required | One of `"image"`, `"audio"`, `"document"`, `"video"`, `"binary"`. |
| `identifier` | `str` | required | URL or path. For `BinaryContent` this falls back to `"binary_data"`. |
| `media_type` | `Optional[str]` | `None` | MIME type — populated only for `BinaryContent` items. |
| `index` | `int` | `0` | Position within the message; required for `Chat.remove_attachment(message_index, attachment_index)` to work. |

`to_dict()` serializes to `{type, identifier, index}` and conditionally adds `media_type`.

#### `ChatMessage` (dataclass)

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `content` | `str` | required | Joined plain text. |
| `role` | `Literal["user", "assistant"]` | required | Set by the `_from_model_request` / `_from_model_response` branch. |
| `timestamp` | `float` | required | `time.time()` at conversion time (note: not the original message timestamp). |
| `attachments` | `Optional[List[ChatAttachment]]` | `None` | Only populated for user messages. |
| `tool_calls` | `Optional[List[Dict[str, Any]]]` | `None` | Only populated for assistant messages. Each dict has `tool_name`, `tool_call_id`, `args`. |
| `metadata` | `Optional[Dict[str, Any]]` | `None` | Assistant-only: `model_name`, `provider_name`, `usage` (input/output tokens), `finish_reason`, `timestamp` (ISO format from the original `ModelResponse`). |
| `message_index` | `int` | `-1` | Position in `session.messages`; used by `Chat.delete_message`, `Chat.remove_attachment`, etc. |

##### Conversion methods

| Method | Signature | Behaviour |
| --- | --- | --- |
| `from_model_message` | `(cls, message: ModelMessage, message_index: int = -1)` | Type-dispatches: `ModelRequest` → `_from_model_request`, `ModelResponse` → `_from_model_response`, else `_from_unknown_message`. |
| `from_model_messages` | `(cls, messages: List[ModelMessage]) -> List[ChatMessage]` | Iterates with `enumerate`, sets `message_index` per item, swallows per-message conversion exceptions silently. |
| `_from_model_request` | private classmethod | Walks `message.parts`. For each `UserPromptPart` it calls `_extract_user_prompt_content`. `SystemPromptPart` parts are skipped (system prompts don't surface as chat messages). |
| `_extract_user_prompt_content` | `(cls, content, content_parts, attachments=None)` | Heart of the multimodal flattening. `None` and non-Sequence content are no-ops. A bare `str` is appended directly. For a sequence it iterates: `str` → text, `CachePoint` → skip, `ImageUrl` → text marker `"[Image: …]"` plus `ChatAttachment(type="image")`, `AudioUrl` / `DocumentUrl` / `VideoUrl` → analogous, `BinaryContent` → uses `media_type` and emits `"[Binary: id (mt)]"`. The local `attachment_index` counter ensures `ChatAttachment.index` is monotonic. Unknown items with a `.content` string attribute fall through into `content_parts`. |
| `_from_model_response` | private classmethod | Walks parts. `TextPart.content` is collected, `ToolCallPart` and `BuiltinToolCallPart` produce dicts with `args_as_dict()` (or `{}`), `ThinkingPart` is silently dropped. If text is empty and tool calls exist, the synthetic content `"Used N tool(s)"` is used. The metadata dict is opportunistically populated from `model_name`, `provider_name`, `usage`, `finish_reason`, `timestamp` if present. |
| `_from_unknown_message` | fallback classmethod | Stringifies the message; if it has a `.parts` attribute, joins each `part.content` for a best-effort body. Always emits `role="assistant"`. |
| `to_dict` | instance method | Produces a JSON-friendly dict (skips `None` blocks for `attachments`, `tool_calls`, `metadata`). |
| `__repr__` | instance method | `"ChatMessage(role='…', content='first 50 chars…', attachments=N)"` |

### 3.5 `session_manager.py`

The largest file in the folder. It owns three public symbols.

#### `SessionState(Enum)`

Pure runtime state machine — *not* persisted to storage.

| Member | Value | Meaning |
| --- | --- | --- |
| `IDLE` | `"idle"` | No invocation in flight; ready. |
| `AWAITING_RESPONSE` | `"awaiting_response"` | A blocking `invoke` call is mid-flight. |
| `STREAMING` | `"streaming"` | A streaming `invoke` / `stream` call is mid-flight. |
| `ERROR` | `"error"` | A previous invocation hit a non-retryable or all-retries-exhausted failure. `can_accept_invocation()` returns `False` until `reset_session()` is called. |

#### `SessionMetrics` (dataclass)

A snapshot DTO returned by `SessionManager.get_session_metrics()` and `Chat.get_session_metrics()`. It mirrors what's in storage's `RunUsage` but adds runtime fields.

| Field | Type | Source |
| --- | --- | --- |
| `session_id`, `user_id` | `str` | `SessionManager` constructor |
| `start_time`, `end_time` | `float`, `Optional[float]` | Local runtime timing |
| `total_input_tokens`, `total_output_tokens` | `int` | `RunUsage.input_tokens` / `output_tokens` |
| `cache_write_tokens`, `cache_read_tokens`, `reasoning_tokens` | `int` | `RunUsage.*` |
| `total_cost` | `float` | `RunUsage.cost or 0.0` |
| `total_requests`, `total_tool_calls` | `int` | `RunUsage.requests` / `tool_calls` |
| `run_duration`, `time_to_first_token` | `Optional[float]` | `RunUsage.duration` / `time_to_first_token` |
| `message_count` | `int` | `len(session.messages)` |
| `average_response_time` | `float` | Local `_response_times` average |
| `last_activity_time` | `float` | Local clock |

Computed properties: `duration` (uses `end_time or time.time()`), `total_tokens` (`input + output`), `messages_per_minute` (`message_count / (duration/60)` with zero-guard).

`to_dict()` always emits the core block, then conditionally adds `cache_write_tokens`, `cache_read_tokens`, `reasoning_tokens`, `run_duration`, `time_to_first_token` only when non-zero / non-`None`.

#### `SessionManager`

The bridge layer. Holds a reference to a `Storage` (required, raises `ValueError` if `None`) and never caches the session object — every read goes through `_get_session()` so callers always see fresh data even when other writers (the agent's `Memory`) update storage between calls.

##### Constructor

```python
SessionManager(
    session_id: str,
    user_id: str,
    storage: Storage,
    *,
    debug: bool = False,
    debug_level: int = 1,
    max_concurrent_invocations: int = 1,
)
```

Initializes runtime-only fields: `_state = SessionState.IDLE`, `_concurrent_invocations = 0`, `_max_concurrent_invocations`, `_start_time = time.time()`, `_end_time = None`, `_last_activity_time = time.time()`, `_response_times: List[float] = []`, `_is_closed = False`.

##### Storage helpers (private)

| Method | Signature | Behaviour |
| --- | --- | --- |
| `_get_session` | `() -> Optional[AgentSession]` | `storage.get_session(session_id, SessionType.AGENT, deserialize=True)`. |
| `_aget_session` | async | Uses `storage.aget_session` if `isinstance(storage, AsyncStorage)`, otherwise falls back to the sync method. |
| `_upsert_session` | `(AgentSession) -> None` | Stamps `updated_at = now_epoch_s()` then `storage.upsert_session(session, deserialize=True)`. |
| `_aupsert_session` | async | Same with `AsyncStorage` dispatch. |
| `_get_or_create_session` | `() -> AgentSession` | Fetches; if absent, constructs a fresh `AgentSession(session_id, user_id, SessionType.AGENT, created_at=now_epoch_s(), messages=[], runs={})` and upserts it. |
| `_aget_or_create_session` | async | Async sibling. |

##### State management

| Method | Behaviour |
| --- | --- |
| `state` (property) | Returns `_state`. |
| `transition_state(new_state)` | Logs and assigns `_state`. |
| `can_accept_invocation()` | `state != ERROR and concurrent < max`. |
| `start_invocation()` | Increments concurrent counter and refreshes activity. |
| `end_invocation()` | Decrements counter (clamped at `0`) and refreshes activity. |

##### Message access (sync + async pairs)

| Method | Returns | Notes |
| --- | --- | --- |
| `all_messages` (property) | `List[ChatMessage]` | Calls `ChatMessage.from_model_messages(session.messages)`. |
| `aget_all_messages()` | async `List[ChatMessage]` | |
| `get_recent_messages(count=10)` | `List[ChatMessage]` | Slices `session.messages[-count:]` (returns full list if `count <= 0`). |
| `aget_recent_messages(count=10)` | async | |
| `get_message_count()` / `aget_message_count()` | `int` | `len(session.messages)`. |
| `get_raw_messages()` / `aget_raw_messages()` | `List[ModelMessage]` | Returns a *copy* (`list(session.messages)`) for safe manipulation. |
| `set_messages(messages)` / `aset_messages(messages)` | `None` | Calls `_get_or_create_session`, assigns `session.messages = messages`, upserts. |
| `delete_message(message_index)` / `adelete_message(...)` | `bool` | Bounds-checks the index, deletes by index, upserts. Returns `False` on miss. |

##### Attachment surgery

`remove_attachment_from_message(message_index, attachment_index)` and its async sibling implement an in-place edit of `UserPromptPart.content`:

1. Fetch the session and validate `message_index`.
2. Verify the target is a `ModelRequest` (assistant responses don't have user attachments).
3. Find the first `UserPromptPart`. Bail if `content` is a plain `str` (no attachments possible).
4. Walk the content sequence, distinguish attachments via `isinstance(item, (ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent))`, and skip the one whose monotonic `current_attachment_idx` matches `attachment_index`.
5. If `current_attachment_idx <= attachment_index` after the walk, the index was out of range → return `False`.
6. Collapse a single-item string list back to a bare `str`, otherwise replace `part.content` with the new sequence. Upsert.

`remove_attachment_by_path(path)` and its async sibling do a bulk pass: for every `ModelRequest`, every `UserPromptPart`, drop attachments whose `identifier` (or `url` fallback) satisfies `path in identifier or identifier in path or path == identifier`. Returns the count removed. If no attachments remain, the part is set to `""` (string), if exactly one string remains, it's collapsed, otherwise the list is preserved.

##### History / lifecycle

| Method | Behaviour |
| --- | --- |
| `clear_history()` / `aclear_history()` | Sets `session.messages = []` and upserts. Other session data (usage, runs, user analysis) is preserved. |
| `reset_session()` / `areset_session()` | Calls `storage.delete_session(session_id)` (or `adelete_session` if `AsyncStorage`). Clears local state, sets a fresh `_start_time`, clears `_end_time`, `_is_closed = False`. |
| `close_session()` / `aclose_session()` | Idempotent: if not yet closed, sets `_end_time = time.time()` and `_is_closed = True`. Forces `_state = IDLE`, zeros `_concurrent_invocations`. Does *not* delete from storage. |
| `reopen_session()` / `areopen_session()` | If `_is_closed`, computes `closed_duration = self.duration` (which was frozen by `close_session`) then sets `_start_time = time.time() - closed_duration` and clears `_end_time` / `_is_closed`. The arithmetic preserves *cumulative* duration semantics so a session that ran 30s, was closed for 1h, and reopened will continue counting from 30s. |
| `is_session_active()` | `_state != ERROR`. |

##### Usage / cost properties (all read from `session.usage`, all return `0` / `None` if absent)

`input_tokens`, `output_tokens`, `total_tokens`, `total_cost`, `total_requests`, `total_tool_calls`, `run_duration`, `time_to_first_token`, `cache_write_tokens`, `cache_read_tokens`, `reasoning_tokens`. Each is a property that calls `_get_session()` and reads the corresponding `RunUsage` field. `get_usage()` / `aget_usage()` returns the full `RunUsage` object (or `None`).

##### Response timing

| Method | Behaviour |
| --- | --- |
| `start_response_timer()` | Returns `time.time()` (caller stores it). |
| `end_response_timer(start_time)` | Appends `time.time() - start_time` to `_response_times`, returns the duration. |
| `average_response_time` (property) | Mean of `_response_times`, or `0.0` if empty. |

##### Wall-clock properties

`start_time`, `end_time`, `last_activity` (seconds since `_last_activity_time`), `last_activity_time`, `duration` (uses `_end_time` if closed, otherwise `time.time() - _start_time`), `is_closed`.

##### Reporting

`get_session_metrics()` / `aget_session_metrics()` build a `SessionMetrics` from current `RunUsage` plus runtime fields. `get_session_summary()` produces a multi-line human-readable string; conditional lines for `run_duration`, `time_to_first_token`, cache tokens, reasoning tokens. `get_debug_info()` returns a dict aggregating session id, user, state, counters, message count, duration, timing, `is_closed`, `last_activity`, and `can_accept_invocation`.

### 3.6 `chat.py`

The main public facade. Imports `Task`, `Memory`, `Storage`, `InMemoryStorage`, `SessionManager`, `SessionState`, `ChatMessage`, `InvokeResult`. `Agent` and `AgentEvent` are TYPE_CHECKING-only imports to break import cycles (the agent layer imports the chat layer for prebuilt agents).

#### Class `Chat`

##### `__init__`

Long signature broken into three groups (see table). All inputs are validated up front: empty strings raise `ValueError`, `agent is None` raises, `max_concurrent_invocations < 1`, `retry_attempts < 0`, `retry_delay < 0`, `num_last_messages < 1` (when not `None`) all raise.

| Group | Parameters |
| --- | --- |
| Identity | `session_id: str`, `user_id: str`, `agent: Agent` |
| Storage | `storage: Optional[Storage] = None` (defaults to `InMemoryStorage()`) |
| Memory save flags | `full_session_memory=True`, `summary_memory=False`, `user_analysis_memory=False` |
| Memory load flags | `load_full_session_memory=True`, `load_summary_memory=None`, `load_user_analysis_memory=None` |
| Profile config | `user_profile_schema=None`, `dynamic_user_profile=False` |
| Memory tuning | `num_last_messages=None`, `feed_tool_call_results=False`, `user_memory_mode='update'\|'replace'` |
| Chat ops | `debug=False`, `debug_level=1`, `max_concurrent_invocations=1`, `retry_attempts=3`, `retry_delay=1.0` |

After validation it creates:

1. `self._storage` (caller-provided or `InMemoryStorage()`).
2. `self._memory = Memory(storage=self._storage, session_id, user_id, all the memory flags, model=agent.model, …)`.
3. **Side effect**: `self.agent.memory = self._memory`. This monkey-patches the agent so its runs persist through the chat's storage.
4. `self._session_manager = SessionManager(session_id, user_id, self._storage, debug, debug_level, max_concurrent_invocations)`.
5. `self._retry_attempts`, `self._retry_delay`, `self._max_concurrent_invocations` for retry logic.

##### Read-only state properties

All forward to `_session_manager` — there is *no* duplicate state in `Chat`.

| Property | Returns |
| --- | --- |
| `state` | `SessionState` |
| `all_messages` | `List[ChatMessage]` |
| `input_tokens`, `output_tokens`, `total_tokens` | `int` |
| `total_cost` | `float` |
| `total_requests`, `total_tool_calls` | `int` |
| `run_duration`, `time_to_first_token` | `Optional[float]` |
| `start_time`, `end_time` | `float`, `Optional[float]` |
| `duration` | `float` |
| `last_activity`, `last_activity_time` | `float` |
| `is_closed` | `bool` |

##### Reporting / queries

| Method | Behaviour |
| --- | --- |
| `get_usage()` | Returns `RunUsage` from session (`Any` typed). |
| `get_session_metrics()` | `SessionMetrics`. |
| `get_session_summary()` | Multiline `str`. |
| `get_recent_messages(count=10)` | `List[ChatMessage]`. |

##### History mutators (sync + `a`-prefixed async pairs)

| Method | Forwards to | Description |
| --- | --- | --- |
| `clear_history()` | `clear_history()` | Empties `session.messages`. |
| `reset_session()` | `reset_session()` | Deletes session record + resets timing. |
| `reopen()` | `reopen_session()` | Cumulative-duration restart. |
| `get_raw_messages()` | `get_raw_messages()` | Raw `ModelMessage` copy. |
| `set_messages(messages)` | `set_messages(...)` | Replace storage messages. |
| `delete_message(message_index)` | `delete_message(...)` | Drop one. |
| `remove_attachment(message_index, attachment_index)` | `remove_attachment_from_message(...)` | Surgical attachment removal. |
| `remove_attachment_by_path(path)` | `remove_attachment_by_path(...)` | Bulk path-based removal. |

##### Internal helpers

| Method | Behaviour |
| --- | --- |
| `_transition_state(new_state)` | Thin wrapper over `_session_manager.transition_state`. |
| `_normalize_input(input_data, context=None)` | `None` → `ValueError`; `str` → `Task(description=stripped, context=context)` (rejecting empty/whitespace); `Task` → returned as-is (with description validation); other → `TypeError`. |
| `_execute_with_retry(coro_func, *args, **kwargs)` | Retry loop. On exception: classify via `_is_retryable_error`. Non-retryable → state to `ERROR`, re-raise. Retryable & attempts left → `await asyncio.sleep(retry_delay * 2 ** attempt)` (exponential backoff). All attempts exhausted → state to `ERROR`, raise the last exception. |
| `_is_retryable_error(error)` | True if the error is `ConnectionError`, `TimeoutError`, `asyncio.TimeoutError`, or `OSError`, OR its lowercased message contains any of: `'timeout'`, `'connection'`, `'network'`, `'rate limit'`, `'temporary'`, `'service unavailable'`, `'internal server error'`, `'bad gateway'`, `'gateway timeout'`. |

##### `invoke` — the primary entry point

`invoke` carries five overload signatures so type-checkers can narrow the return type based on `stream`, `events`, and `return_run_output`:

| `stream` | `events` | `return_run_output` | Return |
| --- | --- | --- | --- |
| `False` | `False` | `True` | `InvokeResult` |
| `False` | `False` | `False` | `str` |
| `True` | `False` | — | `AsyncIterator[str]` |
| `True` | `True` | — | `AsyncIterator[AgentEvent]` |
| `False` | `True` | — | `AsyncIterator[AgentEvent]` (events forces stream=True internally) |

Runtime body:

1. Normalize flags: `events=True` forces `stream=True`. `stream` and `return_run_output` together → `return_run_output` is dropped (it only applies to blocking).
2. Concurrency / state guard via `_session_manager.can_accept_invocation()`. If `state == ERROR` raise `RuntimeError("Chat is in error state…")`. Otherwise raise `RuntimeError("Maximum concurrent invocations exceeded…")` with current/max counts.
3. `_normalize_input` → `Task`.
4. `_session_manager.start_invocation()`, transition state to `STREAMING` or `AWAITING_RESPONSE`.
5. `response_start_time = _session_manager.start_response_timer()`.
6. Dispatch:
   * `stream and events` → `_invoke_streaming_events(...)`.
   * `stream and not events` → `_invoke_streaming(...)`.
   * else → `await _invoke_blocking_async(..., return_run_output=return_run_output, **kwargs)`.

##### `_invoke_blocking_async`

Defines an inner `_execute()` coroutine. If `return_run_output`, calls `agent.do_async(task, debug=self.debug, return_output=True, **kwargs)`. If the result is an `AgentRunOutput` and `result.is_paused`, returns `InvokeResult(text=str(result.output), run_output=result)` — the HITL pause path. Otherwise returns `InvokeResult(text=…, run_output=None)`. If `return_run_output` is `False`, calls `agent.do_async(task, debug, **kwargs)` and stringifies the result. Wraps the call in `_execute_with_retry`. The `finally` always calls `end_response_timer`, `end_invocation`, and transitions state back to `IDLE`.

##### `_invoke_streaming` / `_invoke_streaming_events`

Both follow the same pattern. They define an inner async generator (`_execute_streaming` / `_execute_streaming_events`) that calls `agent.astream(task, debug=self.debug, events=False/True, **kwargs)` and yields chunks (`str`) or `AgentEvent`s. Then they wrap that in `_stream_with_retry` / `_stream_events_with_retry`:

* For each attempt up to `_retry_attempts + 1`, get a fresh generator and `async for` through it, yielding to the caller.
* On exception: best-effort `await stream_generator.aclose()` (suppressing close errors), null out the generator, then:
  * If the message contains `"context manager is already active"` (a known pydantic-ai streaming corner case), backoff is `retry_delay * 3 ** attempt`.
  * Else if attempts remain, backoff is `retry_delay * 2 ** attempt`.
  * Else re-raise the last exception.
* The outer `try/finally` always closes the generator if any, calls `end_response_timer`, `end_invocation`, and transitions to `IDLE`.

This means that even mid-stream cancellation, exceptions, or abandonment never leak open generators or stuck state.

##### `stream` — dedicated streaming entry point

Two overloads (events false/true). Same input normalization and state setup as `invoke`, then dispatches directly to `_invoke_streaming_events` (events=True) or `_invoke_streaming` (events=False). Skips the blocking path entirely.

##### Lifecycle

| Method | Behaviour |
| --- | --- |
| `close()` (async) | `await self._session_manager.aclose_session()`. Logs if debug. |
| `__aenter__` / `__aexit__` | Returns `self` on enter; calls `await self.close()` on exit (regardless of exception). |
| `__repr__` | `Chat(session_id='…', user_id='…', state=…, messages=N, cost=$0.XXXX)`. |

## 4. Subfolders

This package has no subfolders — all six files live directly under `src/upsonic/chat/`.

## 5. Cross-file relationships

```
Chat (chat.py)
  ├── owns 1 SessionManager (session_manager.py)
  │     ├── reads/writes AgentSession via Storage
  │     ├── exposes ChatMessage views via message.py
  │     ├── returns SessionMetrics dataclass
  │     └── tracks SessionState (runtime only)
  ├── owns 1 Memory (upsonic.storage.memory.memory)
  │     └── monkey-patched onto self.agent.memory
  ├── delegates to Agent (upsonic.agent.agent) for actual model runs
  ├── normalizes input into Task (upsonic.tasks.tasks)
  ├── may return InvokeResult (schemas.py) wrapping AgentRunOutput
  └── may yield AgentEvent (upsonic.run.events.events) when streaming events
```

* `Chat.__init__` is the only place that constructs both `Memory` and `SessionManager` against the same `Storage`. Both then read and write the same `AgentSession` row.
* `SessionManager` does not know about `Memory`; they coordinate purely through `AgentSession` rows in storage.
* `ChatMessage` only depends on `upsonic.messages.messages` types — it is otherwise independent of the rest of the package and can be imported standalone for serialization use cases.
* `CostTracker` is *not* used by `Chat` directly — `Chat.total_cost` reads `RunUsage.cost` via `SessionManager`. `CostTracker` is exposed for callers who want ad-hoc cost calculations outside a session.
* `InvokeResult` is the only entity that lets the chat layer surface a paused `AgentRunOutput` back to user code for HITL flows.

## 6. Public API

The names below are the supported import surface from `upsonic.chat`. Anything else is implementation detail.

| Symbol | Where defined | Kind | Purpose |
| --- | --- | --- | --- |
| `Chat` | `chat.py` | class | Session orchestrator. |
| `InvokeResult` | `schemas.py` | dataclass | HITL-aware blocking return. |
| `ChatMessage` | `message.py` | dataclass | Developer view of a turn. |
| `ChatAttachment` | `message.py` | dataclass | Developer view of an attachment. |
| `SessionManager` | `session_manager.py` | class | Storage-bound session bridge. |
| `SessionState` | `session_manager.py` | enum | Runtime state machine. |
| `SessionMetrics` | `session_manager.py` | dataclass | Snapshot DTO. |
| `CostTracker` | `cost_calculator.py` | class | Cost calculation helpers (static). |
| `format_cost` | `cost_calculator.py` | function | Currency-style float formatter. |
| `format_tokens` | `cost_calculator.py` | function | Token count abbreviator. |

## 7. Integration with the rest of Upsonic

* **`upsonic.agent.agent.Agent`** — `Chat` holds an `Agent` instance, replaces `agent.memory` with its own `Memory`, and dispatches all model work to `agent.do_async(task, ...)` (blocking) or `agent.astream(task, events=…)` (streaming).
* **`upsonic.tasks.tasks.Task`** — Strings passed to `Chat.invoke` / `Chat.stream` are wrapped in `Task(description=…, context=…)`. Task instances are passed through verbatim after description validation.
* **`upsonic.storage.base.Storage` / `AsyncStorage`** — `SessionManager` checks `isinstance(self._storage, AsyncStorage)` to decide whether to call sync or async storage methods. Both `Storage.get_session` / `upsert_session` / `delete_session` and their async counterparts are used.
* **`upsonic.storage.in_memory.in_memory.InMemoryStorage`** — Default storage when none is supplied to the `Chat` constructor.
* **`upsonic.storage.memory.memory.Memory`** — The chat layer instantiates `Memory` with the same storage and forwards every memory flag (`full_session_memory`, `summary_memory`, `user_analysis_memory`, `load_*`, `user_profile_schema`, `dynamic_user_profile`, `num_last_messages`, `feed_tool_call_results`, `user_memory_mode`, `model=agent.model`).
* **`upsonic.session.agent.AgentSession` / `upsonic.session.base.SessionType`** — `_get_or_create_session` constructs `AgentSession(session_id, user_id, SessionType.AGENT, created_at=now_epoch_s(), messages=[], runs={})` if the session doesn't exist.
* **`upsonic.usage.RunUsage`** — Every cost / token / request property on `Chat` and `SessionManager` reads through `session.usage: RunUsage`.
* **`upsonic.utils.usage`** — `CostTracker` is a thin shim over `get_estimated_cost`, `get_estimated_cost_from_usage`, `get_estimated_cost_from_run_output`, `get_model_name`.
* **`upsonic.utils.dttm.now_epoch_s`** — Used to stamp `created_at` and `updated_at` on `AgentSession` rows.
* **`upsonic.messages.messages`** — `ChatMessage` and `SessionManager` introspect `ModelRequest`, `ModelResponse`, `UserPromptPart`, `SystemPromptPart`, `TextPart`, `ToolCallPart`, `BuiltinToolCallPart`, `ThinkingPart`, plus the multimodal types `ImageUrl`, `AudioUrl`, `DocumentUrl`, `VideoUrl`, `BinaryContent`, `CachePoint`.
* **`upsonic.run.agent.output.AgentRunOutput`** — Returned by `agent.do_async(..., return_output=True)`; held inside `InvokeResult.run_output` when `result.is_paused`.
* **`upsonic.run.events.events.AgentEvent`** — Yielded by `Chat.invoke(..., stream=True, events=True)` and `Chat.stream(..., events=True)`.
* **`upsonic.utils.printing.debug_log` / `debug_log_level2`** — All debug logging goes through these helpers; `Chat` and `SessionManager` import them lazily inside `if self.debug:` blocks to keep cold-start cheap.
* **Top-level package** — `upsonic/__init__.py` re-exports `Chat` so user code can do `from upsonic import Chat, Agent`.

## 8. End-to-end flow

### 8.1 Constructing a chat

```python
chat = Chat(
    session_id="user123_session1",
    user_id="user123",
    agent=Agent("openai/gpt-4o"),
    storage=SqliteStorage("chat.db"),
    full_session_memory=True,
    summary_memory=True,
)
```

* Constructor validates the strings and integers.
* Creates `Memory(storage, session_id, user_id, …)`.
* **Mutates the agent**: `agent.memory = self._memory`. From this point any direct call to `agent.do(task)` outside the chat will also persist into this session.
* Creates `SessionManager(session_id, user_id, storage)`.
* Records local `_start_time`.

### 8.2 Blocking invocation

```python
text = await chat.invoke("Hello!")
```

1. `invoke` enters the non-stream branch.
2. `can_accept_invocation()` → `True` (state is `IDLE`, concurrent counter is `0 < 1`).
3. `_normalize_input("Hello!")` → `Task(description="Hello!")`.
4. `start_invocation()` increments concurrent counter to `1`. State transitions to `AWAITING_RESPONSE`.
5. `start_response_timer()` returns `t0 = time.time()`.
6. `_invoke_blocking_async` runs `_execute_with_retry(_execute)`:
   * `_execute` calls `agent.do_async(task, debug=self.debug, **kwargs)` and stringifies the result.
   * The agent internally pulls history out of `Memory` (which reads `session.messages`), runs the model, writes back the new `ModelRequest` + `ModelResponse` and updates `session.usage`.
   * Any retryable exception triggers exponential backoff (`retry_delay * 2 ** attempt`).
7. `_session_manager.end_response_timer(t0)` records the duration in `_response_times`.
8. `finally`: `end_invocation()` decrements the counter; state → `IDLE`.
9. Returns `text`.

### 8.3 HITL invocation

```python
result = await chat.invoke("Confirm this action", return_run_output=True)
if result.run_output and result.run_output.is_paused:
    decision = await prompt_user(result.run_output)
    final = await agent.continue_run_async(result.run_output, decision)
```

In step 6 of the blocking flow above, the inner `_execute()` path with `return_run_output=True` calls `agent.do_async(task, return_output=True)`. If the resulting `AgentRunOutput.is_paused`, it constructs `InvokeResult(text=str(result.output), run_output=result)` and returns. The chat returns to `IDLE` state immediately so subsequent `invoke` calls aren't blocked while the human deliberates. Resumption is the agent's responsibility (`continue_run_async`), not the chat's.

### 8.4 Streaming text chunks

```python
async for chunk in chat.invoke("Tell me a story", stream=True):
    print(chunk, end="", flush=True)
```

1. `invoke` sees `stream=True`, returns the `_invoke_streaming(...)` coroutine without awaiting it.
2. Caller starts iterating. The generator opens a fresh `agent.astream(task, events=False)` and yields each `str` chunk.
3. On any failure, the generator is closed cleanly, backoff is applied, and a new `agent.astream(...)` is opened for the next attempt.
4. On normal completion or final exception, the `finally` block closes the generator, ends the response timer, ends the invocation, and transitions state to `IDLE`.

### 8.5 Streaming events

```python
async for event in chat.invoke("Calculate 5+3", stream=True, events=True):
    if isinstance(event, FinalResultEvent):
        ...
```

Same as 8.4 but `_invoke_streaming_events` opens `agent.astream(task, events=True)` and yields raw `AgentEvent` objects (text deltas, tool-call starts/ends, etc.) for the caller to inspect.

### 8.6 Editing history

```python
# Drop the second turn entirely
chat.delete_message(1)

# Strip a specific PDF from any message that referenced it
chat.remove_attachment_by_path("/tmp/old.pdf")

# Or surgical removal
for msg in chat.all_messages:
    if msg.attachments:
        for att in msg.attachments:
            if att.type == "image":
                chat.remove_attachment(msg.message_index, att.index)
```

Each of these reads the live `session.messages`, mutates a copy, and upserts the session. The next `invoke` call will see the edited history because `Memory` (the agent's loader) re-reads the session every run.

### 8.7 Closing and reopening

```python
async with Chat(session_id="s1", user_id="u1", agent=agent) as chat:
    await chat.invoke("Hello")
# session is closed by __aexit__; storage row remains, _end_time is set

# Later:
chat2 = Chat(session_id="s1", user_id="u1", agent=agent, storage=same_storage)
chat2.reopen()           # if you've been re-using the same chat object
await chat2.invoke("Continue")
```

`close()` freezes `duration` at `_end_time - _start_time`. `reopen()` shifts `_start_time` backwards by that frozen duration so the cumulative `duration` keeps advancing as if the session never paused. Storage data (messages, usage) is left untouched, so the agent picks up exactly where it left off.

### 8.8 Reading metrics

```python
print(chat.total_cost)            # via session.usage.cost
print(chat.total_tokens)          # via session.usage.input_tokens + output_tokens
print(chat.run_duration)          # via session.usage.duration
print(chat.get_session_summary()) # multiline human-readable
print(chat.get_session_metrics()) # SessionMetrics dataclass
```

Every read goes through storage on demand; there is no cached scalar to invalidate.
