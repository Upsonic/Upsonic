---
name: tasks-package-reference
description: Use when working with the Upsonic Task class, its lifecycle, fields, or how it threads through agents, teams, workflows, RAG, cache, policy, guardrails, reliability, and pricing. Use when a user asks to define a Task, configure response_format, attachments, context, tools, skills, guardrails, caching (vector_search/llm_call), HITL pause/resume, RunStatus transitions, price_id_/task_id_, or to_dict/from_dict serialization. Trigger when the user mentions Task, src/upsonic/tasks/tasks.py, task_start, task_end, task_response, additional_description, add_canvas, add_tool_call, get_total_cost, RunStatus, is_problematic, is_completed, _run_id, continue_run_async, _original_input, _anonymization_map, _reliability_sub_agent_usage, TaskUsage, ToolManager, KnowledgeBase context, vector_search_top_k, guardrail, cache_method, cache_threshold, policy_apply_to_*, _task_todos, response_format, attachments_base64.
---

# `src/upsonic/tasks/` — Task definition & lifecycle

This document is a deep, file-level reference for the `tasks` package. It covers the
single public class `Task`, every attribute and method on it, and the way it is
threaded through every other component of Upsonic. Every `Agent.do(task)` call,
every `Team.do(task)` call, every `Workflow.run(...)` execution, and every prebuilt
autonomous agent ultimately wraps the user's request inside one of these
`Task` objects.

The `tasks` package itself is small: it ships exactly one source file
(`tasks.py`). All the heavy lifting around context formatting, RAG retrieval,
tool resolution, caching, guardrails, anonymization, policy enforcement, HITL
pause/resume, and pricing is performed by other Upsonic subsystems that read
from and write to the `Task` instance during a pipeline run. The Task is, in
effect, the **shared in-memory scratch-pad** for an entire `do()` invocation.

---

## 1. What this folder is

`Task` is the unit of work in Upsonic. A `Task` instance carries:

| Concern | Held on the Task |
|---|---|
| **Definition** | `description` (the user prompt), `response_format` (str or Pydantic model), `response_lang`, `attachments` (file paths), `context` (free-form list/dict/object), `agent` reference |
| **Identification** | `task_id_` (uuid4), `price_id_` (uuid4 — used by the pricing/usage subsystem to aggregate cost across sub-runs and reliability sub-agents), `_run_id` (set when the task is executed by a pipeline) |
| **Tools & skills** | `tools` (raw user-supplied list), `skills` (a `Skills` registry), `task_builtin_tools`, `registered_task_tools`, `_tool_manager` (a `ToolManager` instance owned by the task) |
| **Cache** | `enable_cache`, `cache_method` (`vector_search` or `llm_call`), `cache_threshold`, `cache_embedding_provider`, `cache_duration_minutes`, `_cache_manager`, `_cache_hit`, `_last_cache_entry`, `_original_input` |
| **Guardrails** | `guardrail` (callable validator), `guardrail_retries` |
| **Status / lifecycle** | `start_time`, `end_time`, `status` (a `RunStatus`), `is_paused`, `is_problematic`, `is_completed`, `_response`, `_tool_calls`, `_usage` |
| **Knowledge-base / RAG** | `query_knowledge_base`, `vector_search_top_k`, `vector_search_alpha`, `vector_search_fusion_method`, `vector_search_similarity_threshold`, `vector_search_filter`, `_context_formatted` |
| **Policy / safety** | `policy_apply_to_description`, `policy_apply_to_context`, `policy_apply_to_system_prompt`, `policy_apply_to_chat_history`, `policy_apply_to_tool_outputs`, `_policy_originals`, `_policy_blocked`, `_saved_context_for_policy`, `_anonymization_map` |
| **Reasoning / planning** | `enable_thinking_tool`, `enable_reasoning_tool`, `_task_todos` (a `TodoList` from the deep-agent planning toolkit) |
| **Misc** | `not_main_task`, `_promptlayer_request_id`, `_reliability_sub_agent_usage`, `_upsonic_tool_config`, `_upsonic_is_tool`, `_cached_result` |

All of these slots are declared on the `Task` Pydantic model with
`model_config = {"arbitrary_types_allowed": True}` so any subsystem can attach
its own internal objects (an `EmbeddingProvider`, a cache manager, a usage
tracker, …) onto the task without subclassing.

---

## 2. Folder layout (tree)

```
src/upsonic/tasks/
├── __pycache__/          # bytecode (excluded)
└── tasks.py              # Task class — the only source file in this package
```

That is the entire `tasks` package. There are no submodules, no helpers, no
factories. Everything lives in `tasks.py` (1418 lines).

---

## 3. Top-level files

### 3.1 `tasks.py`

`tasks.py` defines:

* The type aliases `CacheMethod = Literal["vector_search", "llm_call"]` and
  `CacheEntry = Dict[str, Any]`.
* The `Task` class — a `pydantic.BaseModel` with `arbitrary_types_allowed`.
* The module-level helper `_rebuild_task_model()` which calls
  `Task.model_rebuild()` at import time so forward references (`TodoList`,
  `Skills`, `ToolManager`, `ToolDefinition`, `TaskUsage`) resolve correctly.

The file imports lazily wherever heavy modules are touched. For example:

```python
# Inside Task.task_start
from upsonic.usage import TaskUsage  # delayed import

# Inside Task.additional_description
from upsonic.knowledge_base.knowledge_base import KnowledgeBase
```

This pattern keeps `from upsonic import Task` cheap regardless of how many
optional extras (`rag`, `storage`, …) are installed.

#### 3.1.1 Type aliases

```python
CacheMethod = Literal["vector_search", "llm_call"]
CacheEntry  = Dict[str, Any]
```

`CacheMethod` is the type of the `cache_method` field. `CacheEntry` is a
documentation alias for the dict structure used by `_last_cache_entry`.

#### 3.1.2 Class header

```python
class Task(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
```

Pydantic v2 model. `arbitrary_types_allowed` is required because `Task`
embeds non-pydantic objects: a `ToolManager`, a `Skills` registry, a `Canvas`,
custom embedding providers, and arbitrary tool callables.

---

## 4. Subfolders

There are **no subfolders** under `src/upsonic/tasks/` (other than the bytecode
cache). Anything you might expect in a "tasks runner" — schedulers, queues,
workers — lives elsewhere:

| Subsystem you might think lives here | Actual location |
|---|---|
| Pipeline / step runner | `src/upsonic/agent/pipeline/` |
| Tool execution | `src/upsonic/tools/` |
| Run cancellation / events | `src/upsonic/run/` |
| Task accounting (tokens / cost) | `src/upsonic/usage/` |
| HITL pause / resume | `src/upsonic/agent/pipeline/` (driven by `Task.is_paused` / `Task.status`) |

The `tasks` package is intentionally just the data class.

---

## 5. Cross-file relationships

Even though the package is one file, the `Task` is **the** shared object in
Upsonic. Below is a partial map of who reads/writes which slots on it.

| Subsystem | Reads | Writes |
|---|---|---|
| `Agent` / `Direct` (`src/upsonic/agent/agent.py`) | `description`, `response_format`, `response_lang`, `tools`, `skills`, `attachments`, `context`, `enable_thinking_tool`, `enable_reasoning_tool`, `is_paused`, `status`, `_run_id` | `agent`, `start_time`, `end_time`, `_response`, `_usage`, `_run_id`, `status` |
| `ToolManager` (`src/upsonic/tools/`) | `tools`, `task_builtin_tools`, `registered_task_tools` | `_tool_manager`, `registered_task_tools` |
| `KnowledgeBase` (`src/upsonic/knowledge_base/`) | `description`, `query_knowledge_base`, `vector_search_*` fields | `_context_formatted` (indirectly via the Context manager) |
| `CacheManager` (under `src/upsonic/cache/`) | `enable_cache`, `cache_method`, `cache_threshold`, `cache_embedding_provider`, `cache_duration_minutes`, `_original_input` | `_cache_manager`, `_cache_hit`, `_last_cache_entry` |
| `SafetyEngine` (`src/upsonic/safety_engine/`) | `policy_apply_to_*`, `description`, `context`, `_tool_calls` | `_policy_blocked`, `_policy_originals`, `_saved_context_for_policy`, `_policy_scope_tool_outputs`, `_anonymization_map` |
| `ReliabilityLayer` (`src/upsonic/reliability_layer/`) | `_response`, `description`, `response_format`, `price_id_` | `_reliability_sub_agent_usage`, `_response` (after editor / verifier rounds) |
| Pipeline `InitStep` (`src/upsonic/agent/pipeline/`) | (resets) | `start_time`, `end_time`, `_usage`, `_context_formatted`, `_cached_result`, `_policy_blocked` (via `task_start`) |
| Deep-agent planning (`src/upsonic/agent/deepagent/tools/planning_toolkit.py`) | `description` | `_task_todos` |
| Pricing / printing (`src/upsonic/utils/printing.py`) | `price_id_` | (cost ledger keyed on `price_id_`) |
| Canvas (`src/upsonic/...canvas`) | — | `tools` (appends canvas functions), `description` (appends canvas instruction) via `add_canvas` |
| HITL / continuation | `_run_id`, `is_paused`, `status` | `status`, `is_paused` |

The two **identifiers** are noteworthy:

* `task_id_` — uniquely identifies the task instance.
* `price_id_` — identifies the **billing/usage bucket**. Sub-tasks created by
  reliability rounds, verifier agents, or team delegations can re-use the
  parent's `price_id_` so cost aggregation in the pricing module rolls up to a
  single bucket per user-visible task.

---

## 6. Public API

### 6.1 Imports

```python
from upsonic import Task          # re-exported at the package root
# or
from upsonic.tasks.tasks import Task
```

### 6.2 `Task` constructor

```python
Task(
    description: str,
    attachments: Optional[List[str]] = None,
    tools: list[Any] = None,
    skills: Optional[Skills] = None,
    response_format: Union[Type[BaseModel], type[str], None] = str,
    response: Optional[Union[str, bytes]] = None,
    context: Any = None,
    _context_formatted: Optional[str] = None,
    price_id_: Optional[str] = None,
    task_id_: Optional[str] = None,
    not_main_task: bool = False,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    agent: Optional[Any] = None,
    response_lang: Optional[str] = None,
    enable_thinking_tool: Optional[bool] = None,
    enable_reasoning_tool: Optional[bool] = None,
    guardrail: Optional[Callable] = None,
    guardrail_retries: Optional[int] = None,
    is_paused: bool = False,
    enable_cache: bool = False,
    cache_method: Literal["vector_search", "llm_call"] = "vector_search",
    cache_threshold: float = 0.7,
    cache_embedding_provider: Optional[Any] = None,
    cache_duration_minutes: int = 60,
    _task_todos: Optional[TodoList] = None,
    vector_search_top_k: Optional[int] = None,
    vector_search_alpha: Optional[float] = None,
    vector_search_fusion_method: Optional[Literal['rrf', 'weighted']] = None,
    vector_search_similarity_threshold: Optional[float] = None,
    vector_search_filter: Optional[Dict[str, Any]] = None,
    query_knowledge_base: bool = True,
    policy_apply_to_description: Optional[bool] = None,
    policy_apply_to_context: Optional[bool] = None,
    policy_apply_to_system_prompt: Optional[bool] = None,
    policy_apply_to_chat_history: Optional[bool] = None,
    policy_apply_to_tool_outputs: Optional[bool] = None,
)
```

Validation performed in `__init__`:

| Check | Behavior |
|---|---|
| `guardrail` is not `None` and not callable | raises `TypeError("The 'guardrail' parameter must be a callable function.")` |
| `cache_method` not in `{"vector_search","llm_call"}` | raises `ValueError` |
| `cache_threshold` outside `[0.0, 1.0]` | raises `ValueError` |
| `enable_cache=True` + `cache_method="vector_search"` + no `cache_embedding_provider` | tries `auto_detect_best_embedding()`; if that fails, raises `ValueError` |
| `tools is None` | normalised to `[]` |
| `context is None` | normalised to `[]` |
| File / folder paths in `context` | extracted via `_extract_files_from_context`, removed from `context`, appended to `attachments` |
| Missing `price_id_` / `task_id_` | filled with `uuid.uuid4()` strings |

After `super().__init__(...)`, the constructor calls `self.validate_tools()` to
invoke any `__control__()` hooks on tool classes.

### 6.3 Attribute reference (every field on `Task`)

#### Definition / identity

| Attribute | Type | Default | Description |
|---|---|---|---|
| `description` | `str` | required | The user prompt / task statement. Mutated by `add_canvas` to append canvas instructions. Used as the input key for caching. |
| `attachments` | `Optional[List[str]]` | `None` → `[]` | File paths to attach. Files extracted from `context` are appended here. Convertible to base-64 via the `attachments_base64` property. |
| `tools` | `list[Any]` | `None` → `[]` | Raw tool list — functions, agents, MCP handlers, ToolKit instances, AbstractBuiltinTool instances. Processed at run time. |
| `skills` | `Optional[Skills]` | `None` | A `Skills` registry forwarded to the agent's runtime. |
| `response_format` | `Type[BaseModel] \| type[str] \| None` | `str` | Output schema — either `str` for plain text, or a Pydantic class for structured output. `None` is permitted and treated as "no constraint". |
| `response_lang` | `Optional[str]` | `"en"` (declared) but `None` from constructor | Hint for the model to respond in a particular language. |
| `_response` | `Optional[Union[str, bytes]]` | `None` | The model's final output. Set by `task_response` after the LLM returns. |
| `context` | `Any` | `[]` | Free-form context — list, dict, KnowledgeBase, file paths, folder paths, primitive values. RAG sources here are queried in `additional_description`. |
| `_context_formatted` | `Optional[str]` | `None` | The pre-rendered context block for the prompt. Set by the Context manager subsystem. Read-only via `context_formatted` property; settable via the property setter. |
| `price_id_` | `Optional[str]` | `uuid4()` | Pricing bucket id. Sub-tasks may share this with their parent. |
| `task_id_` | `Optional[str]` | `uuid4()` | Task instance id. Exposed via `task_id` and `id` properties. |
| `not_main_task` | `bool` | `False` | Marks helper / sub-tasks (verifier, editor, team-member tasks) so the printing layer can dim or hide them. |
| `agent` | `Optional[Any]` | `None` | Reference to the agent that owns this task. Set when an Agent picks the task up. |

#### Timing & status

| Attribute | Type | Default | Description |
|---|---|---|---|
| `start_time` | `Optional[int]` | `None` | Wall-clock seconds when `task_start` ran. |
| `end_time` | `Optional[int]` | `None` | Wall-clock seconds when `task_end` ran. |
| `status` | `Optional[RunStatus]` | `None` | One of `running`, `completed`, `paused`, `cancelled`, `error`. Used by `is_problematic` and `is_completed`. |
| `is_paused` | `bool` | `False` | True while a HITL approval is pending. |
| `_run_id` | `Optional[str]` | `None` | Pipeline run id. Persists through pause/resume so a fresh agent can `continue_run_async(task)`. Exposed via `run_id` property. |
| `_usage` | `Optional[TaskUsage]` | `None` | Per-task usage object (tokens, cost, durations). Created in `task_start`, stopped in `task_end`. |

#### Reasoning / planning

| Attribute | Type | Default | Description |
|---|---|---|---|
| `enable_thinking_tool` | `Optional[bool]` | `None` | Auto-attach the `plan_and_execute` thinking tool if `True`. |
| `enable_reasoning_tool` | `Optional[bool]` | `None` | Enable the in-prompt reasoning tool (provider-native chain-of-thought hooks). |
| `_task_todos` | `Optional[TodoList]` | `[]` | Deep-agent planning artifact. Type-checked in `__setattr__`: must be a `list` or `None`. |

#### Tools / skills internals

| Attribute | Type | Default | Description |
|---|---|---|---|
| `_tool_calls` | `List[Dict[str, Any]]` | `[]` | Per-call records appended by `add_tool_call`. Each record typically has `tool_name`, `params`, `tool_result`. Exposed via the `tool_calls` property. |
| `registered_task_tools` | `Dict[str, Any]` | `{}` | Name → tool object map maintained by the task's `ToolManager`. |
| `task_builtin_tools` | `List[Any]` | `[]` | List of `AbstractBuiltinTool` instances bound at the task level. |
| `_tool_manager` | `Optional[ToolManager]` | `None` | The task's own `ToolManager`. Lazily constructed by `_ensure_tool_manager`. Exposed via the `tool_manager` property + setter. |
| `_promptlayer_request_id` | `Optional[int]` | `None` | Returned by the PromptLayer integration when telemetry is enabled. |

#### Guardrails

| Attribute | Type | Default | Description |
|---|---|---|---|
| `guardrail` | `Optional[Callable]` | `None` | A function `(task) -> bool \| (bool, str)` run after the model produces output. Constructor enforces callability. |
| `guardrail_retries` | `Optional[int]` | `None` | How many times the agent will re-prompt to satisfy the guardrail before giving up. |

#### Cache

| Attribute | Type | Default | Description |
|---|---|---|---|
| `enable_cache` | `bool` | `False` | Master switch. |
| `cache_method` | `Literal["vector_search","llm_call"]` | `"vector_search"` | `"vector_search"` uses an embedding provider for similarity; `"llm_call"` uses an LLM judge. |
| `cache_threshold` | `float` | `0.7` | Similarity threshold (0–1) for a hit. |
| `cache_embedding_provider` | `Optional[Any]` | auto-detected | Embedding provider; required for `vector_search`. |
| `cache_duration_minutes` | `int` | `60` | TTL of cache entries. |
| `_cache_manager` | `Optional[Any]` | `None` | Set by the agent before execution. |
| `_cache_hit` | `bool` | `False` | True if the last response came from cache. Exposed via `cache_hit` property. |
| `_original_input` | `Optional[str]` | `description` | Snapshot of the input used as the cache key (so anonymization or RAG augmentation does not poison the key). |
| `_last_cache_entry` | `Optional[Dict[str, Any]]` | `None` | The cache entry hydrated for the last hit. |
| `_cached_result` | `bool` | `False` | True when the agent returned early from the cache check. |

#### Knowledge base / RAG

| Attribute | Type | Default | Description |
|---|---|---|---|
| `query_knowledge_base` | `bool` | `True` | If `False`, `additional_description` skips RAG queries even when `context` contains a `KnowledgeBase`. |
| `vector_search_top_k` | `Optional[int]` | `None` | Override per-task k. |
| `vector_search_alpha` | `Optional[float]` | `None` | Hybrid search dense/sparse mix. |
| `vector_search_fusion_method` | `Optional[Literal['rrf','weighted']]` | `None` | Hybrid fusion algorithm. |
| `vector_search_similarity_threshold` | `Optional[float]` | `None` | Minimum acceptable score. |
| `vector_search_filter` | `Optional[Dict[str, Any]]` | `None` | Metadata-level filter passed to the vector store. |

#### Policy / safety

| Attribute | Type | Default | Description |
|---|---|---|---|
| `policy_apply_to_description` | `Optional[bool]` | `None` | If `True`, run input policies on `description`. |
| `policy_apply_to_context` | `Optional[bool]` | `None` | Run policies on `context`. |
| `policy_apply_to_system_prompt` | `Optional[bool]` | `None` | Run policies on the agent's system prompt. |
| `policy_apply_to_chat_history` | `Optional[bool]` | `None` | Run policies on prior chat messages. |
| `policy_apply_to_tool_outputs` | `Optional[bool]` | `None` | Run policies on tool results before re-feeding to the model. |
| `_saved_context_for_policy` | `Optional[str]` | `None` | Pre-redaction context snapshot, kept for restoration. |
| `_policy_originals` | `Optional[Dict[str, Any]]` | `None` | Original values of fields that policy may have rewritten. |
| `_policy_scope_tool_outputs` | `bool` | `False` | Internal flag set when a policy run targets tool outputs only. |
| `_policy_blocked` | `bool` | `False` | True when a policy outright denied the task. |
| `_anonymization_map` | `Optional[Dict[int, Dict[str, str]]]` | `None` | `idx → {"original","anonymous","pii_type"}`. Used to reverse PII anonymization on the model response before it reaches the user. |

#### Reliability & misc

| Attribute | Type | Default | Description |
|---|---|---|---|
| `_reliability_sub_agent_usage` | `Optional[Any]` | `None` | Aggregated usage from verifier/editor sub-agents. |
| `_upsonic_tool_config` | `Optional[Any]` | `None` | When a Task is itself wrapped as a tool (delegation), this holds tool-side configuration. |
| `_upsonic_is_tool` | `bool` | `False` | True if this task acts as a callable tool inside another agent. |

### 6.4 Properties (read-only and computed)

| Property | Backed by | Description |
|---|---|---|
| `usage` | `_usage` | The `TaskUsage` object. |
| `duration` | `_usage.duration` or `end_time - start_time` | Total wall-clock seconds. |
| `model_execution_time` | `_usage.model_execution_time` | Time spent waiting on the LLM. |
| `tool_execution_time` | `_usage.tool_execution_time` | Time spent inside tool calls. |
| `upsonic_execution_time` | `_usage.upsonic_execution_time` | Framework overhead. |
| `id` | `task_id_` | Alias of `task_id`. |
| `task_id` | `task_id_` | Public id. |
| `price_id` | `price_id_` | Pricing-bucket id. |
| `is_problematic` | `status` | `True` iff status ∈ `{paused, cancelled, error}`. Means the next call must be `continue_run_async(task)`, not `do_async(task)`. |
| `is_completed` | `status` | `True` iff status == `completed`. A completed task cannot be re-run. |
| `tool_manager` (+ setter) | `_tool_manager` | Exposes the task's `ToolManager`. |
| `context_formatted` (+ setter) | `_context_formatted` | The rendered context block injected into the prompt. |
| `run_id` (+ setter) | `_run_id` | Pipeline run id used for HITL continuation. |
| `attachments_base64` | `attachments` | Reads each file in binary mode and returns a list of base-64 strings. Failed files emit a warning_log and are skipped. |
| `response` | `_response` | Returns the final output (str passthrough; bytes / structured returned as-is). |
| `cache_hit` | `_cache_hit` | True iff the last call hit the cache. |
| `tool_calls` | `_tool_calls` | Lazy list of all recorded tool invocations. |
| `total_cost` | `get_total_cost()` | Estimated USD cost from the printing module's pricing ledger. |
| `total_input_token` | `get_total_cost()` | Total input tokens. |
| `total_output_token` | `get_total_cost()` | Total output tokens. |

### 6.5 Methods

| Method | Purpose |
|---|---|
| `__setattr__(name, value)` | Type-guards `_task_todos` to be `list` or `None`. |
| `_is_file_path(item)` (static) | True if `item` is a string referring to an existing readable file. Heuristically raises `FileNotFoundError` when the string looks like a file path (extension or separator) but the file is missing. |
| `_is_folder_path(item)` (static) | Mirror of the above for directories. |
| `_get_files_from_folder(folder_path)` (static) | Recursively walks a folder via `os.walk` and returns every file path. Raises `FileNotFoundError` on `OSError`/`PermissionError`. |
| `_extract_files_from_context(context)` (static) | Walks list/dict/string `context`, pulls out file paths and folder contents, and returns `(cleaned_context, extracted_files)`. Drives the auto-attachment behavior. |
| `validate_tools()` | Iterates `self.tools` and, for any tool that is a class or instance with a `__control__` callable, invokes it. (The boolean returned is captured but not currently asserted on, leaving room for tools to raise from `__control__`.) |
| `_ensure_tool_manager()` | Lazily constructs a `ToolManager` and stores it in `_tool_manager`. |
| `get_tool_defs()` | Returns `self._tool_manager.get_tool_definitions()` or `[]`. |
| `get_skill_metrics()` | Returns `{name: metrics_dict}` from `self.skills.get_metrics()`. |
| `add_tools(tools)` | Append-only: normalises to list, initialises `self.tools` if needed, dedupes by identity. |
| `remove_tools(tools, agent=None)` | Removes by name (`str`), by callable, by agent, by MCP handler, by class, or by `AbstractBuiltinTool`. Splits into builtin vs regular, uses the task's `ToolManager` for the regular case, filters `task_builtin_tools` by `unique_id` for builtins, and finally rewrites `self.tools`. The `agent=` parameter is deprecated. |
| `additional_description(client)` (async) | Iterates `self.context`, for every `KnowledgeBase` calls `setup_async()` (if `query_knowledge_base`) and `query_async(self.description, task=self)`. Returns a single string `The following is the RAG data: <rag>...</rag>` with each result formatted as `[i] [metadata: source: …, page: …, chunk_id: …, score: …] <text>`. |
| `canvas_agent_description()` | Returns the canvas agent system prompt: `"You are a canvas agent. You have tools. You can edit the canvas and get the current text of the canvas."`. |
| `add_canvas(canvas)` | Adds canvas tool functions and the canvas system prompt — guarded against duplicates by checking `tools` and `description` for prior insertion. |
| `task_start(agent)` | Initialises a fresh pipeline run: sets `start_time = time.time()`, clears `end_time`, builds a new `TaskUsage`, starts its timer, resets `_context_formatted`, `_cached_result`, `_policy_blocked`, and adds the agent's canvas if any. Called by the pipeline's `InitStep`. |
| `task_end()` | Sets `end_time` if unset and stops the `TaskUsage` timer. |
| `task_response(model_response)` | Sets `_response = model_response.output`. |
| `set_cache_manager(cache_manager)` | Stores a `CacheManager` on the task. |
| `get_cached_response(input_text, llm_provider=None)` (async) | Delegates to `_cache_manager.get_cached_response(...)`. On hit, sets `_cache_hit = True` and stores the entry under `_last_cache_entry`. |
| `store_cache_entry(input_text, output)` (async) | Delegates to `_cache_manager.store_cache_entry(...)` with the configured method/embedding provider. |
| `get_cache_stats()` | Returns `{total_entries, cache_hits, cache_misses, hit_rate, cache_method, cache_threshold, cache_duration_minutes, session_id}` (zeros if no cache manager) plus `cache_hit` from the live state. |
| `clear_cache()` | Calls the cache manager's `clear_cache()` and resets `_cache_hit`. |
| `_pickle(obj)` (static) | `cloudpickle.dumps` + base64; returns `{"__pickled__": "..."}` or `None`. |
| `_unpickle(obj)` (static) | Inverse of `_pickle`; returns `obj` unchanged if not a `{"__pickled__": ...}` dict. |
| `to_dict(serialize_flag=False)` | Full task → dict. With `serialize_flag=True`, cloudpickles `tools`, `skills`, `registered_task_tools`, `task_builtin_tools`, `guardrail`, `_tool_manager`, `response_format`. With `serialize_flag=False`, returns simple JSON-friendly placeholders for those fields and uses tagged shapes (`__builtin_type__`, `__pydantic_type__`, `__type__`) for `response_format`. |
| `from_dict(data, deserialize_flag=False)` (classmethod) | Inverse of `to_dict`. Re-imports Pydantic / built-in types when `response_format` was JSON-tagged; falls back to `str` with a `warning_log` on import failure. Restores `status`, `_task_todos` (validating each `Todo`), `registered_task_tools`, `task_builtin_tools`, `_tool_manager`, every `_*` private flag, `_usage` (via `TaskUsage.from_dict`), and `agent`/`cache_embedding_provider`/`_cache_manager` when present. |
| `add_tool_call(tool_call)` | Append `tool_call` (dict with `tool_name`, `params`, `tool_result`) to `_tool_calls`. |
| `get_total_cost()` | Looks up `get_price_id_total_cost(self.price_id)` from `upsonic.utils.printing`. Returns `None` if `price_id_` is unset. |

### 6.6 Module-level helper

```python
def _rebuild_task_model():
    """Rebuild Task model after all dependencies are imported."""
    try:
        Task.model_rebuild()
    except Exception:
        pass

_rebuild_task_model()
```

This runs at import time so Pydantic resolves the `TYPE_CHECKING` forward refs
(`TodoList`, `Skills`, `ToolManager`, `ToolDefinition`, `TaskUsage`).

---

## 7. Integration with the rest of Upsonic

### 7.1 Agent.do(task) — the central path

Every direct execution path in Upsonic flows through one of these entrypoints
on `Agent` / `Direct`:

* `Agent.do(task)` — synchronous wrapper.
* `Agent.do_async(task)` — fresh pipeline run.
* `Agent.continue_run_async(task)` — resumes a paused / cancelled / errored
  task, gated on `task.is_problematic`.
* `Agent.print_do(task)` — `do` with rich console printing.
* `Agent.do_stream(task)` / `Agent.do_stream_async(task)` — streaming variants.

In all cases the input is a `Task`. The agent:

1. Reads `task.tools`, `task.skills`, `task.attachments`, `task.context`,
   `task.response_format`, etc., to seed the pipeline.
2. Calls `task.task_start(self)` (via `InitStep`) on a fresh run.
3. Drives the pipeline: context formatting → policy → cache lookup → RAG →
   tool prep → model call → tool dispatch loop → guardrail → reliability layer
   → cache store → policy on outputs → de-anonymization.
4. Calls `task.task_response(model_response)` and `task.task_end()`.
5. Returns `task.response`.

If at any point a HITL pause is needed (a tool requires approval, a verifier
escalates, etc.), the pipeline sets `task.is_paused = True` and
`task.status = RunStatus.paused`, which makes `task.is_problematic == True`.
The next `continue_run_async(task)` skips `InitStep` (so `task_start` is not
re-run, preserving `_usage`) and resumes from the recorded step.

### 7.2 Team.do(task)

Team coordination wraps a primary `Task` and creates **sub-tasks** (one per
team-member call). The sub-tasks inherit `price_id_` from the parent so the
pricing module aggregates cost into one bucket. They are created with
`not_main_task=True` so the printer presents them as nested.

### 7.3 Workflow

A Workflow is a sequence/graph of `Task` instances. Each task carries its own
status; the workflow engine reads `task.is_completed` and `task.is_problematic`
to decide when to advance, skip, or surface a problem to the user.

### 7.4 Knowledge base / RAG

`Task.additional_description(client)` is called by the prompt-builder to
expand any `KnowledgeBase` instance found inside `context`. RAG output appears
in the prompt as `<rag>[i] [metadata: source: ..., page: ..., chunk_id: ..., score: ...] <text></rag>`.
Each KB query receives `task=self` so the KB can apply per-task overrides
(`vector_search_top_k`, `vector_search_alpha`, `vector_search_fusion_method`,
`vector_search_similarity_threshold`, `vector_search_filter`).

### 7.5 Cache

The agent picks an embedding provider (or accepts the auto-detected one),
constructs a `CacheManager`, and calls `task.set_cache_manager(...)`. Before
the LLM call:

```python
cached = await task.get_cached_response(task._original_input, llm_provider)
if cached is not None:
    task._cached_result = True
    return cached
```

After a successful generation:

```python
await task.store_cache_entry(task._original_input, task.response)
```

The key is `_original_input` (the description before policy / RAG augmentation)
so the cache remains stable across cosmetic prompt changes.

### 7.6 Safety engine

The safety engine inspects `task.policy_apply_to_*` flags to decide which
prompt segments to scan. On a redaction it stashes the originals into
`_policy_originals` and `_saved_context_for_policy`. On a hard block it sets
`_policy_blocked = True`, which the agent surfaces as a refusal response.
On PII anonymization it builds `_anonymization_map`, which is then used to
reverse-substitute pseudonyms in `_response` before returning to the user.

### 7.7 Reliability layer

Verifier and editor sub-agents are themselves run as `Task` instances with
`not_main_task=True` and the parent's `price_id_`. Their aggregated usage is
written to `_reliability_sub_agent_usage`. A successful editor pass replaces
`_response` with the corrected output.

### 7.8 Pricing & accounting

`price_id_` is the join key for the pricing ledger maintained inside
`upsonic.utils.printing`. `total_cost`, `total_input_token`, and
`total_output_token` all delegate to `get_price_id_total_cost(price_id)`. The
in-flight `TaskUsage` (`_usage`) holds wall-clock + token totals that
`task_end()` finalises.

### 7.9 Prebuilt autonomous agents

Prebuilts under `src/upsonic/prebuilt/<agent>/` build a `Task` from a template
(system prompt + first message). The base class `PrebuiltAgentBase`
populates `description`, optional `attachments`, `tools` (skills), and a
`response_format` Pydantic model, then calls `Agent.do(task)`. See
`Docs/ai/new_prebuilt_agent_adding.md` for the canonical pattern.

### 7.10 Serialization / persistence

`to_dict` / `from_dict` are how tasks travel across process boundaries:

* In-memory or JSON sessions in `src/upsonic/storage/` use
  `serialize_flag=False`.
* Cross-process or remote-worker scenarios use `serialize_flag=True` so
  unpicklable objects (functions, classes, ToolKits, embedding providers, the
  guardrail callable) survive the round-trip via cloudpickle.

```python
data = task.to_dict(serialize_flag=True)
restored = Task.from_dict(data, deserialize_flag=True)
```

`response_format` has special tagged-dict shapes (`__builtin_type__`,
`__pydantic_type__`, `__type__`) so JSON-only consumers can still survive a
`str` round-trip and re-import a Pydantic class by its `(module, name)`.

---

## 8. End-to-end flow of a Task lifecycle

### 8.1 States

The lifecycle is governed by `RunStatus` (defined in `src/upsonic/run/base.py`):

```python
class RunStatus(str, Enum):
    running   = "RUNNING"
    completed = "COMPLETED"
    paused    = "PAUSED"
    cancelled = "CANCELLED"
    error     = "ERROR"
```

Two derived booleans on `Task`:

* `is_completed` ⇔ `status == completed`
* `is_problematic` ⇔ `status ∈ {paused, cancelled, error}` — the task must be
  resumed via `continue_run_async`, not re-run via `do_async`.

### 8.2 The fresh pipeline (start → response → end)

```text
User code:
    task = Task("write a blog post", response_format=BlogPost,
                tools=[search_web], enable_cache=True,
                context=[my_kb, "/path/to/style_guide.pdf"])
    agent = Agent("openai/gpt-4o")
    result = await agent.do_async(task)
```

What happens:

1. **Construction** (`Task.__init__`)
   * `description = "write a blog post"`.
   * `context` walked: `my_kb` (KnowledgeBase) is kept;
     `"/path/to/style_guide.pdf"` is recognised as a file by
     `_is_file_path` and moved into `attachments`.
   * `tools = [search_web]`; passed through `validate_tools` (no
     `__control__` so a no-op).
   * `enable_cache=True` and `cache_method="vector_search"` →
     `auto_detect_best_embedding()` is called to pick an embedding provider.
   * `task_id_` and `price_id_` get fresh UUIDs.

2. **Agent.do_async(task)** entry
   * Sets `task.agent = agent`, `task._cache_manager = CacheManager(...)`,
     and enters the pipeline.

3. **InitStep** invokes `task.task_start(agent)`:
   ```python
   self.start_time = time.time()
   self.end_time = None
   self._usage = TaskUsage()
   self._usage.start_timer()
   self._context_formatted = None
   self._cached_result = False
   self._policy_blocked = False
   if agent.canvas:
       self.add_canvas(agent.canvas)
   ```
   At this point `task.status` will be set by the pipeline to
   `RunStatus.running`.

4. **Context / policy / RAG**
   * Safety engine reads `policy_apply_to_*` and may rewrite `description`
     and `context`, saving originals to `_policy_originals`.
   * `additional_description(client)` calls `my_kb.setup_async()` and
     `my_kb.query_async(self.description, task=self)`, formatting the hits
     into an `<rag>...</rag>` block.
   * Attachments are encoded via `attachments_base64` and forwarded to the
     LLM where supported.

5. **Cache lookup**
   ```python
   cached = await task.get_cached_response(task._original_input, llm_provider)
   if cached is not None:
       task._cached_result = True
       task.task_response(...)  # _response = cached
       task.task_end()
       task.status = RunStatus.completed
       return cached
   ```

6. **Tool registration**
   * The agent's `ToolManager` (or `task._ensure_tool_manager()`) processes
     `tools`, `task_builtin_tools`, and `skills` into ToolDefinitions. These
     can be inspected with `task.get_tool_defs()`.

7. **Model call**
   * The LLM is invoked. Each tool call is appended via `add_tool_call`.

8. **Guardrail loop**
   * If `task.guardrail` is set, the agent invokes it. On failure the agent
     re-prompts up to `guardrail_retries` times.

9. **Reliability layer** (optional)
   * Verifier / editor sub-tasks are created as `Task(not_main_task=True,
     price_id_=task.price_id_, ...)`. Their usage rolls up to
     `task._reliability_sub_agent_usage`.

10. **Cache store**
    ```python
    await task.store_cache_entry(task._original_input, task.response)
    ```

11. **Policy on outputs**
    * If `policy_apply_to_tool_outputs` was on, tool outputs are scanned;
      if anonymization was used, `_anonymization_map` is reversed on the
      final response.

12. **Finalisation**
    ```python
    task.task_response(model_response)   # _response = model_response.output
    task.task_end()                      # end_time, _usage.stop_timer()
    task.status = RunStatus.completed
    ```

13. **User access**
    * `task.response` → final output (str or Pydantic instance).
    * `task.duration`, `task.total_cost`, `task.total_input_token`,
      `task.total_output_token` → accounting.
    * `task.tool_calls` → ordered tool history.
    * `task.cache_hit` → whether step 5 served the response.

### 8.3 Problematic — paused (HITL)

If the pipeline reaches a tool that needs human approval, it pauses:

```python
task.is_paused = True
task.status    = RunStatus.paused
# pipeline persists the run-state keyed on task._run_id
```

`Agent.do_async(task)` returns control to the caller. `task.is_problematic`
is now `True`.

To resume:

```python
result = await agent.continue_run_async(task)
```

The continuation path:

* Skips `InitStep`, so `task_start` is **not** re-invoked. The existing
  `_usage` accumulates more rather than being replaced. (The docstring on
  `task_start` explicitly notes: *"Must NOT be called during HITL resume — the
  pipeline skips InitStep when resuming from a later step."*)
* Loads pipeline state by `task._run_id`.
* Continues from the step that paused.
* On success: `task.status = RunStatus.completed`, `task.is_paused = False`,
  `task_end()` runs.

### 8.4 Problematic — cancelled

```python
# Some upstream signal, e.g. an event from src/upsonic/run/cancel.py
task.status = RunStatus.cancelled
```

`is_problematic` is `True`. Calling `continue_run_async(task)` either resumes
(if the run-state still exists and the pipeline allows it) or surfaces an
error. Calling `do_async(task)` would treat it as a fresh run and discard the
prior pipeline state.

### 8.5 Problematic — error

```python
task.status = RunStatus.error
```

The pipeline has captured an exception. The user can inspect `task._response`
(may be partial), `task.tool_calls`, and `task._usage`. A retry typically
calls `continue_run_async(task)` after fixing whatever caused the error
(missing API key, rate limit, etc.).

### 8.6 Completed

```python
task.status == RunStatus.completed
task.is_completed == True
task.end_time != None
task._usage is finalised
```

Calling `do_async(task)` or `continue_run_async(task)` again on a completed
task is a misuse — the agent is expected to refuse or to require an explicit
fresh `Task` instance.

### 8.7 Cache-hit short-circuit

The fastest happy path:

```text
__init__ → InitStep/task_start → policy_check → cache_get HIT
        → task_response (cached) → task_end → status=completed
```

Indicators afterwards:

* `task.cache_hit is True`
* `task._cached_result is True`
* `task.tool_calls == []` (no tools fired)
* `task._usage.duration` is dominated by `upsonic_execution_time`

### 8.8 Visualised state machine

```
                    ┌────────────────────────────────────────┐
                    │                                        │
   __init__ ──▶ status=None  ──▶  do_async()  ──▶  RUNNING   │
                                                    │       │
                                                    │       │
                                       ┌────────────┼───────┼──────────┐
                                       │            │       │          │
                                       ▼            ▼       ▼          ▼
                                   COMPLETED    PAUSED   CANCELLED   ERROR
                                       │            │       │          │
                                       │            │       │          │
                                       └─ done      └─ continue_run_async() ─┐
                                                                             │
                                                                             ▼
                                                                          RUNNING
```

`is_completed`  → `COMPLETED`.
`is_problematic` → `{PAUSED, CANCELLED, ERROR}`.

---

## 9. Worked example — the full surface

```python
from pydantic import BaseModel
from upsonic import Task, Agent
from upsonic.knowledge_base.knowledge_base import KnowledgeBase

class BlogPost(BaseModel):
    title: str
    body: str
    tags: list[str]

def search_web(query: str) -> str:
    """Pretend search."""
    return f"<results for {query}>"

def style_guard(task: Task) -> bool:
    # Reject empty / too-short outputs
    return bool(task.response) and len(str(task.response)) > 200

kb = KnowledgeBase(sources=["docs/handbook.pdf"])

task = Task(
    description="Draft a 600-word post about reliability in AI agents.",
    attachments=["assets/cover.png"],
    tools=[search_web],
    response_format=BlogPost,
    response_lang="en",
    context=[kb, "docs/style_guide.pdf"],         # PDF auto-attached
    enable_thinking_tool=True,
    guardrail=style_guard,
    guardrail_retries=2,
    enable_cache=True,
    cache_method="vector_search",
    cache_threshold=0.85,
    cache_duration_minutes=15,
    query_knowledge_base=True,
    vector_search_top_k=5,
    vector_search_fusion_method="rrf",
    policy_apply_to_description=True,
    policy_apply_to_tool_outputs=True,
)

agent = Agent("openai/gpt-4o")
result: BlogPost = await agent.do_async(task)

# Lifecycle introspection
assert task.is_completed
print(task.duration, task.total_cost)
print(task.tool_calls)            # every tool call recorded
print(task.cache_hit)             # False on cold runs, True on warm
```

If `style_guard` rejects the model's first attempt:

1. The agent re-prompts (up to `guardrail_retries=2`).
2. Each retry adds to `task._usage`.
3. If still failing, the agent surfaces the last attempt and
   `task.status` may end at `RunStatus.error` depending on agent config.

If a tool requires approval:

1. `task.is_paused = True`, `task.status = RunStatus.paused`.
2. `await agent.continue_run_async(task)` resumes from the same step,
   without re-running `task_start`.

If the same description is asked again within 15 minutes and the embedding
similarity exceeds 0.85:

1. `get_cached_response(_original_input, ...)` returns the prior output.
2. `_cache_hit = True`, `_cached_result = True`.
3. The pipeline short-circuits and finalises immediately.

---

## 10. Summary

* The `tasks` package is a single file (`tasks.py`, 1418 lines) that defines
  the `Task` Pydantic model.
* `Task` is the **shared run-state** for an Upsonic execution. Every
  `Agent.do(...)` consumes one and writes back into it.
* The constructor enforces guardrail/cache invariants, expands file/folder
  paths from `context` into `attachments`, and assigns UUIDs.
* Cache (vector / LLM judge), guardrails, policy/PII, RAG, planning, tool
  manager, and reliability all attach their own slots to the task — but the
  `Task` itself is dumb data; its only "logic" is in `task_start`,
  `task_end`, `task_response`, `add_canvas`, `additional_description`, the
  cache delegation methods, and the `to_dict`/`from_dict` serialization
  bridge.
* Lifecycle is tracked by `status` (a `RunStatus` enum):
  `running → completed`, with branch-offs to `paused`, `cancelled`, `error`,
  recoverable via `continue_run_async(task)` while `is_problematic` is
  `True`.
