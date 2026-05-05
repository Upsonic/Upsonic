---
name: graphv2-state-graph-engine
description: Use when working with Upsonic's second-generation stateful graph execution engine in `src/upsonic/graphv2/`, building agentic state machines with typed TypedDict state, reducers, checkpointing, human-in-the-loop interrupts, dynamic fan-out, TTL caching, and retry policies. Use when a user asks to build a StateGraph, compile a graph, add nodes/edges/conditional edges, persist state across threads, resume from interrupts, parallel-fan-out via Send, or compare graphv2 vs the legacy `/graph` module. Trigger when the user mentions StateGraph, CompiledStateGraph, graphv2, START, END, MemorySaver, SqliteCheckpointer, BaseCheckpointer, Checkpoint, StateSnapshot, Command, Send, interrupt, InterruptException, BaseStore, InMemoryStore, BaseCache, InMemoryCache, SqliteCache, CachePolicy, RetryPolicy, TaskFunction, @task decorator, GraphRecursionError, GraphValidationError, thread_id, checkpoint_id, recursion_limit, superstep, durability, reducers, Annotated state, conditional_edges, add_conditional_edges, fan-out, time-travel, LangGraph-style execution.
---

# `upsonic.graphv2` — Stateful Graph Execution Engine (LangGraph-style)

This document is a deep technical reference for `src/upsonic/graphv2/`. It covers every Python module in the folder, explains how graphv2 differs from the older `src/upsonic/graph/` module, and shows how the pieces wire together at runtime. The implementation is a self-contained re-imagining of the LangGraph-style execution model on top of Upsonic's UEL `Runnable` base class.

---

## 1. What this folder is + how it differs from `/graph`

`graphv2` is Upsonic's second-generation orchestration layer. Where the original `upsonic.graph` module is a *Pydantic-task DAG* in which each node wraps an Upsonic `Task` and is executed by an `Agent`/`Direct`, `graphv2` is a **state-machine graph runtime**: nodes are arbitrary functions over a typed `state` dict, edges describe deterministic or conditional transitions, and the engine itself owns durability (checkpointing), human-in-the-loop pauses (interrupts), dynamic fan-out (`Send`), TTL caching, retry policies, and per-thread vs cross-thread memory.

| Concern | `upsonic.graph` (legacy) | `upsonic.graphv2` (new) |
| --- | --- | --- |
| Core type | `Graph(BaseModel)` containing `TaskNode` / `DecisionFunc` / `DecisionLLM` / `TaskChain` (Pydantic models) | `StateGraph` builder + `CompiledStateGraph` runtime (plain dataclasses) |
| Node payload | An Upsonic `Task` (description + agent + tools) | An arbitrary `Callable[[state, config?], dict | Command | Send | List[Send]]` |
| State | `State(BaseModel)` with `data: dict` and `task_outputs: dict[node_id, Any]` | A user-defined `TypedDict` whose `Annotated[T, reducer]` fields drive merge semantics |
| Wiring | Operator `>>` building a `TaskChain` plus `if_true` / `if_false` branches on `DecisionFunc` / `DecisionLLM` | Explicit `add_node`, `add_edge`, `add_conditional_edges` against `START` / `END` constants |
| Routing | Static topology computed from edges; LLM/function decisions prune branches | Static edges + conditional functions returning either node names or `Send` objects; node bodies can return `Command(update=…, goto=…)` |
| Execution | `await graph.run_async(verbose, show_progress)` — DAG topological scheduler, optional thread-pool parallel layer | `compiled.ainvoke(input, config={"configurable": {"thread_id": ...}})` — superstep loop with `asyncio.gather` parallelism |
| Persistence | None first-class (relies on `Storage` for tasks) | `BaseCheckpointer` (`MemorySaver`, `SqliteCheckpointer`) snapshots state per `thread_id` after each superstep |
| Cross-thread memory | n/a | `BaseStore` (`InMemoryStore`) keyed by `(namespace tuple, key)` |
| Caching | n/a | `BaseCache` (`InMemoryCache`, `SqliteCache`) + per-node `CachePolicy` and `@task(cache_policy=…)` |
| Retry | n/a (handled by Agent layer) | Per-node `RetryPolicy` with exponential backoff, jitter, configurable `retry_on` |
| Human-in-the-loop | n/a | `interrupt(value)` raises `InterruptException`, run resumes via `Command(resume=…)` |
| Dynamic fan-out | `parallel_execution=True` runs ready DAG nodes via `ThreadPoolExecutor` | `Send(node, state)` from a conditional edge spawns N parallel workers with per-task substate |
| Visualisation | `display_graph_tree` Rich rendering of the DAG | Implements `Runnable.get_graph()` returning a `RunnableGraph` (ASCII + Mermaid) |
| Composability | Self-contained executor | Inherits `Runnable[Dict, Dict]`, can be piped (`|`) into UEL chains |
| Recursion guard | None | `recursion_limit` (default 100) raises `GraphRecursionError` |

Both modules coexist; nothing in graphv2 imports from `/graph`. The legacy module remains the path for "task graphs of LLM tasks", graphv2 is the path for "agentic state machines".

---

## 2. Folder layout

```
src/upsonic/graphv2/
├── __init__.py          # Lazy-loading public re-exports (no eager imports)
├── state_graph.py       # StateGraph builder + CompiledStateGraph runtime (1586 lines)
├── checkpoint.py        # Checkpoint/StateSnapshot dataclasses + BaseCheckpointer / MemorySaver / SqliteCheckpointer
├── primitives.py        # Command, Send, interrupt(), END marker, InterruptException, ContextVars
├── store.py             # BaseStore + InMemoryStore (cross-thread memory)
├── cache.py             # BaseCache + InMemoryCache / SqliteCache + CachePolicy + default_cache_key
├── task.py              # @task decorator, RetryPolicy, TaskFunction, TaskResult, retry helpers
└── errors.py            # GraphRecursionError, GraphValidationError, GraphInterruptError
```

There are no subfolders — graphv2 is intentionally flat. Total: 8 modules, ~3331 source lines (excluding the legacy `/graph` for comparison).

---

## 3. Top-level files file-by-file

### 3.1 `__init__.py` (177 lines)

Exposes the entire public surface but defers every import via a `__getattr__` hook. Heavy modules (`state_graph`, `checkpoint`, `cache`, etc.) are loaded only when the corresponding name is first accessed.

| Lazy loader | Yields |
| --- | --- |
| `_get_graphv2_core_classes()` | `StateGraph`, `START`, `END` |
| `_get_graphv2_checkpoint_classes()` | `BaseCheckpointer`, `MemorySaver`, `SqliteCheckpointer`, `StateSnapshot`, `Checkpoint` |
| `_get_graphv2_primitives()` | `Command`, `interrupt`, `Send` |
| `_get_graphv2_store_classes()` | `BaseStore`, `InMemoryStore` |
| `_get_graphv2_cache_classes()` | `BaseCache`, `InMemoryCache`, `SqliteCache`, `CachePolicy` |
| `_get_graphv2_task_classes()` | `task`, `RetryPolicy`, `TaskFunction` |
| `_get_graphv2_error_classes()` | `GraphRecursionError`, `GraphValidationError`, `GraphInterruptError` |

`__all__` simply enumerates these names. The `if TYPE_CHECKING:` block re-imports them so that IDEs / type checkers see static types.

```python
from upsonic.graphv2 import StateGraph, START, END, MemorySaver, Command, Send, interrupt
```

### 3.2 `state_graph.py` (1586 lines)

This is the core engine. It contains four key dataclasses/classes:

| Symbol | Role |
| --- | --- |
| `START = "__start__"` | Module-level sentinel for the entry edge. (Note: `END` lives in `primitives.py` and is re-imported here.) |
| `NodeConfig` | Dataclass: `func`, `retry_policy: RetryPolicy?`, `cache_policy: CachePolicy?`. |
| `Edge` | Dataclass: `from_node: str`, `to_node: str`. |
| `ConditionalEdge` | Dataclass: `from_node`, `condition: Callable[[state], str]`, `targets: List[str]`. |
| `StateGraph` | Builder. Owns `nodes: Dict[str, NodeConfig]`, `edges`, `conditional_edges`, `_reducers`. Inherits `Runnable[Dict, Dict]` so it can be piped. Calling `.invoke()` on the builder is forbidden — it raises `RuntimeError` directing users to `.compile()`. |
| `CompiledStateGraph` | Runtime. Same fields plus checkpointer/store/cache/interrupt configuration. Holds `normal_routes: Dict[str, List[str]]` and `conditional_routes: Dict[str, ConditionalEdge]`. |

Builder API:

| Method | Behaviour |
| --- | --- |
| `__init__(state_schema, *, input_schema=None, output_schema=None, context_schema=None)` | Captures schemas, eagerly extracts reducers via `_extract_reducers` (looks at `Annotated[T, reducer]` metadata in `state_schema.__annotations__`). |
| `add_node(name, func, *, retry_policy=None, cache_policy=None)` | Rejects duplicate names and the reserved `START` / `END`. |
| `add_edge(from_node, to_node)` | Appends an `Edge`. Multiple edges from the same source are allowed; the engine treats them as fan-out. |
| `add_conditional_edges(from_node, condition, targets)` | Stores a `ConditionalEdge`. `targets` is used during validation only. |
| `compile(*, checkpointer, store, cache, interrupt_before, interrupt_after, durability)` | Validates the graph (`_validate_graph`) and returns a `CompiledStateGraph`. `durability` ∈ `{"exit", "async", "sync"}`. |
| `get_graph()` | Returns a `RunnableGraph` (UEL visualiser); see §5. |
| `invoke` / `ainvoke` | Raise `RuntimeError`. |

Runtime API on `CompiledStateGraph`:

| Method | Purpose |
| --- | --- |
| `invoke(input, config, context)` | Sync wrapper: `asyncio.run(self.ainvoke(...))`. |
| `ainvoke(input, config, context)` | Main entry point. Resolves `thread_id`, `checkpoint_id`, `recursion_limit`. Dispatches to `_resume_from_interrupt` if `input` is a `Command`, or to `_execute_graph` for fresh / resume-from-checkpoint runs. Catches `InterruptException` and returns `{**state, "__interrupt__": [{"value": e.value}]}`. |
| `_validate_input(input)` | If `input_schema` is set, requires all annotated keys present, else raises `GraphValidationError`. |
| `_filter_output(state)` | Reduces the final state to the keys declared on `output_schema`. |
| `_execute_graph(state, next_nodes, …, recursion_limit)` | The superstep loop (see §8 for full flow). |
| `_execute_nodes_parallel(node_names, state, …)` | Wraps `_execute_node_with_policies` in `asyncio.gather`. Disallows `interrupt()` inside parallel nodes (would raise `RuntimeError`). |
| `_execute_node(func, state, config)` | Inspects function arity to decide whether to pass `(state)` or `(state, config)`; awaits if coroutine. Sets the `_current_cache` ContextVar so `@task` calls inside the node can read the graph's cache. |
| `_execute_node_with_policies(node_name, node_config, state, config)` | Handles `cache_policy` (read-through cache keyed by `(node_name,)`/`cache_policy.key_func(state)`) and `retry_policy` (delegating to `_execute_with_retry`). |
| `_execute_with_retry(func, state, retry_policy, config)` | Loops up to `max_attempts`; on each `Exception` calls `should_retry` + `calculate_retry_delay` (see §3.7). |
| `_execute_send_objects(send_objects, state, config)` | Spawns N `_execute_node_with_policies` tasks via `asyncio.gather`. Each Send merges `state | send_obj.state` before invoking the target. Raises `RuntimeError` on failure. |
| `_resolve_goto(goto)` | Normalises `Command.goto` to a `List[str | Send]`. |
| `_save_checkpoint_with_durability(...)` | Writes per `durability`: `sync` blocks via `run_in_executor`, `async` schedules in background, `exit` queues into `self._pending_checkpoints`. |
| `_flush_exit_checkpoints()` | Drains the deferred queue when the graph exits (success, interrupt, or error). |
| `_merge_state(current, updates)` | For each updated key, applies the registered reducer if any (e.g. `operator.add` for lists), else assignment. |
| `_get_next_nodes(current_node, state)` | If a conditional edge exists for `current_node`, evaluates it; the result may be a string, a `Send`, a list of strings, or a list of `Send`. Otherwise returns `normal_routes[current_node]` or `[END]`. |
| `_save_checkpoint(state, next_nodes, …, metadata)` | Convenience deepcopy + `checkpointer.put`. |
| `_resume_from_interrupt(command, thread_id, checkpoint_id, config, recursion_limit)` | Loads checkpoint; for `interrupted_before` / `interrupted_after` simply restarts execution with the stored `next_nodes`. For "interrupted from inside the node body", wraps execution in two ContextVars (`_interrupt_resume_values`, `_interrupt_counter`) so the next call(s) to `interrupt()` return the resume value(s) instead of raising. |
| `get_state(config)` / `get_state_history(config, limit)` | Project checkpoints into `StateSnapshot` objects. |
| `update_state(config, values, as_node)` | Manual time-travel: writes a new checkpoint whose state is `_merge_state(current, values)` and whose parent is the loaded checkpoint. |
| `get_graph()` | Returns a `RunnableGraph` for visualisation (also defined on the builder). |

Notable invariants:

* Nodes cannot return raw `Send` instances. The only valid producers of `Send` are conditional edge functions and the engine raises `ValueError` ("Send objects can only be returned from conditional edge functions") if a node body emits one.
* The recursion guard increments per *superstep* (one iteration of the while loop), regardless of how many nodes execute in parallel that step.
* When `input` is provided alongside an existing checkpoint that ended in `END`, the engine merges `input` into the saved state and restarts from `START`.

### 3.3 `checkpoint.py` (366 lines)

| Symbol | Role |
| --- | --- |
| `Checkpoint` (dataclass) | Fields: `checkpoint_id` (uuid), `thread_id`, `state`, `next_nodes`, `parent_checkpoint_id`, `timestamp` (default `now_utc()`), `metadata`. Provides `to_dict` / `from_dict` for JSON-friendly export. |
| `StateSnapshot` (dataclass) | Public projection used by `get_state` / `get_state_history`: `values`, `next`, `config`, `metadata`, `parent_config`. |
| `BaseCheckpointer` (ABC) | Three abstract methods (`put`, `get`, `list`) plus a concrete `get_history(thread_id, limit)` built on `list`. |
| `MemorySaver` | In-process dict `_storage: Dict[thread_id, List[Checkpoint]]`. New checkpoints are inserted at index 0 so the latest is always `[0]`. |
| `SqliteCheckpointer(connection)` | Persists into `checkpoints(checkpoint_id PK, thread_id, parent_checkpoint_id, state BLOB, next_nodes TEXT, timestamp TEXT, metadata TEXT)` with a `(thread_id, timestamp DESC)` index. State is `pickle`-serialised to support arbitrary Python objects. |
| `generate_checkpoint_id()` | `str(uuid.uuid4())`. |

The SQLite implementation accepts an arbitrary `sqlite3.Connection` rather than a path, which keeps pooling/threading concerns at the call site.

### 3.4 `primitives.py` (207 lines)

Defines the control-flow vocabulary:

| Symbol | Definition |
| --- | --- |
| `END = "__end__"` | Termination sentinel (referenced from `state_graph.py` and elsewhere). |
| `InterruptException(Exception)` | Engine-only signal carrying `value`. Engine catches it; user code must not. |
| `_interrupt_resume_values: ContextVar[Optional[List[Any]]]` | Per-execution list of resume values (set by `_resume_from_interrupt`). |
| `_interrupt_counter: ContextVar[int]` | Tracks which `interrupt()` call within a single resumed node we're at. |
| `interrupt(value)` | If `_interrupt_resume_values` is not set, raises `InterruptException(value)`. Otherwise pops the next resume value off the list (indexed by counter), increments the counter, and returns it. Raises `RuntimeError` if more interrupts fire than resume values were supplied (catches conditional/inconsistent interrupt patterns). |
| `Command[GotoT]` (dataclass) | Carries `update: dict?`, `goto: str | Send | List[Send] | None`, `resume: Any`, `graph: str?`. Class constant `PARENT = "__parent__"`. Returned by node bodies to merge state and route in one step. |
| `Send` (dataclass) | `node: str`, `state: dict`. Returned by conditional edges to dynamically fan out to N parallel workers. |

The use of `ContextVar` rather than thread-local state makes `interrupt()` safe under `asyncio` (each task has its own context).

### 3.5 `store.py` (187 lines)

Cross-thread, cross-session memory. Unlike checkpointed state (scoped to `thread_id`), the store is global and addressed by `(namespace: Tuple[str, ...], key: str)`.

| Symbol | Role |
| --- | --- |
| `MemoryItem` | Dataclass: `namespace`, `key`, `value`, `timestamp`. |
| `BaseStore` (ABC) | `put(namespace, key, value)`, `get(namespace, key) -> MemoryItem?`, `search(namespace, query=None, limit=10) -> List[MemoryItem]`, `delete(namespace, key) -> bool`. The docstring notes that `query` may eventually be semantic via embeddings; today it is metadata-only. |
| `InMemoryStore` | Backed by `Dict[Tuple[str, ...], Dict[str, MemoryItem]]`. `search` simply returns up to `limit` items sorted newest-first; `query` is currently ignored. |

### 3.6 `cache.py` (296 lines)

Per-node and per-task caching with optional TTL.

| Symbol | Role |
| --- | --- |
| `CachePolicy` | `key_func: Callable[[Any], str]` (defaults to `default_cache_key`) and `ttl: Optional[int]`. |
| `default_cache_key(state)` | `sha256(pickle.dumps(state))`; falls back to `sha256(str(state).encode())` if pickling fails. |
| `CacheEntry` | `value`, `timestamp` (epoch float), `ttl` + `is_expired()`. |
| `BaseCache` (ABC) | `get`, `put`, `clear`. |
| `InMemoryCache` | Two-level dict, expires-on-read. |
| `SqliteCache(connection)` | Table `cache_entries(namespace, key, value BLOB, timestamp REAL, ttl INTEGER, PK(namespace, key))`. Namespaces are flattened to `"|"`-joined strings. Expired entries are deleted on read. |

`CompiledStateGraph._execute_node_with_policies` calls `self.cache.get((node_name,), key)` and `self.cache.put((node_name,), key, result, ttl=cache_policy.ttl)`. `@task`-level caching (see §3.7) uses namespace `(f"task:{name}",)`.

### 3.7 `task.py` (458 lines)

Decorator-based "durable task" abstraction independent of the graph engine, but reusable inside graph nodes through the `_current_cache` ContextVar wired up by `_execute_node`.

| Symbol | Role |
| --- | --- |
| `_current_cache: ContextVar[Optional[BaseCache]]` | Set by `CompiledStateGraph._execute_node` for the duration of a node call. |
| `RetryPolicy` (dataclass) | `initial_interval=0.5`, `backoff_factor=2.0`, `max_interval=128.0`, `max_attempts=3`, `jitter=True`, `retry_on: Type[Exception] | Sequence[Type[Exception]] | Callable[[Exception], bool] = Exception`. |
| `default_retry_on(exception)` | Excludes `ValueError`, `TypeError`, `ArithmeticError`, `ImportError`, `LookupError`, `NameError`, `SyntaxError`, `RuntimeError`, `ReferenceError`, `StopIteration`, `StopAsyncIteration`, `OSError`. For `requests`/`httpx`-style errors with a `.response.status_code`, refuses to retry on 4xx. |
| `should_retry(exception, retry_policy)` | Dispatches based on whether `retry_on` is a tuple/list, a class, or a callable. |
| `calculate_retry_delay(attempt, retry_policy)` | `delay = min(initial * factor^attempt, max_interval)`; if `jitter`, multiplies by `0.5 + random()`. |
| `TaskResult[T]` | Future-like wrapper used when `@task`-decorated functions are called directly. `.result()` blocks if needed; `.aresult()` is async. |
| `TaskFunction[T]` | Callable wrapper holding `func`, `name`, `retry_policy`, `cache_policy`. Supports both sync and async underlying functions. Cache key combines `args` and `kwargs` into a dict that is fed to `cache_policy.key_func`. The cache namespace is `(f"task:{self.name}",)`. |
| `task(func=None, *, name=None, retry_policy=None, cache_policy=None)` | Decorator returning a `TaskFunction`. Supports both `@task` and `@task(...)` forms. |
| `set_cache_context(cache)` / `get_cache_context()` | Public helpers for setting/reading the `_current_cache` ContextVar from non-graph code. |

Crucially, the engine threads its own cache into `@task`-decorated calls: `CompiledStateGraph._execute_node` sets `_current_cache.set(self.cache)` before invoking the node and resets it on exit, so `@task(cache_policy=…)` inside a node uses the same cache backend the graph was compiled with.

### 3.8 `errors.py` (55 lines)

Three custom exceptions:

| Class | Base | When raised |
| --- | --- | --- |
| `GraphRecursionError` | `RecursionError` | `_execute_graph` exceeds `recursion_limit` (default 100) supersteps. |
| `GraphInterruptError` | `Exception` | Reserved for surface-level interrupt failures (distinct from the internal `InterruptException` in `primitives.py`). |
| `GraphValidationError` | `ValueError` | `_validate_input` (and the docstring for `_validate_graph`) when schema or topology is invalid. |

---

## 4. Subfolders file-by-file

There are **no subfolders** in `src/upsonic/graphv2/`. All modules live at the top level. The legacy `src/upsonic/graph/` similarly contains only `graph.py` and `__init__.py`.

---

## 5. Cross-file relationships

```
__init__.py
    └── lazy re-exports: state_graph.{StateGraph, START},
                         primitives.{Command, Send, interrupt, END},
                         checkpoint.{BaseCheckpointer, MemorySaver, SqliteCheckpointer, StateSnapshot, Checkpoint},
                         store.{BaseStore, InMemoryStore},
                         cache.{BaseCache, InMemoryCache, SqliteCache, CachePolicy},
                         task.{task, RetryPolicy, TaskFunction},
                         errors.{GraphRecursionError, GraphValidationError, GraphInterruptError}

state_graph.py
    ├── imports primitives.{Command, END, InterruptException, Send}
    ├── imports primitives.{_interrupt_resume_values, _interrupt_counter}  # inside _resume_from_interrupt
    ├── imports task.{RetryPolicy, _current_cache, should_retry, calculate_retry_delay}
    ├── imports cache.{BaseCache, CachePolicy}
    ├── imports checkpoint.{BaseCheckpointer, Checkpoint, StateSnapshot, generate_checkpoint_id}
    ├── imports store.{BaseStore}
    ├── imports errors.{GraphRecursionError, GraphValidationError}
    ├── inherits upsonic.uel.runnable.Runnable[Dict, Dict]
    └── lazy-imports upsonic.uel.graph.RunnableGraph in get_graph()

task.py
    └── imports cache.{BaseCache, CachePolicy}; sets _current_cache ContextVar

cache.py, store.py, checkpoint.py, primitives.py, errors.py
    └── leaf modules with no graphv2-internal dependencies (only stdlib + upsonic._utils.now_utc)
```

External (non-graphv2) dependencies:

| File | External imports |
| --- | --- |
| `state_graph.py` | `upsonic.uel.runnable.Runnable`, `upsonic.uel.graph.RunnableGraph` (lazy, in `get_graph()`) |
| `checkpoint.py`, `store.py` | `upsonic._utils.now_utc` |
| `task.py` | stdlib only (`asyncio`, `functools`, `inspect`, `time`, `random`, `contextvars`) |
| `cache.py`, `primitives.py`, `errors.py` | stdlib only |

The reverse-dependency from `upsonic/uel/graph.py` is what enables `compiled_graph.get_graph().to_mermaid()`. `RunnableGraph._is_state_graph` does an `isinstance(runnable, (StateGraph, CompiledStateGraph))` check; `_build_state_graph` then walks `graph.nodes`, `graph.edges`, `graph.conditional_edges` and renders normal edges as solid arrows (`==>`) and conditional edges as dashed (`-.->|label|`).

---

## 6. Public API

The names exposed by `from upsonic.graphv2 import …`:

| Name | Module | Kind | Notes |
| --- | --- | --- | --- |
| `StateGraph` | `state_graph` | Class | Builder; subclass of `Runnable[Dict, Dict]`. |
| `START` | `state_graph` | str sentinel `"__start__"` | Used in `add_edge(START, "first_node")`. |
| `END` | `primitives` | str sentinel `"__end__"` | Used in `add_edge("last_node", END)` and `Command(goto=END)`. |
| `BaseCheckpointer` | `checkpoint` | ABC | Implement to add new persistence backends. |
| `MemorySaver` | `checkpoint` | Class | Default in-process checkpointer. |
| `SqliteCheckpointer` | `checkpoint` | Class | Takes a `sqlite3.Connection`. |
| `StateSnapshot` | `checkpoint` | Dataclass | Returned by `get_state` / `get_state_history`. |
| `Checkpoint` | `checkpoint` | Dataclass | Internal-but-useful for custom checkpointers. |
| `Command` | `primitives` | Generic dataclass | Node return type for combined update+goto. |
| `Send` | `primitives` | Dataclass | Conditional edge return type for fan-out. |
| `interrupt` | `primitives` | Function | Pause graph and yield `value` to caller. |
| `BaseStore` | `store` | ABC | |
| `InMemoryStore` | `store` | Class | |
| `BaseCache` | `cache` | ABC | |
| `InMemoryCache` | `cache` | Class | |
| `SqliteCache` | `cache` | Class | |
| `CachePolicy` | `cache` | Dataclass | Attached per node via `add_node(... , cache_policy=…)`. |
| `task` | `task` | Decorator | Durable task wrapper for use *inside* nodes. |
| `RetryPolicy` | `task` | Dataclass | Attached per node via `add_node(..., retry_policy=…)`. |
| `TaskFunction` | `task` | Class | Type of a `@task`-decorated function. |
| `GraphRecursionError` | `errors` | Exception | |
| `GraphValidationError` | `errors` | Exception | |
| `GraphInterruptError` | `errors` | Exception | |

Not re-exported (internal but importable from submodules): `CompiledStateGraph`, `NodeConfig`, `Edge`, `ConditionalEdge`, `InterruptException`, `MemoryItem`, `CacheEntry`, `TaskResult`, `default_cache_key`, `default_retry_on`, `should_retry`, `calculate_retry_delay`, `set_cache_context`, `get_cache_context`, `generate_checkpoint_id`.

Method surface on a compiled graph:

```python
class CompiledStateGraph(Runnable[Dict, Dict]):
    def invoke(self, input=None, config=None, context=None) -> Dict[str, Any]: ...
    async def ainvoke(self, input=None, config=None, context=None) -> Dict[str, Any]: ...
    def get_state(self, config) -> Optional[StateSnapshot]: ...
    def get_state_history(self, config, limit=None) -> List[StateSnapshot]: ...
    def update_state(self, config, values, as_node=None) -> None: ...
    def get_graph(self) -> "RunnableGraph": ...
```

The `config` dict follows the convention:

```python
config = {
    "configurable": {"thread_id": "...", "checkpoint_id": "..."},
    "recursion_limit": 100,
    "context": {...}  # auto-populated when a `context=` kwarg is passed
}
```

---

## 7. Integration with the rest of Upsonic

* **UEL (`upsonic.uel`)**: `StateGraph` and `CompiledStateGraph` inherit `upsonic.uel.runnable.Runnable[Dict, Dict]`. This means a compiled graph can be piped into any UEL chain (`prompt | model | compiled_graph | parser`). `upsonic/uel/graph.py` consumes graphv2 to render visualisations:
  ```python
  from upsonic.graphv2.state_graph import StateGraph, CompiledStateGraph
  return isinstance(runnable, (StateGraph, CompiledStateGraph))
  ```
* **Visualisation**: `RunnableGraph._build_state_graph` walks `graph.nodes`, `graph.edges`, `graph.conditional_edges` and emits ASCII / Mermaid:
  - normal edges → solid arrows (`==>` in Mermaid)
  - conditional edges → dashed arrows with target labels (`-.->|target|`)
  - both `StateGraph` (builder) and `CompiledStateGraph` (runtime) work, since they share the same field names.
* **Utilities**: `upsonic._utils.now_utc` is used for timestamps in `Checkpoint` and `MemoryItem` to keep timezone-aware UTC consistent across the codebase.
* **No coupling to `upsonic.tasks.tasks.Task` or `upsonic.agent.*`**: graphv2 is provider-agnostic. Nodes can call any agent (`Direct`, `Team`, `Agent`) inside their bodies, but the engine itself never reaches into Upsonic's agent runtime. This is the main behavioural divergence from `/graph`, where every node is intrinsically an Upsonic `Task`.
* **No coupling to `upsonic.storage.*`**: graphv2 ships its own `BaseStore` and `BaseCheckpointer` rather than reusing the framework's session/memory storage. Bridging is the responsibility of user code.

---

## 8. End-to-end flow

A canonical run looks like this:

```python
from typing_extensions import TypedDict, Annotated
from operator import add
from upsonic.graphv2 import (
    StateGraph, START, END,
    MemorySaver, Command, Send, interrupt,
    InMemoryCache, CachePolicy, RetryPolicy,
)

class State(TypedDict):
    messages: Annotated[list[str], add]   # reducer = list concat
    count: int

def classify(state: State) -> Command:
    intent = "urgent" if "now" in state["messages"][-1] else "normal"
    return Command(update={"count": state["count"] + 1}, goto=intent)

def urgent(state: State) -> dict:
    answer = interrupt({"prompt": "Approve urgent route?"})
    return {"messages": [f"urgent ack: {answer}"]}

def normal(state: State) -> dict:
    return {"messages": ["normal handled"]}

def fan_out(state: State):
    return [Send("worker", {"item": i}) for i in range(3)]

def worker(state: dict) -> dict:
    return {"messages": [f"worker:{state['item']}"]}

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("urgent", urgent, retry_policy=RetryPolicy(max_attempts=2))
builder.add_node("normal", normal, cache_policy=CachePolicy(ttl=60))
builder.add_node("worker", worker)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", lambda s: ["urgent", "normal"][s["count"] % 2], ["urgent", "normal"])
builder.add_conditional_edges("normal", fan_out, ["worker"])
builder.add_edge("urgent", END)
builder.add_edge("worker", END)

graph = builder.compile(checkpointer=MemorySaver(), cache=InMemoryCache(), durability="async")

cfg = {"configurable": {"thread_id": "t1"}, "recursion_limit": 50}
out = graph.invoke({"messages": ["help me now"], "count": 0}, config=cfg)

# If urgent path fired interrupt(...):
#   out == {..., "__interrupt__": [{"value": {"prompt": "Approve urgent route?"}}]}
# Resume by:
out = graph.invoke(Command(resume="yes"), config=cfg)
```

Step-by-step trace of what happens during `graph.invoke(...)`:

1. **Entry / config** — `ainvoke` resolves `thread_id="t1"`, `checkpoint_id=None`, `recursion_limit=50`. `input` is a plain dict, so `input_schema` validation runs (no-op here).
2. **State seed** — Since the checkpointer is a fresh `MemorySaver`, `existing_checkpoint` is `None`, so `state = input` and `next_nodes = [START]`.
3. **Superstep loop** — `_execute_graph` increments `superstep_count` and checks against `recursion_limit`. With `START` in `current_nodes`, it calls `_get_next_nodes(START, state)` → `["classify"]` (from the normal edge).
4. **Node execution** — `_execute_node_with_policies("classify", node_config, state, config)`:
   - No `cache_policy`, so straight to execution.
   - No `retry_policy`, so `_execute_node` runs.
   - `_execute_node` sets `_current_cache.set(self.cache)` so any `@task` calls inside `classify` would hit `InMemoryCache`.
   - `classify` returns `Command(update={"count": 1}, goto="urgent")`.
5. **Command processing** — engine extracts `goto="urgent"`, merges `update` into state via `_merge_state` (no reducer for `count`, so direct assignment), and calls `_resolve_goto` → `["urgent"]`.
6. **Checkpointing** — `durability="async"` → `_save_checkpoint_with_durability` schedules `checkpointer.put(checkpoint)` in a thread executor and returns immediately. `current_parent` is updated.
7. **Next superstep** — `current_nodes = ["urgent"]`. `urgent` is not in `interrupt_before` (none configured here), so `_execute_node_with_policies` runs.
   - Inside `urgent`, `interrupt(...)` is called. `_interrupt_resume_values.get()` is `None`, so it raises `InterruptException`.
8. **Interrupt handling** — `_execute_graph` catches `InterruptException`, saves a checkpoint with metadata `{"interrupted_node": "urgent", "interrupt_value": {...}}` and re-raises.
9. **Outer `ainvoke`** catches the same exception, flushes nothing extra (`durability != "exit"`), and returns:
   ```python
   {**state_before_urgent, "__interrupt__": [{"value": {"prompt": "Approve urgent route?"}}]}
   ```
10. **Resume call** — `graph.invoke(Command(resume="yes"), config=cfg)`:
    - `ainvoke` sees `isinstance(input, Command)` and calls `_resume_from_interrupt`.
    - It loads the latest checkpoint (the one tagged `interrupted_node`).
    - Sets `_interrupt_resume_values.set(["yes"])` and `_interrupt_counter.set(0)`.
    - Re-executes `urgent`. Inside, the new `interrupt(...)` call sees `resume_values is not None`, returns `"yes"` (counter advances), and the rest of `urgent` runs to produce `{"messages": [...]}`.
    - Updates merge with the `Annotated[list, add]` reducer → list concatenation.
    - Saves a new checkpoint with metadata `{"resumed_from": <prev id>}`.
    - Resets the ContextVars in `finally:` so subsequent nodes start with a clean interrupt slate.
    - Continues the superstep loop with the routed `next_nodes`.
11. **Termination** — When `next_nodes` becomes `[END]` (or empty), the loop exits. `_filter_output(state)` honours `output_schema` if configured. The final dict is returned.

For the parallel-fan-out branch (`normal → worker × 3`):

1. After `normal` finishes, `_get_next_nodes("normal", state)` evaluates the conditional edge `fan_out`, which returns `[Send("worker", {"item": 0}), Send("worker", {"item": 1}), Send("worker", {"item": 2})]`.
2. Engine detects `isinstance(next_result[0], Send)` and dispatches to `_execute_send_objects`, which builds 3 coroutines via `_execute_node_with_policies("worker", ...)` with substate `{**state, "item": i}` and runs them under a single `asyncio.gather`.
3. Each result is merged into state in turn (so `Annotated[list, add]` accumulates `["worker:0", "worker:1", "worker:2"]`).
4. After Send execution, `_get_next_nodes("worker", state)` resolves to `[END]` for each, deduplicated into `current_nodes = [END]`, and the loop exits on the next iteration.

Durability summary:

| Mode | Checkpoint write timing | Failure mode |
| --- | --- | --- |
| `"sync"` | Blocks the executor on each superstep until `checkpointer.put` completes. | Slowest; safest for crash recovery. |
| `"async"` (default) | Schedules `put` in a background thread; loop continues immediately. | Tiny window where the next node runs before the previous checkpoint is durable. |
| `"exit"` | Buffers checkpoints in `_pending_checkpoints`; flushes on success, `InterruptException`, or any other exception. | Best throughput, no intermediate persistence. |

Cache + retry interaction: `_execute_node_with_policies` only writes to cache *after* the function returns successfully (post-retry). Cache hits are taken from `cache.get((node_name,), key_func(state))` and bypass retry entirely.

Recursion guard: incremented per superstep iteration; exceeding `recursion_limit` raises `GraphRecursionError`. With dynamic fan-out, N `Send` workers count as one superstep, so the limit measures structural iteration rather than total node executions.
