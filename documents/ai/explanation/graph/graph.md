---
name: graph-orchestration
description: Use when working with Upsonic's first-generation DAG-based agent workflow orchestration in `src/upsonic/graph/`. Use when a user asks to chain tasks with the `>>` operator, build conditional decision branches, inject predecessor outputs into successor task context, or run topological task workflows with branch pruning. Trigger when the user mentions Graph, TaskNode, TaskChain, DecisionFunc, DecisionLLM, DecisionResponse, State, create_graph, TaskOutputSource, graph_execution_id, branch pruning, topological execution, `task_a >> task_b`, or `upsonic.graph`.
---

# `src/upsonic/graph/` — Graph-Based Agent Orchestration

This document is a deep dive into Upsonic's first-generation graph orchestration package. It covers every public class, helper function, internal method, and how the pieces compose to enable declarative DAG-style agent workflows.

> Module path: `upsonic.graph`
> Files: 2 (`__init__.py`, `graph.py`)
> Total lines (Python): ~1060

---

## 1. What this folder is

The `graph` package implements a **declarative, DAG-like orchestration layer** built on top of Upsonic's `Task`/`Agent` primitives. Instead of running a single `Task` in isolation (or chaining them imperatively in user code), `graph` lets developers describe a **directed graph of nodes** — task nodes plus conditional decision nodes — connect them with the Pythonic `>>` operator, and then execute the whole workflow with topological ordering, automatic context injection between predecessor and successor tasks, optional parallel scheduling, branch pruning for decisions, and Rich-based progress / debug visualization.

Conceptually it sits between two other Upsonic concepts:

| Layer                  | Role                                                                         |
| ---------------------- | ---------------------------------------------------------------------------- |
| `tasks.tasks.Task`     | A single unit of agent work (description, tools, response_format, context).  |
| **`graph.Graph`**      | A topology of `Task`s plus decisions, with shared `State` and edge wiring.   |
| `team.Team`            | Higher-level multi-agent collaboration (delegation, hand-offs).              |
| `graphv2`              | A newer state-machine-style graph (StateGraph, conditional edges, START/END). |

The original `graph` (this folder) emphasizes a fluent, operator-based DSL (`task_a >> decision >> task_b`) and was designed to plug directly into the `Direct`/`BaseAgent.do_async` pipeline through a shared `state=` keyword. `graphv2` generalizes this with explicit StateGraphs and is documented elsewhere.

Key capabilities provided by this package:

- A small AST of node types: `TaskNode`, `DecisionFunc`, `DecisionLLM`, plus `TaskChain`.
- Operator overloading (`>>`) on `Task`, `TaskNode`, `DecisionFunc`, `DecisionLLM`, and `TaskChain` to build DAGs without manually managing edges.
- A `Graph` class (Pydantic `BaseModel`) that owns nodes, edges, default agent, optional storage, debug flags, and a single `State`.
- Sequential, decision-aware execution with **branch pruning** and **automatic `TaskOutputSource` context injection**.
- Rich progress bars (`rich.progress.Progress`) and an optional debug tree renderer (`display_graph_tree`).
- Sync (`run`) and async (`run_async`) entry points, where `run` falls back to a `ThreadPoolExecutor` when an event loop is already running.

---

## 2. Folder layout

```
src/upsonic/graph/
├── __init__.py          # Lazy public API surface
└── graph.py             # All graph types, decisions, chain, state, runner, helpers
```

That's it — this is a deliberately small, self-contained module. Every class and function lives in `graph.py`; `__init__.py` only re-exports them.

---

## 3. Top-level files — file-by-file walkthrough

### 3.1 `__init__.py`

A tiny lazy-loading shim. It exists so that `from upsonic.graph import Graph` doesn't pay the import cost for `Direct`, `Storage`, `rich`, etc. until the user actually accesses one of the symbols.

| Symbol             | Type                | Purpose                                                                  |
| ------------------ | ------------------- | ------------------------------------------------------------------------ |
| `_get_graph_classes()` | private function | Imports `from .graph import …` and returns a `dict` of all public names. |
| `__getattr__(name)`    | module-level hook  | PEP 562 lazy attribute access; calls `_get_graph_classes()` once.        |
| `__all__`              | list[str]          | Public re-exports listed below.                                          |

`TYPE_CHECKING` is used so that static type checkers still see the real classes without runtime import cost.

Public re-exports (`__all__`):

```
Graph, State, TaskNode, TaskChain,
DecisionFunc, DecisionLLM, DecisionResponse,
task, node, create_graph
```

If you `import upsonic.graph as g` and access `g.Foo` where `Foo` is not in this list, `__getattr__` raises a clear `AttributeError`.

### 3.2 `graph.py`

This single file contains the entire framework. The structure (top → bottom) is:

1. Imports — stdlib (`asyncio`, `time`, `uuid`, `concurrent.futures`, `typing`, `copy`), Pydantic, Rich, Upsonic internals (`BaseAgent`, optional `Direct`, optional `Storage`, `Task`), printing helpers (`console`, `escape_rich_markup`, `spacing`, `display_graph_tree`), and crucially `TaskOutputSource` from `..context.sources`.
2. `DecisionResponse` — Pydantic model used as the LLM response schema.
3. `DecisionLLM` — LLM-evaluated decision node.
4. `DecisionFunc` — function-evaluated decision node.
5. `TaskNode` — Pydantic wrapper around a `Task`.
6. `TaskChain` — mutable container of nodes + edges.
7. `State` — Pydantic state shared across the graph.
8. `Graph` — the orchestrator (Pydantic, owns nodes/edges/state/storage/agent/flags).
9. Module-level helpers: `task`, `node`, `create_graph`, `_task_rshift`.
10. Monkey-patch line: `Task.__rshift__ = _task_rshift` (executed at import time).

Below is a class-by-class, method-by-method walkthrough.

---

#### 3.2.1 `class DecisionResponse(BaseModel)`

```python
class DecisionResponse(BaseModel):
    result: bool
```

Used as the `response_format` for any `Task` constructed inside `Graph._evaluate_decision` when the decision node is a `DecisionLLM`. Forces the LLM to return a structured `{"result": true/false}` object that the runner can read deterministically.

#### 3.2.2 `class DecisionLLM(BaseModel)`

LLM-driven branching node. The actual model call is **not** performed inside this class — it's delegated to `Graph._evaluate_decision` so the graph's `default_agent` (or any task's agent) can be reused.

| Field           | Type                                                            | Default                  | Notes                                       |
| --------------- | --------------------------------------------------------------- | ------------------------ | ------------------------------------------- |
| `description`   | `str`                                                           | required positional      | Question/criterion for the LLM.             |
| `true_branch`   | `Optional[Union[TaskNode, TaskChain, DecisionFunc, DecisionLLM]]` | `None`                  | Branch when LLM responds `true`.            |
| `false_branch`  | same                                                            | `None`                   | Branch when LLM responds `false`.           |
| `id`            | `str`                                                           | `uuid.uuid4()` factory   | Identity for edge maps and state lookups.   |

`model_config = ConfigDict(arbitrary_types_allowed=True)` allows `TaskChain` (a non-Pydantic class) to appear inside a Pydantic field.

Methods:

| Method                       | Behaviour                                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| `__init__(description, *, …)`| Custom init so callers can pass `description` positionally; assigns a UUID if `id` is `None`.          |
| `evaluate(data)` (async)     | Placeholder returning `True`. Real evaluation lives in `Graph._evaluate_decision`.                     |
| `_generate_prompt(data)`     | Builds the prompt: includes `description` and wraps the previous output in `<data>…</data>` tags.      |
| `if_true(branch)`            | Sets `true_branch`; coerces a bare `Task` to `TaskNode(task=branch)`. Returns `self` (chainable).      |
| `if_false(branch)`           | Same as `if_true`, but for the false branch.                                                           |
| `__rshift__(other)`          | Implements `decision >> next`; constructs a fresh `TaskChain`, adds `self`, then `other`.              |

The prompt template:

```
You are an decision node in a graph.

Decision question: {description}

Previous node output:
<data>
{data}
</data>
```

#### 3.2.3 `class DecisionFunc(BaseModel)`

Pure-Python function decision — no LLM involved. Identical shape to `DecisionLLM` but adds:

| Field  | Type        | Notes                                              |
| ------ | ----------- | -------------------------------------------------- |
| `func` | `Callable`  | A unary function `f(latest_output) -> bool`.       |

`evaluate(data)` (synchronous) just returns `self.func(data)`. The other methods (`if_true`, `if_false`, `__rshift__`) mirror `DecisionLLM`.

This is the lightweight choice when the routing logic is deterministic — e.g. `lambda out: out.score > 0.7`.

#### 3.2.4 `class TaskNode(BaseModel)`

```python
class TaskNode(BaseModel):
    task: Task
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
```

Thin wrapper around an Upsonic `Task`. Two reasons it exists:

1. To attach a stable `id` that `Graph.edges`, `State.task_outputs`, and `TaskOutputSource` keys can reference.
2. To opt the `Task` into the `>>` operator overloading without monkey-patching `Task` directly (the file does monkey-patch `Task.__rshift__` later, but for symmetry every operand becomes a `TaskNode` internally).

`__rshift__(other)` builds a new `TaskChain`, adds `self`, then `other`, returning the chain.

#### 3.2.5 `class TaskChain`

Plain Python class (NOT a Pydantic model). Acts as a builder/container with two mutable structures:

```python
self.nodes: List[Union[TaskNode, DecisionFunc, DecisionLLM]] = []
self.edges: Dict[str, List[str]] = {}   # node_id -> [next_node_id]
```

`edges` is "the single source of truth for graph topology" (per the docstring). `nodes` is just a flat registry.

Methods:

##### `_get_leaf_nodes() -> List[…]`
Returns nodes that are **not** the source of any edge — i.e. current "tails" of the chain. This is what `add` uses to know where to attach new entry points.

##### `add(node_or_chain) -> TaskChain`
The heart of chain construction. Handles three cases for the new operand:

1. **`Task`** → coerced to `TaskNode(task=…)` first.
2. **`TaskNode | DecisionFunc | DecisionLLM`** → appended to `nodes` (if absent), and registered as an entry point. For decisions, both `true_branch` and `false_branch` are recursively unrolled:
   - If a branch is a `TaskChain`, its nodes/edges are merged in, and the decision adds an edge to the branch's first node.
   - If a branch is a single node, it's appended and an edge `decision.id -> branch.id` is recorded.
3. **`TaskChain`** → all nodes/edges merged. The chain's *entry points* (nodes that are never targets of edges within the chain) become the entry points to attach to current leaves.

After collecting `entry_points`, if the host chain already had `previous_leaves`, edges are added from each previous leaf to each new entry point. This is what enables both **branching** (one leaf → many entries via decision) and **convergence** (many leaves → one entry).

##### `__rshift__(other)`
Convenience operator: `chain >> other` simply calls `self.add(other)` and returns `self`.

#### 3.2.6 `class State(BaseModel)`

The shared blackboard between task executions.

| Field         | Type             | Default                 | Notes                                                |
| ------------- | ---------------- | ----------------------- | ---------------------------------------------------- |
| `data`        | `Dict[str, Any]` | `dict()`                | Free-form bag for user data unrelated to outputs.    |
| `task_outputs`| `Dict[str, Any]` | `dict()`                | `node_id -> last output`. Insertion-ordered.         |

| Method                  | Returns | Description                                                              |
| ----------------------- | ------- | ------------------------------------------------------------------------ |
| `update(node_id, output)` | `None` | Stores `output` under `node_id`.                                         |
| `get_task_output(node_id)` | `Any` | Looks up by node ID; returns `None` if missing.                          |
| `get_latest_output()`     | `Any` | Returns the most recently written output (last value in `task_outputs`). |

`State` is intentionally a Pydantic model so it can be serialized via standard Pydantic mechanisms (e.g. `model_dump`) and passed through agents that accept Pydantic objects.

#### 3.2.7 `class Graph(BaseModel)`

The orchestrator. All execution lives here.

##### Fields

| Field               | Type                                                  | Default | Purpose                                            |
| ------------------- | ----------------------------------------------------- | ------- | -------------------------------------------------- |
| `default_agent`     | `Optional[BaseAgent | Direct]`                        | `None`  | Used when a task doesn't specify its own agent.    |
| `parallel_execution`| `bool`                                                | `False` | Reserved flag (current runner is sequential).      |
| `max_parallel_tasks`| `int`                                                 | `4`     | Reserved cap for future parallel runner.           |
| `show_progress`     | `bool`                                                | `True`  | Default for the Rich progress bar in `run`/`run_async`. |
| `debug`             | `bool`                                                | `False` | Master debug switch.                               |
| `debug_level`       | `int`                                                 | `1`     | `>=2` triggers `display_graph_tree` + verbose logs.|
| `nodes`             | `List[TaskNode | DecisionFunc | DecisionLLM]`         | `[]`    | Flat registry.                                     |
| `edges`             | `Dict[str, List[str]]`                                | `{}`    | `node_id -> [next_id]`.                            |
| `state`             | `State`                                               | `State()` | Reset at the start of `run_async`.                |
| `storage`           | `Optional[Storage]`                                   | `None`  | Pluggable persistence (validated against `Storage`). |

`model_config = ConfigDict(arbitrary_types_allowed=True)` keeps the door open for non-Pydantic agent classes.

##### `__init__(**data)` — overrides

Adds runtime type checks beyond what Pydantic does automatically:

- If `default_agent` is provided, ensures it is either a `BaseAgent` or `Direct` instance, **and** that it exposes a callable `do_async`.
- If `storage` is provided, validates `isinstance(storage, Storage)`.
- Normalizes `debug_level` based on `debug` flag.

Then defers to `super().__init__`.

##### `add(tasks_chain) -> Graph`

Public API for inserting a workflow. Accepts any of `Task | TaskNode | TaskChain | DecisionFunc | DecisionLLM`. Non-`TaskChain` inputs are wrapped in `TaskChain().add(…)` first. Then the chain's `nodes` and `edges` are merged into `self.nodes` and `self.edges` (deduplicating by id).

##### `_get_available_agent() -> Any`

Fallback resolution: if `self.default_agent` is set, return it; otherwise scan `self.nodes` for the first `TaskNode` whose `task.agent` is non-None. Returns `None` if no agent is available.

##### `async _execute_task(node, state, verbose=False, *, graph_execution_id=None)`

Runs a single `TaskNode`. Steps:

1. Pick the runner — `task.agent` → `self.default_agent` → first available agent in the graph. Raises `ValueError` if still none.
2. Optional verbose pre-execution panel via `rich.table.Table` and `rich.panel.Panel` (Task description, runner type, tool class names).
3. If `debug_level >= 2`, calls `upsonic.utils.printing.debug_log_level2` with the node id, runner info, tool names, and graph execution id.
4. Delegates to the agent: `await runner.do_async(task, state=state, graph_execution_id=graph_execution_id)`. Falls back to `runner.do(task)` (synchronous) if `do_async` is missing.
5. Optional post-execution panels and debug logs (output preview, cost, duration).
6. Re-raises with a Rich error line on failure.

Note: this method **does not** mutate `state` itself — that happens in `_run_sequential` after the `await` returns successfully (`self.state.update(node.id, output)`).

##### `async _evaluate_decision(decision_node, state, verbose=False)`

Returns the branch (`TaskNode | TaskChain | None`) to follow.

- For `DecisionFunc`: synchronously call `decision_node.evaluate(state.get_latest_output())`.
- For `DecisionLLM`:
  1. Resolve an agent (default → available).
  2. Build prompt via `decision_node._generate_prompt(latest_output)`.
  3. Construct an ephemeral `Task(prompt, response_format=DecisionResponse)`.
  4. Execute via `agent.do_async(decision_task, state=state)` (or `agent.do` fallback).
  5. Read `response.result` (defaults to `False` if attribute is missing).
- Verbose mode prints a yellow "🔀 Evaluating Decision" panel.
- Returns `decision_node.true_branch if result else decision_node.false_branch`.

##### `_format_output_for_display(output) -> str`

Cosmetic helper for verbose mode. If `output` is a Pydantic model, JSON-dumps `model_dump()`; otherwise `str()`. Truncates to 200 chars with an ellipsis. Always passes through `escape_rich_markup` so Rich tags in user output don't blow up the renderer.

##### `_get_predecessors(node) -> List[…]`

Inverts `self.edges` ad-hoc to find all nodes whose edge list contains `node.id`.

##### `_get_task_predecessors_through_decisions(decision_node, executed_node_ids) -> List[TaskNode]`

Crucial helper for context injection. When a `TaskNode` follows a chain of decisions, we want to inject the **last real task output** as context — not the decisions themselves (which produce no `state.task_outputs` entry). This BFS walks `self.edges` backwards from a decision node, skipping over other decision nodes, and collects only `TaskNode` predecessors that have actually been executed.

##### `_get_start_nodes() -> List[…]`

Nodes whose IDs do not appear as a target in any edge — i.e. the roots.

##### `_get_next_nodes(node) -> List[…]`

`self.edges.get(node.id, [])` mapped back through `self.nodes`.

##### `_get_all_branch_node_ids(branch) -> Set[str]`

Recursively collects every node ID inside a sub-branch (used for **pruning** the branch that wasn't taken). Handles three input shapes:

- `None` → empty set.
- `TaskChain` → enqueue every node, recurse on decisions inside.
- single node → recurse into its `true_branch`/`false_branch` if it's a decision.

##### `async _run_sequential(verbose, show_progress, *, graph_execution_id)`

The actual runner. Implements topological execution with decisions and pruning:

1. Initialize execution queue with `self._get_start_nodes()`. Track `queued_node_ids`, `executed_node_ids`, `pruned_node_ids`, `failed_node_ids`.
2. Optionally start a `rich.progress.Progress` with spinner + bar + elapsed time. Total = `_count_all_possible_nodes()`.
3. If `debug_level >= 2`, render the initial graph tree via `display_graph_tree`.
4. Main loop: pop the next node from the queue.
   - If `TaskNode`:
     - Reset `node.task.context = []`.
     - For each predecessor:
       - If predecessor is a decision, call `_get_task_predecessors_through_decisions` and append `TaskOutputSource(task_description_or_id=task_pred.id)` for each executed task predecessor.
       - Otherwise (predecessor is a `TaskNode`), append `TaskOutputSource` directly if it's already executed.
     - Run `await self._execute_task(...)`, update `state`, add to `executed_node_ids`. On failure, record in `failed_node_ids` and re-raise.
   - If decision node:
     - `branch_to_follow = await self._evaluate_decision(...)`.
     - Add to `executed_node_ids`.
     - Determine the **pruned** branch (`false_branch` if true won, else `true_branch`) and union `_get_all_branch_node_ids(pruned_branch)` into `pruned_node_ids`.
5. Re-render the debug tree if `debug_level >= 2`.
6. Enqueue successors only if **all** of their predecessors have either executed or been pruned (this is the "topological gate" that handles convergence).
7. Update progress bar with `len(executed) + len(pruned)`.
8. On exit: stop the progress bar in `finally`, render final tree if debug, print "Graph Execution Completed", return `self.state`.

##### `_count_all_possible_nodes() -> int`

`max(len(self.nodes), 1)` — guards against zero-division in progress totals.

##### `async run_async(verbose=True, show_progress=None) -> State`

Public async entry point. Resets `self.state = State()`, allocates a new `graph_execution_id = str(uuid.uuid4())`, prints a "Starting Graph Execution" header (when verbose), and delegates to `_run_sequential`.

##### `run(verbose=True, show_progress=None) -> State`

Public sync entry point with three execution paths to handle nested event loops gracefully:

1. `try: loop = asyncio.get_event_loop()` — if it raises `RuntimeError` (no loop in this thread), `asyncio.run(self.run_async(...))`.
2. If a loop exists and `loop.is_running()` is `True` (e.g. inside Jupyter, FastAPI, etc.), spawn a `ThreadPoolExecutor` and run `asyncio.run` in a worker thread (avoids "this event loop is already running").
3. Otherwise, `loop.run_until_complete(self.run_async(...))`.

##### `get_output() -> Any`

Sugar for `self.state.get_latest_output()`.

##### `get_task_output(description) -> Any`

Iterates `self.nodes` (in insertion order), matches `TaskNode.task.description == description`, returns the corresponding entry from `state.task_outputs`. Returns `None` if not found or not yet executed.

#### 3.2.8 Module-level helpers

| Function                                                                  | Returns       | Notes                                                                 |
| ------------------------------------------------------------------------- | ------------- | --------------------------------------------------------------------- |
| `task(description: str, **kwargs) -> Task`                                | `Task`        | Convenience constructor; defaults `agent=None` if omitted.            |
| `node(task_instance: Task) -> TaskNode`                                   | `TaskNode`    | Pure wrapping constructor.                                            |
| `create_graph(default_agent=None, parallel_execution=False, show_progress=True) -> Graph` | `Graph` | Functional alternative to `Graph(...)` constructor for symmetry with other Upsonic factory helpers. |
| `_task_rshift(self, other)`                                               | `TaskChain`   | Implements `>>` for `Task` itself; a fresh `TaskChain` with two adds. |

The very last line of `graph.py`:

```python
Task.__rshift__ = _task_rshift
```

This is a module-import-time monkey-patch that adds the `>>` operator to the canonical `Task` class so users can write `task_a >> task_b` without ever touching `TaskNode` themselves. The patch only takes effect once `upsonic.graph` (or anything that imports it) has been imported.

---

## 4. Subfolders

There are no subfolders. The `graph` package is intentionally flat — two files only.

---

## 5. Cross-file relationships

Inside this package, only `__init__.py` references `graph.py`. The interesting cross-file relationships are with the rest of Upsonic:

| External symbol                              | Where used in `graph.py`                              | Why                                                                    |
| -------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| `upsonic.agent.base.BaseAgent`               | `Graph.default_agent` typing & isinstance check       | Validates that the agent has the required `do_async` interface.        |
| `upsonic.direct.Direct` (optional import)    | Same                                                  | Allows the lighter-weight `Direct` agent class to be used as default.  |
| `upsonic.storage.base.Storage` (optional)    | `Graph.storage` typing & isinstance check             | Pluggable persistence; not used by current runner but reserved.        |
| `upsonic.tasks.tasks.Task`                   | `TaskNode.task`, `TaskChain.add`, `_evaluate_decision`| The unit of work executed per node.                                    |
| `upsonic.context.sources.TaskOutputSource`   | `_run_sequential` (auto-injection)                    | Wires upstream outputs into downstream `task.context` lists.           |
| `upsonic.utils.printing` (`console`, `escape_rich_markup`, `spacing`, `display_graph_tree`, `debug_log_level2`) | Throughout `_execute_task`, `_evaluate_decision`, `_run_sequential` | Rich-based UX. |
| `rich.progress`, `rich.panel`, `rich.table`  | Verbose & debug output                                 | Pretty rendering of execution status.                                   |

The `try/except ImportError` for both `Direct` and `Storage` keeps the graph module importable in stripped-down installations (e.g. without storage extras).

---

## 6. Public API

What you should care about when using this package as a consumer.

### 6.1 Re-exported names

```python
from upsonic.graph import (
    Graph,
    State,
    TaskNode,
    TaskChain,
    DecisionFunc,
    DecisionLLM,
    DecisionResponse,
    task,        # factory: task(description, **kwargs) -> Task
    node,        # factory: node(task_instance) -> TaskNode
    create_graph # factory: create_graph(default_agent=…, parallel_execution=…, show_progress=…) -> Graph
)
```

### 6.2 Operators

| Left operand  | Operator | Right operand                           | Result      |
| ------------- | -------- | --------------------------------------- | ----------- |
| `Task`        | `>>`     | `Task | TaskNode | DecisionFunc | DecisionLLM | TaskChain` | `TaskChain` |
| `TaskNode`    | `>>`     | same                                    | `TaskChain` |
| `TaskChain`   | `>>`     | same                                    | `TaskChain` (mutated, returned) |
| `DecisionFunc`/`DecisionLLM` | `>>` | same                          | `TaskChain` |

### 6.3 Decision builders

```python
DecisionFunc(
    description="output looks confident",
    func=lambda out: getattr(out, "score", 0) > 0.7,
).if_true(task_a).if_false(task_b)
```

```python
DecisionLLM(
    description="Does the previous output indicate the user is angry?",
).if_true(escalation_chain).if_false(continue_chain)
```

### 6.4 Graph lifecycle

```python
g = create_graph(default_agent=agent)
g.add(task_a >> decision >> task_b)
state = g.run(verbose=False)        # sync, returns final State
out  = g.get_output()                # latest output
```

---

## 7. Integration with the rest of Upsonic

### 7.1 Relationship to `agent` and `tasks`

- A `Graph` is a *coordinator* that ultimately calls `agent.do_async(task, state=…, graph_execution_id=…)`. Both `BaseAgent` (via `agent/agent.py`) and `Direct` (`direct.py`) accept the `state` keyword and the `graph_execution_id` keyword for traceability.
- Tasks executed in a graph have their `task.context` *rewritten* on every iteration of `_run_sequential`. The graph erases stale context (`node.task.context = []`) and re-builds it from `TaskOutputSource`s pointing at predecessors. This is why a single `Task` can be safely re-used inside a graph without manual context bookkeeping.
- Cost/duration metrics (`task.duration`, `task.total_cost`) are filled in by the agent pipeline and read back by the graph for verbose panels.

### 7.2 Relationship to `context.sources.TaskOutputSource`

`TaskOutputSource(task_description_or_id=...)` is the bridge. The graph injects sources keyed by **node id (UUID)**, not by task description, ensuring that two tasks with the same description don't collide. The agent pipeline reads `task.context`, sees a `TaskOutputSource`, and resolves it against either the active `State` (preferred) or a fallback registry.

### 7.3 Relationship to `graphv2`

Both `graph` and `graphv2` exist in parallel. They are **not** linked at runtime — `graphv2` reimplements graphs as an explicit `StateGraph` with `add_node`, `add_edge`, `add_conditional_edges`, `START`/`END` sentinels, and pluggable channels. `graph` (this folder) is the original DSL-style API; new features (e.g. richer parallel execution, persistence with the `Storage` contract, breakpoints) typically land in `graphv2`. Existing user code that uses `task_a >> task_b` should continue to work against `graph`.

### 7.4 Relationship to `team`, `prebuilt`, `reliability_layer`

- `Team` orchestrates *agents* dynamically (delegation), whereas `Graph` orchestrates *tasks* statically (DAG). Use a graph when the workflow shape is known up-front; use a team when an agent must decide who handles what at runtime.
- `prebuilt` agents (`upsonic/prebuilt/<agent>/`) can serve as the `default_agent` of a `Graph`; they implement `BaseAgent.do_async`, which is the only contract the graph requires.
- The reliability layer is invoked **inside** the agent's `do_async`, so a `Graph` automatically benefits from verifier/editor passes if the underlying agent is configured with them.

### 7.5 Telemetry & debug

- `graph_execution_id` is generated once per `run_async` call and threaded through every `do_async` invocation, so external telemetry (Langfuse, Logfire, OpenTelemetry sinks) can group all task spans under a single graph run.
- `debug_level >= 2` triggers `display_graph_tree(graph, executed_node_ids, pruned_node_ids, executing_node_id, failed_node_ids)`, a Rich-tree renderer in `upsonic.utils.printing` that shows each node's status with colors. This is invoked three times per node (before, after, and once at the end) plus once at startup.

---

## 8. End-to-end flow of a graph execution

Below is the complete lifecycle from user code to final `State`.

### 8.1 Construction

```python
from upsonic import Task
from upsonic.graph import (
    create_graph, DecisionFunc, DecisionLLM, TaskNode,
)

t1 = Task("Summarize this PDF", agent=summarizer_agent)
t2 = Task("Translate to Turkish", agent=translator_agent)
t3 = Task("Send to Slack",        agent=slack_agent)
t4 = Task("Save to disk",         agent=file_agent)

decision = DecisionFunc(
    description="result mentions Slack?",
    func=lambda out: "slack" in str(out).lower(),
).if_true(t3).if_false(t4)

chain = t1 >> t2 >> decision

g = create_graph(default_agent=summarizer_agent, show_progress=True)
g.add(chain)
```

What happened:

1. `Task.__rshift__` (monkey-patched at module import) wraps `t1` and `t2` into `TaskNode`s via a fresh `TaskChain`.
2. `chain >> decision` calls `TaskChain.__rshift__`, which invokes `add(decision)`. Inside `add`:
   - The decision's `true_branch` (`t3`) and `false_branch` (`t4`) are auto-coerced to `TaskNode`s (they were `Task`s) and registered.
   - Edges `decision.id -> t3.id` and `decision.id -> t4.id` are added.
   - The previous leaf (`t2`) gets an edge `t2.id -> decision.id`.
3. `g.add(chain)` merges nodes/edges into the `Graph`.

Topology after construction:

```
t1 ──► t2 ──► decision ──► t3   (true)
                       └──► t4   (false)
```

### 8.2 `g.run()`

1. `run` resolves the right event-loop strategy (new loop, ThreadPoolExecutor, or current loop) and calls `await self.run_async(...)`.
2. `run_async` resets `self.state = State()`, mints `graph_execution_id`, prints "Starting Graph Execution", and calls `_run_sequential`.

### 8.3 `_run_sequential`

| Iter | Queue (head) | Action                                                                                              | State                       |
| ---- | ------------ | --------------------------------------------------------------------------------------------------- | --------------------------- |
| 1    | `t1`         | TaskNode → no predecessors → execute via `summarizer_agent.do_async(t1, state, graph_execution_id)` | `task_outputs[t1.id] = …`   |
| 2    | `t2`         | TaskNode → predecessor is `t1`, append `TaskOutputSource(t1.id)` to `t2.task.context` → execute     | `task_outputs[t2.id] = …`   |
| 3    | `decision`   | DecisionFunc → `decision.evaluate(state.get_latest_output())` → `True`                              | `executed += {decision.id}` |
| 3a   |              | `pruned_branch = decision.false_branch` (=`t4` node) → `pruned_node_ids += {t4.id}`                 |                             |
| 4    | `t3`         | Predecessor is `decision`. `_get_task_predecessors_through_decisions(decision, executed_node_ids)` walks back to find `t2`. `TaskOutputSource(t2.id)` is appended to `t3.task.context`. Execute. | `task_outputs[t3.id] = …` |
| —    | (`t4` skipped) | Successor `t4` is never enqueued because its only predecessor (`decision`) is "executed", but its inclusion in `pruned_node_ids` blocks it from the queue gate (the gate also uses `pruned_node_ids` and we never enqueue pruned nodes). |                             |

Progress bar updates happen on every iteration; the debug tree is re-rendered if `debug_level >= 2`.

### 8.4 Post-execution

`run_async` returns `self.state`. The caller can:

- `g.get_output()` → output of `t3` (most recently written).
- `g.get_task_output("Translate to Turkish")` → output of `t2`.
- Inspect `state.task_outputs` directly.

### 8.5 Failure modes

- **No agent resolvable** → `_execute_task` raises `ValueError` with a Rich-escaped task description.
- **Task itself raises** → `_run_sequential` adds the node to `failed_node_ids`, optionally prints a red banner, and re-raises. The graph stops; `state` is left in a partial state (any successfully completed predecessors retain their outputs).
- **Decision node raises** → handled symmetrically; the failed decision goes into `failed_node_ids` and the exception propagates.
- **Nested event loop** → `run` detects via `loop.is_running()` and runs in a thread; this avoids the classic `RuntimeError: This event loop is already running` issue.

---

## Reference: quick API tables

### Classes

| Class             | Kind            | Location              | Key responsibility                                |
| ----------------- | --------------- | --------------------- | ------------------------------------------------- |
| `DecisionResponse`| `BaseModel`     | `graph.py:34`         | Structured `{result: bool}` for LLM decisions.    |
| `DecisionLLM`     | `BaseModel`     | `graph.py:39`         | LLM-evaluated routing node.                       |
| `DecisionFunc`    | `BaseModel`     | `graph.py:152`        | Pure-function routing node.                       |
| `TaskNode`        | `BaseModel`     | `graph.py:242`        | Wraps a `Task` with an id and `>>`.               |
| `TaskChain`       | plain class     | `graph.py:269`        | Builder of nodes + edges.                         |
| `State`           | `BaseModel`     | `graph.py:348`        | Shared blackboard.                                |
| `Graph`           | `BaseModel`     | `graph.py:392`        | Orchestrator + runner.                            |

### Functions

| Function           | Location          | Role                                            |
| ------------------ | ----------------- | ----------------------------------------------- |
| `task(...)`        | `graph.py:938`    | `Task` factory with `agent=None` default.       |
| `node(...)`        | `graph.py:952`    | `TaskNode` factory.                             |
| `create_graph(...)`| `graph.py:964`    | `Graph` factory.                                |
| `_task_rshift`     | `graph.py:978`    | Patched onto `Task.__rshift__`.                 |

### Graph methods (selected)

| Method                                       | Async | Public | Purpose                                                              |
| -------------------------------------------- | ----- | ------ | -------------------------------------------------------------------- |
| `add(tasks_chain)`                           | no    | yes    | Merge a chain/task into this graph.                                  |
| `run(verbose, show_progress)`                | no    | yes    | Sync entry point with event-loop juggling.                           |
| `run_async(verbose, show_progress)`          | yes   | yes    | Async entry point.                                                   |
| `get_output()`                               | no    | yes    | Latest output from `state`.                                          |
| `get_task_output(description)`               | no    | yes    | Lookup by task description.                                          |
| `_run_sequential`                            | yes   | no     | Topological executor with branch pruning.                            |
| `_execute_task`                              | yes   | no     | Single-task runner with verbose/debug.                               |
| `_evaluate_decision`                         | yes   | no     | Branch selection for `DecisionFunc`/`DecisionLLM`.                   |
| `_get_predecessors`/`_get_next_nodes`        | no    | no     | Topology queries.                                                    |
| `_get_task_predecessors_through_decisions`   | no    | no     | Skip-decision predecessor walk for context injection.                |
| `_get_all_branch_node_ids`                   | no    | no     | Collect IDs to mark as pruned.                                       |
| `_get_start_nodes`                           | no    | no     | Roots (no incoming edges).                                           |
| `_count_all_possible_nodes`                  | no    | no     | Progress total.                                                      |
| `_get_available_agent`                       | no    | no     | Agent fallback resolver.                                             |
| `_format_output_for_display`                 | no    | no     | Truncated, Rich-safe stringification.                                |

---

## Mental model in one sentence

> `upsonic.graph` is a 1000-line, two-file DSL that turns Python's `>>` operator into a topological scheduler over `Task`/`Agent` calls, where every node is either a unit of agent work (`TaskNode`) or a routing decision (`DecisionFunc`/`DecisionLLM`), and a single shared `State` plus auto-injected `TaskOutputSource`s carry data along the edges.
