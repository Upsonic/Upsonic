---
name: team-multi-agent-coordination
description: Use when working with Upsonic's multi-agent orchestration layer in src/upsonic/team/, including building teams of agents, nesting teams, or wiring leader/router agents. Use when a user asks to coordinate multiple agents, delegate tasks across agents, route a request to the right specialist, combine results from multiple tasks, propagate memory or skills across team members, stream from a team, or expose a team as an MCP server. Trigger when the user mentions Team, multi-agent, sequential mode, coordinate mode, route mode, leader agent, router agent, delegate_task, route_request_to_member, CoordinatorSetup, DelegationManager, ContextSharing, TaskAssignment, ResultCombiner, entities, ask_other_team_members, as_mcp, nested teams, or team.do/astream.
---

# `src/upsonic/team/` — Multi-Agent Team Coordination

## 1. What this folder is

The `team/` folder contains the **multi-agent orchestration layer** of Upsonic. It defines the
`Team` class and its supporting helpers. A `Team` groups one or more `Agent` instances (and/or
nested `Team` instances) and runs `Task` objects across them according to one of three operational
modes:

| Mode          | Picks who runs each task                                      | Uses a leader/router agent? |
|---------------|---------------------------------------------------------------|-----------------------------|
| `sequential`  | An ad-hoc selection LLM picks an entity per task, in order    | No (auxiliary selection LLM)|
| `coordinate`  | A persistent leader agent delegates via a `delegate_task` tool| Yes (leader)                |
| `route`       | A router agent picks one specialist for the whole mission     | Yes (router)                |

The folder is **all about coordination**: it does not own model wrappers, tool execution loops, or
storage. It composes the `Agent` class (from `upsonic.agent.agent`) and the `Task` class (from
`upsonic.tasks.tasks`) into higher-order workflows.

Key properties:

- **Heterogeneous membership**: an entity can be an `Agent` or a nested `Team`, so teams nest.
- **Three modes** with different selection strategies (`sequential` / `coordinate` / `route`).
- **Memory & skills propagation**: a `Team`-level `Memory` or `Skills` is propagated down to all
  contained agents (recursively into sub-teams).
- **Streaming**: text-only `astream`/`stream` for sequential, coordinate (leader's text), and route
  (chosen specialist's text) modes.
- **MCP exposure**: a team can be wrapped as a `FastMCP` server with `Team.as_mcp()`.

The package's public surface is small: `Team`, plus the helper classes `ContextSharing`,
`TaskAssignment`, `ResultCombiner` (re-exported by `__init__.py`).

## 2. Folder layout (tree)

```
src/upsonic/team/
├── __init__.py             # Lazy re-exports: Team, ContextSharing, TaskAssignment, ResultCombiner
├── team.py                 # Team class (entry point) — modes, streaming, MCP, propagation
├── coordinator_setup.py    # CoordinatorSetup — builds leader/router system prompts (manifest)
├── delegation_manager.py   # DelegationManager — generates delegate_task / route_request_to_member tools
├── context_sharing.py      # ContextSharing — describes entities, builds + enhances task context
├── task_assignment.py      # TaskAssignment — registry build + LLM-based entity selection
└── result_combiner.py      # ResultCombiner — combines multiple task responses into a final answer
```

There are no sub-packages. All files are at the top level of the `team/` package.

## 3. Top-level files

### 3.1 `__init__.py`

A lazy-loading module shim. It defines `__getattr__` so `from upsonic.team import Team` (and the
helpers) only imports the heavy modules on first access. Public attributes are listed in
`__all__`:

```python
__all__ = ['Team', 'ContextSharing', 'TaskAssignment', 'ResultCombiner']
```

Lazy loading keeps `import upsonic.team` cheap, since `Team` pulls in `Agent`, `Task`, FastMCP, and
the rest of the framework.

### 3.2 `team.py` — the `Team` class

This is the user-facing entry point. The file defines a single class, `Team`, plus an internal
helper to find a model.

#### Constructor signature

```python
class Team:
    def __init__(self,
                 entities: Optional[List[Union[Agent, "Team"]]] = None,
                 tasks: Optional[List[Task]] = None,
                 name: Optional[str] = None,
                 role: Optional[str] = None,
                 goal: Optional[str] = None,
                 model: Optional[Any] = None,
                 response_format: Any = str,
                 ask_other_team_members: bool = False,
                 mode: Literal["sequential", "coordinate", "route"] = "sequential",
                 leader: Optional[Agent] = None,
                 router: Optional[Agent] = None,
                 memory: Optional[Memory] = None,
                 skills: Optional[Any] = None,
                 debug: bool = False,
                 debug_level: int = 1,
                 agents: Optional[List[Union[Agent, "Team"]]] = None,
                 print: Optional[bool] = None):
```

| Parameter                | Purpose                                                                                  |
|--------------------------|------------------------------------------------------------------------------------------|
| `entities`               | Members of the team (Agents and/or nested Teams). Required (≥1).                         |
| `tasks`                  | Default tasks if `do()` is called without args. Stored as a list.                        |
| `name`                   | Display ID used in nested manifests, MCP server name, headers.                           |
| `role`                   | Free-text role description; surfaced in coordinator/router prompts when this team is nested. |
| `goal`                   | Free-text goal description; surfaced in coordinator/router prompts when this team is nested. |
| `model`                  | Fallback model used by the auto-created leader/router and by the result combiner.        |
| `response_format`        | Pydantic model or `str` for the team's final response.                                   |
| `ask_other_team_members` | If true, every Agent member is added as a tool to every initial task (see `add_tool`).   |
| `mode`                   | Operational mode (`sequential` / `coordinate` / `route`).                                |
| `leader`                 | Pre-built leader agent for `coordinate` mode (else built from `model`).                  |
| `router`                 | Pre-built router agent for `route` mode (else built from `model`).                       |
| `memory`                 | A `Memory` propagated to every Agent member (and sub-team) lacking its own.              |
| `skills`                 | A `Skills` set; copied per agent, registered via `agent.add_tools()`. Merges with existing skills. |
| `debug` / `debug_level`  | Enables `info_log` and `debug_log_level2` traces. Level 2 prints task-level details.     |
| `agents`                 | **Backwards-compatible alias** for `entities`.                                           |
| `print`                  | Tri-state (`None` / `True` / `False`) controlling output printing.                       |

The constructor:

1. Resolves `entities` from `entities` or `agents` (alias). Raises `ValueError` if empty.
2. Stores everything; `_leader` and `_router` are kept private until the relevant mode runs.
3. Resolves the `print` flag: `UPSONIC_AGENT_PRINT` env > constructor `print=` > defaults.
4. If `memory` is given, calls `_propagate_memory` to recursively assign it to agents/sub-teams
   that don't already have one.
5. If `skills` is given, calls `_propagate_skills` to recursively register the skills as tools
   (each agent gets its own copy so metrics are tracked independently).
6. If `ask_other_team_members=True`, calls `add_tool()` to expose Agent members as tools on each
   initial task.

#### Properties and aliases

| Member                         | Behavior                                                                  |
|--------------------------------|---------------------------------------------------------------------------|
| `agents` (property + setter)   | Read/write alias to `self.entities` for legacy code.                      |
| `leader_agent`                 | Set during `coordinate`/`route` execution (the active leader/router).     |
| `routed_entity` (via DM)       | The entity chosen by the router in `route` mode.                          |

#### Internal helpers

| Method                                    | Role                                                                                |
|-------------------------------------------|-------------------------------------------------------------------------------------|
| `_propagate_memory(entities, memory)`     | Recursively assigns `memory` to Agents and Teams that have `memory is None`.        |
| `_propagate_skills(entities, skills)`     | Per-agent skills copy + register; merges with existing skills (removes stale tools).|
| `get_entity_id()`                         | Returns `self.name` if set, else `f"Team_{id(self)}"`. Used for routing/manifests.  |
| `_resolve_print_flag(method_default)`     | ENV > constructor `print` > method default (do=False, print_do=True).               |
| `_find_first_model()`                     | DFS over entities to find any agent's model (used by combiner & by inner teams).    |
| `_format_stream_header(entity_id, type)`  | Builds the `--- [<Type>] <id> ---` separator used in streaming output.              |
| `_entity_astream(entity, task, debug)`    | Yields text chunks from an `Agent.astream` / `Team.astream` / `do_async` fallback.  |

#### Public execution methods

| Method                          | Sync? | What it does                                                                |
|---------------------------------|-------|-----------------------------------------------------------------------------|
| `do(tasks=None)`                | sync  | Runs `multi_agent_async` over `tasks` (or `self.tasks`). Returns final.     |
| `print_do(tasks=None)`          | sync  | Same as `do()` but resolves `print=True` (subject to env override).         |
| `complete(tasks=None)`          | sync  | Alias for `do(tasks)`.                                                      |
| `print_complete(tasks=None)`    | sync  | Alias for `print_do(tasks)`.                                                |
| `do_async(task)`                | async | Single task (str or `Task`). Wraps strings in `Task`. Sets `.response`.     |
| `print_do_async(task)`          | async | Like `do_async` but with print enabled (subject to env override).           |
| `ado(task)`                     | async | Async alias for `do_async`.                                                 |
| `multi_agent_async(...)`        | async | The dispatch core (see flow below). Public so it can be called manually.    |
| `astream(tasks=None, debug=False)` | async | Yields text chunks (no events).                                          |
| `stream(tasks=None, debug=False)`  | sync  | Threaded wrapper around `astream`.                                       |
| `as_mcp(name=None)`             | sync  | Builds a `FastMCP` server exposing this team as a `do(task: str)` tool.     |
| `add_tool()`                    | sync  | Adds Agent members as tools to each task in `self.tasks`.                   |

#### `_run_sync(coro)`

A small utility that runs a coroutine synchronously even when called from inside an existing
event loop. If `asyncio.get_running_loop()` succeeds, it spawns a `ThreadPoolExecutor` and runs
`asyncio.run(coro)` in a worker thread; otherwise it just calls `asyncio.run(coro)` directly.
This makes `do()` safe to call from inside Jupyter or another async runtime.

#### `multi_agent_async` — the dispatcher

This is the heart of the file. Behavior depends on `self.mode`:

##### Sequential (default)

1. Build `ContextSharing`, `TaskAssignment`, and a `ResultCombiner` (using `self.model` or the
   first model found in any entity).
2. Pick `last_debug` from the last `Agent` in `entities` (used as a fallback flag for the
   combiner agent).
3. `task_assignment.prepare_entities_registry(entities)` returns `(registry: dict, names: list)`.
4. Iterate over each task in order:
   - Build a **selection context** of (current task + other tasks + entity descriptions).
   - Call `task_assignment.select_entity_for_task(...)` — uses an LLM with a
     `SelectedEntity(BaseModel)` response format and up to 3 attempts (with substring fallback,
     and the first entity as last resort).
   - When `task.agent` is preset, the LLM step is skipped.
   - Call `context_sharing.enhance_task_context(...)` to merge other tasks and entity
     configs into the task's `.context`.
   - `await selected_entity.do_async(current_task, _print_method_default=...)`.
   - Append `current_task` to `all_results`.
5. If only one result, `result_combiner.get_single_result(all_results)`; else
   `result_combiner.combine_results(...)` synthesizes a final answer with a fresh `Agent`.

##### Coordinate

1. Validate that either a `leader` Agent or a `model` has been provided.
2. Build a `tool_mapping: Dict[str, Callable]` from all callable tools attached to each
   `self.tasks` entry.
3. Instantiate `CoordinatorSetup(entities, tasks, mode="coordinate")` and
   `DelegationManager(entities, tool_mapping, debug)`.
4. Use the user-supplied `leader` (and inherit team memory if leader has none) or build a fresh
   `Agent(model=self.model, memory=self.memory)`. Store it as `self.leader_agent`.
5. Set `leader.system_prompt = setup_manager.create_leader_prompt()` (a long prompt detailing
   the team roster + initial task manifest).
6. Build a master `Task`:
   - `description`: a fixed "Begin your mission..." string.
   - `attachments`: union of all task attachments.
   - `tools`: `[delegation_manager.get_delegation_tool()]` — the dynamic `delegate_task` tool.
   - `response_format`: `self.response_format`.
7. `await self.leader_agent.do_async(master_task, ...)`. The leader iteratively calls
   `delegate_task(member_id, description, tools=[names], context=..., attachments=...)`. Each
   invocation routes the sub-task through the chosen member (`Agent` or nested `Team`).
8. Return the leader's final response.

##### Route

1. Validate that either a `router` Agent or a `model` has been provided.
2. Build a `CoordinatorSetup` in `route` mode plus an empty-mapping `DelegationManager`.
3. Use `self._router` or build `Agent(model=self.model)` as the router.
4. Set the router's system prompt with the route-mode template.
5. Build a router `Task` whose only tool is `delegation_manager.get_routing_tool()`
   (the `route_request_to_member` tool). Run it asynchronously.
6. Read `delegation_manager.routed_entity`. If empty, raise `ValueError`. Otherwise build a
   `final_task` that consolidates **all** task descriptions, attachments, and tools (deduped).
7. `await chosen_entity.do_async(final_task, ...)` and return `final_task.response`.

#### `astream(tasks=None, debug=False)`

A reduced-feature analogue of `multi_agent_async` that yields `str` chunks:

| Mode         | What is streamed                                                                   |
|--------------|------------------------------------------------------------------------------------|
| `sequential` | Per task: a `--- [<Type>] <id> ---` header + the agent/team's text chunks. After the last task, the combined final answer (if it had to be combined) is appended. |
| `coordinate` | Header + the leader agent's text chunks. Member outputs are NOT streamed.          |
| `route`      | Router runs (non-stream), then header for the chosen specialist + its text chunks. |

`stream(...)` wraps `astream` in a daemon thread driving an `asyncio.run(...)` and a `queue.Queue`,
so callers in synchronous code can iterate without setting up an event loop themselves.

#### `as_mcp(name=None)`

Wraps the team as a FastMCP server (lazy import; missing fastmcp triggers an
`upsonic.utils.printing.import_error` with the right install command). Adds a single tool:

```python
@server.tool(description=tool_description)
def do(task: str) -> str:
    task_obj = Task(description=task, response_format=team_ref.response_format)
    result = team_ref.print_do(tasks=task_obj)
    return "" if result is None else str(result)
```

`tool_description` includes the team's role, goal, member names, and mode.

#### `add_tool()`

When `ask_other_team_members=True`, every `Agent` in `entities` is appended to each initial
task's `tools` list (creating it if missing). Sub-Team entities are skipped — they aren't
callable as tools.

### 3.3 `coordinator_setup.py` — `CoordinatorSetup`

Generates the leader/router system prompts for `coordinate` and `route` modes.

| Method                              | Purpose                                                                |
|-------------------------------------|------------------------------------------------------------------------|
| `__init__(members, tasks, mode)`    | Stores members + tasks + mode (`"coordinate"` or `"route"`).           |
| `_summarize_tool(tool)`             | Returns `"<name>: <docstring>"` (or `"No description available."`).    |
| `_format_entity_manifest()`         | Produces the **TEAM ROSTER** block. Differentiates Agents vs Teams.    |
| `_serialize_context_item(item)`     | String-coerces context items (`str`, `Task`, `KnowledgeBase`, fallback `str()`). |
| `_format_tasks_manifest()`          | XML-ish `<Tasks><Task index='i'>...</Task></Tasks>` block.             |
| `create_leader_prompt()`            | Dispatches to one of the two builders below.                           |
| `_create_coordinate_prompt()`       | The coordinate-mode prompt (intel package + delegation protocol).      |
| `_create_route_prompt()`            | The route-mode prompt (one tool call, one decision, then stop).        |

Manifest format for an Agent member:

```
- Member ID: `<entity_id>`
  - Role: <role>
  - Goal: <goal>
  - System Prompt: <agent.system_prompt>
  - Agent Tools:
    - <tool_name>: <docstring>
    - ...
```

For a nested Team member:

```
- Member ID: `<entity_id>`
  - Type: Team (<mode> mode)
  - Role: <role>
  - Goal: <goal>
  - Sub-entities: <comma-separated child ids>
```

The coordinate prompt instructs the leader to use `delegate_task(member_id, description, tools,
context, attachments)`. The route prompt instructs the router to call
`route_request_to_member(member_id)` exactly once and then stop.

### 3.4 `delegation_manager.py` — `DelegationManager`

Generates the dynamic tools that the leader/router actually calls. It is **mode-agnostic**: the
same instance can produce both tools, but only one is used per mode.

```python
class DelegationManager:
    def __init__(self, members, tool_mapping: Dict[str, Callable], debug: bool = False):
        self.members = members
        self.tool_mapping = tool_mapping
        self.debug = debug
        self.routed_entity = None  # Filled by route_request_to_member
```

#### `get_delegation_tool()` → `delegate_task(...)`

Returns an async function:

```python
async def delegate_task(
    member_id: str,
    description: str,
    tools: Optional[List[str]] = None,
    context: Any = None,
    attachments: Optional[List[str]] = None,
    expected_output: Union[Type[BaseModel], type[str], None] = None,
) -> str: ...
```

Steps inside the function:

1. Look up `member_id` in `self.members` by `entity.get_entity_id()`. Return an error string
   if not found (the leader sees the message and can retry).
2. Resolve task-level tools by name from `self.tool_mapping`. Unknown names are silently
   dropped.
3. Build a fresh `Task(description=..., tools=[...], context=context, attachments=attachments,
   response_format=str|expected_output)`.
4. If `debug=True`, emit an `info_log("Coordinate mode — Delegating to <Type> '<id>' | ...")`.
5. `await member_entity.do_async(sub_task)` and return `sub_task.response or "The team member
   did not return a result."`.
6. Any exception is caught and surfaced as the tool's return string (so the leader can recover).

The function is decorated with two attributes the Upsonic tool layer checks:

```python
delegate_task._upsonic_tool_config = ToolConfig(timeout=None, max_retries=0)
delegate_task._upsonic_is_tool = True
```

`timeout=None` is critical: a delegated sub-task can take far longer than the default 30-second
tool timeout because it runs an entire sub-pipeline.

#### `get_routing_tool()` → `route_request_to_member(member_id)`

Async function. Looks up the member, sets `self.routed_entity = chosen_entity`, and returns a
confirmation string. After the router calls this tool, the outer `Team.multi_agent_async`
reads `delegation_manager.routed_entity` to decide who actually runs the consolidated final
task.

### 3.5 `context_sharing.py` — `ContextSharing`

Pure utility class. All static methods.

| Method                                                           | Returns                                                             |
|------------------------------------------------------------------|---------------------------------------------------------------------|
| `_describe_entity(entity)`                                       | Human-readable description string for an Agent or Team.             |
| `enhance_task_context(current_task, all_tasks, idx, ents, done)` | Mutates `current_task.context` in place.                            |
| `build_selection_context(current_task, all_tasks, idx, ents, done)` | Returns a list passed to the entity-selection LLM.               |

`_describe_entity` produces:

- For Teams: `"Entity '<id>' (Team, mode=<mode>) — Role: ... Goal: ... Sub-entities: [...]"`.
- For Agents: `"Entity '<id>' (Agent) — Role: ... Goal: ... Tools: [<names>]"`.

`enhance_task_context` ensures `task.context` is a list, then appends every other task and every
entity configuration. This gives the chosen executor visibility into siblings and team
composition.

`build_selection_context` returns `[current_task] + other_tasks + [entity descriptions]`. Note
that **entity objects are converted to strings** here — the selection LLM receives readable
descriptions, not raw object references.

### 3.6 `task_assignment.py` — `TaskAssignment`

Picks an entity for a sequential-mode task.

| Method                                                                                  | Role                                                                                  |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `prepare_entities_registry(entity_configurations)`                                       | `(dict[id → entity], list[ids])`. IDs come from `get_entity_id()`.                    |
| `_find_selection_model(entity_configurations)`                                           | First-choice: any Agent's model. Then any Team's `model`. Then nested team search.    |
| `select_entity_for_task(current_task, context, registry, names, configs)`                | Returns the chosen entity name or `None`.                                             |

Selection algorithm:

1. **Pre-bound agent**: if `current_task.agent` is set and matches an entity (by identity or by
   `get_entity_id()`), return that name immediately.
2. Define an inline `SelectedEntity(BaseModel)` with one field `selected_agent: str`.
3. Resolve a model via `_find_selection_model`. Raise `ValueError` if none.
4. Loop up to 3 times:
   - Build a `selecting_task` (`description="Select the most appropriate agent ...",
     attachments=current_task.attachments, response_format=SelectedEntity, context=context`).
   - `await Agent(model=selection_model).do_async(selecting_task)`.
   - If response is the structured `SelectedEntity`, take `selected_agent`.
   - Try exact match against `registry`; otherwise try case-insensitive substring match against
     each known name (in either direction).
5. If all 3 attempts fail, return the **first** entity name as a last-resort fallback (or `None`
   if there are no entities).

This is the only place in the folder that calls an LLM purely for routing logic.

### 3.7 `result_combiner.py` — `ResultCombiner`

Used in sequential mode to fuse N task responses into one final answer.

```python
class ResultCombiner:
    def __init__(self, model=None, debug=False): ...
    def should_combine_results(self, results: List[Task]) -> bool: ...
    def get_single_result(self, results: List[Task]) -> Any: ...
    async def combine_results(self, results, response_format=str, entities=None) -> Any: ...
```

| Method                       | Behavior                                                              |
|------------------------------|-----------------------------------------------------------------------|
| `should_combine_results`     | `len(results) > 1`.                                                   |
| `get_single_result`          | Returns `results[0].response` (or `None` if empty).                   |
| `combine_results`            | Builds a fixed-prompt `Task`, runs it on a fresh `Agent`, returns it. |

The combiner builds a `Task` with:

- A long instruction string that says "combine results from all previous tasks ... if one
  question, return that answer; if multiple, return all with a summary".
- `context = results` (the list of completed `Task` objects, used by the agent for prior
  responses).
- `response_format = response_format` (so the user-specified format is enforced for the final
  combined output).

`debug_setting` is the combiner's own `self.debug`; if false, it falls back to the **last
entity's** `debug` flag (matching Sequential's heuristic). The combiner raises `ValueError` if
no model is available.

## 4. Subfolders

There are no subfolders inside `src/upsonic/team/`. All code lives at the top level.

## 5. Cross-file relationships

```
                                ┌──────────────────────────┐
                                │         Team             │
                                │  (team.py, entry point)  │
                                └────────────┬─────────────┘
                                             │
            ┌────────────────────────────────┼─────────────────────────────────┐
            │                                │                                 │
            ▼                                ▼                                 ▼
   ┌────────────────────┐         ┌──────────────────────┐         ┌────────────────────────┐
   │  Sequential mode   │         │   Coordinate mode    │         │     Route mode         │
   └────────┬───────────┘         └──────────┬───────────┘         └─────────────┬──────────┘
            │                                │                                   │
            │                                │                                   │
            ▼                                ▼                                   ▼
 ┌────────────────────┐         ┌──────────────────────────┐       ┌──────────────────────────┐
 │  ContextSharing    │         │   CoordinatorSetup       │       │   CoordinatorSetup       │
 │  TaskAssignment    │         │   DelegationManager      │       │   DelegationManager      │
 │  ResultCombiner    │         │   (delegate_task tool)   │       │   (route_request tool)   │
 └────────────────────┘         └──────────────────────────┘       └──────────────────────────┘
            │                                │                                   │
            └────────────────┬───────────────┴─────────────────┬─────────────────┘
                             ▼                                 ▼
                ┌─────────────────────────┐       ┌────────────────────────────┐
                │    upsonic.agent.Agent  │       │     upsonic.tasks.Task     │
                │  (executes sub-tasks)   │       │   (request/response unit)  │
                └─────────────────────────┘       └────────────────────────────┘
```

| File                    | Imports from team                                  | Imports from outside                                                                       |
|-------------------------|----------------------------------------------------|--------------------------------------------------------------------------------------------|
| `team.py`               | `coordinator_setup`, `delegation_manager`, `context_sharing`, `task_assignment`, `result_combiner` | `Task`, `Agent` (lazy), `Memory` (TYPE_CHECKING), `FastMCP` (lazy), `utils.logging_config`, `utils.printing`, `Skills` (lazy) |
| `coordinator_setup.py`  | —                                                  | `Task`, `Agent`, `KnowledgeBase`, `Team` (all TYPE_CHECKING)                               |
| `delegation_manager.py` | —                                                  | `Task`, `tools.config.ToolConfig`/`tool`, `utils.printing.info_log` (lazy), `Agent`/`Team` (TYPE_CHECKING) |
| `context_sharing.py`    | —                                                  | `Task`, `Agent`/`Team` (TYPE_CHECKING)                                                     |
| `task_assignment.py`    | —                                                  | `Task`, `Agent` (lazy), `pydantic.BaseModel`, `Team` (TYPE_CHECKING)                       |
| `result_combiner.py`    | —                                                  | `Task`, `Agent`, `Team` (TYPE_CHECKING)                                                    |

Notable patterns:

- All cross-package imports of `Agent` are **lazy or `TYPE_CHECKING`** to avoid circular imports
  with `upsonic.agent`. The actual `from upsonic.agent.agent import Agent as AgentClass` happens
  inside method bodies.
- `result_combiner.py` is the only file that imports `Agent` at module top-level (because it
  always uses it).
- `Team` instances are detected duck-style in helper modules via
  `hasattr(entity, "entities") and hasattr(entity, "mode")` rather than `isinstance` checks
  (this allows nested teams without a circular `Team` import).

## 6. Public API

What `from upsonic.team import ...` gives you (per `__all__`):

| Symbol           | Kind  | Purpose                                                                  |
|------------------|-------|--------------------------------------------------------------------------|
| `Team`           | class | Multi-agent coordination entry point.                                    |
| `ContextSharing` | class | Helper for building/enhancing task context (mostly internal).            |
| `TaskAssignment` | class | Helper for entity selection in sequential mode (mostly internal).        |
| `ResultCombiner` | class | Helper for combining sequential-mode results (mostly internal).          |

`CoordinatorSetup` and `DelegationManager` are **not** exported via `__init__.py` — they are
implementation details, but they are importable directly from their submodules if needed.

### Typical usage

```python
from upsonic import Agent, Task
from upsonic.team import Team

researcher = Agent(name="Researcher", role="Research", goal="Find facts",
                   model="openai/gpt-4o")
writer = Agent(name="Writer", role="Writing", goal="Compose prose",
               model="openai/gpt-4o")

# Sequential mode: pick an entity per task with an LLM, combine at the end.
team = Team(
    entities=[researcher, writer],
    tasks=[Task("Research climate change in 2025"),
           Task("Write a 200-word summary")],
    mode="sequential",
    model="openai/gpt-4o-mini",  # used for selection + combiner
)
result = team.do()

# Coordinate mode: a leader agent delegates via delegate_task.
team = Team(entities=[researcher, writer], mode="coordinate",
            model="openai/gpt-4o")
result = team.do(Task("Produce a research-backed essay on climate change"))

# Route mode: a router picks one specialist for the whole mission.
team = Team(entities=[researcher, writer], mode="route",
            model="openai/gpt-4o-mini")
result = team.do(Task("What is the boiling point of water at 1 atm?"))

# Streaming
async for chunk in team.astream("Write a haiku about Saturn"):
    print(chunk, end="")

# Expose as MCP server
mcp_server = team.as_mcp(name="My Research Team")
mcp_server.run()  # stdio by default
```

### Backwards compatibility

- `agents=` is accepted as an alias for `entities=` in the constructor.
- The `agents` property has both a getter and a setter that proxy to `entities`.

## 7. Integration with the rest of Upsonic

### 7.1 Composition rather than inheritance

`Team` does **not** inherit from `Agent`. Instead:

- It **owns** Agent instances (and possibly nested Teams) via `self.entities`.
- It builds an `Agent` on the fly for `coordinate`/`route` (when no leader/router is provided)
  and for sequential's selection LLM and result combiner.
- It calls `agent.do_async(task)` and `agent.astream(task, events=False)` for execution.

Both `Agent` and `Team` implement a uniform interface:

| Method               | Contract                                                                 |
|----------------------|--------------------------------------------------------------------------|
| `get_entity_id()`    | Stable display ID used by manifests/registries.                          |
| `do_async(task, **)` | Run the task; populate `task.response`. Accepts `_print_method_default`. |
| `astream(task)`      | Yield text chunks (Team accepts a `debug` keyword; Agent accepts `events`).|

This duck-typed interface is what makes nested teams transparent.

### 7.2 Tasks (`upsonic.tasks.tasks.Task`)

Every concrete unit of work is a `Task`. `Team` mutates and reads:

| Field             | Used by                                                                   |
|-------------------|---------------------------------------------------------------------------|
| `description`     | All modes; `CoordinatorSetup` task manifest; `route` mode consolidation.  |
| `tools`           | All modes; coordinate mode collects `tool_mapping`; route mode dedupes.   |
| `attachments`     | Master / final tasks aggregate attachments across all tasks.              |
| `context`         | Sequential mode appends siblings + entity descriptions.                   |
| `response`        | Read by `ResultCombiner.get_single_result`, set after `do_async`.         |
| `response_format` | Forwarded to the master/final task and the combiner.                     |
| `agent`           | Pre-binding hint: if set, sequential mode skips the LLM selection step.   |

### 7.3 Memory and skills

| Concern  | How `Team` integrates                                                                          |
|----------|------------------------------------------------------------------------------------------------|
| Memory   | `Team(memory=...)` is propagated into every contained `Agent` (including those in sub-teams) that does not already have a `Memory`. In `coordinate` mode, the auto-created or supplied leader inherits team memory if it has none. |
| Skills   | `Team(skills=...)` is **copied per agent** so that metrics stay independent. Skills get registered via `agent.add_tools(skills.get_tools())`. If an agent already has skills, they are merged with `Skills.merge(skills, entity.skills)` and old `get_skill_*` tools are removed first via `agent.remove_tools(...)`. |

### 7.4 Tool layer

`DelegationManager.get_delegation_tool` produces a function decorated with two markers consumed
by `upsonic.tools.config`:

- `_upsonic_is_tool = True` — registers it as a tool.
- `_upsonic_tool_config = ToolConfig(timeout=None, max_retries=0)` — disables the default 30s
  tool timeout and retry logic, since delegation can take an arbitrary amount of time and the
  underlying agent already handles retries.

### 7.5 MCP

`Team.as_mcp()` lazily imports `fastmcp.FastMCP`. If `fastmcp` is not installed, it surfaces a
formatted error from `upsonic.utils.printing.import_error` advising
`pip install 'upsonic[mcp]'`. The single MCP tool (`do(task: str)`) constructs a `Task` and
calls `print_do(...)`, returning the stringified response. The tool description is built from
the team's role, goal, member names, and mode.

### 7.6 Logging

Several places call `from upsonic.utils.printing import info_log` or `debug_log_level2`:

- `team.py`: per-task selection log in sequential mode (level 2 dump of selection details and
  result preview).
- `team.py`: per-mode action log (`Sequential mode — Calling ...`, `Route mode — Routed to ...`).
- `delegation_manager.py`: `Coordinate mode — Delegating to ...` per delegation call.

All gated on `self.debug` (and `debug_level >= 2` for the verbose dumps).

### 7.7 Print/output control

The print resolution order is exactly the same as `Agent`:

```
UPSONIC_AGENT_PRINT (env)  >  Team(print=...)  >  method default
                                                    ├── do(): False
                                                    └── print_do(): True
```

Setting `UPSONIC_AGENT_PRINT=false` globally disables all team output; `Team(print=False)`
suppresses output for both `do()` and `print_do()` unless the env says otherwise. `do_async`
and `print_do_async` follow the same hierarchy via `_print_method_default`.

## 8. End-to-end flow of a team execution

### 8.1 Sequential mode (`team.do(tasks=[T1, T2])`)

```
Caller
  │  team.do([T1, T2])
  ▼
Team._run_sync(team.multi_agent_async(entities, tasks, _print_method_default=...))
  │
  ▼
Team.multi_agent_async (mode="sequential")
  │ 1. ContextSharing(); TaskAssignment(); ResultCombiner(model = self.model or _find_first_model())
  │ 2. registry, names = TaskAssignment.prepare_entities_registry(entities)
  │ 3. for i, T in enumerate(tasks):
  │      a. ctx = ContextSharing.build_selection_context(T, tasks, i, entities, all_results)
  │      b. selected_id = await TaskAssignment.select_entity_for_task(T, ctx, registry, names, entities)
  │           - if T.agent set: skip LLM
  │           - else: ask SelectedEntity LLM up to 3 times; substring match; fallback to first
  │      c. ContextSharing.enhance_task_context(T, tasks, i, entities, all_results)
  │      d. await registry[selected_id].do_async(T, _print_method_default=...)
  │      e. all_results.append(T)
  │ 4. if len(all_results) == 1: return all_results[0].response
  │    else: await ResultCombiner.combine_results(all_results, self.response_format, entities)
  ▼
Result  (raw response or combined final answer; may be a Pydantic model if response_format is one)
```

### 8.2 Coordinate mode (`team.do(Task("..."))` with `mode="coordinate"`)

```
Caller
  │  team.do(Task("Big mission"))
  ▼
Team.multi_agent_async (mode="coordinate")
  │ 1. validate leader-or-model
  │ 2. tool_mapping = {tool.__name__: tool for tool in (t.tools for t in self.tasks)}
  │ 3. setup = CoordinatorSetup(entities, tasks, "coordinate")
  │    dm    = DelegationManager(entities, tool_mapping, debug)
  │ 4. self.leader_agent = self._leader or Agent(model=self.model, memory=self.memory)
  │    self.leader_agent.system_prompt = setup.create_leader_prompt()
  │ 5. master = Task(description="Begin your mission...", attachments=..., 
  │                  tools=[dm.get_delegation_tool()], response_format=self.response_format)
  │ 6. final = await self.leader_agent.do_async(master, ...)
  │
  │   Inside the leader's tool loop:
  │     leader → delegate_task(member_id, description, tools=[names], context=..., attachments=...)
  │       │ 1. find member by get_entity_id() (Agent or nested Team)
  │       │ 2. resolve task-level tools by name from tool_mapping
  │       │ 3. sub_task = Task(...)
  │       │ 4. (debug?) info_log(...)
  │       │ 5. await member.do_async(sub_task)
  │       │ 6. return sub_task.response or "did not return a result"
  │     leader can call delegate_task many times, threading earlier results as `context`.
  │     Final assistant message becomes `final`.
  ▼
Result  (leader's final synthesized answer in self.response_format)
```

### 8.3 Route mode (`team.do(Task("..."))` with `mode="route"`)

```
Caller
  │  team.do(Task("Single mission"))
  ▼
Team.multi_agent_async (mode="route")
  │ 1. validate router-or-model
  │ 2. setup = CoordinatorSetup(entities, tasks, "route")
  │    dm    = DelegationManager(entities, {}, debug)
  │ 3. self.leader_agent = self._router or Agent(model=self.model)
  │    self.leader_agent.system_prompt = setup.create_leader_prompt()
  │ 4. router_task = Task("Analyze ... and route", tools=[dm.get_routing_tool()])
  │ 5. await self.leader_agent.do_async(router_task)
  │       router → route_request_to_member(member_id) → dm.routed_entity = chosen
  │ 6. chosen = dm.routed_entity (raise ValueError if None)
  │ 7. final_task = Task(
  │        description=" ".join(t.description for t in tasks),
  │        attachments=union of attachments,
  │        tools=set(union of tools),
  │        response_format=self.response_format)
  │ 8. (debug?) info_log("Route mode — Routed to ...")
  │ 9. await chosen.do_async(final_task, ...)
  ▼
Result  (final_task.response from the chosen specialist)
```

### 8.4 Streaming variant (sequential mode)

```
async for chunk in team.astream(tasks):
    ...
```

For each task:
1. Build selection context (same as `multi_agent_async`).
2. Pick an entity via `TaskAssignment.select_entity_for_task`.
3. Yield a header `--- [Agent|Team] <id> ---`.
4. Yield each text chunk from `entity.astream(...)` (or `_entity_astream` fallback).
5. Append the task to `all_results`.

After all tasks: if multiple, `ResultCombiner.combine_results(...)` runs and its final string is
yielded as one chunk (no header).

In coordinate mode, only the leader's `astream(events=False)` is forwarded; member sub-streams
are not surfaced. In route mode, the router's call is awaited (non-stream), then the chosen
specialist is streamed.

### 8.5 Memory propagation flow (constructor)

```
Team(entities=[A, SubTeam([B, C])], memory=M)
   │
   ▼
_propagate_memory(entities=[A, SubTeam], memory=M)
   │  for A:        if A.memory is None → A.memory = M
   │  for SubTeam:  if SubTeam.memory is None → SubTeam.memory = M
   │                _propagate_memory(SubTeam.entities, M)
   │                   │  for B: if B.memory is None → B.memory = M
   │                   │  for C: if C.memory is None → C.memory = M
   ▼
All agents and sub-teams that had no memory now share M.
```

### 8.6 Skills propagation flow (constructor)

```
Team(entities=[A, SubTeam([B])], skills=S)
   │
   ▼
_propagate_skills(entities=[A, SubTeam], skills=S)
   │  for A (Agent):
   │     if A.skills is None:
   │        A.skills = S.copy()           # independent metrics per agent
   │        A.add_tools(A.skills.get_tools())
   │     else:
   │        old = [t for t in A.tools if t.__name__.startswith('get_skill_')]
   │        A.remove_tools(old)
   │        A.skills = Skills.merge(S, A.skills)
   │        A.add_tools(A.skills.get_tools())
   │  for SubTeam (Team):
   │     if SubTeam.skills is None: SubTeam.skills = S
   │     _propagate_skills(SubTeam.entities, S)
```

### 8.7 Sync-from-async safety net

```
team.do(...)
   │
   ▼
Team._run_sync(coro)
   │  try: asyncio.get_running_loop()
   │       → already in an event loop?
   │           ├── yes: ThreadPoolExecutor → asyncio.run(coro) in worker thread
   │           └── no:  asyncio.run(coro) directly
   ▼
Result returned to caller as if synchronous.
```

This is what allows `Team(...).do()` to work in Jupyter, scripts, and async frameworks alike.

---

## Quick reference: file ↔ class ↔ public methods

| File                    | Class               | Public methods                                                                                                                       |
|-------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `team.py`               | `Team`              | `do`, `print_do`, `complete`, `print_complete`, `do_async`, `print_do_async`, `ado`, `multi_agent_async`, `astream`, `stream`, `as_mcp`, `add_tool`, `get_entity_id`, `agents` |
| `coordinator_setup.py`  | `CoordinatorSetup`  | `create_leader_prompt`                                                                                                               |
| `delegation_manager.py` | `DelegationManager` | `get_delegation_tool`, `get_routing_tool`                                                                                            |
| `context_sharing.py`    | `ContextSharing`    | `enhance_task_context` (static), `build_selection_context` (static)                                                                  |
| `task_assignment.py`    | `TaskAssignment`    | `prepare_entities_registry`, `select_entity_for_task`                                                                                |
| `result_combiner.py`    | `ResultCombiner`    | `should_combine_results`, `get_single_result`, `combine_results`                                                                     |
| `__init__.py`           | —                   | Re-exports `Team`, `ContextSharing`, `TaskAssignment`, `ResultCombiner` (lazy).                                                      |

## Quick reference: behavior by mode

| Concern                         | sequential                                       | coordinate                                          | route                                                |
|---------------------------------|--------------------------------------------------|-----------------------------------------------------|------------------------------------------------------|
| Who picks the executor          | An ad-hoc selection LLM (per task)               | The leader agent (per delegation)                   | The router agent (one decision)                      |
| Setup helper                    | `TaskAssignment` + `ContextSharing`              | `CoordinatorSetup` + `DelegationManager`            | `CoordinatorSetup` + `DelegationManager`             |
| Tool the LLM must call          | (none — selection is response-format LLM)        | `delegate_task(...)` (many calls)                   | `route_request_to_member(member_id)` (one call)      |
| Task fan-out                    | One executor per task; results combined at end   | Leader fans out as needed; assembles final response | All tasks consolidated into one final task           |
| Final synthesis                 | `ResultCombiner.combine_results` (if N > 1)      | Leader's last assistant message                     | Chosen specialist's `final_task.response`            |
| Leader/router required          | No                                               | Yes (auto-built from `model` if not supplied)       | Yes (auto-built from `model` if not supplied)        |
| Streaming behavior              | Per-task headers + chunks + combined tail        | Leader header + leader chunks                       | Specialist header + specialist chunks                |
| Validates                       | Sees a model anywhere in the team                | `leader is not None or model is not None`           | `router is not None or model is not None`            |
