---
name: ralph-autonomous-dev-loop
description: Use when working with Upsonic's RalphLoop / Groundhog autonomous AI development loop in src/upsonic/ralph/, including its three-phase pipeline (Requirements, TODO, Incremental), workspace state files (PROMPT.md, specs/*.md, fix_plan.md, AGENT.md), backpressure validation, and disposable fresh-context agents. Use when a user asks to build, configure, debug, or extend RalphLoop, RalphConfig, RalphLoopResult, IncrementalPhase, RequirementsPhase, TodoPhase, StateManager, BackpressureGate, or the Ralph toolkits (PlanUpdaterToolKit, LearningsUpdaterToolKit, BackpressureToolKit, RalphFilesystemToolKit, SubagentSpawnerToolKit). Trigger when the user mentions Ralph, RalphLoop, Groundhog technique, fix_plan.md, AGENT.md, PROMPT.md, backpressure gate, spawn_subagent, autonomous coding loop, eventually-consistent agent, disposable agents, or workspace-sandboxed filesystem tools.
---

# `src/upsonic/ralph/` — RalphLoop: Autonomous AI Development Loop

## 1. What this folder is

`src/upsonic/ralph/` is Upsonic's **implementation of the "Ralph / Groundhog" technique** for autonomous, eventually-consistent AI-driven software development. It exposes a single high-level entry point — `RalphLoop` — that turns a *high-level project goal* (e.g. `"Build a FastAPI TODO app"`) into actual working code on disk by running an open-ended loop where:

1. A **fresh `Agent` instance** is spawned every iteration with clean context.
2. The agent picks **exactly one** unchecked item from a TODO file (`fix_plan.md`).
3. It uses **subagents** for expensive operations (search / read / write / test) so the primary context stays small.
4. After implementing, it runs a **backpressure gate** (build / test / lint commands) — only if backpressure passes is the item marked `[x]`.
5. Any learnings (build commands that work, gotchas, patterns) are appended to `AGENT.md` so future iterations benefit.
6. The loop repeats until `fix_plan.md` has no `- [ ]` items left, `max_iterations` is hit, or `SIGINT` is received.

The core insight encoded here is: **state lives on disk in markdown files, not in the agent's memory**. Each iteration reloads `PROMPT.md`, `specs/*.md`, `fix_plan.md`, and `AGENT.md` from the workspace; this is the "deterministic stack" that survives the disposable, fresh-context agents.

The name `ralph` comes from the project itself (the Ralph/Groundhog technique — every iteration is "Groundhog Day": clean context, same files, slow forward progress).

### Three-phase pipeline

| Phase | Purpose | Output | Run count |
|-------|---------|--------|-----------|
| **Requirements** | Goal → spec files | `specs/*.md` + `PROMPT.md` | Once (skipped if specs exist or pre-supplied) |
| **TODO** | Specs → prioritized checkbox list | `fix_plan.md` | Once (skipped if file exists) |
| **Incremental** | Loop: pick one unchecked item, implement, validate, mark `[x]` | Source code under `src/`, updated `fix_plan.md`, updated `AGENT.md` | Repeated until done |

---

## 2. Folder layout

```
src/upsonic/ralph/
├── __init__.py                  # Lazy public API: RalphLoop, RalphConfig, RalphLoopResult
├── config.py                    # RalphConfig dataclass (paths, commands, callbacks)
├── result.py                    # IterationRecord + RalphLoopResult Pydantic models
├── loop.py                      # RalphLoop orchestrator (run / arun / stop)
│
├── state/
│   ├── __init__.py              # Lazy exports: StateManager, RalphState
│   ├── models.py                # RalphState (in-memory Pydantic snapshot of disk)
│   └── manager.py               # StateManager (read/write PROMPT.md/specs/*/fix_plan.md/AGENT.md)
│
├── backpressure/
│   ├── __init__.py              # Lazy exports: BackpressureGate, BackpressureResult
│   └── gate.py                  # subprocess runner for build/test/lint + ValidationResult
│
├── phases/
│   ├── __init__.py              # Lazy exports: BasePhase, RequirementsPhase, TodoPhase, IncrementalPhase
│   ├── base.py                  # BasePhase ABC + PhaseResult dataclass
│   ├── requirements.py          # Goal -> specs/*.md (LLM call, regex parse)
│   ├── todo.py                  # specs -> fix_plan.md (LLM call, normalize to "- [ ]")
│   └── incremental.py           # Single-iteration scheduler agent (the inner loop body)
│
└── tools/
    ├── __init__.py              # Lazy exports of all 5 toolkits
    ├── plan_updater.py          # update_fix_plan / update_spec tools
    ├── learnings_updater.py     # update_learnings tool (append to AGENT.md)
    ├── backpressure.py          # run_backpressure tool (sequential=True bottleneck)
    ├── filesystem.py            # read/write/edit/list/search/grep/run_command (workspace-sandboxed)
    └── subagent_spawner.py      # spawn_subagent (fresh Agent w/ filesystem tools, truncated result)
```

---

## 3. Top-level files file-by-file

### 3.1 `__init__.py` — Public surface

```python
from upsonic import RalphLoop, RalphConfig, RalphLoopResult
```

Uses **module-level `__getattr__`** to lazy-import the three classes only when accessed. This avoids pulling in `Agent`, `subprocess`, etc., when the rest of `upsonic` is imported.

```python
def __getattr__(name: str) -> Any:
    if name == "RalphLoop":   return _get_ralph_loop()
    if name == "RalphConfig": return _get_ralph_config()
    if name == "RalphLoopResult": return _get_ralph_result()
    raise AttributeError(...)
```

`__all__` advertises exactly: `["RalphLoop", "RalphConfig", "RalphLoopResult"]`.

### 3.2 `config.py` — `RalphConfig` dataclass

A frozen-style `@dataclass` that holds **everything the loop needs to know**. Validated in `__post_init__`.

| Field | Default | Role |
|-------|---------|------|
| `goal` | *required* | High-level project description (rejected if empty) |
| `model` | `"openai/gpt-4o"` | LLM passed to every `Agent` |
| `workspace` | `cwd / "ralph_project"` | Where files live; coerced to `Path` |
| `test_command` | `None` | Backpressure: e.g. `"pytest"` |
| `build_command` | `None` | Backpressure: e.g. `"pip install -e ."` |
| `lint_command` | `None` | Backpressure: e.g. `"ruff check ."` |
| `specs` | `None` | Pre-defined `dict[name, content]` — skips Requirements phase |
| `max_iterations` | `None` | Safety cap; must be ≥1 if set |
| `max_subagents` | `50` | Hint passed to phases (also validated ≥1) |
| `debug` | `False` | Reserved for verbose logging |
| `on_iteration` | `None` | Callback `IterationResult -> None` after each loop |
| `on_error` | `None` | Callback `Exception -> None` on failure |
| `show_progress` | `True` | Prints box-drawing UI to stdout |

Convenience properties expose canonical paths — used everywhere instead of string concatenation:

```python
config.workspace_path   # Path
config.specs_dir        # workspace/specs
config.src_dir          # workspace/src
config.prompt_file      # workspace/PROMPT.md
config.fix_plan_file    # workspace/fix_plan.md
config.learnings_file   # workspace/AGENT.md
```

`has_backpressure()` returns `True` if any of `build/test/lint_command` is set. `to_dict()` returns a JSON-serialisable summary.

### 3.3 `result.py` — Outcome models (Pydantic)

Two `BaseModel` types tracking what happened.

**`IterationRecord`** — one entry per loop iteration:

| Field | Type | Meaning |
|-------|------|---------|
| `iteration` | `int` | 1-indexed |
| `task_picked` | `str` | First pending item snapshot |
| `success` | `bool` | An item was marked `[x]` this iteration |
| `backpressure_passed` | `bool` | Inferred from `success` |
| `message` | `str` | First 200 chars of agent output (or error) |
| `execution_time` | `float` | Seconds |
| `learnings_added` | `List[str]` | (Reserved; never populated by current code) |

**`RalphLoopResult`** — overall report:

| Field | Notes |
|-------|-------|
| `goal`, `workspace` | Echoed from config |
| `total_iterations`, `successful_iterations`, `failed_iterations` | Counters maintained by `add_iteration()` |
| `final_status` | One of `"completed"`, `"max_iterations"`, `"stopped"`, `"error"` |
| `iterations` | `List[IterationRecord]` |
| `start_time`, `end_time` | `datetime` set by `RalphLoop.run` |
| `specs_generated` | Names from `specs/*.md` |
| `error_message` | `Optional[str]` |

Methods: `duration()`, `success_rate()`, `summary()` (formats a 60-col box-drawing report), `add_iteration()`, `to_dict()`.

### 3.4 `loop.py` — `RalphLoop` orchestrator

The entry-point class. Construction wires together all the pieces:

```python
self.config           = RalphConfig(...)                 # config.py
self.state_manager    = StateManager(workspace)          # state/manager.py
self.backpressure_gate = BackpressureGate(workspace, ...)# backpressure/gate.py
```

It also installs a `SIGINT` handler so Ctrl-C sets `self._should_stop = True` (the loop exits cleanly *after* the current iteration).

#### `run(max_iterations=None) -> RalphLoopResult` — sync driver

The whole machine, top-to-bottom:

```python
1. _print_header()                      # box-drawing UI
2. _run_requirements_phase()            # if no specs, ask LLM for them
3. _run_todo_phase()                    # if no fix_plan.md, generate it
4. while not self._should_stop:
       iter_result = IncrementalPhase(...).execute_iteration()
       self._result.add_iteration(record)
       if config.on_iteration: config.on_iteration(iter_result)
       if iter_result.plan_is_empty: final_status="completed"; break
       if iteration_count > max_iter:  final_status="max_iterations"; break
5. signal.signal(SIGINT, original_handler)   # restore handler in finally
```

`arun` is the same control-flow but `await`s the `_arun_*` / `aexecute_iteration` variants — every phase exposes both `execute()` and `aexecute()`.

`stop()` flips `_should_stop`. `get_state()` returns a snapshot dict from `StateManager.load_state()` (useful for live UIs).

The phase-runner methods (`_run_requirements_phase`, `_run_todo_phase` and async twins) implement the **idempotency** rule:

| Pre-state | Action |
|-----------|--------|
| `config.specs` is set | Save them to disk via `save_spec`, save default `PROMPT.md`, **skip LLM** |
| `state_manager.has_specs()` | Skip phase entirely (already done in a prior run) |
| Otherwise | Run `RequirementsPhase` for real |

The same idempotency holds for `_run_todo_phase` (skipped if `fix_plan.md` already has content). This is what makes RalphLoop **resumable** — kill the process, re-run with the same `workspace`, and it picks up where it left off.

---

## 4. Subfolders

### 4.1 `state/` — The deterministic stack

This is the on-disk layer. The Ralph technique deliberately **does not** trust the LLM to remember anything; everything important is a `.md` file in the workspace.

| File | Owner / mutator | Purpose |
|------|-----------------|---------|
| `PROMPT.md` | Written by `RequirementsPhase` | Main instructions injected into every iteration's `system_prompt` |
| `specs/*.md` | Written by `RequirementsPhase` | One spec file per major feature (parsed from ` ```spec:name ` LLM blocks) |
| `fix_plan.md` | Written by `TodoPhase`, mutated by `PlanUpdaterToolKit` | Checkbox TODO list (`- [ ]` / `- [x]`) |
| `AGENT.md` | Mutated by `LearningsUpdaterToolKit` | Accumulated learnings under `## BUILD` / `## TEST` / `## PATTERN` / `## GOTCHA` headers |

#### `state/models.py` — `RalphState` Pydantic model

A snapshot of all four file types in memory, plus parsing helpers.

| Method | Behavior |
|--------|----------|
| `format_specs_for_context()` | Joins `### {name}\n{content}` for all specs (sorted) |
| `format_for_context()` | Full agent context: `## SPECIFICATIONS` + `## TODO LIST (N pending, M completed)` + `## LEARNINGS` |
| `get_todo_items(include_completed=False)` | Parses `fix_plan.md`; understands `- [ ]`, `- [x]`, `- item`, `* item`, `1. item` |
| `get_pending_items()` / `get_completed_items()` | Filtered views |
| `is_plan_empty()` | True when 0 pending items remain (loop's exit condition) |
| `get_spec_names()`, `has_specs()`, `to_dict()` | Misc helpers |

#### `state/manager.py` — `StateManager`

The disk I/O layer. Constructor calls `_ensure_workspace_exists()` which creates `workspace/`, `workspace/specs/`, and `workspace/src/`.

Reading: `load_state() -> RalphState` loads all four files in one shot.

Writing primitives:

```python
save_prompt(content)               # PROMPT.md
save_spec(name, content)           # specs/{name}.md  (auto-appends .md)
update_fix_plan(content)           # replace fix_plan.md
append_to_fix_plan(item)           # normalize + append "- [ ] item"
complete_fix_plan_item(item)       # find substring match, "- [ ]" -> "- [x]"
remove_pending_item(item)          # delete unchecked match (completed are protected)
update_learnings(content)          # replace AGENT.md
append_learning(learning, cat)     # append "## CATEGORY\n- learning"
read_learnings()                   # read AGENT.md
has_specs(), has_fix_plan(), get_spec_names()
```

Two important invariants encoded in `complete_fix_plan_item` / `remove_pending_item`:

1. **Completed tasks are immutable history.** `remove_pending_item` explicitly skips lines starting with `- [x]`.
2. **Substring matching is case-insensitive** — the agent doesn't need to quote tasks exactly.

### 4.2 `backpressure/` — The validation gate

This is the **truth oracle**. The agent can claim it implemented something, but the loop only believes it if `subprocess.run(test_command)` exits 0.

#### `backpressure/gate.py`

Two dataclasses:

- **`ValidationResult`** — one command's outcome: `validation_type`, `passed`, `command`, `returncode`, `stdout`, `stderr`, `execution_time`, `no_tests_found`. `get_error_summary(max_length=500)` returns truncated stderr/stdout for LLM context.
- **`BackpressureResult`** — aggregate. `format_for_agent()` returns one of:
  - `"PASS"`
  - `"PASS (no tests found - consider adding tests)"`
  - `"FAIL: {type} failed:\n{error_summary}"`

**`BackpressureGate`** runs commands serially (build → test → lint) and **short-circuits on first failure**:

```python
def validate(self, validation_type="all"):
    for cmd in [build, test, lint]:
        if cmd:
            r = self._run_validation(...)
            if not r.passed: return BackpressureResult(passed=False, ...)
    return BackpressureResult(passed=True, ...)
```

Special handling:

| Case | Behavior |
|------|----------|
| `pytest` test command without `--rootdir` | Auto-rewritten to `pytest --rootdir=. ...` to prevent picking up parent configs |
| `subprocess.TimeoutExpired` (default 300 s) | Returns `passed=False`, stderr `"Command timed out..."` |
| pytest exit-code 5, `"collected 0 items"`, `"no test specified"`, `"no tests found"`, `"no test files"` | `_is_no_tests_collected()` → `no_tests_found=True`, `passed=True` (so missing tests don't block progress) |

Both sync (`validate`) and async (`avalidate`) variants exist; async uses `asyncio.create_subprocess_shell` + `asyncio.wait_for` for the timeout.

### 4.3 `phases/` — The three stages

#### `phases/base.py`

| Class | Role |
|-------|------|
| `PhaseResult` | dataclass: `phase_name`, `success`, `message`, `data: dict`, `errors: list` |
| `BasePhase(ABC)` | Holds `state_manager`, `model`. Abstract: `name`, `execute()`, `aexecute()` |

#### `phases/requirements.py` — `RequirementsPhase`

Owns two prompt strings:

- `REQUIREMENTS_PROMPT` — system prompt instructing the LLM to emit ` ```spec:name ` blocks for each feature.
- `DEFAULT_PROMPT_TEMPLATE` — saved to `PROMPT.md`. This is the **constitution** the iterating agent reads every loop. It enumerates the eight critical rules ("ONE TASK ONLY", "QUICK SEARCH, THEN IMPLEMENT", "READ BEFORE EDITING", "NO PLACEHOLDERS", "WRITE TESTS", "BACKPRESSURE MUST PASS", "MARK COMPLETE", "RECORD LEARNINGS").

Execution:

```python
agent = Agent(model, system_prompt=REQUIREMENTS_PROMPT)
result = agent.do(Task(description=f"...{goal}..."))
specs = self._parse_specs(result_str)        # 3-tier regex fallback
for name, content in specs.items():
    state_manager.save_spec(name, content)
state_manager.save_prompt(DEFAULT_PROMPT_TEMPLATE.format(goal=goal))
```

`_parse_specs` tries (1) ` ```spec:name `, (2) ` ```name.md `, (3) plain markdown with `# ` heading → all under spec name `"main"`.

#### `phases/todo.py` — `TodoPhase`

Different prompt (`TODO_PROMPT`) — it pushes the LLM toward emitting:

```
```todo
- [ ] Set up project with requirements.txt (CRITICAL - do first)
- [ ] Create core data models with type annotations
- [ ] Implement <specific feature> in <specific module>
- [ ] Add unit tests for <specific function> with edge cases
```
```

`_parse_todo_list` extracts the ` ```todo ` block (or falls back to first contiguous list it finds) and **normalizes every variant** to the canonical `- [ ] task` checkbox form, including:

- `- task` → `- [ ] task`
- `* task` → `- [ ] task`
- `1. task` → `- [ ] task`
- Already-`[ ]`/`[x]` lines preserved
- Indented nested items keep their indentation

Notably, `TodoPhase` is itself given a `SubagentSpawnerToolKit` so it can read existing files in the workspace before deciding the TODO list (useful when bootstrapping into an existing codebase).

#### `phases/incremental.py` — `IncrementalPhase`

The body of the inner loop. Defines `IterationResult` dataclass and the `SCHEDULER_PROMPT` — a long instruction string that hammers home the same eight rules and emphasizes:

- Pick **exactly one** `- [ ]` item.
- Use subagents for: **search, read, write, test**.
- Test/build are **bottlenecks** — never spawn parallel test subagents.
- After backpressure passes, **immediately** call `update_fix_plan(action="complete", item=...)`.

Per-iteration flow (`execute_iteration`):

```python
1. _current_iteration += 1
2. state_before = state_manager.load_state()      # fresh snapshot
3. if state_before.is_plan_empty():               # no work left
       return IterationResult(plan_is_empty=True, success=True)
4. pending_count_before = len(state_before.get_pending_items())
5. system_prompt = state_before.prompt + "\n" + SCHEDULER_PROMPT
6. context      = state_before.format_for_context()
7. tools = [PlanUpdaterToolKit, LearningsUpdaterToolKit,
            BackpressureToolKit, RalphFilesystemToolKit,
            SubagentSpawnerToolKit]
8. agent = Agent(model, system_prompt, tools)
9. agent.do(Task(description=f"{context}\n## Your Task This Iteration\n..."))
10. state_after = state_manager.load_state()
11. task_completed = len(state_after.get_pending_items()) < pending_count_before
12. del agent                                     # disposable!
13. return IterationResult(success=task_completed, plan_is_empty=...)
```

The `success` flag is **derived from disk state** — the only way to "succeed" is to mark a `[ ]` as `[x]`. The agent can't lie about it. And since `complete_fix_plan_item` is in `PlanUpdaterToolKit` which writes to `fix_plan.md`, the disk-comparison check at step 11 is the source of truth.

### 4.4 `tools/` — The five toolkits

All five extend `upsonic.tools.ToolKit` and use the `@tool` decorator from `upsonic.tools`. They are passed to `Agent(..., tools=[...])`.

#### 4.4.1 `tools/plan_updater.py` — `PlanUpdaterToolKit`

| Tool | Args | Effect |
|------|------|--------|
| `update_fix_plan` | `action: "add"\|"complete"\|"delete"\|"replace"`, `item`, `new_content` | Dispatches to `StateManager.append_to_fix_plan` / `complete_fix_plan_item` / `remove_pending_item` / `update_fix_plan` |
| `update_spec` | `spec_name`, `content` | `state_manager.save_spec(...)` |

The action verbs map directly to the StateManager primitives. Completed items are protected against deletion at the `StateManager` layer.

#### 4.4.2 `tools/learnings_updater.py` — `LearningsUpdaterToolKit`

| Tool | Args | Effect |
|------|------|--------|
| `update_learnings` | `learning: str`, `category: "build"\|"test"\|"pattern"\|"gotcha"` | `state_manager.append_learning(...)` |

The docstring contains explicit GOOD/BAD examples to bias the LLM toward writing actionable, *why*-focused notes ("When implementing API clients, always add a rate limiter decorator to prevent 429 errors") rather than vague ones ("Added rate limiter").

#### 4.4.3 `tools/backpressure.py` — `BackpressureToolKit`

```python
@tool(sequential=True, timeout=600.0)
def run_backpressure(self, validation_type="all") -> str:
    return self.backpressure_gate.validate(validation_type).format_for_agent()
```

**`sequential=True` is the key flag** — it tells Upsonic's tool runtime that this tool may not run in parallel with itself. This implements the "test/build is a bottleneck" rule at the tool layer.

Returns either `"PASS"`, `"PASS (no tests found - consider adding tests)"`, or `"FAIL: {type} failed:\n{stderr_truncated_to_500_chars}"`. The async twin `arun_backpressure` mirrors the sync one.

#### 4.4.4 `tools/filesystem.py` — `RalphFilesystemToolKit`

A workspace-sandboxed filesystem (used both by the primary scheduler agent **and** by every spawned subagent).

| Tool | Description |
|------|-------------|
| `read_file(path, offset, limit)` | Numbered-line output, summary `[Showing lines X-Y of N total]` |
| `write_file(path, content, create_dirs=True)` | Overwrite; auto-creates parent dirs |
| `edit_file(path, old_string, new_string, replace_all=False)` | String-replacement with "must read first" docstring warning |
| `list_files(directory, recursive=False, exclude_dirs)` | Defaults exclude `node_modules`, `__pycache__`, `.git`, `venv`, `.venv` for recursive |
| `search_files(pattern, directory, exclude_dirs)` | `Path.rglob(pattern)`; truncates at 100 |
| `grep_files(text, directory, file_pattern="*", exclude_dirs)` | Regex (escapes on compile error), case-insensitive, max 100 matches |
| `run_command(command, timeout=60)` | `subprocess.run(shell=True, cwd=workspace)`, output truncated at 5000 chars |

**Sandboxing**: every path is run through `_validate_path(path)` which resolves the path and calls `resolved.relative_to(self.workspace)` — `ValueError` if it escapes. Both relative and absolute inputs are accepted, but absolute ones still must resolve inside the workspace.

#### 4.4.5 `tools/subagent_spawner.py` — `SubagentSpawnerToolKit`

The mechanism that makes "fresh context every iteration" actually work for the *primary* agent. The primary spawns subagents for expensive work; the subagent's huge result gets truncated to ≤1000 chars before going into the primary's context.

```python
@tool(timeout=600.0)
def spawn_subagent(self, task_description: str,
                   purpose: Literal["search","analyze","write","test"]="search") -> str:
    subagent = Agent(model, system_prompt=SUBAGENT_SYSTEM_PROMPT.format(...),
                     tools=[RalphFilesystemToolKit(workspace)])
    result = subagent.do(Task(...))
    del subagent                           # garbage collect immediately
    return self._truncate_result(result)   # max 1000 chars by default
```

`_get_purpose_instructions(purpose)` injects different bullet lists into the subagent's task depending on whether it's `search`, `analyze`, `write`, or `test`. For instance:

| Purpose | Key instruction |
|---------|----------------|
| `search` | "Do ONE thorough search, then report. If nothing found, report 'No existing implementation found' and STOP." |
| `analyze` | "Read the ENTIRE file before analyzing." |
| `write` | "For EXISTING files: read first. For NEW files: just write directly. NO placeholders." |
| `test` | "BOTTLENECK — be efficient. Test actual behavior, not just that code runs." |

Subagent tools are *only* `RalphFilesystemToolKit` — no recursion, no sub-subagents, no `update_fix_plan`. This keeps the delegation tree exactly two levels deep.

---

## 5. Cross-file relationships

### Dependency graph

```
                    ┌──────────────────┐
                    │   __init__.py    │
                    │  (lazy exports)  │
                    └────────┬─────────┘
                             │
                ┌────────────▼──────────────┐
                │         loop.py           │
                │       RalphLoop           │
                └─┬──┬──┬──┬───────────────┘
                  │  │  │  │
        ┌─────────┘  │  │  └────────────────────────┐
        │            │  │                           │
        ▼            ▼  ▼                           ▼
   config.py    state/    backpressure/         phases/
  RalphConfig  ─manager  ─gate                 ─requirements
              ─models     ValidationResult     ─todo
                          BackpressureResult   ─incremental ─┐
                                                             │
                                                             ▼
                                                          tools/
                                                     ─plan_updater  ──► state/manager
                                                     ─learnings_updater ► state/manager
                                                     ─backpressure  ──► backpressure/gate
                                                     ─filesystem    (standalone)
                                                     ─subagent_spawner ► filesystem
                                                                       ► upsonic.agent.Agent
                  result.py
                  RalphLoopResult / IterationRecord
```

### Who creates whom

| Creator | Creates | When |
|---------|---------|------|
| `RalphLoop.__init__` | `RalphConfig`, `StateManager`, `BackpressureGate`, `RalphLoopResult` | At construction |
| `RalphLoop._run_requirements_phase` | `RequirementsPhase` | Phase 1 |
| `RalphLoop._run_todo_phase` | `TodoPhase` | Phase 2 |
| `RalphLoop.run` | `IncrementalPhase` (once, reused across iterations) | Phase 3 |
| `IncrementalPhase._create_tools` | All 5 toolkits + `Agent` | Each iteration |
| `RequirementsPhase.execute` | bare `Agent` (no tools) | Once |
| `TodoPhase.execute` | `SubagentSpawnerToolKit` + `Agent` | Once |
| `SubagentSpawnerToolKit.spawn_subagent` | `RalphFilesystemToolKit` + `Agent` (the subagent) | Per call |

### What flows where

- **Disk → memory**: `StateManager.load_state()` reads four files → returns `RalphState` Pydantic model.
- **Memory → disk**: `StateManager.save_prompt/save_spec/update_fix_plan/append_learning` etc. write back.
- **`RalphState.format_for_context()`** → injected into the `Task.description` for every iteration.
- **`PROMPT.md` content + `SCHEDULER_PROMPT`** → concatenated into the iteration `Agent`'s `system_prompt`.
- **Toolkit method calls (LLM-initiated)** → `StateManager` mutations or `BackpressureGate.validate()` calls.
- **`BackpressureGate.validate()`** → `subprocess.run(...)` against `cwd=workspace`.
- **`IncrementalPhase` post-iteration**: re-loads state, compares pending count, infers `task_completed`. Inferred boolean goes into `IterationResult` → `IterationRecord` → `RalphLoopResult.iterations`.

---

## 6. Public API

### Importable symbols

```python
from upsonic.ralph import RalphLoop, RalphConfig, RalphLoopResult
# (also re-exported at top level: from upsonic import RalphLoop)
```

### `RalphLoop` (the only class users construct)

```python
class RalphLoop:
    def __init__(
        self,
        goal: str,
        model: str = "openai/gpt-4o",
        workspace: Optional[Union[str, Path]] = None,
        test_command: Optional[str] = None,
        build_command: Optional[str] = None,
        lint_command: Optional[str] = None,
        specs: Optional[Dict[str, str]] = None,
        max_iterations: Optional[int] = None,
        max_subagents: int = 50,
        debug: bool = False,
        on_iteration: Optional[Callable[[IterationResult], None]] = None,
        on_error:     Optional[Callable[[Exception], None]] = None,
        show_progress: bool = True,
    ): ...

    # properties
    goal: str                                   # config.goal
    model: str                                  # config.model
    workspace: Path                             # config.workspace_path

    # control
    def run(max_iterations=None) -> RalphLoopResult           # blocking
    async def arun(max_iterations=None) -> RalphLoopResult   # async
    def stop() -> None                                        # graceful stop after iter
    def get_state() -> Dict[str, Any]                         # snapshot for UI
```

### `RalphConfig` (advanced — bypass kwargs)

All kwargs of `RalphLoop.__init__` are mirrored as fields on `RalphConfig`. Construct one directly only when sharing config across multiple loops or persisting it.

### `RalphLoopResult` (return value)

```python
result.summary()              # Human-readable formatted string
result.duration()             # float seconds
result.success_rate()         # float 0..100
result.to_dict()              # JSON-serialisable
result.iterations             # List[IterationRecord]
result.final_status           # "completed" | "max_iterations" | "stopped" | "error"
```

### Minimal usage

```python
from upsonic import RalphLoop

loop = RalphLoop(
    goal="Build a FastAPI TODO app with SQLite persistence",
    model="openai/gpt-4o",
    test_command="pytest",
    build_command="pip install -e .",
    max_iterations=20,
)
result = loop.run()
print(result.summary())
```

### Resume / pre-seed

```python
# Skip the requirements phase by supplying specs up-front:
loop = RalphLoop(
    goal="...",
    specs={
        "auth": "# Auth\n- Use JWT\n- ...",
        "api":  "# API\n- /todos endpoints...\n- ...",
    },
    workspace="./my_project",
)

# Or just re-point at an existing workspace — phases that already produced
# files (specs/*.md, fix_plan.md) are skipped automatically.
loop = RalphLoop(goal="...", workspace="./my_project").run()
```

### Streaming progress

```python
def on_iter(r):
    print(f"#{r.iteration} {'OK' if r.success else 'FAIL'} {r.task_picked}")

RalphLoop(goal="...", on_iteration=on_iter).run()
```

---

## 7. Integration with rest of Upsonic

### Hard dependencies

| Upsonic module | Used as | Where |
|----------------|---------|-------|
| `upsonic.agent.agent.Agent` | The LLM-driven worker | Imported lazily inside `RequirementsPhase.execute`, `TodoPhase.execute`, `IncrementalPhase.execute_iteration`, and `SubagentSpawnerToolKit.spawn_subagent` |
| `upsonic.tasks.tasks.Task` | Wraps prompt strings before `agent.do(...)` | Same locations |
| `upsonic.tools.ToolKit` / `upsonic.tools.tool` | Base class + decorator for the 5 toolkits | All `tools/*.py` files |

The `Agent.do(Task)` / `Agent.do_async(Task)` interface is **the** integration point — every Ralph "phase" is just a configured `Agent` invocation.

### Soft / architectural integration

- **Lazy imports everywhere.** All `__init__.py` files in this folder use module-level `__getattr__` so `import upsonic` does not transitively import `subprocess`, `signal`, or any LLM client. RalphLoop is opt-in.
- **Top-level re-export.** `RalphLoop`, `RalphConfig`, `RalphLoopResult` are re-exported from the package root so users can do `from upsonic import RalphLoop`.
- **No coupling to other Upsonic features.** Ralph does not use `Team`, `KnowledgeBase`, `RAG`, `Storage`, `Safety Engine`, or `Reliability Layer`. It is a self-contained sub-framework that uses only the bare `Agent` + `Task` + `ToolKit` primitives.
- **`@tool(sequential=True, timeout=600.0)`** on `run_backpressure` relies on the Upsonic tool runtime honoring the `sequential` flag for serialization — this is how the "build/test bottleneck" property is enforced at runtime.

### What it's NOT

- It is not a coding *assistant* — there is no chat surface, no UI, no streaming UX. It runs to completion (or `Ctrl-C`).
- It is not a *code-review* tool — there is no diff, no PR. It just writes files.
- It is not safety-checked — `RalphFilesystemToolKit.run_command` runs arbitrary shell commands and `BackpressureGate` does the same. **Run RalphLoop in a sandbox / disposable workspace.**

---

## 8. End-to-end flow

### Concrete walk-through

**Input** (from user):
```python
RalphLoop(
    goal="Build a Python CLI that fetches weather data from OpenWeatherMap",
    model="openai/gpt-4o",
    test_command="pytest",
    workspace="./weather_cli",
    max_iterations=15,
).run()
```

**Step 0 — Construction**:
- `RalphConfig.__post_init__`: workspace coerced to `Path("./weather_cli")`; goal validated.
- `StateManager`: `mkdir -p ./weather_cli/specs ./weather_cli/src`.
- `BackpressureGate`: stores `test_command="pytest --rootdir=. "` (auto-rewritten).
- `signal.SIGINT` handler installed for graceful stop.

**Step 1 — Phase 1: Requirements** (`_run_requirements_phase`):
- `state_manager.has_specs()` → False, no `config.specs` provided.
- Construct `Agent(model="openai/gpt-4o", system_prompt=REQUIREMENTS_PROMPT)`.
- `agent.do(Task("Analyze the goal: '...weather...' Generate ```spec:name blocks"))`.
- LLM emits something like:
  ```
  ```spec:cli
  # CLI
  ## Requirements
  - argparse-based command line
  - --city, --units flags
  ```
  ```spec:api_client
  # API Client
  - GET https://api.openweathermap.org/data/2.5/weather
  - Handle 401, 404, 429
  ```
  ```spec:cache
  # Cache
  - 10-minute TTL on disk
  ```
  ```
- `_parse_specs` extracts 3 specs → `state_manager.save_spec("cli", ...)` etc.
- `state_manager.save_prompt(DEFAULT_PROMPT_TEMPLATE.format(goal=...))` writes `weather_cli/PROMPT.md`.
- `result.specs_generated = ["cli", "api_client", "cache"]`.

**Step 2 — Phase 2: TODO** (`_run_todo_phase`):
- Construct `Agent(system_prompt=TODO_PROMPT, tools=[SubagentSpawnerToolKit])`.
- The agent might spawn a subagent to `list_files(".")` and discover the workspace is empty.
- LLM emits a ` ```todo ` block; `_parse_todo_list` normalizes:
  ```
  - [ ] Set up project with requirements.txt (requests, click)
  - [ ] Create src/api_client.py with WeatherClient.fetch(city)
  - [ ] Add unit tests for WeatherClient.fetch() with mocked responses
  - [ ] Create src/cache.py with TTLCache class
  - [ ] Add unit tests for TTLCache expiration logic
  - [ ] Create src/cli.py with argparse main() function
  - [ ] Add integration test for CLI end-to-end
  - [ ] Add error handling for 401/404/429 in api_client.py
  ```
- `state_manager.update_fix_plan(...)` writes `weather_cli/fix_plan.md`.

**Step 3 — Phase 3: Incremental loop** (per-iteration body):

#### Iteration 1
1. `state_manager.load_state()` → `RalphState` with 8 pending items, no learnings.
2. `pending_count_before = 8`, first item = `"Set up project with requirements.txt (requests, click)"`.
3. Build `system_prompt = PROMPT.md + SCHEDULER_PROMPT`, `context = format_for_context()`.
4. Construct `Agent` with all 5 toolkits.
5. `agent.do(Task(f"This is iteration 1. {context} ## Your Task: pick ONE [ ] item, implement, test, run backpressure, mark complete"))`.
6. Agent reasoning:
   - Calls `update_learnings` → no, not yet.
   - Calls `spawn_subagent(purpose="search", task="grep for requirements.txt")` → returns "No existing implementation found".
   - Calls `spawn_subagent(purpose="write", task="create requirements.txt with requests, click")` → subagent uses `RalphFilesystemToolKit.write_file("requirements.txt", "requests\nclick\n")`.
   - Calls `run_backpressure(validation_type="test")` → `BackpressureGate` runs `pytest --rootdir=. ` → exit code 5 ("collected 0 items") → `_is_no_tests_collected` returns True → `"PASS (no tests found - consider adding tests)"`.
   - Calls `update_fix_plan(action="complete", item="Set up project with requirements.txt")` → `StateManager.complete_fix_plan_item` flips `[ ]` to `[x]`.
   - Calls `update_learnings(learning="Initial pytest fails with exit-5; that's normal until tests exist", category="test")`.
7. After `agent.do` returns, `state_manager.load_state()` again → `pending_count_after = 7`.
8. `task_completed = (7 < 8) = True` → `IterationResult(success=True, plan_is_empty=False, ...)`.
9. `del agent` (free context).
10. `RalphLoop` adds an `IterationRecord` to `result.iterations`, fires `on_iteration` callback if set.

#### Iteration 2..N
Same structure, picking the next `[ ]` item. The `Agent` is reconstructed every time; the only continuity is what's on disk:

- `PROMPT.md` (constant)
- `specs/*.md` (constant unless agent calls `update_spec`)
- `fix_plan.md` (one more `[x]` per success)
- `AGENT.md` (grows monotonically)

The disk is the memory.

#### Iteration N+1 (terminal)
- `state_before.is_plan_empty()` → True (all items are `[x]`).
- Returns `IterationResult(success=True, plan_is_empty=True)`.
- `RalphLoop` sees `iter_result.plan_is_empty` → sets `final_status="completed"`, breaks.

**Step 4 — Wrap-up**:
- `signal.signal(SIGINT, original_handler)` — restore original.
- `result.end_time = datetime.now()`.
- If `show_progress`, prints `result.summary()`:
  ```
  ============================================================
  RalphLoop Execution Summary
  ============================================================
  Goal: Build a Python CLI that fetches weather data...
  Status: COMPLETED
  Workspace: weather_cli
  ------------------------------------------------------------
  Total Iterations: 9
  Successful: 8
  Failed: 1
  Success Rate: 88.9%
  Duration: 412.3s
  ------------------------------------------------------------
  Specs Generated: 3
    - cli
    - api_client
    - cache
  ============================================================
  ```
- Returns the `RalphLoopResult` to the caller.

### State machine summary

```
                  ┌─────────────────────────────┐
                  │  RalphLoop.run() entered    │
                  └──────────────┬──────────────┘
                                 │
                ┌────────────────▼─────────────────┐
                │  Phase 1: Requirements           │
                │  - skipped if specs/ exists      │
                │  - skipped if config.specs given │
                │  - else: LLM -> specs/*.md       │
                └────────────────┬─────────────────┘
                                 │
                ┌────────────────▼─────────────────┐
                │  Phase 2: TODO                   │
                │  - skipped if fix_plan.md exists │
                │  - else: LLM -> fix_plan.md      │
                └────────────────┬─────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────┐
                │  Phase 3: Incremental (loop)     │
                │                                  │
                │   ┌──── load_state ──────────┐   │
                │   │                          │   │
                │   │  is_plan_empty? ────► YES│──► final_status="completed"
                │   │     │ NO                 │   │
                │   │     ▼                    │   │
                │   │  Build fresh Agent       │   │
                │   │  agent.do(Task)          │   │
                │   │     uses tools:          │   │
                │   │     - spawn_subagent     │   │
                │   │     - read/write/edit    │   │
                │   │     - run_backpressure   │   │
                │   │     - update_fix_plan    │   │
                │   │     - update_learnings   │   │
                │   │  del agent               │   │
                │   │                          │   │
                │   │  re-load_state           │   │
                │   │  pending decreased? ── ► success/failure │
                │   │                          │   │
                │   │  on_iteration callback   │   │
                │   │                          │   │
                │   │  iteration > max? ──► YES│──► final_status="max_iterations"
                │   │  SIGINT received? ──► YES│──► final_status="stopped"
                │   └──────────────────────────┘   │
                └────────────────┬─────────────────┘
                                 │
                                 ▼
                          return RalphLoopResult
```

### Why this works (the Ralph principle)

The whole system survives on three properties:

1. **Idempotent state** — every state file can be re-read, every phase can be re-run. Killing and restarting always converges.
2. **Disposable agents** — context window pollution is eliminated by destroying the `Agent` after each iteration. Cost is paid in tokens, not in correctness.
3. **Truth = backpressure exit code 0** — there is exactly one source of truth for "is this work done": the `subprocess` returncode of the test/build command. Everything else (agent claims, LLM rationalizations) is just commentary.

The `ralph` folder is a tight, self-contained encoding of those three properties on top of Upsonic's `Agent` + `Task` + `ToolKit` primitives.
