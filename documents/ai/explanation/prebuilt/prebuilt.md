---
name: prebuilt-autonomous-agents
description: Use when working with the `src/upsonic/prebuilt/` package — ready-to-run autonomous agents whose system prompt, first-message template, and Anthropic-style skills are bundled as repository files and materialized into a workspace via sparse git clone at run time. Use when a user asks to add a new prebuilt, subclass `PrebuiltAutonomousAgentBase`, debug template/first_message placeholder errors, configure `AGENT_REPO`/`AGENT_FOLDER`, or use `AppliedScientist.new_experiment(...)` and the `Experiment` / `ExperimentResult` / `ExperimentRecord` / `ExperimentRegistry` lifecycle (run, run_async, run_stream, run_console, run_in_background, stop, progress_bar). Trigger when the user mentions prebuilt, prebuilt agent, PrebuiltAutonomousAgentBase, AppliedScientist, applied scientist, Experiment, ExperimentResult, ExperimentRecord, ExperimentRegistry, new_experiment, agent_repo, agent_folder, AGENT_REPO, AGENT_FOLDER, template/system_prompt.md, first_message.md, SKILL.md, skills/, sparse clone, _bootstrap, _clone_repo_folder, _copy_inputs, _render_first_message, run_console, progress_bar_live, experiments.json, progress.json, result.json, verdict, comparison, six-phase pipeline, applied_scientist, or new_prebuilt_agent_adding.
---

# `src/upsonic/prebuilt/` — Prebuilt Autonomous Agents

This document is a deep-dive reference for the `prebuilt` package: a folder of
ready-to-use autonomous agents whose system prompt, first-message template,
and Anthropic-style skills are bundled as repository files and pulled into a
user workspace at run time.

---

## 1. What this folder is — prebuilt autonomous agents (ready-to-use)

`src/upsonic/prebuilt/` ships **fully-wired autonomous agents** that a user can
import and run with two arguments (`model`, `workspace`) plus a small,
template-aware payload. Each prebuilt is the combination of:

1. **A Python class** in `<agent>/agent.py` that subclasses
   `PrebuiltAutonomousAgentBase` and pins two constants:
   - `AGENT_REPO` — the git URL to clone (canonical: the Upsonic repo).
   - `AGENT_FOLDER` — the path within that repo to the agent's `template/`.
2. **A `template/` directory** that contains:
   - `system_prompt.md` — the verbatim system prompt for the underlying
     `AutonomousAgent`. Wrapped at run time by the autonomous-agent harness
     with workspace, filesystem, and shell instructions.
   - `first_message.md` — the very first user-style message, written as a
     Python `str.format` template (`{name}`, `{research_source}`, …). Every
     placeholder must be supplied by the caller as a keyword argument.
   - `skills/<skill_name>/SKILL.md` — Anthropic-style skill files. The base
     class copies the entire `skills/` tree into the workspace untouched, so
     any relative reference from `system_prompt.md` resolves on disk.
3. **An optional high-level API** on the agent class (e.g.
   `AppliedScientist.new_experiment(...)`) that hides the placeholder names
   in `first_message.md` behind a typed factory method.

The defining design choice: the **template is data**, not code. It lives in
the public Upsonic repository and is fetched fresh on every run via
`git clone --depth 1 --filter=blob:none --sparse`. There is no caching layer
to bust — fix the template on `master`, every subsequent `agent.run(...)`
sees the change immediately.

> Practical consequence: when you develop a new prebuilt locally, the
> `template/` files are not loaded from your working copy unless your branch
> is the one pulled by the run. Use `agent_repo="https://github.com/<fork>/Upsonic"`
> at construction to point the run at a fork while iterating.

The companion contributor guide
[`docs/new_prebuilt_agent_adding.md`](../../new_prebuilt_agent_adding.md)
(referenced from `CLAUDE.md` as `Docs/ai/new_prebuilt_agent_adding.md`) is
the canonical step-by-step for shipping a new prebuilt — Section 8 of this
document summarizes the rules.

---

## 2. Folder layout (tree showing each prebuilt agent)

```
src/upsonic/prebuilt/
├── __init__.py                        # Lazy re-exports (PrebuiltAutonomousAgentBase,
│                                      #   AppliedScientist, Experiment, ExperimentResult)
├── prebuilt_agent_base.py             # Shared base class for all prebuilts
└── applied_scientist/                 # The first prebuilt — applied scientist agent
    ├── __init__.py                    # Lazy re-export of agent.py classes
    ├── agent.py                       # AppliedScientist + Experiment + ExperimentRecord
    │                                  #   + ExperimentRegistry + ExperimentResult
    └── template/                      # Cloned into user workspace on every run
        ├── system_prompt.md           # 6-phase research-experiment system prompt
        ├── first_message.md           # str.format template — 5 placeholders
        ├── example.sh                 # Reference invocation (claude CLI form)
        └── skills/                    # Anthropic-style skills, one folder per skill
            ├── analyze_current/SKILL.md
            ├── benchmark/SKILL.md
            ├── evaluate/SKILL.md
            ├── experiment_management/SKILL.md
            ├── implement/SKILL.md
            ├── progress/SKILL.md
            └── research/SKILL.md
```

| File | Purpose |
| --- | --- |
| `prebuilt_agent_base.py` | Shared base class. Adds template-fetch + render on top of `AutonomousAgent`. |
| `__init__.py` (top) | Lazy-imports concrete agent classes so `import upsonic` stays cheap. |
| `applied_scientist/agent.py` | Concrete prebuilt + `Experiment` / `ExperimentRecord` / `ExperimentRegistry` / `ExperimentResult`. |
| `applied_scientist/template/system_prompt.md` | The agent's persona, rules, phases, file schemas. |
| `applied_scientist/template/first_message.md` | First user-turn template; placeholders fill from `new_experiment(...)`. |
| `applied_scientist/template/skills/*/SKILL.md` | Per-phase skill specs the system prompt references by relative path. |
| `applied_scientist/template/example.sh` | Out-of-the-box reference invocation (claude CLI). |

---

## 3. Shared base class (`prebuilt_agent_base.py`)

The single concrete class in this file —
`PrebuiltAutonomousAgentBase(AutonomousAgent)` — is the lifeline of every
prebuilt. It inherits the autonomous-agent runtime (filesystem + shell
toolkits, sandboxed workspace, multi-turn loop) and adds *template
materialization* on top of it.

### 3.1 Constructor

```python
class PrebuiltAutonomousAgentBase(AutonomousAgent):
    def __init__(
        self,
        *args: Any,
        agent_repo: Optional[str] = None,
        agent_folder: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.agent_repo: Optional[str] = agent_repo
        self.agent_folder: Optional[str] = agent_folder
        self._first_message_template: Optional[str] = None
        self._repo_system_prompt: Optional[str] = None
```

- `agent_repo` — the git URL to clone. Subclasses pin this to
  `https://github.com/Upsonic/Upsonic` via `AGENT_REPO`.
- `agent_folder` — the path inside that repo to the prebuilt's `template/`
  directory. Subclasses pin this via `AGENT_FOLDER`.
- `_first_message_template` and `_repo_system_prompt` are populated during
  `_bootstrap()` for every run.
- All other `*args` / `**kwargs` flow into `AutonomousAgent.__init__`, so
  every constructor knob of the autonomous agent (model, debug, telemetry,
  caching, etc.) is available on the prebuilt too.

### 3.2 The bootstrap pipeline

Every entry point (`run`, `run_async`, `run_stream`, `run_stream_async`,
`run_console`) routes through `_bootstrap(...)`:

```
┌────────────────┐
│ workspace=...  │
└──────┬─────────┘
       │ _apply_workspace(ws)
       │   ─ remove old fs/shell toolkits
       │   ─ register new AutonomousFilesystemToolKit(workspace=ws)
       │   ─ register new AutonomousShellToolKit(workspace=ws)
       ▼
┌──────────────────────────┐
│ git clone (sparse) repo  │
│ copy template into ws    │
└──────┬───────────────────┘
       │ _clone_repo_folder(repo_url, folder, dest, verbose)
       │   ─ tries: git clone --depth 1 --filter=blob:none --sparse
       │   ─ fallback: full shallow clone if sparse-checkout fails
       │   ─ raises RuntimeError if `git` is not on PATH
       ▼
┌──────────────────────────┐
│ copy user inputs into ws │
└──────┬───────────────────┘
       │ _copy_inputs(inputs, ws, verbose)
       │   ─ refuses paths that escape the workspace sandbox
       │   ─ relative paths preserve their layout under workspace root
       │   ─ absolute paths land at workspace root by basename
       ▼
┌──────────────────────────┐
│ load prompts from disk   │
└──────┬───────────────────┘
       │ _load_repo_files(ws, verbose)
       │   ─ system_prompt.md → wrap with autonomous-agent harness
       │     via self._build_autonomous_system_prompt(...)
       │   ─ first_message.md → store as _first_message_template
       ▼
┌──────────────────────────┐
│ render first message     │
└──────┬───────────────────┘
       │ _render_first_message(template, params)
       │   ─ extracts {placeholders} via string.Formatter().parse(...)
       │   ─ raises ValueError listing every missing placeholder
       ▼
┌──────────────────────────┐
│ Task(description=...)    │
└──────────────────────────┘
```

Key implementation details:

| Helper | Behaviour |
| --- | --- |
| `_apply_workspace(workspace)` | Resolves the path, creates it, removes the previous filesystem/shell toolkits via `Agent.remove_tools(...)`, registers fresh `AutonomousFilesystemToolKit` + `AutonomousShellToolKit` instances, resets `_workspace_greeting_executed`, re-reads `AGENTS.md` if present. |
| `_clone_repo_folder(repo_url, folder, destination, verbose)` | Validates `git` is on PATH; clones into a `tempfile.TemporaryDirectory(prefix="upsonic_prebuilt_repo_")`; if sparse-checkout fails, retries with a full shallow clone; copies the contents of the requested subfolder into the workspace, replacing existing files/directories. |
| `_copy_inputs(inputs, destination, verbose)` | For every input path: resolves `~`, rejects non-existent paths with `FileNotFoundError`, computes a relative destination, ensures `dest.relative_to(dest_root)` succeeds (sandbox check), wipes any pre-existing target, then `shutil.copytree` / `shutil.copy2`. |
| `_load_repo_files(workspace, verbose)` | Reads `system_prompt.md` and feeds it to `self._build_autonomous_system_prompt(...)` so the autonomous-agent harness wraps it with workspace/fs/shell instructions. Reads `first_message.md` into `self._first_message_template`. |
| `_extract_template_fields(template)` | Pure helper using `string.Formatter().parse(...)` to enumerate the named fields in the template. |
| `_render_first_message(template, params)` | Renders, raising `ValueError` listing every missing key (so callers see exactly which kwargs to add). |
| `_bootstrap(workspace, inputs, verbose, template_params)` | Orchestrates the whole pipeline and returns `Task(description=rendered_first_message)`. |
| `_log(verbose, message)` | Prints `"[ClassName] message"` only when `verbose=True`. |

### 3.3 Public API

| Method | Sync/Async | What it does |
| --- | --- | --- |
| `run(*, workspace, inputs, verbose, return_output, timeout, partial_on_timeout, **template_params)` | sync | Calls `run_async` via `_run_in_bg_loop`. Returns the agent's final result. |
| `run_async(...)` | async | `await self.do_async(task, ...)`. Temporarily flips `print` / `show_tool_calls` / `debug` when `verbose=True`. |
| `run_stream(*, workspace, inputs, verbose, events, **template_params)` | sync iterator | Bridges async streaming to a sync iterator via a background thread + `queue.Queue`. Yields text deltas (or `AgentStreamEvent` if `events=True`). |
| `run_stream_async(...)` | async iterator | Calls `self.astream(task, events=events)` directly. |
| `run_console(*, workspace, inputs, verbose, preview_chars, **template_params)` | sync | Pretty TTY rendering using `rich.console.Console` + `rich.panel.Panel`. Shows assistant text, tool calls, tool results, errors, run-completed/cancelled events live. Returns the concatenated assistant text. |

The `run_console` event handling is worth highlighting because it's the path
most users hit:

```python
for event in self.stream(task, events=True):
    if isinstance(event, TextDeltaEvent):       # streaming assistant tokens
        ...
    elif isinstance(event, ToolCallEvent):      # 🔧 yellow tool name + dim args
        ...
    elif isinstance(event, ToolResultEvent):    # ✓ green or ✗ red, with timing
        ...
    elif isinstance(event, RunCancelledEvent):  # ⚠ red banner
        ...
    elif isinstance(event, RunCompletedEvent):  # close any open text block
        ...
```

`preview_chars` (default `400`) caps both tool-call argument previews and
tool-result previews so a single huge payload doesn't drown the terminal.

### 3.4 Failure modes

| Raises | When |
| --- | --- |
| `ValueError` | `agent_repo` set without `agent_folder`; missing required `first_message.md` placeholder; no workspace at any layer. |
| `FileNotFoundError` | `agent_folder` is missing inside the cloned repo; user-supplied `inputs` path does not exist; required `first_message.md` is missing from the template. |
| `RuntimeError` | `git` not on PATH; full shallow clone failed (with original `stderr`). |

---

## 4. Per-prebuilt subfolder walkthrough

There is currently exactly one prebuilt: **`AppliedScientist`**. The
following walkthrough mirrors the layout each new prebuilt should follow.

### 4.1 `applied_scientist/agent.py`

`agent.py` is intentionally self-contained. Reading just this file gives a
complete picture of the agent class, its supporting result/record classes,
and nothing else.

| Class | Role |
| --- | --- |
| `AppliedScientist` | The prebuilt. `AGENT_REPO = "https://github.com/Upsonic/Upsonic"`, `AGENT_FOLDER = "src/upsonic/prebuilt/applied_scientist/template"`. Default model `openai/gpt-4o`. Exposes `new_experiment(...)`, `experiments`, `progress_bar_live(...)`, `list_experiments(...)`. |
| `Experiment` | Prepared-but-not-yet-running experiment. Holds `template_params` + `inputs`. Methods: `run` (TTY), `run_async`, `run_stream`, `run_in_background`, `wait`, `stop`. Properties: `is_running`, `is_done`, `record`, `result`, `output`, `error`, `progress_bar`, `last_logs(n)`, `stop_requested`. |
| `ExperimentResult` | Structured view over `result.json`. Properties: `verdict`, `summary`, `explanation`, `table`, `file_locations`, plus `to_dict()` and a `_repr_html_` for Jupyter. |
| `ExperimentRecord` | Read-only view of an experiment on disk. Re-reads `progress.json`, `log.json`, `result.json` on every property access. Normalises phase status aliases (`completed`/`finished`/`success` → `done`, etc.). |
| `ExperimentRegistry` | Dict-like view over `experiments.json`. Re-reads on every operation. Implements `__iter__`, `__len__`, `__contains__`, `__getitem__`, `get`, `keys`, `values`, `items`. |

The `_auto_inputs(...)` module-level helper derives the sandbox `inputs`
list from the free-form user arguments to `new_experiment`. The rule is:
*if the value happens to point at an existing local file or directory,
keep it; otherwise drop it silently and let the agent fetch it itself*.
URLs, git/Kaggle references, arXiv IDs, and free-text method ideas all
fall through.

#### `AppliedScientist`

```python
class AppliedScientist(PrebuiltAutonomousAgentBase):
    AGENT_REPO: str = "https://github.com/Upsonic/Upsonic"
    AGENT_FOLDER: str = "src/upsonic/prebuilt/applied_scientist/template"

    def __init__(
        self,
        *,
        model: Union[str, "Model"] = "openai/gpt-4o",
        workspace: Optional[str] = None,
        experiments_directory: str = "./experiments",
        **kwargs: Any,
    ) -> None:
        kwargs.pop("agent_repo", None)
        kwargs.pop("agent_folder", None)
        super().__init__(
            model=model,
            workspace=workspace,
            agent_repo=self.AGENT_REPO,
            agent_folder=self.AGENT_FOLDER,
            **kwargs,
        )
        self._experiments_directory: str = experiments_directory
```

The defining method is `new_experiment(...)`:

```python
exp = scientist.new_experiment(
    name="tabpfn_adult",                         # used verbatim by the agent
    research_source="example_1/tabpfn.pdf",      # local file, URL, git/Kaggle, idea, …
    current_notebook="example_1/baseline.ipynb",
    current_data=None,                           # None → infer from notebook
    experiments_directory=None,                  # defaults to ./experiments
    inputs=None,                                 # None → auto-derived (see below)
)
exp.run()                                        # pretty TTY output
```

Inputs auto-derivation: when `inputs=None`, `_auto_inputs(research_source,
current_notebook, current_data or "")` is called. Each value that resolves
to an existing local file or directory becomes an entry in the workspace
sandbox. Values like `"https://arxiv.org/abs/2207.01848"` or
`"swap XGBoost for CatBoost"` are dropped — the agent retrieves or
interprets them at Phase 0.

#### `Experiment` lifecycle

```
new_experiment()
       │
       ▼
   pending  ──── run() ─────────────► running (foreground TTY)
       │  ──── run_async() ────────► running (async)
       │  ──── run_stream() ───────► running (sync iterator)
       │  ──── run_in_background() ► running (thread; quiet)
       │
       ▼
   running  ──── stop() ───────────► cancellation requested
       │
       ▼
    done   ──── result / output ────► ExperimentResult | raw output
            ──── error ──────────────► raised exception (if any)
```

Background runs silence `print` / `show_tool_calls` / `debug` for the
duration of the run so a Jupyter cell does not flood with output. Polling:

```python
exp.run_in_background()
while exp.is_running:
    display(exp.progress_bar)        # IPython HTML; reads progress.json
    time.sleep(5)
print(exp.result.verdict)
```

`Experiment.stop(wait_for_run_id=5.0)` uses
`upsonic.run.cancel.cancel_run(run_id)` to signal cooperative cancellation;
it does not kill the thread. The agent raises at its next pipeline
checkpoint.

#### `ExperimentResult`

`ExperimentResult` is the structured projection of `result.json`:

| Property | Returns |
| --- | --- |
| `verdict` | `"BETTER"` / `"WORSE"` / `"INCONCLUSIVE"` / `"FAILED"` |
| `summary` | 2-3 paragraph prose description of the new method |
| `explanation` | 2-3 sentence reason for the verdict, with concrete numbers |
| `table` | List of metric dicts (`name`, `current`, `new`, `diff`, `diff_display`, `unit`, `higher_is_better`, `better`) |
| `file_locations` | Mapping from canonical names to paths inside the experiment folder |
| `to_dict()` | The full `result.json` payload as a plain dict |

It also defines `_repr_html_` so a Jupyter cell that ends with `exp.result`
renders a colour-coded verdict badge plus a metric comparison table.

#### `ExperimentRecord` / `ExperimentRegistry`

`ExperimentRegistry` is the dict-like, always-fresh view over
`{workspace}/{experiments_directory}/experiments.json`. It is exposed via
`AppliedScientist.experiments` and is the recommended way to retrieve past
runs.

```python
scientist.experiments                       # ExperimentRegistry(...)
scientist.experiments["tabpfn_adult"]       # ExperimentRecord(...)
scientist.experiments["tabpfn_adult"].status     # "completed"
scientist.experiments["tabpfn_adult"].verdict    # "BETTER"
scientist.experiments["tabpfn_adult"].summary    # from result.json
scientist.experiments["tabpfn_adult"].comparison # list[dict]
scientist.experiments["tabpfn_adult"].phases     # normalised list
```

Every property (`progress`, `log`, `result`, derived views) re-reads the
on-disk file each time, so the same record reflects mid-run state without
any explicit refresh.

`ExperimentRecord.phases` accepts both the documented list shape and a
dict-keyed variant; status synonyms (`completed`, `complete`, `finished`,
`success`, `ok`, `running`, `in_progress`, `in-progress`, `active`, `todo`,
`waiting`, `queued`, `error`, `errored`) are normalised to the fixed
vocabulary `{"done", "current", "pending", "failed"}` so UI code can rely
on a stable enum.

### 4.2 `applied_scientist/template/system_prompt.md`

Defines the agent's persona, the six-phase pipeline, the file structure it
produces, and every JSON schema it has to emit. Highlights:

| Phase | Skill | Goal |
| --- | --- | --- |
| 0 | `experiment_management` | Create an isolated experiment folder, copy notebook + data, materialize `research_source` into the folder, init `log.json`, `progress.json`, register in `experiments.json`. |
| 1 | `analyze_current` | Read the baseline notebook, extract model / preprocessing / hyperparameters / metrics / data shape, write `current_requirements.txt`, append a Phase 1 entry to `log.json`. |
| 2 | `research` | Read the materialized research source (PDF, URL, git repo, Kaggle, HF, idea), extract method summary / pros / cons / requirements / compatibility, append a Phase 2 entry. |
| 3 | `benchmark` | Define the comparison metrics list, extract baseline values from `current.ipynb`, append a Phase 3 entry. |
| 4 | `implement` | Write `new_requirements.txt`, create `new.ipynb` with the prescribed 7-section structure, run it end-to-end, append a Phase 4 entry. |
| 5 | `evaluate` | Build the comparison table, choose a verdict, write `result.json`, update `experiments.json`, append a row to `comparison.json`, append a Phase 5 entry. |

Critical rules baked into the prompt:

1. Never modify original files — only the copies inside
   `experiments/{research_name}/`.
2. Follow the phases in order. No skipping.
3. Log everything as JSON (never markdown).
4. Same data, same split, same seed — fair comparison only.
5. If the notebook fails, write `result.json` with `verdict="FAILED"` —
   honest failure beats fake success.
6. Run autonomously — no per-phase confirmation prompts.
7. Update `progress.json` *before* every long operation.
8. All bookkeeping files must be valid JSON.
9. Use `research_name` verbatim in folders and JSON `"name"` fields.

### 4.3 `applied_scientist/template/first_message.md`

```markdown
New experiment.

**Experiment name:** {research_name}
**Research source:** {research_source}
**Current notebook:** {current_notebook}
**Current data:** {current_data}
**Experiments directory:** {experiments_directory}

The research source describes the new method to evaluate. ...
Use `{research_name}` **exactly as given** ...
Run the full experiment pipeline. Go from Phase 0 through Phase 5 without stopping. ...

Start now.
```

The five `str.format` placeholders correspond exactly to the keys built by
`AppliedScientist.new_experiment(...)`:

```python
template_params = {
    "research_name":         name,
    "research_source":       research_source,
    "current_notebook":      current_notebook,
    "current_data":          current_data or "(not provided — infer it from the current notebook's data-loading cells)",
    "experiments_directory": exp_dir,
}
```

Forgetting any of them surfaces as:

```
ValueError: run() is missing required template parameter(s) for first_message.md:
research_source. Pass them as keyword arguments.
```

### 4.4 `applied_scientist/template/skills/`

Each subfolder holds one Anthropic-style `SKILL.md`. The system prompt
references them by relative path (e.g. `skills/evaluate/SKILL.md`), and the
base class copies the entire `skills/` tree into the workspace untouched
so the agent can read those references on disk.

| Skill | Phase | One-line role |
| --- | --- | --- |
| `experiment_management/SKILL.md` | 0 | Create the experiment folder, copy baseline files, materialize the research source (PDF / git / Kaggle / HF / URL / text idea), seed `log.json` + `progress.json` + `experiments.json`. |
| `analyze_current/SKILL.md` | 1 | Read `current.ipynb`, extract model + preprocessing + hyperparameters + metrics + data shape, write `current_requirements.txt`, append Phase 1 entry. |
| `research/SKILL.md` | 2 | Read the materialized research source and extract method summary / pros / cons / requirements / compatibility. Handles ideas (no fabricated citations) the same as papers. |
| `benchmark/SKILL.md` | 3 | Define the comparison metric list, extract baseline values, mark anything missing with `needs_computation: true`. |
| `implement/SKILL.md` | 4 | Build `new.ipynb` (fixed 7-section layout), enforce same data / same split / same seed, install new dependencies, run end-to-end, capture metrics. |
| `evaluate/SKILL.md` | 5 | Build the metric comparison table (`diff`, `diff_display`, `better`), choose a verdict, write `result.json`, update `experiments.json` and `comparison.json`. |
| `progress/SKILL.md` | n/a (cross-cutting) | Strict canonical schema for `progress.json`. Six phases, fixed status vocabulary, overwrite-not-append, refresh `updated_at` on every change. |

The `progress/SKILL.md` schema is the contract `Experiment.progress_bar`
and `ExperimentRecord.phases` rely on, which is why
`ExperimentRecord.phases` only normalises *aliases* — the canonical values
(`done`/`current`/`pending`/`failed`) are exactly what `SKILL.md` mandates.

### 4.5 `AppliedScientist` quick-reference table

| Concern | Who handles it |
| --- | --- |
| Role/goal | Prebuilt applied-scientist that runs the full Phase 0→5 experiment pipeline and writes a machine-readable verdict. |
| Skills | `experiment_management`, `analyze_current`, `research`, `benchmark`, `implement`, `evaluate`, `progress`. |
| Inputs | `research_name`, `research_source` (free-form), `current_notebook`, `current_data` (optional), `experiments_directory` (optional), `inputs` (optional override). |
| Outputs | `experiments/{name}/result.json` (final verdict + metric table), plus `experiments.json` registry update and `comparison.json` row. |
| Cancellation | `Experiment.stop()` → cooperative `cancel_run(run_id)`. |
| Jupyter integration | `Experiment.progress_bar`, `Experiment.last_logs(n)`, `ExperimentResult._repr_html_`, `AppliedScientist.progress_bar_live(experiment, interval=5.0)`. |

---

## 5. Cross-file relationships and how prebuilts compose with `AutonomousAgent`

The class graph (top → bottom is "is-a"):

```
              upsonic.agent.agent.Agent
                       ▲
                       │
              upsonic.agent.autonomous_agent.AutonomousAgent
              (workspace, AutonomousFilesystemToolKit,
               AutonomousShellToolKit, _build_autonomous_system_prompt,
               _read_workspace_agents_md, _workspace_greeting_executed)
                       ▲
                       │
              PrebuiltAutonomousAgentBase
              (agent_repo, agent_folder, _bootstrap, _clone_repo_folder,
               _copy_inputs, _load_repo_files, _render_first_message,
               run, run_async, run_stream, run_stream_async, run_console)
                       ▲
                       │
              AppliedScientist
              (AGENT_REPO, AGENT_FOLDER, new_experiment(...),
               experiments, progress_bar_live(...), list_experiments(...))
```

What each layer adds:

| Layer | Adds |
| --- | --- |
| `Agent` | LLM call loop, tool registry, streaming events, retry + telemetry. |
| `AutonomousAgent` | Sandboxed `workspace`, filesystem + shell toolkits, autonomous system-prompt wrapping, `AGENTS.md` workspace greeting. |
| `PrebuiltAutonomousAgentBase` | Template materialization (git clone), input copy with sandbox enforcement, first-message rendering with placeholder validation, sync/async/stream/console entry points. |
| `AppliedScientist` | Pinned repo + folder, default `openai/gpt-4o`, `Experiment` + `ExperimentRegistry` + `ExperimentResult`, Jupyter helpers. |

Compositional contract:

```python
# This is what the bootstrap pipeline effectively does on every run:
self._apply_workspace(workspace)                # AutonomousAgent: fs + shell toolkits
self._clone_repo_folder(self.agent_repo,        # PrebuiltAutonomousAgentBase
                        self.agent_folder,
                        destination=self.autonomous_workspace)
self._copy_inputs(inputs, self.autonomous_workspace)
self._load_repo_files(self.autonomous_workspace)  # populates _repo_system_prompt and
                                                  # _first_message_template
task = Task(description=self._render_first_message(
    self._first_message_template, template_params))
return await self.do_async(task, ...)            # Agent: standard run loop
```

The autonomous-agent harness (`_build_autonomous_system_prompt`) wraps the
template's `system_prompt.md` with workspace + filesystem + shell
instructions before the model ever sees it, so a prebuilt template can
remain agnostic to those concerns.

Cooperative cancellation:

```python
Experiment.stop()
   │
   ▼
upsonic.run.cancel.cancel_run(self._agent.run_id)
   │
   ▼
AutonomousAgent next pipeline checkpoint raises → Experiment._error set
```

Streaming:

| Entry point | Underlying call |
| --- | --- |
| `run_stream_async(...)` | `self.astream(task, events=events)` |
| `run_stream(...)` | spawns a daemon thread → `_get_bg_loop()` → `astream(...)` → `queue.Queue` for sync iteration |
| `run_console(...)` | `self.stream(task, events=True)` consumed event-by-event for the rich TTY layout |

---

## 6. Public API (which prebuilts users can import)

`src/upsonic/prebuilt/__init__.py` uses module-level `__getattr__` so
imports stay lazy:

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .prebuilt_agent_base import PrebuiltAutonomousAgentBase
    from .applied_scientist.agent import (
        AppliedScientist, Experiment, ExperimentResult,
    )


def _get_classes() -> dict[str, Any]:
    from .prebuilt_agent_base import PrebuiltAutonomousAgentBase
    from .applied_scientist.agent import (
        AppliedScientist, Experiment, ExperimentResult,
    )
    return {
        "PrebuiltAutonomousAgentBase": PrebuiltAutonomousAgentBase,
        "AppliedScientist": AppliedScientist,
        "Experiment": Experiment,
        "ExperimentResult": ExperimentResult,
    }


def __getattr__(name: str) -> Any:
    classes = _get_classes()
    if name in classes:
        return classes[name]
    raise AttributeError(...)


__all__ = [
    "PrebuiltAutonomousAgentBase",
    "AppliedScientist",
    "Experiment",
    "ExperimentResult",
]
```

The package-level public API:

| Symbol | Origin | Use case |
| --- | --- | --- |
| `PrebuiltAutonomousAgentBase` | `prebuilt_agent_base.py` | Direct, untyped use of the base class with arbitrary `agent_repo` / `agent_folder` (advanced — most users prefer a concrete subclass). |
| `AppliedScientist` | `applied_scientist/agent.py` | The prebuilt applied-scientist agent. |
| `Experiment` | `applied_scientist/agent.py` | The deferred-run object returned by `AppliedScientist.new_experiment(...)`. |
| `ExperimentResult` | `applied_scientist/agent.py` | Structured result wrapper (verdict + metric table). |

The `applied_scientist` subpackage exposes a slightly richer surface for
power users:

```python
from upsonic.prebuilt.applied_scientist import (
    AppliedScientist,
    Experiment,
    ExperimentRecord,
    ExperimentRegistry,
    ExperimentResult,
)
```

`ExperimentRecord` and `ExperimentRegistry` are not promoted to the top
level because the recommended access pattern is via
`scientist.experiments[name]`.

### 6.1 Direct base-class usage

```python
from upsonic.prebuilt import PrebuiltAutonomousAgentBase

agent = PrebuiltAutonomousAgentBase(
    model="anthropic/claude-sonnet-4-5",
    agent_repo="https://github.com/Upsonic/Upsonic",
    agent_folder="src/upsonic/prebuilt/applied_scientist/template",
)

agent.run_console(
    workspace="./ws",
    inputs=["example_1/"],
    research_name="tabpfn_adult",
    research_source="example_1/paper.pdf",
    current_notebook="example_1/baseline.ipynb",
    current_data="(not provided — infer it from the current notebook's data-loading cells)",
    experiments_directory="./experiments",
)
```

This works because `PrebuiltAutonomousAgentBase` only requires the generic
five-placeholder `first_message.md`.

### 6.2 High-level usage (recommended)

```python
from upsonic.prebuilt import AppliedScientist

scientist = AppliedScientist(model="openai/gpt-4o", workspace="./ws")

exp = scientist.new_experiment(
    name="tabpfn_adult",
    research_source="example_1/tabpfn.pdf",
    current_notebook="example_1/baseline.ipynb",
)
exp.run()                                           # pretty TTY

# Or in Jupyter:
exp = scientist.new_experiment(name="catboost_adult",
                               research_source="swap XGBoost for CatBoost",
                               current_notebook="example_1/baseline.ipynb")
exp.run_in_background()
scientist.progress_bar_live(exp, interval=5.0)
print(exp.result.verdict)
```

---

## 7. Integration with the rest of Upsonic

| Upsonic surface | How prebuilts plug in |
| --- | --- |
| `upsonic.agent.agent.Agent` | All prebuilts are `Agent` subclasses by transitivity. They participate in `Agent.cost`, `Agent.run_id`, telemetry, retries, and tool registration. |
| `upsonic.agent.autonomous_agent.AutonomousAgent` | Provides the workspace, `AutonomousFilesystemToolKit`, `AutonomousShellToolKit`, `_build_autonomous_system_prompt`. Re-applied on every run by `_apply_workspace`. |
| `upsonic.agent.autonomous_agent.filesystem_toolkit.AutonomousFilesystemToolKit` | Sandboxed file read/write. Re-instantiated on workspace change. |
| `upsonic.agent.autonomous_agent.shell_toolkit.AutonomousShellToolKit` | Sandboxed shell execution. Re-instantiated on workspace change. |
| `upsonic.tasks.tasks.Task` | The first-message template renders into `Task(description=...)`, which is what every entry point hands to `do_async` / `astream`. |
| `upsonic.run.events.events` (`TextDeltaEvent`, `ToolCallEvent`, `ToolResultEvent`, `RunCompletedEvent`, `RunCancelledEvent`) | Consumed event-by-event in `run_console` to drive the rich TTY layout. |
| `upsonic.run.cancel.cancel_run(run_id)` | Backs `Experiment.stop()` for cooperative cancellation of in-flight prebuilt runs. |
| `upsonic.agent.agent._run_in_bg_loop` / `_get_bg_loop` | Used by the sync wrappers (`run`, `run_stream`, `Experiment.run_in_background`) to bridge async coroutines from plain Python / Jupyter cells. |
| `upsonic.models.Model` | Optional explicit model object for the `model=` argument on `AppliedScientist`. |
| `rich.console.Console` / `rich.panel.Panel` | TTY rendering inside `run_console`. |
| `IPython.display.HTML`, `IPython.display.clear_output`, `IPython.display.display` | Jupyter integration in `Experiment.progress_bar`, `Experiment.last_logs`, `AppliedScientist.progress_bar_live`. |

Because prebuilts are still ordinary `Agent`s, any cost/telemetry feature
that works on a plain `Agent` (e.g. `agent.cost` aggregation across tasks
introduced in commit `eec836c1`) works on prebuilts unchanged.

---

## 8. How to add a new prebuilt

The full canonical guide is
[`docs/new_prebuilt_agent_adding.md`](../../new_prebuilt_agent_adding.md)
(referenced by `CLAUDE.md` as `Docs/ai/new_prebuilt_agent_adding.md`).
Quick summary:

### 8.1 File layout

```
src/upsonic/prebuilt/
└── <your_agent>/
    ├── __init__.py                  # lazy re-export of agent.py
    ├── agent.py                     # YourAgent + helper classes
    └── template/
        ├── system_prompt.md         # required
        ├── first_message.md         # required, str.format placeholders
        └── skills/
            ├── <skill_a>/SKILL.md
            └── ...
```

### 8.2 Subclass the base

```python
from upsonic.prebuilt.prebuilt_agent_base import PrebuiltAutonomousAgentBase

class YourAgent(PrebuiltAutonomousAgentBase):
    AGENT_REPO: str = "https://github.com/Upsonic/Upsonic"
    AGENT_FOLDER: str = "src/upsonic/prebuilt/<your_agent>/template"

    def __init__(
        self,
        *,
        model: Union[str, "Model"] = "openai/gpt-4o",
        workspace: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("agent_repo", None)
        kwargs.pop("agent_folder", None)
        super().__init__(
            model=model,
            workspace=workspace,
            agent_repo=self.AGENT_REPO,
            agent_folder=self.AGENT_FOLDER,
            **kwargs,
        )
```

That alone is enough to call
`agent.run(workspace="./ws", **template_params)`.

### 8.3 (Optional) High-level template-aware API

Wrap the placeholder names in a typed factory method so users do not have
to read `first_message.md`. See `AppliedScientist.new_experiment` and the
`Experiment` class for a reference implementation when you also need:

- foreground vs. background runs,
- live progress polling friendly to Jupyter,
- an on-disk record / registry of past runs.

### 8.4 Wire up `__init__.py`

`src/upsonic/prebuilt/<your_agent>/__init__.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import YourAgent


def _get_classes() -> dict[str, Any]:
    from .agent import YourAgent
    return {"YourAgent": YourAgent}


def __getattr__(name: str) -> Any:
    classes = _get_classes()
    if name in classes:
        return classes[name]
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Available: {list(classes.keys())}"
    )


__all__ = ["YourAgent"]
```

`src/upsonic/prebuilt/__init__.py`: extend `_get_classes()` and `__all__`
the same way.

### 8.5 Conventions to follow

| Convention | Why |
| --- | --- |
| `AGENT_REPO == "https://github.com/Upsonic/Upsonic"` | Users installing from PyPI must be able to clone the template. Private repos break the contract. |
| `AGENT_FOLDER == "src/upsonic/prebuilt/<your_agent>/template"` | Keeps source code and template colocated. |
| Self-contained `agent.py` | A reader landing on it should see the agent class and any companion result/record classes — nothing else. |
| Pop `agent_repo` / `agent_folder` from `kwargs` before `super().__init__` | Otherwise users could override the pinned template at construction. |
| Pass placeholders via `**template_params` | The base class extracts the placeholder set from `first_message.md` and raises listing what's missing — typos surface as clear `ValueError`s. |
| Templates are runtime data | No imports, no f-strings, only `str.format` placeholders. |

### 8.6 Reviewer checklist

Before opening a PR adding `<your_agent>`:

- [ ] `src/upsonic/prebuilt/<your_agent>/__init__.py` lazy-imports from `agent.py`.
- [ ] `src/upsonic/prebuilt/__init__.py` re-exports the agent class and lists it in `__all__`.
- [ ] `agent.py` subclasses `PrebuiltAutonomousAgentBase`, sets `AGENT_REPO` and `AGENT_FOLDER`, and pops `agent_repo` / `agent_folder` from `kwargs`.
- [ ] `template/system_prompt.md` and `template/first_message.md` exist.
- [ ] Every placeholder in `first_message.md` is documented in the docstring of the high-level method that supplies it.
- [ ] The agent appears in the README and any concept-level docs that enumerate the available prebuilts.
- [ ] A smoke test calls `from upsonic.prebuilt import YourAgent` and instantiates it (no API calls required).

Once the PR lands on `master`, the next user who runs
`from upsonic.prebuilt import YourAgent` pulls the freshly-merged template
at run time — there is no caching layer to bust.

---

## Appendix A — Output JSON contract for `AppliedScientist`

For reference, the canonical schema of `experiments/{research_name}/result.json`
(from `system_prompt.md` and `skills/evaluate/SKILL.md`):

```json
{
  "name": "tabpfn_adult",
  "verdict": "BETTER",
  "summary": "2-3 short paragraphs describing the new method and its trade-offs.",
  "explanation": "2-3 sentences explaining WHY this verdict was reached, referencing concrete metric numbers.",
  "comparison": {
    "metrics": [
      {
        "name": "accuracy",
        "current": 0.853,
        "new":     0.872,
        "diff":    0.019,
        "diff_display": "+0.019",
        "unit": null,
        "higher_is_better": true,
        "better": "new"
      },
      {
        "name": "training_time_seconds",
        "current": 2.0,
        "new":     45.0,
        "diff":    43.0,
        "diff_display": "+43.0",
        "unit": "seconds",
        "higher_is_better": false,
        "better": "current"
      }
    ]
  },
  "file_locations": {
    "current_notebook":  "experiments/tabpfn_adult/current.ipynb",
    "current_data":      "experiments/tabpfn_adult/current_data/",
    "new_notebook":      "experiments/tabpfn_adult/new.ipynb",
    "research_source":   "experiments/tabpfn_adult/research.pdf",
    "experiment_log":    "experiments/tabpfn_adult/log.json"
  }
}
```

And `experiments.json`:

```json
{
  "experiments": [
    {
      "name": "tabpfn_adult",
      "date": "YYYY-MM-DD",
      "status": "completed",
      "paper": "TabPFN",
      "baseline_model": "XGBoost",
      "new_method":     "TabPFN",
      "verdict":        "BETTER",
      "key_metric":     {"name": "accuracy", "baseline": 0.85, "new": 0.87},
      "path":           "experiments/tabpfn_adult/"
    }
  ]
}
```

And the canonical `progress.json` schema (from `skills/progress/SKILL.md`):

```json
{
  "name": "tabpfn_adult",
  "status": "RUNNING",
  "started_at": "2026-04-17T10:00:00Z",
  "updated_at": "2026-04-17T10:25:00Z",
  "phases": [
    {"index": 0, "name": "Setup",           "status": "done",    "summary": "Copied notebook, data, paper."},
    {"index": 1, "name": "Analyze Current", "status": "done",    "summary": "Baseline is XGBoost, 85.3% accuracy."},
    {"index": 2, "name": "Research",        "status": "current", "summary": null},
    {"index": 3, "name": "Benchmark",       "status": "pending", "summary": null},
    {"index": 4, "name": "Implement",       "status": "pending", "summary": null},
    {"index": 5, "name": "Evaluate",        "status": "pending", "summary": null}
  ],
  "current_activity": "Reading research.pdf — extracting method summary and requirements.",
  "issues": []
}
```

`Experiment.progress_bar`, `ExperimentRecord.phases`, and
`ExperimentRecord.current_activity` all rely on this contract being
preserved by the agent — which is why `progress/SKILL.md` is so strict
about field names, casing, and shape.

---

## Appendix B — Reference invocation outside Upsonic (`example.sh`)

`applied_scientist/template/example.sh` shows how to drive the same
prompts from the Anthropic `claude` CLI directly, useful as a sanity
check while authoring the template:

```bash
claude \
  --system-prompt-file "./system_prompt.md" \
  --dangerously-skip-permissions \
  --effort "medium" \
  "New experiment.
**Research paper:** example_1/tabpfn.pdf
**Current notebook:** example_1/Baseline XGBoost Adult.ipynb
**Current data:** downloaded in notebook (ucimlrepo, id=2)

Run the full experiment pipeline. Go from Phase 0 through Phase 5 without stopping. I want to see \`result.md\` at the end telling me whether this new method is better than what we have.

Start now."
```

Note the divergence from the in-package `first_message.md`: the shell
script predates the structured template parameters and uses a more
free-form first message. The Python prebuilt is the canonical entry
point — `example.sh` is preserved only as a hand-test fixture.
