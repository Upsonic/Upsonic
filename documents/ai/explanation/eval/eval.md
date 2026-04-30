---
name: evaluation-harness
description: Use when working on Upsonic's evaluation subsystem under `src/upsonic/eval/` — scoring agent correctness with an LLM-as-a-judge, profiling latency and memory, or asserting tool-call behavior. Use when a user asks to score agent outputs, run accuracy/performance/reliability evaluations, build pytest assertions on tool usage, profile an Agent/Graph/Team, or stream eval data to PromptLayer or Langfuse. Trigger when the user mentions AccuracyEvaluator, PerformanceEvaluator, ReliabilityEvaluator, EvaluationScore, ToolCallCheck, AccuracyEvaluationResult, PerformanceEvaluationResult, ReliabilityEvaluationResult, PerformanceRunResult, LLM-as-a-judge, judge agent, tracemalloc, tool-call assertions, expected_tool_calls, order_matters, exact_match, num_iterations, warmup_runs, promptlayer, langfuse, dataset_mode, run_with_output, assert_passed, _pl_helpers, accumulate_agent_usage, or extract_model_parameters.
---

# `src/upsonic/eval/` — Evaluation Subsystem

The `eval/` package is Upsonic's first-class evaluation harness. It provides three orthogonal evaluators — **accuracy**, **performance**, and **reliability** — that operate uniformly across the framework's main execution targets (`Agent`, `Graph`, `Team`) and emit Pydantic-typed result objects suitable for assertion in test suites or for downstream logging into observability platforms (PromptLayer, Langfuse).

This document is a deep technical walk-through of every Python file in the folder.

---

## 1. What this folder is — its role in Upsonic

Upsonic agents are non-deterministic systems. A reliability-focused framework therefore needs a way to:

1. **Score correctness** of generated responses against a known gold standard (LLM-as-a-judge).
2. **Profile latency and memory** under repeatable load with warmup and statistical aggregation.
3. **Verify tool-call behavior** — which tools the agent invoked, how often, and in which order.

The `eval/` folder provides these three capabilities as standalone evaluators that can be plugged into a developer's testing workflow. Each evaluator:

- Accepts a target (`Agent`, `Graph`, or `Team`) and a runtime input (`Task`, query, or post-hoc output).
- Executes the target one or more times under instrumentation.
- Returns a typed `pydantic.BaseModel` containing all raw measurements plus aggregated/derived statistics.
- Optionally streams the same data to PromptLayer / Langfuse via fire-and-forget background threads.

The design is deliberately functional — none of the evaluators mutate the agent under test, the result objects are immutable Pydantic models, and observability logging is non-blocking so it never affects the measured wall-clock latency.

---

## 2. Folder layout — tree diagram

```
src/upsonic/eval/
├── __init__.py            # Lazy-loaded public API surface
├── _pl_helpers.py         # Shared helpers (PromptLayer parameter / usage extraction)
├── models.py              # Pydantic schema for every input and result type
├── accuracy.py            # AccuracyEvaluator (LLM-as-a-judge)
├── performance.py         # PerformanceEvaluator (latency + tracemalloc)
└── reliability.py         # ReliabilityEvaluator (tool-call assertions)
```

| File                | Role                                                                 | Public class / function                                                                                                                |
| ------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`       | Lazy-loaded re-export hub                                            | `__getattr__`                                                                                                                          |
| `_pl_helpers.py`    | PromptLayer-related extractors                                       | `extract_model_parameters`, `accumulate_agent_usage`                                                                                   |
| `models.py`         | Typed input/result schema                                            | `EvaluationScore`, `PerformanceRunResult`, `ToolCallCheck`, `AccuracyEvaluationResult`, `PerformanceEvaluationResult`, `ReliabilityEvaluationResult` |
| `accuracy.py`       | LLM-as-a-judge evaluator                                             | `AccuracyEvaluator`                                                                                                                    |
| `performance.py`    | Latency / memory profiler                                            | `PerformanceEvaluator`                                                                                                                 |
| `reliability.py`    | Tool-call verification engine                                        | `ReliabilityEvaluator`                                                                                                                 |

---

## 3. Top-level files — file-by-file walkthrough

### 3.1 `__init__.py`

The package init implements a **lazy-loading public API** via `__getattr__`. Heavy modules (which transitively pull in `Agent`, `Graph`, `Team`, `tracemalloc`, etc.) are not imported when `upsonic.eval` is first referenced — they are pulled in only when a specific attribute is accessed.

| Symbol                       | Purpose                                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| `_get_evaluator_classes()`   | Returns a dict mapping `'AccuracyEvaluator'`, `'PerformanceEvaluator'`, `'ReliabilityEvaluator'` to their actual classes. |
| `_get_model_classes()`       | Returns a dict mapping all six `models.py` classes to their actual implementations.       |
| `__getattr__(name)`          | PEP 562 module-level fallback. Resolves a missing attribute by checking the evaluator and model dictionaries, raising `AttributeError` if unknown. |
| `__all__`                    | Declares the official public surface for star-imports and IDE completion.                 |

`TYPE_CHECKING` is also used to make every public symbol available to static type-checkers without triggering import at runtime.

```python
from upsonic.eval import AccuracyEvaluator, PerformanceEvaluator, ReliabilityEvaluator
# All three classes are loaded lazily on first attribute access.
```

### 3.2 `_pl_helpers.py`

A small, internal helper module that centralizes the extraction of two pieces of state from a live `Agent` instance — the model's settings dict and the most recent token-usage / cost trio. These are reused by every evaluator that ships data to PromptLayer.

| Function                              | Signature                                                                                  | Returns                            | Description                                                                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `extract_model_parameters(agent)`     | `Agent -> Optional[Dict[str, Any]]`                                                        | A copy of `agent.model._settings` | Returns `None` if the agent's model has no `_settings` dict or it is empty. The dict is shallow-copied to avoid downstream mutation. |
| `accumulate_agent_usage(agent)`       | `Agent -> Tuple[int, int, float]` (`input_tokens, output_tokens, price`)                    | Triple of integers / float         | Reads `agent._agent_run_output.usage` (set by the most recent run) for token counts, and calls `agent._calculate_aggregated_cost()` for the dollar price. Falls back to `0` for any missing piece. |

These helpers operate on private attributes (`_settings`, `_agent_run_output`, `_calculate_aggregated_cost`) and are therefore deliberately under-documented in the public docs — they are an implementation contract between `eval/` and `agent/`.

### 3.3 `models.py`

All Pydantic models used as inputs to and outputs of the evaluators. Every field carries an explicit `description` so the schemas double as documentation and as stable interfaces for the LLM-as-a-judge prompt.

#### 3.3.1 `EvaluationScore` (judge output)

The structured response a judge agent must produce.

| Field        | Type      | Constraints     | Meaning                                                                       |
| ------------ | --------- | --------------- | ----------------------------------------------------------------------------- |
| `score`      | `float`   | `1 <= s <= 10`  | Numerical quality score.                                                      |
| `reasoning`  | `str`     | required        | Step-by-step inner monologue justifying the score.                            |
| `is_met`     | `bool`    | required        | Definitive pass/fail flag against the *spirit* of the expected output.         |
| `critique`   | `str`     | required        | Actionable feedback for improvement.                                          |

#### 3.3.2 `PerformanceRunResult` (atomic perf measurement)

| Field                     | Type   | Meaning                                                          |
| ------------------------- | ------ | ---------------------------------------------------------------- |
| `latency_seconds`         | `float` | Wall-clock seconds for this single run (`time.perf_counter`).    |
| `memory_increase_bytes`   | `int`   | `end_mem - start_mem` from `tracemalloc.get_traced_memory`.      |
| `memory_peak_bytes`       | `int`   | `peak_mem - start_mem`, isolating per-run peak.                  |

#### 3.3.3 `ToolCallCheck`

| Field          | Type   | Meaning                                                  |
| -------------- | ------ | -------------------------------------------------------- |
| `tool_name`    | `str`  | The expected tool's name.                                |
| `was_called`   | `bool` | `True` iff the tool appeared in the actual call history. |
| `times_called` | `int`  | Total number of calls observed.                          |

#### 3.3.4 `AccuracyEvaluationResult`

The aggregated accuracy report returned to the user.

| Field                | Type                       | Notes                                                    |
| -------------------- | -------------------------- | -------------------------------------------------------- |
| `evaluation_scores`  | `List[EvaluationScore]`    | One entry per iteration.                                 |
| `average_score`      | `float`                    | `mean(scores)`.                                           |
| `user_query`         | `str`                      | The original prompt.                                     |
| `expected_output`    | `str`                      | Ground-truth answer.                                     |
| `generated_output`   | `str`                      | The agent's actual answer (last iteration's output).      |

`Config.from_attributes = True` enables ORM-style construction.

#### 3.3.5 `PerformanceEvaluationResult`

| Field                    | Type                         | Notes                                                                |
| ------------------------ | ---------------------------- | -------------------------------------------------------------------- |
| `all_runs`               | `List[PerformanceRunResult]` | Raw per-iteration data.                                              |
| `num_iterations`         | `int`                        | Configured count.                                                    |
| `warmup_runs`            | `int`                        | Configured warmup count.                                             |
| `latency_stats`          | `Dict[str, float]`           | Keys: `average`, `median`, `min`, `max`, `std_dev` (seconds).        |
| `memory_increase_stats`  | `Dict[str, float]`           | Same keys, in bytes.                                                 |
| `memory_peak_stats`      | `Dict[str, float]`           | Same keys, in bytes.                                                 |

#### 3.3.6 `ReliabilityEvaluationResult`

| Field                     | Type                   | Notes                                                         |
| ------------------------- | ---------------------- | ------------------------------------------------------------- |
| `passed`                  | `bool`                 | Top-level pass/fail flag.                                     |
| `summary`                 | `str`                  | Human-readable explanation.                                    |
| `expected_tool_calls`     | `List[str]`            | The user's specification.                                      |
| `actual_tool_calls`       | `List[str]`            | Flat ordered list extracted from `Task.tool_calls`.            |
| `checks`                  | `List[ToolCallCheck]`  | One per expected tool.                                         |
| `missing_tool_calls`      | `List[str]`            | Tools never observed.                                          |
| `unexpected_tool_calls`   | `List[str]`            | Populated only when `exact_match=True`.                        |

It also defines `assert_passed()`, which raises `AssertionError(f"Reliability evaluation failed: {self.summary}")` if `passed` is False — the bridge into pytest-style assertion-based tests.

---

## 4. Subfolder-level deep dives (per evaluator file)

The eval/ folder is flat — there are no subfolders — but each evaluator file is dense enough to warrant its own section.

### 4.1 `accuracy.py` — `AccuracyEvaluator`

**Purpose.** Drive an agent / graph / team against a query, capture its output, then ask a *judge agent* (an LLM-as-a-judge) to score it against an expected answer.

#### 4.1.1 Constructor

```python
AccuracyEvaluator(
    judge_agent: Agent,
    agent_under_test: Union[Agent, Graph, Team],
    query: str,
    expected_output: str,
    additional_guidelines: Optional[str] = None,
    num_iterations: int = 1,
    promptlayer: Optional[PromptLayer] = None,
    promptlayer_dataset_name: Optional[str] = None,
    promptlayer_dataset_mode: str = "log_only",   # or "new_version"
    langfuse: Optional[Langfuse] = None,
    langfuse_dataset_name: Optional[str] = None,
    langfuse_run_name: Optional[str] = None,
)
```

Validation:

- `judge_agent` must be an `Agent`.
- `agent_under_test` must be an `Agent`, `Graph`, or `Team` (raises `TypeError` otherwise).
- `num_iterations` must be a positive integer.

The constructor stores everything as instance state and initializes `self._results: List[EvaluationScore] = []`.

#### 4.1.2 `async run(print_results=True) -> AccuracyEvaluationResult`

The main happy path.

| Step | Action                                                                                                                                     |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | Reset `self._results`, capture `eval_start_time = time.time()`.                                                                            |
| 2    | For each iteration: build `Task(description=self.query)`, dispatch to the appropriate target type:                                          |
|      | - `Agent`: `await agent.do_async(task)`, read `task.response`, accumulate token usage via `accumulate_agent_usage`, capture `_agent_run_output.trace_id`. |
|      | - `Graph`: `state = await graph.run_async(verbose=False)`, then `state.get_latest_output()`.                                                |
|      | - `Team`: `await asyncio.to_thread(team.complete, task)` (the synchronous `complete` is offloaded to a thread).                              |
| 3    | Raise `ValueError` if the generated output is `None`.                                                                                       |
| 4    | Stringify the output and call `_get_judge_score(...)` to score it. Add the judge's own usage to the running totals.                         |
| 5    | Append the `EvaluationScore` to `self._results`.                                                                                            |
| 6    | After all iterations: `_aggregate_and_present_results(...)`.                                                                                |
| 7    | If a PromptLayer client was provided, kick off two background loggers:                                                                      |
|      | - `_log_eval_to_promptlayer_background` (sends one summary `alog` per evaluation).                                                          |
|      | - `_log_to_promptlayer_dataset_background` (creates / updates a PromptLayer dataset).                                                       |
| 8    | If a Langfuse client was provided, kick off `_log_to_langfuse_dataset_background` with the captured `trace_id`.                             |

#### 4.1.3 `async run_with_output(output, print_results=True, trace_id=None) -> AccuracyEvaluationResult`

A "score-only" variant that *skips* executing the agent under test and instead scores a pre-existing `output` string `num_iterations` times. Useful when the output was already generated elsewhere (e.g., from a saved log) — the evaluator becomes a pure judge wrapper. PromptLayer / Langfuse logging follows the same pattern as `run()`, but token usage is sourced exclusively from the judge agent.

#### 4.1.4 PromptLayer integration

| Method                                                | Role                                                                                         |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `_log_eval_to_promptlayer_background`                 | Spawns a daemon `threading.Thread` running `asyncio.run(_log_eval_to_promptlayer(...))`.      |
| `async _log_eval_to_promptlayer`                      | Builds the PromptLayer `alog` payload: provider/model parsed from `agent.model_name`, average score scaled to 0–100, per-iteration scores, model parameters, and metadata including `expected_output` and `all_met`. |
| `_log_to_promptlayer_dataset_background` / `_sync`    | Either ensures a dataset group exists (`log_only`) or uploads a new CSV version (`new_version`). The CSV columns are `query, expected_output, generated_output, score, reasoning`, base64-encoded before upload. |

#### 4.1.5 Langfuse integration

`_log_to_langfuse_dataset_sync` (always invoked from a daemon thread):

1. Calls `langfuse.create_dataset(name)` (idempotent).
2. Calls `langfuse.create_dataset_item(name, input=query, expected_output=expected)`.
3. If a `trace_id` was captured: `langfuse.flush()` then `time.sleep(10)` to let the OTel pipeline ingest the trace, then `update_trace(trace_id, output=generated_output)`, `create_dataset_run_item(...)`, and finally `score(trace_id, name="accuracy_eval_score", value=avg_score, data_type="NUMERIC")`.

#### 4.1.6 Internal helpers

| Method                                | Role                                                                                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `async _get_judge_score(generated)`   | Builds the judge prompt, wraps it in a `Task(response_format=EvaluationScore, not_main_task=True)`, runs the judge, and validates the response. |
| `_aggregate_and_present_results`      | Computes the simple `mean` of all `score` values, builds an `AccuracyEvaluationResult`, and optionally prints it.                          |
| `_construct_judge_prompt`             | Templated XML-style prompt with `<UserQuery>`, `<ExpectedAnswer>`, `<GeneratedAnswer>`, `<EvaluationGuidelines>` sections.                   |
| `_print_formatted_results`            | Renders a Rich `Panel` colored green / yellow / red based on `>= 8`, `>= 5`, otherwise.                                                    |

### 4.2 `performance.py` — `PerformanceEvaluator`

**Purpose.** A profiler that measures latency and memory for an `Agent`, `Graph`, or `Team` over many iterations, with optional warmup, and aggregates the data into descriptive statistics.

#### 4.2.1 Constructor

```python
PerformanceEvaluator(
    agent_under_test: Union[Agent, Graph, Team],
    task: Union[Task, List[Task]],
    num_iterations: int = 10,
    warmup_runs: int = 2,
    promptlayer: Optional[PromptLayer] = None,
)
```

Validation enforces the agent type, the task type (single or list), and that `num_iterations >= 1` and `warmup_runs >= 0`.

#### 4.2.2 `async run(print_results=True) -> PerformanceEvaluationResult`

The measurement loop is wrapped in a `try / finally` so that `tracemalloc` is always stopped even if a run throws.

| Phase | Action                                                                                                                                          |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | `tracemalloc.start()`.                                                                                                                          |
| 1     | Run `warmup_runs` warmups using `copy.deepcopy(self.task)` for each, dispatched through `_execute_component`. No measurements are kept.          |
| 2     | For each measurement iteration, deep-copy the task again, then:                                                                                 |
|       | - `tracemalloc.clear_traces()`, capture `start_mem`.                                                                                            |
|       | - `start_time = time.perf_counter()`.                                                                                                            |
|       | - Execute the component.                                                                                                                         |
|       | - Capture `latency = time.perf_counter() - start_time`.                                                                                          |
|       | - Capture `end_mem, peak_mem = tracemalloc.get_traced_memory()`.                                                                                 |
|       | - If the component is an `Agent`, accumulate token usage / cost via `accumulate_agent_usage`.                                                    |
|       | - Build a `PerformanceRunResult(latency, end_mem - start_mem, peak_mem - start_mem)`.                                                            |
| 3     | `tracemalloc.stop()` in `finally`.                                                                                                              |
| 4     | `_aggregate_results(...)` builds the final report; optionally print and optionally fire-and-forget log to PromptLayer.                          |

#### 4.2.3 `async _execute_component(agent, task)`

Dispatches based on the agent class:

| Target type | Call                                                                                              |
| ----------- | ------------------------------------------------------------------------------------------------- |
| `Agent`     | `await agent.do_async(task[0] if list else task)`                                                 |
| `Graph`     | `await agent.run_async(verbose=False, show_progress=False)`                                       |
| `Team`      | `await agent.multi_agent_async(entity_configurations=agent.entities, tasks=task)`                 |

#### 4.2.4 Statistics

`_calculate_stats(data)` produces a stable dictionary of `{"average", "median", "min", "max", "std_dev"}`. `std_dev` falls back to `0.0` when there is only one sample (since `statistics.stdev` requires `len >= 2`).

`_aggregate_results` decomposes the per-run results into three lists (`latencies`, `mem_increases`, `mem_peaks`) and feeds each through `_calculate_stats`, returning a `PerformanceEvaluationResult`.

#### 4.2.5 PromptLayer logging

`_log_eval_to_promptlayer_background -> _log_eval_to_promptlayer` constructs a single `alog` payload representing the entire evaluation. Notable computations:

- `latency_score = max(0, min(100, 100 - int(round(avg_latency * 10))))` — a 0–100 score where 100 = "instantaneous", and each tenth of a second deducts a point.
- `task_desc` is built by joining `task.description` over the list (or using the single task's description).
- All three stat dictionaries (latency / memory increase / memory peak) are sent in `metadata`.

#### 4.2.6 Pretty-printing

`_print_formatted_results` renders a Rich `Table` with one row per metric (Latency, Memory Increase, Memory Peak) and columns Average / Median / Min / Max / Std. Dev. Latency is shown in milliseconds, and memory bytes are formatted by `format_mem`:

| Range         | Output         |
| ------------- | -------------- |
| `|x| < 1 KB`  | `"<x> B"`     |
| `< 1 MB`      | `"<x> KB"`    |
| `>= 1 MB`     | `"<x> MB"`    |

### 4.3 `reliability.py` — `ReliabilityEvaluator`

**Purpose.** A *post-execution* assertion engine that inspects the tool calls recorded on already-executed `Task`(s) (or a `Graph`) and verifies that the agent invoked the expected tools.

This evaluator is intentionally synchronous — it does not run the agent itself; the user runs the agent first, then passes the resulting `Task` objects to `run()`.

#### 4.3.1 Constructor

```python
ReliabilityEvaluator(
    expected_tool_calls: List[str],
    order_matters: bool = False,
    exact_match: bool = False,
    promptlayer: Optional[PromptLayer] = None,
    agent_under_test: Optional[Agent] = None,
)
```

| Flag             | Effect                                                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `order_matters`  | If `True`, the evaluator checks that `expected_tool_calls` appears as a subsequence of `actual_tool_calls`.                                  |
| `exact_match`    | If `True`, the evaluator additionally fails when *unexpected* tools (those not in `expected_tool_calls`) appear in the actual call list.    |
| `agent_under_test` | Optional. Only used to enrich PromptLayer logs with provider, model, and parameters.                                                       |

The constructor enforces that `expected_tool_calls` is a non-empty `List[str]`.

#### 4.3.2 `run(run_result, print_results=True) -> ReliabilityEvaluationResult`

| Step | Action                                                                                                                                         |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Capture `eval_start_time`.                                                                                                                     |
| 2    | Normalize the tool-call history via `_normalize_tool_call_history` (see below).                                                                |
| 3    | Walk `expected_tool_calls`: count occurrences in the actual history, append a `ToolCallCheck`, and accumulate `missing_tool_calls`.            |
| 4    | If `order_matters`, validate subsequence membership using `it = iter(actual); all(t in it for t in expected)` — a Pythonic short-circuit subsequence test. |
| 5    | If `exact_match`, compute `set(actual) - set(expected)` and fail if non-empty.                                                                  |
| 6    | Concatenate human-readable summary messages and build the `ReliabilityEvaluationResult`.                                                       |
| 7    | Optionally print, then (if PromptLayer is configured) fire-and-forget log.                                                                     |

#### 4.3.3 `_normalize_tool_call_history(run_result)`

Accepts `Task`, `List[Task]`, or `Graph` and produces a flat ordered list of `tool_name` strings:

- `Task`: `extend(call['tool_name'] for call in task.tool_calls)`.
- `List[Task]`: same, iterated.
- `Graph`: iterates over `graph.nodes`, restricted to `TaskNode`s whose `id` appears in `graph.state.task_outputs.keys()` — i.e., only nodes that actually executed.
- Anything else: raises `TypeError`.

#### 4.3.4 `_extract_task_usage(run_result)`

Mirrors the normalizer but pulls per-task `total_input_token`, `total_output_token`, `total_cost` to provide PromptLayer with realistic token / cost numbers. Missing values default to zero.

#### 4.3.5 PromptLayer logging

`_log_eval_to_promptlayer_background -> _log_eval_to_promptlayer` (note: synchronous `log`, not `alog`, so no `asyncio.run` is needed inside the thread). Builds a per-tool score dict like:

```python
per_tool_scores = {
    f"tool_{check.tool_name}_called": 100 if check.was_called else 0
    for check in result.checks
}
```

If `agent_under_test` was supplied, the actual provider/model/parameters are extracted; otherwise it falls back to `provider="upsonic", model="reliability_eval"`.

#### 4.3.6 Rich rendering

`_print_formatted_results` produces a Rich `Panel` containing a table with rows `(✅/❌, tool_name, times_called)` and a subtitle showing both the expected and actual lists.

---

## 5. Cross-file relationships

```
                    ┌──────────────────────┐
                    │   models.py          │
                    │  (Pydantic schema)   │
                    └─────────┬────────────┘
                              │ used by
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌────────────┐       ┌─────────────┐       ┌─────────────┐
  │ accuracy.py│       │performance  │       │reliability  │
  │            │       │   .py       │       │   .py       │
  └─────┬──────┘       └──────┬──────┘       └──────┬──────┘
        │                     │                     │
        │   imports           │   imports           │   imports
        ▼                     ▼                     ▼
              ┌──────────────────────────────┐
              │   _pl_helpers.py             │
              │ (PromptLayer extractors)     │
              └──────────────────────────────┘
                              │
                              │ also uses
                              ▼
              ┌──────────────────────────────┐
              │ upsonic.agent.agent.Agent    │
              │ upsonic.graph.graph.Graph    │
              │ upsonic.team.team.Team       │
              │ upsonic.tasks.tasks.Task     │
              │ upsonic.utils.printing       │
              └──────────────────────────────┘
```

| Concern                  | Centralized in    | Consumed by                                                                  |
| ------------------------ | ----------------- | ---------------------------------------------------------------------------- |
| Pydantic result types     | `models.py`       | All three evaluators + the `__init__.py` lazy loader.                         |
| PromptLayer model params  | `_pl_helpers.py`  | `accuracy.py`, `performance.py`, `reliability.py`.                            |
| PromptLayer agent usage   | `_pl_helpers.py`  | `accuracy.py`, `performance.py` (reliability uses task-level data instead).   |
| Rich console / debug log  | `utils.printing`  | All three evaluators for human-readable output.                               |
| Background logging pattern| Each evaluator    | All three use `threading.Thread(target=..., daemon=True)` with try/except.    |

---

## 6. Public API

The eval folder exports exactly nine names. Everything else is internal.

| Name                          | Kind            | Module          |
| ----------------------------- | --------------- | --------------- |
| `AccuracyEvaluator`           | class           | `accuracy.py`   |
| `PerformanceEvaluator`        | class           | `performance.py`|
| `ReliabilityEvaluator`        | class           | `reliability.py`|
| `EvaluationScore`             | Pydantic model  | `models.py`     |
| `AccuracyEvaluationResult`    | Pydantic model  | `models.py`     |
| `ToolCallCheck`               | Pydantic model  | `models.py`     |
| `ReliabilityEvaluationResult` | Pydantic model  | `models.py`     |
| `PerformanceRunResult`        | Pydantic model  | `models.py`     |
| `PerformanceEvaluationResult` | Pydantic model  | `models.py`     |

```python
from upsonic.eval import (
    AccuracyEvaluator,
    PerformanceEvaluator,
    ReliabilityEvaluator,
    EvaluationScore,
    AccuracyEvaluationResult,
    ToolCallCheck,
    ReliabilityEvaluationResult,
    PerformanceRunResult,
    PerformanceEvaluationResult,
)
```

---

## 7. Integration with the rest of Upsonic

The eval folder is one of the most integration-heavy components in Upsonic. The exact integration surface:

| Upsonic module                                  | How eval/ uses it                                                                                                  |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `upsonic.agent.agent.Agent`                     | Type-checked via `isinstance`. Accuracy uses `agent.do_async(task)` and `agent._agent_run_output`. Performance reuses both. Reliability optionally takes an agent only for PromptLayer enrichment. |
| `upsonic.graph.graph.Graph` / `TaskNode`        | Accuracy: `await graph.run_async(verbose=False)` then `state.get_latest_output()`. Performance: `run_async(verbose=False, show_progress=False)`. Reliability: walks `graph.nodes` filtered by `graph.state.task_outputs`. |
| `upsonic.team.team.Team`                        | Accuracy: `await asyncio.to_thread(team.complete, task)`. Performance: `team.multi_agent_async(entity_configurations=team.entities, tasks=task)`. |
| `upsonic.tasks.tasks.Task`                      | All three evaluators consume `Task` objects. Reliability requires `task.tool_calls` (list of `{'tool_name': ...}` dicts) and `task.total_input_token`, `total_output_token`, `total_cost` for usage extraction. Accuracy uses `Task(response_format=EvaluationScore, not_main_task=True)` for the judge. |
| `upsonic.utils.printing.console / debug_log`    | The shared Rich `Console` and a debug logger.                                                                       |
| `upsonic.integrations.promptlayer.PromptLayer`  | Optional dependency. Accuracy, Performance, and Reliability all log via `alog` / `log`. Accuracy additionally calls `list_datasets`, `create_dataset_group`, `create_dataset_version_from_file`, and `_parse_provider_model`. |
| `upsonic.integrations.langfuse.Langfuse`        | Optional dependency, used by `AccuracyEvaluator` only: `create_dataset`, `create_dataset_item`, `flush`, `update_trace`, `create_dataset_run_item`, `score`. |

The agent module's private API (`agent.model._settings`, `agent._agent_run_output`, `agent._calculate_aggregated_cost()`) is touched only via `_pl_helpers.py`, isolating the coupling. Any change to those private interfaces should keep `_pl_helpers.py` in sync.

---

## 8. End-to-end flow of an evaluation run

Below are concrete walk-throughs of each evaluator type.

### 8.1 Accuracy — full LLM-as-a-judge cycle

```python
from upsonic import Agent
from upsonic.eval import AccuracyEvaluator

judge = Agent(model="openai/gpt-4o")
hero  = Agent(model="anthropic/claude-3-sonnet")

evaluator = AccuracyEvaluator(
    judge_agent=judge,
    agent_under_test=hero,
    query="What is the capital of France?",
    expected_output="Paris",
    additional_guidelines="Accept only single-word answers.",
    num_iterations=3,
)
result = await evaluator.run()
print(result.average_score, result.evaluation_scores[0].critique)
```

Internally, on each of three iterations:

1. A new `Task(description=query)` is created.
2. `hero.do_async(task)` runs the agent; `task.response` becomes the generated output.
3. `_pl_helpers.accumulate_agent_usage(hero)` records tokens and cost.
4. `_get_judge_score(generated)` builds a templated XML prompt, wraps it in `Task(response_format=EvaluationScore, not_main_task=True)`, and runs the judge agent. The judge's structured response is validated.
5. The score is appended to `self._results`.

After the loop, `_aggregate_and_present_results` builds the `AccuracyEvaluationResult` with `average_score = mean(scores)` and prints a Rich panel colored green / yellow / red. PromptLayer / Langfuse loggers fire on background daemon threads if configured.

### 8.2 Performance — `tracemalloc` profiling

```python
from upsonic import Task
from upsonic.eval import PerformanceEvaluator

evaluator = PerformanceEvaluator(
    agent_under_test=hero,
    task=Task(description="Summarize the latest tech news."),
    num_iterations=20,
    warmup_runs=3,
)
report = await evaluator.run()
print(report.latency_stats["median"], report.memory_peak_stats["max"])
```

Per iteration, the evaluator:

1. `copy.deepcopy(self.task)` — every run gets a fresh task object so previous responses do not leak into measurements.
2. `tracemalloc.clear_traces()` resets the allocation state.
3. Captures `start_mem`, `start_time = time.perf_counter()`.
4. Dispatches to `_execute_component`, which calls the appropriate async method.
5. Captures `end_time`, `end_mem`, `peak_mem`.
6. Builds a `PerformanceRunResult(latency_seconds, memory_increase_bytes, memory_peak_bytes)` (peaks are stored as `peak_mem - start_mem` to isolate the run's contribution).

After all iterations, `_aggregate_results` collapses the lists into stat dicts. The Rich-printed table renders latency in milliseconds and memory in human-friendly units.

### 8.3 Reliability — tool-call assertions

```python
from upsonic import Task
from upsonic.eval import ReliabilityEvaluator

task = Task(description="Find the weather in Paris and email me.")
await hero.do_async(task)   # populates task.tool_calls

evaluator = ReliabilityEvaluator(
    expected_tool_calls=["search_web", "send_email"],
    order_matters=True,
    exact_match=False,
)
report = evaluator.run(task)
report.assert_passed()  # raises AssertionError on failure
```

Internally:

1. `_normalize_tool_call_history(task)` flattens the per-task `tool_calls` list into `["search_web", "send_email"]`.
2. For each expected tool, count occurrences and build a `ToolCallCheck`. Any zero-count tool is also pushed onto `missing_tool_calls`.
3. `order_matters=True` runs `it = iter(actual); all(t in it for t in expected)` — a one-pass subsequence test that returns `True` iff `expected_tool_calls` appears in the same order within `actual_tool_calls`.
4. `exact_match=True` (not used here) would compute the set difference and fail if any unexpected tools were called.
5. The evaluator builds the `ReliabilityEvaluationResult`, prints a Rich panel, and optionally logs to PromptLayer (with `_extract_task_usage` providing realistic token / cost data).

`report.assert_passed()` is the final glue into pytest — raising `AssertionError` if `passed is False`.

---

## 9. Design notes & implementation contracts

### 9.1 Background logging pattern

All three evaluators follow the identical pattern for non-blocking observability:

```python
def _log_eval_to_promptlayer_background(self, ...):
    def _run():
        try:
            asyncio.run(self._log_eval_to_promptlayer(...))   # or sync .log(...)
        except Exception as e:
            _logger.warning("Background ... logging failed: %s", e)

    threading.Thread(target=_run, daemon=True).start()
```

Three guarantees:

1. **Daemon=True** — threads don't prevent the process from exiting.
2. **Try/except** — any logging failure is downgraded to a warning, never an exception in the user's code.
3. **`asyncio.run` inside the thread** — the async version uses a fresh event loop; the sync version (Reliability) skips this.

### 9.2 Task isolation

The `PerformanceEvaluator` is the only one that *must* `copy.deepcopy` its task per run, because each `Task` accumulates state (`response`, `tool_calls`, token counters). Without the deep copy, latency measurements would degrade over time as tasks grow. Accuracy uses a fresh `Task(description=self.query)` per iteration for the same reason. Reliability operates on already-completed tasks, so no copy is needed.

### 9.3 Graph node filtering

Both `AccuracyEvaluator` (when target is a `Graph`) and `ReliabilityEvaluator` filter `graph.nodes` by membership in `graph.state.task_outputs.keys()`. This guarantees that *only executed* nodes contribute to the evaluation — partial graph runs (e.g., conditional paths not taken) do not count.

### 9.4 Provider/model parsing

When PromptLayer logs are constructed, the evaluators call `self.promptlayer._parse_provider_model(str(agent.model_name))`. This is an internal hook in the `PromptLayer` integration that splits an Upsonic-style `"<provider>/<model>"` string back into a 2-tuple suitable for PromptLayer's API.

### 9.5 Score scaling

The evaluators normalize their scores into PromptLayer's 0–100 range:

| Evaluator    | Source range                | Scaling                                                                                          |
| ------------ | --------------------------- | ------------------------------------------------------------------------------------------------ |
| Accuracy     | `score: float [1..10]`      | `int(round(score * 10))`, clamped to `[0, 100]`. `is_met` becomes `100 / 0`.                      |
| Performance  | `avg_latency_seconds`       | `100 - int(round(avg_latency * 10))`, clamped — penalizes 0.1 s per point.                         |
| Reliability  | `passed: bool`              | `100` for pass, `0` for fail. Per-tool: `100` if called, `0` otherwise.                            |

### 9.6 Pytest hook

`ReliabilityEvaluationResult.assert_passed()` is the only evaluator result that ships with a built-in pytest bridge. The other two return rich data and let the user decide on assertion thresholds (e.g., `assert result.average_score >= 8`).

---

## 10. Quick reference cheat sheet

| Need                                              | Use                                                                        |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| Score an agent's answer with an LLM-as-a-judge     | `AccuracyEvaluator(...).run()`                                             |
| Score a *pre-existing* answer (no agent run)       | `AccuracyEvaluator(...).run_with_output(output, trace_id=...)`             |
| Profile latency / memory                           | `PerformanceEvaluator(agent, task, num_iterations=20, warmup_runs=3).run()` |
| Verify which tools were used                       | `ReliabilityEvaluator(["t1", "t2"]).run(task_or_list_or_graph)`            |
| Order-sensitive tool check                         | `ReliabilityEvaluator(..., order_matters=True)`                            |
| Forbid unexpected tool calls                       | `ReliabilityEvaluator(..., exact_match=True)`                              |
| Pytest assertion                                   | `result.assert_passed()` on `ReliabilityEvaluationResult`                  |
| Stream eval data to PromptLayer                    | Pass `promptlayer=PromptLayer(...)` to any evaluator                       |
| Stream accuracy eval to Langfuse                   | Pass `langfuse=Langfuse(...)` to `AccuracyEvaluator`                       |
| Append evals into a PromptLayer dataset version    | `promptlayer_dataset_mode="new_version"` on `AccuracyEvaluator`            |
