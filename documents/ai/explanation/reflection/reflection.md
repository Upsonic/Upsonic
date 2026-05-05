---
name: agent-reflection-self-evaluation
description: Use when working on Upsonic's agent reflection / self-evaluation subsystem that scores LLM responses on accuracy, completeness, relevance, and clarity, then iteratively improves them. Use when a user asks to enable reflection on an Agent, configure evaluator models, tune acceptance thresholds or max iterations, debug the reflection loop, customize evaluation/improvement prompts, or understand how ReflectionStep integrates into the pipeline. Trigger when the user mentions ReflectionConfig, ReflectionProcessor, ReflectionStep, ReflectionAction, EvaluationCriteria, EvaluationResult, ReflectionState, ReflectionResult, ReflectionPrompts, evaluator agent, acceptance_threshold, max_iterations, self-critique, improvement loop, ACCEPT/REVISE/RETRY/CLARIFY, evaluation_prompt, improved_output, sub_agent_usage, src/upsonic/reflection/, or AutonomousAgent reflection.
---

# `src/upsonic/reflection/` — Agent Reflection & Self-Evaluation

## 1. What this folder is

The `reflection/` package implements the **self-evaluation and iterative
improvement** subsystem of an Upsonic `Agent`. Once the main LLM has produced
an answer for a `Task`, reflection takes that answer and runs an
*evaluator agent* (typically a second LLM run) over it. The evaluator scores
the response on four criteria — **accuracy, completeness, relevance, clarity**
— and either accepts the response or returns structured feedback that drives
a follow-up *improvement* pass on the same task. The cycle repeats until the
score crosses an acceptance threshold, an evaluator action terminates the
loop, or the maximum iteration count is reached.

The folder is intentionally small and self-contained:

- A **data-model layer** (`models.py`) that defines the Pydantic
  configuration object (`ReflectionConfig`), the structured evaluation
  envelope (`EvaluationCriteria`, `EvaluationResult`, `ReflectionAction`),
  loop-state tracking (`ReflectionState`), the result envelope returned to
  the pipeline (`ReflectionResult`), and the prompt templates
  (`ReflectionPrompts`).
- A **runtime layer** (`processor.py`) — `ReflectionProcessor` — that owns
  the evaluator-agent construction, the evaluation/improvement loop, prompt
  formatting, structured-output handling, sub-agent usage aggregation, and
  fallback behaviour when the evaluator itself fails.
- A **lazy-import shim** (`__init__.py`) that makes the public symbols
  importable as `from upsonic.reflection import …` without paying the import
  cost unless the symbol is actually used.

Reflection is wired into the agent execution pipeline via `ReflectionStep`
(see `src/upsonic/agent/pipeline/steps.py`). The step is a no-op unless the
`Agent` was constructed with `reflection=True` and/or
`reflection_config=ReflectionConfig(...)`. When enabled, it runs **after** the
main model call (and tool-result post-processing) and **before**
ReliabilityStep / final output materialisation.

```
Agent.do_async(task)
  → pipeline runs steps 1..N
      …
      → ModelCallStep              # main LLM call produces context.output
      …
      → ReflectionStep             # ←── this folder powers it
          → ReflectionProcessor.process_with_reflection(agent, task, output)
              → evaluator_agent.do_async(...)   # score+feedback
              → agent.do_async(improved_task)   # re-run with feedback (if revise/retry)
              → … repeat …
          ← returns ReflectionResult(evaluation_prompt, improved_output, …)
      → ReliabilityStep
      → CallManagementStep
      → return AgentRunOutput
```

## 2. Folder layout

```
src/upsonic/reflection/
├── __init__.py        # lazy re-exports of the public API
├── models.py          # Pydantic schemas, enums, prompt templates, loop state
└── processor.py       # ReflectionProcessor — evaluation/improvement loop
```

Three files, all pure Python, all import-light. There are no subfolders; the
"subfolders" section below is therefore intentionally minimal.

| File          | LOC (approx) | Responsibility                                                      |
| ------------- | ------------ | ------------------------------------------------------------------- |
| `models.py`   | 158          | Schemas, enums, state, prompt templates                             |
| `processor.py`| 343          | Reflection loop, evaluator-agent factory, improvement task building |
| `__init__.py` |  68          | Lazy attribute access via `__getattr__`, public `__all__`           |

## 3. Top-level files

### 3.1 `models.py`

Defines every dataclass-style object that flows in and out of the reflection
loop. All models are Pydantic v2 (`BaseModel` with `Field(...)`).

#### 3.1.1 `ReflectionAction` (Enum)

A string-valued enum that the evaluator must return as part of its
structured response. It is the *control-flow signal* of the loop.

| Member             | Value       | Meaning to the loop                                              |
| ------------------ | ----------- | ---------------------------------------------------------------- |
| `ACCEPT`           | `"accept"`  | Stop. Use the current response as-is. Sets `terminated_reason="evaluator_accepted"`. |
| `REVISE`           | `"revise"`  | Run an improvement pass with the evaluator's feedback.           |
| `RETRY`            | `"retry"`   | Same as `REVISE` from a control-flow standpoint — also triggers an improvement pass. |
| `CLARIFY`          | `"clarify"` | Stop. Marks `terminated_reason="clarification_needed"`. The agent should ask the user. |

#### 3.1.2 `EvaluationCriteria`

The four-axis rubric the evaluator must fill in. All fields are floats in
`[0, 1]`.

```python
class EvaluationCriteria(BaseModel):
    accuracy:     float   # how factually correct the response is
    completeness: float   # whether the task was fully addressed
    relevance:    float   # how on-topic the response is
    clarity:      float   # readability / structure

    def overall_score(self) -> float:
        return (accuracy + completeness + relevance + clarity) / 4.0
```

#### 3.1.3 `EvaluationResult`

The full structured payload returned by the evaluator agent (used as
`response_format=EvaluationResult` on the evaluator's `Task`):

| Field                    | Type                | Notes                                              |
| ------------------------ | ------------------- | -------------------------------------------------- |
| `criteria`               | `EvaluationCriteria`| The four rubric scores                             |
| `overall_score`          | `float`             | Aggregate score in `[0, 1]` (also re-derivable)    |
| `feedback`               | `str`               | Detailed prose feedback                            |
| `suggested_improvements` | `List[str]`         | Bulletable, actionable suggestions                 |
| `action`                 | `ReflectionAction`  | Control signal: accept / revise / retry / clarify  |
| `confidence`             | `float`             | Evaluator's confidence in its own judgement (0-1)  |

There is a `__post_init__` hook that recomputes `overall_score` from
`criteria.overall_score()`. Note: with Pydantic v2 `BaseModel` this hook is
not auto-invoked — the LLM is expected to fill `overall_score` itself, and
the loop reads it directly via `evaluation.overall_score`.

#### 3.1.4 `ReflectionConfig` — public configuration object

The single object users construct to enable reflection on an `Agent`.

```python
class ReflectionConfig(BaseModel):
    max_iterations: int                = 3
    acceptance_threshold: float        = 0.8
    evaluator_model: Optional[str]     = None   # falls back to main agent's model
    enable_self_critique: bool         = True
    enable_improvement_suggestions: bool = True
```

| Field                            | Default | Effect                                                                                |
| -------------------------------- | ------- | ------------------------------------------------------------------------------------- |
| `max_iterations`                 | `3`     | Hard cap on (evaluate + maybe improve) cycles.                                        |
| `acceptance_threshold`           | `0.8`   | First evaluation whose `overall_score ≥ threshold` ends the loop.                     |
| `evaluator_model`                | `None`  | Provider/model string for the evaluator agent. `None` → reuse main agent's model.     |
| `enable_self_critique`           | `True`  | Reserved flag — referenced from `ReflectionPrompts.SELF_CRITIQUE_PROMPT` template.    |
| `enable_improvement_suggestions` | `True`  | Reserved flag — wired through to the improvement prompt builder.                      |

#### 3.1.5 `ReflectionResult` — pipeline-facing return type

`ReflectionProcessor.process_with_reflection(...)` returns an instance of
this. It carries everything `ReflectionStep` needs to update
`AgentRunOutput`/`Task` state and to record the round-trip in chat history.

| Field                | Type                       | Purpose                                                                                                            |
| -------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `evaluation_prompt`  | `str`                      | The **first** evaluation prompt (used as the `UserPromptPart` content of the `ModelRequest` appended to chat).     |
| `improved_output`    | `Any`                      | The **last** output (final, post-reflection answer in the original task's `response_format`).                      |
| `improvement_made`   | `bool`                     | `True` iff `str(original) != str(final)`.                                                                          |
| `original_output`    | `Any`                      | Whatever the main agent originally produced (untouched).                                                           |
| `final_evaluation`   | `Optional[EvaluationResult]`| The last `EvaluationResult` recorded by the loop (None if the evaluator never ran successfully).                  |
| `termination_reason` | `Optional[str]`            | `"acceptance_threshold_met"`, `"evaluator_accepted"`, `"clarification_needed"`, `"improvement_failed"`, or `"max_iterations_reached"`. |
| `sub_agent_usage`    | `Optional[RunUsage]`       | Aggregate token usage from all evaluator + improvement sub-agent calls.                                             |

`model_config = {"arbitrary_types_allowed": True}` is set so that
`improved_output` and `original_output` can be Pydantic models, plain
strings, or any user-defined `response_format` instance.

#### 3.1.6 `ReflectionState` — internal loop bookkeeping

Not exposed to user code. Tracks per-iteration history and is the source of
the `should_continue` decision.

| Method / field                            | Purpose                                                              |
| ----------------------------------------- | -------------------------------------------------------------------- |
| `iteration: int`                          | Current iteration counter (incremented in `add_evaluation`).         |
| `evaluations: List[EvaluationResult]`     | Full history of evaluator outputs, in order.                          |
| `responses: List[str]`                    | Full history of candidate response texts that were evaluated.         |
| `final_response: Optional[str]`           | Set when the loop terminates.                                         |
| `terminated_reason: Optional[str]`        | Mirrors what gets surfaced as `ReflectionResult.termination_reason`.  |
| `add_evaluation(response, evaluation)`    | Append a `(response, evaluation)` tuple and increment iteration.      |
| `get_latest_evaluation()`                 | Returns the most recent evaluation, or `None`.                        |
| `should_continue(config)`                 | Loop guard: stops at `max_iterations` or when score ≥ threshold.      |

`should_continue` logic:

```python
def should_continue(self, config: ReflectionConfig) -> bool:
    if self.iteration >= config.max_iterations:
        return False
    latest_eval = self.get_latest_evaluation()
    if latest_eval and latest_eval.overall_score >= config.acceptance_threshold:
        return False
    return True
```

#### 3.1.7 `ReflectionPrompts`

A namespace (plain class with three string-template class attributes) that
the processor uses to build prompts. Each template is `.format(...)`-ed at
runtime.

| Template               | Placeholders                                                                              | When used                                            |
| ---------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `EVALUATION_PROMPT`    | `task_description`, `response`, `context`                                                 | Every evaluator-agent call.                          |
| `IMPROVEMENT_PROMPT`   | `task_description`, `previous_response`, `feedback`, `improvements`, `context`            | Every improvement re-run.                            |
| `SELF_CRITIQUE_PROMPT` | `task_description`, `response`, `context`                                                 | Reserved (controlled by `enable_self_critique`).     |

The evaluator is steered toward returning a structured `EvaluationResult` by
passing `response_format=EvaluationResult` on the `Task` it receives, not
by inlining JSON-schema instructions in the prompt itself.

### 3.2 `processor.py` — `ReflectionProcessor`

The single class that drives reflection at runtime. It is constructed by
`Agent.__init__` whenever a `ReflectionConfig` is present and is then used
exactly once per `Agent.do_async(task)` call (by `ReflectionStep`).

```python
class ReflectionProcessor:
    def __init__(self, config: ReflectionConfig): ...

    async def process_with_reflection(
        self,
        agent: "Agent",
        task: "Task",
        initial_response: Any,
    ) -> ReflectionResult: ...
```

#### 3.2.1 Public method: `process_with_reflection`

End-to-end orchestration. The order of operations is:

1. Construct an empty `ReflectionState`.
2. Coerce `initial_response` to a string via `_extract_response_text`.
3. Emit `reflection_started(iteration=1, max_iterations=...)` log event.
4. Spin up a fresh evaluator `Agent` via `_create_evaluator_agent(agent)`.
5. Pre-format the **first** evaluation prompt and stash it for the result
   envelope (this is what `ReflectionStep` will feed into chat history as
   the *FIRST INPUT*).
6. Initialise an aggregate `RunUsage` for sub-agent token bookkeeping.
7. Loop while `state.should_continue(config)`:
   - **Evaluate** current response — `await _evaluate_response(...)`.
   - Emit `reflection_evaluation(...)` log event.
   - `state.add_evaluation(...)`.
   - **Threshold check** — if `overall_score ≥ acceptance_threshold`, set
     `final_response`, mark `terminated_reason="acceptance_threshold_met"`,
     break.
   - **Action dispatch:**
     - `ACCEPT`   → break with `"evaluator_accepted"`
     - `CLARIFY`  → break with `"clarification_needed"`
     - `REVISE` / `RETRY` → call `_generate_improved_response(...)`. If it
       returns `None`, break with `"improvement_failed"`. Otherwise, replace
       `current_response` with the improved one and loop.
8. If the loop exited without setting `state.final_response`, fall through
   to `terminated_reason="max_iterations_reached"`.
9. Emit `reflection_completed(...)` log event.
10. Convert `final_response` (a string) back into the original
    `response_format` via `_convert_to_response_format`.
11. Compute `improvement_made = str(original) != str(final)`.
12. Return `ReflectionResult(...)`.

#### 3.2.2 Private helpers

| Helper                                | Returns                                  | Notes                                                                                                               |
| ------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `_create_evaluator_agent(main_agent)` | `Agent`                                  | Spawns a sibling `Agent` with `role="Response Evaluator"` and a hard-coded evaluator persona. Inherits `debug` flag. |
| `_evaluate_response(evaluator, task, response, state)` | `(EvaluationResult, RunUsage?)`         | Builds an `EVALUATION_PROMPT`, wraps it in a `Task(response_format=EvaluationResult, not_main_task=True)`, runs the evaluator. On any exception, falls back to `_create_fallback_evaluation`. |
| `_generate_improved_response(agent, task, prev, eval, state)` | `(Any, RunUsage?)`                    | Builds an `IMPROVEMENT_PROMPT`, wraps it in a `Task(response_format=task.response_format, tools=task.tools, attachments=task.attachments, not_main_task=True)`, runs the **main agent again** with the feedback. |
| `_extract_response_text(response)`    | `str`                                    | Coerces strings, Pydantic models (`model_dump_json`), and arbitrary objects into a stable text form for the LLM.    |
| `_convert_to_response_format(text, original, task)` | `Any`                            | Inverse of the above. If `task.response_format` is a Pydantic model, attempts `model_validate_json`. Falls back to plain text. |
| `_build_evaluation_context(task, state)` | `str`                                 | Concatenates `task.context` items + iteration counter into the `{context}` placeholder.                              |
| `_build_improvement_context(task, state)`| `str`                                 | Same as above but tailored for the improvement template; includes guidance after the second iteration.              |
| `_create_fallback_evaluation(response, error)` | `EvaluationResult`                | Returns a 0.5-across-the-board "ACCEPT, low confidence" evaluation, used when the evaluator agent itself raises.    |

#### 3.2.3 Sub-agent usage aggregation

Both `_evaluate_response` and `_generate_improved_response` request the full
`AgentRunOutput` (via `evaluator.do_async(eval_task, return_output=True)` /
`agent.do_async(improved_task, return_output=True)`) so they can read
`.usage` off the wrapper. Each non-`None` `RunUsage` is folded into a
running `aggregated_usage = RunUsage()` inside `process_with_reflection` via
`aggregated_usage.incr(eval_usage)`. The aggregate is surfaced as
`ReflectionResult.sub_agent_usage`, which `ReflectionStep` then folds into
the *parent* `AgentRunOutput.usage` so token cost reporting is end-to-end
correct even when reflection ran multiple cycles.

#### 3.2.4 Failure modes & fallbacks

| Situation                                      | Behaviour                                                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Evaluator agent raises any exception           | `_create_fallback_evaluation` produces a "0.5 / ACCEPT / 0.3 confidence" record; loop exits cleanly.   |
| Improvement re-run raises any exception        | `_generate_improved_response` returns `(None, None)`; loop sets `terminated_reason="improvement_failed"` and uses the previous good response. |
| Evaluator returns malformed structured output  | Pydantic parse error → caught by the exception path above → fallback evaluation.                       |
| `task.response_format` parse fails on rebuild  | `_convert_to_response_format` silently falls back to the raw improved-response string.                  |

### 3.3 `__init__.py`

A *lazy* package init using PEP-562 module-level `__getattr__`. Importing
`upsonic.reflection` itself does **not** import `processor.py` — and
therefore does not pull in `Agent`, `Task`, `RunUsage`, etc. — until you
actually access one of the public symbols.

```python
def _get_model_classes():
    from .models import (...)
    return { 'ReflectionAction': ..., 'EvaluationCriteria': ..., ... }

def _get_processor_classes():
    from .processor import ReflectionProcessor
    return { 'ReflectionProcessor': ReflectionProcessor }

def __getattr__(name: str) -> Any:
    if name in _get_model_classes(): return _get_model_classes()[name]
    if name in _get_processor_classes(): return _get_processor_classes()[name]
    raise AttributeError(...)
```

The `__all__` list defines the official public API:

```python
__all__ = [
    "ReflectionAction",
    "EvaluationCriteria",
    "EvaluationResult",
    "ReflectionConfig",
    "ReflectionState",
    "ReflectionPrompts",
    "ReflectionProcessor",
]
```

`TYPE_CHECKING`-only imports at the top of the file mean static type
checkers (mypy, IDEs) still see the full set of names with their real
types, even though they are loaded lazily at runtime.

## 4. Subfolders

There are no subfolders. The reflection feature is small enough to fit in
two implementation files plus a re-export shim. If the loop ever needs a
provider-specific specialisation (e.g. distinct evaluators per provider, or
streaming evaluators), the natural growth path is a `strategies/`
subpackage hung off `processor.py`.

## 5. Cross-file relationships

```
                ┌─────────────────────────────────────────────────────────────┐
                │                  src/upsonic/reflection/                    │
                │                                                             │
   __init__.py  │  lazy re-exports                                            │
        │       │                                                             │
        │       │   ┌──────────────────────┐    ┌──────────────────────────┐  │
        │       │   │       models.py      │    │       processor.py       │  │
        ▼       │   │                      │    │                          │  │
  user code     │   │ ReflectionConfig     │◀───│ self.config: …Config     │  │
                │   │ ReflectionAction     │◀───│ ReflectionAction.ACCEPT… │  │
                │   │ EvaluationCriteria   │◀───│ build EvaluationResult   │  │
                │   │ EvaluationResult     │◀───│ response_format=…        │  │
                │   │ ReflectionState      │◀───│ ReflectionState() loop   │  │
                │   │ ReflectionPrompts    │◀───│ EVALUATION_PROMPT.format │  │
                │   │ ReflectionResult     │───▶│ returned to caller       │  │
                │   └──────────────────────┘    └──────────────────────────┘  │
                └─────────────────────────────────────────────────────────────┘
                                      │                       │
                                      │                       ▼
                                      │           upsonic.agent.agent.Agent
                                      │           upsonic.tasks.tasks.Task
                                      │           upsonic.usage.RunUsage
                                      ▼
                              upsonic.utils.printing
                              (reflection_started, reflection_evaluation,
                               reflection_improvement_started,
                               reflection_completed)
```

Concrete inbound dependencies of `processor.py`:

| Imported symbol                                              | From                                  | Used for                                       |
| ------------------------------------------------------------ | ------------------------------------- | ---------------------------------------------- |
| `Agent`                                                      | `upsonic.agent.agent` (deferred)      | Building the evaluator agent.                  |
| `Task`                                                       | `upsonic.tasks.tasks` (deferred)      | Wrapping evaluation/improvement prompts.       |
| `RunUsage`                                                   | `upsonic.usage` (deferred)            | Aggregating sub-agent token usage.             |
| `reflection_started`, `reflection_evaluation`, `reflection_improvement_started`, `reflection_completed` | `upsonic.utils.printing` (deferred) | CLI/log output during the loop.                |

All `Agent`/`Task`/`RunUsage`/printing imports are **deferred inside method
bodies** — this is what keeps `from upsonic.reflection import ReflectionConfig`
cheap for users who only need to type-annotate.

## 6. Public API

Importable names (all also listed in `__all__`):

```python
from upsonic.reflection import (
    ReflectionConfig,        # construct to enable reflection on an Agent
    ReflectionProcessor,     # advanced: directly drive process_with_reflection()

    # data shapes you may receive in callbacks / inspection
    ReflectionAction,
    EvaluationCriteria,
    EvaluationResult,
    ReflectionState,
    ReflectionPrompts,
)
```

Everything else (`ReflectionResult`, `_create_evaluator_agent`,
`_create_fallback_evaluation`, `_extract_response_text`, …) is implementation
detail and not re-exported from the package root. Note that
`ReflectionResult` is importable from `upsonic.reflection.models` if you
need to reference its type — pipeline code does exactly that:
`from .models import …, ReflectionResult`.

### 6.1 Typical usage

```python
from upsonic import Agent, Task
from upsonic.reflection import ReflectionConfig

agent = Agent(
    name="Researcher",
    model="openai/gpt-4o",
    reflection_config=ReflectionConfig(
        max_iterations=4,
        acceptance_threshold=0.85,
        evaluator_model="anthropic/claude-3-5-sonnet-latest",  # cross-provider eval
    ),
)

# Or, the shortcut: reflection=True with default config (3 iters, 0.8 threshold)
# agent = Agent(..., reflection=True)

task = Task(description="Summarise the attached PDF in three bullet points.")
result = await agent.do_async(task)
print(result.output)
print(result.usage)   # token totals include evaluator + improvement passes
```

### 6.2 Direct (advanced) usage

You can also call the processor outside of an `Agent.do_async` pipeline —
useful for retrofitting reflection onto an external response, unit-testing
the rubric, or building a custom evaluator harness:

```python
from upsonic.reflection import ReflectionProcessor, ReflectionConfig

proc = ReflectionProcessor(ReflectionConfig(max_iterations=2))
reflection_result = await proc.process_with_reflection(
    agent=my_main_agent,
    task=my_task,
    initial_response="The capital of France is Berlin.",
)
print(reflection_result.improved_output)
print(reflection_result.termination_reason)
print(reflection_result.final_evaluation.overall_score)
```

## 7. Integration with the rest of Upsonic

The reflection package is owned by `Agent` and consumed by the agent
**pipeline** as a single `Step`.

### 7.1 `Agent` constructor wiring

In `src/upsonic/agent/agent.py`, `Agent.__init__` accepts both a
`reflection: bool` flag and a `reflection_config: Optional[ReflectionConfig]`
keyword. The wiring is:

```python
# src/upsonic/agent/agent.py
if reflection and not reflection_config:
    from upsonic.reflection import ReflectionConfig
    reflection_config = ReflectionConfig()

self.reflection_config = reflection_config
if reflection_config:
    from upsonic.reflection import ReflectionProcessor
    self.reflection_processor = ReflectionProcessor(reflection_config)
else:
    self.reflection_processor = None
```

The truthy-check `agent.reflection_processor and agent.reflection` inside
`ReflectionStep.execute` is what gates the entire feature on or off at
pipeline runtime.

### 7.2 `ReflectionStep` (pipeline integration)

Defined in `src/upsonic/agent/pipeline/steps.py` (`class ReflectionStep(Step)`),
registered in `src/upsonic/agent/pipeline/__init__.py`, and slotted into the
default pipeline order around step #15–16 (after the main model call /
tool-result post-processing, before reliability and call-management).

| Pipeline check / branch                       | What `ReflectionStep` does                                                                            |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `agent.run_id` is cancelled                   | `raise_if_cancelled` propagates `RunCancelledException`.                                              |
| Reflection not configured on the agent        | Returns a no-op `StepResult("Reflection not enabled")`, optionally yields a streaming event.          |
| `task._cached_result`                         | Skipped (`"Skipped due to cache hit"`).                                                               |
| `task._policy_blocked`                        | Skipped (`"Skipped due to policy block"`).                                                            |
| `task.is_paused`                              | Skipped (`"Skipped due to external pause"`).                                                          |
| Otherwise                                     | Calls `agent.reflection_processor.process_with_reflection(agent, task, context.output)`.              |

Once the processor returns, `ReflectionStep`:

1. Folds `reflection_result.sub_agent_usage` into the parent
   `AgentRunOutput.usage` via `context._ensure_usage().incr(...)`.
2. Builds two pieces of chat history (so reflection is auditable by anything
   reading `chat_history`):
   - `ModelRequest(parts=[UserPromptPart(content=evaluation_prompt)])`
   - `ModelResponse(parts=[TextPart(content=str(improved_output))])`
3. Updates `context.response`, `context.output`, and `task._response` so that
   downstream steps (and the eventual `AgentRunOutput`) see the
   *post-reflection* answer.
4. If streaming, yields a `reflection` event via
   `ayield_reflection_event(...)` carrying `reflection_applied`,
   `improvement_made`, and `original_preview` / `improved_preview`.

This is why `ReflectionResult` deliberately exposes `evaluation_prompt`
(the *first* evaluation prompt) and `improved_output` (the *last* output) as
top-level fields: they are the canonical FIRST INPUT / LAST OUTPUT pair the
pipeline writes into chat history for that step.

### 7.3 `AutonomousAgent` parity

`src/upsonic/agent/autonomous_agent/autonomous_agent.py` accepts the same
`reflection_config: Optional[ReflectionConfig]` keyword in its constructor,
giving prebuilt autonomous agents the same reflection capability via the
same configuration surface.

### 7.4 Logging hooks

`processor.py` calls four helpers from `upsonic.utils.printing`:

| Helper                              | Called when                                           |
| ----------------------------------- | ----------------------------------------------------- |
| `reflection_started(...)`           | First entry into `process_with_reflection`.           |
| `reflection_evaluation(...)`        | After each evaluator-agent call (passes scores+action). |
| `reflection_improvement_started(...)` | Just before each `_generate_improved_response`.       |
| `reflection_completed(...)`         | After the loop exits; passes final score, total iterations, termination reason. |

These integrate with Upsonic's existing CLI/printing layer so that
reflection is observable in the same place every other agent stage is.

## 8. End-to-end flow

The full life of one reflection cycle, from the moment the main LLM
finishes producing `context.output` to the moment `AgentRunOutput.output`
contains the improved answer:

```
                         ┌───────────────────────────────────────────────┐
                         │ Pipeline reaches ReflectionStep.execute(...)  │
                         └───────────────────────────────────────────────┘
                                              │
                                              ▼
                       ┌─────────────────────────────────────────────────┐
                       │ Guard checks                                    │
                       │   • agent.reflection_processor truthy?          │
                       │   • task not cached / blocked / paused?         │
                       └─────────────────────────────────────────────────┘
                                              │ (yes)
                                              ▼
              ┌─────────────────────────────────────────────────────────────────┐
              │ ReflectionProcessor.process_with_reflection(agent, task, output)│
              └─────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────┐
                    │ state = ReflectionState()                       │
                    │ current = _extract_response_text(initial_resp)  │
                    │ evaluator = _create_evaluator_agent(agent)      │
                    │ first_eval_prompt = EVALUATION_PROMPT.format()  │
                    │ aggregated_usage = RunUsage()                   │
                    └─────────────────────────────────────────────────┘
                                              │
                                              ▼
              ┌─────────────────────────────────────────────────────────────────┐
              │ while state.should_continue(config):                            │
              │                                                                 │
              │   ┌─────────────────────────────────────────────────────────┐   │
              │   │ EVALUATE                                                │   │
              │   │   eval_task = Task(description=EVALUATION_PROMPT,       │   │
              │   │                    response_format=EvaluationResult,    │   │
              │   │                    not_main_task=True)                  │   │
              │   │   out = await evaluator.do_async(eval_task,             │   │
              │   │                                  return_output=True)    │   │
              │   │   evaluation = out.output                               │   │
              │   │   aggregated_usage.incr(out.usage)                      │   │
              │   │   reflection_evaluation(...)                            │   │
              │   │   state.add_evaluation(current, evaluation)             │   │
              │   └─────────────────────────────────────────────────────────┘   │
              │                                                                 │
              │   if evaluation.overall_score >= acceptance_threshold:          │
              │       state.final_response = current                            │
              │       termination = "acceptance_threshold_met"; break           │
              │                                                                 │
              │   match evaluation.action:                                      │
              │       ACCEPT     → final = current; "evaluator_accepted"; break │
              │       CLARIFY    → final = current; "clarification_needed"; brk │
              │       REVISE/RETRY ↓                                            │
              │                                                                 │
              │   ┌─────────────────────────────────────────────────────────┐   │
              │   │ IMPROVE                                                 │   │
              │   │   reflection_improvement_started(feedback=…)            │   │
              │   │   improved_task = Task(description=IMPROVEMENT_PROMPT,  │   │
              │   │                        response_format=task.rf,         │   │
              │   │                        tools=task.tools,                │   │
              │   │                        attachments=task.attachments,    │   │
              │   │                        not_main_task=True)              │   │
              │   │   out2 = await agent.do_async(improved_task,            │   │
              │   │                                return_output=True)      │   │
              │   │   aggregated_usage.incr(out2.usage)                     │   │
              │   │   if out2.output is None:                               │   │
              │   │       termination = "improvement_failed"; break         │   │
              │   │   current = _extract_response_text(out2.output)         │   │
              │   └─────────────────────────────────────────────────────────┘   │
              │ (loop)                                                          │
              └─────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────┐
                    │ if state.final_response is None:                │
                    │     state.final_response = current              │
                    │     termination = "max_iterations_reached"      │
                    │ reflection_completed(final_score, iters, …)     │
                    └─────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────┐
                    │ final_output = _convert_to_response_format(     │
                    │     state.final_response, initial_response, t)  │
                    │ improvement_made = original != final            │
                    │ return ReflectionResult(                        │
                    │     evaluation_prompt = first_eval_prompt,      │
                    │     improved_output   = final_output,           │
                    │     improvement_made,                           │
                    │     original_output  = initial_response,        │
                    │     final_evaluation = state.evaluations[-1],   │
                    │     termination_reason = state.terminated_reason│
                    │     sub_agent_usage = aggregated_usage,         │
                    │ )                                               │
                    └─────────────────────────────────────────────────┘
                                              │
                                              ▼
              ┌─────────────────────────────────────────────────────────────────┐
              │ ReflectionStep — post-processing                                │
              │   context._ensure_usage().incr(reflection_result.sub_agent_usage)│
              │   chat_history.append(ModelRequest(eval_prompt))                │
              │   chat_history.append(ModelResponse(improved_output))           │
              │   context.response = ModelResponse(...)                         │
              │   context.output   = improved_output                            │
              │   task._response   = improved_output                            │
              │   if streaming: ayield_reflection_event(...)                    │
              │   return StepResult("Reflection applied", COMPLETED)            │
              └─────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                       ┌─────────────────────────────────────────────────┐
                       │ Pipeline continues with ReliabilityStep,        │
                       │ CallManagementStep, … final AgentRunOutput.     │
                       └─────────────────────────────────────────────────┘
```

### 8.1 Termination matrix

The five possible exit paths and their semantics:

| `termination_reason`           | Trigger                                                                       | `improvement_made` likely | Surface to user                                                  |
| ------------------------------ | ----------------------------------------------------------------------------- | ------------------------- | ---------------------------------------------------------------- |
| `acceptance_threshold_met`     | `evaluation.overall_score >= config.acceptance_threshold`                     | True or False             | "Reflection accepted by score."                                  |
| `evaluator_accepted`           | Evaluator returned `ReflectionAction.ACCEPT` regardless of numeric score      | Often False               | "Reflection accepted by evaluator action."                       |
| `clarification_needed`         | Evaluator returned `ReflectionAction.CLARIFY`                                 | False                     | Caller should ask the user a follow-up question.                 |
| `improvement_failed`           | A `_generate_improved_response` call raised or returned `None`                | False                     | Last good response is kept; failure logged.                      |
| `max_iterations_reached`       | Loop exited via `should_continue` returning `False` on the iteration count    | True or False             | Best-effort answer after `config.max_iterations` cycles.          |

### 8.2 Token-cost accounting

All evaluator and improvement sub-runs go through the standard
`agent.do_async(..., return_output=True)` path, so they emit normal
`RunUsage` objects. `process_with_reflection` aggregates these into
`ReflectionResult.sub_agent_usage`, and `ReflectionStep` then folds that
aggregate into the parent run's usage. As a result, **`AgentRunOutput.usage`
and `agent.cost` already include the reflection overhead** — there is no
hidden token cost.

### 8.3 Why FIRST INPUT / LAST OUTPUT?

The pipeline's chat-history convention is "for any agentic step, store one
request/response pair: the very first prompt that started the step, and the
very last output that ended it." Reflection internally runs `1..N`
evaluation+improvement pairs, so the processor explicitly:

- captures the **first** evaluation prompt (`first_evaluation_prompt`)
  *before* entering the loop, and
- emits the **last** improved output (`state.final_response` →
  `final_output`)

…via `ReflectionResult.evaluation_prompt` / `ReflectionResult.improved_output`.
`ReflectionStep` then writes exactly that pair into `context.chat_history`,
keeping reflection consistent with every other pipeline step's audit trail.
