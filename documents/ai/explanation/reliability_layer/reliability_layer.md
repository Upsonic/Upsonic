---
name: reliability-layer
description: Use when working on Upsonic's post-generation hallucination defense pipeline that runs verifier and editor sub-agents on agent output. Use when a user asks to enable, configure, debug, or extend the reliability layer, add a new verifier (URL/number/code/information/citation), tune `prevent_hallucination` levels, understand how `task._response` gets cleaned, or trace usage accounting for verifier sub-agents. Trigger when the user mentions reliability_layer, ReliabilityProcessor, ReliabilityStep, ReliabilityManager, ValidationPoint, ValidationResult, SourceReliability, prevent_hallucination, hallucination prevention, verifier agent, editor agent, url_validation_prompt, number_validation_prompt, code_validation_prompt, information_validation_prompt, editor_task_prompt, strip_context_tags, find_urls_in_text, find_numbers_in_text, find_code_in_text, _reliability_sub_agent_usage, or trusted source vs untrusted source verification.
---

# `src/upsonic/reliability_layer/` — Production Reliability via Verifier and Editor Agents

## 1. What This Folder Is

The `reliability_layer/` package implements Upsonic's **post-generation reliability pipeline**, a defense layer that runs *after* an `Agent` produces a response and *before* the response is handed back to the caller. Its purpose is to detect and remove **hallucinations** in agent output — fabricated URLs, invented numbers, untraceable claims, and synthesized code — by re-executing a constellation of small **verifier sub-agents** against the original task context and then routing any flagged content through an **editor sub-agent** that scrubs (or nulls out) suspicious values.

Conceptually it is an iterative, agent-on-agent quality improvement round:

| Concept | What It Does |
|---|---|
| **Verifier agents** | Independent specialist agents that each check one dimension of the answer (URLs, numbers, information, code) against the trusted source context. |
| **Editor agent** | A sanitizer agent that, when *any* verifier raises suspicion, rewrites the final response — replacing suspicious fields with `None` while preserving everything else. |
| **`ReliabilityProcessor`** | The static orchestrator that wires verifiers, editor, regex pre-filters, and usage accounting together. |
| **`ReliabilityStep`** | The pipeline step (in `agent/pipeline/steps.py`) that invokes the processor as part of every agent run. |
| **`ReliabilityManager`** | The thin async context-manager wrapper (in `agent/context_managers/reliability_manager.py`) used by the step. |

The whole layer is **opt-in**: an agent without a `reliability_layer` argument simply passes through. When enabled (typically via a config object exposing `prevent_hallucination >= 10`), every successful response is fan-out validated, and only obviously-suspicious responses pay the editor-rewrite cost.

The folder itself is intentionally tiny — a single file — because all the complexity is concentrated into one disciplined module. Everything else lives outside the folder, in the agent pipeline.

---

## 2. Folder Layout

```
src/upsonic/reliability_layer/
└── reliability_layer.py     # Sole source file — entire reliability subsystem
```

That is the entire tree. There are no subpackages, no `__init__.py` re-exports, no helper modules. Consumers reach in directly:

```python
from upsonic.reliability_layer.reliability_layer import ReliabilityProcessor
```

For context, here is how external code references it:

```
src/upsonic/
├── reliability_layer/
│   └── reliability_layer.py            # ReliabilityProcessor + helpers + prompts
├── agent/
│   ├── agent.py                        # Agent.reliability_layer attribute (line ~471)
│   ├── pipeline/
│   │   └── steps.py                    # ReliabilityStep (line 2494) — pipeline integration
│   └── context_managers/
│       └── reliability_manager.py      # ReliabilityManager — thin async wrapper
```

---

## 3. Top-Level File: `reliability_layer.py`

This single file is structured into six logical zones. They are listed below in source order.

### 3.1 Zone Map

| Lines (approx.) | Zone | Responsibility |
|---|---|---|
| 1–11 | Imports & `TYPE_CHECKING` | Lazy-imports `Task` to avoid circulars; pulls `Model`, `BaseModel`, regex, asyncio. |
| 13–42 | `strip_context_tags` | Pre-processing helper that removes Upsonic's structural XML-ish framing (`<Context>`, `<Tasks>`, `<Knowledge Base>`, etc.) so the verifiers see only raw text. |
| 45–118 | Validation prompts | Five string constants — one per verifier and one for the editor. |
| 120–142 | `SourceReliability`, `ValidationPoint`, `ValidationResult` | Pydantic models used as **structured outputs** from the verifier agents. |
| 144–202 | `ValidationResult.calculate_suspicion` | The roll-up: aggregates every verifier's verdict into an overall feedback string and a single boolean. |
| 204–464 | `ReliabilityProcessor` | The orchestrator; class with `process_task` static method. |
| 466–528 | Regex pre-filters | `find_urls_in_text`, `find_numbers_in_text`, `find_code_in_text`, and their batch counterparts (`contains_*`). |

### 3.2 `strip_context_tags(text: str) -> str`

Upsonic's task pipeline injects framing tags around context blocks. When that context is forwarded to a verifier, the tags inflate the prompt and confuse pattern recognition. This helper performs an idempotent regex pass that removes both opening and closing instances of:

- `<Context>` / `</Context>`
- `<Knowledge Base>` / `</Knowledge Base>`
- `<Agents>` / `</Agents>`
- `<Tasks>` / `</Tasks>`
- `<Default Prompt>` / `</Default Prompt>`

After tag removal it collapses runs of blank lines (`\n\s*\n` → `\n\n`) and trims. Any non-string input is returned untouched, so it is safe to call inside a list comprehension over heterogeneous context items.

### 3.3 Validation Prompts (the **rubric** for verifiers)

Four prompts, all of which share the same anti-hallucination thesis:

> *"Check if the source comes from the content. Don't make assumptions, just check the context and try to find exact things. If not, flag it. If you can see the things in the context, everything is okay (Trusted Source)."*

| Prompt constant | Verifier role |
|---|---|
| `url_validation_prompt` | Confirms every URL in the response actually appeared verbatim in the trusted context. |
| `number_validation_prompt` | Same deal for numbers — currencies, percentages, scientific notation. |
| `code_validation_prompt` | Same for code blocks, function names, etc. |
| `information_validation_prompt` | General-purpose claim verification — covers anything the other three missed. |

All four are written to push the verifier toward **conservative refusal**: an unverifiable item must be flagged, not justified. The editor prompt below is what cleans up the consequences.

### 3.4 `editor_task_prompt`

A single prompt template with one `{validation_feedback}` slot. Its rules are deliberately rigid:

```text
Processing Rules:
1. For ANY suspicious content identified in validation:
   - Replace the suspicious value with None
   - Do not suggest alternatives
   - Do not provide explanations
   - Do not modify other parts of the content

2. For non-suspicious content:
   - Keep the original value unchanged
   - Do not enhance or modify
   - Do not add additional information
```

The editor never *fixes* hallucinations (you cannot fix something that is fabricated). It deletes them. This is a critical invariant: the layer is a **subtractive filter**, not a corrective rewriter.

### 3.5 Pydantic Models

```python
class SourceReliability(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class ValidationPoint(BaseModel):
    is_suspicious: bool
    feedback: str
    suspicious_points: list[str]
    source_reliability: SourceReliability = SourceReliability.UNKNOWN
    verification_method: str = ""
    confidence_score: float = 0.0

class ValidationResult(BaseModel):
    url_validation: ValidationPoint
    number_validation: ValidationPoint
    information_validation: ValidationPoint
    code_validation: ValidationPoint
    any_suspicion: bool
    suspicious_points: list[str]
    overall_feedback: str
    overall_confidence: float = 0.0
```

Each verifier agent is configured with `response_format=ValidationPoint`, so it is forced to return a structured JSON object; the orchestrator then plugs each `ValidationPoint` into the corresponding slot of a single `ValidationResult`.

#### `ValidationResult.calculate_suspicion()`

A pure aggregator — no LLM calls. It:

1. Sets `any_suspicion = OR` of all four `is_suspicious` flags.
2. Concatenates each verifier's `suspicious_points` into a flat list.
3. For each suspicious verifier, builds a "Header + bullet list" feedback section.
4. Computes `overall_confidence = mean(four confidence_scores)`.
5. Joins everything into a "Validation Summary" multi-line string and **returns** it. (Side effect *and* return: the caller stores it on the model and also passes the returned string into the editor.)

### 3.6 `ReliabilityProcessor` (the orchestrator)

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, confidence_threshold: float = 0.7)` | Threshold field is stored but not currently consulted by `process_task`. Reserved for future "soft suspicion" gating. |
| `process_task` | `static async (task, reliability_layer=None, model=None) -> Task` | The single externally-called entry point. Mutates and returns `task` with a possibly-cleaned `task._response`. |

The control flow inside `process_task` is described in detail in section 8 below.

### 3.7 Regex Pre-Filters

Before spending tokens on a verifier, the orchestrator asks: *does the response actually contain anything of this kind?* If not, the verifier is short-circuited with a synthetic "no X found" `ValidationPoint` (confidence 1.0).

| Helper | Pattern (summary) | Returns |
|---|---|---|
| `find_urls_in_text(text)` | `http[s]?://...` (also matches `ftp://`) | `List[str]` of URL matches |
| `find_numbers_in_text(text)` | Integers, floats, percentages, currencies (`$1.99`), scientific notation (`1e5`) | `List[str]` of number matches |
| `find_code_in_text(text)` | Code fences, inline backticks, keywords (`def`/`class`/`import`/...), brackets, `f(x)`-shaped tokens, `obj.method` | `bool` |
| `contains_urls(texts)` / `contains_numbers(texts)` / `contains_code(texts)` | List variants. Skip non-strings. | `bool` |

Note: there is **no** `find_information_in_text` — information is always a candidate, so `information_validation` always runs.

---

## 4. Subfolders Walked Through

There are no subfolders. The package is a single Python module. This is a deliberate choice: the reliability layer has one job, and it is implemented in one file so the prompts, the data model, the orchestrator, and the helper regexes are all reviewable on a single screen scroll.

If subfolders were added in the future, the natural divisions would be:

| Hypothetical subfolder | What would live there |
|---|---|
| `prompts/` | The four validation prompts and the editor prompt as `.txt` templates. |
| `models/` | `SourceReliability`, `ValidationPoint`, `ValidationResult`. |
| `detectors/` | The regex pre-filter functions. |
| `processors/` | `ReliabilityProcessor` itself. |

But until additional reliability strategies are added (e.g. citation-grounded checking, tool-call replay, formal verification), the flat layout is the right call.

---

## 5. Cross-File Relationships

The reliability layer is **not** a self-contained library — it is invoked by, and contributes data to, several other Upsonic subsystems.

```
┌───────────────────────────────────────────────────────────────────────┐
│  Agent.do_async / Agent.run                                            │
│    └── pipeline.run_step(...)                                          │
│         ├── ... (other steps) ...                                      │
│         └── ReliabilityStep            (src/upsonic/agent/pipeline/    │
│              │                          steps.py — line 2494)          │
│              │                                                         │
│              ▼                                                         │
│         ReliabilityManager             (src/upsonic/agent/context_     │
│              │                          managers/reliability_manager.py│
│              │                                                         │
│              ▼                                                         │
│   ReliabilityProcessor.process_task    (THIS FOLDER)                   │
│              │                                                         │
│              ├──► spawns AgentConfiguration("URL Validation Agent")    │
│              ├──► spawns AgentConfiguration("Number Validation Agent") │
│              ├──► spawns AgentConfiguration("Information Validation A.")│
│              ├──► spawns AgentConfiguration("Code Validation Agent")   │
│              │      (asyncio.gather'd in parallel)                     │
│              │                                                         │
│              └──► if any suspicion:                                    │
│                    spawns AgentConfiguration("Information Editor Agent")│
│                    rewrites task._response                             │
└───────────────────────────────────────────────────────────────────────┘
```

### 5.1 Inputs the orchestrator pulls from outside

| Source | What is read | Why |
|---|---|---|
| `task.response` / `task.response.output` | Current AI answer | The "untrusted" payload to verify. |
| `task.description` | Original user prompt | Restated to the verifier as "Given Task: ...". |
| `task.context` (str or list) | User-provided trusted material | Forms the "Trusted Source" block. |
| `task.response_format` | Pydantic schema | Reminds verifier of the requested shape; reused unchanged when the editor reissues. |
| `task.images`, `task.tools`, `task.price_id` | Multimodal & cost plumbing | Re-attached to verifier and editor sub-tasks so cost tracking and tool availability are preserved. |
| `reliability_layer.prevent_hallucination` | Integer level (0 = off, 10 = full) | Gates the entire pipeline. |
| `model` (parameter) | LLM to use | Each verifier and the editor use *the same* model as the parent agent. |

### 5.2 Outputs the orchestrator writes back

| Destination | What is written |
|---|---|
| `task._response` | Cleaned response from the editor (only when `any_suspicion=True`). |
| `task._reliability_sub_agent_usage` | A `RunUsage` accumulator with token/cost stats for every verifier and the editor. The pipeline step later folds this into the parent context. |

### 5.3 Files that import from this folder

```
src/upsonic/agent/context_managers/reliability_manager.py
    from upsonic.reliability_layer.reliability_layer import ReliabilityProcessor
```

That is the only direct import in the codebase. Every other interaction is mediated by the manager. The pipeline step (`ReliabilityStep`) imports `ReliabilityManager`, never `ReliabilityProcessor` directly.

---

## 6. Public API

The folder exposes a small, intentionally narrow surface. Only `ReliabilityProcessor.process_task` is meant to be called by code outside the module; everything else is internal but is documented here because it is freely importable.

### 6.1 `ReliabilityProcessor`

```python
class ReliabilityProcessor:
    def __init__(self, confidence_threshold: float = 0.7) -> None: ...

    @staticmethod
    async def process_task(
        task: "Task",
        reliability_layer: Optional[Any] = None,
        model: Optional[Union[Model, str]] = None,
    ) -> "Task":
        """
        Mutate `task` so that any verifiable hallucinations are removed.

        Behaviour matrix
        -----------------
        reliability_layer is None         -> return task unchanged
        prevent_hallucination == 0        -> return task unchanged
        prevent_hallucination == 10       -> run full verifier+editor pipeline
        other positive values             -> reserved (currently no-op)

        Returns the same `task` instance. On suspicion, `task._response`
        is replaced with the editor's cleaned response.
        """
```

### 6.2 Pydantic models

```python
class SourceReliability(Enum):
    HIGH | MEDIUM | LOW | UNKNOWN

class ValidationPoint(BaseModel):
    is_suspicious: bool
    feedback: str
    suspicious_points: list[str]
    source_reliability: SourceReliability
    verification_method: str
    confidence_score: float

class ValidationResult(BaseModel):
    url_validation: ValidationPoint
    number_validation: ValidationPoint
    information_validation: ValidationPoint
    code_validation: ValidationPoint
    any_suspicion: bool
    suspicious_points: list[str]
    overall_feedback: str
    overall_confidence: float

    def calculate_suspicion(self) -> str: ...
```

### 6.3 Free functions (regex layer)

```python
def strip_context_tags(text: str) -> str: ...

def find_urls_in_text(text: str) -> List[str]: ...
def find_numbers_in_text(text: str) -> List[str]: ...
def find_code_in_text(text: str) -> bool: ...

def contains_urls(texts: List[str]) -> bool: ...
def contains_numbers(texts: List[str]) -> bool: ...
def contains_code(texts: List[str]) -> bool: ...
```

### 6.4 Module-level prompt constants (string-typed)

```
url_validation_prompt
number_validation_prompt
information_validation_prompt
code_validation_prompt
editor_task_prompt          # contains a `{validation_feedback}` slot
```

These are public in the sense that nothing prevents importing them, but they are part of the implementation contract and may change. Consumers should treat them as opaque.

---

## 7. Integration With the Rest of Upsonic

### 7.1 How it slots into the agent pipeline

`ReliabilityStep` is enrolled as one fixed step in the canonical agent pipeline declared in `agent/agent.py`:

```python
# src/upsonic/agent/agent.py — excerpt of the canonical pipeline
ReliabilityStep(),             # step 18 in standard pipeline
# ...
ReliabilityStep(),             # step 16 in reflection-enabled pipeline
                               #     <-- "Verify and clean output"
```

In the streaming variant the step *also* emits a `reliability` event (via `ayield_reliability_event`) so downstream observers know whether modifications were made.

### 7.2 How it is enabled

An `Agent` is constructed with an optional `reliability_layer` argument (any object with a `prevent_hallucination` integer attribute):

```python
class HighReliability:
    prevent_hallucination = 10  # full pipeline ON

agent = Agent(
    model="openai/gpt-4o",
    reliability_layer=HighReliability(),
)
```

The same plumbing flows through `AutonomousAgent.__init__`, which forwards `reliability_layer` to its inner `Agent` (see `agent/autonomous_agent/autonomous_agent.py`).

### 7.3 What the pipeline step does around the processor

```python
# src/upsonic/agent/pipeline/steps.py (excerpt, ReliabilityStep.execute)

if not agent.reliability_layer:
    # no-op + emit "reliability_applied=False" if streaming
    return StepResult(..., status=COMPLETED, message="No reliability layer")

if task._cached_result:
    return StepResult(..., message="Skipped due to cache hit")
if task._policy_blocked:
    return StepResult(..., message="Skipped due to policy block")

original_output = context.output

reliability_manager = ReliabilityManager(task, agent.reliability_layer, model)
await reliability_manager.aprepare()
try:
    processed_task = await reliability_manager.process_task(task)
    task = processed_task
    context.output = processed_task.response
finally:
    await reliability_manager.afinalize()

# Fold sub-agent usage back into the parent run
reliability_usage = getattr(task, '_reliability_sub_agent_usage', None)
if reliability_usage is not None:
    usage = context._ensure_usage()
    usage.incr(reliability_usage)
    task._reliability_sub_agent_usage = None
```

Three things to notice:

1. **Cache and policy blocks bypass the layer.** If the task already returned a cached answer or was blocked by an `AgentPolicyStep`, there is no fresh content to verify — running it would be wasted tokens.
2. **Usage telemetry is preserved.** Verifier and editor calls accumulate into `task._reliability_sub_agent_usage` (a `RunUsage`); the step folds that back into the run-level usage counter so cost reporting stays honest.
3. **The original output is captured before processing**, so the step can compute `modifications_made = str(original) != str(processed)` for streaming events and debug logs.

### 7.4 Telemetry

When `agent.debug_level >= 2`, the step emits a `debug_log_level2` entry tagged `"ReliabilityStep"` containing:

```python
{
  "modifications_made": bool,
  "original_output_preview": str[:300],
  "processed_output_preview": str[:300],
  "reliability_layer_type": type(agent.reliability_layer).__name__,
}
```

This is the primary observability hook for diagnosing false positives.

---

## 8. End-to-End Flow

A complete walk-through of one invocation, from the moment `ReliabilityStep.execute` calls into `ReliabilityProcessor.process_task` (`prevent_hallucination = 10` path):

### 8.1 Phase A — Bail-out checks

```python
if reliability_layer is None:
    return task                                   # no layer attached
prevent_hallucination = getattr(reliability_layer, 'prevent_hallucination', 0)
if prevent_hallucination == 0:
    return task                                   # explicitly disabled
if prevent_hallucination != 10:
    return task                                   # only level 10 is wired up
```

### 8.2 Phase B — Snapshot and unwrap the response

```python
copy_task = deepcopy(task)
copy_task._response = task.response

# Try to extract the raw `output` rather than the wrapper
old_task_output = task.response
if hasattr(task.response, 'output'):
    old_task_output = task.response.output
elif hasattr(task.response, 'model_dump'):
    old_task_output = task.response.model_dump()
```

The deepcopy ensures the original `task` is preserved while the processor builds context for the verifiers.

### 8.3 Phase C — Build the shared context bundle

The same context block is shipped to every verifier:

```python
context_strings = [f"Given Task: {copy_task.description}"]

if copy_task.context:
    items = copy_task.context if isinstance(copy_task.context, list) else [copy_task.context]
    if copy_task.response_format:
        items.append(copy_task.response_format)

    for item in items:
        if isinstance(item, str):
            cleaned = strip_context_tags(item)
            context_strings.append(
                f"Context That Came From User (Trusted Source): "
                f"```User given context {cleaned}```"
            )
        elif <pydantic BaseModel subclass>:
            context_strings.append(
                f"User requested output: ```Requested Output {item.model_fields}```"
            )

context_strings.append(
    f"Current AI Response (Untrusted Source, last AI response that we are checking now): "
    f"{old_task_output}"
)

context_strings = [strip_context_tags(s) if isinstance(s, str) else s
                   for s in context_strings]
```

This is the heart of the trust model: user-supplied context is labelled **Trusted Source**, the AI's own output is labelled **Untrusted Source**, and the verifier is told to check that one is grounded in the other.

### 8.4 Phase D — Spawn verifier sub-agents

For each of the four validation types:

| Step | Detail |
|---|---|
| Build a named `Agent` | `AgentConfiguration(model=model, name=f"{validation_type.title()} Agent")`. Same model as the parent. |
| Pre-filter | If `validation_type` is `url_validation` / `number_validation` / `code_validation`, run the corresponding `contains_*` regex over `[prompt] + context_strings`. If nothing of that kind exists, write a synthetic `ValidationPoint(is_suspicious=False, confidence_score=1.0, verification_method="regex_*_detection")` directly into `validation_result` and `continue`. **Information validation has no pre-filter and always runs.** |
| Build verifier task | `Task(prompt, response_format=ValidationPoint, context=context_strings, images=task.images, tools=task.tools, price_id_=task.price_id, not_main_task=True)` |
| Append to `validation_tasks`, `validation_types` | Will be gathered next. |

### 8.5 Phase E — Run verifiers in parallel

```python
validation_coroutines = [
    validator_agents[vt].do_async(validation_tasks[i], return_output=True)
    for i, vt in enumerate(validation_types)
]
validation_outputs = await asyncio.gather(*validation_coroutines)

# Usage accounting
from upsonic.usage import RunUsage
if task._reliability_sub_agent_usage is None:
    task._reliability_sub_agent_usage = RunUsage()
for v_output in validation_outputs:
    if hasattr(v_output, 'usage') and v_output.usage:
        task._reliability_sub_agent_usage.incr(v_output.usage)

for i, vt in enumerate(validation_types):
    setattr(validation_result, vt, validation_tasks[i].response)
```

`asyncio.gather` is used rather than sequential awaits — the four verifiers are independent, so the wall-clock cost is roughly *one* LLM call, not four.

### 8.6 Phase F — Aggregate

```python
validation_result.calculate_suspicion()
```

This populates `any_suspicion`, the flat `suspicious_points` list, the human-readable `overall_feedback`, and the averaged `overall_confidence`.

| `any_suspicion` | Action |
|---|---|
| `False` | Return `task` unchanged. The verifiers signed off. |
| `True` | Continue to Phase G — invoke the editor. |

### 8.7 Phase G — Editor rewrite

```python
editor_agent = AgentConfiguration(model=model, name="Information Editor Agent")

formatted_prompt = editor_task_prompt.format(
    validation_feedback=validation_result.overall_feedback
)
formatted_prompt += (
    f"\n\nORIGINAL AI RESPONSE TO CLEAN:\n{old_task_output}\n\n"
    f"Return the cleaned version of this response in the same format "
    f"(not as code or XML):"
)

cleaned_context = []
if copy_task.context:
    for item in copy_task.context:
        if isinstance(item, str):
            cleaned_context.append(strip_context_tags(item))
        else:
            cleaned_context.append(item)

editor_task = Task(
    formatted_prompt,
    images=task.images,
    context=[copy_task.response_format, validation_result] + cleaned_context,
    response_format=task.response_format,    # SAME schema as the parent task
    tools=task.tools,
    price_id_=task.price_id,
    not_main_task=True,
)

editor_output = await editor_agent.do_async(editor_task, return_output=True)

# Editor usage accounting
if hasattr(editor_output, 'usage') and editor_output.usage:
    task._reliability_sub_agent_usage.incr(editor_output.usage)

task._response = editor_task.response
return task
```

Three load-bearing details:

1. **`response_format` is preserved** — the cleaned output must still match the caller's Pydantic schema, so the editor receives the same schema and the original Pydantic model class as a context element.
2. **The `ValidationResult` is part of the editor's context** — so the editor sees not just *what* to clean but *why* (per-field `suspicious_points`, the verification feedback).
3. **`not_main_task=True`** — both verifier and editor sub-tasks are tagged so other parts of the framework (telemetry, callbacks) can distinguish reliability-internal calls from the user's main task.

### 8.8 Phase H — Pipeline returns the cleaned task

The processor returns `task`. The pipeline step then:

```python
context.output = processed_task.response                  # set the new output
usage.incr(task._reliability_sub_agent_usage)             # roll up costs
task._reliability_sub_agent_usage = None                  # clear marker
modifications_made = str(original_output) != str(context.output)
# emit reliability event / debug log
```

…and the agent run continues to whatever steps come after (typically result formatting and finalization).

### 8.9 Sequence diagram

```
ReliabilityStep        ReliabilityManager      ReliabilityProcessor    Verifier×4         Editor
      │                       │                         │                  │                 │
      │── aprepare() ────────►│                         │                  │                 │
      │                       │                         │                  │                 │
      │── process_task ──────►│── process_task ────────►│                  │                 │
      │                       │                         │                                    │
      │                       │                         │── strip_context_tags / regex ──    │
      │                       │                         │                                    │
      │                       │                         │── Task(URL)  ────►Agent.do_async ──┤
      │                       │                         │── Task(Num)  ────►Agent.do_async ──┤
      │                       │                         │── Task(Info) ────►Agent.do_async ──┤  asyncio.gather
      │                       │                         │── Task(Code) ────►Agent.do_async ──┤
      │                       │                         │                                    │
      │                       │                         │◄── 4×ValidationPoint ──────────────┤
      │                       │                         │                                    │
      │                       │                         │── calculate_suspicion()            │
      │                       │                         │                                    │
      │                       │                         │── if any_suspicion: ───────────────►│
      │                       │                         │                                    │── do_async
      │                       │                         │◄── cleaned response ───────────────│
      │                       │                         │                                    │
      │                       │◄── task (mutated) ──────│                                    │
      │◄── processed_task ────│                         │                                    │
      │── afinalize() ───────►│                         │                                    │
      │                       │                         │                                    │
      │── fold usage / emit reliability event ─────────────────────────────────────────────► │
```

### 8.10 Cost characteristics

| Scenario | LLM calls (worst case) | Notes |
|---|---|---|
| `reliability_layer = None` | 0 | Step is a no-op. |
| `prevent_hallucination = 0` | 0 | Processor is a no-op. |
| Level 10, response has only prose | 1 (information verifier only) | URL/number/code regex pre-filters short-circuit. |
| Level 10, prose + numbers | 2 (information + number) | |
| Level 10, mixed content, **clean** | up to 4 verifiers, in parallel | Editor not invoked. |
| Level 10, mixed content, **suspicious** | up to 4 verifiers + 1 editor | Editor adds one final sequential call. |

Every one of those calls contributes to `task._reliability_sub_agent_usage`, which the pipeline step then merges into the parent run's usage so the user sees a single accurate cost number.

### 8.11 Failure modes & invariants

| Invariant | Enforced where |
|---|---|
| The original `task` object is always returned. | Every branch in `process_task` ends in `return task`. |
| `task._response` is *only* mutated when `any_suspicion=True`. | Only the editor branch assigns `task._response`. |
| Verifier failures cannot upgrade content quality, only flag/remove. | The editor prompt forbids alternatives and explanations. |
| Sub-agent usage is always accounted for, even when the editor is skipped. | Verifier loop `incr`s into `_reliability_sub_agent_usage` unconditionally; pipeline step folds it into the parent usage. |
| Cached or policy-blocked tasks bypass the layer entirely. | `ReliabilityStep.execute` checks `task._cached_result` and `task._policy_blocked` before constructing the manager. |

---

## Appendix A — Reference: source-of-truth file paths

| Concern | Absolute path |
|---|---|
| `ReliabilityProcessor`, prompts, models, regex helpers | `/Users/dogankeskin/Desktop/Upsonic/src/upsonic/reliability_layer/reliability_layer.py` |
| `ReliabilityManager` (async lifecycle wrapper) | `/Users/dogankeskin/Desktop/Upsonic/src/upsonic/agent/context_managers/reliability_manager.py` |
| `ReliabilityStep` (pipeline integration) | `/Users/dogankeskin/Desktop/Upsonic/src/upsonic/agent/pipeline/steps.py` (line 2494) |
| Pipeline registration of the step | `/Users/dogankeskin/Desktop/Upsonic/src/upsonic/agent/pipeline/__init__.py` |
| `Agent.reliability_layer` attribute | `/Users/dogankeskin/Desktop/Upsonic/src/upsonic/agent/agent.py` (line ~471) |
| `AutonomousAgent` forwarding | `/Users/dogankeskin/Desktop/Upsonic/src/upsonic/agent/autonomous_agent/autonomous_agent.py` (line ~314) |

## Appendix B — Adding a new verifier

To add, say, a *citation* verifier:

1. Add a new prompt constant near the existing four (e.g. `citation_validation_prompt`).
2. Add `citation_validation: ValidationPoint` to `ValidationResult` and update `calculate_suspicion` to include its flags, points, and confidence in the aggregation.
3. Inside `process_task`, append `("citation_validation", citation_validation_prompt)` to the iteration list.
4. (Optional) Add a `find_citations_in_text` regex pre-filter and a corresponding `if validation_type == "citation_validation": ...` short-circuit.
5. No pipeline changes needed — `ReliabilityStep` is data-driven.

The minimal-touch design here is intentional: every new verifier is one prompt + one model field + one tuple entry.
