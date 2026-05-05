---
name: agent-culture
description: Use when working with agent persona, behavior policy, tone, scope enforcement, or system-prompt identity injection in Upsonic. Use when a user asks to define how an agent should speak, what topics it should avoid or help with, or how to combat prompt drift across long conversations. Trigger when the user mentions Culture, CultureManager, CulturalKnowledge, culture package, persona, tone of speech, topics to avoid, scope enforcement, extracted_guidelines, format_for_system_prompt, repeat_interval, aprepare, ExtractedCulture, or `<CulturalKnowledge>` block.
---

# `src/upsonic/culture/` — Agent Culture & Behavior Guidelines

This document is a deep technical walkthrough of the `culture` package inside the Upsonic AI agent framework. The package is small (four Python modules) but plays a focused role: it lets users define **how** an agent should behave, communicate and constrain itself — and it injects those rules into the system prompt so the underlying LLM is forced to comply.

---

## 1. What this folder is — its role in Upsonic

The `culture` package is Upsonic's **persona / behavior-policy layer**. It sits one level above plain prompt engineering and one level below the safety engine.

Where the rest of the framework focuses on *capability* (tools, RAG, reliability, multi-agent), `culture` focuses on *identity* and *scope*:

| Concern                                        | Owned by `culture`? |
|------------------------------------------------|---------------------|
| Who is the agent (role, name, character)?      | Yes                 |
| How does the agent speak (tone)?               | Yes                 |
| What topics is the agent allowed to discuss?   | Yes                 |
| What topics must the agent decline?            | Yes                 |
| What principles must the agent always honor?   | Yes                 |
| Hard content filtering / policy enforcement    | No (`safety_engine`)|
| User-specific memory                           | No (`memory`)       |
| Tool routing / execution                       | No (`tools`)        |

The package ships two main public classes:

* **`Culture`** (dataclass) — declarative description of desired agent behavior.
* **`CultureManager`** — runtime that takes a `Culture`, calls a sub-Agent to *extract* structured guidelines from the free-text description, and renders an injectable system-prompt block.

A third dataclass, **`CulturalKnowledge`**, models a richer, persisted unit of "shared knowledge / principles / practices" that other parts of the framework (or higher-level KB layers) can store and ship across agents. It is intentionally storage-agnostic in this folder.

A key design statement is captured in the manager's docstring:

> Storage operations are NOT handled by `CultureManager` (removed from storage).

So `culture/` is purely a **behavioral / formatting** module — no I/O, no persistence — leaving storage decisions to the surrounding agent runtime.

---

## 2. Folder layout — tree diagram

```
src/upsonic/culture/
├── __init__.py             # Lazy-loading public surface (Culture, CultureManager)
├── culture.py              # Culture dataclass — user-facing config object
├── cultural_knowledge.py   # CulturalKnowledge dataclass — shared KB-style record
└── manager.py              # CultureManager — extraction + system-prompt formatting
```

Four files. Roughly ~700 lines total. No subfolders, no tests inside the package itself.

---

## 3. Top-level files — file-by-file walkthrough

### 3.1 `__init__.py`

The package entry point uses **PEP 562 lazy attribute loading** to avoid importing heavy modules (Pydantic, the Agent runtime, etc.) until a class is actually accessed.

| Symbol                  | Kind     | Purpose                                                         |
|-------------------------|----------|-----------------------------------------------------------------|
| `_get_culture_classes`  | function | Imports `Culture` and `CultureManager` on demand; returns dict. |
| `__getattr__(name)`     | hook     | Module-level `__getattr__`; dispatches lazy imports.            |
| `__all__`               | tuple    | Declares the public surface: `Culture`, `CultureManager`.       |

```python
from upsonic.culture import Culture, CultureManager
```

Both names are wired up via `__getattr__`. `CulturalKnowledge` is **not** re-exported here — it lives in its module and is imported directly by callers that need it.

A `TYPE_CHECKING` block guards type hints so static analyzers see the real classes while runtime stays cheap.

### 3.2 `culture.py`

A small, validated dataclass capturing the user-supplied culture configuration.

```python
@dataclass
class Culture:
    description: str                       # required, non-empty
    add_system_prompt: bool = True
    repeat: bool = False
    repeat_interval: int = 5               # must be >= 1
```

#### Fields

| Field               | Type   | Default | Meaning                                                                                  |
|---------------------|--------|---------|------------------------------------------------------------------------------------------|
| `description`       | `str`  | —       | Free-text description of the desired agent behavior (the "persona" prompt source).       |
| `add_system_prompt` | `bool` | `True`  | If true, the manager injects culture into the very first system prompt.                  |
| `repeat`            | `bool` | `False` | If true, culture is re-injected periodically into ongoing message turns.                 |
| `repeat_interval`   | `int`  | `5`     | Number of messages between re-injections (only used when `repeat=True`).                 |

#### Methods

| Method                   | Signature                                | Behavior                                                                            |
|--------------------------|------------------------------------------|-------------------------------------------------------------------------------------|
| `__post_init__`          | `(self) -> None`                         | Validates: `description` non-empty after strip; `repeat_interval >= 1`. Raises `ValueError`. |
| `to_dict`                | `(self) -> dict[str, Any]`               | Plain dict serialization of all four fields.                                        |
| `from_dict` (classmethod)| `(cls, data: dict) -> Culture`           | `cls(**data)` — round-trip constructor.                                             |
| `__repr__`               | `(self) -> str`                          | Truncates `description` to 50 chars for log readability.                            |

Note: `from_dict`'s annotation uses the lowercase `any` (Python builtin) rather than `typing.Any`. This still works at runtime because dataclasses don't enforce type hints, but it's effectively `dict[str, any]` (i.e., the builtin `any` callable as an annotation). It is a typo of historical interest, not a behavior change.

### 3.3 `cultural_knowledge.py`

A dataclass for storing a single unit of "cultural knowledge" — a longer-form, structured record that could be persisted across agents.

#### Helper functions (module-level)

| Function                  | Signature                                                | Purpose                                                                |
|---------------------------|----------------------------------------------------------|------------------------------------------------------------------------|
| `_now_epoch_s`            | `() -> int`                                              | Current UTC epoch seconds.                                             |
| `_to_epoch_s`             | `(value: int\|float\|str\|datetime) -> int`              | Normalizes int/float/ISO-string/`datetime` into UTC epoch seconds.     |
| `_epoch_to_rfc3339_z`     | `(ts: int\|float) -> str`                                | Converts epoch seconds to `RFC3339` with trailing `Z`.                 |

`_to_epoch_s` accepts:

* `int` / `float` — assumed already in seconds.
* `datetime` — naive datetimes are treated as UTC.
* `str` — ISO 8601; trailing `Z` is converted to `+00:00` before `fromisoformat`. Raises `ValueError` on parse failure, `TypeError` on unknown types.

#### `CulturalKnowledge` dataclass

| Field         | Type                            | Notes                                                                 |
|---------------|---------------------------------|-----------------------------------------------------------------------|
| `id`          | `Optional[str]`                 | Unique identifier (auto-generated externally if not provided).        |
| `name`        | `Optional[str]`                 | Short title; **must not be an empty/whitespace string** if provided.  |
| `content`     | `Optional[str]`                 | Main body — the principle / rule / guideline.                         |
| `categories`  | `Optional[List[str]]`           | Tags, e.g. `['guardrails', 'rules', 'principles', 'practices']`.      |
| `notes`       | `Optional[List[str]]`           | Free-form notes / examples / rationale.                               |
| `summary`     | `Optional[str]`                 | One-line takeaway.                                                    |
| `metadata`    | `Optional[Dict[str, Any]]`      | Arbitrary structured metadata (source, author, version...).           |
| `input`       | `Optional[str]`                 | Original input that produced this knowledge.                          |
| `created_at`  | `Optional[int]`                 | Epoch seconds (UTC); auto-filled in `__post_init__`.                  |
| `updated_at`  | `Optional[int]`                 | Epoch seconds (UTC); defaults to `created_at` if not given.           |
| `agent_id`    | `Optional[str]`                 | Owning agent's id.                                                    |
| `team_id`     | `Optional[str]`                 | Owning team's id.                                                     |

#### Methods

| Method                | Signature                                            | Behavior                                                                                     |
|-----------------------|------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `__post_init__`       | `(self) -> None`                                     | Validates `name` (non-empty if not None). Initializes `created_at`/`updated_at` from helpers.|
| `bump_updated_at`     | `(self) -> None`                                     | Sets `updated_at = _now_epoch_s()`.                                                          |
| `preview`             | `(self) -> Dict[str, Any]`                           | Token-saving short form: truncates `summary`/`content`/each `note` to 100 chars + `"..."`.   |
| `to_dict`             | `(self) -> Dict[str, Any]`                           | Full serialization; converts timestamps to RFC3339 (`Z`); strips `None` values.              |
| `from_dict` (cls)     | `(data: Dict) -> CulturalKnowledge`                  | Re-normalizes timestamps with `_to_epoch_s`. Preserves explicit zeros and `None`s.           |
| `__repr__`            | `(self) -> str`                                      | Compact `id`, `name`, `categories` summary.                                                  |

`preview()` is specifically designed for **LLM context budgets** — it returns just the keys that are non-None and aggressively truncates string fields so they can be inlined into prompts without blowing the token budget.

### 3.4 `manager.py`

The runtime brain of the package. It owns:

1. The currently-set `Culture` instance.
2. A cached `extracted_guidelines` dict produced by a sub-Agent.
3. The repeat counter for periodic re-injection.
4. Aggregated LLM usage from the extraction sub-call.

#### Module-level constant

```python
CULTURE_EXTRACTION_SYSTEM_PROMPT = """You are a Culture Extraction Agent ..."""
```

This is the system prompt fed to a one-shot sub-Agent that turns the raw user description into a structured **four-axis** guideline:

* Tone of Speech
* Topics I Shouldn't Talk About
* Topics I Can Help With
* Things I Should Pay Attention To

These four axes are the **canonical schema** of a culture in Upsonic and they appear again in `format_for_system_prompt()`.

#### `class CultureManager`

Constructor:

```python
CultureManager(
    model: Optional[Union[Model, str]] = None,
    enabled: bool = True,
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
    debug: bool = False,
    debug_level: int = 1,
    print: Optional[bool] = None,
)
```

Internal state set in `__init__`:

| Attribute                | Type                              | Purpose                                                                |
|--------------------------|-----------------------------------|------------------------------------------------------------------------|
| `_model_spec`            | `Optional[Model\|str]`            | Model used by the extraction sub-Agent.                                |
| `enabled`                | `bool`                            | Master kill-switch (consumed by upstream agent runtime).               |
| `agent_id`, `team_id`    | `Optional[str]`                   | Identity context, mostly for logging / KB linking.                     |
| `debug`, `debug_level`   | `bool`, `int`                     | Verbosity controls (used by `info_log` / `warning_log`).               |
| `print`                  | `bool`                            | Controls printing of sub-Agent output. Defaults to `True` if `None`.   |
| `_culture`               | `Optional[Culture]`               | Currently-set `Culture` instance.                                      |
| `_extracted_guidelines`  | `Optional[Dict[str, str]]`        | Result of `_extract_guidelines`.                                       |
| `_prepared`              | `bool`                            | Has `aprepare()` been run for the current culture?                     |
| `_message_count`         | `int`                             | Counter for repeat-injection logic.                                    |
| `_last_llm_usage`        | `Optional[Any]`                   | Accumulator for `RunUsage` from extraction sub-calls.                  |

#### Properties

| Property               | Returns                       | Notes                                  |
|------------------------|-------------------------------|----------------------------------------|
| `culture`              | `Optional[Culture]`           | The current `Culture` instance.        |
| `extracted_guidelines` | `Optional[Dict[str, str]]`    | `tone_of_speech`, etc.                 |
| `prepared`             | `bool`                        | Whether extraction has been performed. |

#### Methods — full reference

| Method                          | Signature                                                    | Description                                                                                                                                                              |
|---------------------------------|--------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `set_culture`                   | `(self, culture: Culture) -> None`                           | Replaces the current culture and **invalidates** any prior extraction (`_prepared=False`, `_extracted_guidelines=None`).                                                 |
| `aprepare`                      | `async (self) -> None`                                       | Idempotent. If not prepared and a culture exists, runs `_extract_guidelines(self._culture.description)`.                                                                 |
| `prepare`                       | `(self) -> None`                                             | Sync wrapper. Detects a running event loop and offloads to a `ThreadPoolExecutor` to avoid `RuntimeError`. Falls back to `asyncio.run` when no loop is running.          |
| `_extract_guidelines`           | `async (self, description: str) -> Dict[str, str]`           | The core extraction routine. Builds an inline Pydantic schema, spawns an `Agent` + `Task`, parses output, accumulates usage. Multiple fallbacks ensure a usable result.  |
| `format_for_system_prompt`      | `(self) -> Optional[str]`                                    | Produces the `<CulturalKnowledge>...</CulturalKnowledge>` block. Returns `None` when no culture or no extracted guidelines.                                              |
| `should_repeat`                 | `(self) -> bool`                                             | Increments `_message_count`; returns `True` and resets the counter when `_message_count >= repeat_interval` and `repeat=True`.                                           |
| `drain_accumulated_usage`       | `(self) -> Optional[Any]`                                    | Returns the accumulated `RunUsage` and clears it. Used by the parent agent to fold sub-Agent costs into its own usage report.                                            |
| `reset_message_count`           | `(self) -> None`                                             | Resets `_message_count` to 0 (testing / manual control).                                                                                                                 |
| `to_dict`                       | `(self) -> Dict[str, Any]`                                   | Serializes config + state (including nested `Culture.to_dict()` and `_extracted_guidelines`).                                                                            |
| `from_dict` (classmethod)       | `(data: Dict, model=None) -> CultureManager`                 | Inverse of `to_dict`. Reconstructs the manager and its `Culture` if present; restores `_prepared`, `_message_count`, `_extracted_guidelines`.                            |

#### `_extract_guidelines` in detail

This is the most logic-heavy method in the package. Pseudocode:

```
if no _model_spec:
    log warning, set generic fallback guidelines, return them.

define ExtractedCulture(BaseModel) with 4 string fields.
build extraction_prompt embedding `description`.

try:
    Agent(model=..., system_prompt=CULTURE_EXTRACTION_SYSTEM_PROMPT, ...)
    Task(description=extraction_prompt, response_format=ExtractedCulture)
    result = await agent.do_async(task, return_output=True)

    accumulate usage into self._last_llm_usage (RunUsage).

    if result:
        store guideline dict, optionally info_log success, return.
    else:
        warn, set generic fallback, return.
except Exception as e:
    warn with error, set generic fallback, return.
```

Three properties worth highlighting:

1. **Three-layer fallback**: no model → generic; model returns `None` → generic; exception → generic. The agent **never** crashes due to extraction failing.
2. **Usage aggregation**: even though `__init__` already sets `_last_llm_usage = None`, the method also re-checks `hasattr(self, '_last_llm_usage')` defensively — useful when state is reconstituted from `from_dict` with no extraction run.
3. **Output schema is local**: `ExtractedCulture` is defined *inside* the method to avoid pulling Pydantic at import time and to keep the schema logically scoped to extraction.

#### `format_for_system_prompt` in detail

The output is a strongly worded, **MANDATORY**-style policy block wrapped in `<CulturalKnowledge>...</CulturalKnowledge>` XML-ish tags. The structure is fixed:

```
<CulturalKnowledge>
## MANDATORY AGENT CULTURE GUIDELINES - STRICT COMPLIANCE REQUIRED
... critical compliance preamble ...
### Your Identity
**{culture.description}**
... identity reinforcement ...
**SCOPE ENFORCEMENT:** ... bullet list ...
### Tone of Speech
{guidelines.tone_of_speech}
### Topics I Shouldn't Talk About
{guidelines.topics_to_avoid}
... conditional warning if not "N/A" ...
### Topics I Can Help With
{guidelines.topics_to_help}
... conditional warning if not "N/A" ...
### Things I Should Pay Attention To
{guidelines.things_to_pay_attention}
**FINAL REMINDER:** ...
</CulturalKnowledge>
```

The conditional warnings (e.g. *"If a user asks about any of these topics, you MUST politely decline..."*) are appended **only** when the corresponding axis is non-empty and not `"N/A"`. This avoids polluting the prompt with reminders for empty sections.

The XML-style tag is intentional: it gives the model a clear scope boundary it can reliably attend to and helps tools that scrape or post-process the system prompt to find/strip the block.

---

## 4. Subfolders

There are **no subfolders** in `src/upsonic/culture/`. All four modules sit at the package root.

---

## 5. Cross-file relationships

```
                ┌────────────────────────┐
                │   __init__.py          │
                │ (lazy public surface)  │
                └─────────┬──────────────┘
                          │ exposes
                          ▼
        ┌──────────────────────┐         ┌─────────────────────────┐
        │ culture.py           │         │ cultural_knowledge.py   │
        │ class Culture        │         │ class CulturalKnowledge │
        └──────────┬───────────┘         └─────────────────────────┘
                   │ used by                  (independent)
                   ▼
        ┌─────────────────────────────────────────────────────────┐
        │ manager.py                                               │
        │ class CultureManager                                     │
        │   - holds Culture                                        │
        │   - calls Agent + Task to extract structured guidelines  │
        │   - formats system-prompt block                          │
        └────────────────────────┬─────────────────────────────────┘
                                 │ imports lazily
                                 ▼
                ┌────────────────────────────┐
                │ upsonic.agent.agent.Agent  │
                │ upsonic.tasks.tasks.Task   │
                │ upsonic.usage.RunUsage     │
                │ upsonic.utils.printing.*   │
                └────────────────────────────┘
```

| Relation                                                | Where                                          |
|---------------------------------------------------------|------------------------------------------------|
| `CultureManager.set_culture` ← `Culture`                | `manager.py` accepts a `Culture` instance.     |
| `CultureManager.from_dict` → `Culture.from_dict`        | Round-tripping nested state.                   |
| `CultureManager._extract_guidelines` → `Agent`, `Task`  | Lazy import inside method body.                |
| `CultureManager._extract_guidelines` → `RunUsage`       | For usage accumulation.                        |
| `CultureManager._extract_guidelines` → `info_log`, `warning_log` | Logging via `upsonic.utils.printing`. |
| `CulturalKnowledge`                                     | **Stand-alone** — not consumed by `manager.py`.|

`CulturalKnowledge` is intentionally decoupled: it's a record format the surrounding framework can store and ship across agents (think "principles library"), but the local manager doesn't depend on it.

---

## 6. Public API

What other modules should import:

```python
from upsonic.culture import Culture, CultureManager
from upsonic.culture.cultural_knowledge import CulturalKnowledge  # not re-exported
```

### Culture

```python
Culture(
    description: str,                # required, non-empty
    add_system_prompt: bool = True,
    repeat: bool = False,
    repeat_interval: int = 5,        # >= 1
)
Culture.to_dict() -> dict
Culture.from_dict(data: dict) -> Culture
```

### CultureManager

```python
CultureManager(
    model: Optional[Model | str] = None,
    enabled: bool = True,
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
    debug: bool = False,
    debug_level: int = 1,
    print: Optional[bool] = None,
)

# Lifecycle
manager.set_culture(culture)
await manager.aprepare()      # or: manager.prepare()

# Use
prompt_block: str | None = manager.format_for_system_prompt()
manager.should_repeat()       # increment + threshold check
manager.reset_message_count()
manager.drain_accumulated_usage()  # → RunUsage | None

# Persistence
state = manager.to_dict()
restored = CultureManager.from_dict(state, model="openai/gpt-4o")

# Inspection
manager.culture
manager.extracted_guidelines
manager.prepared
```

### CulturalKnowledge

```python
CulturalKnowledge(
    id=None, name=None, content=None,
    categories=None, notes=None, summary=None,
    metadata=None, input=None,
    created_at=None, updated_at=None,
    agent_id=None, team_id=None,
)
CulturalKnowledge.bump_updated_at()
CulturalKnowledge.preview() -> dict   # token-friendly
CulturalKnowledge.to_dict() -> dict   # RFC3339 timestamps
CulturalKnowledge.from_dict(data) -> CulturalKnowledge
```

---

## 7. Integration with the rest of Upsonic

The `culture` package is consumed by the **agent runtime**. The connection points are:

| Upsonic component                                | How `culture` integrates                                                                  |
|--------------------------------------------------|-------------------------------------------------------------------------------------------|
| `upsonic.agent.agent.Agent`                      | A parent Agent owns a `CultureManager`; instantiates a sub-Agent inside `_extract_guidelines`. |
| `upsonic.tasks.tasks.Task`                       | The extraction sub-call is expressed as a `Task` with `response_format=ExtractedCulture`. |
| `upsonic.models.Model`                           | `CultureManager._model_spec` accepts a `Model` instance or a `"provider/name"` string.    |
| `upsonic.usage.RunUsage`                         | Sub-Agent token usage is folded back via `drain_accumulated_usage()`.                     |
| `upsonic.utils.printing.info_log`, `warning_log` | Debug output respects `debug` / `debug_level`.                                            |

Because the manager lazily imports `Agent`, `Task`, and `RunUsage` *inside* `_extract_guidelines`, importing `upsonic.culture` itself is **cheap and side-effect free** — important for CLIs, tests, and quick dataclass usage.

The `enabled` flag is purely a signal to the upstream caller; the manager never inspects it on its own. The upstream agent runtime is responsible for skipping `aprepare()` / `format_for_system_prompt()` when `enabled=False`.

The repeat mechanism (`repeat`, `repeat_interval`, `should_repeat`) lets the agent loop re-inject the culture block every N messages to combat **prompt drift** — a common failure mode where the model gradually loses track of identity/scope across long conversations.

---

## 8. End-to-end flow

A typical lifecycle, from user code to LLM call:

```python
from upsonic.culture import Culture, CultureManager

# 1. User declares culture
culture = Culture(
    description="You are a 5-star hotel receptionist at the Grand Plaza Hotel. "
                "Greet guests warmly, only help with hotel-related queries, "
                "and never discuss politics or competitors.",
    add_system_prompt=True,
    repeat=True,
    repeat_interval=10,
)

# 2. User constructs a manager
manager = CultureManager(model="openai/gpt-4o", debug=True)
manager.set_culture(culture)

# 3. Extract guidelines (one-shot LLM call)
await manager.aprepare()
# manager.extracted_guidelines is now:
# {
#   "tone_of_speech": "Warm, polite, professional...",
#   "topics_to_avoid": "Politics, competitors...",
#   "topics_to_help":  "Hotel reservations, amenities, check-in...",
#   "things_to_pay_attention": "Always greet by name when known..."
# }

# 4. Build system-prompt block
block = manager.format_for_system_prompt()
# Wrapped in <CulturalKnowledge>...</CulturalKnowledge>, MANDATORY-style copy.

# 5. Parent agent injects `block` into its system prompt.
# (This step lives in the agent runtime, not in this package.)

# 6. After every user/agent turn, the runtime calls:
if manager.should_repeat():
    # Re-inject the culture block to combat drift.
    ...

# 7. After the conversation completes, fold sub-Agent costs in:
sub_usage = manager.drain_accumulated_usage()
if sub_usage:
    parent_run_usage.incr(sub_usage)
```

### Sequence diagram — extraction phase

```
User Code            CultureManager           Agent (extractor)        LLM
   │                       │                          │                  │
   │ set_culture(culture)  │                          │                  │
   │──────────────────────>│                          │                  │
   │                       │                          │                  │
   │ await aprepare()      │                          │                  │
   │──────────────────────>│                          │                  │
   │                       │ build ExtractedCulture   │                  │
   │                       │ build extraction_prompt  │                  │
   │                       │ Agent(...)               │                  │
   │                       │─────────────────────────>│                  │
   │                       │ Task(response_format=…)  │                  │
   │                       │ await do_async(task)     │                  │
   │                       │─────────────────────────>│ run prompt       │
   │                       │                          │─────────────────>│
   │                       │                          │ structured JSON  │
   │                       │                          │<─────────────────│
   │                       │ result, usage            │                  │
   │                       │<─────────────────────────│                  │
   │                       │ store _extracted_guidelines                 │
   │                       │ accumulate _last_llm_usage                  │
   │<──────────────────────│                          │                  │
   │                       │                          │                  │
   │ format_for_system_prompt()                       │                  │
   │──────────────────────>│                          │                  │
   │ <CulturalKnowledge>…  │                          │                  │
   │<──────────────────────│                          │                  │
```

### Failure modes & fallbacks

| Condition                             | Result                                                              |
|---------------------------------------|---------------------------------------------------------------------|
| `description` empty                   | `Culture.__post_init__` raises `ValueError`.                        |
| `repeat_interval < 1`                 | `Culture.__post_init__` raises `ValueError`.                        |
| `aprepare()` called twice             | No-op the second time (`_prepared=True` guard).                     |
| `set_culture()` after `aprepare()`    | `_prepared` flips back to `False`; next `aprepare()` re-extracts.   |
| No `model` provided to manager        | Generic fallback guidelines; `info_log/warning_log` if debug.       |
| Extraction returns `None`             | Generic fallback; warning log.                                      |
| Extraction raises                     | Generic fallback; warning log with exception text.                  |
| `format_for_system_prompt` w/o prep   | Returns `None`.                                                     |
| `should_repeat` w/o `culture.repeat`  | Returns `False`, counter not incremented.                           |

---

## 9. Design notes & gotchas

* **No storage.** The package was deliberately stripped of storage concerns; persistence belongs to the agent/team layer above. `to_dict` / `from_dict` exist only for external serialization.
* **Lazy imports everywhere.** `__init__.py` uses module `__getattr__`; `manager._extract_guidelines` imports `Agent`, `Task`, `RunUsage`, `printing` at call time. This keeps `import upsonic.culture` essentially free.
* **`CulturalKnowledge.preview()` vs `to_dict()`** — choose `preview()` for prompt context (truncated, lossy) and `to_dict()` for storage (full, RFC3339 timestamps).
* **`prepare()` thread-pool dance.** When called from inside an existing event loop, it spawns a thread to run `asyncio.run` in isolation, avoiding `"asyncio.run() cannot be called from a running event loop"`. This is a common but subtle pattern.
* **`Culture.from_dict` annotation typo.** Uses lowercase `any` instead of `Any`. Functionally harmless, but worth fixing if you tighten typing.
* **Deliberately strong prompt language.** `format_for_system_prompt` uses ALL-CAPS "MUST"/"MANDATORY" phrasing because empirical experience shows this materially improves scope adherence on weaker models.

---

## 10. Quick reference cheat sheet

```python
# Create a culture
from upsonic.culture import Culture, CultureManager

culture = Culture(description="…", repeat=True, repeat_interval=5)

# Wire up a manager
mgr = CultureManager(model="openai/gpt-4o")
mgr.set_culture(culture)
await mgr.aprepare()

# Inject into system prompt
sys_prompt_block = mgr.format_for_system_prompt()

# Loop hook
if mgr.should_repeat():
    re_inject(sys_prompt_block)

# Cost folding
parent_usage.incr(mgr.drain_accumulated_usage())

# Persist / restore
state = mgr.to_dict()
mgr2 = CultureManager.from_dict(state, model="openai/gpt-4o")
```

That is the entirety of `src/upsonic/culture/` — a focused, side-effect-free behavioral layer that converts a single English sentence into a strict, structured, repeatable system-prompt directive.
