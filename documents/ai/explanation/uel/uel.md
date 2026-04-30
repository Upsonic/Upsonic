---
name: uel-expression-language
description: Use when composing Upsonic chains with the pipe operator or wiring prompts, models, parsers, and Python callables into a Runnable pipeline. Use when a user asks to build sequential, parallel, branching, or passthrough chains, write @chain-decorated functions, parse model output into strings or Pydantic objects, or visualize a chain or StateGraph as ASCII/Mermaid. Trigger when the user mentions Upsonic Expression Language, UEL, Runnable, RunnableSequence, RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough, ChatPromptTemplate, StrOutputParser, PydanticOutputParser, BaseOutputParser, RunnableGraph, @chain decorator, itemgetter, coerce_to_runnable, pipe operator, LCEL-style composition, invoke/ainvoke, or rendering a chain to Mermaid.
---

# UEL — Upsonic Expression Language

Source folder: `/Users/dogankeskin/Desktop/Upsonic/src/upsonic/uel/`

## 1. What this folder is

`upsonic.uel` is the **Upsonic Expression Language** — a small, LangChain‑LCEL–style composition primitive layer that lets every component in Upsonic (prompts, models, parsers, plain Python functions, dicts, sub‑graphs) be wired together with the pipe operator (`|`) and executed with a uniform `invoke` / `ainvoke` interface.

The folder is intentionally **decoupled from any specific model/provider**. It defines:

- A single abstract base class `Runnable[Input, Output]` with a `__or__` overload that turns `a | b` into a `RunnableSequence`.
- A handful of concrete runnable types that cover the structural patterns of an LLM pipeline: sequence, parallel, branch, lambda, passthrough/assign, prompt template, output parser.
- A `@chain` decorator that turns ordinary Python functions into runnables.
- A graph view (`RunnableGraph`) that introspects an arbitrary chain (or a `graphv2` `StateGraph`) and renders it as ASCII or Mermaid.

UEL is the *plumbing* on which the higher‑level Upsonic primitives sit:

| Upsonic class                                  | Inherits from        | Effect |
|------------------------------------------------|----------------------|--------|
| `upsonic.models.Model` (base of every provider) | `Runnable[Any, Any]` | Every model can be used as `prompt \| model \| parser`. |
| `upsonic.graphv2.state_graph.StateGraph`        | `Runnable`           | Compiled graphs are first‑class chain steps. |
| `upsonic.uel.ChatPromptTemplate`                | `Runnable`           | Renders to a `ModelRequest` / list of `ModelMessage`s. |
| `upsonic.uel.BaseOutputParser`                  | `Runnable`           | Consumes a `ModelResponse`. |

The intent is the same as LangChain’s LCEL: declarative chain construction, automatic sync/async dispatch, and trivial composition with plain Python (functions, dicts, `operator.itemgetter`).

## 2. Folder layout

```
src/upsonic/uel/
├── __init__.py            # Lazy public API surface + drop-in `itemgetter`
├── runnable.py            # Abstract Runnable[Input, Output] + pipe operator
├── sequence.py            # RunnableSequence — chain of steps run in order
├── parallel.py            # RunnableParallel — fan-out, dict result
├── branch.py              # RunnableBranch — conditional routing
├── lambda_runnable.py     # RunnableLambda + coerce_to_runnable() helper
├── passthrough.py         # RunnablePassthrough(.assign) — input fan-through
├── prompt.py              # ChatPromptTemplate — string / message templating
├── output_parser.py       # BaseOutputParser, StrOutputParser, PydanticOutputParser
├── decorator.py           # @chain decorator (function -> Runnable)
└── graph.py               # RunnableGraph — ASCII / Mermaid visualization
```

11 files, ~2 KLoC. No subfolders.

## 3. Top-level files

### `__init__.py` — public surface, lazy imports

`__getattr__` is used so that `from upsonic.uel import X` only triggers the actual import when the attribute is accessed. This keeps `import upsonic` cheap.

| Exported name           | Where it lives             | Type                                       |
|-------------------------|----------------------------|--------------------------------------------|
| `Runnable`              | `runnable.py`              | abstract base class                        |
| `RunnableSequence`      | `sequence.py`              | linear chain                               |
| `RunnableParallel`      | `parallel.py`              | concurrent fan-out                         |
| `RunnableLambda`        | `lambda_runnable.py`       | wraps a function                           |
| `RunnableBranch`        | `branch.py`                | conditional routing                        |
| `RunnablePassthrough`   | `passthrough.py`           | identity / `.assign(...)`                  |
| `ChatPromptTemplate`    | `prompt.py`                | prompt builder                             |
| `BaseOutputParser`      | `output_parser.py`         | abstract parser                            |
| `StrOutputParser`       | `output_parser.py`         | text extractor                             |
| `PydanticOutputParser`  | `output_parser.py`         | JSON → Pydantic                            |
| `chain`                 | `decorator.py`             | decorator                                  |
| `itemgetter`            | `__init__.py` (synthesized)| pipe-aware drop-in for `operator.itemgetter` |

`itemgetter` is unique to this module: it returns a `RunnableLambda` so `itemgetter("k") | something` works directly inside a UEL chain.

```python
from upsonic.uel import itemgetter
chain = itemgetter("key") | (lambda x: f"Value: {x}")
chain.invoke({"key": "test"})   # "Value: test"
```

### `runnable.py` — the protocol

`Runnable[Input, Output]` is the only abstract class users typically subclass.

```python
class Runnable(ABC, Generic[Input, Output]):
    @abstractmethod
    def invoke(self, input: Input, config: dict | None = None) -> Output: ...

    async def ainvoke(self, input, config=None):
        # default: run sync invoke in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, input, config)

    def __or__(self, other) -> "Runnable":
        # a | b   ->  RunnableSequence(steps=[a, b])
        # if self is already a sequence, the new step is appended in-place
        # `other` is coerced via coerce_to_runnable()
    def __ror__(self, other) -> "Runnable":
        # supports  dict | runnable, callable | runnable
```

Key design points:

- **`__or__` does the coercion**, so `prompt | model | (lambda r: r.text)` works without the user wrapping the lambda.
- **`__ror__`** lets a leading non-Runnable (a callable or a dict) participate in the pipeline; this is what makes `{"a": chain1, "b": chain2} | next_step` work.
- The default `ainvoke` falls back to `invoke` in an executor, so any subclass can implement only `invoke` and still be awaitable.

### `sequence.py` — `RunnableSequence`

A flat `list[Runnable]` of `steps`. `invoke` walks the list, threading the output of step *n* into the input of step *n+1*.

```python
class RunnableSequence(Runnable[Any, Any]):
    def __init__(self, steps: list[Runnable]): ...
    def invoke(self, input, config=None):
        result = input
        for step in self.steps:
            result = step.invoke(result, config)
        return result
    async def ainvoke(self, input, config=None):
        result = input
        for step in self.steps:
            result = await step.ainvoke(result, config)
        return result
    def __or__(self, other):
        # extends rather than nesting
        return RunnableSequence(steps=self.steps + [coerce_to_runnable(other)])
    def get_graph(self) -> "RunnableGraph": ...
    def get_prompts(self) -> list[ChatPromptTemplate]: ...
```

`get_prompts()` walks `steps` recursively (descending into `RunnableSequence`/`RunnableParallel` via `.steps`) and harvests every `ChatPromptTemplate`. Useful for prompt management / inspection.

### `parallel.py` — `RunnableParallel`

Fan-out: pass the **same input** to N named runnables, gather their outputs into a dict.

```python
RunnableParallel(joke=joke_chain, poem=poem_chain).invoke({"topic": "bears"})
# -> {"joke": ModelResponse(...), "poem": ModelResponse(...)}
```

Behaviour notes:

- `invoke()` calls `asyncio.run(self.ainvoke(...))`. Inside `ainvoke()` it uses `asyncio.gather` over `runnable.ainvoke(input, config)` for true parallelism. **If any branch raises, all remaining tasks are cancelled** and the exception propagates.
- Each kwarg value is coerced: `Runnable` is kept; `dict` becomes a nested `RunnableParallel.from_dict`; any callable is wrapped in `RunnableLambda`. Other types raise `TypeError`.
- A plain `dict[str, Runnable | Callable]` can be used in a chain because `coerce_to_runnable(dict)` returns `RunnableParallel.from_dict(dict)`. So:

  ```python
  chain = {"joke": joke_chain, "poem": poem_chain} | next_step
  ```

  is equivalent to `RunnableParallel(joke=…, poem=…) | next_step`.

### `branch.py` — `RunnableBranch`

Imperative `if/elif/.../else` for chains.

```python
RunnableBranch(
    (lambda x: "upsonic"   in x["topic"].lower(), upsonic_chain),
    (lambda x: "anthropic" in x["topic"].lower(), anthropic_chain),
    general_chain,                # default — must be the LAST argument
)
```

Validation in `__init__`:

| Position           | Required form                              | Failure |
|--------------------|--------------------------------------------|---------|
| Last positional    | `Runnable` or callable (NOT a tuple)       | `ValueError` |
| Earlier positionals| `(condition, runnable)` 2-tuple            | `ValueError` |
| `condition`        | callable                                   | `ValueError` |

Runtime behaviour:

- Conditions are evaluated **in order**. The first one that returns truthy wins; subsequent branches are not evaluated.
- A `condition(input)` that **raises** is treated as “does not match” (`continue`), not as a hard failure.
- `ainvoke` additionally `await`s a condition if it’s a coroutine function (`inspect.iscoroutinefunction`).
- When no branch matches, the default runnable is invoked.

### `lambda_runnable.py` — `RunnableLambda` + `coerce_to_runnable`

`RunnableLambda(func)` wraps any callable.

```python
class RunnableLambda(Runnable[Input, Output]):
    def __init__(self, func): ...
    def invoke(self, input, config=None):
        if self.is_coroutine:
            # If called from an async context: dispatch to a thread that owns its own loop
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    return ex.submit(asyncio.run, self.func(input)).result()
            except RuntimeError:
                return asyncio.run(self.func(input))
        return self.func(input)
    async def ainvoke(self, input, config=None):
        return await self.func(input) if self.is_coroutine \
               else await asyncio.get_event_loop().run_in_executor(None, self.func, input)
```

`coerce_to_runnable(thing)` is the central type-bridge used by `__or__`, `RunnableSequence.__or__`, `RunnablePassthrough`, `RunnableBranch`:

| Input type | Returned                                     |
|------------|----------------------------------------------|
| `Runnable` | unchanged                                    |
| `dict`     | `RunnableParallel.from_dict(dict)`           |
| callable   | `RunnableLambda(callable)`                   |
| else       | `TypeError`                                  |

### `passthrough.py` — `RunnablePassthrough`

Identity by default. With `.assign(**kwargs)` it merges new keys into the input dict, and **assignments are computed sequentially** — so a later assignment can read earlier ones.

```python
chain = (
    RunnablePassthrough.assign(
        formatted_question=lambda x: f"Question: {x['question']}",
    ).assign(
        context=lambda x: retrieve_context(x['question']),
    )
    | prompt
    | model
)
```

`assign` is implemented as a custom **descriptor** (`AssignDescriptor`) so the same identifier works as a classmethod (`RunnablePassthrough.assign(...)`) and an instance method (`existing.assign(...)` merges its assignments).

Input rules:

- No assignments → returns `input` unchanged (any type).
- With assignments → `input` MUST be a `dict`, otherwise `TypeError`.
- Each assignment value is `coerce_to_runnable`-d, then invoked with the **running result dict**, and the return is stored under the assignment key.

### `prompt.py` — `ChatPromptTemplate`

Builds either:

- a plain formatted string (legacy `template=` constructor), or
- a `ModelRequest` / `list[ModelMessage]` for multi-message prompts.

Two factory methods:

```python
ChatPromptTemplate.from_template("Tell me a {adj} joke about {topic}")
# -> single human-message template; variables auto-extracted by regex
ChatPromptTemplate.from_messages([
    ("system",      "You are a helpful assistant"),
    ("placeholder", {"variable_name": "chat_history"}),
    ("human",       "Tell me about {topic}"),
])
```

Supported message roles:

| Role            | Lowered to                                      |
|-----------------|-------------------------------------------------|
| `"system"`      | `SystemPromptPart` (placed in the **first** `ModelRequest`) |
| `"human"`/`"user"`     | `UserPromptPart`                          |
| `"ai"`/`"assistant"`   | `ModelResponse(parts=[TextPart])` (for few-shot examples) |
| `"placeholder"` | dynamic injection from `input[variable_name]`   |

Conversion rules enforced by `invoke`:

- Exactly **one** `SystemPromptPart`, attached to the first `ModelRequest` only.
- One `UserPromptPart` per `ModelRequest`. Adjacent human messages are merged with a space.
- One `TextPart` per `ModelResponse` (for `ai` few-shot turns).
- The conversation alternates Request → Response → Request → Response…
- If `input` is already a `ModelMessage` or a `list[ModelMessage]` (detected via `.kind in ('request', 'response')`), it is returned **as-is** — this lets memory/history flow through a prompt step unchanged.
- Missing required variables raise `KeyError`; non-dict / non-message input raises `TypeError`.

Placeholders accept either a list of `(role, content)` tuples (treated as chat history) or a string (appended as a user prompt).

### `output_parser.py` — `BaseOutputParser` family

`BaseOutputParser[T]` is `Runnable[ModelResponse, T]`. Subclasses implement only `parse(response: ModelResponse) -> T`.

| Class                          | `parse` behaviour                                                |
|--------------------------------|------------------------------------------------------------------|
| `StrOutputParser`              | Returns `response.parts[-1].content` if last part is a `TextPart`, else `""`. |
| `PydanticOutputParser(Model)`  | JSON-loads the last `TextPart` and validates against the supplied Pydantic class (`model_validate` if available, else `Model(**parsed)`). Raises `ValueError` on empty parts, non-text last part, JSON errors, or schema validation errors. |

Both are drop-in tail steps for `prompt | model | parser` chains.

### `decorator.py` — `@chain`

Turn a function (sync or async) into a runnable.

```python
@chain
def custom_chain(text):
    p = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    out = infer_model("openai/gpt-4o").invoke(p.invoke({"topic": text}))
    return (ChatPromptTemplate.from_template("Subject of joke: {joke}")
            | infer_model("openai/gpt-4o")).invoke({"joke": out})

custom_chain.invoke("bears")
```

Notable semantics:

- Internally builds a private `ChainRunnable(Runnable)` and calls `functools.update_wrapper` so `__name__`, `__doc__`, etc. are preserved.
- If the wrapped function returns a `Runnable`, the decorator **invokes that returned runnable with the same input** (and same `config`). This enables dynamic chains:

  ```python
  @chain
  def dynamic(input_):
      return complex_chain if input_["use_complex"] else simple_chain
  ```

- Sync `invoke` with an `async` body intentionally fails fast when called from inside a running event loop — it raises `RuntimeError` directing the caller to `await runnable.ainvoke(...)`.

### `graph.py` — `RunnableGraph`

Static introspection / visualization.

| Method                  | Output                                                        |
|-------------------------|---------------------------------------------------------------|
| `to_ascii()` / `print_ascii()` | Indented ASCII tree with `|`/`v` connectors and parallel branch markers. |
| `to_mermaid()`          | `graph TD ...` Mermaid source. Parallel splits use dotted edges with the branch key as a label; sequential edges use `==>`. |
| `get_structure_details()` | Per-node breakdown: parallel branches, merge targets, sequential edges. |

It accepts either a UEL `Runnable` (auto-expands `RunnableSequence`/`RunnableParallel` via `_build_graph`) **or** a `graphv2.StateGraph` / `CompiledStateGraph` (via `_build_state_graph`):

- Normal `StateGraph` edges become sequential `edges_to`.
- Conditional edges become `parallel_branches` (rendered with dashed Mermaid arrows) so dynamic routing is visually distinct.
- `START` (`__start__`) and `END` markers are added when referenced.
- A node that has *both* a normal edge and a conditional edge keeps only the conditional view (conditionals take precedence at runtime).

The internal `_StateGraphNodeRef` is a non-executable `Runnable` proxy used purely as a label carrier when visualizing graphs whose nodes aren’t themselves runnables.

## 4. Subfolders

None. UEL is intentionally flat.

## 5. Cross-file relationships

```
                      runnable.py  (Runnable, __or__)
                          ▲   ▲
                          │   │
   coerce_to_runnable     │   │  used by every Runnable subclass
   ────────────────────►  │   │
   lambda_runnable.py ────┘   │
        ▲      ▲              │
        │      │              │
        │   parallel.py ──────┤
        │   branch.py ────────┤
        │   passthrough.py ───┤
        │   prompt.py ────────┤
        │   output_parser.py ─┤
        │                     │
        └── decorator.py ─────┤
                              │
                  sequence.py ┘
                       │
                       ▼
                   graph.py
                       ▲
                       │
            graphv2.state_graph (external; introspected)
```

Concrete edges:

| Edge                                   | Purpose |
|----------------------------------------|---------|
| every concrete runnable → `runnable.py`| inherits `Runnable` |
| `runnable.__or__` → `sequence.py`, `lambda_runnable.coerce_to_runnable` | builds a `RunnableSequence` from `a \| b` |
| `lambda_runnable.coerce_to_runnable` → `parallel.py` | converts `dict` to `RunnableParallel` |
| `parallel.py` (constructor / `__or__`) → `lambda_runnable.RunnableLambda`, `sequence.RunnableSequence` | wraps callables, builds chains |
| `branch.py` → `lambda_runnable.coerce_to_runnable` | normalizes branch arms |
| `passthrough.py` → `lambda_runnable.coerce_to_runnable` (per-assignment, lazy import) | normalizes `assign` values |
| `decorator.py` → `runnable.py`, `lambda_runnable.py` | builds an inner `Runnable` subclass |
| `sequence.get_graph()` → `graph.RunnableGraph` | visualization entry-point |
| `sequence.get_prompts()` → `prompt.ChatPromptTemplate` | recursive harvest |
| `graph.py` → `sequence.RunnableSequence`, `parallel.RunnableParallel`, `graphv2.state_graph.StateGraph` | structural introspection |
| `__init__.py` → all of the above (lazy via `__getattr__`) | public surface |

There are **no cycles at import time** — every module-level import in this folder either imports `runnable.py` (a sink) or defers via local imports inside functions.

## 6. Public API / syntax

### Building a chain

```python
from upsonic.uel import (
    ChatPromptTemplate, RunnableParallel, RunnableLambda,
    RunnablePassthrough, RunnableBranch, StrOutputParser,
    PydanticOutputParser, chain, itemgetter,
)
from upsonic import infer_model
from pydantic import BaseModel

model = infer_model("openai/gpt-4o")

# 1) Linear: prompt | model | parser
prompt = ChatPromptTemplate.from_template("Summarize: {text}")
linear  = prompt | model | StrOutputParser()
linear.invoke({"text": "..."})

# 2) Parallel (dict syntax fan-out)
fanout = (
    {
        "summary":  ChatPromptTemplate.from_template("Summarize: {text}") | model | StrOutputParser(),
        "keywords": ChatPromptTemplate.from_template("Top 5 keywords: {text}") | model | StrOutputParser(),
    }
    | (lambda d: f"{d['summary']}\nKeywords: {d['keywords']}")
)
fanout.invoke({"text": "..."})

# 3) Conditional routing
router = RunnableBranch(
    (lambda x: x["lang"] == "fr",  fr_chain),
    (lambda x: x["lang"] == "de",  de_chain),
    en_chain,            # default — must be last
)

# 4) Pass-through with derived fields
enriched = (
    RunnablePassthrough.assign(
        word_count=lambda x: len(x["text"].split()),
        has_url   =lambda x: "http" in x["text"],
    )
    | prompt
    | model
)

# 5) Itemgetter inside chains
get_q = itemgetter("question") | (lambda q: q.upper())

# 6) Custom function as a chain step
@chain
def maybe_complex(input_):
    return complex_chain if input_["complex"] else simple_chain

# 7) Pydantic-typed output
class Answer(BaseModel):
    summary: str
    confidence: float
typed = prompt | model | PydanticOutputParser(Answer)

# 8) Visualization
print(linear.get_graph().to_ascii())
print(linear.get_graph().to_mermaid())
```

### `Runnable` ABI summary

| Method                                  | Signature                                                                |
|-----------------------------------------|--------------------------------------------------------------------------|
| `invoke(input, config=None)`            | sync execution, MUST be implemented                                       |
| `ainvoke(input, config=None)`           | async execution, defaults to `invoke` in a thread pool                    |
| `__or__(other)`                         | coerces `other` and returns a `RunnableSequence`                          |
| `__ror__(other)`                        | reverse pipe support for plain dicts/callables                            |

### Pipe-coercion rules (recap)

| Left side | Right side               | Resulting type after `\|`                    |
|-----------|--------------------------|---------------------------------------------|
| Runnable  | Runnable                 | `RunnableSequence`                          |
| Runnable  | callable                 | `RunnableSequence` (rhs wrapped in `RunnableLambda`) |
| Runnable  | dict[str, Runnable\|callable] | `RunnableSequence` (rhs becomes `RunnableParallel`) |
| dict      | Runnable                 | `RunnableSequence` (lhs becomes `RunnableParallel`) |
| callable  | Runnable                 | `RunnableSequence` (lhs wrapped in `RunnableLambda`) |

## 7. Integration with the rest of Upsonic

UEL is a *foundational* layer; several other modules depend on it.

| Module                              | How it uses UEL |
|-------------------------------------|-----------------|
| `src/upsonic/models/__init__.py`    | The base `Model` class is `class Model(Runnable[Any, Any])`. `Model` overrides `invoke`/`ainvoke` so any provider-specific subclass (OpenAI, Anthropic, Bedrock, …) can be dropped into a UEL chain. The model module also imports `ChatPromptTemplate` (when a chain’s input is a dict it calls the prompt step itself) and the parsers (`StrOutputParser`, `PydanticOutputParser`) from `upsonic.uel.output_parser` for structured-output configuration. |
| `src/upsonic/graphv2/state_graph.py`| `StateGraph` (and `CompiledStateGraph`) inherit from `upsonic.uel.runnable.Runnable`, so a state graph is itself a chain step. The graph module also calls `RunnableGraph` from `upsonic.uel.graph` to render its node/edge structure. |
| `src/upsonic/messages` (consumed)   | UEL imports `ModelRequest`, `ModelResponse`, `SystemPromptPart`, `UserPromptPart`, `TextPart` from `upsonic.messages` (used in `prompt.py` and `output_parser.py`). |
| `src/upsonic/usage`, `src/upsonic/_utils` | `prompt.py` imports `RequestUsage` and `now_utc()` to build placeholder `ModelResponse`s for few-shot AI turns. |

A typical Upsonic agent path is therefore:

```
ChatPromptTemplate (Runnable) | Model (Runnable, provider-specific subclass) | OutputParser (Runnable)
```

…and the same composition is used inside `graphv2` nodes and inside higher-level prebuilt agents (`src/upsonic/prebuilt/...`) wherever a deterministic prompt-model-parse step is wanted alongside the agent loop.

## 8. End-to-end flow / example usage

### 8.1 Linear pipeline trace

```python
from upsonic.uel import ChatPromptTemplate, StrOutputParser
from upsonic import infer_model

prompt  = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model   = infer_model("openai/gpt-4o")
parser  = StrOutputParser()

chain   = prompt | model | parser
result  = chain.invoke({"topic": "bears"})
```

What happens at runtime:

1. `prompt | model` — `Runnable.__or__` runs:
   - `coerce_to_runnable(model)` → returns `model` (already a `Runnable`).
   - Builds `RunnableSequence(steps=[prompt, model])`.
2. `(prompt | model) | parser` — `RunnableSequence.__or__` runs:
   - Returns `RunnableSequence(steps=[prompt, model, parser])` (flat, not nested).
3. `chain.invoke({"topic": "bears"})`:
   - `prompt.invoke({"topic": "bears"})` → `ModelRequest(parts=[UserPromptPart("Tell me a joke about bears")])`.
   - `model.invoke(<ModelRequest>)` → calls the provider, returns a `ModelResponse`.
   - `parser.invoke(<ModelResponse>)` → returns `response.parts[-1].content` as `str`.

### 8.2 Parallel + assign + branch

```python
from upsonic.uel import (
    ChatPromptTemplate, RunnablePassthrough, RunnableBranch,
    RunnableParallel, StrOutputParser, itemgetter,
)
from upsonic import infer_model

model = infer_model("openai/gpt-4o")

summarize  = ChatPromptTemplate.from_template("Summarize: {text}") | model | StrOutputParser()
classify   = ChatPromptTemplate.from_template("Classify topic of: {text}") | model | StrOutputParser()

short_chain = ChatPromptTemplate.from_template("Short reply about: {text}") | model | StrOutputParser()
long_chain  = ChatPromptTemplate.from_template("Long essay about: {text}")   | model | StrOutputParser()

pipeline = (
    RunnablePassthrough.assign(
        word_count=lambda x: len(x["text"].split()),
    )
    | RunnableParallel(
        summary=summarize,
        topic=classify,
        original=itemgetter("text"),
    )
    | RunnableBranch(
        (lambda d: len(d["original"]) < 200, short_chain | (lambda s: {"reply": s})),
        long_chain | (lambda s: {"reply": s}),
    )
)

print(pipeline.invoke({"text": "Once upon a time ..."}))
```

Step-by-step:

1. `RunnablePassthrough.assign(word_count=...)` — input dict is copied, `word_count` is computed against the running dict and stored.
2. `RunnableParallel(summary=…, topic=…, original=itemgetter("text"))` — fans out to 3 concurrent `ainvoke`s via `asyncio.gather`. Returns `{"summary": str, "topic": str, "original": str}`.
3. `RunnableBranch` — checks `len(d["original"]) < 200`; routes to `short_chain` or the default `long_chain`. The branch arms here are themselves `RunnableSequence`s that include a final lambda to wrap the string in a dict.

### 8.3 Visualizing a chain

```python
print(pipeline.get_graph().to_ascii())
```

Produces an indented tree where parallel branches are shown with `├─>` / `└─>` markers and the parallel key name. `pipeline.get_graph().to_mermaid()` returns Mermaid source that can be pasted into any Mermaid renderer; parallel splits are emitted as dotted arrows labelled with the branch key, and sequential edges as `==>`.

### 8.4 Visualizing a `StateGraph`

```python
from upsonic.graphv2.state_graph import StateGraph
g = StateGraph(SomeState).add_node("a", a_fn).add_node("b", b_fn).add_edge("__start__", "a").add_edge("a", "b")
print(g.get_graph().to_mermaid())   # internally uses RunnableGraph
```

Conditional edges in the `StateGraph` are rendered as dashed Mermaid arrows so they’re visually distinct from static edges; `START` and `END` are added when referenced.

### 8.5 Dynamic chains via `@chain`

```python
@chain
def smart_route(data: dict):
    if data["needs_rag"]:
        return rag_chain        # ChainRunnable will invoke this with `data`
    return ChatPromptTemplate.from_template("Answer: {q}") | model | StrOutputParser()

smart_route.invoke({"needs_rag": True, "q": "What is Upsonic?"})
```

`smart_route` itself is a `Runnable` and therefore a valid step in any larger chain.

---

UEL stays small on purpose: it only formalizes how steps connect (`Runnable`, pipe operator, sequence/parallel/branch/passthrough), how arbitrary Python is brought into a chain (`RunnableLambda`, `coerce_to_runnable`, `@chain`), and how chains are introspected (`RunnableGraph`). Everything domain-specific (models, tools, memory, agents, RAG, safety) lives outside this folder and merely *uses* the `Runnable` protocol.
