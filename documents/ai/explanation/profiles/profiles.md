---
name: model-capability-profiles
description: Use when working on per-model capability metadata, provider-specific request/response shaping, or JSON-Schema transformation for LLM providers in Upsonic. Use when a user asks to add a new model provider profile, tweak structured output behavior, configure thinking tags, narrow builtin tools, override `tool_choice='required'`, or debug streaming whitespace/reasoning quirks. Trigger when the user mentions ModelProfile, ModelProfileSpec, DEFAULT_PROFILE, OpenAIModelProfile, GoogleModelProfile, GrokModelProfile, GroqModelProfile, openai_model_profile, anthropic_model_profile, google_model_profile, grok_model_profile, groq_model_profile, deepseek_model_profile, qwen_model_profile, moonshotai_model_profile, nvidia_model_profile, meta_model_profile, amazon_model_profile, harmony_model_profile, mistral_model_profile, cohere_model_profile, OpenAIJsonSchemaTransformer, GoogleJsonSchemaTransformer, InlineDefsJsonSchemaTransformer, JsonSchemaTransformer, supports_tools, supports_json_schema_output, supports_json_object_output, default_structured_output_mode, thinking_tags, ignore_streamed_leading_whitespace, supported_builtin_tools, src/upsonic/profiles/, profile factory, model capability flags, or provider quirks.
---

# `src/upsonic/profiles/` — Model Capability Profiles

This document describes the profiles package that lives at
`src/upsonic/profiles/`. Profiles are the layer that lets Upsonic talk to
many different LLM providers (OpenAI, Anthropic, Google, Mistral, Cohere,
Grok/xAI, Groq, DeepSeek, Qwen, MoonshotAI, NVIDIA, Meta, Amazon, …) using
a single unified `Model` abstraction without hardcoding provider quirks
into the request/response code paths.

A `ModelProfile` is **declarative metadata** that describes the
capabilities and dialect of a model family: does it support tools, does it
support JSON-schema structured output, what JSON-Schema transformations
must be applied to make function-calling work, what tags wrap "thinking"
content, which builtin tools (web search, code execution, MCP) are
permitted, etc. Provider-specific subclasses (`OpenAIModelProfile`,
`GoogleModelProfile`, `GrokModelProfile`, `GroqModelProfile`) extend the
base with extra prefixed fields used only by that provider's `Model`
class.

---

## 1. What this folder is — model capability profiles

The `profiles` package is the **per-model capability registry** of the
Upsonic framework. It answers the question "given a model name like
`gpt-5.1`, `claude-opus-4-5`, `gemini-3-pro` or `qwen-3-coder`, how should
Upsonic shape requests, parse responses, and what optional features are
allowed?".

Three things live here:

1. The `ModelProfile` dataclass (`__init__.py`) — a base, provider-agnostic
   record of capability flags shared by every model.
2. **Provider-specific subclasses** (e.g. `OpenAIModelProfile`,
   `GoogleModelProfile`, `GrokModelProfile`, `GroqModelProfile`) that add
   prefixed fields (`openai_…`, `google_…`, `grok_…`, `groq_…`). The
   prefix rule means profiles can be safely merged across providers when
   one provider re-uses another's wire format (e.g. xAI/Grok uses the
   OpenAI Chat Completions wire format).
3. **Per-provider factory functions** (`openai_model_profile`,
   `anthropic_model_profile`, `google_model_profile`, …) that take a model
   name string and return the appropriate profile instance, encoding all
   model-name-based heuristics (e.g. "is this an o-series reasoning
   model?", "is this a `-search-preview` variant?", "is this `gemini-3+`?").

These profiles are consumed by:

- `Provider.model_profile()` in `src/upsonic/providers/__init__.py` — each
  concrete `Provider` (e.g. `OpenAIProvider`) returns a profile for the
  model name it was asked to instantiate.
- `Model.profile` in `src/upsonic/models/__init__.py` — every `Model`
  caches its profile and uses it during request shaping (which structured
  output mode to use, which JSON-Schema transformer to apply, which
  builtin tools are permitted) and response parsing (which thinking tags
  to look for, whether to ignore leading whitespace in streamed deltas).
- `Agent.__init__(profile=…)` in `src/upsonic/agent/agent.py` — users can
  override the profile per-agent by passing a `ModelProfile` instance,
  which is then attached to the model via `self.model._profile = profile`.

---

## 2. Folder layout

```
src/upsonic/profiles/
├── __init__.py        # ModelProfile base dataclass, ModelProfileSpec, DEFAULT_PROFILE
├── openai.py          # OpenAIModelProfile + openai_model_profile + OpenAIJsonSchemaTransformer
├── anthropic.py       # anthropic_model_profile (uses base ModelProfile, no subclass)
├── google.py          # GoogleModelProfile + google_model_profile + GoogleJsonSchemaTransformer
├── grok.py            # GrokModelProfile + grok_model_profile (xAI / Grok)
├── groq.py            # GroqModelProfile + groq_model_profile
├── mistral.py         # mistral_model_profile (returns None — defaults to base)
├── cohere.py          # cohere_model_profile (returns None — defaults to base)
├── deepseek.py        # deepseek_model_profile (toggles ignore_streamed_leading_whitespace for r1)
├── qwen.py            # qwen_model_profile (special-cases qwen-3-coder)
├── moonshotai.py      # moonshotai_model_profile (ignore_streamed_leading_whitespace=True)
├── nvidia.py          # nvidia_model_profile (NVIDIA NIM / Nemotron)
├── meta.py            # meta_model_profile (Llama family — InlineDefs JSON-Schema transformer)
├── amazon.py          # amazon_model_profile (Titan/Nova on Bedrock — InlineDefs transformer)
└── harmony.py         # harmony_model_profile (OpenAI Harmony Response format)
```

Sister modules referenced from this package:

- `src/upsonic/_json_schema.py` — defines `JsonSchema`, the abstract
  `JsonSchemaTransformer`, and the concrete `InlineDefsJsonSchemaTransformer`
  that several providers re-use.
- `src/upsonic/tools/builtin_tools.py` — defines `AbstractBuiltinTool`
  and `SUPPORTED_BUILTIN_TOOLS` (the universe of builtin tool types like
  `WebSearchTool`, `CodeExecutionTool`, `MCPTool`).
- `src/upsonic/output.py` — defines `StructuredOutputMode`
  (`'tool' | 'native' | 'prompted'`).

---

## 3. Top-level files

### 3.1 `__init__.py` — the `ModelProfile` base class

This module defines the public contract of the package. It exports four
names:

```python
__all__ = [
    'ModelProfile',
    'ModelProfileSpec',
    'DEFAULT_PROFILE',
    'InlineDefsJsonSchemaTransformer',
    'JsonSchemaTransformer',
]
```

#### 3.1.1 `ModelProfile` dataclass

`ModelProfile` is a `@dataclass(kw_only=True)`. Every field has a sensible
default so a bare `ModelProfile()` is the conservative "I know nothing
special about this model" baseline, exposed as `DEFAULT_PROFILE`.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `supports_tools` | `bool` | `True` | Whether the model supports function/tool calling. |
| `supports_json_schema_output` | `bool` | `False` | Native JSON-schema structured output (`NativeOutput`). |
| `supports_json_object_output` | `bool` | `False` | Loose JSON-object mode (e.g. OpenAI `response_format={"type":"json_object"}`). |
| `supports_image_output` | `bool` | `False` | Whether the model can return images. |
| `default_structured_output_mode` | `StructuredOutputMode` | `'tool'` | Which mode `Model._get_output_mode` falls back to when the user did not explicitly choose one. |
| `prompted_output_template` | `str` | "Always respond with a JSON object…" | Template used when structured output is enforced via prompt rather than API. `{schema}` placeholder. |
| `native_output_requires_schema_in_instructions` | `bool` | `False` | If `True`, the prompted template is **also** included for `NativeOutput` mode. |
| `json_schema_transformer` | `type[JsonSchemaTransformer] \| None` | `None` | Provider-specific schema rewriter applied to tool/output schemas. |
| `thinking_tags` | `tuple[str, str]` | `('<think>', '</think>')` | Tags wrapping reasoning/thinking spans in plain-text channels. |
| `ignore_streamed_leading_whitespace` | `bool` | `False` | Workaround for models (Qwen-3 on Ollama, MoonshotAI) that emit empty `<think></think>` blocks or leading whitespace before tool calls during streaming. |
| `supported_builtin_tools` | `frozenset[type[AbstractBuiltinTool]]` | `SUPPORTED_BUILTIN_TOOLS` | Builtin tool types the model is allowed to use (`WebSearchTool`, `CodeExecutionTool`, `MCPTool`, …). Profile factories should narrow this. |

#### 3.1.2 Profile merging — `from_profile()` and `update()`

Profiles compose. The two key methods are:

```python
@classmethod
def from_profile(cls, profile: ModelProfile | None) -> Self: ...

def update(self, profile: ModelProfile | None) -> Self: ...
```

`from_profile(profile)`:

- If `profile` is already an instance of `cls` (e.g. asking
  `OpenAIModelProfile.from_profile(...)` on an existing
  `OpenAIModelProfile`), return it as-is.
- Otherwise, construct a fresh `cls()` and call `.update(profile)` so
  the subclass's prefixed fields take their defaults while shared base
  fields are copied from the input profile.

`update(profile)`:

- Walks the **other** profile's fields and copies any field whose value
  is **not the default** into a new copy of `self`. This is how
  `harmony_model_profile` layers Harmony-specific overrides on top of an
  OpenAI base, or how `Agent.__init__(profile=...)` lets the user mix in
  custom flags without erasing model-derived defaults.

This pattern is used heavily in `src/upsonic/models/openai.py`, e.g.

```python
openai_profile = OpenAIModelProfile.from_profile(_profile)
if not openai_profile.openai_chat_supports_web_search:
    new_tools = _profile.supported_builtin_tools - {WebSearchTool}
    _profile = replace(_profile, supported_builtin_tools=new_tools)
```

#### 3.1.3 (De)serialization

`ModelProfile.to_dict()` / `ModelProfile.from_dict()` and the private
`_to_serializable_dict()` are used by the storage layer to persist the
effective profile alongside session/agent state. Notes:

- `json_schema_transformer` is serialized as the **class name string**
  (e.g. `"OpenAIJsonSchemaTransformer"`), with a small lookup table in
  `from_dict` that lazily imports the matching class to avoid circular
  imports.
- `thinking_tags` round-trips as a list ↔ tuple.
- The `supported_builtin_tools` set is **not** persisted — it is treated
  as a behavior-of-code value, not user state.

#### 3.1.4 `ModelProfileSpec` and `DEFAULT_PROFILE`

```python
ModelProfileSpec = ModelProfile | Callable[[str], ModelProfile | None]
DEFAULT_PROFILE = ModelProfile()
```

A `Model` accepts either a fully-built `ModelProfile` **or** a callable
`(model_name: str) -> ModelProfile | None`. The callable form is the
common case — every per-provider factory function in this folder fits
that signature, so a `Model` can be initialized with
`profile=openai_model_profile` and only resolved when the model name is
known. `DEFAULT_PROFILE` is the fallback if the callable returns `None`.

The resolution lives in `Model.profile` (`src/upsonic/models/__init__.py`):

```python
@cached_property
def profile(self) -> ModelProfile:
    _profile = self._profile
    if callable(_profile):
        _profile = _profile(self.model_name)
    if _profile is None:
        _profile = DEFAULT_PROFILE
    # narrow supported_builtin_tools to the intersection of profile-allowed
    # and model-implemented tools
    model_supported = self.__class__.supported_builtin_tools()
    profile_supported = _profile.supported_builtin_tools
    effective_tools = profile_supported & model_supported
    if effective_tools != profile_supported:
        _profile = replace(_profile, supported_builtin_tools=effective_tools)
    return _profile
```

Two key behaviors:

1. **Late-bound model name** — the factory is called once, with the
   actual `model_name`, and the result is cached.
2. **Tool-set intersection** — the *model class* declares the tools it
   technically implements, the *profile* declares which tools the
   model-as-deployed permits, and the effective set is their
   intersection.

---

## 4. Per-profile files (one per provider)

### 4.1 `openai.py` — OpenAI / OpenAI-compatible

Defines `OpenAIModelProfile(ModelProfile)`, the `openai_model_profile`
factory, and the `OpenAIJsonSchemaTransformer`.

#### `OpenAIModelProfile` extra fields (all `openai_`-prefixed)

| Field | Default | Purpose |
|---|---|---|
| `openai_chat_thinking_field` | `None` | Custom field name for thinking content in the Chat Completions wire format (e.g. `'reasoning'` for Ollama/vLLM, `'reasoning_content'` for DeepSeek). |
| `openai_chat_send_back_thinking_parts` | `'auto'` | How to round-trip thinking content: `'auto'`, `'tags'`, `'field'`, `False`. |
| `openai_supports_strict_tool_definition` | `True` | Whether `tool` definitions can use OpenAI strict mode. |
| `openai_supports_sampling_settings` | `True` | **Deprecated**; superseded by `openai_unsupported_model_settings`. |
| `openai_unsupported_model_settings` | `()` | Explicit list of settings (e.g. `'temperature'`) the model rejects. |
| `openai_supports_tool_choice_required` | `True` | Whether `tool_choice='required'` is accepted. False for MoonshotAI, Qwen-3-Coder, Harmony. |
| `openai_system_prompt_role` | `None` | Role used for the system prompt (`'system'`, `'developer'`, `'user'`). `'user'` for `o1-mini`. |
| `openai_chat_supports_web_search` | `False` | Whether Chat Completions exposes built-in web search. True for `*-search-preview` snapshots. |
| `openai_chat_audio_input_encoding` | `'base64'` | `'base64'` (OpenAI native) vs `'uri'` (some compat providers). |
| `openai_chat_supports_file_urls` | `False` | Whether `file_data` can be a URL (OpenRouter) instead of base64 (OpenAI native). |
| `openai_supports_encrypted_reasoning_content` | `False` | Whether the model returns encrypted reasoning blobs that must be echoed back. |
| `openai_supports_reasoning` | `False` | Whether `reasoning={"effort": ...}` is supported. |
| `openai_supports_reasoning_effort_none` | `False` | Whether `reasoning_effort='none'` is allowed (and sampling params can be combined with it). |
| `openai_responses_requires_function_call_status_none` | `False` | vLLM Responses API quirk (pre-PR-26706). |

`__post_init__` enforces a constraint: if `openai_chat_send_back_thinking_parts == 'field'` then `openai_chat_thinking_field` must be set, otherwise it raises `UserError`.

#### `openai_model_profile(model_name)`

Encodes all model-name heuristics OpenAI requires:

```python
is_gpt_5_1_plus      = model_name.startswith(('gpt-5.1', 'gpt-5.2'))
is_gpt_5             = model_name.startswith('gpt-5') and not is_gpt_5_1_plus
is_o_series          = model_name.startswith('o')
thinking_always_enabled = is_o_series or (is_gpt_5 and 'gpt-5-chat' not in model_name)
supports_reasoning   = thinking_always_enabled or is_gpt_5_1_plus
openai_system_prompt_role = 'user' if model_name.startswith('o1-mini') else None
supports_web_search  = '-search-preview' in model_name
supports_image_output = (is_gpt_5 or is_gpt_5_1_plus or 'o3' in model_name
                         or '4.1' in model_name or '4o' in model_name)
```

It always sets `json_schema_transformer=OpenAIJsonSchemaTransformer`,
`supports_json_schema_output=True`, and `supports_json_object_output=True`
because these are wire-format properties of the OpenAI API (the
`default_structured_output_mode` is still `'tool'` so the API is only
exercised when the user explicitly opts into `NativeOutput`).

#### `OpenAIJsonSchemaTransformer`

A `JsonSchemaTransformer` subclass that rewrites JSON Schema produced by
Pydantic into OpenAI strict-mode-compatible form:

- Strips `title`, `$schema`, `discriminator`, and (in strict) `default`.
- Removes strict-incompatible keys (`minLength`, `maxLength`,
  `patternProperties`, `unevaluatedProperties`, `propertyNames`,
  `minProperties`, `maxProperties`, `unevaluatedItems`, `contains`,
  `minContains`, `maxContains`, `uniqueItems`) and folds them into the
  `description` so the LLM still sees the constraint as text.
- Converts `oneOf` → `anyOf` (OpenAI strict doesn't support `oneOf`).
- Forces `additionalProperties: False` on every object.
- In strict mode, marks every property as `required`.
- Handles recursive schemas (root `$ref`) by inlining the root.
- For `format` strings, only allows the OpenAI-strict-compatible whitelist
  (`date-time`, `time`, `date`, `duration`, `email`, `hostname`, `ipv4`,
  `ipv6`, `uuid`); other formats become `description` text.

### 4.2 `anthropic.py` — Claude

```python
def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    models_that_support_json_schema_output = (
        'claude-haiku-4-5', 'claude-sonnet-4-5', 'claude-sonnet-4-6',
        'claude-opus-4-1', 'claude-opus-4-5', 'claude-opus-4-6',
    )
    supports_json_schema_output = model_name.startswith(models_that_support_json_schema_output)
    return ModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
    )
```

Notable: Anthropic's reasoning channel is wrapped in `<thinking>…</thinking>`
rather than the default `<think>…</think>`. There is **no**
`AnthropicModelProfile` subclass — Claude's API quirks are handled
directly inside the model class (`src/upsonic/models/anthropic.py`).

### 4.3 `google.py` — Gemini / Vertex AI

Defines `GoogleModelProfile(ModelProfile)` with one extra field
(`google_supports_native_output_with_builtin_tools`) plus
`GoogleJsonSchemaTransformer`.

```python
is_image_model = 'image' in model_name
is_3_or_newer  = 'gemini-3' in model_name
return GoogleModelProfile(
    json_schema_transformer=GoogleJsonSchemaTransformer,
    supports_image_output=is_image_model,
    supports_json_schema_output=is_3_or_newer or not is_image_model,
    supports_json_object_output=is_3_or_newer or not is_image_model,
    supports_tools=not is_image_model,
    google_supports_native_output_with_builtin_tools=is_3_or_newer,
)
```

`GoogleJsonSchemaTransformer` rewrites Pydantic JSON Schema into the
OpenAPI 3.0.3 subset Gemini accepts:

- Removes `$schema`, `discriminator`, `examples`, `title` (titles
  trip a known bug in `google-genai`'s python client).
- Replaces `const` with single-element `enum`, inferring `type` from the
  const value (string/bool/integer/number — bool checked before int
  because Python `bool` ⊂ `int`).
- For `string` types, moves `format` into the `description` (Gemini
  doesn't support arbitrary string formats but the LLM still benefits
  from seeing the constraint).
- Strips `exclusiveMinimum`/`exclusiveMaximum` (not yet supported).

### 4.4 `grok.py` — xAI / Grok

Defines `GrokModelProfile(ModelProfile)` (used by both `GrokProvider` and
`XaiProvider`) with two extra fields:

| Field | Default | Purpose |
|---|---|---|
| `grok_supports_builtin_tools` | `False` | Whether the model supports `web_search` / `code_execution` / `mcp` builtin tools. |
| `grok_supports_tool_choice_required` | `True` | Whether the API accepts `tool_choice='required'`. |

```python
grok_supports_builtin_tools = model_name.startswith('grok-4') or 'code' in model_name
supported_builtin_tools = SUPPORTED_BUILTIN_TOOLS if grok_supports_builtin_tools else frozenset()

return GrokModelProfile(
    supports_tools=True,
    supports_json_schema_output=True,
    supports_json_object_output=True,
    grok_supports_builtin_tools=grok_supports_builtin_tools,
    supported_builtin_tools=supported_builtin_tools,
)
```

### 4.5 `groq.py` — Groq

Defines `GroqModelProfile(ModelProfile)` with one extra field:

| Field | Default | Purpose |
|---|---|---|
| `groq_always_has_web_search_builtin_tool` | `False` | Whether the model always has the web search builtin (true for the `compound-*` family). |

```python
return GroqModelProfile(
    groq_always_has_web_search_builtin_tool=model_name.startswith('compound-'),
)
```

### 4.6 `mistral.py` and `cohere.py` — minimal stubs

Both providers currently rely entirely on the base `ModelProfile`
defaults, so the factory just returns `None`:

```python
def mistral_model_profile(model_name: str) -> ModelProfile | None:
    return None

def cohere_model_profile(model_name: str) -> ModelProfile | None:
    return None
```

In `Model.profile`, a `None` result triggers the `DEFAULT_PROFILE`
fallback. The provider's model class still applies any provider-specific
behavior internally; it just doesn't need to override profile flags.

### 4.7 `deepseek.py` — DeepSeek

```python
def deepseek_model_profile(model_name: str) -> ModelProfile | None:
    return ModelProfile(ignore_streamed_leading_whitespace='r1' in model_name)
```

The `r1` reasoning model emits leading whitespace / empty thinking
blocks in streamed deltas; this flag tells the streaming response handler
to skip them so they don't get treated as the first chunk of the final
text answer.

### 4.8 `qwen.py` — Qwen

```python
def qwen_model_profile(model_name: str) -> ModelProfile | None:
    if model_name.startswith('qwen-3-coder'):
        return OpenAIModelProfile(
            json_schema_transformer=InlineDefsJsonSchemaTransformer,
            openai_supports_tool_choice_required=False,
            openai_supports_strict_tool_definition=False,
            ignore_streamed_leading_whitespace=True,
        )
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        ignore_streamed_leading_whitespace=True,
    )
```

Two interesting things:

1. `qwen-3-coder` returns a full `OpenAIModelProfile` (because the
   provider speaks the OpenAI Chat Completions wire format) but disables
   strict tool definitions and `tool_choice='required'`.
2. All Qwen variants use `InlineDefsJsonSchemaTransformer` to inline
   `$defs`/`$ref` because the model handles the flattened form better.

### 4.9 `moonshotai.py` and `nvidia.py`

```python
def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    return ModelProfile(ignore_streamed_leading_whitespace=True)

def nvidia_model_profile(model_name: str) -> ModelProfile | None:
    return ModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )
```

`nvidia.py` covers NVIDIA's own models served via NIM (Nemotron, etc.).
NVIDIA NIM also hosts third-party models — those should resolve to their
respective vendor profiles, not this one.

### 4.10 `meta.py` and `amazon.py` — InlineDefs schema transformer

```python
# meta.py
def meta_model_profile(model_name: str) -> ModelProfile | None:
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)

# amazon.py
def amazon_model_profile(model_name: str) -> ModelProfile | None:
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)
```

Both Llama (Meta) and Titan/Nova (Amazon) function-calling implementations
prefer flattened JSON Schemas without `$defs`/`$ref`, so they reuse the
shared `InlineDefsJsonSchemaTransformer` from `_json_schema.py`.

### 4.11 `harmony.py` — OpenAI Harmony Response format

```python
def harmony_model_profile(model_name: str) -> ModelProfile | None:
    profile = openai_model_profile(model_name)
    return OpenAIModelProfile(
        openai_supports_tool_choice_required=False,
        ignore_streamed_leading_whitespace=True,
    ).update(profile)
```

This is the canonical example of profile composition. It first computes
the regular OpenAI profile (so model-name heuristics like reasoning
support still apply), then layers Harmony-specific overrides on top via
`OpenAIModelProfile.update()`. Order matters: the Harmony profile is the
**target** and the OpenAI profile is the **delta**, because `update()`
copies non-default fields from the delta onto the target. Harmony's own
overrides (tool_choice='required' off, ignore leading whitespace) survive.

---

## 5. Cross-file relationships

```
                      ┌──────────────────────┐
                      │  ModelProfile (base) │  __init__.py
                      └─────────┬────────────┘
                                │ subclassed by
        ┌──────────┬────────────┼──────────────┬─────────────┐
        │          │            │              │             │
 OpenAIModelProfile│       GoogleModelProfile  │       GroqModelProfile
        │     GrokModelProfile                 │
        │                                       │
   openai.py        grok.py    google.py      groq.py
        │
        │ used as base by
        ▼
   harmony.py, qwen.py(qwen-3-coder branch)


   anthropic.py, mistral.py, cohere.py,
   deepseek.py, moonshotai.py, nvidia.py,
   meta.py, amazon.py        ── return base ModelProfile or None


   _json_schema.py
   ├── JsonSchemaTransformer  ← OpenAIJsonSchemaTransformer (openai.py)
   │                          ← GoogleJsonSchemaTransformer (google.py)
   └── InlineDefsJsonSchemaTransformer  ← used by qwen, meta, amazon
```

Key relationships:

- **All provider-specific profiles inherit from `ModelProfile`**. This
  makes `ModelProfile` interchangeable across the whole framework — code
  that reads only base fields (e.g. `Model._get_output_mode`,
  `Model.profile.json_schema_transformer`) doesn't care which subclass
  it received.

- **All extra fields are prefixed by provider** (`openai_*`, `google_*`,
  `grok_*`, `groq_*`). This is enforced by docstring convention, not
  type system, but it lets two profiles be merged via `update()` without
  field collisions even when, e.g., `harmony.py` blends OpenAI-derived
  data into a fresh Harmony profile.

- **Schema transformers live next to the profile that needs them.**
  `OpenAIJsonSchemaTransformer` and `GoogleJsonSchemaTransformer` are
  defined inside `openai.py` and `google.py` respectively. The shared
  `InlineDefsJsonSchemaTransformer` lives in `_json_schema.py` and is
  used by `meta`, `amazon`, and both `qwen` branches.

- **Builtin-tool constraints flow through `supported_builtin_tools`.**
  The default is "all known builtins". Profile factories narrow it
  (Grok narrows to empty for non-grok-4 models). `Model.profile` then
  intersects this with the model class's own implemented set.

- **Model-name heuristics live entirely inside the factory functions.**
  No other code in Upsonic should ever do `if model_name.startswith('o')`
  — it should ask `model.profile.openai_supports_reasoning` instead.

---

## 6. Public API

The package's external surface is intentionally small:

```python
from upsonic.profiles import (
    ModelProfile,                       # base dataclass
    ModelProfileSpec,                   # ModelProfile | (str -> ModelProfile | None)
    DEFAULT_PROFILE,                    # bare ModelProfile() singleton
    InlineDefsJsonSchemaTransformer,    # re-export from _json_schema
    JsonSchemaTransformer,              # re-export from _json_schema
)

# Provider-specific profiles + their factories
from upsonic.profiles.openai     import OpenAIModelProfile, openai_model_profile, OpenAIJsonSchemaTransformer
from upsonic.profiles.google     import GoogleModelProfile, google_model_profile, GoogleJsonSchemaTransformer
from upsonic.profiles.grok       import GrokModelProfile, grok_model_profile
from upsonic.profiles.groq       import GroqModelProfile, groq_model_profile
from upsonic.profiles.anthropic  import anthropic_model_profile
from upsonic.profiles.mistral    import mistral_model_profile
from upsonic.profiles.cohere     import cohere_model_profile
from upsonic.profiles.deepseek   import deepseek_model_profile
from upsonic.profiles.qwen       import qwen_model_profile
from upsonic.profiles.moonshotai import moonshotai_model_profile
from upsonic.profiles.nvidia     import nvidia_model_profile
from upsonic.profiles.meta       import meta_model_profile
from upsonic.profiles.amazon     import amazon_model_profile
from upsonic.profiles.harmony    import harmony_model_profile
```

Common usage shapes:

```python
# 1) End-user override on an Agent — overrides whatever the provider returned.
from upsonic import Agent
from upsonic.profiles import ModelProfile

agent = Agent(
    model="openai/gpt-4o",
    profile=ModelProfile(supports_json_schema_output=True),
)

# 2) Custom subclass field — e.g. force MoonshotAI's tool_choice quirk
from upsonic.profiles.openai import OpenAIModelProfile
agent = Agent(
    model="moonshotai/kimi-k2",
    profile=OpenAIModelProfile(openai_supports_tool_choice_required=False),
)

# 3) Inside a custom Provider — return a factory callable.
from upsonic.providers import Provider
from upsonic.profiles.openai import openai_model_profile

class MyProvider(Provider):
    def model_profile(self, model_name: str) -> ModelProfile | None:
        return openai_model_profile(model_name)
```

### 6.1 Capability matrix (defaults from each factory, ignoring per-name heuristics)

| Provider factory | `supports_tools` | `supports_json_schema_output` | `supports_json_object_output` | `supports_image_output` | `json_schema_transformer` | `thinking_tags` | `ignore_streamed_leading_whitespace` |
|---|---|---|---|---|---|---|---|
| `openai_model_profile` | `True` | `True` | `True` | model-dependent | `OpenAIJsonSchemaTransformer` | `<think>` | `False` |
| `anthropic_model_profile` | `True` | model-dependent | `False` | `False` | `None` | **`<thinking>`** | `False` |
| `google_model_profile` | model-dependent (False for image) | `True` (False for legacy image) | `True` (False for legacy image) | model-dependent | `GoogleJsonSchemaTransformer` | `<think>` | `False` |
| `grok_model_profile` | `True` | `True` | `True` | `False` | `None` | `<think>` | `False` |
| `groq_model_profile` | `True` | `False` | `False` | `False` | `None` | `<think>` | `False` |
| `mistral_model_profile` | `True` (default) | `False` (default) | `False` (default) | `False` | `None` | `<think>` | `False` |
| `cohere_model_profile` | `True` (default) | `False` (default) | `False` (default) | `False` | `None` | `<think>` | `False` |
| `deepseek_model_profile` | `True` | `False` | `False` | `False` | `None` | `<think>` | `True` for `*r1*` |
| `qwen_model_profile` | `True` | `False` (`True` via OpenAIModelProfile branch for qwen-3-coder) | same | `False` | `InlineDefsJsonSchemaTransformer` | `<think>` | `True` |
| `moonshotai_model_profile` | `True` | `False` | `False` | `False` | `None` | `<think>` | `True` |
| `nvidia_model_profile` | `True` | `True` | `True` | `False` | `None` | `<think>` | `False` |
| `meta_model_profile` | `True` | `False` | `False` | `False` | `InlineDefsJsonSchemaTransformer` | `<think>` | `False` |
| `amazon_model_profile` | `True` | `False` | `False` | `False` | `InlineDefsJsonSchemaTransformer` | `<think>` | `False` |
| `harmony_model_profile` | `True` | `True` | `True` | model-dependent | `OpenAIJsonSchemaTransformer` | `<think>` | `True` |

### 6.2 OpenAI per-model heuristics

| Model name pattern | `supports_reasoning` | `openai_system_prompt_role` | `openai_chat_supports_web_search` | `supports_image_output` |
|---|---|---|---|---|
| `gpt-4o*`, `gpt-4.1*` | `False` | `None` | only `*-search-preview` | `True` |
| `o1-mini*` | `True` | `'user'` | no | only `o3*` matches the rule |
| `o*` (other) | `True` | `None` | no | only `o3*` matches the rule |
| `gpt-5*` (not chat) | `True` | `None` | no | `True` |
| `gpt-5-chat*` | `False` | `None` | no | `True` |
| `gpt-5.1*`, `gpt-5.2*` | `True` (with `effort=none` allowed) | `None` | no | `True` |
| `*-search-preview*` | inherits | inherits | `True` | inherits |

---

## 7. Integration with the rest of Upsonic

### 7.1 `Provider` (in `src/upsonic/providers/`)

Each concrete `Provider` subclass implements `model_profile(model_name)`
that delegates to the matching factory in this folder. For example,
`src/upsonic/providers/openai.py`:

```python
from upsonic.profiles.openai import openai_model_profile

class OpenAIProvider(Provider[AsyncOpenAI]):
    def model_profile(self, model_name: str) -> ModelProfile | None:
        return openai_model_profile(model_name)
```

The same pattern is repeated in `providers/anthropic.py`,
`providers/google.py`, `providers/grok.py`, `providers/groq.py`,
`providers/deepseek.py`, `providers/qwen.py`, `providers/moonshotai.py`,
`providers/nvidia.py`, etc. OpenAI-compatible providers (DeepSeek,
MoonshotAI, OpenRouter, Groq, Together, Fireworks, Cerebras, vLLM,
Ollama, LM Studio, NVIDIA NIM, SambaNova, Heroku, OVHcloud, …) typically
mix-and-match: most return either their own factory or a thin wrapper
that calls `openai_model_profile` and post-processes.

### 7.2 `Model` (in `src/upsonic/models/`)

`Model.__init__` accepts `profile: ModelProfileSpec | None = None` and
stores it as `self._profile`. The `profile` cached property resolves the
spec lazily using the model name.

When a request is constructed, `Model._get_output_mode` and
`Model._customize_request_parameters` consult the profile:

```python
# src/upsonic/models/__init__.py
if transformer := self.profile.json_schema_transformer:
    ...
output_mode = self.profile.default_structured_output_mode
...
if params.output_mode == 'native' and not self.profile.supports_json_schema_output:
    raise ...
if params.output_mode == 'tool' and not self.profile.supports_tools:
    raise ...
if params.allow_image_output and not self.profile.supports_image_output:
    raise ...
supported_types = self.profile.supported_builtin_tools
```

The `OpenAIChatModel` reads the OpenAI-specific subclass via
`OpenAIModelProfile.from_profile(self.profile)` so it can access fields
like `openai_system_prompt_role`, `openai_chat_supports_web_search`,
`openai_supports_strict_tool_definition`, `openai_supports_reasoning`,
etc. The same pattern applies in `models/google.py`
(`GoogleModelProfile.from_profile`), `models/xai.py`
(`GrokModelProfile.from_profile`), and so on.

### 7.3 `Agent` (in `src/upsonic/agent/agent.py`)

`Agent.__init__` accepts an optional `profile: ModelProfile`:

```python
profile: Optional["ModelProfile"] = None,
...
if profile:
    self.model._profile = profile
```

This **replaces** the model's profile spec with the user-supplied
instance (the `Model.profile` cached property is recomputed on next
access). The agent also exposes `self.model.profile` to downstream
consumers, and includes it in observability payloads:

```python
model_provider_profile=self.model.profile if self.model else None,
```

### 7.4 Storage / serialization

`ModelProfile.to_dict()` / `from_dict()` are the persistence hooks. The
storage layer (e.g. SQLite, Postgres, JSON file) can round-trip a
profile when checkpointing an agent's full state, with the
`json_schema_transformer` class identified by name and rehydrated via a
small lookup table at load time. The `supported_builtin_tools` set is
considered code-derived and is **not** persisted.

---

## 8. End-to-end flow

The diagram below traces a single agent run from `Agent(...)` through to
the model issuing an HTTP request, showing where the profile is consulted.

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. User                                                              │
│    agent = Agent(model="openai/gpt-5.1", profile=None)               │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. Agent.__init__                                                    │
│    - resolves "openai/gpt-5.1" via providers.infer_provider_class    │
│    - instantiates OpenAIProvider                                     │
│    - instantiates OpenAIChatModel(model_name='gpt-5.1',              │
│                                   provider=OpenAIProvider, ...)      │
│    - if user passed `profile=`, sets self.model._profile = profile   │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3. OpenAIChatModel.__init__                                          │
│    super().__init__(profile=profile or provider.model_profile)       │
│    -> Model._profile = OpenAIProvider.model_profile (a callable)     │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 4. First access to self.model.profile  (cached_property)             │
│    _profile = self._profile                                           │
│    if callable(_profile):                                             │
│        _profile = _profile('gpt-5.1')                                 │
│    # -> openai_model_profile('gpt-5.1')                               │
│    #    => OpenAIModelProfile(                                        │
│    #         json_schema_transformer=OpenAIJsonSchemaTransformer,     │
│    #         supports_json_schema_output=True,                        │
│    #         supports_json_object_output=True,                        │
│    #         supports_image_output=True,                              │
│    #         openai_system_prompt_role=None,                          │
│    #         openai_chat_supports_web_search=False,                   │
│    #         openai_supports_encrypted_reasoning_content=True,        │
│    #         openai_supports_reasoning=True,                          │
│    #         openai_supports_reasoning_effort_none=True,              │
│    #       )                                                          │
│    # intersect supported_builtin_tools with model's implemented set   │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 5. agent.do(task)  → Model.request(...)                              │
│    Model._customize_request_parameters consults:                     │
│      - profile.json_schema_transformer    (apply to tool schemas)    │
│      - profile.default_structured_output_mode  (default 'tool')      │
│      - profile.supports_json_schema_output                           │
│      - profile.supports_tools                                        │
│      - profile.supports_image_output                                 │
│      - profile.supported_builtin_tools                               │
│      - profile.prompted_output_template                              │
│      - profile.native_output_requires_schema_in_instructions         │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 6. OpenAIChatModel layer (provider-specific quirks)                  │
│    profile = OpenAIModelProfile.from_profile(self.profile)            │
│      - profile.openai_system_prompt_role ......... 'system'/'user'    │
│      - profile.openai_supports_strict_tool_definition ...             │
│      - profile.openai_chat_supports_web_search                        │
│      - profile.openai_supports_reasoning                              │
│      - profile.openai_supports_reasoning_effort_none                  │
│      - profile.openai_chat_thinking_field                             │
│      - profile.openai_chat_send_back_thinking_parts                   │
│    HTTP POST to /v1/chat/completions or /v1/responses                 │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 7. Streaming response                                                 │
│    profile.thinking_tags          → split <think>…</think>           │
│    profile.ignore_streamed_leading_whitespace                         │
│        → drop empty <think></think> + leading whitespace              │
│    profile.json_schema_transformer → reverse-validate structured     │
│                                       output if needed                │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 8. Result returned to Agent → Task                                   │
└──────────────────────────────────────────────────────────────────────┘
```

The same flow applies to every other provider: only the **factory
function** changes (e.g. `anthropic_model_profile` is invoked instead of
`openai_model_profile`), and the model class fetches a different
subclass via `from_profile()` if it needs provider-specific fields.

### 8.1 Adding a new provider's profile

To onboard a new model family, you typically:

1. Create `src/upsonic/profiles/<provider>.py`.
2. If the provider has unique knobs not covered by the base `ModelProfile`,
   declare a `@dataclass(kw_only=True) class FooModelProfile(ModelProfile)`
   with `foo_`-prefixed fields.
3. Define `def foo_model_profile(model_name: str) -> ModelProfile | None`
   and encode all model-name heuristics there.
4. In your `Provider` subclass (under `src/upsonic/providers/`), implement
   `model_profile(self, model_name)` to call `foo_model_profile(model_name)`.
5. In your `Model` subclass (under `src/upsonic/models/`), use
   `FooModelProfile.from_profile(self.profile)` whenever you need to read
   the prefixed fields, and consult base fields directly via
   `self.profile.supports_tools`, `self.profile.json_schema_transformer`, etc.

Reuse `InlineDefsJsonSchemaTransformer` from `_json_schema.py` if your
function-calling backend prefers flat schemas. Reuse
`OpenAIJsonSchemaTransformer` if you're an OpenAI-compatible provider
that supports strict mode. Otherwise, write a transformer next to your
profile and reference it via `json_schema_transformer=...`.
