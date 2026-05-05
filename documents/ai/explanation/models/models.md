---
name: models-multi-provider-layer
description: Use when working on the multi-provider LLM integration layer in `src/upsonic/models/`, adding or debugging a provider backend, wiring `Agent.model`, `infer_model`, `Model.request`/`request_stream`/`count_tokens`, `ModelRequestParameters`, `ModelSettings`, `StreamedResponse`, `InstrumentedModel` OpenTelemetry spans, token/cost histograms, the `KnownModelName` literal, `normalize_model_id`, `cached_async_http_client`, `download_item`, `model_registry`, or `model_selector`. Use when a user asks to add an OpenAI-compatible provider, route through Vercel AI Gateway, configure Anthropic cache control or thinking betas, Bedrock Converse cache points, Google Gemini safety/thinking config, xAI gRPC, Outlines local structured output, OpenRouter routing, or troubleshoot provider message mapping, output_mode (text/tool/native/prompted/auto), tool_choice, or streamed final-result detection. Trigger when the user mentions Model, infer_model, ModelRequestParameters, ModelSettings, StreamedResponse, InstrumentedModel, KnownModelName, OpenAIChatModel, OpenAIResponsesModel, AnthropicModel, GoogleModel, BedrockConverseModel, XaiModel, GroqModel, MistralModel, CohereModel, HuggingFaceModel, OutlinesModel, OpenRouterModel, Azure, Cerebras, GitHub Models, Grok, Heroku, LiteLLM, LM Studio, MoonshotAI, Nvidia NIM, Ollama, OVHcloud, SambaNova, Together AI, Vercel AI Gateway, vLLM, ALLOW_MODEL_REQUESTS, ModelHTTPError, RequestUsage, genai_prices, ThinkingPart, or `<think>` tag splitting.
---

# `src/upsonic/models/` — Model Abstraction & Multi-Provider Integration Layer

This document is an exhaustive walkthrough of the `upsonic.models` package. It covers
the abstract `Model` base, the `infer_model` factory, request/response data structures
(`ModelRequestParameters`, `StreamedResponse`, `ModelSettings`), the OpenTelemetry
instrumentation wrapper, the metadata-driven `model_selector`, and every per-provider
implementation: OpenAI (Chat + Responses), Anthropic, Google (Gemini), AWS Bedrock,
Mistral, Cohere, Groq, xAI (native + OpenAI-compatible Grok), HuggingFace, Outlines
(local), and the long list of OpenAI-compatible "thin" wrappers (Azure, Cerebras,
GitHub Models, Heroku, LiteLLM, LM Studio, MoonshotAI, Nvidia NIM, Ollama, OpenRouter,
OVHcloud, SambaNova, Together AI, Vercel AI Gateway, vLLM).

This is the largest and most stable surface area in Upsonic: it is what `Agent.model`
talks to, what `ModelExecutionStep` invokes, what `InstrumentedModel` wraps for traces,
and what the rest of the framework treats as the unified provider API.

---

## 1. What this folder is

The `upsonic.models` package is **the multi-provider LLM integration layer**. Its job
is to expose one Python interface — `Model.request(...)`, `Model.request_stream(...)`,
and `Model.count_tokens(...)` — that all higher-level Upsonic constructs (Agents,
Teams, ModelExecutionStep in `agent/run.py`, the reliability layer, prebuilt agents,
RAG retrievers that ask for embeddings, …) can use without knowing which vendor SDK
is doing the actual HTTPS call.

It does the following heavy lifting:

1. **Defines the `Model` abstract base class** (`models/__init__.py`). This is the
   contract: every concrete model implements `request`, optionally `request_stream`
   and `count_tokens`, and exposes properties like `model_name`, `system`,
   `base_url`, `provider`, and `profile`.
2. **Defines `ModelRequestParameters`** — the per-call configuration object describing
   tools, output mode, output schemas, builtin-tool requests, and image-output flags.
   This is the *parameters* side of a model call. The *settings* side is the
   `ModelSettings` `TypedDict` in `settings.py`.
3. **Defines `StreamedResponse`** — the abstract iterator + final-result-detection
   wrapper used for streamed model output across all providers.
4. **Provides `infer_model(...)`** — a string→Model factory that interprets ids like
   `openai/gpt-4o`, `bedrock/claude-3-5-sonnet:v2`, `gateway/anthropic/claude-haiku-4-5`
   and returns the correct concrete `Model` subclass with the correct `Provider`.
5. **Provides per-provider concrete `Model` classes** — translating Upsonic's
   `ModelMessage`/`ModelResponse` part graph into the wire format each SDK expects
   (Anthropic content blocks, Bedrock Converse messages, Google Gemini `Part`s, OpenAI
   chat completions / Responses items, etc.) and translating the response back.
6. **Provides `InstrumentedModel`** (`instrumented.py`) — a `WrapperModel` that adds
   OpenTelemetry GenAI semantic-convention spans, prompt/completion attributes, token
   histograms, and cost histograms around every call.
7. **Provides a model registry + selector** (`model_registry.py`,
   `model_selector.py`) — high-level metadata about ~20 popular models (capabilities,
   benchmark scores, cost/speed tiers) plus rule-based and LLM-based recommenders that
   suggest a model id for a given task description.
8. **Provides `KnownModelName`** — a giant `Literal[...]` union of every named
   `provider/model` string the framework recognises, used for type-checked agent
   construction.
9. **Provides plumbing helpers** — `cached_async_http_client`, `download_item`
   (used for fetching images / PDFs / videos referenced in user prompts, with SSRF
   protection), `check_allow_model_requests` (a global kill-switch for tests),
   `normalize_model_id` (resolves simplified ids like `bedrock/claude-3-5-sonnet:v2`
   to canonical Bedrock ARNs), and a thinking-tag splitter (`_thinking_part.py`) for
   models that interleave `<think>...</think>` blocks in plain text.

---

## 2. Folder layout

```
src/upsonic/models/
├── __init__.py              (2434 lines) Model base, infer_model, KnownModelName,
│                            ModelRequestParameters, StreamedResponse,
│                            ALLOW_MODEL_REQUESTS, normalize_model_id,
│                            cached_async_http_client, download_item, helpers.
├── _thinking_part.py        (  30 lines) split_content_into_text_and_thinking
│                            tag splitter (<think>...</think> → ThinkingPart).
├── settings.py              ( 196 lines) ModelSettings TypedDict + merge_model_settings
├── wrapper.py               (  90 lines) WrapperModel base for delegation.
├── instrumented.py          ( 562 lines) InstrumentedModel + InstrumentationSettings:
│                            OpenTelemetry GenAI tracing, token + cost histograms.
├── model_registry.py        (1047 lines) ModelMetadata, ModelCapability, ModelTier,
│                            BenchmarkScores; static registry of ~20 popular models.
├── model_selector.py        ( 626 lines) RuleBasedSelector, LLMBasedSelector,
│                            select_model / select_model_async, ModelRecommendation.
│
├── openai.py                (3133 lines) OpenAIChatModel (Chat Completions API) +
│                            OpenAIResponsesModel (Responses API), settings,
│                            OpenAIStreamedResponse, _MapModelResponseContext.
├── anthropic.py             (1477 lines) AnthropicModel (Beta Messages API) +
│                            AnthropicModelSettings (cache control, thinking,
│                            betas, container), AnthropicStreamedResponse.
├── google.py                (1411 lines) GoogleModel (genai/Vertex), GoogleModelSettings
│                            (safety, thinking_config, image config), GeminiStreamedResponse.
├── bedrock.py               (1173 lines) BedrockConverseModel (boto3 Converse API),
│                            cache-point insertion, BedrockStreamedResponse.
├── xai.py                   (1233 lines) XaiModel (native xai-sdk; gRPC).
├── groq.py                  ( 747 lines) GroqModel (Groq SDK; OpenAI-shaped).
├── mistral.py               ( 810 lines) MistralModel (mistralai SDK).
├── cohere.py                ( 356 lines) CohereModel (cohere AsyncClientV2).
├── huggingface.py           ( 540 lines) HuggingFaceModel (huggingface_hub
│                            AsyncInferenceClient).
├── outlines.py              ( 574 lines) OutlinesModel (local Transformers/LlamaCpp/
│                            MLXLM/SGLang/VLLMOffline; structured-output-first).
├── openrouter.py            ( 741 lines) OpenRouterModel (extends OpenAIChatModel
│                            with OpenRouter-specific routing/reasoning settings).
│
├── azure.py                 ( 105 lines) ┐
├── cerebras.py              (  88 lines) │
├── github.py                (  84 lines) │
├── grok.py                  ( 105 lines) │  Thin OpenAI-Chat-compatible wrappers.
├── heroku.py                (  84 lines) │  Each is a 1-line subclass of
├── litellm.py               ( 101 lines) │  OpenAIChatModel that just sets a
├── lmstudio.py              (  83 lines) │  default `provider=` literal.
├── moonshotai.py            (  84 lines) │
├── nvidia.py                ( 101 lines) │
├── ollama.py                ( 102 lines) │
├── ovhcloud.py              (  84 lines) │
├── sambanova.py             (  84 lines) │
├── together.py              (  84 lines) │
├── vercel.py                (  84 lines) │
└── vllm.py                  ( 105 lines) ┘
```

Total: ~18,500 LoC across 33 modules.

---

## 3. Top-level files

### 3.1 `__init__.py` — the contract

This module defines the public surface of `upsonic.models`. Five things matter:

#### 3.1.1 `KnownModelName`

```python
KnownModelName = TypeAliasType(
    'KnownModelName',
    Literal[
        'anthropic/claude-3-5-haiku-20241022',
        'anthropic/claude-haiku-4-5',
        'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0',
        'bedrock/us.anthropic.claude-sonnet-4-6',
        'cerebras/qwen-3-coder-480b',
        'cohere/command-r-plus-08-2024',
        'deepseek/deepseek-reasoner',
        'gateway/openai/gpt-5',                # routed through Vercel AI Gateway
        'google-gla/gemini-2.5-pro',
        'google-vertex/gemini-3-pro-preview',
        'grok/grok-4-fast-reasoning',
        'groq/openai/gpt-oss-120b',
        'heroku/claude-4-5-sonnet',
        'huggingface/Qwen/Qwen3-235B-A22B',
        'mistral/mistral-large-latest',
        'moonshotai/kimi-thinking-preview',
        'openai/gpt-5.1', 'openai/o4-mini',
        'xai/grok-4-fast-reasoning-latest',
        'test',                                # special: used by test/mock provider
        ...
    ],
)
```

Hundreds of literal strings, organised by provider prefix. Anything in this union is
guaranteed to round-trip through `infer_model`. The list also feeds the alias index
used by `normalize_model_id` (see §3.1.6).

Two complementary unions are also defined here:

| Alias | Members | Purpose |
| --- | --- | --- |
| `OpenAIChatCompatibleProvider` | `alibaba`, `azure`, `cerebras`, `deepseek`, `fireworks`, `github`, `grok`, `heroku`, `litellm`, `lmstudio`, `moonshotai`, `nebius`, `nvidia`, `ollama`, `openai`, `openai-chat`, `openrouter`, `ovhcloud`, `sambanova`, `together`, `vercel`, `vllm` | Providers whose endpoint speaks the OpenAI Chat Completions wire format. |
| `OpenAIResponsesCompatibleProvider` | `azure`, `deepseek`, `fireworks`, `grok`, `nebius`, `openrouter`, `ovhcloud`, `sambanova`, `together` | Providers whose endpoint speaks the OpenAI **Responses** API wire format. |

These are how `infer_model` decides which `Model` subclass to instantiate — see
§3.1.7.

#### 3.1.2 `ModelRequestParameters`

Per-call configuration. Distinct from `ModelSettings` (the latter is sampling /
sampling-tier knobs; this one is "what kind of completion are you asking for?"):

```python
@dataclass(repr=False, kw_only=True)
class ModelRequestParameters:
    function_tools: list[ToolDefinition] = field(default_factory=list)
    builtin_tools: list[AbstractBuiltinTool] = field(default_factory=list)

    output_mode: OutputMode = 'text'      # 'text' | 'tool' | 'native' | 'prompted' | 'auto'
    output_object: OutputObjectDefinition | None = None

    output_tools: list[ToolDefinition] = field(default_factory=list)
    prompted_output_template: str | None = None
    allow_text_output: bool = True
    allow_image_output: bool = False

    @cached_property
    def tool_defs(self) -> dict[str, ToolDefinition]: ...
    @cached_property
    def prompted_output_instructions(self) -> str | None: ...
```

Every concrete provider receives this object and translates it into provider-specific
shapes (e.g. Anthropic `betas`, OpenAI `tool_choice`, Bedrock `toolConfig`).

#### 3.1.3 `Model` (abstract base)

```python
class Model(Runnable[Any, Any]):
    _profile: ModelProfileSpec | None = None
    _settings: ModelSettings | None = None
    _memory: Any = None             # UEL chains
    _tools: list[Any] | None = None # UEL bind_tools()
    _response_format: Any = None    # UEL with_structured_output()
    _tool_call_limit: int = 5

    @abstractmethod
    async def request(self, messages, model_settings, model_request_parameters) -> ModelResponse: ...

    async def count_tokens(self, ...) -> RequestUsage:
        raise NotImplementedError(...)         # opt-in per provider

    @asynccontextmanager
    async def request_stream(self, ...) -> AsyncIterator[StreamedResponse]:
        raise NotImplementedError(...)         # opt-in per provider

    def customize_request_parameters(self, params) -> ModelRequestParameters:
        # Applies profile.json_schema_transformer to function_tools, output_tools,
        # output_object — each provider has different JSON-Schema dialect quirks.

    def prepare_request(self, model_settings, model_request_parameters) -> tuple[ModelSettings | None, ModelRequestParameters]:
        # 1. merge_model_settings(self.settings, model_settings)
        # 2. customize_request_parameters(...)
        # 3. de-duplicate builtin_tools by unique_id
        # 4. resolve output_mode == 'auto' → profile.default_structured_output_mode
        # 5. clear irrelevant output_tools / output_object / prompted_output_template
        # 6. set default prompted_output_template if profile requires
        # 7. validate native/tool/image output is supported by profile
        # 8. validate builtin_tools against profile.supported_builtin_tools

    # UEL chain integration (see Runnable):
    def add_memory(self, history=False, memory=None, mode='auto', debug=False) -> Model: ...
    def bind_tools(self, tools, *, tool_call_limit=None) -> Model: ...
    def with_structured_output(self, schema) -> Model: ...
    async def ainvoke(self, input, config=None) -> ModelResponse: ...
    def invoke(self, input, config=None) -> ModelResponse: ...

    # Identity & metadata:
    @property @abstractmethod
    def model_name(self) -> str: ...
    @property
    def model_id(self) -> str:                 # f'{self.system}:{self.model_name}'
    @property
    def label(self) -> str:                    # human-friendly: 'GPT 5', 'Claude Sonnet 4.5'
    @property @abstractmethod
    def system(self) -> str:                   # OTel gen_ai.system value
    @property
    def base_url(self) -> str | None: ...
    @property
    def provider(self) -> Provider[Any] | None: ...
    @property
    def provider_name(self) -> str | None: ...
    @cached_property
    def profile(self) -> ModelProfile: ...
    @classmethod
    def supported_builtin_tools(cls) -> frozenset[type[AbstractBuiltinTool]]:
        return frozenset()                      # subclasses declare what builtins they support

    @staticmethod
    def _get_instructions(messages, params=None) -> str | None:
        # Walks messages in reverse and pulls the most recent ModelRequest.instructions,
        # with a special case to skip mock requests created for tool-result returns.
```

`prepare_request` is the universal pre-flight pass. Every concrete `request(...)`
starts with `model_settings, params = self.prepare_request(model_settings, params)`
before talking to the SDK. The Anthropic and Google models override it to add their
own thinking/output validation.

#### 3.1.4 `StreamedResponse`

```python
@dataclass
class StreamedResponse(ABC):
    model_request_parameters: ModelRequestParameters
    final_result_event: FinalResultEvent | None = None
    provider_response_id: str | None = None
    provider_details: dict[str, Any] | None = None
    finish_reason: FinishReason | None = None

    _parts_manager: ModelResponsePartsManager
    _event_iterator: AsyncIterator[ModelResponseStreamEvent] | None
    _usage: RequestUsage

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        # Pipeline: _get_event_iterator → iterator_with_final_event → iterator_with_part_end
        # Detects when a TextPart, FilePart-image, or output-ToolCallPart starts and emits
        # FinalResultEvent. Emits PartEndEvent before each new part start.

    @abstractmethod
    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]: ...

    def get(self) -> ModelResponse:
        return ModelResponse(parts=self._parts_manager.get_parts(), ..., usage=self.usage())
```

Provider streamed-response classes (`AnthropicStreamedResponse`, `OpenAIStreamedResponse`,
`GeminiStreamedResponse`, `BedrockStreamedResponse`, …) implement `_get_event_iterator`
to translate vendor SSE chunks into `ModelResponseStreamEvent`s using `_parts_manager`.

#### 3.1.5 `ALLOW_MODEL_REQUESTS` kill-switch

```python
ALLOW_MODEL_REQUESTS = True

def check_allow_model_requests() -> None:
    if not ALLOW_MODEL_REQUESTS:
        raise RuntimeError('Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False')

@contextmanager
def override_allow_model_requests(allow: bool) -> Iterator[None]: ...
```

Every concrete `Model.request` should call `check_allow_model_requests()` first. This
is the test-suite hook that prevents accidental real LLM calls when `ALLOW_MODEL_REQUESTS`
has been monkey-patched to `False`.

#### 3.1.6 `normalize_model_id`

Resolves user-friendly aliases to canonical model ids:

```python
'bedrock/claude-3-5-sonnet:v2'   → 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0'
'openai/gpt-4o:latest'           → 'openai/gpt-4o'
'ollama/llama3.1:8b'             → 'ollama/llama3.1:8b'   # unknown provider, pass-through
```

Backed by a cached alias index that is built once from `KnownModelName` by stripping
date suffixes, prefer-non-prefixed-over-regional rules for Bedrock, and a generic
`name[:version]` parser. Called automatically by `infer_model`.

#### 3.1.7 `infer_model` factory

```python
def infer_model(model: Model | KnownModelName | str,
                provider_factory=infer_provider) -> Model:
```

The single public entry point used everywhere `Agent(model='openai/gpt-4o')` and
similar are accepted. Logic:

1. If `os.getenv('LLM_CUSTOM_PROVIDER')` is set, use it as a provider override.
2. If `os.getenv('LLM_MODEL_KEY')` is set (and not the default), use it as the model.
3. If a Celery `bypass_llm_model` kwarg is on the current task, use it.
4. If `model` is already a `Model`, return it.
5. Run `normalize_model_id(model)`.
6. Split on `/` → `(provider_name, model_name)`. If no `/` and the bare model name
   starts with `gpt`, `o1`, `o3`, `claude`, or `gemini`, fall back with a
   deprecation warning.
7. Build a `Provider` via `provider_factory(provider_name)`.
8. Compute the **provider kind**:
   - `gateway/<x>` → strip `gateway/`, then run `normalize_gateway_provider(x)`.
   - Membership in `_OPENAI_CHAT_COMPATIBLE_PROVIDERS` → kind = `'openai-chat'`.
   - Membership in `_GOOGLE_PROVIDERS` (`google`/`google-gla`/`google-vertex`/`gemini`)
     → kind = `'google'`.
   - Otherwise the kind is just the provider name.
9. Dispatch on the kind to a concrete class:

| Kind            | Concrete `Model` class        | Module                |
| --------------- | ------------------------------ | --------------------- |
| `openai-chat`   | `OpenAIChatModel`              | `openai.py`           |
| `openai-responses` | `OpenAIResponsesModel`      | `openai.py`           |
| `openrouter`    | `OpenRouterModel`              | `openrouter.py`       |
| `vercel`        | `VercelModel`                  | `vercel.py`           |
| `together`      | `TogetherModel`                | `together.py`         |
| `sambanova`     | `SambaNovaModel`               | `sambanova.py`        |
| `ovhcloud`      | `OVHcloudModel`                | `ovhcloud.py`         |
| `moonshotai`    | `MoonshotAIModel`              | `moonshotai.py`       |
| `heroku`        | `HerokuModel`                  | `heroku.py`           |
| `github`        | `GitHubModel`                  | `github.py`           |
| `cerebras`      | `CerebrasModel`                | `cerebras.py`         |
| `vllm`          | `VLLMModel`                    | `vllm.py`             |
| `nvidia`        | `NvidiaModel`                  | `nvidia.py`           |
| `ollama`        | `OllamaModel`                  | `ollama.py`           |
| `lmstudio`      | `LMStudioModel`                | `lmstudio.py`         |
| `google`        | `GoogleModel`                  | `google.py`           |
| `groq`          | `GroqModel`                    | `groq.py`             |
| `cohere`        | `CohereModel`                  | `cohere.py`           |
| `mistral`       | `MistralModel`                 | `mistral.py`          |
| `anthropic`     | `AnthropicModel`               | `anthropic.py`        |
| `bedrock`       | `BedrockConverseModel`         | `bedrock.py`          |
| `huggingface`   | `HuggingFaceModel`             | `huggingface.py`      |
| `xai`           | `XaiModel`                     | `xai.py`              |

Anything else raises `UserError(f'Unknown model: {model}')`.

#### 3.1.8 `cached_async_http_client`, `download_item`, `get_user_agent`

- `cached_async_http_client(provider=...)` returns an `httpx.AsyncClient` with a
  `User-Agent` of `upsonic/<version>` and 600s default timeout, cached per-provider.
- `download_item(item, data_format='bytes'|'base64'|'base64_uri'|'text', type_format='mime'|'extension')`
  fetches an `ImageUrl`/`DocumentUrl`/`VideoUrl`/`AudioUrl` (rejecting YouTube
  videos here — only Google handles those natively) using `_ssrf.safe_download`,
  which blocks private IPs and cloud metadata endpoints unless
  `item.force_download == 'allow-local'`.
- `get_user_agent()` is `f'upsonic/{__version__}'`.

### 3.2 `settings.py` — `ModelSettings`

```python
class ModelSettings(TypedDict, total=False):
    max_tokens: int
    temperature: float
    top_p: float
    timeout: float | httpx.Timeout
    parallel_tool_calls: bool
    seed: int
    presence_penalty: float
    frequency_penalty: float
    logit_bias: dict[str, int]
    stop_sequences: list[str]
    extra_headers: dict[str, str]
    extra_body: object


def merge_model_settings(base, overrides) -> ModelSettings | None:
    if base and overrides:
        return base | overrides
    return base or overrides
```

Each provider subclasses this with a `*ModelSettings(ModelSettings, total=False)`
that adds vendor-prefixed keys: `anthropic_thinking`, `openai_reasoning_effort`,
`google_safety_settings`, `bedrock_guardrail_config`, `groq_reasoning_format`,
`xai_logprobs`, etc. The naming convention is *strict*: every vendor-specific key
starts with the vendor prefix so `merge_model_settings` (a plain dict-merge) can
combine settings from differently-typed providers without collisions when the
same dict is used as a default across model swaps.

### 3.3 `_thinking_part.py` — thinking-tag splitter

A 30-line helper used by providers that emit reasoning inside the same text stream
(DeepSeek, Moonshot via OpenAI shape, GPT-OSS via Ollama, HuggingFace inference
endpoints). Given content like:

```
<think>Let me work this out…</think>
The answer is 42.
```

it returns `[ThinkingPart(content='Let me work this out…'), TextPart(content='The answer is 42.')]`.
The `(start_tag, end_tag)` pair comes from `model.profile.thinking_tags`.

### 3.4 `wrapper.py` — `WrapperModel` decorator base

```python
@dataclass(init=False)
class WrapperModel(Model):
    wrapped: Model

    def __init__(self, wrapped: Model | KnownModelName):
        super().__init__()
        self.wrapped = infer_model(wrapped)

    async def request(self, ...): return await self.wrapped.request(...)
    async def count_tokens(self, ...): return await self.wrapped.count_tokens(...)
    @asynccontextmanager
    async def request_stream(self, ...):
        async with self.wrapped.request_stream(...) as s: yield s

    def customize_request_parameters(self, params): return self.wrapped.customize_request_parameters(params)
    def prepare_request(self, settings, params): return self.wrapped.prepare_request(settings, params)

    @property
    def model_name(self): return self.wrapped.model_name
    @property
    def system(self): return self.wrapped.system
    @cached_property
    def profile(self): return self.wrapped.profile
    @property
    def settings(self): return self.wrapped.settings

    def __getattr__(self, item): return getattr(self.wrapped, item)
```

`InstrumentedModel` (and any future fallback / retry / cache wrapper) extends this.
Note the `__getattr__` fall-through — wrappers expose every attribute of the wrapped
model transparently.

### 3.5 `instrumented.py` — OpenTelemetry GenAI instrumentation

Two classes:

#### `InstrumentationSettings`

```python
@dataclass(init=False)
class InstrumentationSettings:
    tracer: Tracer
    logger: Logger
    event_mode: Literal['attributes', 'logs'] = 'attributes'
    include_binary_content: bool = True
    include_content: bool = True
    version: Literal[1, 2, 3, 4] = 4              # DEFAULT_INSTRUMENTATION_VERSION
    use_aggregated_usage_attribute_names: bool = False

    tokens_histogram: Histogram   # gen_ai.client.token.usage    {token}
    cost_histogram: Histogram     # operation.cost                {USD}

    def __init__(self, *, tracer_provider=None, meter_provider=None, ..., version=4): ...

    def messages_to_otel_events(self, messages, parameters=None) -> list[LogRecord]: ...   # v1
    def messages_to_otel_messages(self, messages) -> list[ChatMessage]: ...                # v2-4
    def handle_messages(self, input_messages, response, system, span, parameters=None): ...
    def system_instructions_attributes(self, instructions: str | None) -> dict[str, str]: ...
    def record_metrics(self, response, price_calculation, attributes): ...
```

The settings object holds the **OTel scope** (`upsonic` named tracer/logger/meter at
the package version) and the bucketed token histogram. Token-histogram boundaries
follow the GenAI metrics spec: `(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864)`.

Versions in `version=4` (default) emit:

- `gen_ai.input.messages` — JSON of input messages in OTel `ChatMessage` shape.
- `gen_ai.output.messages` — JSON of the assistant `OutputMessage`.
- `gen_ai.system_instructions` — JSON of the system prompt as a `TextPart`.
- `gen_ai.tool.definitions` — JSON of all `function_tools` + `output_tools`.
- `logfire.json_schema` — array-vs-object hints for Logfire.
- Multimodal content uses `type=uri` (URL refs with mime_type/modality) and
  `type=blob` (inline base64 with mime_type/modality).

Version 1 is legacy event-based and emits `gen_ai.system.message` / `gen_ai.choice`
log events.

#### `InstrumentedModel`

```python
@dataclass(init=False)
class InstrumentedModel(WrapperModel):
    instrumentation_settings: InstrumentationSettings

    async def request(self, messages, model_settings, model_request_parameters):
        prepared_settings, prepared_params = self.wrapped.prepare_request(...)
        with self._instrument(messages, prepared_settings, prepared_params) as finish:
            response = await self.wrapped.request(messages, model_settings, params)
            finish(response, prepared_params)
            return response

    @asynccontextmanager
    async def request_stream(self, ...):
        # Same shape, calls finish(response_stream.get(), prepared_params) on exit.
```

`_instrument(...)` opens a `chat <model_name>` span with attributes:

- `gen_ai.operation.name` = `'chat'`
- `gen_ai.provider.name` (new), `gen_ai.system` (legacy back-compat) = `model.system`
- `gen_ai.request.model` = `model.model_name`
- `server.address`, `server.port` (parsed from `model.base_url`)
- `gen_ai.tool.definitions` (if any tools)
- `gen_ai.request.{max_tokens, top_p, seed, temperature, presence_penalty, frequency_penalty}`
- `model_request_parameters` (full JSON dump)

Inside `finish(response, parameters)`:

- Calls `self.instrumentation_settings.handle_messages(...)` to set
  `gen_ai.input.messages` / `gen_ai.output.messages` / `gen_ai.system_instructions`.
- Sets `response.usage.opentelemetry_attributes()` (input/output/cache/cost token
  counts) on the span.
- Calls `response.cost()`. On success, sets `operation.cost = float(price.total_price)`.
  `LookupError` is silenced ("cost unknown for this provider/model" is common).
  Other exceptions raise `CostCalculationFailedWarning`.
- Sets `gen_ai.response.id` and `gen_ai.response.finish_reasons` if present.
- Renames the span to `chat <request_model>`.

Outside the `with` block, `record_metrics()` records the token histogram and the
cost histogram with attributes `{provider, system, operation_name, request_model,
response_model, gen_ai.token.type='input'|'output'}`. Metrics fire after the span
finishes to avoid double-counting in Logfire.

`instrument_model(model, instrument)` is the public wrapper:

```python
def instrument_model(model: Model, instrument: InstrumentationSettings | bool) -> Model:
    if instrument and not isinstance(model, InstrumentedModel):
        if instrument is True:
            instrument = InstrumentationSettings()
        return InstrumentedModel(model, instrument)
    return model
```

### 3.6 `model_registry.py` — static metadata catalog

A hand-curated catalog of ~20 popular models with rich metadata. Three enums and
two dataclasses:

```python
class ModelCapability(str, Enum):
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    MATHEMATICS = "mathematics"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"
    MULTILINGUAL = "multilingual"
    VISION = "vision"
    AUDIO = "audio"
    LONG_CONTEXT = "long_context"
    FAST_INFERENCE = "fast_inference"
    COST_EFFECTIVE = "cost_effective"
    FUNCTION_CALLING = "function_calling"
    STRUCTURED_OUTPUT = "structured_output"
    ETHICAL_SAFETY = "ethical_safety"
    RESEARCH = "research"
    PRODUCTION = "production"


class ModelTier(str, Enum):
    FLAGSHIP = "flagship"     # GPT-4o, Claude Opus, Gemini Pro, Grok-4
    ADVANCED = "advanced"     # Claude Sonnet, Llama 70B, Mistral Large
    STANDARD = "standard"     # DeepSeek Chat
    FAST = "fast"             # GPT-4o-mini, Haiku, Gemini Flash
    SPECIALIZED = "specialized"  # o1-pro, o1-mini, DeepSeek-R1


@dataclass
class BenchmarkScores:
    mmlu: float | None        # General knowledge / multi-task
    gpqa: float | None        # Graduate-level
    math: float | None        # MATH benchmark
    gsm8k: float | None       # Grade-school math
    aime: float | None        # American Invitational Math Exam
    humaneval: float | None   # Python coding
    mbpp: float | None        # Mostly Basic Python
    drop: float | None        # Reading comprehension
    mgsm: float | None        # Multilingual grade-school math
    arc_challenge: float | None  # AI2 Reasoning Challenge

    def overall_score(self) -> float:
        # Weighted: MMLU x2, HumanEval x1.5, MATH x1.5, the rest x1


@dataclass
class ModelMetadata:
    name: str
    provider: str
    tier: ModelTier
    release_date: str
    capabilities: list[ModelCapability]
    context_window: int = 8192
    benchmarks: BenchmarkScores | None = None
    strengths: list[str]
    ideal_for: list[str]
    limitations: list[str]
    cost_tier: int = 5     # 1=cheapest, 10=most expensive
    speed_tier: int = 5    # 1=slowest, 10=fastest
    notes: str = ""
```

The registry currently includes `GPT_4O`, `GPT_4O_MINI`, `O1_PRO`, `O1_MINI`,
`CLAUDE_4_OPUS`, `CLAUDE_3_7_SONNET`, `CLAUDE_3_5_HAIKU`, `GEMINI_2_5_PRO`,
`GEMINI_2_5_FLASH`, `LLAMA_3_3_70B`, `DEEPSEEK_R1`, `DEEPSEEK_CHAT`, `QWEN_3_235B`,
`MISTRAL_LARGE`, `MISTRAL_SMALL`, `COHERE_COMMAND_R_PLUS`, `GROK_4` and exposes
each one under multiple alias keys (with and without provider prefix, with/without
date suffix, plus Azure mirrors of OpenAI metadata).

Three lookup helpers:

```python
def get_model_metadata(model_name: str) -> ModelMetadata | None: ...
def get_models_by_capability(c: ModelCapability) -> list[ModelMetadata]: ...
def get_models_by_tier(t: ModelTier) -> list[ModelMetadata]: ...
def get_top_models(n: int = 10, by_benchmark: str | None = None) -> list[ModelMetadata]: ...
```

### 3.7 `model_selector.py` — recommender system

Two selectors that turn a free-form task description into a `ModelRecommendation`.

#### `ModelRecommendation`

```python
class ModelRecommendation(BaseModel):
    model_name: str
    reason: str
    confidence_score: float       # 0..1
    alternative_models: list[str]
    estimated_cost_tier: int      # 1..10
    estimated_speed_tier: int     # 1..10
    selection_method: Literal["llm", "rule_based"]
```

#### `SelectionCriteria`

```python
class SelectionCriteria(BaseModel):
    requires_reasoning: bool | None
    requires_code_generation: bool | None
    requires_math: bool | None
    requires_creative_writing: bool | None
    requires_vision: bool | None
    requires_audio: bool | None
    requires_long_context: bool | None
    prioritize_speed: bool = False
    prioritize_cost: bool = False
    prioritize_quality: bool = False
    max_cost_tier: int | None
    min_context_window: int | None
    preferred_provider: str | None
    require_open_source: bool = False
    require_production_ready: bool = False
    required_capabilities: list[ModelCapability]
```

#### `RuleBasedSelector`

A keyword-based scorer:

1. **Keyword analysis** — searches the task description for capability keywords
   (e.g. "reason", "analyze", "complex" → `REASONING`; "image", "photo", "visual"
   → `VISION`; etc.) and assigns each capability a normalised score.
2. **Per-model scoring** for every entry in `MODEL_REGISTRY`:
   - Capability match (×100 weight per matched capability).
   - Hard requirements (`requires_reasoning`, `min_context_window`,
     `max_cost_tier`, `require_open_source`) → score 0 if not met.
   - `prioritize_cost` → `+10*(10-cost_tier)`, `prioritize_speed` →
     `+10*speed_tier`, `prioritize_quality` → `+benchmarks.overall_score()*2`.
   - Tier bonus (FLAGSHIP=100, ADVANCED=80, STANDARD=60, FAST=40, SPECIALIZED=70).
3. Picks the top score, computes confidence from the gap between #1 and #2,
   returns a `ModelRecommendation` with up to 5 alternatives.

#### `LLMBasedSelector`

Sends the registry summary and task to GPT-4o (or any provided agent) with a
carefully-structured prompt instructing the LLM to return a JSON
`ModelRecommendation`. Falls back to `RuleBasedSelector` on any error.

Public API:

```python
def select_model(task_description, criteria=None, *, use_llm=False, agent=None,
                 default_model='openai/gpt-4o') -> ModelRecommendation: ...
async def select_model_async(...): ...
```

---

## 4. Per-provider walkthrough

### 4.1 The OpenAI-compatible "thin wrapper" tier

Fifteen providers share one wire format (OpenAI Chat Completions) and therefore
collapse into a tiny wrapper around `OpenAIChatModel`. Each follows the same
template:

```python
class GitHubModelSettings(OpenAIChatModelSettings, total=False):
    """Settings for GitHub Models API model requests."""

class GitHubModel(OpenAIChatModel):
    """Convenience wrapper around OpenAIChatModel with provider='github' default."""
    def __init__(self, model_name, *, provider: ... = 'github',
                 profile=None, settings=None):
        super().__init__(model_name=model_name, provider=provider,
                         profile=profile, settings=settings)
```

The full table:

| File              | Class                  | Default `provider=` | Notes / endpoint |
| ----------------- | ---------------------- | --------------------- | ---------------- |
| `azure.py`        | `AzureModel`           | `'azure'`             | Azure OpenAI; supports `vllm`, `nvidia` providers too. |
| `cerebras.py`     | `CerebrasModel`        | `'cerebras'`          | Doesn't support `frequency_penalty`, `logit_bias`, `presence_penalty`, `parallel_tool_calls`, `service_tier`. |
| `github.py`       | `GitHubModel`          | `'github'`            | GitHub Models API marketplace. |
| `grok.py`         | `GrokModel`            | `'grok'`              | xAI Grok via OpenAI-compatible endpoint. |
| `heroku.py`       | `HerokuModel`          | `'heroku'`            | Heroku Inference. |
| `litellm.py`      | `LiteLLMModel`         | `'litellm'`           | Calls a LiteLLM proxy. |
| `lmstudio.py`     | `LMStudioModel`        | `'lmstudio'`          | Default endpoint `http://localhost:1234/v1`. |
| `moonshotai.py`   | `MoonshotAIModel`      | `'moonshotai'`        | Kimi (`moonshot-v1-128k`, `kimi-thinking-preview`). |
| `nvidia.py`       | `NvidiaModel`          | `'nvidia'`            | NVIDIA NIM. |
| `ollama.py`       | `OllamaModel`          | `'ollama'`            | Default endpoint `http://localhost:11434/api`. |
| `ovhcloud.py`     | `OVHcloudModel`        | `'ovhcloud'`          | OVHcloud AI Endpoints. |
| `sambanova.py`    | `SambaNovaModel`       | `'sambanova'`         | SambaNova Cloud. |
| `together.py`     | `TogetherModel`        | `'together'`          | Together AI marketplace. |
| `vercel.py`       | `VercelModel`          | `'vercel'`            | Vercel AI Gateway. |
| `vllm.py`         | `VLLMModel`            | `'vllm'`              | vLLM OpenAI-compatible endpoint, default `http://localhost:8000/v1`. |

These wrappers exist so users get nice import paths (`from upsonic.models.cerebras
import CerebrasModel`) and so that each provider can grow vendor-specific tweaks
later without breaking imports — but today they share 100% of the runtime
behaviour with `OpenAIChatModel`.

### 4.2 OpenAI (`openai.py`) — Chat + Responses

The largest single file (~3133 lines). Two concrete model classes:

#### `OpenAIChatModel`

```python
@dataclass(init=False)
class OpenAIChatModel(Model):
    client: AsyncOpenAI
    _model_name: OpenAIModelName
    _provider: Provider[AsyncOpenAI]

    def __init__(self, model_name, *, provider='openai', profile=None,
                 system_prompt_role=None, settings=None): ...
```

Provider literal: `OpenAIChatCompatibleProvider | 'openai' | 'openai-chat' | 'gateway' | Provider[AsyncOpenAI]`.
`'gateway'` is silently rewritten to `'gateway/openai'`.

`OpenAIChatModelSettings` (the canonical one inherited by every "thin wrapper"
above) adds:

| Setting | Type | Notes |
| --- | --- | --- |
| `openai_reasoning_effort` | `'low' \| 'medium' \| 'high'` | For reasoning models (o-series, GPT-5+). |
| `openai_logprobs` | `bool` | Include token logprobs in `provider_details`. |
| `openai_top_logprobs` | `int` | Top-N logprobs per token. |
| `openai_store` | `bool \| None` | OpenAI conversation storage. |
| `openai_user` | `str` | End-user id for abuse monitoring. |
| `openai_service_tier` | `'auto' \| 'default' \| 'flex' \| 'priority'` | |
| `openai_prediction` | `ChatCompletionPredictionContentParam` | Predicted outputs. |
| `openai_prompt_cache_key` | `str` | Prompt cache routing key. |
| `openai_prompt_cache_retention` | `'in_memory' \| '24h'` | Extended caching. |
| `openai_continuous_usage_stats` | `bool` | Cumulative usage in stream. |

Request flow (`_completions_create`):

1. `_get_tools` + `_get_web_search_options` (if any `WebSearchTool` builtin and
   the profile sets `openai_chat_supports_web_search`).
2. `tool_choice` = `'required'` if `not allow_text_output` and the profile sets
   `openai_supports_tool_choice_required`, else `'auto'`, else `None`.
3. `_map_messages` — converts each `ModelMessage` to OpenAI message params,
   including `chat.ChatCompletionContentPart{Text,Image,InputAudio}Param`,
   `File` (PDFs), `tool_calls`, `assistant.refusal`. Reasoning content for gpt-oss
   / Moonshot / DeepSeek goes into a custom field per `profile.openai_chat_thinking_field`.
4. `response_format` from `output_mode == 'native'` (json_schema) or `'prompted'`
   (json_object).
5. `_drop_sampling_params_for_reasoning` — for o-series/GPT-5/GPT-5.1+ models with
   reasoning enabled, drop `temperature/top_p/presence_penalty/...` (warn user).
6. `_drop_unsupported_params` — for Cerebras and other restricted providers.
7. Build `extra_headers` with `User-Agent: upsonic/<version>`.
8. Call `client.chat.completions.create(...)`.
9. Special-case `_check_azure_content_filter` for Azure 400s with content-filter
   errors → return a synthetic `ModelResponse` with `finish_reason='content_filter'`.

Response processing (`_process_response`):

- If `choice.message.refusal` → `finish_reason='content_filter'`, empty parts.
- Otherwise: `_process_thinking` extracts reasoning from `choice.message.reasoning`
  / `choice.message.reasoning_content` / a custom field (per profile).
- `split_content_into_text_and_thinking(choice.message.content, profile.thinking_tags)`
  splits inline `<think>` blocks.
- Function tool calls become `ToolCallPart`s.
- `provider_details` includes `finish_reason`, `timestamp`, and may include
  Azure content-filter results.

Streaming: `OpenAIStreamedResponse` yields `ChatCompletionChunk`s and uses
`_parts_manager` to accumulate text/thinking/tool-call deltas.

#### `OpenAIResponsesModel`

A second model class for the OpenAI Responses API (more capable, supports
encrypted reasoning content + many builtin tools). Uses `client.responses.create`
instead of `client.chat.completions.create`. Adds settings:

| Setting | Type |
| --- | --- |
| `openai_builtin_tools` | `Sequence[FileSearchToolParam | WebSearchToolParam | ComputerToolParam]` |
| `openai_reasoning_summary` | `'detailed' \| 'concise' \| 'auto'` |
| `openai_send_reasoning_ids` | `bool` |
| `openai_truncation` | `'disabled' \| 'auto'` |
| `openai_text_verbosity` | `'low' \| 'medium' \| 'high'` |
| `openai_previous_response_id` | `'auto' \| str` |
| `openai_include_code_execution_outputs` | `bool` |
| `openai_include_web_search_sources` | `bool` |
| `openai_include_file_search_results` | `bool` |
| `openai_include_raw_annotations` | `bool` |

Builtin tools translated:
- `WebSearchTool` → `responses.WebSearchToolParam`.
- `FileSearchTool` → `responses.FileSearchToolParam` with `vector_store_ids`.
- `CodeExecutionTool` → `{type: 'code_interpreter', container: {type: 'auto'}}`.
- `MCPServerTool` → `responses.tool_param.Mcp(server_label, server_url|connector_id, allowed_tools, headers)`.
- `ImageGenerationTool` → `responses.tool_param.ImageGeneration(...)` with size
  derived from `_resolve_openai_image_generation_size(tool)` (mapping `aspect_ratio`
  '1:1'→'1024x1024', '2:3'→'1024x1536', '3:2'→'1536x1024').

Returned items can be `ResponseFunctionToolCall`, `ResponseCodeInterpreterToolCall`,
`ResponseFunctionWebSearch`, `ImageGenerationCall`, `ResponseFileSearchToolCall`,
`McpCall`, `McpListTools` — each translated into `(BuiltinToolCallPart,
BuiltinToolReturnPart, FilePart?)` triples by the corresponding `_map_*_call`
helpers.

`openai_previous_response_id == 'auto'` walks back through messages to find the
most recent `provider_response_id` for `system == 'openai'`, then trims the
message list and lets the server reuse its conversation state.

#### Supported builtin tools

| Tool                  | `OpenAIChatModel` | `OpenAIResponsesModel` |
| --------------------- | :---------------: | :--------------------: |
| `WebSearchTool`       | ✓ (only if profile says so) | ✓ |
| `FileSearchTool`      | —                 | ✓ |
| `CodeExecutionTool`   | —                 | ✓ |
| `MCPServerTool`       | —                 | ✓ |
| `ImageGenerationTool` | —                 | ✓ |

### 4.3 Anthropic (`anthropic.py`)

`AnthropicModel` uses the Anthropic Beta Messages API
(`client.beta.messages.create`). Provider literal: `'anthropic' | 'gateway' | Provider[AsyncAnthropicClient]`.

`AnthropicModelSettings` adds rich cache + thinking knobs:

| Setting | Type | Notes |
| --- | --- | --- |
| `anthropic_metadata` | `BetaMetadataParam` | `user_id` for tagging. |
| `anthropic_thinking` | `BetaThinkingConfigParam` | Extended thinking. |
| `anthropic_cache_tool_definitions` | `bool \| '5m' \| '1h'` | Cache last tool def. |
| `anthropic_cache_instructions` | `bool \| '5m' \| '1h'` | Cache system prompt. |
| `anthropic_cache_messages` | `bool \| '5m' \| '1h'` | Cache last user msg. |
| `anthropic_effort` | `'low' \| 'medium' \| 'high' \| 'max' \| None` | |
| `anthropic_container` | `BetaContainerParams \| Literal[False]` | Reuse / fresh. |
| `anthropic_betas` | `list[AnthropicBetaParam]` | Manual feature flags. |

Supported builtin tools:
`{WebSearchTool, CodeExecutionTool, WebFetchTool, MemoryTool, MCPServerTool}`.

Per-call flow:

1. `prepare_request` — special-cases the combination of `anthropic_thinking` +
   output tools (Anthropic doesn't allow `tool_choice=any` with thinking; auto-
   switches to `NativeOutput` or `PromptedOutput` and raises if the user explicitly
   asked for tool output without `allow_text_output`). Also forces
   `output_object.strict=True` when `output_mode == 'native'`.
2. `_get_tools` — maps each `ToolDefinition` to `BetaToolParam`. Adds
   `cache_control` to the *last* tool if `anthropic_cache_tool_definitions` is set.
3. `_add_builtin_tools` — translates each Upsonic builtin to its Anthropic Beta
   counterpart and accumulates required beta feature names:
    - `WebSearchTool` → `BetaWebSearchTool20250305Param` (with optional
      `user_location`, `allowed_domains`, `blocked_domains`, `max_uses`).
    - `CodeExecutionTool` → `BetaCodeExecutionTool20250522Param` + beta
      `code-execution-2025-05-22`.
    - `WebFetchTool` → `BetaWebFetchTool20250910Param` + beta `web-fetch-2025-09-10`.
    - `MemoryTool` → `BetaMemoryTool20250818Param` + beta `context-management-2025-06-27`.
    - `MCPServerTool` → adds an entry to the separate `mcp_servers` list with
      `BetaRequestMCPServerURLDefinitionParam`, plus beta `mcp-client-2025-04-04`.
4. `_infer_tool_choice` — `{type: 'any'}` if `not allow_text_output`, else
   `{type: 'auto'}`; sets `disable_parallel_tool_use` from
   `parallel_tool_calls`.
5. `_map_message` — full multimodal mapping including:
    - System prompts joined with `\n\n`.
    - User content can include `BetaImageBlockParam`, `BetaRequestDocumentBlockParam`
      (PDF/text), URL-based images and PDFs (with optional `force_download`).
    - `CachePoint` in user content → `_add_cache_control_to_last_param`.
    - Assistant messages preserve `BetaThinkingBlockParam` /
      `BetaRedactedThinkingBlockParam` (using Upsonic's `ThinkingPart.signature`).
    - Builtin tool returns are mapped back to `BetaWebSearchToolResultBlockParam`
      / `BetaCodeExecutionToolResultBlockParam` / `BetaWebFetchToolResultBlockParam`
      / `BetaMCPToolResultBlock`.
6. `_limit_cache_points` — Anthropic enforces a max of **4** cache points per
   request. Counts cache points in system_prompt + tools, calculates the remaining
   budget, then walks message content blocks newest→oldest, removing
   `cache_control` from older blocks if the budget is exceeded. Raises
   `UserError` if system+tools alone exceed 4.
7. `_get_betas_and_extra_headers` — accumulates auto-betas (`structured-outputs-2025-11-13`
   for native output / strict tools), user-supplied `anthropic_betas`, and any
   `anthropic-beta` header from `extra_headers`.
8. `_get_container` — reuses `provider_details['container_id']` from previous
   responses unless `anthropic_container=False`.
9. `client.beta.messages.create(...)`.

Streaming: `AnthropicStreamedResponse` consumes
`BetaRawMessageStartEvent` / `BetaRawContentBlockStartEvent` /
`BetaRawContentBlockDeltaEvent` (BetaTextDelta, BetaThinkingDelta,
BetaSignatureDelta, BetaInputJSONDelta, BetaCitationsDelta) /
`BetaRawMessageDeltaEvent` / stop events, calling `_parts_manager.handle_*`
helpers.

`count_tokens` is implemented via `client.beta.messages.count_tokens` (raises if
the client is `AsyncAnthropicBedrock` since Bedrock doesn't support it).

### 4.4 Google (`google.py`)

`GoogleModel` uses the official `google-genai` SDK, which abstracts both
Gemini API (`google-gla`) and Vertex AI (`google-vertex`). Provider literal:
`'google-gla' | 'google-vertex' | 'gateway' | Provider[Client]`. Default is
`'google-gla'`.

`GoogleModelSettings`:

| Setting | Type |
| --- | --- |
| `google_safety_settings` | `list[SafetySettingDict]` |
| `google_thinking_config` | `ThinkingConfigDict` |
| `google_labels` | `dict[str, str]` (Vertex only) |
| `google_video_resolution` | `MediaResolution` |
| `google_cached_content` | `str` |
| `google_logprobs` | `bool` (Vertex only, non-streaming) |
| `google_top_logprobs` | `int` |

Supported builtin tools:
`{WebSearchTool, CodeExecutionTool, FileSearchTool, WebFetchTool, ImageGenerationTool}`.

Builtin-tool mapping:
- `WebSearchTool` → `ToolDict(google_search=GoogleSearchDict())`.
- `WebFetchTool` → `ToolDict(url_context=UrlContextDict())`.
- `CodeExecutionTool` → `ToolDict(code_execution=ToolCodeExecutionDict())`.
- `FileSearchTool` → `ToolDict(file_search=FileSearchDict(file_search_store_names=...))`.
- `ImageGenerationTool` → builds `ImageConfigDict` (size '1K'/'2K'/'4K',
  `aspect_ratio`, Vertex-only `output_mime_type` and `output_compression_quality`).

Function tools and builtin tools are mutually exclusive (Google's restriction).

Output mode handling:
- `'native'`: sets `response_mime_type='application/json'` and a
  `response_json_schema` from `_map_response_schema`. Cannot be combined with
  function tools.
- `'prompted'` (only when no tools): `response_mime_type='application/json'` and
  the schema goes into instructions.

`prepare_request` validates that builtin_tools and output_tools aren't combined
unless `google_supports_native_output_with_builtin_tools` (per profile).

Message mapping:
- System prompts go into a separate `system_instruction: ContentDict(role='user', parts=[...])`.
- User parts: text, `BinaryContent` (inline_data), `VideoUrl` for YouTube
  (passed through as `file_data`), GCS `gs://` URIs (Vertex only), other
  `FileUrl` (downloaded for `google-gla`, passed through for Vertex).
- `CachePoint` is silently ignored (Google requires pre-created cache objects
  via `google_cached_content`).
- Assistant `ThinkingPart.signature` round-trips as `part['thought_signature']`
  (base64-encoded). Function calls require a thought_signature on the *first*
  call, with `b'skip_thought_signature_validator'` as a sentinel.

Tool result format: function responses become
`ContentDict{role:'user', parts:[{function_response: {name, response, id}}]}`.

A Gemini-specific bug-workaround: contents containing `function_response` parts
get split into separate `role='user'` entries when interleaved with non-function
parts.

Response processing: `_process_response_from_parts` turns Google `Part`s into
Upsonic parts and prepends synthetic `(BuiltinToolCallPart, BuiltinToolReturnPart)`
pairs from `grounding_metadata` (web search), `grounding_metadata` (file search),
and `url_context_metadata` (web fetch).

Streaming: `GeminiStreamedResponse` handles incremental file-search /
code-execution detection and emits the synthetic builtin-tool parts after the
text delta (because Google streams grounding metadata after the text it
references).

`count_tokens` is implemented via `client.aio.models.count_tokens`. Vertex
supports the full config; google-gla only accepts a subset.

### 4.5 AWS Bedrock (`bedrock.py`)

`BedrockConverseModel` uses the boto3 `bedrock-runtime` Converse API. Provider
literal: `'bedrock' | 'gateway' | Provider[BaseClient]`.

`BedrockModelSettings`:

| Setting | Type |
| --- | --- |
| `bedrock_guardrail_config` | `GuardrailConfigurationTypeDef` |
| `bedrock_performance_configuration` | `PerformanceConfigurationTypeDef` |
| `bedrock_request_metadata` | `dict[str, str]` |
| `bedrock_additional_model_response_fields_paths` | `list[str]` |
| `bedrock_prompt_variables` | `Mapping[str, PromptVariableValuesTypeDef]` |
| `bedrock_additional_model_requests_fields` | `Mapping[str, Any]` |
| `bedrock_cache_tool_definitions` | `bool` |
| `bedrock_cache_instructions` | `bool` |
| `bedrock_cache_messages` | `bool` |
| `bedrock_service_tier` | `ServiceTierTypeDef` |

Supported builtin tools: `{CodeExecutionTool}` (mapped to
`{'systemTool': {'name': 'nova_code_interpreter'}}` for Nova models).

Cache-point handling is delicate. AWS rejects cache points that *immediately
follow* documents/videos (but allows them after images). The helper
`_insert_cache_point_before_trailing_documents` walks back from the end of a
content list, finds the start of the trailing contiguous document/video group,
and inserts the cache point *before* that group. `_limit_cache_points` enforces
the same 4-cache-point ceiling Anthropic does.

Bedrock's API is sync-only, so every call uses `anyio.to_thread.run_sync(
functools.partial(self.client.converse, **params))` to avoid blocking the
event loop.

Message mapping:
- Sequential user messages get merged into one (Bedrock disallows consecutive
  user turns).
- `BinaryContent` for documents / images / videos is sent as base64 bytes.
- `s3://` URLs are sent as `S3LocationTypeDef`; everything else is downloaded
  and inlined.
- Tool results are sent as `text` or `json` based on
  `BedrockModelProfile.bedrock_tool_result_format`.
- `ThinkingPart.signature` round-trips as `reasoningContent.reasoningText` or
  `reasoningContent.redactedContent` if `bedrock_send_back_thinking_parts` is set.

`count_tokens` uses the `count_tokens` API (only works on a subset of Bedrock
models).

Streaming: `BedrockStreamedResponse` consumes `EventStream` events
(`messageStart`, `contentBlockStart`, `contentBlockDelta`, `contentBlockStop`,
`messageStop`, `metadata`) via `anyio.to_thread.run_sync` per chunk, since
botocore eventstreams are sync iterators.

### 4.6 xAI native SDK (`xai.py`)

`XaiModel` uses the official `xai-sdk` (gRPC-based, distinct from the
OpenAI-shaped `GrokModel` in `grok.py`).

`XaiModelSettings`:

| Setting | Type |
| --- | --- |
| `xai_logprobs` | `bool` |
| `xai_top_logprobs` | `int` |
| `xai_user` | `str` |
| `xai_store_messages` | `bool` |
| `xai_previous_response_id` | `str` |
| `xai_include_encrypted_content` | `bool` |
| `xai_include_code_execution_output` | `bool` |
| `xai_include_web_search_output` | `bool` |
| `xai_include_inline_citations` | `bool` |
| `xai_include_mcp_output` | `bool` |

Settings mapping table `_XAI_MODEL_SETTINGS_MAPPING` translates Upsonic
keys to xAI SDK kwarg names (e.g. `stop_sequences` → `stop`, `xai_logprobs` →
`logprobs`).

Builtin tools: `WebSearchTool`, `CodeExecutionTool`, `MCPServerTool` translated
to `xai_sdk.tools.{web_search, code_execution, mcp}`.

Streaming uses the xAI SDK's native gRPC stream and proto FinishReason
(`sample_pb2.FinishReason.{REASON_STOP, REASON_MAX_LEN, REASON_TOOL_CALLS}`).

### 4.7 Groq (`groq.py`)

`GroqModel` uses the Groq Python SDK (OpenAI-compatible shape but with extra
features). Provider literal: `'groq' | 'gateway' | Provider[AsyncGroq]`.

`GroqModelSettings`:

| Setting | Type | Notes |
| --- | --- | --- |
| `groq_reasoning_format` | `'hidden' \| 'raw' \| 'parsed'` | How to surface reasoning. |

Supported builtin tools: `{WebSearchTool}`.

Special error handling: when Groq's SDK raises a `ModelHTTPError` because
generated tool arguments don't match the JSON Schema, `_parse_tool_use_failed_error`
extracts the failed generation and returns it as a `ToolCallPart` (or `TextPart`)
with `finish_reason='error'` so Upsonic's retry / re-prompt logic can recover.

### 4.8 Mistral (`mistral.py`)

`MistralModel` uses the official `mistralai` SDK. Provider literal:
`'mistral' | Provider[Mistral]`. Adds a `json_mode_schema_prompt` instance
attribute (the prompt prefix used when injecting a JSON schema into the
system prompt for prompted-output mode).

Multimodal: `MistralImageURLChunk`, `MistralDocumentURLChunk`. Tool calls use
`MistralToolCall` / `MistralFunctionCall`. Reasoning content arrives as
`MistralThinkChunk` and is mapped to `ThinkingPart`.

### 4.9 Cohere (`cohere.py`)

`CohereModel` uses the Cohere v2 client (`AsyncClientV2`). Provider literal:
`'cohere' | Provider[AsyncClientV2]`.

`CohereModelSettings` is a placeholder (no Cohere-specific keys today). Multimodal
inputs are not supported (raises `RuntimeError` if user content contains anything
other than a string). Reasoning content arrives as
`ThinkingAssistantMessageV2ContentOneItem` and is round-tripped on the way back.

Usage parsing extracts billed-units details (`input_tokens`, `output_tokens`,
`search_units`, `classifications`).

### 4.10 HuggingFace (`huggingface.py`)

`HuggingFaceModel` uses `huggingface_hub.AsyncInferenceClient` (which fronts both
HF Inference API and any Inference Provider — TGI, vLLM, Replicate, Together,
Fireworks, …). Provider literal: `'huggingface' | Provider[AsyncInferenceClient]`.

`HuggingFaceModelSettings` is a placeholder. Multimodal: text and images only
(audio / documents / video raise `NotImplementedError`). `CachePoint` is
silently ignored. Reasoning is handled via the inline `<think>` tag splitter
(see §3.3).

### 4.11 OpenRouter (`openrouter.py`)

`OpenRouterModel` *extends* `OpenAIChatModel` rather than wrapping it (unlike
the thin wrappers in §4.1) because OpenRouter has many features that need real
support: provider routing, reasoning configuration, video-URL content parts,
max-price routing, etc.

Highlights of the dedicated settings/types:

- `KnownOpenRouterProviders` — Literal of ~70 marketplace providers (`anthropic`,
  `groq`, `cerebras`, `google-vertex`, `together`, etc.).
- `OpenRouterTransforms = Literal['middle-out']`.
- `OpenRouterProviderConfig` (TypedDict) — `order`, `allow_fallbacks`,
  `require_parameters`, `data_collection`, `zdr` (Zero Data Retention),
  `only`, `ignore`, `quantizations`, `sort` (`'price' | 'throughput' | 'latency'`),
  `max_price` (with prompt/completion/image/audio/request).
- `OpenRouterReasoning` — `effort` ('high'/'medium'/'low') OR `max_tokens` (Anthropic
  style), but not both.
- `_VideoURL` / `_ChatCompletionContentPartVideoUrlParam` — adds video-URL
  content-part support that the upstream OpenAI SDK lacks.

The class overrides `_completions_create`'s `extra_body` to inject `provider`,
`reasoning`, `transforms`, and other OpenRouter-only fields.

### 4.12 Outlines (`outlines.py`)

`OutlinesModel` is the local-inference path. Wraps any `outlines.models.*`
model:

```python
@classmethod
def from_transformers(cls, hf_model, hf_tokenizer_or_processor, *, ...): ...
@classmethod
def from_llamacpp(cls, llama_model, *, ...): ...
@classmethod
def from_mlxlm(cls, mlx_model, mlx_tokenizer, *, ...): ...
@classmethod
def from_sglang(cls, base_url, api_key=None, model_name=None, *, ...): ...
@classmethod
def from_vllm_offline(cls, vllm_model, *, ...): ...
```

System prompt = `'outlines'`. `model_name = 'outlines-model'`. Function tools are
not supported (raises `UserError`). Prompts are turned into `outlines.inputs.Chat`
(with `add_system_message` / `add_user_message` / `add_assistant_message`).
Output type, when `output_object` is set, becomes
`outlines.types.dsl.JsonSchema(output_object.json_schema)`.

Each backend has its own supported-args filter:
- Transformers: `max_tokens, temperature, top_p, logit_bias, extra_body`.
- LlamaCpp: `+ seed, presence_penalty, frequency_penalty`.
- MLXLM: `extra_body` only.
- SGLang: `max_tokens, temperature, top_p, presence_penalty, frequency_penalty, extra_body`.
- vLLMOffline: builds a `vllm.SamplingParams`, supports the full set.

Streaming requires SGLang (the only outlines backend with native async
streaming); other backends use a sync iterator wrapped in an `async def
async_response()` generator.

---

## 5. Cross-file relationships

### 5.1 Provider registry — how `infer_model` finds the right SDK

`infer_model` doesn't know about Provider classes directly. It calls
`infer_provider(provider_name)` from `upsonic/providers/__init__.py`, which
returns a `Provider[T]` instance carrying:

- `Provider.client: T` — the actual SDK client instance (`AsyncOpenAI`,
  `AsyncAnthropicClient`, `Mistral`, `BedrockRuntimeClient`, etc.).
- `Provider.name: str` — used as `Model.system` and `gen_ai.system` OTel attribute.
- `Provider.base_url: str` — used as `Model.base_url`.
- `Provider.model_profile(model_name)` — returns the per-model `ModelProfile` that
  encodes feature flags (supports_tools, supports_json_schema_output,
  default_structured_output_mode, thinking_tags, json_schema_transformer, …).

The model class then unwraps the client and stores it as
`self.client = provider.client`. Settings/profile/system all flow from the
provider unless overridden in `__init__`.

### 5.2 Settings cascade

There are three levels of `ModelSettings` for any one call:

```
Model._settings  ← passed at construction time
   merged with
ModelSettings passed to request() (per-call override)
   = merge_model_settings(self.settings, model_settings)
```

`prepare_request(self.settings, model_settings)` produces the merged value, and
also materialises `customize_request_parameters(model_request_parameters)`.
Vendor-specific keys from different providers (`anthropic_thinking`,
`openai_reasoning_effort`, …) coexist harmlessly because of the prefix
convention.

### 5.3 Instrumented wrapper composition

```
infer_model('openai/gpt-4o')         →  OpenAIChatModel
        │
        ▼
instrument_model(model, settings)    →  InstrumentedModel(WrapperModel)
                                          ├── wrapped: OpenAIChatModel
                                          └── instrumentation_settings: InstrumentationSettings
```

The `Agent` constructor calls `instrument_model(self.model, instrument)` once,
so the same `Model` instance is reused across runs and OTel context naturally
propagates through `tracer.start_as_current_span(...)`.

Because `WrapperModel.__getattr__` falls through to `self.wrapped`, every
attribute and method (including `client`, `customize_request_parameters`,
`supported_builtin_tools`, etc.) works transparently on an `InstrumentedModel`
just as if it were the raw model — except `request`, `request_stream`, and
`count_tokens` which are intercepted to add spans.

### 5.4 `ModelRequestParameters` lifecycle

```
Agent build-time: builds initial ModelRequestParameters from
                  - tools (TOOL definitions for each registered tool)
                  - output_type (sets output_mode/output_object/output_tools/...)
                  - builtin_tools

  ↓ passed to `model.request(messages, settings, params)`

Model.prepare_request(settings, params):
  1. merge settings with self.settings.
  2. customize_request_parameters(params)
       → applies profile.json_schema_transformer to all tools and the
         output_object's json_schema (e.g. strips unsupported features for
         models that don't speak full JSON Schema).
  3. resolve `output_mode == 'auto'` → profile.default_structured_output_mode.
  4. delete now-irrelevant fields (output_tools when not 'tool',
     output_object when not 'native'/'prompted', etc.).
  5. validate `output_mode == 'native'` against profile.supports_json_schema_output.
  6. validate builtin_tools against profile.supported_builtin_tools.

  ↓ resulting (settings, params) used to build the SDK call.
```

---

## 6. Public API

Symbols re-exported (or accessible as `upsonic.models.X`) and what to use them
for:

| Symbol | Where | What for |
| --- | --- | --- |
| `Model` | `models/__init__.py` | Subclass to define a custom provider. |
| `infer_model` | `models/__init__.py` | The factory used by `Agent(model=...)`. |
| `KnownModelName` | `models/__init__.py` | Type alias for the giant `Literal[...]` of supported ids. |
| `ModelRequestParameters` | `models/__init__.py` | Build a per-call config with tools / output_mode / builtin_tools. |
| `StreamedResponse` | `models/__init__.py` | Subclass when implementing `request_stream` for a custom provider. |
| `ModelSettings` | `models/settings.py` | TypedDict for sampling / generic knobs. |
| `merge_model_settings` | `models/settings.py` | Combine agent defaults with per-call overrides. |
| `WrapperModel` | `models/wrapper.py` | Base for decorators (e.g. retries, caching). |
| `InstrumentedModel`, `InstrumentationSettings`, `instrument_model` | `models/instrumented.py` | OpenTelemetry tracing/metrics wrapper. |
| `ALLOW_MODEL_REQUESTS`, `check_allow_model_requests`, `override_allow_model_requests` | `models/__init__.py` | Test-suite kill-switch. |
| `cached_async_http_client`, `download_item`, `get_user_agent` | `models/__init__.py` | HTTP plumbing helpers. |
| `normalize_model_id` | `models/__init__.py` | Resolve simplified model ids. |
| `OpenAIChatCompatibleProvider`, `OpenAIResponsesCompatibleProvider` | `models/__init__.py` | Providers that speak OpenAI Chat / Responses. |
| `ModelMetadata`, `ModelCapability`, `ModelTier`, `BenchmarkScores`, `MODEL_REGISTRY`, `get_model_metadata`, `get_models_by_capability`, `get_models_by_tier`, `get_top_models` | `models/model_registry.py` | Static metadata catalog. |
| `ModelRecommendation`, `SelectionCriteria`, `RuleBasedSelector`, `LLMBasedSelector`, `select_model`, `select_model_async` | `models/model_selector.py` | Recommend a model id for a task description. |
| `OpenAIChatModel`, `OpenAIResponsesModel`, `OpenAIChatModelSettings`, `OpenAIResponsesModelSettings`, `OpenAIModelName` | `models/openai.py` | OpenAI Chat / Responses APIs. |
| `AnthropicModel`, `AnthropicModelSettings`, `AnthropicModelName` | `models/anthropic.py` | Anthropic Beta Messages API. |
| `GoogleModel`, `GoogleModelSettings`, `GoogleModelName` | `models/google.py` | Gemini (Google AI / Vertex). |
| `BedrockConverseModel`, `BedrockModelSettings`, `BedrockModelName` | `models/bedrock.py` | AWS Bedrock Converse API. |
| `XaiModel`, `XaiModelSettings`, `XaiModelName` | `models/xai.py` | xAI native gRPC SDK. |
| `GrokModel` | `models/grok.py` | xAI via OpenAI-compatible Chat Completions endpoint. |
| `GroqModel`, `GroqModelSettings`, `GroqModelName` | `models/groq.py` | Groq Cloud. |
| `MistralModel`, `MistralModelSettings`, `MistralModelName` | `models/mistral.py` | Mistral AI. |
| `CohereModel`, `CohereModelSettings`, `CohereModelName` | `models/cohere.py` | Cohere v2. |
| `HuggingFaceModel`, `HuggingFaceModelSettings`, `HuggingFaceModelName` | `models/huggingface.py` | HF Inference Providers. |
| `OutlinesModel` | `models/outlines.py` | Local Outlines (Transformers / LlamaCpp / MLXLM / SGLang / vLLMOffline). |
| `OpenRouterModel`, `OpenRouterModelSettings` | `models/openrouter.py` | OpenRouter marketplace. |
| `AzureModel`, `CerebrasModel`, `GitHubModel`, `HerokuModel`, `LiteLLMModel`, `LMStudioModel`, `MoonshotAIModel`, `NvidiaModel`, `OllamaModel`, `OVHcloudModel`, `SambaNovaModel`, `TogetherModel`, `VercelModel`, `VLLMModel` | `models/<name>.py` | Thin OpenAI-Chat-compatible wrappers (see §4.1). |

---

## 7. Integration with the rest of Upsonic

### 7.1 `Agent.model`

The `Agent` (`Direct`) class accepts a `model: Model | KnownModelName | str`
constructor argument. Internally:

```python
# upsonic/agent/agent.py (essence)
self.model = infer_model(model)
if instrument:
    self.model = instrument_model(self.model, instrument)
```

The instrumented model is reused across every `agent.do(task)` call, which keeps
HTTP connection pools warm (via `cached_async_http_client`) and accumulates
metrics in the same OTel meter scope.

### 7.2 `ModelExecutionStep`

The agent's run engine (`upsonic/agent/run.py`) builds a `ModelExecutionStep`
per LLM round:

```python
async def _execute_model(self, step: ModelExecutionStep) -> ModelResponse:
    model_request_parameters = ModelRequestParameters(
        function_tools=tool_defs,
        builtin_tools=builtin_tools,
        output_mode=output_mode,
        output_object=output_object,
        output_tools=output_tools,
        allow_text_output=allow_text_output,
        allow_image_output=allow_image_output,
    )
    if stream:
        async with self.model.request_stream(messages, settings, model_request_parameters) as resp:
            async for event in resp:
                ...
    else:
        return await self.model.request(messages, settings, model_request_parameters)
```

The streaming code path drives `StreamedResponse.__aiter__`, which interleaves
provider-specific events with synthetic `FinalResultEvent` and `PartEndEvent`
markers so the agent loop can short-circuit as soon as a final answer is
detected.

### 7.3 `Task` cost tracking

After each `Model.request(...)`, the agent reads `response.usage`
(`RequestUsage`) and `response.cost()` to populate `Task.usage` and
`Task.cost`. The `Agent.cost` property aggregates over all completed tasks (see
the recent commits `eec836c1`, `92321149`, `48dc6c40`).

`InstrumentedModel` also calls `response.cost()` and writes the result to the
span as `operation.cost` and to the `cost_histogram`. A `LookupError` from
`genai_prices` is silenced (the model/provider has no published pricing); other
exceptions become `CostCalculationFailedWarning`.

### 7.4 Reliability layer

The `reliability_layer` builds verifier and editor sub-agents that use the same
`Model` interface — they're `Agent` instances under the hood, so they go through
`infer_model` + `instrument_model` like any other.

### 7.5 Model selection helpers

The `model_selector` module is occasionally used by user code (and by some
prebuilt agents) to pick a model dynamically:

```python
from upsonic.models.model_selector import select_model, SelectionCriteria

rec = select_model(
    "Write a Python function to parse CSV files",
    criteria=SelectionCriteria(
        requires_code_generation=True,
        prioritize_cost=True,
        max_cost_tier=4,
    ),
)
agent = Agent(model=rec.model_name)
```

---

## 8. End-to-end flow of a `model.request(...)` call

Putting it all together, here is the lifecycle of a single non-streamed model
invocation, e.g. `agent.do("What is 2+2?")` against `openai/gpt-4o`:

```
1. Agent boot
   ─ Agent(__init__) calls
       self.model = infer_model('openai/gpt-4o')
       self.model = instrument_model(self.model, instrument)
   ─ infer_model:
       a. normalize_model_id('openai/gpt-4o') → 'openai/gpt-4o'
       b. provider = infer_provider('openai')  # OpenAIProvider
       c. model_kind 'openai' is in _OPENAI_CHAT_COMPATIBLE_PROVIDERS → 'openai-chat'
       d. return OpenAIChatModel('gpt-4o', provider=provider)
   ─ instrument_model wraps it: InstrumentedModel(OpenAIChatModel).

2. agent.do("What is 2+2?")
   ─ Builds messages: [ModelRequest(parts=[UserPromptPart('What is 2+2?')])]
   ─ Builds ModelRequestParameters with function_tools, output_mode='text', etc.
   ─ Calls model.request(messages, settings=None, params)

3. InstrumentedModel.request
   a. prepared_settings, prepared_params = self.wrapped.prepare_request(...)
   b. with self._instrument(...) as finish:
        ─ Open span "chat gpt-4o" with attributes:
            gen_ai.operation.name='chat'
            gen_ai.system='openai', gen_ai.provider.name='openai'
            gen_ai.request.model='gpt-4o'
            server.address='api.openai.com', server.port=443
            gen_ai.tool.definitions=[{type:'function', name:..., parameters:...}, ...]
            gen_ai.request.{max_tokens, top_p, seed, temperature, ...}
            model_request_parameters=<full JSON dump>
            logfire.json_schema=...
        ─ response = await self.wrapped.request(messages, model_settings, params)

4. OpenAIChatModel.request
   a. check_allow_model_requests()  # raises if ALLOW_MODEL_REQUESTS is False
   b. settings, params = self.prepare_request(settings, params)
        ─ merge_model_settings(self.settings, settings)
        ─ customize_request_parameters: walks function_tools / output_object,
          applies profile.json_schema_transformer (e.g. strict-mode polyfill).
        ─ output_mode 'auto' → profile.default_structured_output_mode (e.g. 'tool').
        ─ Validate native/tool/image output support against profile.
        ─ Validate builtin_tools against profile.supported_builtin_tools.
   c. response = await self._completions_create(messages, False, settings, params)

5. _completions_create
   a. tools = self._get_tools(params)
   b. tool_choice = 'required' / 'auto' / None per allow_text_output and profile.
   c. openai_messages = await self._map_messages(messages, params)
        ─ For each ModelMessage:
            • SystemPromptPart → {role: 'system'|'developer', content: ...}
              (role decided by profile.openai_system_prompt_role)
            • UserPromptPart with multimodal items → ChatCompletionContentPart{Text|Image|InputAudio}Param
            • ToolReturnPart → {role: 'tool', tool_call_id, content}
            • RetryPromptPart → either retry text or a tool result with error
            • Assistant ModelResponse → _MapModelResponseContext.map_assistant_message
              splitting TextPart/ThinkingPart/ToolCallPart, building
              ChatCompletionAssistantMessageParam.
   d. response_format from output_mode == 'native' (json_schema) or 'prompted' (json_object).
   e. _drop_sampling_params_for_reasoning, _drop_unsupported_params.
   f. extra_headers['User-Agent'] = 'upsonic/<version>'.
   g. await client.chat.completions.create(model='gpt-4o', messages=...,
                                            tools=..., tool_choice=...,
                                            response_format=..., timeout=...,
                                            reasoning_effort=..., user=..., ...)

6. APIStatusError handling
   ─ _check_azure_content_filter: if Azure 400 with code='content_filter',
     return synthetic ModelResponse(parts=[], finish_reason='content_filter').
   ─ Otherwise raise ModelHTTPError(status_code, model_name, body).

7. Response processing (_process_response)
   ─ Validate the ChatCompletion (OpenAI's SDK skips real validation).
   ─ _process_thinking: extract reasoning from message.reasoning /
     message.reasoning_content / a custom field.
   ─ split_content_into_text_and_thinking: split <think> tags in message.content.
   ─ Each ChatCompletionMessageFunctionToolCall → ToolCallPart.
   ─ provider_details: finish_reason, timestamp, optional Azure content-filter info.
   ─ usage = _map_usage(response, provider_name, base_url, model_name)
       (uses RequestUsage.extract for genai_prices).
   ─ Build ModelResponse with parts, usage, model_name, provider_response_id,
     provider_name, provider_url, finish_reason='stop'|'length'|'tool_call'|...,
     provider_details={...}.

8. Back in InstrumentedModel.request
   ─ finish(response, prepared_params)
       • span.set_attributes(response.usage.opentelemetry_attributes(),
                             gen_ai.response.model=...,
                             gen_ai.response.id=...,
                             gen_ai.response.finish_reasons=[...])
       • try: price = response.cost()
              span.set_attribute('operation.cost', float(price.total_price))
         except LookupError: pass        # pricing unknown — common
         except Exception:   warn(CostCalculationFailedWarning)
       • span.update_name('chat gpt-4o')
       • Stash record_metrics callable for after-span execution.
   ─ return response
   ─ After span ends: record_metrics() emits
       gen_ai.client.token.usage histogram (input tokens) with
         {provider, system, operation_name='chat', request_model, response_model,
          gen_ai.token.type='input'} attributes,
       gen_ai.client.token.usage histogram (output tokens), and
       operation.cost histogram (if pricing was calculable).

9. Back in agent.do
   ─ Inspect response.parts, route ToolCallParts to the tool runner, append
     ToolReturnParts to the message list, loop until allow_text_output yields a
     final TextPart (or a final ToolCallPart for output tools).
   ─ Update Task.usage and Task.cost from response.usage / response.cost().
```

For `request_stream`, replace step 5g with `client.chat.completions.create(...,
stream=True)` and route the `AsyncStream[ChatCompletionChunk]` through
`OpenAIStreamedResponse._get_event_iterator`, which calls
`_parts_manager.handle_text_delta` / `handle_thinking_delta` /
`handle_tool_call_delta` and accumulates `_usage` from the final usage chunk.
The agent loop iterates the stream via `async for event in stream` and gets
`PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FinalResultEvent` markers
from `StreamedResponse.__aiter__`.

For other providers, the only thing that changes is **step 5**: the message
mapping, tool definitions, builtin-tool translation, response format, error
handling, and response parsing all swap to the vendor's wire format. Everything
else — `prepare_request`, the `InstrumentedModel` wrapper, the OTel attributes,
the `RequestUsage` / cost extraction, the `StreamedResponse` final-result
detection — is identical.
