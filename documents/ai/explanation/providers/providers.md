---
name: providers-auth-transport
description: Use when working with Upsonic's provider auth and transport layer in src/upsonic/providers/, configuring credentials, base URLs, headers, or HTTP clients for upstream LLM APIs. Use when a user asks to add or debug a provider, wire up an SDK client (AsyncOpenAI, AsyncAnthropic, boto3 bedrock-runtime, google.genai.Client, mistralai, cohere, groq, AsyncInferenceClient, xai_sdk), resolve API keys from environment variables, route through the Pydantic AI Gateway, or select a ModelProfile. Trigger when the user mentions Provider ABC, infer_provider, infer_provider_class, OpenAIProvider, AnthropicProvider, GoogleProvider, BedrockProvider, AzureProvider, GroqProvider, MistralProvider, CohereProvider, HuggingFaceProvider, XaiProvider, DeepSeekProvider, OpenRouterProvider, VercelProvider, CerebrasProvider, FireworksProvider, TogetherProvider, HerokuProvider, MoonshotAIProvider, GitHubProvider, NvidiaProvider, OVHcloudProvider, SambaNovaProvider, OllamaProvider, LMStudioProvider, VLLMProvider, LiteLLMProvider, GrokProvider, OutlinesProvider, gateway_provider, cached_async_http_client, BedrockModelProfile, AnthropicJsonSchemaTransformer, VertexAILocation, OPENAI_API_KEY, ANTHROPIC_API_KEY, AWS_BEARER_TOKEN_BEDROCK, GATEWAY_API_KEY, or PAIG_API_KEY.
---

# `src/upsonic/providers/` — Provider Auth & Transport Layer

## 1. What this folder is

The `providers/` package is Upsonic's **provider abstraction layer**. Its job
is *not* to format chat requests, parse tool calls, or interpret streaming
responses — that lives in `src/upsonic/models/`. Instead, every file in this
folder answers a much narrower set of questions for one upstream LLM API:

1. **Authentication** — Which environment variables, API keys, bearer tokens,
   AWS credentials, OAuth tokens, or `Credentials` objects unlock this
   provider, and how do they get injected into outbound requests?
2. **Base URL & routing** — Where do requests physically go (regional
   endpoints, gateway proxies, OpenAI-compatible aliases, local servers)?
3. **Header shaping** — Which `default_headers`, attribution headers,
   `User-Agent`, traceparent injection, third-party-integration markers, etc.
   need to ride on each request?
4. **HTTP client construction** — How is the underlying `httpx.AsyncClient`
   configured (timeouts, pooling, request hooks), and how is it wired into
   the vendor SDK (`AsyncOpenAI`, `AsyncAnthropic`, `Mistral`,
   `AsyncInferenceClient`, boto3 `BaseClient`, `google.genai.Client`,
   `xai_sdk.AsyncClient`, `cohere.AsyncClientV2`, `groq.AsyncGroq`)?
5. **Profile selection** — Given a `model_name` string, which
   `ModelProfile` from `src/upsonic/profiles/` describes the model's
   capabilities (JSON schema flavor, thinking field, tool-choice support,
   prompt caching, etc.)? Profiles flow from here into `models/` so the
   request-shaping code can branch on them.

The split is deliberate: a `Provider` is a *credentialed transport*. A
`Model` (in `src/upsonic/models/`) is a *protocol implementation*. Multiple
providers can share one `Model` class — for example, `OpenAIChatModel` is
driven by `OpenAIProvider`, `AzureProvider`, `OllamaProvider`,
`DeepSeekProvider`, `GroqProvider`, `OpenRouterProvider`, `VercelProvider`,
`CerebrasProvider`, `FireworksProvider`, `TogetherProvider`,
`HerokuProvider`, `LMStudioProvider`, `VLLMProvider`, `NvidiaProvider`,
`SambaNovaProvider`, `MoonshotAIProvider`, `OVHcloudProvider`,
`GitHubProvider`, `LiteLLMProvider`, and `GrokProvider` (deprecated). The
provider supplies the configured `AsyncOpenAI` client; the model owns the
chat-completion / responses-API request and response logic.

## 2. Folder layout

```
src/upsonic/providers/
├── __init__.py            # Provider ABC + infer_provider / infer_provider_class
├── gateway.py             # Pydantic AI Gateway proxy (multi-upstream)
│
│  --- Native vendor SDKs ---
├── openai.py              # OpenAIProvider          → AsyncOpenAI
├── anthropic.py           # AnthropicProvider       → AsyncAnthropic*
├── google.py              # GoogleProvider          → google.genai.Client
├── bedrock.py             # BedrockProvider         → boto3 bedrock-runtime
├── mistral.py             # MistralProvider         → mistralai.Mistral
├── cohere.py              # CohereProvider          → cohere.AsyncClientV2
├── groq.py                # GroqProvider            → groq.AsyncGroq
├── huggingface.py         # HuggingFaceProvider     → AsyncInferenceClient
├── xai.py                 # XaiProvider             → xai_sdk.AsyncClient
│
│  --- OpenAI-compatible (uses AsyncOpenAI under the hood) ---
├── azure.py               # AzureProvider           → AsyncAzureOpenAI
├── deepseek.py            # DeepSeekProvider
├── grok.py                # GrokProvider (deprecated, OpenAI-compat shim)
├── openrouter.py          # OpenRouterProvider
├── vercel.py              # VercelProvider          (Vercel AI Gateway)
├── cerebras.py            # CerebrasProvider
├── fireworks.py           # FireworksProvider
├── together.py            # TogetherProvider
├── heroku.py              # HerokuProvider
├── moonshotai.py          # MoonshotAIProvider
├── github.py              # GitHubProvider          (GitHub Models)
├── nvidia.py              # NvidiaProvider          (NVIDIA NIM)
├── ovhcloud.py            # OVHcloudProvider
├── sambanova.py           # SambaNovaProvider
├── ollama.py              # OllamaProvider          (local)
├── lmstudio.py            # LMStudioProvider        (local)
├── vllm.py                # VLLMProvider            (self-hosted)
├── litellm.py             # LiteLLMProvider         (multi-provider proxy)
│
│  --- Special ---
└── outlines.py            # OutlinesProvider        (no client; profile-only)
```

A total of **30 .py files** — 28 concrete provider modules, the package
`__init__.py` (the `Provider` ABC), and the `gateway.py` multiplexer.

## 3. Top-level files

### `__init__.py` — the `Provider[InterfaceClient]` ABC

This is the only file in the package that defines a public type. It pins
the contract every provider must satisfy and supplies the discovery helpers
that the rest of Upsonic uses to lift a string like `"anthropic"` into a
ready-to-use, authenticated client.

```python
class Provider(ABC, Generic[InterfaceClient]):
    """The provider is in charge of providing an authenticated client to the API."""

    _client: InterfaceClient

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def base_url(self) -> str: ...

    @property
    @abstractmethod
    def client(self) -> InterfaceClient: ...

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return None
```

Three abstract members (`name`, `base_url`, `client`) and one optional hook
(`model_profile`). The generic parameter `InterfaceClient` is the *vendor
SDK type* — `AsyncOpenAI`, `AsyncAnthropic`, `boto3.BaseClient`,
`google.genai.Client`, etc. — so static analyzers can prove that
`OpenAIChatModel` only accepts a `Provider[AsyncOpenAI]`.

Two factory functions live alongside the ABC:

| Function                         | Purpose |
|----------------------------------|---------|
| `infer_provider_class(provider)` | Map a provider name to its class *without* constructing it. Used by code that needs to introspect the type (e.g. typing, registries). |
| `infer_provider(provider)`       | Construct the provider from environment variables, returning a fully wired instance. Handles two special prefixes: `gateway/<upstream>` routes to `gateway.gateway_provider`, and `google-vertex` / `google-gla` / `gemini` route to `GoogleProvider(vertexai=...)`. |

`infer_provider_class` is one big `if/elif` keyed on the provider string,
with lazy imports inside each branch so that an `import upsonic` does not
drag in `boto3`, `mistralai`, `cohere`, `huggingface_hub`, `xai_sdk`,
`google.genai`, etc. Each optional dependency is imported only when its
provider is actually requested. Concrete provider modules also wrap their
SDK imports in `try/except ImportError` and call
`upsonic.utils.printing.import_error(...)` to print an actionable
`pip install …` hint when the package is missing.

### `gateway.py` — Pydantic AI Gateway proxy

```python
from upsonic.providers import infer_provider
provider = infer_provider("gateway/openai")
```

`gateway_provider(upstream_provider, *, route, api_key, base_url, http_client)`
returns a fully configured **upstream** provider whose traffic is routed
through `https://gateway.pydantic.dev/proxy` (or a region-specific URL
inferred from a `pylf_v…` token). Its responsibilities:

- **Auth resolution** — `api_key` from arg, `GATEWAY_API_KEY`, then
  `PAIG_API_KEY` env vars. Errors via `UserError` when nothing is set.
- **Base URL inference** — `_infer_base_url(api_key)` matches Pydantic's
  token format `pylf_v<version>_<region>_…` to pick
  `https://gateway-<region>.pydantic.dev/proxy` or staging URL.
- **Path routing** — `_merge_url_path(base_url, route)` joins the gateway
  base with a per-upstream route (e.g. `openai`, `openai-responses`,
  `bedrock`, `google-vertex`). Default route comes from
  `normalize_gateway_provider(upstream_provider)`.
- **Header injection via httpx event hook** — `_request_hook(api_key)`
  installs an `event_hooks={'request': [...]}` callback on the cached
  client that:
  1. injects OpenTelemetry `traceparent` via `opentelemetry.propagate.inject`
  2. sets `Authorization: Bearer <api_key>` if absent
- **Upstream construction** — Then it builds the *real* upstream provider
  (`OpenAIProvider`, `GroqProvider`, `AnthropicProvider`,
  `BedrockProvider`, `GoogleProvider`) but with the gateway base URL and
  the gateway-hooked HTTP client substituted in. For Anthropic it bypasses
  `api_key=` and uses `auth_token=` directly on `AsyncAnthropic`; for
  Bedrock it passes a fake region (`'upsonic-gateway'`) to dodge boto3's
  `NoRegionError` since the gateway terminates the AWS sigv4 dance.

The end result: from the model's perspective, a gateway provider looks
identical to the real upstream provider. Only the URL and an extra header
hook differ.

## 4. Per-provider files

Below, each provider summarised by **interface client**, **default base
URL**, **auth env vars**, and **special headers / extras**. All entries
inherit `Provider[…]` and share a uniform constructor pattern: positional
or keyword args fall back to `os.getenv(...)`, then either accept an
existing client or build a new one over `cached_async_http_client(provider=…)`.

### Native-SDK providers

| File           | Class                | Client type                                      | Default base URL                                              | Env vars                                                                                          | Notes |
|----------------|----------------------|--------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------|-------|
| `openai.py`    | `OpenAIProvider`     | `openai.AsyncOpenAI`                             | OpenAI default (`api.openai.com`)                             | `OPENAI_API_KEY`, `OPENAI_BASE_URL`                                                                | If `base_url` is set without a key, falls back to `'api-key-not-set'` so locally hosted OpenAI-compatible servers work. |
| `anthropic.py` | `AnthropicProvider`  | `AsyncAnthropic` / `Bedrock` / `Foundry` / `Vertex` | Anthropic default (`api.anthropic.com`)                       | `ANTHROPIC_API_KEY`                                                                                | Also defines `AnthropicJsonSchemaTransformer` which calls `anthropic.transform_schema` when `strict=True`. |
| `google.py`    | `GoogleProvider`     | `google.genai.Client`                            | Google GenAI default; Vertex regional endpoint                | `GOOGLE_API_KEY` (or legacy `GEMINI_API_KEY`); `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`     | Two modes: GLA (API-key) vs Vertex (Application Default Credentials, project, location). Builds `HttpOptions` with explicit timeout + `User-Agent`. Lists 30 supported Vertex regions in `VertexAILocation`. |
| `bedrock.py`   | `BedrockProvider`    | `boto3.client('bedrock-runtime')`                | AWS regional Bedrock endpoint                                 | `AWS_REGION` / `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_BEARER_TOKEN_BEDROCK`, `AWS_READ_TIMEOUT`, `AWS_CONNECT_TIMEOUT` | If `api_key` (or `AWS_BEARER_TOKEN_BEDROCK`) is set, uses a custom `_BearerTokenSession` and `signature_version='bearer'` instead of sigv4. Defines `BedrockModelProfile` with `bedrock_*` flags and complex per-vendor profile resolution (anthropic / mistral / cohere / amazon / meta / deepseek). Strips geo prefixes (`us.`, `eu.`, `apac.`, `jp.`, `au.`, `ca.`, `global.`, `us-gov.`) from inference profile IDs. |
| `mistral.py`   | `MistralProvider`    | `mistralai.Mistral`                              | Mistral SDK default                                           | `MISTRAL_API_KEY`                                                                                  | Reads `base_url` from `sdk_configuration.get_server_details()`. |
| `cohere.py`    | `CohereProvider`     | `cohere.AsyncClientV2`                           | Cohere SDK default                                            | `CO_API_KEY`, `CO_BASE_URL`                                                                        | Also exposes a `v1_client: AsyncClient` for legacy Cohere v1 endpoints (e.g. embeddings). |
| `groq.py`      | `GroqProvider`       | `groq.AsyncGroq`                                 | `https://api.groq.com`                                        | `GROQ_API_KEY`, `GROQ_BASE_URL`                                                                    | Multi-vendor profile dispatch by name prefix (`llama`, `meta-llama/`, `gemma`, `qwen`, `deepseek`, `mistral`, `moonshotai/`, `compound-`, `openai/`). |
| `huggingface.py`| `HuggingFaceProvider`| `huggingface_hub.AsyncInferenceClient`           | `INFERENCE_PROXY_TEMPLATE` for chosen `provider_name`         | `HF_TOKEN`                                                                                         | `http_client` is intentionally rejected — must use `hf_client` instead. `base_url` and `provider_name` are mutually exclusive. |
| `xai.py`       | `XaiProvider`        | `xai_sdk.AsyncClient`                            | `https://api.x.ai/v1`                                         | `XAI_API_KEY`                                                                                      | Wraps the gRPC client in `_LazyAsyncClient` because gRPC channels bind to the event loop at creation; recreates the client when `asyncio.get_running_loop()` differs from the cached loop. |

### OpenAI-compatible providers (interface = `AsyncOpenAI`)

These all instantiate `openai.AsyncOpenAI` (or `AsyncAzureOpenAI`) pointed
at a non-OpenAI URL and overlay `OpenAIModelProfile(...)` so the
`OpenAIChatModel` in `models/` produces a request the upstream understands.

| File             | Class                  | Default base URL                                                  | Env vars                                                                  | Default headers / extras |
|------------------|------------------------|-------------------------------------------------------------------|---------------------------------------------------------------------------|--------------------------|
| `azure.py`       | `AzureProvider`        | from `AsyncAzureOpenAI` (deployment-derived)                      | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`     | Profile dispatch keyed on prefix `llama`, `meta-`, `deepseek`, `mistralai-`, `mistral`, `cohere-`, `grok`. Always wraps with `OpenAIJsonSchemaTransformer`. |
| `deepseek.py`    | `DeepSeekProvider`     | `https://api.deepseek.com`                                        | `DEEPSEEK_API_KEY`                                                        | Profile sets `openai_chat_thinking_field='reasoning_content'`, `openai_chat_send_back_thinking_parts='field'`, `openai_supports_tool_choice_required=False` for `deepseek-reasoner`. |
| `grok.py`        | `GrokProvider`         | `https://api.x.ai/v1`                                             | `GROK_API_KEY`                                                            | **Deprecated** in favor of `XaiProvider`. Disables strict tool defs (`openai_supports_strict_tool_definition=False`). |
| `openrouter.py`  | `OpenRouterProvider`   | `https://openrouter.ai/api/v1`                                    | `OPENROUTER_API_KEY`, `OPENROUTER_APP_URL`, `OPENROUTER_APP_TITLE`        | Adds `HTTP-Referer` and `X-Title` attribution headers. Multi-provider profile dispatch (`google`, `openai`, `anthropic`, `mistralai`, `qwen`, `x-ai`, `cohere`, `amazon`, `deepseek`, `meta-llama`, `moonshotai`). Custom `_OpenRouterGoogleJsonSchemaTransformer` to keep older Gemini schemas working. |
| `vercel.py`      | `VercelProvider`       | `https://ai-gateway.vercel.sh/v1`                                 | `VERCEL_AI_GATEWAY_API_KEY`, `VERCEL_OIDC_TOKEN`                          | Adds `http-referer: https://upsonic.ai/` and `x-title: upsonic`. Per-prefix profile dispatch (`anthropic`, `bedrock`, `cohere`, `deepseek`, `mistral`, `openai`, `vertex`, `xai`). |
| `cerebras.py`    | `CerebrasProvider`     | `https://api.cerebras.ai/v1`                                      | `CEREBRAS_API_KEY`                                                        | Adds `X-Cerebras-3rd-Party-Integration: upsonic`. Marks `frequency_penalty`, `logit_bias`, `presence_penalty`, `parallel_tool_calls`, `service_tier` as `openai_unsupported_model_settings`. |
| `fireworks.py`   | `FireworksProvider`    | `https://api.fireworks.ai/inference/v1`                           | `FIREWORKS_API_KEY`                                                       | Strips `accounts/fireworks/models/` prefix before profile lookup. |
| `together.py`    | `TogetherProvider`     | `https://api.together.xyz/v1`                                     | `TOGETHER_API_KEY`                                                        | Profile dispatch on prefix (`deepseek-ai`, `google`, `qwen`, `meta-llama`, `mistralai`, `mistral`). |
| `heroku.py`      | `HerokuProvider`       | `https://us.inference.heroku.com/v1`                              | `HEROKU_INFERENCE_KEY`, `HEROKU_INFERENCE_URL`                            | Trailing `/v1` is appended to the Heroku endpoint manually. |
| `moonshotai.py`  | `MoonshotAIProvider`   | `https://api.moonshot.ai/v1`                                      | `MOONSHOTAI_API_KEY`                                                      | `openai_supports_tool_choice_required=False` (Kimi limitation), `openai_chat_thinking_field='reasoning_content'`. |
| `github.py`      | `GitHubProvider`       | `https://models.github.ai/inference`                              | `GITHUB_API_KEY`                                                          | Profile dispatch by GitHub Models vendor prefix (`xai`, `meta`, `microsoft`, `mistral-ai`, `cohere`, `deepseek`); falls through to `openai_model_profile` for unprefixed names. |
| `nvidia.py`      | `NvidiaProvider`       | `https://integrate.api.nvidia.com/v1`                             | `NVIDIA_API_KEY`, `NGC_API_KEY`, `NVIDIA_BASE_URL`                        | Big prefix table covering Meta, Google, Microsoft Phi, Mistral families, DeepSeek, Qwen, AI21 Jamba, NVIDIA Nemotron, IBM Granite, Snowflake Arctic, Upstage, Databricks DBRX, Cohere Command. |
| `ovhcloud.py`    | `OVHcloudProvider`     | `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1`                | `OVHCLOUD_API_KEY`                                                        | Profile dispatch on prefix (`llama`, `meta-`, `deepseek`, `mistral`, `gpt` → harmony, `qwen`). |
| `sambanova.py`   | `SambaNovaProvider`    | `https://api.sambanova.ai/v1`                                     | `SAMBANOVA_API_KEY`, `SAMBANOVA_BASE_URL`                                 | — |
| `ollama.py`      | `OllamaProvider`       | from `OLLAMA_BASE_URL` (required)                                 | `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`                                       | Falls back to `'api-key-not-set'` placeholder. Sets `openai_chat_thinking_field='reasoning'`. |
| `lmstudio.py`    | `LMStudioProvider`     | from `LMSTUDIO_BASE_URL` (required)                               | `LMSTUDIO_BASE_URL`, `LMSTUDIO_API_KEY`                                   | Same placeholder pattern as Ollama. |
| `vllm.py`        | `VLLMProvider`         | from `VLLM_BASE_URL` (required, e.g. `http://localhost:8000/v1`)  | `VLLM_BASE_URL`, `VLLM_API_KEY`                                           | Self-hosted; placeholder API key allowed. |
| `litellm.py`     | `LiteLLMProvider`      | from `LITELLM_BASE_URL` (required, e.g. `http://localhost:4000/v1`)| `LITELLM_BASE_URL`, `LITELLM_API_KEY`                                     | Profile dispatch covers anthropic, openai, google, mistral, cohere, amazon/bedrock, meta-llama, groq, deepseek, moonshotai, x-ai, qwen — falls back to OpenAI profile. |

### Special providers

| File           | Class             | Notes |
|----------------|-------------------|-------|
| `outlines.py`  | `OutlinesProvider`| Has **no** real `client` or `base_url` — both `@property` methods raise `NotImplementedError`. Used purely for its `model_profile()` which returns a `ModelProfile(supports_tools=False, supports_json_schema_output=True, supports_json_object_output=True, default_structured_output_mode='native', native_output_requires_schema_in_instructions=True)`. The associated `OutlinesModel` in `models/` owns the actual transport. |

## 5. Cross-file relationships

The folder is mostly flat — providers do not import each other except in
two places:

1. **`gateway.py`** imports the concrete providers it wraps (`OpenAIProvider`,
   `GroqProvider`, `AnthropicProvider`, `BedrockProvider`, `GoogleProvider`)
   to delegate construction once it has rewritten the URL and installed the
   gateway request hook.
2. **`__init__.py`** lazily imports every provider class inside
   `infer_provider_class` to avoid forcing optional deps on import.

The recurring cross-cutting helpers (used by almost every provider) live
**outside** this folder:

| Helper                                        | Location                                | Purpose |
|-----------------------------------------------|-----------------------------------------|---------|
| `cached_async_http_client(provider=...)`      | `src/upsonic/models/__init__.py:2260`   | One shared `httpx.AsyncClient` per provider name, with `User-Agent: upsonic/<version>`. Connection pools survive across agent calls. |
| `get_user_agent()`                            | `src/upsonic/models/__init__.py:2392`   | Returns `f"upsonic/{__version__}"`. Used by `GoogleProvider` directly. |
| `DEFAULT_HTTP_TIMEOUT`                        | `src/upsonic/models/__init__.py:67`     | `600` seconds; read by `GoogleProvider` to mirror the timeout into `HttpOptions`. |
| `import_error(...)`                           | `src/upsonic/utils/printing`            | Prints actionable install instructions when an optional SDK is missing. |
| `UserError`                                   | `src/upsonic/utils/package/exception`   | Raised by every provider when auth env vars are missing. |
| `ModelProfile` and per-vendor profile fns     | `src/upsonic/profiles/`                 | Source of truth for capability flags. Providers compose these into final profiles. |
| `JsonSchemaTransformer`, `JsonSchema`         | `src/upsonic/_json_schema`              | Used by `AnthropicJsonSchemaTransformer` and `_OpenRouterGoogleJsonSchemaTransformer`. |

The dependency graph is therefore strictly downward:

```
upsonic/agent/         → uses Model
upsonic/models/<name>  → uses Provider[<sdk client>]
upsonic/providers/     → uses upsonic/profiles + upsonic/models helpers (cached_async_http_client, get_user_agent)
                       → uses upsonic/_json_schema for transformers
                       → uses vendor SDK (openai, anthropic, boto3, …)
```

A subtle cycle-avoidance trick: `providers/*.py` imports
`cached_async_http_client` from `upsonic.models` (the package
`__init__.py`), but **does not** import any concrete model class. Concrete
model classes import providers eagerly. This keeps the dependency direction
one-way.

## 6. Public API

Re-exported / used externally:

```python
# Type / abstract base
from upsonic.providers import Provider

# Discovery
from upsonic.providers import infer_provider, infer_provider_class

# Concrete providers (each via its own module)
from upsonic.providers.openai     import OpenAIProvider
from upsonic.providers.anthropic  import AnthropicProvider, AnthropicJsonSchemaTransformer
from upsonic.providers.google     import GoogleProvider, VertexAILocation
from upsonic.providers.bedrock    import BedrockProvider, BedrockModelProfile, remove_bedrock_geo_prefix
from upsonic.providers.azure      import AzureProvider
from upsonic.providers.groq       import GroqProvider
from upsonic.providers.cohere     import CohereProvider
from upsonic.providers.mistral    import MistralProvider
from upsonic.providers.huggingface import HuggingFaceProvider
from upsonic.providers.xai        import XaiProvider
from upsonic.providers.grok       import GrokProvider          # deprecated
from upsonic.providers.ollama     import OllamaProvider
from upsonic.providers.lmstudio   import LMStudioProvider
from upsonic.providers.vllm       import VLLMProvider
from upsonic.providers.deepseek   import DeepSeekProvider, DeepSeekModelName
from upsonic.providers.openrouter import OpenRouterProvider
from upsonic.providers.vercel     import VercelProvider
from upsonic.providers.cerebras   import CerebrasProvider
from upsonic.providers.fireworks  import FireworksProvider
from upsonic.providers.together   import TogetherProvider
from upsonic.providers.heroku     import HerokuProvider
from upsonic.providers.moonshotai import MoonshotAIProvider, MoonshotAIModelName
from upsonic.providers.github     import GitHubProvider
from upsonic.providers.nvidia     import NvidiaProvider
from upsonic.providers.ovhcloud   import OVHcloudProvider
from upsonic.providers.sambanova  import SambaNovaProvider
from upsonic.providers.litellm    import LiteLLMProvider
from upsonic.providers.outlines   import OutlinesProvider

# Gateway helpers
from upsonic.providers.gateway    import gateway_provider, normalize_gateway_provider, GATEWAY_BASE_URL
```

A typical user almost never instantiates a provider directly — they pass
the provider name as a string to the model class:

```python
from upsonic.models.openai import OpenAIChatModel
model = OpenAIChatModel("gpt-4o-mini", provider="openai")
```

`OpenAIChatModel.__init__` calls `infer_provider("openai")` internally,
which walks the `if/elif` chain in `infer_provider_class`, lazily imports
`OpenAIProvider`, calls its zero-arg constructor, and that constructor
reads `OPENAI_API_KEY` from the env, builds an `AsyncOpenAI` over a cached
`httpx.AsyncClient`, and returns. The model then captures
`provider.client` for transport and `provider.model_profile(model_name)`
for capability flags.

## 7. Integration with the rest of Upsonic

### Used by `src/upsonic/models/`

Every `*Model` class in `models/` accepts `provider:` as either a string,
the literal `'gateway'`, or a fully constructed `Provider[...]` of the
correct generic type. Inside `__init__` you'll see:

```python
# src/upsonic/models/openai.py:546
if isinstance(provider, str):
    provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
self._provider = provider
self.client = provider.client                  # the SDK client used to call the API
super().__init__(profile=profile or provider.model_profile)   # capability flags
```

So the contract between `providers/` and `models/` is exactly:

1. `provider.client` — SDK object the model calls.
2. `provider.model_profile(model_name)` — capability flags consumed when
   shaping the request (which JSON-schema flavor to emit, whether to send
   thinking parts back, what `tool_choice` values are legal, etc.).
3. `provider.base_url` — informational; logged and used for telemetry.
4. `provider.name` — used for logging and to key the cached httpx client.

### Used by `infer_model(...)`

The top-level model factory at `src/upsonic/models/__init__.py` resolves
`"openai:gpt-4o"`, `"anthropic:claude-3-7-sonnet"`,
`"bedrock:anthropic.claude-3-5-sonnet-…"`, etc. into the appropriate
`Model` subclass and forwards the provider string into the model
constructor. From there control passes to `infer_provider`.

### Used by agent / team layers

`src/upsonic/agent/agent.py` and `src/upsonic/team/team.py` only ever see
`Model` instances; they never instantiate or import providers directly. The
provider layer is therefore a **transitive** dependency: it is exercised on
every API call but is hidden from the high-level surface.

### Used by reliability and safety layers

`src/upsonic/reliability_layer/` may run "verifier" or "editor" sub-agents
that themselves are built on top of `Model`s — and through them, providers.
There is no direct coupling.

## 8. End-to-end flow: provider auth → request

The diagram below traces a single `agent.run("hello")` call from string
parsing all the way to an HTTP request leaving the host:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ User code                                                                │
│ Agent(model="openai/gpt-4o-mini").do_async(Task(...))                    │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ src/upsonic/models/__init__.py :: infer_model("openai/gpt-4o-mini")      │
│   -> picks OpenAIChatModel  (logic = chat-completions request shape)     │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ OpenAIChatModel.__init__(model_name=..., provider="openai")              │
│   if isinstance(provider, str):                                          │
│       provider = infer_provider("openai")                                │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ src/upsonic/providers/__init__.py :: infer_provider("openai")            │
│   -> infer_provider_class("openai") -> OpenAIProvider                    │
│   -> OpenAIProvider() constructed                                        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ src/upsonic/providers/openai.py :: OpenAIProvider.__init__               │
│   1. api_key = os.getenv("OPENAI_API_KEY")                               │
│   2. http_client = cached_async_http_client(provider="openai")           │
│        - same httpx.AsyncClient reused across agents/calls               │
│        - User-Agent: upsonic/<version>                                   │
│        - timeout=600s, connect=5s, default httpx pool                    │
│   3. self._client = AsyncOpenAI(base_url=None, api_key=..., http_client) │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Back in OpenAIChatModel:                                                 │
│   self.client  = provider.client       # AsyncOpenAI                     │
│   self.profile = provider.model_profile("gpt-4o-mini")                   │
│                  -> openai_model_profile("gpt-4o-mini")                  │
│                  -> e.g. supports_json_schema_output=True, etc.          │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Per-request (agent.run -> model.request):                                │
│   - models/openai.py builds chat-completion params using self.profile    │
│   - calls self.client.chat.completions.create(...) (AsyncOpenAI)         │
│   - AsyncOpenAI:                                                         │
│       * sets Authorization: Bearer $OPENAI_API_KEY                       │
│       * uses the shared httpx.AsyncClient                                │
│       * issues POST https://api.openai.com/v1/chat/completions           │
└──────────────────────────────────────────────────────────────────────────┘
```

The same flow with `provider="gateway"`:

```
infer_provider("openai")    →    OpenAIProvider              (direct)
infer_provider("gateway/openai")
   ↳ gateway.gateway_provider("openai", ...)
       1. api_key = GATEWAY_API_KEY
       2. base_url = _infer_base_url(api_key)            # region-aware
       3. http_client = cached_async_http_client(provider="gateway/openai")
          http_client.event_hooks["request"] = [_request_hook(api_key)]
              # injects traceparent + Bearer token
       4. base_url = _merge_url_path(base_url, "openai")
       5. return OpenAIProvider(api_key=api_key,
                                base_url=base_url,
                                http_client=http_client)
```

And with `provider="anthropic"`:

```
infer_provider("anthropic")
   ↳ AnthropicProvider()
       1. api_key = ANTHROPIC_API_KEY                  # mandatory
       2. http_client = cached_async_http_client(provider="anthropic")
       3. self._client = AsyncAnthropic(api_key=..., base_url=..., http_client=...)
   ↳ provider.model_profile(name)
       = ModelProfile(json_schema_transformer=AnthropicJsonSchemaTransformer)
         .update(anthropic_model_profile(name))
   ↳ AnthropicModel uses self._client.messages.create(...)
```

And with `provider="bedrock"`:

```
infer_provider("bedrock")
   ↳ BedrockProvider()
       1. region = AWS_REGION or AWS_DEFAULT_REGION       # or UserError
       2. read_timeout / connect_timeout from env or default 300/60
       3. if api_key (or AWS_BEARER_TOKEN_BEDROCK):
              session = boto3.Session(botocore_session=_BearerTokenSession(api_key),
                                      region_name=region, profile_name=...)
              config["signature_version"] = "bearer"
          else:
              session = boto3.Session(aws_access_key_id=...,
                                      aws_secret_access_key=...,
                                      aws_session_token=...,
                                      region_name=region,
                                      profile_name=...)
       4. self._client = session.client(
              "bedrock-runtime",
              config=Config(read_timeout=…, connect_timeout=…, signature_version=…),
              endpoint_url=base_url,
          )
   ↳ provider.model_profile("us.anthropic.claude-…-v2:0")
       = strip geo prefix → "anthropic.claude-…-v2:0"
       → split on "." → ("anthropic", "claude-…-v2:0")
       → strip version "v2:0" → "claude-…"
       → BedrockModelProfile(...).update(_without_builtin_tools(anthropic_model_profile("claude-…")))
```

### Reading the auth contract at a glance

| Provider              | Required env var(s)                                            | Optional env var(s)                                                | Failure mode if missing                                  |
|-----------------------|----------------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------|
| OpenAI                | `OPENAI_API_KEY` (unless `base_url` provided)                  | `OPENAI_BASE_URL`                                                  | Falls back to placeholder when `base_url` is set         |
| Anthropic             | `ANTHROPIC_API_KEY`                                            | —                                                                  | `UserError`                                              |
| Google (GLA)          | `GOOGLE_API_KEY` (or legacy `GEMINI_API_KEY`)                  | —                                                                  | `UserError`                                              |
| Google (Vertex)       | ADC + `GOOGLE_CLOUD_PROJECT`                                   | `GOOGLE_CLOUD_LOCATION` (defaults `us-central1`)                   | boto/google libs raise; defaults filled                  |
| Bedrock (sigv4)       | `AWS_REGION`/`AWS_DEFAULT_REGION` + AWS creds                  | `AWS_SESSION_TOKEN`, profile, timeouts                             | `UserError` for region; `NoRegionError` re-raised        |
| Bedrock (bearer)      | `AWS_REGION` + `AWS_BEARER_TOKEN_BEDROCK`                      | —                                                                  | `UserError`                                              |
| Azure                 | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION` | —                                                            | `UserError` for each missing                             |
| Groq                  | `GROQ_API_KEY`                                                 | `GROQ_BASE_URL`                                                    | `UserError`                                              |
| Cohere                | `CO_API_KEY`                                                   | `CO_BASE_URL`                                                      | `UserError`                                              |
| Mistral               | `MISTRAL_API_KEY`                                              | —                                                                  | `UserError`                                              |
| HuggingFace           | `HF_TOKEN`                                                     | —                                                                  | `UserError`                                              |
| xAI                   | `XAI_API_KEY`                                                  | —                                                                  | `UserError`                                              |
| Grok (deprecated)     | `GROK_API_KEY`                                                 | —                                                                  | `UserError`                                              |
| DeepSeek              | `DEEPSEEK_API_KEY`                                             | —                                                                  | `UserError`                                              |
| OpenRouter            | `OPENROUTER_API_KEY`                                           | `OPENROUTER_APP_URL`, `OPENROUTER_APP_TITLE`                       | `UserError`                                              |
| Vercel                | `VERCEL_AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN`             | —                                                                  | `UserError`                                              |
| Cerebras              | `CEREBRAS_API_KEY`                                             | —                                                                  | `UserError`                                              |
| Fireworks             | `FIREWORKS_API_KEY`                                            | —                                                                  | `UserError`                                              |
| Together              | `TOGETHER_API_KEY`                                             | —                                                                  | `UserError`                                              |
| Heroku                | `HEROKU_INFERENCE_KEY`                                         | `HEROKU_INFERENCE_URL`                                             | `UserError`                                              |
| MoonshotAI            | `MOONSHOTAI_API_KEY`                                           | —                                                                  | `UserError`                                              |
| GitHub Models         | `GITHUB_API_KEY`                                               | —                                                                  | `UserError`                                              |
| NVIDIA NIM            | `NVIDIA_API_KEY` or `NGC_API_KEY`                              | `NVIDIA_BASE_URL`                                                  | `UserError`                                              |
| OVHcloud              | `OVHCLOUD_API_KEY`                                             | —                                                                  | `UserError`                                              |
| SambaNova             | `SAMBANOVA_API_KEY`                                            | `SAMBANOVA_BASE_URL`                                               | `UserError`                                              |
| Ollama                | `OLLAMA_BASE_URL` (mandatory)                                  | `OLLAMA_API_KEY`                                                   | `UserError` for base URL                                 |
| LM Studio             | `LMSTUDIO_BASE_URL` (mandatory)                                | `LMSTUDIO_API_KEY`                                                 | `UserError` for base URL                                 |
| vLLM                  | `VLLM_BASE_URL` (mandatory)                                    | `VLLM_API_KEY`                                                     | `UserError` for base URL                                 |
| LiteLLM               | `LITELLM_BASE_URL` (mandatory)                                 | `LITELLM_API_KEY`                                                  | `UserError` for base URL                                 |
| Gateway               | `GATEWAY_API_KEY` (or `PAIG_API_KEY`)                          | `GATEWAY_BASE_URL` (or `PAIG_BASE_URL`)                            | `UserError`                                              |
| Outlines              | n/a (no transport)                                             | n/a                                                                | `NotImplementedError` if `client` accessed               |

### Custom-client escape hatch

Every provider exposes an "I already built my own SDK client, just use it"
constructor — the kwargs vary in name (`openai_client=`,
`anthropic_client=`, `mistral_client=`, `cohere_client=`, `groq_client=`,
`hf_client=`, `xai_client=`, `bedrock_client=`, `client=` for Google,
`openai_client=` for the OpenAI-compatible providers). When you pass it,
all auth/URL/`http_client` arguments must be `None`, and the provider
becomes a thin shim around your client. This is how you plug Upsonic into
existing observability proxies, mTLS clients, custom retry layers, etc.

### Worked example: OpenRouter with attribution

```python
from upsonic.providers.openrouter import OpenRouterProvider
from upsonic.models.openai import OpenAIChatModel

provider = OpenRouterProvider(
    api_key="...",
    app_url="https://my-app.example/",
    app_title="My App",
)

# OpenRouterProvider builds an AsyncOpenAI pointed at https://openrouter.ai/api/v1
# and injects default headers:
#     HTTP-Referer: https://my-app.example/
#     X-Title:      My App
# The OpenAIChatModel below sees a normal AsyncOpenAI; OpenRouter adds the headers.

model = OpenAIChatModel("anthropic/claude-3.5-sonnet", provider=provider)
```

`provider.model_profile("anthropic/claude-3.5-sonnet")` first splits on
`/`, dispatches to `anthropic_model_profile("claude-3.5-sonnet")`, then
wraps with `OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer,
openai_chat_send_back_thinking_parts='field',
openai_chat_thinking_field='reasoning',
openai_chat_supports_file_urls=True)` so that `OpenAIChatModel` knows to
relay reasoning content and accept file URLs.

### Worked example: gateway-routed Bedrock

```python
from upsonic.providers import infer_provider
provider = infer_provider("gateway/bedrock")
# Internally:
#   gateway_provider("bedrock", ...)
#     api_key   = GATEWAY_API_KEY
#     base_url  = https://gateway-<region>.pydantic.dev/proxy/bedrock
#     hook      = adds traceparent + Authorization on every request
#     return BedrockProvider(api_key=api_key,
#                            base_url=base_url,
#                            region_name="upsonic-gateway")  # bypass NoRegionError
```

The signed AWS request is now produced by boto3 with `signature_version="bearer"`
and shipped through the Pydantic gateway, which performs the real sigv4
handshake and forwards to AWS Bedrock.

---

## TL;DR

- `providers/` = *who you are and where to talk to*; `models/` = *what to say*.
- One `Provider` ABC + one factory pair (`infer_provider_class`,
  `infer_provider`) gates lazy import of every optional SDK.
- The shared `cached_async_http_client(provider=...)` from
  `upsonic.models.__init__` is the single source of HTTP plumbing —
  every provider passes it into the SDK client constructor.
- Model profiles flow up from `upsonic.profiles.*` through each provider's
  `model_profile()` method. This is how the same `OpenAIChatModel` can
  speak to twenty different OpenAI-compatible backends with the right
  capability flags for each.
- `gateway.py` is the only file that re-wraps other providers; it does so
  by changing `base_url` and installing an httpx request hook.
