---
name: embeddings-subsystem
description: Use when working on Upsonic's vector embedding layer, swapping or adding embedding providers, debugging embedding ingestion/retrieval, or wiring KnowledgeBase/RAG to a vector store. Use when a user asks to embed chunks or queries, configure OpenAI/Azure/Bedrock/HuggingFace/FastEmbed/Ollama/Gemini embeddings, tune batching/retry/normalization, or pick a model by dimension/cost/locality. Trigger when the user mentions EmbeddingProvider, EmbeddingConfig, EmbeddingMode, EmbeddingMetrics, embed_documents, embed_query, _embed_batch, create_embedding_provider, auto_detect_best_embedding, create_best_available_embedding, OpenAIEmbedding, AzureOpenAIEmbedding, BedrockEmbedding, HuggingFaceEmbedding, FastEmbedProvider, OllamaEmbedding, GeminiEmbedding, text-embedding-3-small, text-embedding-3-large, ada-002, Titan, Cohere, BGE, MiniLM, MPNet, nomic-embed-text, gemini-embedding-001, Matryoshka/MRL, SPLADE sparse embeddings, ONNX runtime, managed identity, Vertex AI, adaptive batching, exponential backoff, or pricing_info.
---

# `src/upsonic/embeddings/` — Embeddings Subsystem

## 1. What this folder is — its role in Upsonic

The `embeddings/` package is Upsonic's **vector-embedding abstraction layer**. It provides a unified, async-first interface for converting text (and Upsonic `Chunk` objects) into dense (or sparse) numeric vectors using a wide selection of providers — both cloud-hosted APIs (OpenAI, Azure OpenAI, AWS Bedrock, Google Gemini, HuggingFace Inference) and local runtimes (HuggingFace transformers, FastEmbed/ONNX, Ollama).

In Upsonic's architecture, embeddings are the bridge between three layers:

| Consumer | What it asks for |
|----------|------------------|
| `KnowledgeBase` (`src/upsonic/knowledge_base/`) | Embed `Chunk` objects from documents during ingestion (`embed_documents`) |
| `RAG` retrieval (`src/upsonic/rag/`) | Embed user queries before vector search (`embed_query`) |
| Vector stores (Pinecone, Chroma, Qdrant, FAISS adapters in `src/upsonic/vectordb/`) | Receive `List[List[float]]` vectors keyed by chunk ID |

The package solves four orthogonal concerns:

1. **Provider polymorphism** — every backend implements the same `EmbeddingProvider` ABC.
2. **Robust execution** — batching, retries with exponential backoff, adaptive sub-batch splitting, normalization, optional caching, and metrics.
3. **Discovery & selection** — a registry-backed factory with environment-aware auto-detection (`auto_detect_best_embedding`) and use-case-aware selection (`create_best_available_embedding`).
4. **Lazy dependency loading** — heavy SDKs (`torch`, `transformers`, `boto3`, `google-genai`, `fastembed`, `aiohttp`, …) are imported only when their concrete provider is actually instantiated, so a user installing only `openai` does not pay the import cost of `torch`.

## 2. Folder layout

```
src/upsonic/embeddings/
├── __init__.py                  # Public surface; lazy __getattr__ for all symbols
├── base.py                      # EmbeddingProvider ABC, EmbeddingConfig, EmbeddingMode, EmbeddingMetrics
├── factory.py                   # Provider registry, factories, auto-detection, recommendations
├── openai_provider.py           # OpenAIEmbedding + OpenAIEmbeddingConfig + helpers
├── azure_openai_provider.py     # AzureOpenAIEmbedding + AzureOpenAIEmbeddingConfig + Managed Identity
├── bedrock_provider.py          # BedrockEmbedding + BedrockEmbeddingConfig (Titan, Cohere)
├── huggingface_provider.py      # HuggingFaceEmbedding (local transformers + Inference API)
├── fastembed_provider.py        # FastEmbedProvider (Qdrant ONNX-runtime, dense + sparse)
├── ollama_provider.py           # OllamaEmbedding (HTTP client for local Ollama daemon)
└── gemini_provider.py           # GeminiEmbedding (Google AI / Vertex AI)
```

There are no subfolders — the package is intentionally flat, with one `*_provider.py` file per backend.

## 3. Top-level files — file-by-file walkthrough

### 3.1 `base.py` — the contract

This file defines the abstract base class that every provider extends, plus the shared configuration model, mode enum, and metrics model.

#### `EmbeddingMode` (enum, str)

Maps abstract intent to provider-specific request shaping.

| Value | Meaning |
|-------|---------|
| `DOCUMENT` | Embedding a chunk for index storage (default for `embed_documents`) |
| `QUERY` | Embedding a user query for retrieval (default for `embed_query`) |
| `SYMMETRIC` | Bi-encoder symmetric similarity (e.g. clustering pairs) |
| `CLUSTERING` | Embedding optimized for clustering tasks |

Each provider declares which subset it supports via the abstract `supported_modes` property. For example, OpenAI/Azure only return `[DOCUMENT, QUERY]`, while Gemini returns all four.

#### `EmbeddingMetrics` (Pydantic model)

Per-provider running counters: `total_chunks`, `total_tokens`, `embedding_time_ms`, `avg_time_per_chunk`, `dimension`, `model_name`, `provider`. Populated inside `embed_texts` and exposed via `get_metrics()`.

#### `EmbeddingConfig` (Pydantic base model)

Common knobs that every concrete config inherits from. The most important fields:

| Field | Default | Purpose |
|-------|---------|---------|
| `model_name` | required | Provider-specific model identifier |
| `dimension` | `None` | Optional expected output dim (for validation) |
| `batch_size` | `100` | Texts per `_embed_batch` call |
| `max_retries` | `3` | Used by `_embed_with_retry` |
| `retry_delay` | `1.0` | Initial backoff seconds |
| `timeout` | `30.0` | Per-request timeout |
| `normalize_embeddings` | `True` | Apply unit-norm normalization post-call |
| `show_progress` | `True` | Render Rich progress table |
| `cache_embeddings` | `False` | In-process dict cache keyed by `(model, text, metadata)` |
| `truncate_input` | `True` | Truncate inputs that exceed `max_input_length` |
| `enable_retry_with_backoff` | `True` | Use exponential rather than constant delay |
| `enable_adaptive_batching` | `True` | On failure, split batch in half and recurse |
| `enable_compression` | `False` | Hook for dim-reduction (`_compress_embeddings` is a no-op stub) |
| `compression_ratio` | `0.5` | Target ratio for compression |

#### `EmbeddingProvider` (Pydantic + ABC)

The base provider. Inherits from `BaseModel` so its `config` field is validated, and from `ABC` so `_embed_batch`, `get_model_info`, `supported_modes`, and `pricing_info` must be overridden.

| Member | Kind | Purpose |
|--------|------|---------|
| `config` | field | The provider's typed `EmbeddingConfig` subclass |
| `_cache` | dict | In-memory `{cache_key: vector}` |
| `_metrics` | `EmbeddingMetrics` | Initialized in `__init__` from `self.__class__.__name__` |
| `_executor` | `ThreadPoolExecutor` (optional) | Used by some subclasses for sync→async offloading |
| `_embed_batch` | abstract async | The single method each provider must implement |
| `get_model_info` | abstract | Return dimensions / max_tokens / description dict |
| `supported_modes` | abstract property | List of supported `EmbeddingMode` values |
| `pricing_info` | abstract property | `{"per_million_tokens": float, ...}` |
| `embed_documents(chunks)` | concrete | Pulls `chunk.text_content` and calls `embed_texts(..., DOCUMENT)` |
| `embed_query(query)` | concrete | Calls `embed_texts([query], QUERY)` and returns `[0]` |
| `embed_texts(texts, mode, show_progress, metadata)` | concrete | The orchestrator — see §8 for flow |
| `_embed_with_retry(texts, mode)` | concrete | Retry + backoff + adaptive split |
| `_check_cache(texts, metadata)` | concrete | Returns `(partial_results, uncached_indices)` |
| `_update_cache(...)` | concrete | Persists newly computed vectors |
| `_get_cache_key(text, metadata)` | concrete | `model_name|text|sorted(metadata)` |
| `_normalize_embeddings(embeddings)` | concrete | Unit-norm using NumPy if available, else pure Python |
| `_compress_embeddings(embeddings)` | concrete (stub) | Currently identity; reserved for future dim-reduction |
| `_create_progress_display(total)` | concrete | Builds a Rich `Table` panel via `upsonic.utils.printing` |
| `_update_progress(current, total)` | concrete | Currently a `pass` — reserved hook |
| `validate_connection()` | concrete async | `embed_texts(["test"])`; returns `bool` |
| `get_metrics()` | concrete | Returns a copy of `_metrics` |
| `get_model_name()` / `aget_model_name()` | concrete | Read `config.model_name` (or `'Unknown'`) |
| `clear_cache()` / `get_cache_info()` | concrete | Manage in-memory cache |
| `warmup(sample_texts)` | concrete async | Call `embed_texts(sample_texts)` to load model |
| `estimate_cost(num_texts, avg_text_length=100)` | concrete | Uses `pricing_info['per_million_tokens']` and `chars/4 ≈ tokens` |
| `close()` | concrete async | Shuts down the optional `ThreadPoolExecutor`; subclasses override to close clients |

The retry logic is worth highlighting:

```python
async def _embed_with_retry(self, texts, mode):
    for attempt in range(self.config.max_retries):
        try:
            return await self._embed_batch(texts, mode)
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise                              # never retry config errors
            last_error = e
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt) \
                        if self.config.enable_retry_with_backoff \
                        else self.config.retry_delay
                await asyncio.sleep(delay)
                if self.config.enable_adaptive_batching and len(texts) > 1:
                    mid = len(texts) // 2                       # split & recurse
                    first = await self._embed_with_retry(texts[:mid], mode)
                    second = await self._embed_with_retry(texts[mid:], mode)
                    return first + second
    raise ModelConnectionError("Failed to embed texts after N attempts: ...",
                               error_code="EMBEDDING_FAILED",
                               original_error=last_error)
```

This is the central reliability primitive for the whole subsystem: configuration mistakes fail fast, transient errors back off, and overlong batches are bisected.

### 3.2 `__init__.py` — the lazy public surface

The package exposes a long `__all__` list, but **none of it is imported eagerly**. Instead, the module defines a `__getattr__(name)` that resolves attributes on demand:

1. Base classes (`EmbeddingProvider`, `EmbeddingConfig`, `EmbeddingMode`, `EmbeddingMetrics`) → loaded via `_get_base_classes()` from `.base`.
2. Factory helpers (`create_embedding_provider`, `auto_detect_best_embedding`, …) → loaded via `_get_factory_functions()` from `.factory`.
3. Provider classes and helper factories → each has its own `_get_*` thunk that imports only the relevant module.

The result: `from upsonic.embeddings import OpenAIEmbedding` triggers `openai_provider.py` only; `from upsonic.embeddings import FastEmbedProvider` triggers `fastembed_provider.py` and its `fastembed`/`onnxruntime` deps only. A `TYPE_CHECKING` block at the top redeclares the same names for static analyzers.

If a name is unknown, `__getattr__` raises `AttributeError("module 'upsonic.embeddings' has no attribute '{name}'. Please import from the appropriate sub-module.")`.

### 3.3 `factory.py` — registry, factories, recommendations

Provides three registries and a family of helper functions.

#### Registries (module-level dicts)

```python
_PROVIDER_IMPORTERS: Dict[str, Callable[[], Tuple[Type[EmbeddingProvider], Type[EmbeddingConfig]]]]
_PROVIDER_REGISTRY:  Dict[str, Type[EmbeddingProvider]]
_PROVIDER_CONFIGS:   Dict[str, Type[EmbeddingConfig]]
```

Only `_PROVIDER_IMPORTERS` is populated up front; the other two are filled lazily by `_lazy_import_provider`.

#### Helpers

| Function | Role |
|----------|------|
| `_register_provider_importer(name, import_func)` | Adds a key → thunk entry |
| `_lazy_import_provider(name)` | Calls the thunk, caches the resolved classes, raises `ConfigurationError("UNKNOWN_PROVIDER")` or `("PROVIDER_IMPORT_ERROR")` |
| `_setup_provider_importers()` | Idempotent registration of the 7 built-ins; called by every public function |

Aliases registered: `azure` → `azure_openai`, `aws` → `bedrock`, `hf` → `huggingface`, `qdrant` → `fastembed`, `google` → `gemini`. Names are normalized via `provider.lower().replace("-", "_")`.

#### Public functions

| Function | Behavior |
|----------|----------|
| `list_available_providers()` (`@lru_cache(maxsize=1)`) | Returns `list(_PROVIDER_IMPORTERS.keys())` |
| `get_provider_info()` (`@lru_cache(maxsize=1)`) | Static dictionary of descriptions, deps, features, models, filtered by what's registered |
| `create_embedding_provider(provider, config=None, **kwargs)` | Main factory — see §6 |
| `create_openai_embedding(**kwargs)` … `create_gemini_vertex_embedding(**kwargs)` | Thin wrappers around `create_embedding_provider("<name>", **kwargs)` |
| `create_best_available_embedding(use_case="general", preference="balanced", **kwargs)` | Looks up a preference matrix (`enterprise`/`local`/`cost_effective`/`general` × `fast`/`quality`/`balanced`/`cheap`) and picks the first available provider; merges use-case defaults like `{"batch_size": 100, "cache_embeddings": True}` |
| `auto_detect_best_embedding(**kwargs)` | Inspects env vars (`OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`+`_ENDPOINT`, `GOOGLE_AI_API_KEY`/`GEMINI_API_KEY`, `AWS_ACCESS_KEY_ID`/`AWS_PROFILE`) and `torch.cuda.is_available()`; falls back to local providers; raises `ConfigurationError("NO_CONFIGURED_PROVIDERS")` if none usable |
| `get_embedding_recommendations(use_case, budget, privacy)` | Returns a list of `{provider, reason, pros, cons}` dicts — purely advisory metadata |

### 3.4 `openai_provider.py`

#### `OpenAIEmbeddingConfig(EmbeddingConfig)`

| Field | Default | Notes |
|-------|---------|-------|
| `api_key` | `None` | Falls back to `OPENAI_API_KEY` env |
| `organization` | `None` | OpenAI org header |
| `base_url` | `None` | For OpenAI-compatible endpoints |
| `model_name` | `"text-embedding-3-small"` | Validated; aliases like `"3-large"`, `"small"`, `"large"`, `"ada-002"` are auto-mapped |
| `enable_rate_limiting` | `True` | Toggles internal token-bucket-style guard |
| `requests_per_minute` | `3000` | Upper bound on RPM |
| `tokens_per_minute` | `1_000_000` | Upper bound on TPM |
| `parallel_requests` | `5` | Reserved (current code is sequential per batch) |
| `request_timeout` | `60.0` | Passed to `AsyncOpenAI(timeout=...)` |

A `field_validator('model_name')` rejects unknown models, listing the three valid ones.

#### `OpenAIEmbedding(EmbeddingProvider)`

- `__init__`: raises `ConfigurationError("DEPENDENCY_MISSING")` if `openai` isn't installed; calls `_setup_client()`; initializes `_request_times`, `_token_usage` lists and `_model_info` cache.
- `_setup_client`: builds `AsyncOpenAI(api_key=..., timeout=..., organization=?, base_url=?)`. Raises `ConfigurationError("API_KEY_MISSING")` if no key.
- `supported_modes`: `[DOCUMENT, QUERY]`.
- `pricing_info`: hard-coded prices per million tokens (`ada-002`: 0.10, `3-small`: 0.02, `3-large`: 0.13).
- `get_model_info`: maps each model to `{dimensions, max_tokens, description}`; `3-small`/`ada-002` → 1536, `3-large` → 3072; appends provider/type.
- `_check_rate_limits(estimated_tokens)`: trims `_request_times` and `_token_usage` to last 60 s, sleeps if either bucket would be exceeded, then records the new entry.
- `_estimate_tokens(texts)`: `int(total_chars / 4)`.
- `_embed_batch(texts, mode)`: enforces rate limits, calls `client.embeddings.create(model=..., input=texts, encoding_format="float")`, extracts `data.embedding`, accumulates `total_tokens`. Translates exceptions:
  - `openai.RateLimitError` → sleep parsed `retry-after` seconds (capped at 300) and re-raise (so the outer retry loop sees it).
  - `openai.AuthenticationError` → `ConfigurationError("AUTHENTICATION_ERROR")`.
  - `openai.BadRequestError` → `ModelConnectionError("API_REQUEST_ERROR")`.
  - other → `ModelConnectionError("EMBEDDING_FAILED")`.
- `validate_connection`: calls `_embed_batch(["test connection"], QUERY)`.
- `get_usage_stats`: returns recent RPM/TPM counters.
- `estimate_cost_detailed(num_texts, avg_text_length)`: extends `estimate_cost` with `cost_per_token`, `cost_breakdown`, and `comparison.vs_<other_model>` deltas.
- `close`: awaits `client.aclose()` / `client.close()` then `super().close()`.

Module-level helpers: `create_openai_embedding(model_name, api_key, **kwargs)`, `create_ada_002_embedding`, `create_3_small_embedding`, `create_3_large_embedding`.

### 3.5 `azure_openai_provider.py`

#### `AzureOpenAIEmbeddingConfig`

Adds Azure-specific fields:

| Field | Notes |
|-------|-------|
| `azure_endpoint` | Validator forces `https://` prefix and `.openai.azure.com/` suffix |
| `api_version` | Default `"2024-02-01"` |
| `deployment_name` | Validator allows letters/digits/`-`/`_` only |
| `use_managed_identity` | Toggle for Azure AD authentication |
| `tenant_id`, `client_id` | For `ManagedIdentityCredential` |
| `enable_content_filtering` | Reported in `get_compliance_info` |
| `data_residency_region` | Reported only |
| `parallel_requests` (3), `requests_per_minute` (240), `tokens_per_minute` (240_000) | Lower defaults than vanilla OpenAI |

#### `AzureOpenAIEmbedding`

- `_setup_managed_identity()`: requires `azure-identity`; uses `ManagedIdentityCredential(client_id=...)` when `client_id` set, else `DefaultAzureCredential`.
- `_setup_client()`: builds `AsyncAzureOpenAI` with either `azure_ad_token_provider=self._get_azure_token` or `api_key=...`.
- `_get_azure_token()`: awaits `self._credential.get_token("https://cognitiveservices.azure.com/.default")` and returns the `.token` string.
- `_embed_batch(texts, mode)`: identical shape to OpenAI but uses `model_param = deployment_name or model_name`. Adds Azure-flavoured error mapping (`AZURE_AUTH_ERROR`, `AZURE_API_ERROR`, `AZURE_EMBEDDING_FAILED`); on auth failure with managed identity, attempts a token refresh.
- `get_azure_info()` and `get_compliance_info()`: return diagnostic / compliance dictionaries (SOC 2, ISO 27001, HIPAA, FedRAMP, GDPR).
- `close`: closes the OpenAI client, then any `_credential` it created, then `super().close()`.

Helpers: `create_azure_openai_embedding(azure_endpoint, deployment_name, api_key=None, use_managed_identity=False, **kwargs)`, `create_azure_embedding_with_managed_identity(azure_endpoint, deployment_name, **kwargs)`.

### 3.6 `bedrock_provider.py`

#### `BedrockEmbeddingConfig`

| Field | Notes |
|-------|-------|
| `aws_access_key_id` / `aws_secret_access_key` / `aws_session_token` | Optional explicit creds |
| `region_name` | Default `"us-east-1"`; validator warns if not in `[us-east-1, us-west-2, ap-southeast-1, ap-northeast-1, eu-central-1, eu-west-1, eu-west-3]` |
| `profile_name` | AWS named profile alternative |
| `model_name` | Default `"amazon.titan-embed-text-v1"`; validator maps short names like `"titan"`, `"v2"`, `"cohere-embed-english"` |
| `model_id` / `inference_profile` | Override `model_name` for full ARN-style IDs |
| `enable_guardrails`, `guardrail_id` | Reserved (not used by current `_embed_batch`) |
| `enable_cloudwatch_logging`, `log_group_name` | Reported only |

#### `BedrockEmbedding`

- `_setup_aws_session`: builds `boto3.Session` with profile or explicit keys; calls `sts.get_caller_identity()` to validate; raises `ConfigurationError` (`AWS_CREDENTIALS_MISSING` / `AWS_SESSION_ERROR`) otherwise.
- `_setup_bedrock_client`: instantiates `bedrock-runtime` and `bedrock` (info) clients in the configured region.
- `pricing_info`: per-1k-token map (Titan v1: $0.0001, Titan v2: $0.00002, Cohere v3: $0.0001, Marengo: $0.0007) plus `per_million_tokens` derived field.
- `get_model_info`: per-model dim/max-tokens/languages dict; tries `bedrock_info_client.get_foundation_model(modelIdentifier=...)` to enrich with live ARN/status/customizations.
- `_prepare_titan_request(texts)`: `{"inputText": ..., "dimensions": ..., "normalize": ...}`.
- `_prepare_cohere_request(texts)`: `{"texts": [...], "input_type": "search_document"}`; switched to `"search_query"` for `EmbeddingMode.QUERY`.
- `_prepare_request_body(texts, mode)`: dispatches by model prefix.
- `_extract_embeddings(response_body, num_texts)`: unwraps `embedding`/`embeddings`/`vectors` for Titan/Cohere/generic.
- `_embed_batch(texts, mode)`: chunks into Bedrock-friendly sub-batches (`min(config.batch_size, 25)`), calls `bedrock_client.invoke_model(modelId, body=json.dumps(body))`, increments `_invocation_count` and `_total_input_tokens`, sleeps 0.1 s between sub-batches. Maps `ThrottlingException` to a re-raise after sleep, `ValidationException`/`ModelNotReadyException` to `ConfigurationError`, others to `ModelConnectionError`.
- `get_aws_info`, `get_cost_estimate`, `list_available_models()` (uses `bedrock_info_client.list_foundation_models(byOutputModality='EMBEDDING')`): observability helpers.
- `close`: closes both Bedrock clients and the `Session`.

Helpers: `create_titan_embedding(region_name="us-east-1", model_version="v1"|"v2", **kwargs)`, `create_cohere_embedding(language="english"|"multilingual", region_name="us-east-1", **kwargs)`.

### 3.7 `huggingface_provider.py`

#### `HuggingFaceEmbeddingConfig`

| Field | Notes |
|-------|-------|
| `model_name` | Default `"sentence-transformers/all-MiniLM-L6-v2"` |
| `hf_token` | Falls back to `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` |
| `use_api` | If `True`, route through `huggingface_hub.InferenceClient` |
| `use_local` | If `True`, load a local `transformers` model |
| `device` | `cuda` / `mps` / `cpu`; auto-detected when `None` |
| `torch_dtype` | `float16`/`float32`/`bfloat16` (validator) |
| `trust_remote_code` | Forwarded to `from_pretrained` |
| `max_seq_length` | Capped at 512 in `_embed_local` |
| `pooling_strategy` | `mean`/`cls`/`max`/`mean_sqrt_len` (validator) |
| `enable_quantization` | Toggle bitsandbytes |
| `quantization_bits` | `4` (NF4 + bf16 compute) or `8` (load_in_8bit) |
| `enable_gradient_checkpointing` | Memory savings |
| `wait_for_model`, `timeout` | Used by InferenceClient |
| `cache_dir`, `force_download` | Forwarded to `from_pretrained` |

#### `HuggingFaceEmbedding`

- `_setup_device`: auto-picks `cuda` > `mps` > `cpu`.
- `_setup_authentication`: calls `huggingface_hub.login(token=...)` if a token is found.
- `_setup_local_model`: builds `model_kwargs = {trust_remote_code, cache_dir, force_download}`; configures bitsandbytes quantization if requested; loads `AutoTokenizer` + `AutoModel`; moves to device unless `device_map="auto"`; calls `eval()` and optional `gradient_checkpointing_enable()`. Raises `ConfigurationError("DEPENDENCY_MISSING")` mentioning bitsandbytes/accelerate when the missing-dep error matches.
- `_setup_api_session`: creates `InferenceClient(token=hf_token, timeout=...)`.
- `supported_modes`: all four (`DOCUMENT`, `QUERY`, `SYMMETRIC`).
- `pricing_info`: $0 for local, $0.01/M tokens placeholder for API.
- `get_model_info`: includes parameter count, hidden_size, max_position_embeddings, vocab_size when available.
- Pooling: `_mean_pooling` (mask-weighted), `_cls_pooling` (`[CLS]` token), `_max_pooling` (mask-aware), and `_apply_pooling` dispatcher.
- `_embed_local(texts)`: tokenizes with `padding=True, truncation=True, max_length=min(model_max_length, 512)`, runs `model(**inputs)` under `torch.no_grad()`, applies pooling, optional `F.normalize(p=2, dim=1)`, returns `.cpu().numpy().tolist()`.
- `_embed_api(texts)`: wraps `InferenceClient.feature_extraction(text=texts, model=...)` in `asyncio.to_thread`; coerces NumPy arrays to lists.
- `_embed_batch`: dispatches to API or local based on `use_api`.
- `get_memory_usage`: queries `torch.cuda.memory_allocated/reserved/get_device_properties`.
- `close`: deletes model, tokenizer, InferenceClient; calls `torch.cuda.empty_cache()` if on CUDA.
- `remove_local_cache()`: uses `huggingface_hub.try_to_load_from_cache` + `shutil.rmtree` to delete the model from disk.

Helpers: `create_sentence_transformer_embedding`, `create_mpnet_embedding`, `create_minilm_embedding`, `create_huggingface_api_embedding`.

### 3.8 `fastembed_provider.py`

#### `FastEmbedConfig`

| Field | Notes |
|-------|-------|
| `model_name` | Default `"BAAI/bge-small-en-v1.5"` |
| `cache_dir` | ONNX model cache; defaults under `~/.cache/fastembed` |
| `threads` | ONNX intra-op threads |
| `providers` | Validated list of ONNX providers (`CPUExecutionProvider`, `CUDAExecutionProvider`, `ROCMExecutionProvider`, `CoreMLExecutionProvider`, `DmlExecutionProvider`) |
| `enable_gpu` | Auto-prepend GPU provider if the runtime has it |
| `doc_embed_type` | `default` or `passage` |
| `enable_sparse_embeddings` | Use SPLADE-style sparse model |
| `sparse_model_name` | Override sparse model name |
| `model_warmup` | Run a synthetic batch on init |

#### `FastEmbedProvider`

- `_setup_providers`: detects available ONNX providers via `ort.get_available_providers()` and prepends GPU options when `enable_gpu=True`.
- `_initialize_models`: instantiates `TextEmbedding(model_name=..., cache_dir=..., threads=..., providers=...)`; optionally adds `SparseTextEmbedding`; runs warmup if `model_warmup`.
- `pricing_info`: $0 always (local).
- `_embed_batch`: routes to `sparse_model.embed(texts)`, `embedding_model.query_embed(texts)` (for `EmbeddingMode.QUERY`), `embedding_model.passage_embed(texts)` (for `doc_embed_type="passage"`), or `embedding_model.embed(texts)`. Iterators are materialized through `_process_embeddings`. Optional unit-norm normalization (overrides the base implementation with a NumPy-only one).
- `get_performance_info`, `list_available_models()` (`TextEmbedding.list_supported_models()`), `get_cache_info()` (extends base with disk-cache size walk).
- `close`: deletes both models and best-effort closes the ONNX runtime session.

Helpers: `create_bge_small_embedding`, `create_bge_large_embedding`, `create_e5_embedding`, `create_sparse_embedding` (uses `prithivida/Splade_PP_en_v1`), `create_gpu_accelerated_embedding`.

### 3.9 `ollama_provider.py`

#### `OllamaEmbeddingConfig`

| Field | Notes |
|-------|-------|
| `base_url` | Default `"http://localhost:11434"`; validator prepends `http://` and strips trailing `/` |
| `model_name` | Default `"nomic-embed-text"`; validator warns on unknown names; tag-style (`name:tag`) accepted as-is |
| `auto_pull_model` | If `True`, pull the model when missing |
| `keep_alive` | Default `"5m"`; forwarded to Ollama |
| `temperature` / `top_p` / `num_ctx` | Optional model `options` |
| `request_timeout` (120 s), `connection_timeout` (10 s), `max_retries` (3) | Tuned for local-but-slow hosts |
| `enable_keep_alive`, `enable_model_preload` | Preload behavior |

#### `OllamaEmbedding`

- `_setup_http_session`: builds a `requests.Session` with a `urllib3` `Retry(total=max_retries, backoff_factor=1, status_forcelist=[429,500,502,503,504])` adapter on both `http://` and `https://`.
- `_initialize_sync` (called from `__init__`): pings `/api/version`, then `_ensure_model_available_sync` if `auto_pull_model`.
- `_initialize_async`: async variant that also calls `_preload_model()` (a single dummy embed) when `enable_model_preload`.
- `_check_ollama_health[_sync]`: GETs `/api/version`.
- `_list_models[_sync]`: GETs `/api/tags`.
- `_pull_model[_sync]`: POSTs `/api/pull` with `{"name": model_name}`; the async version streams JSON progress lines.
- `_make_embedding_request(texts)`: POSTs `/api/embeddings` with `{"model", "prompt": texts[0]_or_list, "keep_alive", "options": {...}}` via a fresh `aiohttp.ClientSession` (no shared session).
- `_embed_batch(texts, mode)`: throttles a periodic health check (every 60 s), then loops one text at a time through `_make_embedding_request` (Ollama's `/api/embeddings` is single-prompt), accumulating into `all_embeddings`, with a 10 ms sleep between requests. Errors become `ModelConnectionError` with `OLLAMA_CONNECTION_ERROR` or `OLLAMA_EMBEDDING_ERROR`.
- `get_server_info()`, `get_model_details()` (POSTs `/api/show`).
- `close` and `__del__`: best-effort close of the `requests.Session`.

Helpers: `create_nomic_embedding`, `create_mxbai_embedding`, `create_arctic_embedding`, `create_ollama_embedding_with_auto_setup(model_name, base_url, ...)`.

### 3.10 `gemini_provider.py`

#### `GeminiEmbeddingConfig`

| Field | Notes |
|-------|-------|
| `api_key` | Falls back to `GOOGLE_API_KEY` / `GOOGLE_AI_API_KEY` / `GEMINI_API_KEY` |
| `model_name` | Default `"gemini-embedding-001"`; validator warns on names outside `[gemini-embedding-001, gemini-embedding-2-preview, text-embedding-005, text-multilingual-embedding-002, embedding-001]` |
| `enable_safety_filtering`, `safety_settings` | Default settings block harassment / hate / sexual / dangerous content above MEDIUM |
| `task_type` | One of `RETRIEVAL_QUERY/_DOCUMENT`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`, `QUESTION_ANSWERING`, `FACT_VERIFICATION`, `CODE_RETRIEVAL_QUERY` (validator) |
| `title` | Optional context title |
| `enable_batch_processing` | Toggle the batched path (currently still sequential — see below) |
| `use_google_cloud_auth` / `project_id` / `location` (`us-central1`) | Vertex AI mode |
| `requests_per_minute` | 60 |
| `use_vertex_ai` | If `True`, build `genai.Client(vertexai=True, project=..., location=...)` |
| `api_version` | `v1beta` / `v1` / `v1alpha` (validator) |
| `enable_caching`, `cache_ttl_seconds` | Build a `types.CacheConfig` |
| `output_dimensionality` | 128–3072 (validator) — uses Matryoshka Representation Learning to truncate the vector |
| `embedding_config` | Free-form extra fields applied to `types.EmbedContentConfig` |

#### `GeminiEmbedding`

- `_setup_authentication`: when Vertex / Cloud auth, calls `google.auth.default()`, captures `credentials` and `project_id`. Otherwise insists on an API key.
- `_setup_client`: builds either `genai.Client(vertexai=True, project=..., location=..., http_options=HttpOptions(api_version=...))` or `genai.Client(api_key=..., http_options=...)`. Optionally sets up `_setup_caching()`.
- `supported_modes`: all four.
- `pricing_info`: per-1k-character price (Gemini bills by characters); `per_million_tokens` is derived as `price * 1000 * 4`.
- `get_model_info`: per-model dims/max-tokens/supported-tasks. `gemini-embedding-001` and `gemini-embedding-2-preview` advertise MRL output sizes `[3072, 1536, 768]`.
- `_get_task_type_for_mode(mode)`: maps `DOCUMENT→RETRIEVAL_DOCUMENT`, `QUERY→RETRIEVAL_QUERY`, `SYMMETRIC→SEMANTIC_SIMILARITY`, `CLUSTERING→CLUSTERING`; falls back to `config.task_type`.
- `_check_rate_limits`: tracks `_request_times` against `requests_per_minute` and sleeps to stay under.
- `_embed_single_text(text, task_type)`: builds an optional `types.EmbedContentConfig(output_dimensionality=..., **embedding_config)`, calls `client.models.embed_content(model=..., contents=text, config=config)`, returns `response.embeddings[0].values`. Empty-string inputs return a zero vector of the appropriate dim.
- `_embed_texts_batch(texts, task_type)`: today's implementation is still sequential per-text but wrapped in a batch-aware fallback path.
- `_embed_batch`: classifies failures into `GEMINI_RATE_LIMIT`, `GEMINI_SAFETY_ERROR` (raised as `ConfigurationError`), or `GEMINI_EMBEDDING_ERROR`.
- `get_usage_stats`, `get_safety_info`, `list_available_models()` (`client.models.list()` filtered by `embedding`).
- `close`: closes client, cache, and any `_credential`.

Helpers: `create_gemini_document_embedding`, `create_gemini_query_embedding`, `create_gemini_semantic_embedding`, `create_gemini_cloud_embedding(project_id, location)`, `create_gemini_vertex_embedding(project_id, location)`.

## 4. Subfolders

There are none. All concrete providers live as sibling modules of `base.py` under `src/upsonic/embeddings/`.

## 5. Cross-file relationships

### 5.1 Inheritance

```
EmbeddingProvider (base.py, ABC + Pydantic)
├── OpenAIEmbedding             (openai_provider.py)
├── AzureOpenAIEmbedding        (azure_openai_provider.py)
├── BedrockEmbedding            (bedrock_provider.py)
├── HuggingFaceEmbedding        (huggingface_provider.py)
├── FastEmbedProvider           (fastembed_provider.py)
├── OllamaEmbedding             (ollama_provider.py)
└── GeminiEmbedding             (gemini_provider.py)

EmbeddingConfig (base.py)
├── OpenAIEmbeddingConfig
├── AzureOpenAIEmbeddingConfig
├── BedrockEmbeddingConfig
├── HuggingFaceEmbeddingConfig
├── FastEmbedConfig
├── OllamaEmbeddingConfig
└── GeminiEmbeddingConfig
```

### 5.2 Registry

`factory.py` keeps `_PROVIDER_IMPORTERS` keyed by canonical names (`openai`, `azure_openai`, `bedrock`, `huggingface`, `fastembed`, `ollama`, `gemini`) with aliases (`azure`, `aws`, `hf`, `qdrant`, `google`). Each value is a thunk that imports the provider module and returns the `(Provider, Config)` pair.

```
list_available_providers() ───▶ list(_PROVIDER_IMPORTERS.keys())
create_embedding_provider(name, ...)
        │
        ▼
_lazy_import_provider(name)  ──▶  thunk()  ──▶  (ProviderCls, ConfigCls)
        │                              │
        └─────────► caches into ──────┘ (_PROVIDER_REGISTRY, _PROVIDER_CONFIGS)
```

### 5.3 Lazy public surface

`__init__.py` declares everything in `__all__` but resolves each name on first attribute access via `__getattr__`. The same lazy thunks are used both at the package level and inside `factory.py`, so a downstream `from upsonic.embeddings import create_embedding_provider; create_embedding_provider("openai")` only imports `base.py`, `factory.py`, and `openai_provider.py`.

### 5.4 Shared utilities

| Imported symbol | Source | Used by |
|-----------------|--------|---------|
| `Chunk` | `upsonic.schemas.data_models` | `base.py` (`embed_documents`) |
| `ConfigurationError`, `ModelConnectionError` | `upsonic.utils.package.exception` | every provider for typed errors |
| `console`, `Panel`, `Table` | `upsonic.utils.printing` | `base.py` progress UI |
| `info_log`, `debug_log`, `warning_log`, `error_log`, `success_log`, `connection_info`, `import_error` | `upsonic.utils.printing` | every provider |

## 6. Public API

The full surface re-exported by `from upsonic.embeddings import …`:

### 6.1 Base

- `EmbeddingProvider`
- `EmbeddingConfig`
- `EmbeddingMode`
- `EmbeddingMetrics`

### 6.2 Provider classes / configs

| Provider | Class | Config |
|----------|-------|--------|
| OpenAI | `OpenAIEmbedding` | `OpenAIEmbeddingConfig` |
| Azure OpenAI | `AzureOpenAIEmbedding` | `AzureOpenAIEmbeddingConfig` |
| AWS Bedrock | `BedrockEmbedding` | `BedrockEmbeddingConfig` |
| HuggingFace | `HuggingFaceEmbedding` | `HuggingFaceEmbeddingConfig` |
| FastEmbed | `FastEmbedProvider` | `FastEmbedConfig` |
| Ollama | `OllamaEmbedding` | `OllamaEmbeddingConfig` |
| Google Gemini | `GeminiEmbedding` | `GeminiEmbeddingConfig` |

### 6.3 Factory & discovery

- `create_embedding_provider(provider, config=None, **kwargs)`
- `list_available_providers()`
- `get_provider_info()`
- `create_best_available_embedding(use_case="general", preference="balanced", **kwargs)`
- `auto_detect_best_embedding(**kwargs)`
- `get_embedding_recommendations(use_case="general", budget="medium", privacy="standard")`

### 6.4 Per-provider one-liners

| Helper | Module |
|--------|--------|
| `create_openai_embedding`, `create_ada_002_embedding`, `create_3_small_embedding`, `create_3_large_embedding` | `openai_provider.py` |
| `create_azure_openai_embedding`, `create_azure_embedding_with_managed_identity` | `azure_openai_provider.py` |
| `create_bedrock_embedding`, `create_titan_embedding`, `create_cohere_embedding` | `bedrock_provider.py` |
| `create_huggingface_embedding`, `create_sentence_transformer_embedding`, `create_mpnet_embedding`, `create_minilm_embedding`, `create_huggingface_api_embedding` | `huggingface_provider.py` |
| `create_fastembed_provider`, `create_bge_small_embedding`, `create_bge_large_embedding`, `create_e5_embedding`, `create_sparse_embedding`, `create_gpu_accelerated_embedding` | `fastembed_provider.py` |
| `create_ollama_embedding`, `create_nomic_embedding`, `create_mxbai_embedding`, `create_arctic_embedding`, `create_ollama_embedding_with_auto_setup` | `ollama_provider.py` |
| `create_gemini_embedding`, `create_gemini_vertex_embedding`, `create_gemini_document_embedding`, `create_gemini_query_embedding`, `create_gemini_semantic_embedding`, `create_gemini_cloud_embedding` | `gemini_provider.py` |

### 6.5 Per-provider feature matrix

| Provider | Modes | Local | Cloud auth | Streaming pull | Adaptive batch | Quantization | Sparse |
|----------|-------|-------|------------|----------------|----------------|--------------|--------|
| OpenAI | DOC, QUERY | — | API key / org | — | yes (base) | — | — |
| Azure OpenAI | DOC, QUERY | — | API key + Managed Identity | — | yes (base) | — | — |
| Bedrock | DOC, QUERY | — | IAM (boto3 session) | — | yes (base) | — | — |
| HuggingFace | DOC, QUERY, SYM | yes | HF token (API) | — | yes (base) | 4-bit / 8-bit | — |
| FastEmbed | DOC, QUERY, SYM | yes (ONNX) | — | — | yes (base) | — | yes (SPLADE) |
| Ollama | DOC, QUERY, SYM | yes (HTTP) | — | yes (`/api/pull` JSON stream) | yes (base) | — | — |
| Gemini | all four | — | API key or ADC + Vertex | — | yes (base) | — | MRL truncation |

## 7. Integration with the rest of Upsonic

### 7.1 Knowledge base / RAG

- `KnowledgeBase` (`src/upsonic/knowledge_base/knowledge_base.py`) accepts an `EmbeddingProvider` and calls `provider.embed_documents(chunks)` during ingestion. The vectors are then handed to a `vectordb` adapter (Pinecone/Chroma/Qdrant/FAISS/Weaviate/Milvus, …) keyed by `Chunk.id`.
- During retrieval, the same provider's `embed_query(query_string)` produces the search vector. Because `embed_documents` and `embed_query` go through `_embed_batch` with different `EmbeddingMode` values, asymmetric models (Gemini, Cohere, BGE-passage) automatically use the correct task type.

### 7.2 Vector store dimension contract

`get_model_info()['dimensions']` is the source of truth for index dimensionality. Adapters in `vectordb/` read this when constructing indices, so swapping providers requires re-indexing if dimensions differ (e.g. `text-embedding-3-large` 3072 vs `nomic-embed-text` 768).

### 7.3 Errors, logging, progress

- All provider errors are normalized to `ConfigurationError` (user-fixable, never retried) or `ModelConnectionError` (retriable, gets the full backoff-and-bisect treatment) from `upsonic.utils.package.exception`.
- Logs flow through `upsonic.utils.printing` (`info_log`, `warning_log`, `debug_log`, `error_log`, `success_log`, `connection_info`, `import_error`). The base class also exposes a Rich progress `Table` via `_create_progress_display`.
- `EmbeddingMetrics` is queryable via `provider.get_metrics()` for cost dashboards.

### 7.4 Optional dependencies

Each provider raises `ConfigurationError("DEPENDENCY_MISSING")` (or `import_error(...)`) if its third-party dependency is missing — this is what makes the lazy `__init__.py` and lazy factory import critical: an Upsonic install with only `openai` will not crash on `import upsonic.embeddings`, only on `create_embedding_provider("fastembed")`.

## 8. End-to-end flow of an embedding call

The reference path is `OpenAIEmbedding` embedding a list of `Chunk`s. The same flow applies to every provider — only `_embed_batch` differs.

```
caller code
└─ provider = create_embedding_provider("openai", model_name="text-embedding-3-small")
   │
   │ factory.py
   │    _setup_provider_importers()
   │    _lazy_import_provider("openai")
   │      └─ from .openai_provider import OpenAIEmbedding, OpenAIEmbeddingConfig   (only now)
   │    OpenAIEmbeddingConfig(model_name=..., **kwargs)   # field validators run
   │    OpenAIEmbedding(config=config)
   │      └─ _setup_client()  →  AsyncOpenAI(api_key=…, timeout=…)
   ▼
caller code
└─ vectors = await provider.embed_documents(chunks)
   │
   ▼  base.py: embed_documents(chunks)
   │      texts = [c.text_content for c in chunks]
   │      return await self.embed_texts(texts, mode=DOCUMENT)
   ▼  base.py: embed_texts(texts, mode, show_progress, metadata)
       1. _metrics.total_chunks = len(texts)
       2. if cache_embeddings: (embeddings, uncached_indices) = _check_cache(texts, metadata)
       3. for batch in chunks(uncached_texts, batch_size):
              batch_embeddings = await _embed_with_retry(batch, mode)
              ...
       4. if cache_embeddings: _update_cache(...)
       5. if normalize_embeddings: embeddings = _normalize_embeddings(embeddings)
       6. _metrics.embedding_time_ms = (time.time() - start) * 1000
       7. return embeddings
   ▼  base.py: _embed_with_retry(batch, mode)
       for attempt in range(max_retries):
           try:    return await self._embed_batch(batch, mode)
           except ConfigurationError:                raise         # fail fast
           except Exception:
               if last attempt:                      raise ModelConnectionError(...)
               sleep(retry_delay * 2**attempt)
               if enable_adaptive_batching and len(batch) > 1:
                   first  = await _embed_with_retry(batch[:mid], mode)
                   second = await _embed_with_retry(batch[mid:], mode)
                   return first + second
   ▼  openai_provider.py: _embed_batch(texts, mode)
       _check_rate_limits(estimated_tokens=int(total_chars/4))
           ├─ trim _request_times / _token_usage to last 60 s
           ├─ if RPM saturated:    asyncio.sleep(60 - (now - oldest_request))
           ├─ if TPM saturated:    asyncio.sleep(60 - (now - oldest_token_window))
           └─ append (now, estimated_tokens)
       response = await client.embeddings.create(model=model_name, input=texts,
                                                 encoding_format="float")
       _metrics.total_tokens += response.usage.total_tokens
       return [d.embedding for d in response.data]
   ▼  base.py: _normalize_embeddings(embeddings)
       for v in embeddings: v / ||v||  (NumPy if available, pure-Python fallback)
   ▼
returned to caller as List[List[float]]
```

For other providers, the only thing that changes is what happens inside `_embed_batch`:

- **Azure OpenAI** uses `deployment_name or model_name`, mints Azure AD tokens via `_get_azure_token()` when managed identity is on.
- **Bedrock** chunks into ≤25-text sub-batches and calls `bedrock_client.invoke_model(modelId, body=json.dumps(_prepare_request_body(...)))`.
- **HuggingFace local** tokenizes → `model(**inputs)` under `torch.no_grad()` → `_apply_pooling` → optional `F.normalize`.
- **HuggingFace API** wraps `InferenceClient.feature_extraction` in `asyncio.to_thread`.
- **FastEmbed** picks `embed`/`query_embed`/`passage_embed`/`sparse_model.embed` based on mode + `doc_embed_type`.
- **Ollama** loops one text at a time through POST `/api/embeddings` (single-prompt API).
- **Gemini** calls `client.models.embed_content(model, contents, config=EmbedContentConfig(output_dimensionality=...))` per text (with batch fallback) and tags the request with the right `task_type` derived from `EmbeddingMode`.

The orchestration layer in `base.py` is invariant across all of them, which is what gives Upsonic a single, predictable embedding contract regardless of the backend the user picks.
