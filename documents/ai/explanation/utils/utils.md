---
name: utils-shared-infrastructure
description: Use when working with Upsonic's cross-cutting utility belt under `src/upsonic/utils/` — Rich console rendering, logging/Sentry/OpenTelemetry bootstrap, retry decorators, error wrapping, token/cost calculation, or per-vendor integration helpers. Use when a user asks to print agent panels, calculate model costs, configure logging, retry flaky calls, extract tokens/tool calls from `AgentRunOutput`, validate attachments, manipulate datetimes, stream pipeline events, or work with Upsonic exceptions. Trigger when the user mentions printing.py, usage.py, MODEL_CONTEXT_WINDOWS, calculate_cost, format_cost, get_estimated_cost, price_id_summary, call_end, agent_end, print_price_id_summary, print_agent_metrics, retryable, RetryMode, upsonic_error_handler, UupsonicError, ModelRetry, GuardrailValidationError, RetryExhaustedError, llm_usage, tool_usage, AsyncExecutionMixin, run_async, Timer, dttm, validators, attachments, file_helpers, image helpers, messages serialization, pipeline step indices, agent events, ayield_*_event, integrations (Apify, WhatsApp, Gmail, Telegram, Discord, Exa, Firecrawl, Crawlee), Configuration, system_id, get_library_version, logging_config, setup_sentry, setup_opentelemetry, or genai_prices.
---

# `src/upsonic/utils/` — Shared Utility Infrastructure

## 1. What this folder is

`src/upsonic/utils/` is the cross-cutting **utility belt** of the Upsonic
framework. Almost every other module under `src/upsonic/` reaches into it for
one of three things:

1. **User-facing presentation** — Rich-based panels, tables, costs, retries,
   pipeline progress, agent metrics, safety/cache/policy banners (`printing.py`).
2. **Background plumbing** — logging configuration that drives both the local
   console and Sentry telemetry, async/sync bridging, retry decorators, error
   wrapping, file/image helpers, message extraction, datetime helpers, timing,
   pipeline step indices, attachment validators.
3. **Token / cost accounting** — model context window registry, token-to-dollar
   pricing via `genai_prices`, cost formatting, and cost extraction from
   `AgentRunOutput`, `RequestUsage`, `RunUsage`, `Agent` and `Task`.

Three sub-areas hang off the package:

- `agent/events.py` — small wrappers that yield run-time event objects
  (`PipelineStartEvent`, `ToolCallEvent`, `ModelResponseEvent`, etc.) for the
  streaming run-loop in both sync and async flavours.
- `integrations/` — small per-vendor helpers (WhatsApp, Apify, Exa, Telegram,
  Discord, Gmail, Crawlee, Firecrawl) that the matching modules under
  `src/upsonic/tools/custom_tools/` rely on for serialization / sanitisation /
  authentication.
- `package/` — process-level concerns: package version detection, persistent
  on-disk configuration in `~/.upsonic/config.json`, system ID management, and
  the framework-wide exception hierarchy (`UupsonicError`,
  `GuardrailValidationError`, `ModelRetry`, `SkillError`, etc.).

The package's `__init__.py` only re-exports a small curated cost / printing
surface; everything else is imported by absolute path (`from upsonic.utils.X`)
to keep the import graph lazy and avoid circular imports with the agent code.

> Code lives at `/Users/dogankeskin/Desktop/Upsonic/src/upsonic/utils/`. Around
> 145 files under `src/upsonic/` import from this package.

## 2. Folder layout

```
src/upsonic/utils/
├── __init__.py              # Lazy public re-exports (printing + cost API)
├── async_utils.py           # run_async + AsyncExecutionMixin
├── dttm.py                  # Datetime / epoch helpers
├── error_wrapper.py         # @upsonic_error_handler decorator + error mapping
├── file_helpers.py          # get_clean_extension
├── image.py                 # Image URL → base64, save, open, list
├── llm_usage.py             # Token usage extraction from AgentRunOutput
├── logging_config.py        # Sentry + Python logging + OpenTelemetry bootstrap
├── messages.py              # ModelResponse part extraction + (de)serialization
├── pipeline.py              # Step name/index lookup helpers
├── printing.py              # ~4090 lines of Rich panels, tables, log banners
├── retry.py                 # @retryable decorator (sync + async)
├── timer.py                 # Timer context manager
├── tool_usage.py            # Tool call extraction from AgentRunOutput
├── usage.py                 # Cost calculation, MODEL_CONTEXT_WINDOWS registry
├── validators.py            # Attachment file existence / readability checks
│
├── agent/
│   ├── __init__.py          # Lazy attr forwarding to events.py
│   └── events.py            # ~1290 lines of (a)yield_*_event helpers
│
├── integrations/
│   ├── __init__.py          # Empty marker (docstring only)
│   ├── apify.py             # Apify client + JSON-schema conversion
│   ├── crawlee.py           # Re-exports run_async
│   ├── discord.py           # sanitize_text_for_discord
│   ├── exa.py               # serialize_exa_response
│   ├── firecrawl.py         # serialize_firecrawl_response
│   ├── gmail.py             # @authenticate decorator + email helpers
│   ├── telegram.py          # sanitize_text_for_telegram
│   └── whatsapp.py          # WhatsApp Cloud API client functions
│
└── package/
    ├── __init__.py          # (intentionally empty)
    ├── exception.py         # All Upsonic exception classes
    ├── get_version.py       # get_library_version()
    ├── storage.py           # Configuration class (~/.upsonic/config.json)
    └── system_id.py         # get_system_id() / generate_system_id()
```

## 3. Top-level files (grouped by purpose)

### 3.1 Logging & telemetry — `logging_config.py`

Bootstraps three subsystems on import:

| Subsystem | Trigger | Effect |
|---|---|---|
| **Sentry** | Always called via `setup_sentry()` at module bottom | Initialises `sentry_sdk` in error-only mode (sample rate 0.0), tags release with `upsonic@<version>`, sets user id from `package.system_id.get_system_id()` and registers an `atexit` flush. Skipped on Python ≥ 3.14 due to pydantic/fastapi compatibility. |
| **Python `logging`** | Only if `UPSONIC_LOG_LEVEL` or `UPSONIC_LOG_FILE` env var is set | Configures a root `upsonic` logger with stream and/or file handlers and three formats (`simple`, `detailed`, `json`). Otherwise just attaches a `NullHandler` (library best practice). |
| **OpenTelemetry** | Only if `UPSONIC_OTEL_ENABLED` env var is set | Imports `upsonic.integrations.tracing.DefaultTracingProvider` and calls `Agent.instrument_all()`. |

Public helpers:

| Symbol | Purpose |
|---|---|
| `setup_logging(level, log_format, log_file, force_reconfigure, enable_console)` | Master configuration entry point. `enable_console=False` is what `printing.py` uses so Rich owns the console while file/Sentry continue to receive logs. |
| `setup_sentry()` | Idempotent; honours `UPSONIC_TELEMETRY` (DSN or `"false"` to disable). |
| `setup_opentelemetry()` | Idempotent OTLP bootstrap. |
| `get_logger(name)` | Auto-configures on first use. Standard module-level pattern. |
| `set_module_log_level(module, level)` | Run-time override for the patterns in `MODULE_PATTERNS`. |
| `disable_logging()` | Replace handlers with `NullHandler`, set level above `CRITICAL`. |
| `get_current_log_levels()` | Snapshot of every pattern logger's level. |
| `memory_debug_log(memory_debug, msg, data)` | Plain `print()` based debug helper for memory subsystem. |
| `get_env_bool` / `get_env_bool_optional` / `get_env_log_level` | Env-var coercion helpers. |

`MODULE_PATTERNS` defines the pattern→pattern mapping for module-scoped levels:
`loaders`, `text_splitter`, `vectordb`, `agent`, `team`, `tools`, `cache`,
`memory`, `embeddings`. Each can be overridden through
`UPSONIC_LOG_LEVEL_<MODULE>` env vars.

### 3.2 Console rendering — `printing.py`

The framework's user-facing presentation layer. Around 4090 lines of Rich-based
helpers, organised below by purpose. They all share a single module-level
`Console(file=_StdoutProxy())` so output respects `redirect_stdout`.

Two background loggers route everything to the right place:

```python
_bg_logger     = get_logger("upsonic.user")    # File / general background
_sentry_logger = get_logger("upsonic.sentry")  # INFO+ → Sentry events via LoggingIntegration
```

Cost helpers in `printing.py` are thin wrappers around `usage.py` that fall
back to `~$0.0000` on failure. The dedup is intentional: external code can
import either, and a tag-only error never surfaces as an exception.

#### Display tables for runs and tasks

| Function | Renders |
|---|---|
| `display_pydantic_structured_output` | Field/Type/Value table for Pydantic results, with metadata panel (model, tokens, cost). |
| `display_llm_result_table` | General LLM result display, falling back to the Pydantic version when applicable. |
| `display_tool_calls_table` | Per-tool panel with parameters and result, with `func_dict` extraction. |
| `display_tool_results_table` | Tabular tool-results summary. |
| `display_graph_tree` | Graph node tree with `✓ Completed`, `⚡ Executing`, `⊘ Pruned`, `✗ Failed`, `○ Pending` markers, used live by `upsonic.graph`. |

#### Run / call lifecycle

| Function | When fired |
|---|---|
| `connected_to_server` | MCP / external server connection panel. |
| `call_end` | LLM call finished — handles `price_id_summary` accumulation, optional output rendering, Sentry event with model/cost/tools. |
| `agent_end` | Agent run finished — same accumulation plus tool/context counts. |
| `agent_total_cost` | Aggregate cost panel. |
| `print_price_id_summary(price_id, task)` | Reads timing from `task.usage` (TaskUsage), prints Task Metrics panel including duration, model time, tool time, pause time, framework overhead. |
| `print_agent_metrics(agent)` | Aggregate panel from `agent.usage` (AgentUsage). |
| `agent_retry` / `call_retry` | Retry banners. |
| `agent_started` | "🚀 Started to work" banner + Sentry event. |

#### Logging helpers (Rich console + background logger)

| Function | Style |
|---|---|
| `info_log(msg, ctx)` | `[INFO]` blue |
| `warning_log(msg, ctx)` | `[WARNING]` yellow |
| `error_log(msg, ctx)` | `[ERROR]` red — also routes to Sentry via `LoggingIntegration`. |
| `success_log(msg, ctx)` | Plain success text + background `INFO`. |
| `debug_log(msg, ctx, debug, debug_level)` | Level-1 debug. |
| `debug_log_level2(msg, ctx, debug, debug_level, **details)` | Level-2 detail panel; full key/value layout. |
| `simple_output(msg)` | Plain console print. |
| `connection_info(provider, version)` | Provider connect line. |

#### Tooling / dependencies / API keys

`mcp_tool_operation`, `tool_operation`, `print_orchestrator_tool_step`,
`error_message`, `missing_dependencies`, `missing_api_key`, `import_error`
(this one **raises `ImportError`** after rendering).

#### Cache

`cache_hit`, `cache_miss`, `cache_stored`, `cache_stats`, `cache_cleared`,
`cache_configuration`.

#### Pipeline lifecycle

`pipeline_started`, `pipeline_step_started`, `pipeline_step_completed`,
`pipeline_completed`, `pipeline_failed`, `pipeline_paused`,
`pipeline_timeline` (sorted bar chart of step durations).

#### Direct (non-agent) execution

`direct_started`, `direct_completed`, `direct_error`, `direct_metrics_summary`,
`direct_configuration`.

#### Safety / Policy / Reflection

`policy_triggered`, `tool_safety_check`, `skill_safety_check`,
`policy_feedback_generated`, `policy_feedback_retry`,
`policy_feedback_exhausted`, `user_policy_feedback_returned`,
`agent_policy_feedback_success`, `reflection_started`, `reflection_evaluation`,
`reflection_improvement_started`, `reflection_completed`,
`anonymization_debug_panel`.

#### Culture, OCR, planning, deep-agent

`culture_info`, `culture_debug`, `culture_warning`, `culture_error`,
`culture_knowledge_added/updated/deleted`, `culture_extraction_started`,
`culture_extraction_completed`, `ocr_loading`, `ocr_initialized`,
`ocr_language_not_supported`, `ocr_language_warning`, `planning_todo_list`,
`planning_todo_update`, `deep_agent_todo_completion_check`,
`deep_agent_all_todos_completed`, `deep_agent_max_iterations_warning`.

#### Misc

`compression_fallback`, `model_recommendation_summary`,
`model_recommendation_error`, `escape_rich_markup`, `spacing`.

#### Cost surfaces in `printing.py`

| Function | Returns | Source of price |
|---|---|---|
| `get_estimated_cost(input, output, model)` | `"~$0.0000"` string | `usage.calculate_cost` via `genai_prices` |
| `get_estimated_cost_from_usage(usage, model)` | string | `usage.calculate_cost_from_usage` |
| `get_estimated_cost_from_agent_run_output(run, model)` | string | `usage.calculate_cost_from_run_output` |
| `get_estimated_cost_from_agent(agent)` | string | `usage.calculate_cost_from_agent` |
| `get_price_id_total_cost(price_id)` | `dict | None` | `price_id_summary` accumulator (filled by `call_end` / `agent_end`). |

### 3.3 Cost calculation — `usage.py`

Single source of truth for token costing. Holds `MODEL_CONTEXT_WINDOWS`, a
giant dict mapping ~250 model names to their context windows (GPT-5.x, GPT-4.x,
o1/o3/o4, Claude 3/4, Gemini 1.x–3.x, Llama 3/4, Grok 3/4, Qwen, DeepSeek,
Mistral, Codestral, Gemma, Cohere Command, etc.).

| Function | Purpose |
|---|---|
| `get_model_name(model)` | Strip provider prefix from `"openai/gpt-4o-mini"` style strings or read `model.model_name`. |
| `normalize_model_name(name)` | Strip prefixes from `PROVIDER_PREFIXES` (`anthropic:`, `openai:`, `google-vertex:`, …). |
| `get_model_context_window(model)` | Lookup against `MODEL_CONTEXT_WINDOWS`. |
| `calculate_cost(input_tokens, output_tokens, model, cache_write_tokens, cache_read_tokens, reasoning_tokens)` | Wraps `genai_prices.calc_price` with `RequestUsage`, adds a separate reasoning-token charge for o-series. |
| `calculate_cost_from_usage(usage_or_dict, model)` | Coerces `RequestUsage` / `RunUsage` / `dict` and dispatches to `calculate_cost`. |
| `calculate_cost_from_run_output(run_output, model)` | Prefers `run_output.usage.cost` when present, else aggregates from `run_output.all_messages()`. |
| `calculate_cost_from_agent(agent)` | Tries `agent.get_session_usage()` first, then `agent.get_run_output()`. |
| `format_cost(cost, approximate=True)` | `~$0.012345` / `$0.0123` formatting. |
| `get_estimated_cost*` | String-returning convenience wrappers (the same ones re-exported from `printing.py`). |

### 3.4 Token / tool usage extraction

| File | Function | Used by |
|---|---|---|
| `llm_usage.py` | `llm_usage(model_response, cumulative=False)` — sum `input_tokens`/`output_tokens` over `new_messages()` (default) or `all_messages()` | `agent.agent.Agent`, pipeline cost steps |
| `tool_usage.py` | `tool_usage(model_response, task)` — pair `ToolCallPart`/`BaseToolReturnPart` by `tool_call_id`, also reads `model_response.tools` (`ToolExecution`) as a fallback, and calls `task.add_tool_call(...)` | `agent.agent.Agent` |

### 3.5 Async glue & timing

| File | Symbol | Purpose |
|---|---|---|
| `async_utils.py` | `run_async(coro)` | Run a coroutine even when an event loop is already active (uses a temporary `ThreadPoolExecutor`). Re-exported through `integrations/crawlee.py`. |
| `async_utils.py` | `AsyncExecutionMixin._run_async_from_sync(awaitable)` | Mixin so classes can call `await` from synchronous methods. |
| `timer.py` | `Timer` | `perf_counter` based context manager. Exposes `start/stop/elapsed/to_dict()`. |
| `dttm.py` | `current_datetime`, `current_datetime_utc`, `current_datetime_utc_str`, `now_epoch_s`, `to_epoch_s` | Datetime ↔ epoch normalisation, supports `int`, `float`, `datetime`, ISO-8601 strings (including `Z` suffix). |

### 3.6 Retry & error handling

#### `retry.py` — `@retryable`

A decorator that supports sync **and** async functions. Resolution order for
the retry count and mode:

1. Decorator args (`@retryable(retries=5, mode="raise")`).
2. If `retries_from_param="X"` is set: instance attribute `self.retry` first, else
   the function parameter named `X` (default fallback `0` for sync, `3` for async).
3. Instance attribute `self.retry` / `self.mode`.
4. Hard defaults (`0`/`3`, `"raise"`).

Behaviour:

- Catches every exception except `GuardrailValidationError` and
  `ExecutionTimeoutError`, which always bubble up immediately.
- Exponential backoff with `delay`, multiplied by `backoff` after each failure.
- On exhaustion: `mode="raise"` re-raises the last exception, `mode="return_false"`
  returns `False`.
- Logs to `printing.warning_log` between attempts and `printing.error_log`
  on exhaustion.

#### `error_wrapper.py` — `@upsonic_error_handler`

Wraps any function (sync or async) and converts third-party errors into
`UupsonicError` subclasses via `map_pydantic_error_to_upsonic`:

| Detected substring | Mapped exception | Code |
|---|---|---|
| `api key`, `unauthorized`, `401` | `NoAPIKeyException` | — |
| `connection`, `network`, `timeout`, `refused`, `unreachable` | `ModelConnectionError` | `CONNECTION_ERROR` |
| `rate limit`, `quota`, `billing`, `usage limit` | `ModelConnectionError` | `QUOTA_EXCEEDED` |
| `validation`, `invalid input`, `bad request`, `400` | `TaskProcessingError` | `VALIDATION_ERROR` |
| `configuration`, `config`, `setup`, `missing` | `ConfigurationError` | `CONFIG_ERROR` |
| `500`, `server error`, `internal error`, `service unavailable` | `ModelConnectionError` | `SERVER_ERROR` |
| `pydantic` (in name or message) | `AgentExecutionError` | `AGENT_ERROR` |
| anything else | `AgentExecutionError` | `UNKNOWN_ERROR` |

`_is_retryable_error` only treats `ModelConnectionError` with codes
`CONNECTION_ERROR` / `SERVER_ERROR` / `TIMEOUT_ERROR` as retryable. After
exhausting retries it raises `RetryExhaustedError` (or returns `None` when
`return_none_on_error=True`). `_display_error` calls `printing.error_message`
with an HTTP-like status code.

### 3.7 Pipeline helpers — `pipeline.py`

Simple lookups for the standard agent pipeline (`upsonic.agent.pipeline`). The
indices are documented in source:

```
0  InitializationStep
1  StorageConnectionStep
2  CacheCheckStep
3  UserPolicyStep
4  LLMManagerStep
5  ModelSelectionStep
6  ToolSetupStep
7  MemoryPrepareStep
8  SystemPromptBuildStep
9  ContextBuildStep
10 UserInputBuildStep
11 ChatHistoryStep              ← get_chat_history_step_index()
12 MessageAssemblyStep          ← get_message_assembly_step_index()
13 CallManagerSetupStep         ← get_call_manager_setup_step_index()
14 ModelExecutionStep           ← get_model_execution_step_index()
```

`find_step_index_by_name(steps, step_name)` walks a list looking for
`step.name == step_name`.

### 3.8 Validators & file helpers

| File | Symbol | Purpose |
|---|---|---|
| `validators.py` | `AttachmentFileNotFoundError` | Rich error with task description and remediation suggestions (absolute-path hint, permission hint, extension hint). |
| `validators.py` | `validate_attachments_exist(task)` | Raises if any attachment doesn't exist or isn't a regular file. |
| `validators.py` | `validate_attachments_readable(task)` | Above plus `os.access(path, os.R_OK)` and a 1024-byte read smoke test. |
| `validators.py` | `get_attachment_info(task)` | Per-attachment dict with `exists`/`is_file`/`readable`/`size`/`extension`. |
| `file_helpers.py` | `get_clean_extension(path)` | Lowercased extension without leading dot, e.g. `"video.MP4" → "mp4"`. |

### 3.9 Image helpers — `image.py`

| Function | Purpose |
|---|---|
| `extract_image_urls(text)` | Regex out `![alt](url)` markdown image URLs. |
| `urls_to_base64(urls)` | Download via `requests` (lazy import — calls `import_error` if missing) and base64-encode. |
| `save_base64_image(b64, name, ext)` | Decode and write. |
| `create_images_folder(path)` | `mkdir -p`. |
| `save_image_to_folder(data, folder, filename, is_base64)` | Bytes-or-base64 to disk. |
| `extract_and_save_images_from_response(text, folder, base_filename)` | Pipeline of the above three. |
| `open_image_file(path)` | macOS `open`, Windows `os.startfile`, Linux `xdg-open`. |
| `open_images_from_folder(folder, limit=None)` | Sorted by mtime descending. |
| `list_images_in_folder(folder)` | Same scan, no opening. |

### 3.10 Message helpers — `messages.py`

Convenience extractors over `upsonic.messages.messages.ModelResponse`:

- `get_text_content(response)` — content of last `TextPart`.
- `get_text_parts(response)`, `get_thinking_parts(response)`,
  `get_thinking_content(response)`, `get_tool_calls(response)`,
  `get_builtin_tool_calls(response)`, `get_tool_returns(response)`.
- `analyze_model_request_messages(messages)` returns
  `(message_details, total_parts)` for tracing (`has_system` flag).
- (De)serialisation: `serialize_model_request`, `deserialize_model_request`,
  `serialize_model_response`, `deserialize_model_response`,
  `serialize_messages`, `deserialize_messages` — all use lazily-built Pydantic
  `TypeAdapter`s with `ser_json_bytes='base64'`.

## 4. Subfolders walked through

### 4.1 `agent/events.py`

Around 1290 lines of helper coroutines / generators that wrap every
`upsonic.run.events.events.*` event class. Each event has a paired
`ayield_*_event` (`AsyncIterator[T]`) and `yield_*_event` (`Iterator[T]`)
function so streaming run-loops can `async for ... in ayield_x(...)` without
constructing event objects directly.

| Group | Events covered |
|---|---|
| **Pipeline** | `PipelineStartEvent`, `PipelineEndEvent` |
| **Step** | `StepStartEvent`, `StepEndEvent` |
| **Text** | `TextDeltaEvent`, `TextCompleteEvent` |
| **Tool** | `ToolCallEvent`, `ToolResultEvent`, `ExternalToolPauseEvent` |
| **Cache** | `CacheCheckEvent`, `CacheHitEvent`, `CacheMissEvent`, `CacheStoredEvent` |
| **Agent** | `AgentInitializedEvent` (currently kept but not exported) |
| **Model** | `ModelSelectedEvent`, `ModelRequestStartEvent`, `ModelResponseEvent` |
| **Tools/Messages** | `ToolsConfiguredEvent`, `MessagesBuiltEvent` |
| **Run** | `RunStartedEvent`, `RunCompletedEvent`, `RunCancelledEvent`, `RunPausedEvent` (last three kept but not exported) |
| **Memory** | `MemoryUpdateEvent`, `MemoryPreparedEvent`, `CultureUpdateEvent` |
| **Pipeline-internal** | `SystemPromptBuiltEvent`, `ContextBuiltEvent`, `UserInputBuiltEvent`, `ChatHistoryLoadedEvent`, `StorageConnectionEvent`, `LLMPreparedEvent` |
| **Policy** | `PolicyCheckEvent`, `PolicyFeedbackEvent` |
| **Reflection / Reliability** | `ReflectionEvent`, `ReliabilityEvent` |
| **Final** | `ExecutionCompleteEvent`, `FinalOutputEvent` |

The package `__init__.py` performs lazy `__getattr__` forwarding so
`from upsonic.utils.agent import yield_pipeline_start_event` works while keeping
import time low.

### 4.2 `integrations/`

Small per-vendor shims used by the matching `upsonic.tools.custom_tools.*`.

| File | Contents |
|---|---|
| `apify.py` | `create_apify_client(token)` with `Origin/upsonic` UA suffix; `actor_id_to_tool_name` (sanitises actor IDs for Python identifiers); `get_actor_latest_build`; `prune_actor_input_schema` (trims long descriptions to 350 chars); `_infer_array_item_type`; `props_to_json_schema` — converts pruned Apify props into a JSON-schema dict, handling proxy/`requestListSources`/array/object editor types and inferring item types from `prefill`/`default`. |
| `crawlee.py` | Re-exports `run_async` (the only thing the Crawlee tool needed). |
| `discord.py` | `sanitize_text_for_discord` — strip control chars, fall back to `"(empty response)"`. |
| `exa.py` | `serialize_exa_response` — recursive `model_dump()` traversal so JSON-encoding never falls through to `default=str`. |
| `firecrawl.py` | `serialize_firecrawl_response` — single-level `model_dump()` for firecrawl-py v4 BaseModel responses. |
| `gmail.py` | `@authenticate` decorator (re-runs OAuth if `self.creds` is invalid and rebuilds `self.service`); `extract_email_address` (handles `Display <user@x.com>` and bare addresses); `validate_email`; `encode_email_address` (RFC 2047 encoded headers for non-ASCII display names). |
| `telegram.py` | `sanitize_text_for_telegram` — `html.unescape`, strip tags, strip control chars. |
| `whatsapp.py` | Sync + async functions for the WhatsApp Cloud API: `get_access_token`, `get_phone_number_id`, `get_media`, `upload_media`, `send_image_message`, `typing_indicator`. Sync uses `requests`, async uses `httpx`. Uses `printing.debug_log`/`error_log` for tracing. |

`integrations/__init__.py` is essentially empty; importers reach files by full
path (e.g. `from upsonic.utils.integrations.gmail import authenticate`).

### 4.3 `package/`

Process-level / cross-version concerns.

| File | Symbol(s) | Purpose |
|---|---|---|
| `get_version.py` | `get_library_version()` | Tries `importlib.metadata.version("upsonic")`, falls back to reading `pyproject.toml` from package root or current working directory. Returns `"Version information not available."` when both fail. |
| `storage.py` | `Configuration` (classmethods only) | Persistent JSON-backed config in `~/.upsonic/config.json` with `_cache`. `get`, `set`, `delete`, `clear`, `all`. |
| `system_id.py` | `generate_system_id()`, `get_system_id()` | UUID4 created once and persisted into `Configuration` under the `system_id` key. Sentry uses this in `sentry_sdk.set_user`. |
| `exception.py` | All Upsonic exception classes | See table below. |

**Exception hierarchy (all live in `upsonic.utils.package.exception` and are
re-exported via `upsonic.exceptions`)**:

| Exception | Base | Notes |
|---|---|---|
| `VectorDBError` | `Exception` | Base for vector DB errors. |
| `VectorDBConnectionError`, `CollectionDoesNotExistError`, `SearchError`, `UpsertError` | `VectorDBError` | Storage-layer errors. |
| `GuardrailValidationError` | `Exception` | Bubbles through `@retryable`. |
| `NoAPIKeyException`, `UnsupportedLLMModelException`, `UnsupportedComputerUseModelException`, `ContextWindowTooSmallException`, `InvalidRequestException`, `CallErrorException`, `ServerStatusException`, `TimeoutException` | `Exception` | Generic provider/runtime errors. |
| `ToolError` | `Exception` | Carries `.message`. |
| `UupsonicError` | `Exception` | Common base with `error_code` and `original_error`. |
| `AgentExecutionError`, `ModelConnectionError`, `TaskProcessingError`, `ConfigurationError`, `RetryExhaustedError`, `ModelCapabilityError` | `UupsonicError` | Used by `error_wrapper.py`. |
| `AgentRunError` | `RuntimeError` | Base for run-time errors. |
| `UsageLimitExceeded`, `UnexpectedModelBehavior`, `ModelHTTPError`, `ModelAPIError` | `AgentRunError` | Provider-level errors with `body`/`status_code`/`model_name`. |
| `UserError` | `RuntimeError` | Developer mistakes. |
| `ModelRetry` | `Exception` | Tools raise this to ask the model to retry. Has a Pydantic core schema for serialisation. |
| `SkillError`, `SkillParseError`, `SkillDownloadError`, `SkillIntegrityError`, `SkillRegistryError`, `SkillValidationError` | `Exception` | Skills subsystem. `SkillValidationError` carries `errors: list[str]`. |

## 5. Cross-file relationships

```
                 ┌────────────────────────────────────────────────────────┐
                 │                  utils/__init__.py                     │
                 │   (lazy re-exports printing.* + usage.*)               │
                 └─────────────────┬──────────────────────────────────────┘
                                   │
       ┌───────────────────────────┼─────────────────────────────────────────────┐
       │                           │                                             │
       ▼                           ▼                                             ▼
  printing.py                 usage.py                                   logging_config.py
  (Rich panels,         (cost calculation,                          (Sentry / logging /
   bg-loggers,           MODEL_CONTEXT_WINDOWS,                      OpenTelemetry bootstrap)
   price_id_summary)     genai_prices wrapper)
       │                           │                                             │
       │ imports                   │ imports                                     │ imports
       ▼                           ▼                                             ▼
  retry.py ───► printing.warning/error_log               package/system_id ──► package/storage
  error_wrapper.py ──► printing.error_message            package/get_version ──► importlib.metadata
                                                                                
  llm_usage.py / tool_usage.py work on AgentRunOutput from the agent layer
  messages.py wraps upsonic.messages (ModelResponse / ModelRequest)
  validators.py works on Task objects (uses task.attachments)
  pipeline.py is consumed by upsonic.agent.pipeline.steps
  agent/events.py wraps upsonic.run.events.events.* event classes
  integrations/* are referenced by tools/custom_tools/*
```

Key invariants:

- **`printing.py` calls `setup_logging(enable_console=False)`** the moment it
  loads, so file/Sentry logging is wired up but Rich keeps console ownership.
- **`usage.py`** is the single source of cost math; `printing.py`, `chat.cost_calculator`,
  and `agent.agent.Agent.cost` all delegate to it.
- **`@retryable`** *must* let `GuardrailValidationError` and
  `ExecutionTimeoutError` bubble up — they signal hard failures (validator
  exhaustion, user-set timeout) that retries cannot fix.
- The lazy `__getattr__` pattern in `utils/__init__.py` and
  `utils/agent/__init__.py` exists to break circular imports with
  `agent.agent.Agent`.

## 6. Public API (consumed externally vs internally)

### 6.1 Imports surfaced via `upsonic.utils` itself

`utils/__init__.py` only exposes **cost surfaces and `AsyncExecutionMixin`**:

```python
from upsonic.utils import (
    AsyncExecutionMixin,
    print_price_id_summary, call_end,
    get_estimated_cost, get_estimated_cost_from_usage,
    get_estimated_cost_from_agent_run_output, get_estimated_cost_from_agent,
    calculate_cost, calculate_cost_from_usage,
    calculate_cost_from_run_output, calculate_cost_from_agent,
    get_model_name, get_model_context_window, format_cost,
)
```

### 6.2 Heavily reused from absolute paths (sample of ~145 importers)

| Import | Used by (representative) |
|---|---|
| `from upsonic.utils.printing import …` | `agent.agent`, `tasks.tasks`, `tools.processor`, `tools.mcp`, `chat.chat`, `embeddings.*`, `providers.*`, `tools.custom_tools.*`, `direct.py` |
| `from upsonic.utils.retry import retryable, RetryMode` | `agent.agent`, validators, RAG layer |
| `from upsonic.utils.error_wrapper import upsonic_error_handler` | `agent.agent`, providers |
| `from upsonic.utils.async_utils import run_async, AsyncExecutionMixin` | Sync wrappers across the codebase, including `tasks`, `tools`, `team`, `chat`, `integrations.crawlee` |
| `from upsonic.utils.timer import Timer` | Pipeline steps, agent / chat code |
| `from upsonic.utils.logging_config import setup_logging, get_logger, sentry_sdk, setup_opentelemetry` | Library bootstrap, integrations.tracing |
| `from upsonic.utils.llm_usage import llm_usage` | `agent.agent.Agent` |
| `from upsonic.utils.tool_usage import tool_usage` | `agent.agent.Agent` |
| `from upsonic.utils.usage import calculate_cost*, get_model_*` | `agent.agent.Agent`, `chat.cost_calculator`, providers |
| `from upsonic.utils.messages import …` | `agent.agent`, `messages.messages`, run-output handling |
| `from upsonic.utils.validators import validate_attachments_exist, validate_attachments_readable` | `tasks.tasks`, `agent.agent` |
| `from upsonic.utils.file_helpers import get_clean_extension` | `tasks.tasks`, attachment processing |
| `from upsonic.utils.image import extract_and_save_images_from_response, …` | Multimodal output handling |
| `from upsonic.utils.dttm import current_datetime_utc, now_epoch_s, to_epoch_s` | Storage providers, sessions |
| `from upsonic.utils.pipeline import get_chat_history_step_index, get_model_execution_step_index, …` | `agent.pipeline.steps` |
| `from upsonic.utils.agent import ayield_*_event, yield_*_event` | `agent.agent`, `agent.pipeline.steps` |
| `from upsonic.utils.package.exception import …` | Re-exported by `upsonic.exceptions`; touched everywhere errors are raised. |
| `from upsonic.utils.package.get_version import get_library_version` | `logging_config.py`, telemetry, CLI banners |
| `from upsonic.utils.package.system_id import get_system_id` | `logging_config.py` (Sentry user id) |
| `from upsonic.utils.package.storage import Configuration` | `system_id.py` and any module that wants a persistent on-disk flag |
| `from upsonic.utils.integrations.<vendor> import …` | The matching `tools.custom_tools.<vendor>` module |

### 6.3 Helpers that deliberately stay private

`_get_terminal_width`, `_format_pydantic_model`, `_get_model_name`,
`_format_cost_for_display`, `_StdoutProxy`, `_should_debug`,
`_is_retryable_error`, `_display_error`, `_get_status_code_from_error_code`,
`_get_retries_from_call`, `_lazy_import` — implementation details for the
public functions above.

## 7. Integration with the rest of Upsonic

- **Agent layer (`upsonic.agent.agent.Agent`)** — Uses `Timer`, `@retryable`,
  `@upsonic_error_handler`, `llm_usage`, `tool_usage`, `agent_started`,
  `agent_end`, `print_agent_metrics`, `pipeline_*`, the entire
  `utils/agent/events.py` event-yielder pack, and `usage.calculate_cost*` to
  populate `Agent.usage` (`AgentUsage`) and the per-call `task.usage`
  (`TaskUsage`). `setup_opentelemetry()` calls `Agent.instrument_all()`
  bidirectionally.
- **Pipeline (`upsonic.agent.pipeline.steps`)** — Uses
  `pipeline.get_*_step_index()` to find resumption points after external tool
  pause/resume; `find_step_index_by_name` for runtime inspection; the
  `pipeline_*` printing helpers and matching `(a)yield_step_*_event` /
  `(a)yield_pipeline_*_event`.
- **Tasks (`upsonic.tasks.tasks.Task`)** — Uses
  `validators.validate_attachments_*`, `file_helpers.get_clean_extension`,
  `print_price_id_summary`, `Timer`, and is the "task" arg threaded through
  `tool_usage(model_response, task)`.
- **Direct (`upsonic.direct.Direct`)** — Top-level synchronous facade. Uses
  `direct_started`, `direct_completed`, `direct_error`,
  `direct_metrics_summary`, `direct_configuration`, `Timer`, and the cost API.
- **Tool subsystem (`upsonic.tools.*`)** — `tools.processor`, `tools.mcp`,
  `tools.orchestration`, `tools.wrappers`, every `tools.custom_tools.*`
  use `printing.tool_operation`, `mcp_tool_operation`, `error_message`,
  `missing_dependencies`, `missing_api_key`, `import_error`, `tool_safety_check`
  and the matching `integrations/*` helpers.
- **Cache layer (`upsonic.cache`)** — Uses `cache_hit`, `cache_miss`,
  `cache_stored`, `cache_stats`, `cache_cleared`, `cache_configuration`
  panels.
- **Safety engine (`upsonic.safety_engine`)** — Calls `policy_triggered`,
  `policy_feedback_*`, `tool_safety_check`, `skill_safety_check`,
  `anonymization_debug_panel`.
- **Reflection / reliability (`upsonic.reflection`, `upsonic.reliability_layer`)**
  — Uses `reflection_*` panels and the matching `(a)yield_reflection_event` /
  `(a)yield_reliability_event` event helpers.
- **Storage / sessions (`upsonic.storage`, `upsonic.session`)** — Uses `dttm.*`
  for timestamps and `Configuration` for any local on-disk state.
- **Embedding & provider layers (`upsonic.embeddings.*`, `upsonic.providers.*`)**
  — Use `printing.import_error`, `missing_api_key`, `connection_info`, and the
  `error_wrapper.upsonic_error_handler` decorator.
- **Telemetry (`upsonic.integrations.tracing.DefaultTracingProvider`)** —
  Bootstrapped from `logging_config.setup_opentelemetry`.

## 8. End-to-end usage examples

### 8.1 Logging (file + Sentry, console handled by Rich)

```python
import os
os.environ["UPSONIC_LOG_LEVEL"] = "DEBUG"
os.environ["UPSONIC_LOG_FILE"] = "/tmp/upsonic.log"
os.environ["UPSONIC_TELEMETRY"] = "https://<your-dsn>@o…ingest.sentry.io/…"
os.environ["UPSONIC_ENVIRONMENT"] = "staging"

from upsonic.utils.logging_config import get_logger, set_module_log_level

logger = get_logger(__name__)
logger.debug("Initialised")

# Bring the noisy embeddings module down to WARNING
set_module_log_level("embeddings", "WARNING")
```

### 8.2 Rich-styled run output

```python
from upsonic.utils.printing import (
    info_log, warning_log, error_log, debug_log, debug_log_level2,
    call_end, agent_end, print_price_id_summary, print_agent_metrics,
)
from upsonic.utils.timer import Timer

info_log("Starting agent run", "Demo")

with Timer() as t:
    result = my_agent.do("Summarise this PDF.")
    usage = {"input_tokens": 1250, "output_tokens": 480}

call_end(
    result=result,
    model=my_agent.model,
    response_format=str,
    start_time=t.start_time,
    end_time=t.end_time,
    usage=usage,
    tool_usage=[],
    debug=False,
    price_id="demo-run",
    print_output=True,
)

# Aggregate cost summary tied to the price_id we passed above.
print_price_id_summary("demo-run", task=my_task, print_output=True)

# Or, if you have an Agent instance:
print_agent_metrics(my_agent)
```

### 8.3 Cost helpers without Rich output

```python
from upsonic.utils.usage import (
    calculate_cost, calculate_cost_from_usage,
    calculate_cost_from_run_output, format_cost,
    get_model_context_window,
)

# Direct token cost
cost = calculate_cost(input_tokens=1000, output_tokens=500, model="openai/gpt-4o-mini")
print(format_cost(cost, approximate=False))

# From a RunUsage / RequestUsage / dict:
cost = calculate_cost_from_usage(
    {"input_tokens": 800, "output_tokens": 200, "cache_read_tokens": 1500},
    model="anthropic/claude-sonnet-4-5",
)

# From an AgentRunOutput (preferred when it exists):
run_output = await agent.run("Hello")
cost = calculate_cost_from_run_output(run_output, model=agent.model)

print(get_model_context_window("openai/gpt-4.1"))   # → 1047576
```

### 8.4 Retry with `@retryable` (sync + async, instance-driven)

```python
from upsonic.utils.retry import retryable, RetryMode

class FlakyClient:
    def __init__(self, retry: int = 3, mode: RetryMode = "raise"):
        self.retry = retry
        self.mode = mode

    @retryable(retries_from_param="retry", delay=0.5, backoff=2.0)
    def fetch(self, url: str, retry: int | None = None) -> str:
        return self._http_get(url)  # may raise

    @retryable(retries_from_param="retry", delay=0.5, backoff=2.0)
    async def afetch(self, url: str, retry: int | None = None) -> str:
        return await self._aiohttp_get(url)
```

If the call raises `GuardrailValidationError` or `ExecutionTimeoutError` it
bypasses retries and bubbles up immediately. After exhaustion the decorator
either re-raises (`mode="raise"`) or returns `False` (`mode="return_false"`).

### 8.5 Wrapping arbitrary code with the Upsonic error mapper

```python
from upsonic.utils.error_wrapper import upsonic_error_handler

@upsonic_error_handler(max_retries=2, show_error_details=True)
async def call_provider(req):
    return await openai.chat.completions.create(**req)
```

A raw `RateLimitError` becomes `ModelConnectionError(error_code="QUOTA_EXCEEDED")`,
a `ConnectionError` becomes `ModelConnectionError(error_code="CONNECTION_ERROR")`,
both get auto-retried with exponential backoff, and on exhaustion you get
`RetryExhaustedError` with the original exception attached on `.original_error`.

### 8.6 Token / tool extraction from `AgentRunOutput`

```python
from upsonic.utils.llm_usage import llm_usage
from upsonic.utils.tool_usage import tool_usage

run_output = await agent.run("What's the weather in Paris?")
print(llm_usage(run_output))               # {'input_tokens': 1200, 'output_tokens': 240}
print(llm_usage(run_output, cumulative=True))  # session totals

print(tool_usage(run_output, task))        # also calls task.add_tool_call(...)
```

### 8.7 Validator + attachment helpers

```python
from upsonic.tasks.tasks import Task
from upsonic.utils.validators import (
    validate_attachments_exist, validate_attachments_readable, get_attachment_info,
    AttachmentFileNotFoundError,
)

task = Task("Summarise", attachments=["report.pdf", "chart.png"])

try:
    validate_attachments_exist(task)
    validate_attachments_readable(task)
except AttachmentFileNotFoundError as e:
    print(e)  # rich error w/ remediation suggestions

for info in get_attachment_info(task):
    print(info["path"], info["readable"], info["size"], info["extension"])
```

### 8.8 Async glue for sync wrappers

```python
from upsonic.utils.async_utils import run_async, AsyncExecutionMixin

class Tool(AsyncExecutionMixin):
    async def _afetch(self, q):
        return await some_async_api(q)

    def fetch(self, q):                       # sync facade
        return self._run_async_from_sync(self._afetch(q))

# Stand-alone usage:
data = run_async(some_async_api("hi"))         # works inside or outside an event loop
```

### 8.9 Image helpers in a multimodal pipeline

```python
from upsonic.utils.image import extract_and_save_images_from_response

response_text = agent.run("Generate three product banners").output
saved = extract_and_save_images_from_response(
    response_text=response_text,
    folder_path="./out/images",
    base_filename="banner",
)
print(saved)  # ['./out/images/banner_1.png', ...]
```

### 8.10 Datetime / epoch normalisation

```python
from upsonic.utils.dttm import to_epoch_s, current_datetime_utc

to_epoch_s("2026-01-15T08:30:00Z")       # → 1768624200
to_epoch_s(1768624200)                    # → 1768624200
to_epoch_s(current_datetime_utc())        # → now in seconds
```

### 8.11 Streaming events from a custom run-loop

```python
from upsonic.utils.agent import (
    ayield_pipeline_start_event, ayield_step_start_event,
    ayield_tool_call_event, ayield_tool_result_event,
    ayield_pipeline_end_event,
)

async def my_run(run_id: str):
    async for ev in ayield_pipeline_start_event(run_id, total_steps=3, task_description="Demo"):
        yield ev

    for step_idx, step_name in enumerate(["plan", "execute", "summarise"], start=0):
        async for ev in ayield_step_start_event(run_id, step_name, "...", step_idx, 3):
            yield ev

    async for ev in ayield_tool_call_event(run_id, "search", {"q": "anthropic"}):
        yield ev
    async for ev in ayield_tool_result_event(run_id, "search", {"q": "anthropic"}, "...result..."):
        yield ev

    async for ev in ayield_pipeline_end_event(run_id, status="completed",
                                              total_duration=1.42, total_steps=3,
                                              executed_steps=3):
        yield ev
```

### 8.12 Per-vendor integration helpers

```python
# Apify — convert pruned actor schema to JSON schema for an LLM tool.
from upsonic.utils.integrations.apify import (
    create_apify_client, get_actor_latest_build,
    prune_actor_input_schema, props_to_json_schema, actor_id_to_tool_name,
)

client = create_apify_client(token="…")
build = get_actor_latest_build(client, "apify/web-scraper")
props, required = prune_actor_input_schema(build["inputSchema"])
schema = props_to_json_schema(props, required)
tool_name = actor_id_to_tool_name("apify/web-scraper")  # "apify_actor_apify_web_scraper"

# Telegram / Discord — strip control chars before sendMessage.
from upsonic.utils.integrations.telegram import sanitize_text_for_telegram
from upsonic.utils.integrations.discord  import sanitize_text_for_discord

safe_t = sanitize_text_for_telegram("<b>hi</b>\x07world")
safe_d = sanitize_text_for_discord("\x01ok")

# Gmail OAuth wrapping.
from upsonic.utils.integrations.gmail import (
    authenticate, validate_email, encode_email_address, extract_email_address,
)

class GmailTool:
    def __init__(self):
        self.creds = None
        self.service = None

    @authenticate
    def list_messages(self):
        return self.service.users().messages().list(userId="me").execute()

# WhatsApp media + image messages (sync or async).
from upsonic.utils.integrations.whatsapp import (
    upload_media, send_image_message,
    upload_media_async, send_image_message_async,
)

mid = upload_media(image_bytes, mime_type="image/png", filename="banner.png")
send_image_message(media_id=mid, recipient="+15551234567", text="Here is your banner.")
```

### 8.13 Persistent on-disk configuration & system ID

```python
from upsonic.utils.package.storage   import Configuration
from upsonic.utils.package.system_id import get_system_id
from upsonic.utils.package.get_version import get_library_version

Configuration.set("last_run", "2026-04-28T08:00:00Z")
Configuration.get("last_run")            # → '2026-04-28T08:00:00Z'
Configuration.all()                      # → {'system_id': '…', 'last_run': '…'}

print(get_system_id())                    # stable UUID4 across runs
print(get_library_version())              # e.g. '0.75.0'
```

### 8.14 Pipeline step lookup

```python
from upsonic.utils.pipeline import (
    find_step_index_by_name,
    get_chat_history_step_index, get_message_assembly_step_index,
    get_call_manager_setup_step_index, get_model_execution_step_index,
)

resume_idx = get_model_execution_step_index()      # 14 — external-tool resumption
hist_idx   = get_chat_history_step_index()         # 11
custom_idx = find_step_index_by_name(steps, "ContextBuildStep")
```

### 8.15 Message extraction & (de)serialisation

```python
from upsonic.utils.messages import (
    get_text_content, get_tool_calls, get_thinking_content,
    serialize_messages, deserialize_messages,
)

text  = get_text_content(response)              # last TextPart content
calls = get_tool_calls(response)                # list[ToolCallPart]
think = get_thinking_content(response)          # last ThinkingPart content

raw = serialize_messages(history)
restored = deserialize_messages(raw)            # round-trips through base64-encoded bytes
```

### 8.16 Custom exceptions

```python
from upsonic.utils.package.exception import (
    UupsonicError, NoAPIKeyException, ModelConnectionError,
    ModelRetry, GuardrailValidationError, ModelCapabilityError,
)

try:
    raise ModelCapabilityError(
        model_name="gpt-4o-mini",
        attachment_path="movie.mp4",
        attachment_extension="mp4",
        required_capability="video",
        supported_extensions=[],
    )
except UupsonicError as e:
    print(e.error_code)   # MODEL_CAPABILITY_MISMATCH
    print(e)              # Rich human-readable message

# A tool function asks the model to retry with feedback:
def lookup(name: str):
    if not name:
        raise ModelRetry("`name` is empty — please provide a non-empty value.")
    return _db.get(name)
```

---

The `utils/` package is therefore best understood as Upsonic's **observability +
plumbing kernel**: every cost number you see in a printed panel, every retry
banner, every tool-call card, every Sentry breadcrumb, every event streamed by
the agent run-loop, and every persisted system identifier flows through the
files documented above.
