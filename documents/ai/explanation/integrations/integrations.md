---
name: observability-tracing-integrations
description: Use when wiring Upsonic agents to external observability, tracing, prompt-registry, or governance backends, or debugging span export, REST logging, and audit signing flows. Use when a user asks to enable OpenTelemetry, send traces to Langfuse/Jaeger/Datadog/Honeycomb, log runs to PromptLayer, sign spans with Asqav, configure samplers/exporters, or build a custom TracingProvider subclass. Trigger when the user mentions TracingProvider, DefaultTracingProvider, Langfuse, PromptLayer, AsqavGovernance, OTLP, OpenTelemetry, BatchSpanProcessor, BaggageSpanProcessor, InstrumentationSettings, instrument=, promptlayer=, OTLPSpanExporter, gen_ai semantic conventions, ML-DSA-65 signing, score configs, annotation queues, dataset runs, /log-request, /api/public/otel/v1/traces, UPSONIC_OTEL_ENDPOINT, LANGFUSE_PUBLIC_KEY, PROMPTLAYER_API_KEY, ASQAV_API_KEY, or src/upsonic/integrations.
---

# `src/upsonic/integrations/` — Third-Party Observability, Tracing & Governance

## 1. What this folder is

This folder contains every adapter that bridges Upsonic agents to **external SaaS / OSS observability, tracing, governance and prompt-management platforms**. It is the single seam where Upsonic emits OpenTelemetry spans, prompt-registry lookups, evaluation/dataset uploads and cryptographic audit signatures.

Concretely, four backends are wired up here:

| File | Backend | Purpose |
|---|---|---|
| `tracing.py` | OTLP / OpenTelemetry (any compatible collector — Jaeger, Tempo, Datadog Agent, Honeycomb, etc.) | Abstract base class **`TracingProvider`** + concrete **`DefaultTracingProvider`** that boots a full OTel pipeline (TracerProvider, MeterProvider, BatchSpanProcessor, BaggageSpanProcessor, sampler). |
| `langfuse.py` | [Langfuse](https://langfuse.com/) cloud (EU/US) or self-hosted | OTel HTTP/protobuf exporter aimed at `/api/public/otel/v1/traces` **plus** a complete REST client for Langfuse Scores, Score Configs, Annotation Queues, Traces, Datasets, Dataset Items, Dataset Runs and Dataset Run Items. |
| `promptlayer.py` | [PromptLayer](https://promptlayer.com/) | Versioned prompt registry, request log ingestion (`/log-request`), workflow (agent) registry, dataset-groups, evaluation reports/scoring. **Not** an OTel exporter — uses its own bespoke HTTP API. |
| `asqav.py` | [Asqav](https://asqav.com/) | Quantum-safe (ML-DSA-65) cryptographic signing of every span. Wraps an `InMemorySpanExporter` with `_AsqavSigningExporter` that classifies spans via OTel GenAI semantic conventions and hands them to the asqav SDK. |
| `__init__.py` | — | Lazy re-exports (`__getattr__`) so `from upsonic import Langfuse, PromptLayer, AsqavGovernance, DefaultTracingProvider, TracingProvider` works without paying import cost on cold start. |

All three TracingProvider subclasses (`DefaultTracingProvider`, `Langfuse`, `AsqavGovernance`) plug into `Agent(instrument=...)`. PromptLayer plugs in via the orthogonal `Agent(promptlayer=...)` parameter and is also accepted by every evaluator (`AccuracyEvaluator`, `ReliabilityEvaluator`, `PerformanceEvaluator`).

## 2. Folder layout

```
src/upsonic/integrations/
├── __init__.py            # Lazy re-exports (PEP 562 module __getattr__)
├── tracing.py             # TracingProvider ABC + DefaultTracingProvider (OTLP)
├── langfuse.py            # Langfuse(TracingProvider) + REST API client
├── promptlayer.py         # PromptLayer (standalone REST client, not OTel)
└── asqav.py               # AsqavGovernance(TracingProvider) + _AsqavSigningExporter
```

There are **no subfolders**.

## 3. Per-integration walkthrough

### 3.1 `__init__.py` — lazy re-exports

Five symbols are re-exported through PEP 562 lazy `__getattr__` so importing `upsonic` doesn't drag in `httpx`, `opentelemetry-sdk`, `asqav`, etc., until they're actually requested:

```python
__all__ = ["TracingProvider", "DefaultTracingProvider",
           "Langfuse", "PromptLayer", "AsqavGovernance"]

def __getattr__(name):
    if name == "TracingProvider":
        from upsonic.integrations.tracing import TracingProvider
        return TracingProvider
    if name == "DefaultTracingProvider":
        from upsonic.integrations.tracing import DefaultTracingProvider
        return DefaultTracingProvider
    if name == "Langfuse":
        from upsonic.integrations.langfuse import Langfuse
        return Langfuse
    if name == "PromptLayer":
        from upsonic.integrations.promptlayer import PromptLayer
        return PromptLayer
    if name == "AsqavGovernance":
        from upsonic.integrations.asqav import AsqavGovernance
        return AsqavGovernance
    raise AttributeError(...)
```

`TYPE_CHECKING` aliases keep static type-checkers happy without runtime imports.

### 3.2 `tracing.py` — `TracingProvider` (ABC) and `DefaultTracingProvider`

Two classes, ~480 lines, with three responsibilities:

1. **Bootstrap a full OpenTelemetry pipeline** once per provider instance.
2. **Provide an HTTP client surface** (`_get/_post/_patch/_delete/_post async/_get async/...`) so subclasses can talk to their backend's REST API without duplicating httpx wiring.
3. **Manage lifecycle**: `shutdown()`, `ashutdown()`, `flush()`, `atexit` registration, idempotent shutdown via `_shutdown_called`.

#### `TracingProvider.__init__`

| Constructor kwarg | Default | Effect |
|---|---|---|
| `service_name` | `"upsonic"` | Becomes `service.name` on the OTel `Resource`. |
| `sample_rate` | `1.0` | Clamped to `[0.0, 1.0]`. Applied via `TraceIdRatioBasedSampler` (or its older `TraceIdRatioBased` alias on legacy SDKs). |
| `include_content` | `True` | Forwarded to `InstrumentationSettings`; controls whether prompt/response bodies appear in spans. |
| `use_aggregated_usage_attribute_names` | `False` | Switches token-usage attributes to the `gen_ai.aggregated_usage.*` namespace on root spans (avoids double-counting in Langfuse). |
| `flush_on_exit` | `True` | Registers `atexit.register(self.shutdown)`. |

The constructor does **two** non-obvious things that are easy to miss:

- It defends against subclasses that set `_api_base_url` / `_client` / `_async_client` *before* calling `super().__init__()` (Langfuse does exactly this).
- It immediately calls `self._setup()` which calls `self._create_exporter()`. This means subclasses must initialise everything the exporter needs **before** `super().__init__()` runs. AsqavGovernance leans heavily on this fact and calls `_init_asqav()` first.

#### `_setup()` — the one place OTel is wired up

```python
def _setup(self) -> Tuple[TracerProvider, MeterProvider, InstrumentationSettings]:
    resource = self._create_resource()             # service.name=...
    sampler  = self._create_sampler()              # None when sample_rate>=1.0
    tracer_provider = TracerProvider(resource=resource, **({"sampler": sampler} if sampler else {}))

    exporter = self._create_exporter()             # subclass hook
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    # BaggageSpanProcessor copies OTel Baggage entries onto every span,
    # so trace-level attrs (user.id, session.id, langfuse.trace.name, ...)
    # appear on ALL child spans — Langfuse needs this.
    try:
        from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS
        tracer_provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
    except ImportError:
        pass

    meter_provider = MeterProvider(resource=resource)

    from upsonic.models.instrumented import InstrumentationSettings
    settings = InstrumentationSettings(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        include_content=self._include_content,
        use_aggregated_usage_attribute_names=self._use_aggregated,
    )
    return tracer_provider, meter_provider, settings
```

The returned `InstrumentationSettings` is what `Agent` actually consumes via `.settings`.

#### Public properties

| Property | Returns | Used by |
|---|---|---|
| `settings` | `InstrumentationSettings` | `Agent._resolve_instrumentation` |
| `tracer_provider` | `TracerProvider` | (rare; debugging) |
| `meter_provider` | `MeterProvider` | (rare; metrics access) |

Plus a fallback `__getattr__(name)` that proxies any non-underscore attribute to `self._settings`, so `provider.tracer`, `provider.meter`, `provider.include_content` all transparently work.

#### HTTP plumbing (mixin-style, used only by REST-aware subclasses)

| Method | Behaviour |
|---|---|
| `_api_headers()` | Default returns `{}`. Overridden by subclasses to supply Bearer/Basic auth headers. |
| `_get_client()` / `_get_async_client()` | Lazy `httpx.Client` / `httpx.AsyncClient` with `base_url=_api_base_url`, `timeout=30.0`. |
| `_post`, `_get`, `_patch`, `_delete`, `_delete_with_body` | Sync wrappers. |
| `_apost`, `_aget`, `_apatch`, `_adelete`, `_adelete_with_body` | Async wrappers. |
| `_raise_api_error()` | Static helper that decodes JSON error body (falling back to text) and re-raises as `httpx.HTTPStatusError` with the body in the message. |

#### Lifecycle

- `shutdown()` — flushes spans, shuts down `tracer_provider` + `meter_provider`, closes sync httpx client. Idempotent via `_shutdown_called`.
- `ashutdown()` — calls `shutdown()` then `await self._async_client.aclose()`.
- `flush()` — `tracer_provider.force_flush()`. Used by `AgentOTelManager.flush()` after each agent run so Langfuse/Jaeger see spans immediately rather than waiting for `BatchSpanProcessor`'s scheduled tick.

#### `DefaultTracingProvider`

The "no specific backend" provider. Reads `UPSONIC_OTEL_*` env vars when arguments are omitted:

| Env var | Constructor arg | Default |
|---|---|---|
| `UPSONIC_OTEL_ENDPOINT` | `endpoint` | `http://localhost:4317` |
| `UPSONIC_OTEL_HEADERS` | `headers` | `""` (parsed as `key=value,key2=value2`) |
| `UPSONIC_OTEL_SERVICE_NAME` | `service_name` | `"upsonic"` |
| `UPSONIC_OTEL_SAMPLE_RATE` | `sample_rate` | `1.0` |

`_create_exporter()` performs a **3-tier exporter fallback**:

```python
# 1. gRPC exporter (preferred)
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    if "insecure" not in kw:
        kw["insecure"] = not endpoint.startswith("https")
    return OTLPSpanExporter(**kw)
except ImportError:
    pass

# 2. HTTP/protobuf exporter — auto-rewrites :4317 → :4318
try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as Http
    if endpoint.endswith(":4317"):
        kw["endpoint"] = endpoint.replace(":4317", ":4318")
    return Http(**kw)
except ImportError:
    pass

# 3. ConsoleSpanExporter — last resort, prints to stderr
return ConsoleSpanExporter()
```

`_parse_headers("foo=bar,baz=qux")` → `{"foo": "bar", "baz": "qux"}`; ignores malformed pairs.

### 3.3 `langfuse.py` — `Langfuse(TracingProvider)`

A subclass of `TracingProvider` that ships **both** an OTel exporter *and* a thick REST client. ~1600 lines.

#### Constructor

| Kwarg | Env fallback | Notes |
|---|---|---|
| `public_key` | `LANGFUSE_PUBLIC_KEY` | `pk-lf-...` — required, raises `ValueError` if missing. |
| `secret_key` | `LANGFUSE_SECRET_KEY` | `sk-lf-...` — required. |
| `host` | `LANGFUSE_HOST` | Strips trailing `/`. |
| `region` | — | `"eu"` or `"us"`. Selects `https://cloud.langfuse.com` vs `https://us.cloud.langfuse.com`. Ignored if `host` is given. |
| `include_content`, `service_name`, `sample_rate`, `flush_on_exit`, `use_aggregated_usage_attribute_names` | inherited | Same as parent. |

Endpoint computation:

```python
self._endpoint    = f"{self._host}/api/public/otel/v1/traces"
self._auth_header = f"Basic {b64(public_key:secret_key)}"
self._api_base_url = self._host    # consumed by parent's _get_client()
```

#### `_create_exporter()`

Forces the **HTTP/protobuf** OTLP exporter (Langfuse does not speak OTLP gRPC):

```python
return OTLPSpanExporter(
    endpoint=self._endpoint,
    headers={"Authorization": self._auth_header},
)
```

If `opentelemetry-exporter-otlp-proto-http` is not installed, raises `ImportError` with an actionable message.

#### `_api_headers()` override

Adds `Authorization: Basic <b64>` and `Content-Type: application/json` so every REST call inherits Basic-Auth.

#### REST surface — full table

| Group | Sync method | Async variant | HTTP | Path |
|---|---|---|---|---|
| **Scores** | `score(...)` | `ascore` | POST | `/api/public/scores` |
| | `get_scores(...)` | `aget_scores` | GET | `/api/public/v2/scores` |
| | `delete_score(score_id)` | `adelete_score` | DELETE | `/api/public/scores/{id}` |
| **Score Configs** | `create_score_config(...)` | `acreate_score_config` | POST | `/api/public/score-configs` |
| | `get_score_configs(...)` | `aget_score_configs` | GET | `/api/public/score-configs` |
| | `get_score_config(id)` | `aget_score_config` | GET | `/api/public/score-configs/{id}` |
| | `update_score_config(...)` | `aupdate_score_config` | PATCH | `/api/public/score-configs/{id}` |
| **Annotation Queues** | `create_annotation_queue(...)` | `acreate_annotation_queue` | POST | `/api/public/annotation-queues` |
| | `get_annotation_queues(...)` | `aget_annotation_queues` | GET | `/api/public/annotation-queues` |
| | `get_annotation_queue(id)` | `aget_annotation_queue` | GET | `/api/public/annotation-queues/{id}` |
| | `delete_annotation_queue(id)` | `adelete_annotation_queue` | (loop of DELETEs) | clears items page-by-page |
| **Queue Items** | `create_annotation_queue_item(...)` | `acreate_annotation_queue_item` | POST | `/api/public/annotation-queues/{q}/items` |
| | `get_annotation_queue_items(...)` | `aget_annotation_queue_items` | GET | `/api/public/annotation-queues/{q}/items` |
| | `get_annotation_queue_item(q,i)` | `aget_annotation_queue_item` | GET | `/api/public/annotation-queues/{q}/items/{i}` |
| | `update_annotation_queue_item(...)` | `aupdate_annotation_queue_item` | PATCH | `/api/public/annotation-queues/{q}/items/{i}` |
| | `delete_annotation_queue_item(q,i)` | `adelete_annotation_queue_item` | DELETE | `/api/public/annotation-queues/{q}/items/{i}` |
| **Queue Assignments** | `create_annotation_queue_assignment(q,u)` | async variant | POST | `/api/public/annotation-queues/{q}/assignments` |
| | `delete_annotation_queue_assignment(q,u)` | async variant | DELETE w/body | `/api/public/annotation-queues/{q}/assignments` |
| **Traces** | `update_trace(...)` | `aupdate_trace` | POST | `/api/public/ingestion` (batch with `trace-create` event) |
| **Datasets** | `create_dataset(...)` | `acreate_dataset` | POST | `/api/public/v2/datasets` |
| | `get_datasets(...)` | `aget_datasets` | GET | `/api/public/v2/datasets` |
| | `get_dataset(name)` | `aget_dataset` | GET | `/api/public/v2/datasets/{name}` |
| **Dataset Items** | `create_dataset_item(...)` | `acreate_dataset_item` | POST | `/api/public/dataset-items` |
| | `get_dataset_items(...)` | `aget_dataset_items` | GET | `/api/public/dataset-items` |
| | `get_dataset_item(id)` | `aget_dataset_item` | GET | `/api/public/dataset-items/{id}` |
| | `delete_dataset_item(id)` | `adelete_dataset_item` | DELETE | `/api/public/dataset-items/{id}` |
| **Dataset Run Items** | `create_dataset_run_item(...)` | `acreate_dataset_run_item` | POST | `/api/public/dataset-run-items` |
| **Dataset Runs** | `get_dataset_runs(name)` | `aget_dataset_runs` | GET | `/api/public/datasets/{name}/runs` |
| | `get_dataset_run(name,run)` | `aget_dataset_run` | GET | `/api/public/datasets/{name}/runs/{run}` |
| | `delete_dataset_run(name,run)` | `adelete_dataset_run` | DELETE | `/api/public/datasets/{name}/runs/{run}` |

#### Notable design choices

- **`update_trace`** does *not* hit a `/traces/{id}` PATCH endpoint — Langfuse exposes no such thing. Instead, it submits a **`trace-create` ingestion event** with the trace's existing `id` and a timestamp 5 seconds in the future, exploiting Langfuse's "last-write-wins by timestamp" upsert semantics:

  ```python
  event_ts = utcnow() + timedelta(seconds=5)
  batch = {"batch": [{"id": uuid4(), "type": "trace-create",
                      "timestamp": event_ts.iso(), "body": {"id": trace_id, ...}}]}
  return self._post("/api/public/ingestion", batch)
  ```

- **`delete_annotation_queue`** simulates a non-existent endpoint by paging all items (50 at a time) and deleting them one by one, returning `{"success": True, "items_removed": <int>}`.

- **`_build_score_body`** is a static helper that constructs the score POST body with conditional inclusion of every optional field (`observationId`, `sessionId`, `dataType`, `comment`, `id`, `configId`, `metadata`, `environment`, `queueId`, `datasetRunId`).

- Both `score()` and `get_scores()` accept all 16 Langfuse v2 filter parameters (`source`, `dataType`, `operator`, `value`, `traceTags`, etc.).

- Score `data_type` may be omitted — Langfuse infers it from the value's type. When given, it is `"NUMERIC" | "CATEGORICAL" | "BOOLEAN"`.

### 3.4 `promptlayer.py` — `PromptLayer` (standalone, **not** a TracingProvider)

A ~1500-line REST client. **Does not** subclass `TracingProvider` and emits no OTel spans — it has its own thread-pool-style background logging model and a wholly different API shape.

#### Constructor

| Kwarg | Env fallback | Default |
|---|---|---|
| `api_key` | `PROMPTLAYER_API_KEY` | required (`pl_...`) |
| `base_url` | `PROMPTLAYER_BASE_URL` | `https://api.promptlayer.com` |

State carried per instance:

| Attribute | Purpose |
|---|---|
| `_client`, `_async_client` | Lazy `httpx.Client` / `httpx.AsyncClient`. |
| `_last_prompt_name`, `_last_prompt_id`, `_last_prompt_version` | Captured by `get_prompt(...)` so `Agent._log_to_promptlayer_unified` can attach them to the `/log-request` body without callers passing them explicitly. |
| `_created_workflows: Dict[str, int]` | Cache of `workflow_name → workflow_id` to avoid double-creating the agent's workflow on subsequent runs. |
| `_pending_threads`, `_threads_lock` | Bookkeeping for fire-and-forget background `log()` calls — drained at `shutdown()`. |

#### Helpers

- `_stringify_metadata({k: v})` — `dict`/`list`/`tuple` → `json.dumps(..., default=str)`, primitives → `str(v)`. Required because PromptLayer's metadata values must be strings.
- `_parse_provider_model(name)` — splits `"openai/gpt-4o"` → `("openai", "gpt-4o")`. Handles `"accuracy_eval:anthropic/claude-sonnet-4-6"` (drops the `accuracy_eval:` prefix) and plain `"reliability_eval"` → `("custom", "reliability_eval")`.
- `_epoch_to_iso(epoch)` — ISO-8601 timestamps used in `request_start_time` / `request_end_time`.

#### HTTP plumbing

Every request adds `X-API-KEY: <api_key>`. Failed responses are logged via the Upsonic logger then `raise_for_status()`'d. Sync and async variants exist for `_get`, `_post`, `_patch`, `_delete`, plus a dedicated `_log_post` / `_alog_post` for `/log-request` (skips the warning log).

#### REST surface

##### Prompt Registry

| Method | Async | Path | Notes |
|---|---|---|---|
| `get_prompt(name, version=, label=, variables=, return_metadata=False)` | `aget_prompt` | `POST /prompt-templates/{name}` | Sets `_last_prompt_*`. Returns the rendered string or `(string, {id, version, label})`. |

`_extract_prompt_text(result)` traverses the response to flatten:

1. `prompt_template.messages[*].content` (chat-style; `content` may itself be a list of `{type: "text"}` blocks).
2. `prompt_template.content[*]` (single-message form).
3. `prompt_template.template` (legacy single-string form).
4. Final fallback: `str(prompt_template)`.

##### Unified logging

The crown jewel: `log(...)` / `alog(...)` post to `POST /log-request` and return the integer `request_id`. They are the **single entry point** for everything PromptLayer-loggable — agent runs, accuracy evals, reliability evals, performance evals.

```python
def log(*, provider, model, input_text, output_text,
        start_time=None, end_time=None,
        input_tokens=0, output_tokens=0, price=0.0,
        parameters=None, tags=None, metadata=None,
        score=None, status="SUCCESS",
        function_name=None,
        prompt_name=None, prompt_id=None, prompt_version=None,
        scores=None,                  # Dict[str, int] — extra named scores
        system_prompt=None,
        tools=None, tool_calls=None, tool_results=None) -> int:
    body = self._build_log_body(...)  # see below
    request_id = self._log_post(body)["id"]
    if scores and request_id:
        for n, v in scores.items():
            self.score(request_id, v, name=n)
    return request_id
```

`_build_log_body(...)` constructs the chat-style payload PromptLayer expects:

- **input messages**: `[system?, user]` — each `content: [{type: "text", text: ...}]`.
- **output messages**: branches on `tool_calls`/`tool_results`:
  - With both: `[assistant(tool_calls), tool, tool, …, assistant(final text)]` so the timeline matches what actually happened.
  - Tool calls only: single assistant message with both `content` and `tool_calls`.
  - No tools: single assistant message with `content`.
- Adds `parameters`, `tags`, `metadata` (stringified), `score` (clamped to `[0, 100]` int), `prompt_name/id/version_number`, `function_name`.

##### Post-hoc score / metadata

| Method | Async | Path | Purpose |
|---|---|---|---|
| `score(request_id, score, name="quality")` | `ascore` | POST `/rest/track-score` | Adds a named numeric score (rounded + clamped to `0..100`). |
| `add_metadata(request_id, metadata)` | `aadd_metadata` | POST `/rest/track-metadata` | Stringifies + attaches metadata to an already-logged request. |

Both swallow exceptions and return `False` on failure, since they're called from background threads.

##### Workflows (Agents)

Used by `Agent._create_promptlayer_workflow()` to register the Upsonic agent as a PromptLayer "workflow" so the dashboard shows the topology of prompts and tools.

| Method | Async | Path | Behaviour |
|---|---|---|---|
| `list_workflows(page=1, per_page=30)` | `alist_workflows` | GET `/workflows` | Paginated list. |
| `create_workflow(nodes, name=, workflow_id=, …)` | `acreate_workflow` | POST `/rest/workflows` | Either creates (give `name`) or versions an existing one (give `workflow_id`/`workflow_name`). |
| `patch_workflow(id_or_name, base_version=, nodes=, …)` | `apatch_workflow` | PATCH `/rest/workflows/{id_or_name}` | Per-node update keyed by name; passing `None` for a node removes it. |

##### Datasets

| Method | Async | Path |
|---|---|---|
| `create_dataset_group(name, workspace_id=)` | `acreate_dataset_group` | POST `/api/public/v2/dataset-groups` |
| `list_datasets(...)` | `alist_datasets` | GET `/api/public/v2/datasets` |
| `create_dataset_version_from_file(group_id, file_name, file_content_base64)` | async variant | POST `/api/public/v2/dataset-versions/from-file` |
| `create_dataset_version_from_filter(group_id, …)` | async variant | POST `/api/public/v2/dataset-versions/from-filter-params` |

##### Reports / Evaluations

| Method | Async | Path | Purpose |
|---|---|---|---|
| `create_report(group_id, …)` | `acreate_report` | POST `/reports` | Build an evaluation pipeline. |
| `get_report(id)` | `aget_report` | GET `/reports/{id}` | Fetch the pipeline definition + status + stats. |
| `get_report_score(id)` | `aget_report_score` | GET `/reports/{id}/score` | Get aggregated overall score. |
| `add_report_column(id, type, name, configuration, position=)` | `aadd_report_column` | POST `/report-columns` | Append a new column (e.g. `LLM_ASSERTION`, `CODE_EXECUTION`). |
| `update_report_score_card(id, column_names, code=, code_language=)` | async variant | POST `/reports/{id}/score-card` | Custom Python/JavaScript scoring code. |
| `run_report(id, name, dataset_id=, refresh_dataset=)` | `arun_report` | POST `/reports/{id}/run` | Execute the pipeline. |
| `delete_report_by_name(name)` | `adelete_report_by_name` | DELETE `/reports/name/{name}` | Archive all reports with that name. |
| `list_evaluations(name=, status=, page=, per_page=)` | `alist_evaluations` | GET `/api/public/v2/evaluations` | List existing evals. |

#### Lifecycle

```python
def shutdown(self) -> None:
    self._drain_threads()       # joins every fire-and-forget log thread
    if self._client is not None:
        self._client.close(); self._client = None
    self._async_client = None   # async client lives in subthread event loops
```

`_register_thread` / `_drain_threads` are how `Agent._log_to_promptlayer_background` keeps logging non-blocking yet still flushes cleanly when the user calls `pl.shutdown()`.

### 3.5 `asqav.py` — `AsqavGovernance(TracingProvider)`

A 315-line OTel-style provider that does **not** export to a backend at all. Its `SpanExporter` is an `InMemorySpanExporter` wrapped in `_AsqavSigningExporter`, which signs every relevant span by calling `agent.sign(action_type=..., context=...)` on the asqav SDK before discarding it.

#### Constructor ordering — important

```python
def __init__(self, *, api_key=None, agent_name="upsonic-agent", endpoint=None,
             sign_tool_calls=True, sign_llm_calls=True, sign_agent_steps=True,
             include_content=True, service_name="upsonic"):
    self._api_key   = api_key  or os.environ.get("ASQAV_API_KEY")  or None
    self._agent_name = agent_name
    self._endpoint   = endpoint or os.environ.get("ASQAV_API_URL") or None
    self._sign_tool_calls  = sign_tool_calls
    self._sign_llm_calls   = sign_llm_calls
    self._sign_agent_steps = sign_agent_steps

    # asqav MUST be initialized BEFORE super().__init__() because the
    # parent's __init__ calls _setup() → _create_exporter() which needs
    # self._agent to already exist.
    self._init_asqav()
    super().__init__(service_name=service_name, include_content=include_content)
```

`_init_asqav()` does:

1. `import asqav` (raises `ImportError` with `pip install 'upsonic[asqav]'` hint if missing).
2. `asqav.init(api_key=..., base_url=...)` (passes `None` to keep asqav's own correct default `https://api.asqav.com/api/v1` — the docstring warns against hardcoding `https://api.asqav.com` because that strips the `/api/v1` suffix).
3. `self._agent = asqav.Agent.create(self._agent_name)` — idempotent: asqav returns the existing agent for the same name.
4. `self._session = self._agent.start_session()`.

#### `_create_exporter()`

```python
return _AsqavSigningExporter(
    inner=InMemorySpanExporter(),
    get_agent=lambda: self._agent,    # late-binding callable
    sign_tool_calls=self._sign_tool_calls,
    sign_llm_calls=self._sign_llm_calls,
    sign_agent_steps=self._sign_agent_steps,
)
```

The `get_agent` is a **callable**, not a captured value, so the signer always sees the *live* `self._agent` regardless of init order quirks.

#### `_AsqavSigningExporter` internals

Span classification uses **OTel GenAI semantic-convention attribute keys**, never substring matching, so internal Upsonic spans like `pipeline.step.tool_setup` are never misclassified as a real tool call:

```python
_LLM_ATTR_KEYS  = ("gen_ai.system", "gen_ai.request.model", "gen_ai.operation.name")
_TOOL_ATTR_KEYS = ("gen_ai.tool.name", "gen_ai.tool.call.id")

def _classify(self, name, attrs):
    if any(k in attrs for k in self._TOOL_ATTR_KEYS) or any(str(k).startswith("gen_ai.tool.") for k in attrs):
        return None if not self._sign_tool_calls else ("tool", str(attrs.get("gen_ai.tool.name", name)))
    if any(k in attrs for k in self._LLM_ATTR_KEYS):
        return None if not self._sign_llm_calls else ("llm", name)
    return None if not self._sign_agent_steps else ("agent", name)
```

For each span that matches:

```python
agent.sign(
    action_type=f"{kind}:{action_name}",
    context={k: str(v) for k, v in attrs.items()},
)
```

Failures are caught and logged via `_logger.warning(...)` — never raised, because tracing must never break the agent.

`shutdown()` and `force_flush(timeout_millis=30000)` proxy through to the inner `InMemorySpanExporter`.

#### Public methods on `AsqavGovernance`

| Method | Returns | Notes |
|---|---|---|
| `export_audit_json(agent_id=None, start_date=None, end_date=None)` | `Dict[str, Any]` | Defaults `agent_id` to **this provider's own** `self._agent.agent_id` so users only see actions they signed. Pass `agent_id=None` explicitly + override to fetch every agent (requires `export:read` scope). |
| `export_audit_csv(...)` | `str` | Same scoping. |
| `shutdown()` | `None` | Best-effort: only ends the session if `agent._session_id` is still set, then calls parent `shutdown()`. Robust to partial init failures via `getattr(self, "_X", None)`. |

## 4. Subfolders walked through

There are no subfolders in `src/upsonic/integrations/`. Every integration lives at the top level as a single file.

## 5. Cross-file relationships

```
                         ┌────────────────────────────┐
                         │   tracing.TracingProvider  │   (ABC)
                         │  ─ _setup()                │
                         │  ─ _create_resource()      │
                         │  ─ _create_sampler()       │
                         │  ─ _create_exporter() ◄──┐ │   (abstract)
                         │  ─ HTTP _get/_post/...   │ │
                         │  ─ shutdown/flush        │ │
                         └─────────────┬────────────┘ │
                                       │              │
              ┌────────────────────────┼──────────────┘
              │                        │              ▲
              ▼                        ▼              │
  tracing.DefaultTracingProvider   langfuse.Langfuse  │
  - OTLP gRPC/HTTP/Console         - HTTP/protobuf to │
  - UPSONIC_OTEL_* env vars          /api/public/otel │
                                   - + ~30 REST methods
                                                      │
                          asqav.AsqavGovernance ──────┘
                          - InMemorySpanExporter wrapped by
                            _AsqavSigningExporter
                          - asqav.init(); agent.sign(...)


  promptlayer.PromptLayer  (standalone — NOT a TracingProvider)
  - Used via Agent(promptlayer=pl) parameter
  - X-API-KEY header on every request
  - Background-thread logging via _register_thread / _drain_threads
```

#### Activation matrix

| Integration | Activation route | Lazy import? | Mandatory env / kwarg |
|---|---|---|---|
| `DefaultTracingProvider` | `Agent(instrument=True)` (creates one from env) **or** `Agent(instrument=DefaultTracingProvider(...))` **or** `Agent.instrument_all(True)` | yes (via `__getattr__`) | none required; falls back to `localhost:4317`, then HTTP `:4318`, then console |
| `Langfuse` | `Agent(instrument=Langfuse(...))` | yes | `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` (or constructor) |
| `AsqavGovernance` | `Agent(instrument=AsqavGovernance(...))` | yes | `ASQAV_API_KEY` (or constructor) |
| `PromptLayer` | `Agent(promptlayer=PromptLayer(...))` and `AccuracyEvaluator(promptlayer=...)` etc. | yes | `PROMPTLAYER_API_KEY` (or constructor) |

You can combine `instrument=Langfuse(...)` **with** `promptlayer=PromptLayer(...)` on the same agent — they are orthogonal.

## 6. Public API

Re-exported through `upsonic.integrations.__init__`:

```python
from upsonic.integrations import (
    TracingProvider,         # base class to subclass for custom backends
    DefaultTracingProvider,  # generic OTLP exporter (gRPC → HTTP → console fallback)
    Langfuse,                # Langfuse OTel + REST
    PromptLayer,             # PromptLayer REST (no OTel)
    AsqavGovernance,         # asqav signing exporter + REST exports
)
```

These are also re-exported from the top-level `upsonic` package, so `from upsonic import Langfuse` works.

#### Stable public methods you can rely on

`TracingProvider` (and every subclass):

```python
provider.settings              # InstrumentationSettings (handed to Agent)
provider.tracer_provider       # opentelemetry.sdk.trace.TracerProvider
provider.meter_provider        # opentelemetry.sdk.metrics.MeterProvider
provider.flush()               # force-flush spans
provider.shutdown()            # idempotent
await provider.ashutdown()     # async variant; closes async httpx client too
```

`Langfuse` adds the full Scores / Score Configs / Annotation Queues / Traces / Datasets / Dataset Items / Dataset Runs / Dataset Run Items REST surface (see §3.3 table). Every method has an `a*` async variant.

`PromptLayer` exposes the prompt registry (`get_prompt`/`aget_prompt`), unified logging (`log`/`alog`), post-hoc scoring/metadata, workflows CRUD, datasets and reports/evaluations.

`AsqavGovernance` exposes `export_audit_json(...)`, `export_audit_csv(...)` and an overridden `shutdown()` that ends the asqav session.

## 7. Integration with the rest of Upsonic

Direct import sites, found via `grep -r "from upsonic.integrations" src/`:

| Caller | Imports | Role |
|---|---|---|
| `src/upsonic/agent/agent.py` | `TracingProvider`, `DefaultTracingProvider`, `PromptLayer` | Accepts `instrument=` and `promptlayer=` kwargs; routes `_resolve_instrumentation` → `AgentOTelManager`; pumps every run through `_log_to_promptlayer_background`. |
| `src/upsonic/agent/otel_manager.py` | `TracingProvider` | Holds a reference; calls `provider.flush()` after each agent run so spans appear in Langfuse/Jaeger immediately rather than on the BatchSpanProcessor's tick. |
| `src/upsonic/agent/autonomous_agent/autonomous_agent.py` | `TracingProvider`, `PromptLayer` | Same plumbing as `Agent`, propagated to autonomous loops. |
| `src/upsonic/eval/accuracy.py` | `PromptLayer`, `Langfuse` | `AccuracyEvaluator(promptlayer=..., langfuse=...)` logs each eval run + uploads dataset items. |
| `src/upsonic/eval/reliability.py` | same | `ReliabilityEvaluator` analogous. |
| `src/upsonic/eval/performance.py` | same | `PerformanceEvaluator` analogous. |
| `src/upsonic/utils/logging_config.py` | (re-export bookkeeping) | — |

#### How `Agent._resolve_instrumentation` works

```python
def _resolve_instrumentation(self, instrument):
    from upsonic.integrations.tracing import TracingProvider as _TP
    from upsonic.agent.otel_manager import AgentOTelManager

    resolved = instrument if instrument is not None else self._global_tracing_provider

    tracing_provider = None
    settings         = None

    if resolved is True:
        from upsonic.integrations.tracing import DefaultTracingProvider
        tracing_provider = DefaultTracingProvider()
        settings = tracing_provider.settings
    elif isinstance(resolved, _TP):                          # Langfuse, AsqavGovernance, …
        tracing_provider = resolved
        settings = resolved.settings
    elif resolved and resolved is not False:                 # raw InstrumentationSettings
        settings = resolved

    if settings is not None:
        from upsonic.models.instrumented import instrument_model
        self.model = instrument_model(self.model, settings)  # wrap the LLM client

    return tracing_provider, settings, AgentOTelManager(settings, tracing_provider)
```

#### How `Agent` logs to PromptLayer

Once per task completion, `Agent` calls `_log_to_promptlayer_background(task, output, start, end)`, which spawns a thread whose body is:

```python
def _runner():
    try:
        asyncio.run(self._log_to_promptlayer_unified(task, output, start, end))
    except Exception as e:
        _pl_logger.warning("Background PromptLayer logging failed: %s", e)
thread = Thread(target=_runner, daemon=True); thread.start()
self.promptlayer._register_thread(thread)
```

`_log_to_promptlayer_unified` then:

1. Splits the model name with `pl._parse_provider_model(model_name)`.
2. Calls `await pl.alog(provider=..., model=..., input_text=task.description, output_text=str(output), start_time=..., end_time=..., input_tokens=..., output_tokens=..., price=..., parameters=..., tags=..., metadata=..., status=..., function_name=..., system_prompt=..., tools=..., tool_calls=..., tool_results=..., prompt_name=pl._last_prompt_name, prompt_id=pl._last_prompt_id, prompt_version=pl._last_prompt_version)`.
3. Stashes the returned `request_id` on `task._promptlayer_request_id` so subsequent eval runs can attach scores.
4. Calls `_create_promptlayer_workflow(...)` which uses `_created_workflows` cache, falls back to `alist_workflows` lookup, and finally either `apatch_workflow` or `acreate_workflow`.

`pl.shutdown()` blocks on `_drain_threads()` so no log is lost.

## 8. End-to-end flows

### 8.1 OTel-traced agent run via `DefaultTracingProvider`

```python
import os
os.environ["UPSONIC_OTEL_ENDPOINT"] = "http://jaeger:4317"
os.environ["UPSONIC_OTEL_SAMPLE_RATE"] = "0.5"

from upsonic import Agent

agent = Agent("openai/gpt-4o", instrument=True)
agent.print_do("Compute 2+2.")
```

Step by step:

1. `Agent.__init__` sees `instrument=True`.
2. `_resolve_instrumentation(True)` instantiates `DefaultTracingProvider()` with no kwargs.
3. `DefaultTracingProvider.__init__` reads `UPSONIC_OTEL_ENDPOINT="http://jaeger:4317"` and `UPSONIC_OTEL_SAMPLE_RATE=0.5`, then calls `super().__init__(...)`.
4. `TracingProvider._setup()` builds:
   - `Resource(service.name="upsonic")`
   - `TraceIdRatioBasedSampler(0.5)`
   - `TracerProvider(resource, sampler)` + `BatchSpanProcessor(_create_exporter())` + `BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS)`
   - `MeterProvider(resource)`
   - `InstrumentationSettings(tracer_provider, meter_provider, include_content=True, use_aggregated_usage_attribute_names=False)`
5. `_create_exporter` succeeds with the gRPC exporter (port 4317, `insecure=True`).
6. `atexit.register(self.shutdown)` is registered.
7. `Agent` wraps `self.model = instrument_model(self.model, settings)` so every LLM call emits `gen_ai.*` spans.
8. `agent.print_do("Compute 2+2.")` → an `agent.run` root span is created via `AgentOTelManager.agent_run_span(...)`. Baggage entries (`langfuse.trace.name`, `user.id`, `session.id`) are propagated by `BaggageSpanProcessor` to every child.
9. Each LLM call emits a child `gen_ai.chat <model>` span; tool calls emit `gen_ai.tool ...` spans.
10. After the run, `AgentOTelManager.flush()` calls `provider.flush()` → `tracer_provider.force_flush()` → BatchSpanProcessor flushes pending spans through the gRPC exporter.
11. On interpreter shutdown, the `atexit` handler calls `provider.shutdown()` → tracer + meter shut down, httpx client closed.

### 8.2 Langfuse-instrumented agent + post-run score

```python
from upsonic import Agent, Langfuse

lf = Langfuse(public_key="pk-lf-...", secret_key="sk-lf-...", region="eu")
agent = Agent("anthropic/claude-3-sonnet", instrument=lf,
              user_id="u-42", session_id="s-99")

result = agent.do("Summarize this PDF.")

# Trace ID is on result.trace_id (set by AgentOTelManager via baggage)
lf.score(result.trace_id, "quality", 0.93,
         data_type="NUMERIC", comment="Reviewer A")

# Update trace metadata after the fact via the ingestion API trick
lf.update_trace(result.trace_id,
                metadata={"reviewer": "alice"}, tags=["batch-42"])

lf.shutdown()
```

What happens under the hood:

1. `Langfuse.__init__` resolves keys, sets `_endpoint = "https://cloud.langfuse.com/api/public/otel/v1/traces"`, builds a Basic-Auth header, and sets `_api_base_url = "https://cloud.langfuse.com"`.
2. Calls `super().__init__(...)`, which calls `_create_exporter()` — forced HTTP/protobuf OTLP exporter aimed at the Langfuse OTel endpoint.
3. `Agent(instrument=lf)` — `_resolve_instrumentation` sees `isinstance(lf, TracingProvider)` and uses it directly; `self.model = instrument_model(self.model, lf.settings)`.
4. During `agent.do(...)`, `AgentOTelManager.agent_run_span` sets baggage (`langfuse.trace.name`, `user.id="u-42"`, `session.id="s-99"`) so `BaggageSpanProcessor` propagates them to every child span. Langfuse uses these for trace-level filtering.
5. Spans are batched and exported via HTTP/protobuf with `Authorization: Basic <b64>`.
6. `lf.score(...)` calls `self._post("/api/public/scores", body)` — the body is built by `_build_score_body` and uses the inherited `_get_client()` (httpx Client with `base_url=_host`).
7. `lf.update_trace(...)` does *not* hit a PATCH — it submits a `trace-create` ingestion event with a +5s timestamp so it wins the upsert.
8. `lf.shutdown()` flushes the BatchSpanProcessor, shuts down tracer/meter, and closes the httpx clients (sync + async).

### 8.3 Asqav-signed agent run + audit export

```python
from upsonic import Agent, Task
from upsonic.integrations.asqav import AsqavGovernance

gov = AsqavGovernance(api_key="sk_...", agent_name="finance-bot",
                      sign_tool_calls=True, sign_llm_calls=True,
                      sign_agent_steps=False)         # only sign LLM + tool spans
agent = Agent("openai/gpt-4o", instrument=gov)
agent.print_do(Task("Reconcile Q3 revenue", tools=[fetch_ledger, send_email]))

# Audit trail is scoped to gov._agent.agent_id by default
audit = gov.export_audit_json()
print(audit)

gov.shutdown()
```

Pipeline:

1. `AsqavGovernance.__init__` resolves api_key, sets sign flags, then `_init_asqav()`:
   - `import asqav` → `asqav.init(api_key=..., base_url=None)` → `agent = asqav.Agent.create("finance-bot")` → `session = agent.start_session()`.
2. `super().__init__(...)` runs — calls `_setup()` → `_create_exporter()` returns `_AsqavSigningExporter(InMemorySpanExporter(), get_agent=lambda: self._agent, ...)`.
3. Spans flow through `BatchSpanProcessor → _AsqavSigningExporter.export(spans)`.
4. For each span, `_classify(name, attrs)` checks GenAI semantic-convention keys:
   - Has `gen_ai.tool.*`? → `("tool", attrs["gen_ai.tool.name"])` → **signed** because `sign_tool_calls=True`.
   - Has `gen_ai.system` / `gen_ai.request.model` / `gen_ai.operation.name`? → `("llm", span.name)` → **signed**.
   - Otherwise → `("agent", span.name)` → **skipped** because `sign_agent_steps=False`.
5. `agent.sign(action_type=f"{kind}:{name}", context={k: str(v) for k, v in attrs.items()})` produces an ML-DSA-65 signature chained to the previous action.
6. The inner `InMemorySpanExporter` swallows the spans (no external backend).
7. `gov.export_audit_json()` defaults `agent_id` to `self._agent.agent_id` and calls `asqav.export_audit_json(...)` — returns the cryptographically-chained audit trail.
8. `gov.shutdown()` ends the asqav session (best-effort, robust to partial init), then calls `super().shutdown()`.

### 8.4 PromptLayer-logged agent run with versioned prompt

```python
from upsonic import Agent
from upsonic.integrations.promptlayer import PromptLayer

pl = PromptLayer(api_key="pl_...")

# Pull a labeled prompt — caches name/id/version on `pl`
system_prompt = pl.get_prompt("finance-summarizer", label="production",
                              variables={"quarter": "Q3"})

agent = Agent("openai/gpt-4o",
              system_prompt=system_prompt,
              promptlayer=pl)

result = agent.do("Summarize Q3 revenue figures.")

# Score the run after the fact (e.g. from a reviewer UI)
pl.score(request_id=result._promptlayer_request_id, score=92, name="accuracy")
pl.add_metadata(result._promptlayer_request_id, {"reviewer": "alice"})

pl.shutdown()
```

Flow:

1. `pl.get_prompt("finance-summarizer", label="production", variables={...})` →
   `POST /prompt-templates/finance-summarizer` body `{"label": "production", "input_variables": {...}}` → `_extract_prompt_text(result)` flattens the chat-style or string-template response → caches `_last_prompt_name/_id/_version` on `pl`.
2. `Agent(promptlayer=pl)` stores `self.promptlayer = pl`.
3. After `agent.do(...)` completes, `_log_to_promptlayer_background(task, output, start, end)` spawns a thread:
   - The thread does `asyncio.run(_log_to_promptlayer_unified(...))`.
   - `_log_to_promptlayer_unified` calls `pl._parse_provider_model("openai/gpt-4o") → ("openai", "gpt-4o")`.
   - Calls `await pl.alog(provider="openai", model="gpt-4o", input_text=task.description, output_text=str(output), input_tokens=..., output_tokens=..., price=..., status="SUCCESS", system_prompt=..., tools=[...], tool_calls=[...], tool_results=[...], prompt_name="finance-summarizer", prompt_id=<id>, prompt_version=<v>, ...)`.
   - `_build_log_body` chats up the input/output messages (system + user → assistant or assistant(tool_calls) + tool + … + assistant(final)), stringifies metadata, clamps score, attaches prompt name/id/version_number.
   - `await self._alog_post(body)` → `POST /log-request` with `X-API-KEY` header → returns `{"id": <int>}`.
   - Sets `task._promptlayer_request_id = <int>` so post-hoc scores can target this run.
4. `_create_promptlayer_workflow(...)` checks `pl._created_workflows` cache → falls back to `alist_workflows(per_page=100)` lookup → either `apatch_workflow(...)` or `acreate_workflow(name=..., nodes=[...], required_input_variables=...)`.
5. `pl.score(request_id, 92, name="accuracy")` → `POST /rest/track-score` body `{"request_id": <id>, "score": 92, "score_name": "accuracy"}`.
6. `pl.add_metadata(request_id, {...})` → `POST /rest/track-metadata` (values stringified).
7. `pl.shutdown()` calls `_drain_threads(timeout=10.0)` to join every background log thread, then closes `_client` and drops `_async_client` (its event loop is dead by now).

### 8.5 Combining Langfuse + PromptLayer on one agent

The two integrations are orthogonal, so a single agent can use both:

```python
from upsonic import Agent, Langfuse
from upsonic.integrations.promptlayer import PromptLayer

lf  = Langfuse()
pl  = PromptLayer()
sys = pl.get_prompt("agent-system", label="production")

agent = Agent("anthropic/claude-3-sonnet",
              system_prompt=sys,
              instrument=lf,        # OTel → Langfuse
              promptlayer=pl)        # REST log → PromptLayer

agent.do("...")

# Cross-link them
lf.score(result.trace_id, "promptlayer_request_id",
         result._promptlayer_request_id, data_type="CATEGORICAL")

lf.shutdown(); pl.shutdown()
```

`Agent._resolve_instrumentation(lf)` configures OTel; `Agent.__init__` separately stashes `pl` for the background log path. `instrument_model` only wraps the LLM client once (with the Langfuse settings); PromptLayer is a parallel record path that does not touch OTel at all.

## 9. Adding a new integration

To implement a fifth backend (say, `Sentry` for error tracking), the recipe is:

1. Create `src/upsonic/integrations/sentry.py`.
2. Subclass `TracingProvider` if it speaks OTel; otherwise, subclass nothing and follow the `PromptLayer` pattern (own httpx clients, own thread bookkeeping).
3. Implement `_create_exporter() -> SpanExporter` (OTel only).
4. Override `_api_headers()` if you need REST auth.
5. Set `_api_base_url` **before** calling `super().__init__()`.
6. Add the lazy re-export to `src/upsonic/integrations/__init__.py` (both the `TYPE_CHECKING` alias and the `__getattr__` branch, plus `__all__`).
7. Re-export from `src/upsonic/__init__.py` if it should be top-level.
8. If it's an `instrument=` integration, no wiring change is needed in `Agent` — `_resolve_instrumentation` accepts any `TracingProvider` subclass via `isinstance(resolved, _TP)`.
9. If it's a sidechannel like PromptLayer, add a constructor kwarg and wire up the log call in `_log_to_*_background` / `_log_to_*_unified`.
