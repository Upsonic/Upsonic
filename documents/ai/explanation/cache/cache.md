---
name: session-task-cache
description: Use when working on Upsonic's session-level Task response cache or debugging cache hits/misses inside an Agent run. Use when a user asks to enable, configure, or troubleshoot Task caching, tune similarity thresholds, switch between vector_search and llm_call modes, wire an embedding provider for caching, inspect cache stats, or understand how cached LLM responses short-circuit pipeline steps. Trigger when the user mentions CacheManager, enable_cache, cache_method, cache_threshold, cache_duration_minutes, cache_embedding_provider, vector_search, llm_call, cosine similarity, MD5 cache key, _cache_manager, _last_llm_usage, get_cache_stats, clear_cache, store_cache_entry, get_cached_response, _find_similar_cache_entry, _llm_batch_compare_queries, or src/upsonic/cache.
---

# `src/upsonic/cache/` — Session-Level Task Response Cache

## 1. What this folder is

The `upsonic/cache/` package implements an in-process, session-scoped cache that stores previously generated `Task` outputs and serves them back on subsequent semantically-similar requests. It is the mechanism behind the `enable_cache=True` flag on `Task`, and behind the `Agent` instance attribute `_cache_manager` that is created during `Agent.__init__`. The intent is to short-circuit expensive LLM calls when an incoming task description is "the same as" or "close enough to" a previously seen one — saving tokens, money, and latency in long-running agent sessions.

The folder is intentionally tiny: a single concrete class (`CacheManager`) plus a lazy-loading `__init__.py`. There is no on-disk persistence, no Redis/SQLite backend, and no global registry. Every `Agent` constructs its own `CacheManager` keyed by `agent_id`, so cache lifetime is bounded by the lifetime of the Agent object. This is by design — Upsonic separates "session memory" (handled by `upsonic/storage/`) from "response caching" (handled here). The cache here is opt-in per `Task` (`enable_cache=False` by default) and never escapes the parent agent's process.

The `CacheManager` supports two retrieval modes that are selected via the `cache_method` field on the Task:

1. **`"vector_search"`** — embed the input text with a user-supplied embedding provider, compute cosine similarity against the stored vectors of every non-expired entry, and return the best match if it crosses the configured similarity threshold.
2. **`"llm_call"`** — when an embedding provider is unavailable or LLM-based semantic equivalence is preferred, send the new query along with the full set of cached query strings to a comparison LLM that picks the index of the most semantically similar cached query (or returns `"NONE"`).

If no LLM provider is supplied for `"llm_call"`, the manager degrades to a pure MD5-hash exact-match lookup, which is useful for unit tests and deterministic re-runs.

## 2. Folder layout

```
src/upsonic/cache/
├── __init__.py          Lazy re-export of CacheManager via module __getattr__
└── cache_manager.py     CacheManager class: storage, expiry, similarity, LLM batch comparison
```

That is the entire surface of the package — there are no submodules, helpers, or sibling files.

## 3. Top-level files

### 3.1 `__init__.py`

A 25-line lazy-loader. It declares `__all__ = ["CacheManager"]` and uses a module-level `__getattr__` so that `from upsonic.cache import CacheManager` only triggers an import of `cache_manager.py` (which transitively imports `numpy` and the heavy `upsonic.models` types) at first access.

| Symbol | Type | Description |
| --- | --- | --- |
| `__getattr__(name)` | module function | Resolves the name `"CacheManager"` to `cache_manager.CacheManager`. Any other name raises `AttributeError`. |
| `__all__` | list[str] | `["CacheManager"]` — the only public re-export. |

There is also a `TYPE_CHECKING`-guarded direct import so that static type-checkers see the symbol without paying the import cost at runtime.

### 3.2 `cache_manager.py`

A single-file module that defines:

- Two type aliases:
  - `CacheMethod = Literal["vector_search", "llm_call"]`
  - `CacheEntry = Dict[str, Any]`
- One feature-flag: `_NUMPY_AVAILABLE` (set during a `try: import numpy as np` block; if numpy is missing the manager still imports, but `_calculate_similarity` raises a friendly install-instruction error).
- One class: `CacheManager`.

Below is an exhaustive enumeration of the class.

#### `CacheManager` — fields

| Attribute | Type | Purpose |
| --- | --- | --- |
| `session_id` | `str` | Either user-supplied or auto-generated as `f"session_{int(time.time())}"`. Stamped onto every cache entry so multiple managers can coexist without collision in logs/exports. |
| `_cache_data` | `Dict[str, CacheEntry]` | The actual storage: a plain dict keyed by the MD5 hash of the input text. |
| `_cache_hits` | `int` | Counter, incremented on every successful `get_cached_response`. |
| `_cache_misses` | `int` | Counter, incremented on every miss. |
| `_last_llm_usage` | `Optional[Any]` | Token-usage record from the most recent `llm_call`-mode comparison agent run. The pipeline reads and clears this so the parent task aggregates sub-agent token spend. |

#### `CacheManager` — methods

| Method | Sync/Async | Role |
| --- | --- | --- |
| `__init__(session_id=None)` | sync | Constructor. Initializes counters and storage; defaults `session_id` to `"session_<unix-ts>"`. |
| `_generate_cache_id(input_text)` | sync | Returns `hashlib.md5(input_text.encode()).hexdigest()`. Treats `None` as an empty string so it never raises on absent input. |
| `_is_cache_expired(cache_entry, duration_minutes)` | sync | Returns `True` if `entry["timestamp"]` is older than `duration_minutes * 60` seconds. Treats missing timestamps as expired. |
| `_cleanup_expired_cache(duration_minutes)` | sync | Walks `_cache_data`, collects expired keys into a list, and deletes them. Called inside both finder methods so expiry is opportunistic — there is no background sweeper. |
| `_calculate_similarity(vector1, vector2)` | sync | Cosine similarity using numpy: `dot / (||v1|| * ||v2||)`. If numpy is missing, defers to `upsonic.utils.printing.import_error` which raises a formatted "please `pip install numpy`" message. Returns `0.0` when either vector has zero norm. |
| `_find_similar_cache_entry(input_text, input_vector, cache_threshold, duration_minutes)` | async | Iterates over non-expired entries that have an `input_vector`, computes similarity, and tracks the highest scorer above `cache_threshold`. Returns a *copy* of the entry with an extra `"similarity"` field added so callers can log the score without mutating cached state. |
| `_find_exact_cache_entry(input_text, duration_minutes, llm_provider=None)` | async | Two paths: when `llm_provider` is `None`, performs an MD5-hash exact lookup; otherwise gathers all non-expired entries and delegates to `_llm_batch_compare_queries`. |
| `_llm_batch_compare_queries(input_query, valid_entries, llm_provider)` | async | Spins up a private `Agent(model=llm_provider, debug=False)`, builds a numbered list of cached query strings, and asks it to either reply with the index of the best semantic match or `"NONE"`. Uses an inner `Task` with `enable_cache=False` (to prevent recursion) and `response_format=str`. Captures the agent's `usage` into `self._last_llm_usage` for parent aggregation. Catches all exceptions, logs via `warning_log("Batch LLM comparison failed: ...", "CacheManager")`, and returns `None` on any failure. |
| `get_cached_response(input_text, cache_method, cache_threshold, duration_minutes, embedding_provider=None, llm_provider=None)` | async | Public read API. Dispatches on `cache_method` — for `"vector_search"`, it requires `embedding_provider` and calls `embedding_provider.embed_query(input_text)`; for `"llm_call"`, it forwards to `_find_exact_cache_entry`. Increments `_cache_hits` or `_cache_misses` and returns just the cached `output` (not the full entry). |
| `store_cache_entry(input_text, output, cache_method, embedding_provider=None)` | async | Public write API. Builds an entry with `input_id`, `input`, `output`, `timestamp`, and `session_id`. If `cache_method == "vector_search"` and an embedding provider is supplied, also embeds the input and attaches it as `input_vector`. Embedding failures are silently swallowed so a transient provider error does not break the calling pipeline. |
| `get_cache_stats()` | sync | Returns a dict with `session_id`, `total_entries`, `cache_hits`, `cache_misses`, and a computed `hit_rate = hits / (hits + misses)` (or `0.0` if no requests yet). |
| `clear_cache()` | sync | Empties `_cache_data` and zeros both counters. Does **not** reset `session_id`. |
| `get_cache_size()` | sync | Returns `len(self._cache_data)`. |
| `get_session_id()` | sync | Returns `self.session_id`. |

#### Cache entry shape

Every entry written by `store_cache_entry` has the following keys:

| Key | Type | Source |
| --- | --- | --- |
| `input_id` | `str` (md5 hex) | `_generate_cache_id(input_text)` |
| `input` | `str` | the raw query text |
| `output` | `Any` | whatever the upstream Task produced (often a Pydantic model or plain str) |
| `timestamp` | `float` | `time.time()` at insertion |
| `session_id` | `str` | the manager's `session_id` |
| `input_vector` | `List[float]` | only present in `"vector_search"` mode when the embedding succeeded |

When a hit is returned via `_find_similar_cache_entry`, the returned entry also carries an extra `"similarity"` float (added on a *copy*, never on the stored object).

## 4. Subfolders

There are no subfolders inside `src/upsonic/cache/`. All logic lives in `cache_manager.py`.

## 5. Cross-file relationships

The cache module is consumed by exactly two layers in Upsonic and feeds back to a third:

### 5.1 Construction site — `Agent`

`src/upsonic/agent/agent.py` (≈line 569–572) imports `CacheManager` lazily inside `Agent.__init__` and wires it up:

```python
from upsonic.cache import CacheManager
...
self._cache_manager = CacheManager(session_id=f"agent_{self.agent_id}")
```

Two convenience methods delegate straight to the manager:

- `Agent.get_cache_stats()` → `self._cache_manager.get_cache_stats()` (line 829)
- `Agent.clear_cache()` → `self._cache_manager.clear_cache()` (line 833)

The manager is **per-Agent**, not global. Cloning, copying, or constructing a new Agent gives you a fresh empty cache.

### 5.2 Per-task wiring — `Task`

`src/upsonic/tasks/tasks.py` declares cache configuration on the `Task` model itself:

| Task field | Default | Description |
| --- | --- | --- |
| `enable_cache` | `False` | Master switch; nothing happens unless this is `True`. |
| `cache_method` | `"vector_search"` | Maps directly onto `CacheMethod`. |
| `cache_threshold` | `0.7` | Minimum cosine similarity for vector-search hits. |
| `cache_embedding_provider` | `None` | An `EmbeddingProvider` (see `upsonic/embeddings/`). Required for `"vector_search"`. |
| `cache_duration_minutes` | `60` | TTL applied during `_is_cache_expired`. |
| `_cache_manager` | `None` | Injected by the agent at runtime, *not* user-supplied. |
| `_cache_hit` | `False` | Set to `True` after a hit so downstream code can short-circuit. |
| `_original_input` | `None` | The pre-policy/pre-anonymization input used as the cache key. |
| `_last_cache_entry` | `None` | The full entry of the last hit, used by logging code to surface similarity. |

`Task` exposes user-facing wrappers:

| Method | Behaviour |
| --- | --- |
| `set_cache_manager(cache_manager)` | Stores the agent's manager on the task. |
| `get_cached_response(input_text, llm_provider=None)` | Guards on `enable_cache and self._cache_manager`, then forwards to `CacheManager.get_cached_response` with the task's own `cache_method`/`cache_threshold`/`cache_duration_minutes`/`cache_embedding_provider`. On a hit, sets `_cache_hit=True` and stores `{"output": cached_response}` into `_last_cache_entry`. |
| `store_cache_entry(input_text, output)` | Forwards to the manager with the task's configured method and embedding provider. |
| `get_cache_stats()` | Wraps `CacheManager.get_cache_stats()` and merges in the task-side fields (`cache_method`, `cache_threshold`, `cache_duration_minutes`, `cache_hit`). Returns a synthetic empty dict if no manager is wired. |
| `clear_cache()` | Calls `CacheManager.clear_cache()` and resets `_cache_hit`. |

Because `_cache_manager` is a non-pickleable runtime reference, `Task`'s custom serialization (around lines 1163–1172 and 1402–1408 in `tasks.py`) explicitly drops or reattaches it during dict-conversion, so cached responses serialize fine but the manager handle does not leak across processes.

### 5.3 Pipeline integration — `Agent.pipeline.steps`

`src/upsonic/agent/pipeline/steps.py` contains the cache-check pipeline step:

1. If the task is paused or `enable_cache` is `False`, it emits a "cache disabled" event and returns `COMPLETED`.
2. Otherwise it calls `task.set_cache_manager(agent._cache_manager)` (line 158) — this is where the agent's manager is bound to the per-call Task.
3. When `agent.debug` is set, it calls `upsonic.utils.printing.cache_configuration(...)` to print the active settings.
4. It uses `task._original_input or task.description` as the key, then `await task.get_cached_response(input_text, model)`.
5. After the call, it pulls `cache_mgr._last_llm_usage` (set by `_llm_batch_compare_queries`) and merges it into the run-level `usage` accounting before clearing the field. This is how comparison-LLM token costs are attributed back to the parent task even though they were spent inside `CacheManager`.
6. On a hit, it logs via `cache_hit(...)` with the similarity, key, and a 100-char input preview.

Storage of new entries (the write side) happens later in the pipeline — once the real LLM response is produced, that step calls `task.store_cache_entry(input_text, output)` which forwards into the manager.

### 5.4 Top-level lazy import

`src/upsonic/__init__.py` invokes `_lazy_import("upsonic.cache.cache_manager", "CacheManager")()` inside `_get_Task()` so that the `Task` Pydantic model can resolve the `_cache_manager: Optional[Any]` forward reference without forcing every Upsonic user to pay numpy/embedding import cost up front.

## 6. Public API

What end users actually import:

```python
from upsonic.cache import CacheManager
```

That is the only sanctioned import. Type aliases (`CacheMethod`, `CacheEntry`) and the `_NUMPY_AVAILABLE` flag are module-private. `CacheManager` itself is rarely instantiated by users — they configure caching via Task fields and let `Agent` build the manager.

| Surface | Visibility |
| --- | --- |
| `CacheManager` | public |
| `CacheManager.__init__(session_id=None)` | public |
| `CacheManager.get_cached_response(...)` | public (called by `Task`) |
| `CacheManager.store_cache_entry(...)` | public (called by `Task`) |
| `CacheManager.get_cache_stats()` | public |
| `CacheManager.clear_cache()` | public |
| `CacheManager.get_cache_size()` | public |
| `CacheManager.get_session_id()` | public |
| `_generate_cache_id`, `_is_cache_expired`, `_cleanup_expired_cache`, `_calculate_similarity`, `_find_similar_cache_entry`, `_find_exact_cache_entry`, `_llm_batch_compare_queries`, `_last_llm_usage` | private (leading underscore; internal use only) |

## 7. Integration with the rest of Upsonic

| Module | Relationship |
| --- | --- |
| `upsonic/agent/agent.py` | Constructs the `CacheManager` per Agent and exposes `get_cache_stats` / `clear_cache` shortcuts. |
| `upsonic/agent/pipeline/steps.py` | Implements the cache-check step that binds, queries, and reads back token usage from the manager. |
| `upsonic/tasks/tasks.py` | Declares the user-facing `enable_cache` / `cache_method` / `cache_threshold` / `cache_duration_minutes` / `cache_embedding_provider` config and the corresponding `get_cached_response` / `store_cache_entry` / `get_cache_stats` / `clear_cache` methods. Custom serialization explicitly handles `_cache_manager`. |
| `upsonic/embeddings/` (the `EmbeddingProvider` interface) | The manager calls `embedding_provider.embed_query(input_text)` in `"vector_search"` mode. The provider type is whatever the Task supplies (e.g., `OpenAIEmbedding`, `GeminiEmbedding`). |
| `upsonic/models` (`Model`) | Imported only to type-annotate `llm_provider` as `Union[Model, str]` in `_llm_batch_compare_queries`. The provider is then passed straight into a fresh `Agent`. |
| `upsonic/agent/agent.Agent` and `upsonic/tasks/tasks.Task` | Imported lazily *inside* `_llm_batch_compare_queries` to avoid a circular import — the cache manager builds a private `Agent` to run the comparison `Task`. |
| `upsonic/utils/printing` | Used for `import_error` (numpy missing), `warning_log` (LLM comparison failure), and indirectly `cache_configuration` / `cache_hit` (rendered by the pipeline step, not by the manager itself). |
| `upsonic/__init__.py` | Lazy-imports `CacheManager` to resolve `Task`'s forward references via `model_rebuild()`. |

The cache module has **no** dependency on `upsonic/storage/`, `upsonic/knowledge_base/`, `upsonic/rag/`, `upsonic/safety_engine/`, or `upsonic/team/`. It is leaf-level infrastructure.

## 8. End-to-end flow

Here is what happens when a user runs an Agent with a cache-enabled Task twice in a row.

### First call (cache miss)

1. User constructs `Agent(model=...)`. Inside `Agent.__init__`, `self._cache_manager = CacheManager(session_id=f"agent_{self.agent_id}")` runs. `_cache_data` is empty.
2. User constructs `Task(description="What is the weather in Paris?", enable_cache=True, cache_method="vector_search", cache_embedding_provider=OpenAIEmbedding(...))`.
3. User calls `await agent.do_async(task)`. The pipeline runs the cache-check step:
   - `task.set_cache_manager(agent._cache_manager)` binds the manager.
   - `input_text = task._original_input or task.description` → `"What is the weather in Paris?"`
   - `await task.get_cached_response(input_text, model)` → enters `CacheManager.get_cached_response` with `cache_method="vector_search"`.
   - The manager calls `embedding_provider.embed_query(...)`, then `_find_similar_cache_entry`. `_cache_data` is empty, so it returns `None`.
   - `_cache_misses` becomes `1`, `get_cached_response` returns `None`.
4. The pipeline proceeds to make the real LLM call. After getting the response, a downstream step calls `await task.store_cache_entry(input_text, output)` → `CacheManager.store_cache_entry`:
   - Builds the entry, embeds the input, stores under `md5("What is the weather in Paris?")`.
5. Final response is returned to the user.

### Second call (cache hit, vector search)

1. The Agent (and therefore its `CacheManager`) is still alive. `_cache_data` contains one entry.
2. User constructs another Task: `Task(description="Tell me Paris weather today.", enable_cache=True, cache_method="vector_search", cache_threshold=0.7, cache_embedding_provider=...)`.
3. Cache-check step runs:
   - Embeds `"Tell me Paris weather today."` → vector `v_new`.
   - `_find_similar_cache_entry` walks the single entry, computes cosine similarity vs. its `input_vector`. Suppose `similarity == 0.83 ≥ 0.7`.
   - Returns a copy of the entry with `"similarity": 0.83` attached.
   - `_cache_hits` becomes `1`. `get_cached_response` returns `entry["output"]`.
4. Back in the pipeline step, the cached response is non-`None`. `task._last_cache_entry` carries the similarity. The pipeline calls `cache_hit(cache_method=..., similarity=0.83, input_preview=...)` for the debug log, packages the cached output into the run result, and skips the real LLM call entirely.
5. Token cost: zero (the original embedding call may still happen, depending on the provider, but the LLM completion is bypassed).

### Second call (cache hit, llm_call mode)

If `cache_method="llm_call"` instead, the flow at step 3 is different:

1. `get_cached_response` dispatches to `_find_exact_cache_entry(input_text, duration_minutes, llm_provider=model)`.
2. Because `llm_provider` is not `None`, it skips the MD5 path and assembles the list `[(cache_id, entry, cached_input)]` of all non-expired entries.
3. `_llm_batch_compare_queries` builds a comparison prompt:

   ```
   NEW QUERY: "Tell me Paris weather today."
   CACHED QUERIES:
   1. "What is the weather in Paris?"
   ```

   It instantiates `Agent(model=llm_provider, debug=False)` and runs `comparison_task` with `enable_cache=False, response_format=str`. The agent's reply is parsed:
   - `"NONE"` → return `None` (miss).
   - `"1"` → return `valid_entries[0][1]` (hit).
4. Sub-agent token usage is captured into `self._last_llm_usage`. The pipeline step pulls it out, accumulates it onto the parent run's usage, and clears the field.
5. On a hit, the cached `output` is returned exactly as in the vector-search case.

### Expiry and cleanup

Both finder methods call `_cleanup_expired_cache(duration_minutes)` before scanning, so any entry older than `cache_duration_minutes * 60` seconds is dropped lazily on the next access. There is no scheduled cleanup. `clear_cache()` is the only way to forcibly empty the store.

### Statistics and introspection

At any point, `agent.get_cache_stats()` (or `task.get_cache_stats()` for a Task wrapper that adds the per-task config fields) returns:

```python
{
    "session_id": "agent_<id>",
    "total_entries": 7,
    "cache_hits": 4,
    "cache_misses": 3,
    "hit_rate": 4 / 7,
    # When called via Task:
    "cache_method": "vector_search",
    "cache_threshold": 0.7,
    "cache_duration_minutes": 60,
    "cache_hit": True,
}
```

That is the complete behavioural surface of the `cache/` package.
