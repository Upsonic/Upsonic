"""Single chokepoint that turns a fresh ``RequestUsage`` into a
``UsageEntry`` and appends it to the default registry.

Phase 2 wires every place an LLM response's usage is first observed (the
"fresh" emission sites) through here. Roll-up sites — where a parent
adds an already-recorded sub-output's usage to its own — are NOT hooked,
since their entries are already in the ledger; double-recording would
silently double-count and break the registry's idempotency promise.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from upsonic.usage_registry.entry import UsageEntry, UsageKind
from upsonic.usage_registry.registry import UsageRegistry, get_default_registry
from upsonic.usage_registry.scope import current_scope_tags

if TYPE_CHECKING:
    from upsonic.usage import RequestUsage


def record_response_usage(
    response: Any,
    *,
    model: Any = None,
    pipeline_step: str,
    model_execution_time: float = 0.0,
    run_output: Any = None,
    compute_cost: bool = True,
) -> Optional[float]:
    """Standard handling for a fresh ``model.request(...)`` response.

    Wraps the three things every emission site does:

    1. Compute ``cost_usd`` via :func:`calculate_cost_from_usage` from
       the response usage + model (skipped when ``compute_cost=False``).
    2. Fold the response usage into the per-run snapshot on
       ``run_output`` (``_ensure_usage().incr(...)`` +
       ``set_usage_cost(...)``) — only when ``run_output`` is given.
    3. Write a ``UsageEntry`` into the default usage registry under
       the active scope contextvars via :func:`record_request_usage`.

    Every step is wrapped in its own ``try/except`` so a pricing
    failure / snapshot failure / registry failure can't propagate and
    mask the real model response. Returns the computed cost (or
    ``None`` when no cost was resolved) so the caller can keep
    using it.

    Args:
        response: The :class:`ModelResponse` from
            ``await model.request(...)``. Only the ``.usage`` attribute
            is touched; everything else is ignored.
        model: The model instance (or string id) used for pricing
            resolution. Pass ``None`` to skip cost lookup.
        pipeline_step: Free-form step name attached to the ledger row
            (``"model_call"`` / ``"model_call_retry"`` /
            ``"guardrail"`` / ...).
        model_execution_time: Per-call elapsed wall-clock from
            ``time.time()`` before/after the request. Becomes both
            ``UsageEntry.model_execution_time`` and
            ``UsageEntry.duration`` so :class:`AggregatedUsage` rolls
            them up.
        run_output: The active :class:`AgentRunOutput` to update the
            per-run snapshot on. ``None`` skips the snapshot fold
            (used by ``direct.py`` where the snapshot lives on the
            task instead).
        compute_cost: When ``False`` the cost-calculation step is
            skipped — useful when the caller already has the value or
            when pricing isn't meaningful for this call.

    Returns:
        The computed ``cost_value`` (USD float), or ``None`` if no
        cost was resolved.
    """
    if not (hasattr(response, "usage") and response.usage):
        return None

    response_usage = response.usage
    cost_value: Optional[float] = None

    if compute_cost and model is not None:
        try:
            from upsonic.utils.usage import calculate_cost_from_usage
            cost_value = calculate_cost_from_usage(response_usage, model)
        except Exception:
            cost_value = None

    if run_output is not None:
        try:
            run_output._ensure_usage().incr(response_usage)
            if cost_value is not None:
                run_output.set_usage_cost(cost_value)
        except Exception:
            pass

    try:
        record_request_usage(
            response_usage,
            model=getattr(model, "model_name", None) if model is not None else None,
            pipeline_step=pipeline_step,
            cost_usd=cost_value,
            model_execution_time=model_execution_time,
        )
    except Exception:
        pass

    return cost_value


def record_request_usage(
    request_usage: Optional["RequestUsage"],
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    kind: UsageKind = "llm",
    pipeline_step: Optional[str] = None,
    parent_entry_id: Optional[str] = None,
    cost_usd: Optional[float] = None,
    model_execution_time: float = 0.0,
    time_to_first_token: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    registry: Optional[UsageRegistry] = None,
) -> Optional[UsageEntry]:
    """Append one ledger row for a fresh model-response usage.

    Args:
        request_usage: The :class:`~upsonic.usage.RequestUsage` that just
            came off ``model.request(...)`` (or its async variant). If
            ``None`` or zero-token, no entry is recorded and ``None`` is
            returned — saves the caller a guard.
        model: Optional model identifier; falls back to whatever the
            request_usage carries (usually nothing).
        provider: Optional provider id, mostly for analytics dashboards.
        kind: Ledger row ``kind`` discriminator. Default ``"llm"``.
        pipeline_step: Free-form step name (``"model_call"``,
            ``"retry"``, ``"summarization"``, ``"culture"``, ...). Lets
            future queries break down spend by pipeline stage.
        parent_entry_id: For sub-agent / nested calls — points at the
            ledger row of the parent's emission, so a query can roll up
            a whole tree.
        cost_usd: Pre-computed USD cost when the caller already has it.
            When ``None`` the row records token counts but no cost; a
            later pricing pass can backfill from ``model``.
        extra: Free-form metadata; never relied on by aggregation.
        registry: Override registry instance — Phase 4 will use this for
            storage-backed instances; tests pass a clean throw-away
            registry to avoid global state. Defaults to
            :func:`get_default_registry`.

    Returns:
        The created :class:`UsageEntry`, or ``None`` if ``request_usage``
        was empty / falsy.
    """
    if request_usage is None:
        return None

    # Token-zero responses are useful to record for cache-hit / refusal
    # analytics, but for now follow the existing ``incr`` guard pattern
    # and skip them.
    input_tokens = getattr(request_usage, "input_tokens", 0) or 0
    output_tokens = getattr(request_usage, "output_tokens", 0) or 0
    if input_tokens == 0 and output_tokens == 0:
        return None

    reg = registry if registry is not None else get_default_registry()
    tags = current_scope_tags()

    entry = UsageEntry(
        kind=kind,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=getattr(request_usage, "cache_read_tokens", 0) or 0,
        cache_write_tokens=getattr(request_usage, "cache_write_tokens", 0) or 0,
        reasoning_tokens=getattr(request_usage, "reasoning_tokens", 0) or 0,
        input_audio_tokens=getattr(request_usage, "input_audio_tokens", 0) or 0,
        output_audio_tokens=getattr(request_usage, "output_audio_tokens", 0) or 0,
        cache_audio_read_tokens=getattr(request_usage, "cache_audio_read_tokens", 0) or 0,
        requests=getattr(request_usage, "requests", 1) or 1,
        cost_usd=cost_usd,
        model_execution_time=model_execution_time,
        duration=model_execution_time,
        time_to_first_token=time_to_first_token,
        pipeline_step=pipeline_step,
        parent_entry_id=parent_entry_id,
        extra=extra or {},
        **tags,
    )
    reg.record(entry)
    return entry
