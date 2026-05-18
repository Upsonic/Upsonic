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


def record_request_usage(
    request_usage: Optional["RequestUsage"],
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    kind: UsageKind = "llm",
    pipeline_step: Optional[str] = None,
    parent_entry_id: Optional[str] = None,
    cost_usd: Optional[float] = None,
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
        pipeline_step=pipeline_step,
        parent_entry_id=parent_entry_id,
        extra=extra or {},
        **tags,
    )
    reg.record(entry)
    return entry
