"""Cache-aware estimated-cost for the per-call cost panels.

QA bug: the per-call "LLM Result" / Direct cost panels priced only
``{input_tokens, output_tokens}`` and charged cached input at full rate, so the
displayed *Estimated Cost* over-estimated relative to the cache-aware
``chat.usage.cost`` / Agent Metrics value (the registry path was already
correct). ``get_estimated_cost_from_usage`` honors ``cache_read_tokens`` /
``cache_write_tokens`` so the panel matches the registry.

These are pure-pricing assertions — ``genai_prices`` ships offline pricing data,
so no network is needed.
"""
import re

import pytest

from upsonic.utils.printing import (
    get_estimated_cost,
    get_estimated_cost_from_usage,
)
from upsonic.utils.usage import calculate_cost_from_usage, format_cost


MODEL = "openai/gpt-4o"


def _to_float(cost_str: str) -> float:
    """Parse a formatted cost string like ``~$0.0158`` to a float."""
    match = re.search(r"[-+]?\d*\.?\d+", cost_str.replace(",", ""))
    return float(match.group()) if match else 0.0


def test_cache_read_tokens_lower_the_estimated_cost():
    """A large cached-read portion must price strictly cheaper than charging
    the same input at full rate — i.e. the cache discount actually flows."""
    blind = _to_float(get_estimated_cost(10_000, 200, MODEL))
    aware = _to_float(
        get_estimated_cost_from_usage(
            {"input_tokens": 10_000, "output_tokens": 200, "cache_read_tokens": 9_000},
            MODEL,
        )
    )
    assert blind > 0.0
    assert aware < blind, f"cache-aware {aware} should be < cache-blind {blind}"


def test_no_cache_fields_matches_plain_estimate():
    """Backward compatibility: a usage dict without cache fields prices
    identically to the plain input/output path — no regression for callers
    that never carried cache tokens."""
    plain = get_estimated_cost(10_000, 200, MODEL)
    via_usage = get_estimated_cost_from_usage(
        {"input_tokens": 10_000, "output_tokens": 200}, MODEL
    )
    assert via_usage == plain


def test_panel_matches_registry_cache_aware_cost():
    """The panel helper must agree with the registry's cache-aware
    ``calculate_cost_from_usage`` for the same usage — this is what removes the
    ~2x panel/registry discrepancy the QA report flagged."""
    usage = {"input_tokens": 10_000, "output_tokens": 200, "cache_read_tokens": 9_000}
    expected = format_cost(calculate_cost_from_usage(usage, MODEL), approximate=True)
    assert get_estimated_cost_from_usage(usage, MODEL) == expected
