"""Tool-call counting flows into the usage ledger (QA bug batch F6).

Before the fix, UsageEntry.tool_calls was never populated, so chat.usage.tool_calls
and the agent metrics "Total Tool Calls" panel stayed 0 even when tools ran. The
fix counts ToolCallParts at the record_response_usage chokepoint (requested basis)
and plumbs the count into the ledger entry.
"""
from __future__ import annotations

import pytest

from upsonic.messages import ModelResponse, TextPart
from upsonic.messages.messages import ToolCallPart
from upsonic.usage import RequestUsage
from upsonic.usage_registry import (
    UsageRegistry,
    get_default_registry,
    record_request_usage,
    record_response_usage,
    scope,
)


def test_record_request_usage_sets_tool_calls():
    reg = UsageRegistry()
    ru = RequestUsage(input_tokens=10, output_tokens=5)
    entry = record_request_usage(ru, tool_calls=2, registry=reg)
    assert entry is not None
    assert entry.tool_calls == 2


def test_zero_token_response_with_tool_calls_still_records():
    # A token-zero response that carried tool calls must NOT be dropped.
    reg = UsageRegistry()
    ru = RequestUsage(input_tokens=0, output_tokens=0)
    entry = record_request_usage(ru, tool_calls=1, registry=reg)
    assert entry is not None
    assert entry.tool_calls == 1
    assert len(reg) == 1


def test_zero_token_response_without_tool_calls_still_skipped():
    # Characterization: zero tokens AND zero tool calls is still skipped.
    reg = UsageRegistry()
    ru = RequestUsage(input_tokens=0, output_tokens=0)
    assert record_request_usage(ru, tool_calls=0, registry=reg) is None
    assert len(reg) == 0


def _make_response(n_tool_calls: int) -> ModelResponse:
    parts = [
        ToolCallPart(tool_name=f"tool_{i}", args={"x": i})
        for i in range(n_tool_calls)
    ]
    parts.append(TextPart(content="done"))
    return ModelResponse(
        parts=parts,
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=RequestUsage(input_tokens=100, output_tokens=20),
        provider_name="test-provider",
        provider_response_id="test-id",
        provider_details={},
        finish_reason="stop",
    )


def test_record_response_usage_counts_tool_call_parts():
    unique_chat = "tool-calls-test-chat-xyz"
    with scope(chat_usage_id=unique_chat):
        record_response_usage(
            _make_response(n_tool_calls=3),
            model=None,
            pipeline_step="model_call",
        )
    agg = get_default_registry().by_chat(unique_chat)
    assert agg.tool_calls == 3


@pytest.mark.asyncio
async def test_response_without_tool_calls_records_zero():
    unique_chat = "no-tool-calls-test-chat-xyz"
    with scope(chat_usage_id=unique_chat):
        record_response_usage(
            _make_response(n_tool_calls=0),
            model=None,
            pipeline_step="model_call",
        )
    agg = get_default_registry().by_chat(unique_chat)
    assert agg.tool_calls == 0
