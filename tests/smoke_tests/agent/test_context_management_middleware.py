"""
Tests for ContextManagementMiddleware.
"""

import json
from typing import Any, List

import pytest

from upsonic.agent.context_managers.context_management_middleware import (
    CONTEXT_FULL_MESSAGE,
    DEFAULT_KEEP_RECENT_COUNT,
    ConversationSummary,
    ContextManagementMiddleware,
    SummarizedRequest,
    SummarizedRequestPart,
    SummarizedResponse,
    SummarizedResponsePart,
)
from upsonic.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from upsonic.models import infer_model
from upsonic.usage import RequestUsage

pytestmark = pytest.mark.timeout(120)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_real_model():
    """Create a real model instance via the framework's infer_model."""
    return infer_model("openai/gpt-4o")


def _make_user_request(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _make_system_request(text: str) -> ModelRequest:
    return ModelRequest(parts=[SystemPromptPart(content=text)])


def _make_text_response(
    text: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model_name: str = "gpt-4o",
) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=text)],
        model_name=model_name,
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_tool_call_response(
    tool_name: str,
    tool_call_id: str,
    args: str = "{}",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)],
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_tool_return_request(
    tool_name: str,
    tool_call_id: str,
    content: str = "result",
) -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=tool_name, content=content, tool_call_id=tool_call_id)]
    )


def _total_content_chars(messages: List[Any]) -> int:
    """Sum the character length of every part's content across all messages."""
    total: int = 0
    for msg in messages:
        if not hasattr(msg, 'parts'):
            continue
        for part in msg.parts:
            if hasattr(part, 'content'):
                content = part.content
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, (dict, list)):
                    total += len(json.dumps(content, default=str))
                else:
                    total += len(str(content))
            if hasattr(part, 'args'):
                args = getattr(part, 'args', '')
                if isinstance(args, str):
                    total += len(args)
                elif isinstance(args, dict):
                    total += len(json.dumps(args, default=str))
    return total


def _old_messages_chars(
    messages: List[Any],
    keep_recent_count: int,
) -> int:
    """Compute total content chars of only the 'old' portion that gets summarized.

    Excludes the system prompt (first message if it has SystemPromptPart)
    and the last ``keep_recent_count`` non-system messages.
    """
    non_system: List[Any] = []
    for i, msg in enumerate(messages):
        if i == 0 and isinstance(msg, ModelRequest):
            if any(isinstance(p, SystemPromptPart) for p in msg.parts):
                continue
        non_system.append(msg)

    if len(non_system) <= keep_recent_count:
        return 0

    old_msgs = non_system[:-keep_recent_count]
    return _total_content_chars(old_msgs)


def _build_conversation(
    n_tool_pairs: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> List[Any]:
    """Build a conversation with a system prompt, user message, and n tool call/return pairs."""
    msgs: List[Any] = [
        _make_system_request("You are a helpful assistant."),
        _make_user_request("Do something."),
        _make_text_response("Sure, let me work on that.", input_tokens=input_tokens, output_tokens=output_tokens),
    ]
    for i in range(n_tool_pairs):
        tc_id = f"tc_{i}"
        msgs.append(_make_tool_call_response(f"tool_{i}", tc_id, f'{{"arg": {i}}}', input_tokens=input_tokens, output_tokens=output_tokens))
        msgs.append(_make_tool_return_request(f"tool_{i}", tc_id, f"result_{i}"))
    return msgs


# ===========================================================================
# Pydantic schema tests
# ===========================================================================

class TestPydanticSchemas:
    def test_summarized_request_part_system_prompt(self) -> None:
        part = SummarizedRequestPart(part_kind="system-prompt", content="You are helpful.")
        assert part.part_kind == "system-prompt"
        assert part.content == "You are helpful."
        assert part.tool_name is None
        assert part.tool_call_id is None

    def test_summarized_request_part_user_prompt(self) -> None:
        part = SummarizedRequestPart(part_kind="user-prompt", content="Hello there")
        assert part.part_kind == "user-prompt"
        assert part.content == "Hello there"

    def test_summarized_request_part_tool_return(self) -> None:
        part = SummarizedRequestPart(
            part_kind="tool-return",
            content="42",
            tool_name="calculator",
            tool_call_id="tc_1",
        )
        assert part.tool_name == "calculator"
        assert part.tool_call_id == "tc_1"
        assert part.content == "42"

    def test_summarized_response_part_text(self) -> None:
        part = SummarizedResponsePart(part_kind="text", content="Hello!")
        assert part.part_kind == "text"
        assert part.content == "Hello!"

    def test_summarized_response_part_tool_call(self) -> None:
        part = SummarizedResponsePart(
            part_kind="tool-call",
            tool_name="search",
            tool_call_id="tc_5",
            args='{"q": "test"}',
        )
        assert part.tool_name == "search"
        assert part.args == '{"q": "test"}'

    def test_conversation_summary_round_trip(self) -> None:
        summary = ConversationSummary(messages=[
            SummarizedRequest(parts=[
                SummarizedRequestPart(part_kind="user-prompt", content="Hi"),
            ]),
            SummarizedResponse(parts=[
                SummarizedResponsePart(part_kind="text", content="Hello!"),
            ]),
        ])
        json_str: str = summary.model_dump_json()
        restored = ConversationSummary.model_validate_json(json_str)
        assert len(restored.messages) == 2
        assert restored.messages[0].kind == "request"
        assert restored.messages[1].kind == "response"

    def test_conversation_summary_json_schema_has_required_keys(self) -> None:
        schema = ConversationSummary.model_json_schema()
        assert "properties" in schema
        assert "messages" in schema["properties"]

    def test_summarized_request_with_multiple_parts(self) -> None:
        req = SummarizedRequest(parts=[
            SummarizedRequestPart(part_kind="system-prompt", content="sys"),
            SummarizedRequestPart(part_kind="user-prompt", content="user"),
        ])
        assert len(req.parts) == 2
        assert req.kind == "request"

    def test_summarized_response_with_mixed_parts(self) -> None:
        resp = SummarizedResponse(parts=[
            SummarizedResponsePart(part_kind="text", content="thinking..."),
            SummarizedResponsePart(part_kind="tool-call", tool_name="fn", tool_call_id="tc_1", args="{}"),
        ])
        assert len(resp.parts) == 2
        assert resp.kind == "response"


# ===========================================================================
# _get_max_context_window (real model)
# ===========================================================================

class TestGetMaxContextWindow:
    def test_returns_known_model_window(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        result = mw._get_max_context_window()
        assert result is not None
        assert result == 128_000

    def test_returns_none_for_unknown_model(self) -> None:
        model = _get_real_model()
        original_name = model.model_name
        model._model_name = "totally-unknown-model-xyz-999"
        object.__setattr__(model, '_model_name_override', "totally-unknown-model-xyz-999")

        mw = ContextManagementMiddleware(model=model)
        # Temporarily override model_name property for this test
        import types
        mw._get_max_context_window_original = mw._get_max_context_window

        def patched_get(self_mw=mw) -> int | None:
            from upsonic.utils.usage import get_model_context_window
            return get_model_context_window("totally-unknown-model-xyz-999")

        mw._get_max_context_window = types.MethodType(lambda self: patched_get(), mw)
        result = mw._get_max_context_window()
        assert result is None


# ===========================================================================
# _estimate_message_tokens
# ===========================================================================

class TestEstimateMessageTokens:
    def test_usage_based_accumulation(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        messages = [
            _make_user_request("Hello"),
            _make_text_response("Hi", input_tokens=100, output_tokens=20),
            _make_user_request("Next question"),
            _make_text_response("Answer", input_tokens=200, output_tokens=30),
        ]
        tokens = mw._estimate_message_tokens(messages)
        assert tokens == (100 + 200) + (20 + 30)

    def test_single_response_usage(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        messages = [
            _make_user_request("Hey"),
            _make_text_response("World", input_tokens=50, output_tokens=10),
        ]
        assert mw._estimate_message_tokens(messages) == 60

    def test_char_fallback_when_no_usage(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        text = "a" * 400  # 400 chars -> 100 tokens
        messages = [_make_user_request(text)]
        assert mw._estimate_message_tokens(messages) == 100

    def test_char_fallback_with_zero_usage(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        messages = [
            _make_user_request("a" * 80),
            _make_text_response("b" * 80, input_tokens=0, output_tokens=0),
        ]
        assert mw._estimate_message_tokens(messages) == (80 + 80) // 4

    def test_tool_parts_counted_in_fallback(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [_make_tool_return_request("my_tool", "tc_1", "x" * 200)]
        tokens = mw._estimate_message_tokens(msgs)
        assert tokens > 0

    def test_mixed_usage_and_no_usage(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        messages = [
            _make_user_request("Hello"),
            _make_text_response("Hi", input_tokens=500, output_tokens=100),
            _make_user_request("More"),
            _make_text_response("End", input_tokens=0, output_tokens=0),
        ]
        assert mw._estimate_message_tokens(messages) == 500 + 100

    def test_multiple_runs_accumulated(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        messages = [
            _make_user_request("Run 1"),
            _make_text_response("R1 answer", input_tokens=1000, output_tokens=200),
            _make_user_request("Run 2"),
            _make_text_response("R2 answer", input_tokens=2000, output_tokens=300),
            _make_user_request("Run 3"),
            _make_text_response("R3 answer", input_tokens=3000, output_tokens=400),
        ]
        # All input+output accumulated: (1000+2000+3000) + (200+300+400) = 6900
        assert mw._estimate_message_tokens(messages) == 6900


# ===========================================================================
# _is_context_exceeded
# ===========================================================================

class TestIsContextExceeded:
    def test_not_exceeded_with_small_context(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, safety_margin_ratio=0.90)
        messages = [_make_text_response("hi", input_tokens=100, output_tokens=10)]
        assert mw._is_context_exceeded(messages) is False

    def test_exceeded_with_large_context(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, safety_margin_ratio=0.90)
        # gpt-4o has 128000 window; 90% = 115200; we exceed that
        messages = [_make_text_response("hi", input_tokens=110_000, output_tokens=10_000)]
        assert mw._is_context_exceeded(messages) is True

    def test_exactly_at_limit_not_exceeded(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, safety_margin_ratio=0.90)
        # 128000 * 0.90 = 115200; exactly 115200 is not exceeded (> not >=)
        messages = [_make_text_response("hi", input_tokens=115_000, output_tokens=200)]
        assert mw._is_context_exceeded(messages) is False

    def test_just_over_limit_exceeded(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, safety_margin_ratio=0.90)
        messages = [_make_text_response("hi", input_tokens=115_000, output_tokens=201)]
        assert mw._is_context_exceeded(messages) is True

    def test_custom_safety_margin(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, safety_margin_ratio=0.50)
        # 128000 * 0.50 = 64000
        messages = [_make_text_response("hi", input_tokens=60_000, output_tokens=5_000)]
        assert mw._is_context_exceeded(messages) is True


# ===========================================================================
# _prune_tool_call_history
# ===========================================================================

class TestPruneToolCallHistory:
    def test_no_pruning_when_under_limit(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=10)
        msgs = _build_conversation(3)
        pruned = mw._prune_tool_call_history(msgs)
        assert len(pruned) == len(msgs)

    def test_prunes_old_tool_messages(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)
        msgs = _build_conversation(5)
        pruned = mw._prune_tool_call_history(msgs)
        assert len(pruned) < len(msgs)

        tool_count = 0
        for msg in pruned:
            for part in msg.parts:
                if isinstance(part, (ToolCallPart, ToolReturnPart)):
                    tool_count += 1
                    break
        assert tool_count == 2

    def test_returns_new_list_object(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=10)
        msgs = _build_conversation(2)
        pruned = mw._prune_tool_call_history(msgs)
        assert pruned is not msgs

    def test_non_tool_messages_preserved(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=1)
        msgs = [
            _make_system_request("system"),
            _make_user_request("hello"),
            _make_text_response("hi"),
            _make_tool_call_response("t1", "tc_0"),
            _make_tool_return_request("t1", "tc_0"),
            _make_tool_call_response("t2", "tc_1"),
            _make_tool_return_request("t2", "tc_1"),
            _make_tool_call_response("t3", "tc_2"),
            _make_tool_return_request("t3", "tc_2"),
        ]
        pruned = mw._prune_tool_call_history(msgs)
        assert any(isinstance(p, SystemPromptPart) for m in pruned for p in m.parts)
        assert any(isinstance(p, UserPromptPart) for m in pruned for p in m.parts)
        assert any(isinstance(p, TextPart) for m in pruned for p in m.parts)

    def test_keep_recent_count_1(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=1)
        msgs = _build_conversation(4)
        pruned = mw._prune_tool_call_history(msgs)

        tool_count = 0
        for msg in pruned:
            for part in msg.parts:
                if isinstance(part, (ToolCallPart, ToolReturnPart)):
                    tool_count += 1
                    break
        assert tool_count == 1

    def test_empty_messages(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=5)
        pruned = mw._prune_tool_call_history([])
        assert pruned == []


# ===========================================================================
# _serialize_messages_for_prompt
# ===========================================================================

class TestSerializeMessagesForPrompt:
    def test_system_prompt_serialized(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [ModelRequest(parts=[SystemPromptPart(content="Be helpful.")])]
        result = mw._serialize_messages_for_prompt(msgs)
        assert "[system-prompt]" in result
        assert "Be helpful." in result

    def test_user_prompt_serialized(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [ModelRequest(parts=[UserPromptPart(content="What time is it?")])]
        result = mw._serialize_messages_for_prompt(msgs)
        assert "[user-prompt]" in result
        assert "What time is it?" in result

    def test_text_response_serialized(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [ModelResponse(parts=[TextPart(content="It's 3pm.")])]
        result = mw._serialize_messages_for_prompt(msgs)
        assert "[text]" in result
        assert "It's 3pm." in result

    def test_tool_call_serialized(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="clock", tool_call_id="tc_99", args='{"tz": "UTC"}'),
            ])
        ]
        result = mw._serialize_messages_for_prompt(msgs)
        assert "[tool-call]" in result
        assert "tool_name=clock" in result
        assert "tool_call_id=tc_99" in result

    def test_tool_return_serialized(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [_make_tool_return_request("search", "tc_7", "some result")]
        result = mw._serialize_messages_for_prompt(msgs)
        assert "[tool-return]" in result
        assert "tool_name=search" in result
        assert "tool_call_id=tc_7" in result
        assert "some result" in result

    def test_message_indices_are_1_based(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [_make_user_request("A"), _make_text_response("B")]
        result = mw._serialize_messages_for_prompt(msgs)
        assert "MESSAGE 1 [REQUEST]:" in result
        assert "MESSAGE 2 [RESPONSE]:" in result

    def test_dict_tool_args_serialized(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="fn", tool_call_id="tc_1", args={"key": "value"}),
            ])
        ]
        result = mw._serialize_messages_for_prompt(msgs)
        assert '"key"' in result
        assert '"value"' in result

    def test_mixed_request_parts(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [
            ModelRequest(parts=[
                SystemPromptPart(content="System"),
                UserPromptPart(content="User"),
                ToolReturnPart(tool_name="fn", content="data", tool_call_id="tc_1"),
            ])
        ]
        result = mw._serialize_messages_for_prompt(msgs)
        assert "[system-prompt]" in result
        assert "[user-prompt]" in result
        assert "[tool-return]" in result

    def test_empty_list_returns_empty_string(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        result = mw._serialize_messages_for_prompt([])
        assert result == ""


# ===========================================================================
# _reconstruct_messages
# ===========================================================================

class TestReconstructMessages:
    def test_request_with_user_prompt(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        summary = ConversationSummary(messages=[
            SummarizedRequest(parts=[
                SummarizedRequestPart(part_kind="user-prompt", content="Hello"),
            ]),
        ])
        result = mw._reconstruct_messages(summary)
        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[0].parts[0], UserPromptPart)
        assert result[0].parts[0].content == "Hello"

    def test_request_with_system_prompt(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        summary = ConversationSummary(messages=[
            SummarizedRequest(parts=[
                SummarizedRequestPart(part_kind="system-prompt", content="Be nice"),
            ]),
        ])
        result = mw._reconstruct_messages(summary)
        assert isinstance(result[0].parts[0], SystemPromptPart)
        assert result[0].parts[0].content == "Be nice"

    def test_request_with_tool_return(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        summary = ConversationSummary(messages=[
            SummarizedRequest(parts=[
                SummarizedRequestPart(
                    part_kind="tool-return",
                    content="42",
                    tool_name="calc",
                    tool_call_id="tc_1",
                ),
            ]),
        ])
        result = mw._reconstruct_messages(summary)
        part = result[0].parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.tool_name == "calc"
        assert part.content == "42"
        assert part.tool_call_id == "tc_1"

    def test_response_with_text(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        summary = ConversationSummary(messages=[
            SummarizedResponse(parts=[
                SummarizedResponsePart(part_kind="text", content="Answer"),
            ]),
        ])
        result = mw._reconstruct_messages(summary)
        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)
        assert isinstance(result[0].parts[0], TextPart)
        assert result[0].parts[0].content == "Answer"

    def test_response_with_tool_call(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        summary = ConversationSummary(messages=[
            SummarizedResponse(parts=[
                SummarizedResponsePart(
                    part_kind="tool-call",
                    tool_name="search",
                    tool_call_id="tc_5",
                    args='{"q": "test"}',
                ),
            ]),
        ])
        result = mw._reconstruct_messages(summary)
        part = result[0].parts[0]
        assert isinstance(part, ToolCallPart)
        assert part.tool_name == "search"
        assert part.tool_call_id == "tc_5"
        assert part.args == '{"q": "test"}'

    def test_response_model_name_set(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        summary = ConversationSummary(messages=[
            SummarizedResponse(parts=[
                SummarizedResponsePart(part_kind="text", content="ok"),
            ]),
        ])
        result = mw._reconstruct_messages(summary)
        assert result[0].model_name == "gpt-4o"

    def test_empty_parts_skipped(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        summary = ConversationSummary(messages=[
            SummarizedRequest(parts=[]),
            SummarizedResponse(parts=[]),
        ])
        result = mw._reconstruct_messages(summary)
        assert len(result) == 0

    def test_full_conversation_reconstruction(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        summary = ConversationSummary(messages=[
            SummarizedRequest(parts=[
                SummarizedRequestPart(part_kind="system-prompt", content="You are helpful."),
                SummarizedRequestPart(part_kind="user-prompt", content="Search for X."),
            ]),
            SummarizedResponse(parts=[
                SummarizedResponsePart(part_kind="tool-call", tool_name="search", tool_call_id="tc_1", args='{"q":"X"}'),
            ]),
            SummarizedRequest(parts=[
                SummarizedRequestPart(part_kind="tool-return", content="Found X", tool_name="search", tool_call_id="tc_1"),
            ]),
            SummarizedResponse(parts=[
                SummarizedResponsePart(part_kind="text", content="Here is X."),
            ]),
        ])
        result = mw._reconstruct_messages(summary)
        assert len(result) == 4
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[1], ModelResponse)
        assert isinstance(result[2], ModelRequest)
        assert isinstance(result[3], ModelResponse)

    def test_tool_return_without_ids_uses_empty_string(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        summary = ConversationSummary(messages=[
            SummarizedRequest(parts=[
                SummarizedRequestPart(part_kind="tool-return", content="data"),
            ]),
        ])
        result = mw._reconstruct_messages(summary)
        part = result[0].parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.tool_name == ""
        assert part.tool_call_id == ""


# ===========================================================================
# _build_context_full_response
# ===========================================================================

class TestBuildContextFullResponse:
    def test_contains_context_full_message(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        resp = mw._build_context_full_response(model_name="gpt-4o")
        assert isinstance(resp, ModelResponse)
        assert len(resp.parts) == 1
        assert isinstance(resp.parts[0], TextPart)
        assert resp.parts[0].content == CONTEXT_FULL_MESSAGE

    def test_model_name_propagated(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        resp = mw._build_context_full_response(model_name="my-model")
        assert resp.model_name == "my-model"

    def test_finish_reason_is_length(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        resp = mw._build_context_full_response()
        assert resp.finish_reason == "length"

    def test_none_model_name(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        resp = mw._build_context_full_response(model_name=None)
        assert resp.model_name is None

    def test_timestamp_is_set(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        resp = mw._build_context_full_response()
        assert resp.timestamp is not None


# ===========================================================================
# _summarize_old_messages (real LLM call)
# ===========================================================================

class TestSummarizeOldMessages:
    @pytest.mark.asyncio
    async def test_short_conversation_unchanged(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=5)
        msgs = [_make_user_request("A"), _make_text_response("B")]
        result = await mw._summarize_old_messages(msgs)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_system_prompt_preserved(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        msgs = [
            _make_system_request("You are a math tutor who explains concepts step by step."),
            _make_user_request(
                "Can you explain to me what calculus is, how it was invented, "
                "who are the key historical figures involved in its development, "
                "and what are the main branches of calculus used today?"
            ),
            _make_text_response(
                "Calculus is a branch of mathematics that deals with continuous change. "
                "It was independently developed by Isaac Newton and Gottfried Wilhelm Leibniz "
                "in the late 17th century. Newton focused on fluxions and their applications to "
                "physics, while Leibniz developed a more systematic notation that is still used today. "
                "The main branches are differential calculus, which studies rates of change and slopes "
                "of curves, and integral calculus, which studies accumulation of quantities and areas "
                "under curves. The Fundamental Theorem of Calculus connects these two branches."
            ),
            _make_user_request(
                "Now explain the difference between limits, derivatives, and integrals "
                "with real-world examples for each one."
            ),
            _make_text_response(
                "A limit describes the value a function approaches as its input approaches a point. "
                "For example, speed at an exact instant is the limit of average speed over shorter intervals. "
                "A derivative measures instantaneous rate of change — like a car's speedometer reading. "
                "An integral accumulates quantities — like calculating total distance from a speed-vs-time graph. "
                "In engineering, derivatives help optimize designs by finding maxima/minima, "
                "while integrals help compute areas, volumes, and total accumulated quantities."
            ),
            _make_user_request("What is the Fundamental Theorem of Calculus?"),
            _make_text_response("It connects differentiation and integration as inverse operations."),
        ]
        original_old_chars: int = _old_messages_chars(msgs, keep_recent_count=2)
        result = await mw._summarize_old_messages(msgs)

        first_msg = result[0]
        assert isinstance(first_msg, ModelRequest)
        has_system = any(isinstance(p, SystemPromptPart) for p in first_msg.parts)
        assert has_system
        sys_part = [p for p in first_msg.parts if isinstance(p, SystemPromptPart)][0]
        assert "math tutor" in sys_part.content.lower()

        result_old_chars: int = _old_messages_chars(result, keep_recent_count=2)
        assert result_old_chars < original_old_chars, (
            f"Summarized old portion ({result_old_chars} chars) must be smaller "
            f"than original old portion ({original_old_chars} chars)"
        )

    @pytest.mark.asyncio
    async def test_recent_messages_kept_verbatim(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        recent_1 = _make_user_request("KEEP_THIS_EXACT_MESSAGE_1")
        recent_2 = _make_text_response("KEEP_THIS_EXACT_MESSAGE_2")

        msgs = [
            _make_system_request("You are a helpful research assistant."),
            _make_user_request(
                "I need a comprehensive analysis of the environmental impact of electric vehicles "
                "compared to traditional combustion engine vehicles. Please cover manufacturing, "
                "battery production, daily operation, and end-of-life recycling considerations."
            ),
            _make_text_response(
                "Electric vehicles have a complex environmental profile. During manufacturing, EVs "
                "require significant energy for battery production, particularly lithium-ion batteries "
                "which involve mining lithium, cobalt, and nickel. The carbon footprint of producing an "
                "EV battery can be 30-40% higher than producing a comparable combustion engine. However, "
                "during operation, EVs produce zero tailpipe emissions. Over the vehicle's lifetime, "
                "the total emissions depend heavily on the electricity grid mix. In regions with clean "
                "energy grids, EVs can reduce lifetime emissions by 50-70%. Battery recycling remains "
                "a challenge, though advances in lithium recovery technology are improving."
            ),
            _make_user_request(
                "What about the impact of lithium mining on local water supplies and ecosystems? "
                "Are there any sustainable alternatives being developed?"
            ),
            _make_text_response(
                "Lithium mining has significant environmental consequences. In South America's Lithium "
                "Triangle, brine extraction consumes vast quantities of water — approximately 500,000 "
                "gallons per ton of lithium. This depletes aquifers and affects local agriculture. "
                "Hard rock mining in Australia produces chemical waste and requires forest clearing. "
                "Alternatives include sodium-ion batteries, solid-state batteries using abundant materials, "
                "and direct lithium extraction (DLE) technology that reduces water usage by 90%. "
                "Companies like Tesla are also researching iron-phosphate cathodes to eliminate cobalt."
            ),
            recent_1,
            recent_2,
        ]
        original_old_chars: int = _old_messages_chars(msgs, keep_recent_count=2)
        result = await mw._summarize_old_messages(msgs)

        assert result[-2] is recent_1
        assert result[-1] is recent_2

        result_old_chars: int = _old_messages_chars(result, keep_recent_count=2)
        assert result_old_chars < original_old_chars, (
            f"Summarized old portion ({result_old_chars} chars) must be smaller "
            f"than original old portion ({original_old_chars} chars)"
        )

    @pytest.mark.asyncio
    async def test_result_is_structured_messages(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        msgs = [
            _make_system_request("You are a knowledgeable programming assistant."),
            _make_user_request(
                "Explain the key differences between Python, JavaScript, and Rust. "
                "Include information about their type systems, memory management, "
                "concurrency models, and typical use cases in the industry."
            ),
            _make_text_response(
                "Python is a dynamically-typed, interpreted language with garbage collection. "
                "It uses the GIL for thread safety, making true parallelism difficult. "
                "Python excels in data science, machine learning, scripting, and web backends with Django/Flask. "
                "JavaScript is also dynamically-typed and uses an event-loop concurrency model. "
                "It runs in browsers and on servers via Node.js, dominating web development. "
                "Rust is a statically-typed, compiled systems language with no garbage collector. "
                "It uses ownership and borrowing for memory safety without runtime overhead. "
                "Rust's async model uses futures and tokio for high-performance concurrent applications. "
                "Rust is used for systems programming, WebAssembly, embedded systems, and performance-critical backends."
            ),
            _make_user_request(
                "Can you go deeper into Rust's ownership model? Explain borrowing, lifetimes, "
                "and how the borrow checker prevents data races at compile time."
            ),
            _make_text_response(
                "Rust's ownership model ensures memory safety without a garbage collector. Each value "
                "has exactly one owner, and when the owner goes out of scope, the value is dropped. "
                "Borrowing allows references to values: immutable borrows (&T) allow multiple simultaneous "
                "readers, while mutable borrows (&mut T) enforce exclusive access. Lifetimes are annotations "
                "that tell the compiler how long references are valid, preventing dangling pointers. "
                "The borrow checker enforces these rules at compile time: you cannot have a mutable reference "
                "while immutable references exist, and all references must be valid for their declared lifetime. "
                "This prevents data races because concurrent mutable access is statically disallowed."
            ),
            _make_user_request("What about async in Rust?"),
            _make_text_response("Rust uses async/await with the tokio runtime for concurrent I/O."),
        ]
        original_old_chars: int = _old_messages_chars(msgs, keep_recent_count=2)
        result = await mw._summarize_old_messages(msgs)

        for msg in result:
            assert isinstance(msg, (ModelRequest, ModelResponse))

        for msg in result:
            assert hasattr(msg, 'parts')
            assert len(msg.parts) > 0

        result_old_chars: int = _old_messages_chars(result, keep_recent_count=2)
        assert result_old_chars < original_old_chars, (
            f"Summarized old portion ({result_old_chars} chars) must be smaller "
            f"than original old portion ({original_old_chars} chars)"
        )

    @pytest.mark.asyncio
    async def test_conversation_with_tool_calls_summarized(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        msgs = [
            _make_system_request("You are a data analysis assistant with access to database and search tools."),
            _make_user_request(
                "I need a full analysis of our Q3 revenue data. First search for the report, "
                "then query the database for detailed breakdowns by region and product category."
            ),
            ModelResponse(parts=[
                TextPart(content=(
                    "I'll start by searching for the Q3 revenue report in your document system, "
                    "then I'll query the database for regional and category breakdowns."
                )),
                ToolCallPart(tool_name="document_search", tool_call_id="tc_1",
                             args='{"query": "Q3 2025 revenue report", "filters": {"type": "financial"}}'),
            ]),
            _make_tool_return_request("document_search", "tc_1",
                "Found: Q3 Revenue Report - Total revenue $12.4M, up 15% YoY. "
                "Key highlights: North America grew 22%, Europe grew 8%, APAC declined 3%. "
                "SaaS subscriptions contributed 65% of total revenue."),
            ModelResponse(parts=[
                TextPart(content=(
                    "I found the Q3 report showing $12.4M total revenue. Now let me query "
                    "the database for the detailed regional and product breakdowns."
                )),
                ToolCallPart(tool_name="database_query", tool_call_id="tc_2",
                             args='{"sql": "SELECT region, product_category, SUM(revenue) FROM sales WHERE quarter=3 GROUP BY region, product_category ORDER BY revenue DESC"}'),
            ]),
            _make_tool_return_request("database_query", "tc_2",
                "Results: NA-SaaS: $5.2M, NA-Services: $1.8M, EU-SaaS: $2.1M, "
                "EU-Services: $0.9M, APAC-SaaS: $1.5M, APAC-Services: $0.4M, Other: $0.5M"),
            _make_text_response(
                "Here's the complete Q3 revenue analysis: Total revenue reached $12.4M, a 15% "
                "increase year-over-year. North America led with $7.0M (56% of total), driven by "
                "strong SaaS growth at $5.2M. Europe contributed $3.0M with steady SaaS performance. "
                "APAC saw a slight decline to $1.9M, primarily in services. SaaS subscriptions "
                "remain the dominant revenue driver across all regions at 65% of total revenue."
            ),
            _make_user_request(
                "Now compare this with Q2 numbers and identify the biggest changes."
            ),
            ModelResponse(parts=[
                TextPart(content="Let me pull the Q2 data for comparison."),
                ToolCallPart(tool_name="database_query", tool_call_id="tc_3",
                             args='{"sql": "SELECT region, product_category, SUM(revenue) FROM sales WHERE quarter=2 GROUP BY region, product_category"}'),
            ]),
            _make_tool_return_request("database_query", "tc_3",
                "Results: NA-SaaS: $4.5M, NA-Services: $1.6M, EU-SaaS: $2.0M, "
                "EU-Services: $0.85M, APAC-SaaS: $1.6M, APAC-Services: $0.45M, Other: $0.4M"),
            _make_text_response(
                "Comparing Q3 vs Q2: The biggest improvement was NA-SaaS, growing from $4.5M to "
                "$5.2M (+15.6%). NA-Services also grew from $1.6M to $1.8M (+12.5%). The most "
                "concerning trend is APAC-SaaS declining from $1.6M to $1.5M (-6.3%) and APAC-Services "
                "dropping from $0.45M to $0.4M (-11.1%). EU showed modest growth across both categories. "
                "Overall, the quarter-over-quarter growth was driven almost entirely by North America."
            ),
            # Recent messages (last 2)
            _make_user_request("What should we focus on for Q4?"),
            _make_text_response("Focus on APAC recovery and maintaining NA SaaS momentum."),
        ]
        result = await mw._summarize_old_messages(msgs)

        for msg in result:
            assert isinstance(msg, (ModelRequest, ModelResponse))

        assert result[-1] is msgs[-1]
        assert result[-2] is msgs[-2]

        first_msg = result[0]
        assert isinstance(first_msg, ModelRequest)
        assert any(isinstance(p, SystemPromptPart) for p in first_msg.parts)

        # Verify summarized portion has fewer or equal messages than original old portion
        # (tool-call metadata must be preserved, so char reduction is not guaranteed)
        result_old_chars: int = _old_messages_chars(result, keep_recent_count=2)
        original_old_chars: int = _old_messages_chars(msgs, keep_recent_count=2)
        assert result_old_chars <= original_old_chars, (
            f"Summarized old portion ({result_old_chars} chars) must not exceed "
            f"original old portion ({original_old_chars} chars)"
        )

    @pytest.mark.asyncio
    async def test_all_messages_within_keep_count_no_summarization(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=10)

        msgs = [
            _make_system_request("sys"),
            _make_user_request("q1"),
            _make_text_response("a1"),
        ]
        result = await mw._summarize_old_messages(msgs)
        # Nothing to summarize, should return same length
        assert len(result) == len(msgs)


# ===========================================================================
# apply (orchestration, real objects)
# ===========================================================================

class TestApply:
    @pytest.mark.asyncio
    async def test_no_action_when_context_not_exceeded(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [
            _make_user_request("Hello"),
            _make_text_response("Hi", input_tokens=100, output_tokens=20),
        ]
        result, ctx_full = await mw.apply(msgs)
        assert ctx_full is False
        assert len(result) == len(msgs)

    @pytest.mark.asyncio
    async def test_returns_new_list_not_same_reference(self) -> None:
        """Verify the aliasing fix: apply() must return a NEW list."""
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [
            _make_user_request("Hello"),
            _make_text_response("Hi", input_tokens=100, output_tokens=20),
        ]
        result, _ = await mw.apply(msgs)
        assert result is not msgs
        # Mutating result must NOT affect original
        result.clear()
        assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_clear_extend_pattern_safe(self) -> None:
        """Simulate the exact pattern used in _handle_model_response."""
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [
            _make_user_request("Hello"),
            _make_text_response("Hi", input_tokens=100, output_tokens=20),
        ]
        original_len = len(msgs)

        managed_msgs, ctx_full = await mw.apply(msgs)
        msgs.clear()
        msgs.extend(managed_msgs)

        assert len(msgs) == original_len
        assert ctx_full is False

    @pytest.mark.asyncio
    async def test_empty_messages_not_exceeded(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        result, ctx_full = await mw.apply([])
        assert ctx_full is False
        assert len(result) == 0


# ===========================================================================
# Constructor defaults
# ===========================================================================

class TestConstructorDefaults:
    def test_default_keep_recent_count(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        assert mw.keep_recent_count == DEFAULT_KEEP_RECENT_COUNT

    def test_default_safety_margin_ratio(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        assert mw.safety_margin_ratio == 0.90

    def test_custom_keep_recent_count(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=10)
        assert mw.keep_recent_count == 10

    def test_custom_safety_margin(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, safety_margin_ratio=0.75)
        assert mw.safety_margin_ratio == 0.75

    def test_model_stored(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        assert mw.model is model
