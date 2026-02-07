"""
Smoke tests for ContextManagementMiddleware — stage-by-stage validation.

Each test class targets a specific stage of the context compression pipeline:
  1. Tool-call pruning (with and without tool calls)
  2. LLM-based summarization (text-only conversations)
  3. LLM-based summarization (conversations with tool calls)
  4. Context-full signal when all strategies fail
  5. Custom context_compression_model usage
  6. Logging output verification
"""

import json
from typing import Any, List, Optional

import pytest

from upsonic.agent.context_managers.context_management_middleware import (
    CONTEXT_FULL_MESSAGE,
    ContextManagementMiddleware,
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

pytestmark = pytest.mark.timeout(180)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LONG_PARAGRAPH: str = (
    "The rapid advancement of artificial intelligence has transformed industries "
    "across the globe. Machine learning algorithms now power recommendation systems, "
    "autonomous vehicles, medical diagnostics, and financial trading platforms. "
    "Deep learning architectures like transformers have revolutionized natural language "
    "processing, enabling large language models to generate human-quality text, translate "
    "between languages, and answer complex questions. Reinforcement learning has achieved "
    "superhuman performance in games like Go, chess, and StarCraft, demonstrating the "
    "potential for AI systems to master complex strategic reasoning. Computer vision "
    "systems can now identify objects, faces, and scenes with accuracy surpassing human "
    "performance in many benchmarks. The ethical implications of these advances continue "
    "to be debated, with concerns about bias, privacy, job displacement, and the long-term "
    "risks of artificial general intelligence. Researchers and policymakers are working "
    "together to establish frameworks for responsible AI development and deployment."
)

TOOL_RESULT_TEXT: str = (
    "Database query returned 500 records spanning Q1-Q4 2024. Revenue breakdown: "
    "North America $45.2M (up 18%), Europe $32.1M (up 12%), Asia-Pacific $28.7M "
    "(up 25%), Latin America $8.9M (up 15%), Middle East & Africa $5.1M (up 8%). "
    "Product categories: Enterprise SaaS $62.3M, SMB SaaS $31.2M, Professional "
    "Services $18.4M, Training & Certification $8.1M. Customer retention rate 94.2%, "
    "net revenue retention 118%. New customer acquisition cost $1,250, down from $1,800 "
    "in the previous year. Total active accounts: 12,847 enterprise, 45,231 SMB."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_real_model() -> Any:
    return infer_model("openai/gpt-4o")


def _get_high_context_model() -> Any:
    return infer_model("openai/gpt-4.1")


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


def _build_long_text_only_conversation(
    n_exchanges: int,
    text_multiplier: int = 50,
) -> List[Any]:
    """Build a conversation with NO tool calls, only user/assistant text exchanges.

    Each exchange contains a long paragraph multiplied to inflate token count.
    """
    msgs: List[Any] = [
        _make_system_request("You are a knowledgeable research assistant."),
    ]
    for i in range(n_exchanges):
        user_text: str = (
            f"[Exchange {i + 1}] Please elaborate on the following topic in detail: "
            + (LONG_PARAGRAPH * text_multiplier)
        )
        assistant_text: str = (
            f"[Response {i + 1}] Here is a comprehensive analysis: "
            + (LONG_PARAGRAPH * text_multiplier)
        )
        msgs.append(_make_user_request(user_text))
        msgs.append(_make_text_response(assistant_text))
    return msgs


def _build_long_tool_conversation(
    n_tool_pairs: int,
    text_multiplier: int = 30,
) -> List[Any]:
    """Build a conversation WITH tool calls and long results.

    Each tool pair has a large tool return content.
    """
    msgs: List[Any] = [
        _make_system_request("You are a data analysis assistant with database access."),
        _make_user_request(
            "Analyze our revenue data across all regions and quarters. " * text_multiplier
        ),
        _make_text_response(
            "I will query the database for comprehensive revenue analysis. " * text_multiplier
        ),
    ]
    for i in range(n_tool_pairs):
        tc_id: str = f"tc_{i}"
        tool_name: str = f"database_query_{i}"
        args: str = json.dumps({
            "sql": f"SELECT region, SUM(revenue) FROM sales WHERE quarter={i + 1} GROUP BY region",
            "batch_id": i,
        })
        msgs.append(_make_tool_call_response(tool_name, tc_id, args))
        msgs.append(_make_tool_return_request(
            tool_name, tc_id,
            (TOOL_RESULT_TEXT + f" [batch {i}] ") * text_multiplier,
        ))
    msgs.append(_make_text_response(
        "Here is the consolidated revenue analysis across all quarters. " * text_multiplier
    ))
    return msgs


def _build_mixed_conversation(
    n_text_exchanges: int,
    n_tool_pairs: int,
    text_multiplier: int = 30,
) -> List[Any]:
    """Build a conversation with both text-only exchanges AND tool calls."""
    msgs: List[Any] = [
        _make_system_request("You are a versatile AI assistant."),
    ]
    for i in range(n_text_exchanges):
        msgs.append(_make_user_request(
            f"[Text exchange {i + 1}] " + (LONG_PARAGRAPH * text_multiplier)
        ))
        msgs.append(_make_text_response(
            f"[Text response {i + 1}] " + (LONG_PARAGRAPH * text_multiplier)
        ))
    for i in range(n_tool_pairs):
        tc_id: str = f"tc_mixed_{i}"
        tool_name: str = f"analysis_tool_{i}"
        msgs.append(_make_tool_call_response(tool_name, tc_id, f'{{"step": {i}}}'))
        msgs.append(_make_tool_return_request(
            tool_name, tc_id,
            (TOOL_RESULT_TEXT + f" [step {i}] ") * text_multiplier,
        ))
    msgs.append(_make_user_request("Summarize all findings."))
    msgs.append(_make_text_response("Summary of all analysis results."))
    return msgs


# ===========================================================================
# Stage 1: Tool-call pruning
# ===========================================================================

class TestStage1ToolPruning:
    """Test that tool-call pruning works independently."""

    def test_prune_removes_old_tool_messages(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)
        msgs = _build_long_tool_conversation(n_tool_pairs=10, text_multiplier=1)

        pruned = mw._prune_tool_call_history(msgs)
        assert len(pruned) < len(msgs)

        tool_msg_count: int = 0
        for msg in pruned:
            for part in msg.parts:
                if isinstance(part, (ToolCallPart, ToolReturnPart)):
                    tool_msg_count += 1
                    break
        assert tool_msg_count == 2

    def test_prune_preserves_non_tool_messages(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)
        msgs = _build_long_tool_conversation(n_tool_pairs=8, text_multiplier=1)

        pruned = mw._prune_tool_call_history(msgs)

        has_system: bool = any(
            isinstance(p, SystemPromptPart)
            for m in pruned for p in m.parts
        )
        has_user: bool = any(
            isinstance(p, UserPromptPart)
            for m in pruned for p in m.parts
        )
        has_text: bool = any(
            isinstance(p, TextPart)
            for m in pruned for p in m.parts
        )
        assert has_system
        assert has_user
        assert has_text

    def test_no_pruning_when_under_limit(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=20)
        msgs = _build_long_tool_conversation(n_tool_pairs=3, text_multiplier=1)

        pruned = mw._prune_tool_call_history(msgs)
        assert len(pruned) == len(msgs)

    def test_has_tool_related_messages_true(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = _build_long_tool_conversation(n_tool_pairs=2, text_multiplier=1)
        assert mw._has_tool_related_messages(msgs) is True

    def test_has_tool_related_messages_false(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = _build_long_text_only_conversation(n_exchanges=3, text_multiplier=1)
        assert mw._has_tool_related_messages(msgs) is False


# ===========================================================================
# Stage 2a: Summarization with text-only conversation (no tool calls)
# ===========================================================================

class TestStage2TextOnlySummarization:
    """Verify that summarization works for conversations with NO tool calls.

    This is the critical path: if there are no tool calls, Step 1 (pruning)
    is skipped entirely, and we go directly to Step 2 (summarization).
    """

    @pytest.mark.asyncio
    async def test_text_only_conversation_gets_summarized(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        msgs = _build_long_text_only_conversation(n_exchanges=6, text_multiplier=3)
        original_chars: int = _total_content_chars(msgs)

        result = await mw._summarize_old_messages(msgs)

        assert len(result) > 0
        result_chars: int = _total_content_chars(result)
        assert result_chars < original_chars, (
            f"Summarized ({result_chars} chars) should be smaller than original ({original_chars} chars)"
        )

    @pytest.mark.asyncio
    async def test_text_only_preserves_system_prompt(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        msgs = _build_long_text_only_conversation(n_exchanges=5, text_multiplier=3)

        result = await mw._summarize_old_messages(msgs)

        first_msg = result[0]
        assert isinstance(first_msg, ModelRequest)
        has_system: bool = any(isinstance(p, SystemPromptPart) for p in first_msg.parts)
        assert has_system

    @pytest.mark.asyncio
    async def test_text_only_preserves_recent_messages(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        msgs = _build_long_text_only_conversation(n_exchanges=5, text_multiplier=3)

        result = await mw._summarize_old_messages(msgs)

        assert result[-1] is msgs[-1]
        assert result[-2] is msgs[-2]

    @pytest.mark.asyncio
    async def test_text_only_result_has_valid_structure(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        msgs = _build_long_text_only_conversation(n_exchanges=5, text_multiplier=3)

        result = await mw._summarize_old_messages(msgs)

        for msg in result:
            assert isinstance(msg, (ModelRequest, ModelResponse))
            assert hasattr(msg, 'parts')
            assert len(msg.parts) > 0


# ===========================================================================
# Stage 2b: Summarization with tool-call conversation
# ===========================================================================

class TestStage2ToolCallSummarization:
    """Verify that summarization works for conversations WITH tool calls."""

    @pytest.mark.asyncio
    async def test_tool_conversation_gets_summarized(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        msgs = _build_long_tool_conversation(n_tool_pairs=6, text_multiplier=3)
        original_chars: int = _total_content_chars(msgs)

        result = await mw._summarize_old_messages(msgs)

        assert len(result) > 0
        result_chars: int = _total_content_chars(result)
        assert result_chars <= original_chars

    @pytest.mark.asyncio
    async def test_tool_conversation_preserves_system_prompt(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        msgs = _build_long_tool_conversation(n_tool_pairs=5, text_multiplier=3)

        result = await mw._summarize_old_messages(msgs)

        first_msg = result[0]
        assert isinstance(first_msg, ModelRequest)
        assert any(isinstance(p, SystemPromptPart) for p in first_msg.parts)


# ===========================================================================
# Stage 2c: Summarization with mixed conversation
# ===========================================================================

class TestStage2MixedSummarization:
    """Verify summarization with both text-only and tool-call messages."""

    @pytest.mark.asyncio
    async def test_mixed_conversation_summarized(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model, keep_recent_count=2)

        msgs = _build_mixed_conversation(
            n_text_exchanges=4, n_tool_pairs=4, text_multiplier=3,
        )
        original_chars: int = _total_content_chars(msgs)

        result = await mw._summarize_old_messages(msgs)

        assert len(result) > 0
        result_chars: int = _total_content_chars(result)
        assert result_chars <= original_chars


# ===========================================================================
# Stage 3: Full apply() pipeline — text-only (skip tool pruning, go to summarization)
# ===========================================================================

class TestApplyTextOnlyPipeline:
    """Test the full apply() flow when there are NO tool calls.

    This validates the critical fix: without tool calls, Step 1 is skipped
    and Step 2 (summarization) is invoked directly.
    """

    @pytest.mark.asyncio
    async def test_apply_skips_pruning_and_summarizes(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(
            model=model,
            keep_recent_count=2,
            safety_margin_ratio=0.90,
        )

        msgs = _build_long_text_only_conversation(n_exchanges=8, text_multiplier=50)

        estimated_tokens: int = mw._estimate_message_tokens(msgs)
        max_window: Optional[int] = mw._get_max_context_window()
        assert max_window is not None
        effective_limit: int = int(max_window * 0.90)
        assert estimated_tokens > effective_limit, (
            f"Test setup error: tokens ({estimated_tokens}) must exceed limit ({effective_limit})"
        )

        result, ctx_full = await mw.apply(msgs)

        assert len(result) > 0
        for msg in result:
            assert isinstance(msg, (ModelRequest, ModelResponse))

    @pytest.mark.asyncio
    async def test_apply_no_action_when_under_limit(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [
            _make_user_request("Hello"),
            _make_text_response("Hi", input_tokens=100, output_tokens=20),
        ]
        result, ctx_full = await mw.apply(msgs)
        assert ctx_full is False
        assert len(result) == len(msgs)


# ===========================================================================
# Stage 4: Full apply() pipeline — with tool calls (prune first, then summarize)
# ===========================================================================

class TestApplyWithToolsPipeline:
    """Test the full apply() flow when tool calls exist."""

    @pytest.mark.asyncio
    async def test_apply_prunes_then_summarizes_if_needed(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(
            model=model,
            keep_recent_count=2,
            safety_margin_ratio=0.90,
        )

        msgs = _build_long_tool_conversation(n_tool_pairs=15, text_multiplier=80)

        estimated_tokens: int = mw._estimate_message_tokens(msgs)
        max_window: Optional[int] = mw._get_max_context_window()
        assert max_window is not None
        effective_limit: int = int(max_window * 0.90)
        assert estimated_tokens > effective_limit, (
            f"Test setup error: tokens ({estimated_tokens}) must exceed limit ({effective_limit})"
        )

        result, ctx_full = await mw.apply(msgs)

        assert len(result) > 0
        for msg in result:
            assert isinstance(msg, (ModelRequest, ModelResponse))


# ===========================================================================
# Stage 5: Context-full signal
# ===========================================================================

class TestContextFullSignal:
    """Test that context_full=True is returned when all strategies fail."""

    @pytest.mark.asyncio
    async def test_context_full_with_massive_conversation(self) -> None:
        """Create a very large conversation and verify the middleware handles it
        gracefully. When the serialized content itself exceeds the summarization
        model's limit, the middleware catches the error and signals context_full.
        """
        model = _get_real_model()
        mw = ContextManagementMiddleware(
            model=model,
            keep_recent_count=2,
            safety_margin_ratio=0.90,
        )

        msgs = _build_long_text_only_conversation(n_exchanges=15, text_multiplier=150)

        estimated_tokens: int = mw._estimate_message_tokens(msgs)
        max_window: Optional[int] = mw._get_max_context_window()
        assert max_window is not None
        effective_limit: int = int(max_window * 0.90)
        assert estimated_tokens > effective_limit, (
            f"Test setup error: tokens ({estimated_tokens}) must exceed limit ({effective_limit})"
        )

        result, ctx_full = await mw.apply(msgs)

        assert len(result) > 0
        assert ctx_full is True
        for msg in result:
            assert isinstance(msg, (ModelRequest, ModelResponse))

    def test_build_context_full_response_structure(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        resp = mw._build_context_full_response(model_name="gpt-4o")

        assert isinstance(resp, ModelResponse)
        assert len(resp.parts) == 1
        assert isinstance(resp.parts[0], TextPart)
        assert resp.parts[0].content == CONTEXT_FULL_MESSAGE
        assert resp.finish_reason == "length"
        assert resp.model_name == "gpt-4o"


# ===========================================================================
# Stage 6: Custom context_compression_model
# ===========================================================================

class TestCustomCompressionModel:
    """Test that a separate model can be used for summarization."""

    def test_get_summarization_model_returns_compression_model(self) -> None:
        agent_model = _get_real_model()
        compression_model = _get_high_context_model()
        mw = ContextManagementMiddleware(
            model=agent_model,
            context_compression_model=compression_model,
        )
        assert mw._get_summarization_model() is compression_model

    def test_get_summarization_model_falls_back_to_agent_model(self) -> None:
        agent_model = _get_real_model()
        mw = ContextManagementMiddleware(model=agent_model)
        assert mw._get_summarization_model() is agent_model

    def test_context_window_check_uses_agent_model(self) -> None:
        agent_model = _get_real_model()
        compression_model = _get_high_context_model()
        mw = ContextManagementMiddleware(
            model=agent_model,
            context_compression_model=compression_model,
        )
        max_window: Optional[int] = mw._get_max_context_window()
        assert max_window == 128_000

    @pytest.mark.asyncio
    async def test_summarization_uses_compression_model(self) -> None:
        """Verify that the compression model with larger context is used for summarization."""
        compression_model = _get_high_context_model()
        agent_model = _get_real_model()
        mw = ContextManagementMiddleware(
            model=agent_model,
            keep_recent_count=2,
            context_compression_model=compression_model,
        )

        msgs = _build_long_text_only_conversation(n_exchanges=6, text_multiplier=3)
        original_chars: int = _total_content_chars(msgs)

        result = await mw._summarize_old_messages(msgs)

        assert len(result) > 0
        result_chars: int = _total_content_chars(result)
        assert result_chars < original_chars


# ===========================================================================
# Stage 7: Logging verification
# ===========================================================================

class TestLogging:
    """Verify that the middleware produces log output during operations.

    Since pytest.ini uses ``--capture=no``, we verify logging through
    the ``logging`` module's captured records instead of capsys.
    """

    @pytest.mark.asyncio
    async def test_apply_logs_when_exceeded(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        model = _get_real_model()
        mw = ContextManagementMiddleware(
            model=model,
            keep_recent_count=2,
            safety_margin_ratio=0.90,
        )

        msgs = _build_long_text_only_conversation(n_exchanges=8, text_multiplier=50)

        with caplog.at_level(logging.INFO):
            await mw.apply(msgs)

        log_text: str = caplog.text
        assert "ContextManagement" in log_text or "Context" in log_text or "Summariz" in log_text

    @pytest.mark.asyncio
    async def test_apply_with_tools_logs_pruning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        model = _get_real_model()
        mw = ContextManagementMiddleware(
            model=model,
            keep_recent_count=2,
            safety_margin_ratio=0.90,
        )

        msgs = _build_long_tool_conversation(n_tool_pairs=15, text_multiplier=80)

        with caplog.at_level(logging.INFO):
            await mw.apply(msgs)

        log_text: str = caplog.text
        assert "ContextManagement" in log_text or "Context" in log_text or "pruning" in log_text


# ===========================================================================
# Stage 8: Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_messages(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        pruned = mw._prune_tool_call_history([])
        assert pruned == []

    @pytest.mark.asyncio
    async def test_apply_empty_messages(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        result, ctx_full = await mw.apply([])
        assert ctx_full is False
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_apply_returns_new_list(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        msgs = [
            _make_user_request("Hello"),
            _make_text_response("Hi", input_tokens=100, output_tokens=20),
        ]
        result, _ = await mw.apply(msgs)
        assert result is not msgs
        result.clear()
        assert len(msgs) == 2

    def test_constructor_stores_compression_model(self) -> None:
        agent_model = _get_real_model()
        compression_model = _get_high_context_model()
        mw = ContextManagementMiddleware(
            model=agent_model,
            context_compression_model=compression_model,
        )
        assert mw.context_compression_model is compression_model
        assert mw.model is agent_model

    def test_constructor_defaults(self) -> None:
        model = _get_real_model()
        mw = ContextManagementMiddleware(model=model)
        assert mw.context_compression_model is None
        assert mw.keep_recent_count == 5
        assert mw.safety_margin_ratio == 0.90
