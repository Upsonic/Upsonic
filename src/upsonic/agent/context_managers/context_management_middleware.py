from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from upsonic.messages.messages import ModelMessage, ModelResponse
    from upsonic.models import Model


CONTEXT_FULL_MESSAGE: str = (
    "[SYSTEM] The conversation context window has been exceeded. "
    "I am unable to process further messages in this session. "
    "Please start a new conversation or reduce the context size."
)

DEFAULT_KEEP_RECENT_COUNT: int = 5


class SummarizedRequestPart(BaseModel):
    """A single part inside a summarized ModelRequest."""
    part_kind: Literal["system-prompt", "user-prompt", "tool-return"] = Field(
        description="The type of part: 'system-prompt', 'user-prompt', or 'tool-return'."
    )
    content: str = Field(
        description="The text content for this part."
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Tool name (required when part_kind is 'tool-return')."
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Tool call ID linking a tool-return to its tool-call (required when part_kind is 'tool-return')."
    )


class SummarizedResponsePart(BaseModel):
    """A single part inside a summarized ModelResponse."""
    part_kind: Literal["text", "tool-call"] = Field(
        description="The type of part: 'text' for assistant text, 'tool-call' for a tool invocation."
    )
    content: Optional[str] = Field(
        default=None,
        description="The text content (required when part_kind is 'text')."
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Tool name (required when part_kind is 'tool-call')."
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Tool call identifier (required when part_kind is 'tool-call')."
    )
    args: Optional[str] = Field(
        default=None,
        description="Tool call arguments as a JSON string (required when part_kind is 'tool-call')."
    )


class SummarizedRequest(BaseModel):
    """A summarized ModelRequest (user/system → model)."""
    kind: Literal["request"] = "request"
    parts: List[SummarizedRequestPart] = Field(
        description="Ordered list of parts in this request."
    )


class SummarizedResponse(BaseModel):
    """A summarized ModelResponse (model → user)."""
    kind: Literal["response"] = "response"
    parts: List[SummarizedResponsePart] = Field(
        description="Ordered list of parts in this response."
    )


class ConversationSummary(BaseModel):
    """The complete summarized conversation returned by the LLM."""
    messages: List[Union[SummarizedRequest, SummarizedResponse]] = Field(
        description=(
            "Ordered list of summarized messages. "
            "Must alternate between 'request' and 'response' kinds. "
            "The first message must be a 'request'."
        )
    )



SUMMARY_SYSTEM_PROMPT: str = """\
You are a conversation summarizer. You will receive a structured conversation \
and must return a CONDENSED version of it as valid JSON matching the schema below.

RULES:
1. Preserve the request/response alternation order.
2. Merge multiple user messages into fewer user messages when possible.
3. Merge multiple assistant text responses into fewer responses when possible.
4. For tool calls and their results: keep only the MOST IMPORTANT ones. \
   Summarize or drop tool calls that are redundant or whose results were not used.
5. Condense long text content into concise summaries while preserving key facts, \
   decisions, and outcomes.
6. Keep system prompts intact — do NOT summarize them.
7. Every tool-return MUST have a matching tool-call with the same tool_call_id \
   in a preceding response.
8. Return ONLY the JSON object. No markdown fences, no extra text.

JSON SCHEMA:
{schema}
"""


_LOG_CONTEXT: str = "ContextManagement"


class ContextManagementMiddleware:
    """Middleware that manages context window overflow for agent conversations.

    When the message history exceeds the model's maximum context window,
    this middleware applies a series of strategies in order:

    1. Prune old tool call/return pairs, keeping only the last ``keep_recent_count``.
    2. Summarize old messages via the LLM into properly structured
       ModelRequest / ModelResponse objects.
    3. If the context is still full after all strategies, inject a fixed
       "context full" response and stop further processing.

    An optional ``context_compression_model`` can be provided to use a
    different (typically higher-context-window) model specifically for
    the summarization step, while still using the agent's primary model
    for context-window limit checks.
    """

    def __init__(
        self,
        model: "Model",
        keep_recent_count: int = DEFAULT_KEEP_RECENT_COUNT,
        safety_margin_ratio: float = 0.90,
        context_compression_model: Optional["Model"] = None,
    ) -> None:
        """
        Args:
            model: The Model instance, used for token counting and context window lookup.
            keep_recent_count: Number of recent tool-call events / messages
                to preserve when pruning or summarizing (default 5).
            safety_margin_ratio: Use this fraction of the max context window as
                the effective limit (default 0.90 = 90%).
            context_compression_model: Optional separate Model instance with a
                larger context window to use for the summarization LLM call.
                If None, the primary ``model`` is used for summarization.
        """
        self.model: "Model" = model
        self.keep_recent_count: int = keep_recent_count
        self.safety_margin_ratio: float = safety_margin_ratio
        self.context_compression_model: Optional["Model"] = context_compression_model

    def _get_summarization_model(self) -> "Model":
        """Return the model to use for the summarization LLM call."""
        if self.context_compression_model is not None:
            return self.context_compression_model
        return self.model

    def _get_max_context_window(self) -> Optional[int]:
        """Get the max context window for the current model."""
        from upsonic.utils.usage import get_model_context_window

        model_name: str = self.model.model_name
        return get_model_context_window(model_name)

    def _estimate_message_tokens(self, messages: List["ModelMessage"]) -> int:
        """Estimate the total token count of the conversation.

        The ``messages`` list may span multiple agent runs. Each
        ``ModelResponse`` carries a ``usage`` field (``RequestUsage``)
        whose ``input_tokens`` reflects the input context sent to the
        model for *that particular turn*, and ``output_tokens`` reflects
        the tokens the model generated in that turn.

        Because the list can contain responses from different runs,
        we accumulate **all** ``input_tokens`` and **all**
        ``output_tokens`` across every ``ModelResponse`` to get the
        total token footprint of the conversation.

        Falls back to a character-based heuristic (~4 chars/token) when no
        ``ModelResponse`` with usage data exists in the message list.
        """
        from upsonic.messages import ModelResponse

        total_input_tokens: int = 0
        total_output_tokens: int = 0
        has_usage: bool = False

        for message in messages:
            if isinstance(message, ModelResponse) and hasattr(message, 'usage'):
                usage = message.usage
                if usage.input_tokens > 0 or usage.output_tokens > 0:
                    has_usage = True
                    total_input_tokens += usage.input_tokens
                    total_output_tokens += usage.output_tokens

        if has_usage:
            return total_input_tokens + total_output_tokens

        total_chars: int = 0
        for message in messages:
            if hasattr(message, 'parts'):
                for part in message.parts:
                    if hasattr(part, 'content'):
                        content = part.content
                        if isinstance(content, str):
                            total_chars += len(content)
                        elif isinstance(content, (dict, list)):
                            total_chars += len(json.dumps(content, default=str))
                        else:
                            total_chars += len(str(content))
                    if hasattr(part, 'tool_name'):
                        total_chars += len(str(getattr(part, 'tool_name', '')))
                    if hasattr(part, 'args'):
                        args = getattr(part, 'args', '')
                        if isinstance(args, str):
                            total_chars += len(args)
                        elif isinstance(args, dict):
                            total_chars += len(json.dumps(args, default=str))
        return total_chars // 4

    def _is_context_exceeded(self, messages: List["ModelMessage"]) -> bool:
        """Check if the current messages exceed the model's context window."""
        max_window: Optional[int] = self._get_max_context_window()
        if max_window is None:
            return False

        effective_limit: int = int(max_window * self.safety_margin_ratio)
        estimated_tokens: int = self._estimate_message_tokens(messages)
        return estimated_tokens > effective_limit

    def _has_tool_related_messages(self, messages: List["ModelMessage"]) -> bool:
        """Check whether the message list contains any tool call or tool return parts."""
        from upsonic.messages import ToolCallPart, ToolReturnPart

        for msg in messages:
            if not hasattr(msg, 'parts'):
                continue
            for part in msg.parts:
                if isinstance(part, (ToolCallPart, ToolReturnPart)):
                    return True
        return False

    def _prune_tool_call_history(
        self,
        messages: List["ModelMessage"],
    ) -> List["ModelMessage"]:
        """Remove old tool call/return pairs, keeping only the most recent ones.

        Args:
            messages: The full message list.

        Returns:
            A new list of messages with old tool call history pruned.
        """
        from upsonic.messages import ToolCallPart, ToolReturnPart

        tool_related_indices: List[int] = []
        for i, msg in enumerate(messages):
            if not hasattr(msg, 'parts'):
                continue
            for part in msg.parts:
                if isinstance(part, (ToolCallPart, ToolReturnPart)):
                    tool_related_indices.append(i)
                    break

        if len(tool_related_indices) <= self.keep_recent_count:
            return list(messages)

        indices_to_remove: set[int] = set(tool_related_indices[:-self.keep_recent_count])
        return [msg for i, msg in enumerate(messages) if i not in indices_to_remove]


    def _serialize_messages_for_prompt(
        self,
        messages: List["ModelMessage"],
    ) -> str:
        """Serialize a list of ModelMessage objects into a human-readable
        structured text representation for the LLM prompt.
        """
        from upsonic.messages import (
            ModelRequest,
            ModelResponse,
            SystemPromptPart,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )

        lines: List[str] = []
        for idx, msg in enumerate(messages):
            if isinstance(msg, ModelRequest):
                lines.append(f"MESSAGE {idx + 1} [REQUEST]:")
                for p_idx, part in enumerate(msg.parts):
                    prefix = f"  Part {p_idx + 1}"
                    if isinstance(part, SystemPromptPart):
                        lines.append(f"{prefix} [system-prompt]: {part.content}")
                    elif isinstance(part, UserPromptPart):
                        content_str = part.content if isinstance(part.content, str) else str(part.content)
                        lines.append(f"{prefix} [user-prompt]: {content_str}")
                    elif isinstance(part, ToolReturnPart):
                        content_str = part.content if isinstance(part.content, str) else json.dumps(part.content, default=str)
                        lines.append(
                            f"{prefix} [tool-return] tool_name={part.tool_name} "
                            f"tool_call_id={part.tool_call_id}: {content_str}"
                        )
                    else:
                        lines.append(f"{prefix} [unknown-request-part]: {part}")
            elif isinstance(msg, ModelResponse):
                lines.append(f"MESSAGE {idx + 1} [RESPONSE]:")
                for p_idx, part in enumerate(msg.parts):
                    prefix = f"  Part {p_idx + 1}"
                    if isinstance(part, TextPart):
                        lines.append(f"{prefix} [text]: {part.content}")
                    elif isinstance(part, ToolCallPart):
                        args_str = part.args if isinstance(part.args, str) else json.dumps(part.args, default=str)
                        lines.append(
                            f"{prefix} [tool-call] tool_name={part.tool_name} "
                            f"tool_call_id={part.tool_call_id} args={args_str}"
                        )
                    else:
                        lines.append(f"{prefix} [other-response-part]: {part}")
            else:
                lines.append(f"MESSAGE {idx + 1} [UNKNOWN]: {msg}")

        return "\n".join(lines)

    def _reconstruct_messages(
        self,
        summary: ConversationSummary,
    ) -> List["ModelMessage"]:
        """Reconstruct proper ModelRequest / ModelResponse objects from
        the Pydantic ``ConversationSummary`` returned by the LLM.
        """
        from upsonic.messages import (
            ModelRequest,
            ModelResponse,
            SystemPromptPart,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )
        from upsonic._utils import now_utc

        reconstructed: List["ModelMessage"] = []

        for msg in summary.messages:
            if msg.kind == "request":
                parts: List[Any] = []
                for p in msg.parts:
                    if p.part_kind == "system-prompt":
                        parts.append(SystemPromptPart(content=p.content))
                    elif p.part_kind == "user-prompt":
                        parts.append(UserPromptPart(content=p.content))
                    elif p.part_kind == "tool-return":
                        parts.append(ToolReturnPart(
                            tool_name=p.tool_name or "",
                            content=p.content,
                            tool_call_id=p.tool_call_id or "",
                        ))
                if parts:
                    reconstructed.append(ModelRequest(parts=parts))

            elif msg.kind == "response":
                parts_resp: List[Any] = []
                for p in msg.parts:
                    if p.part_kind == "text":
                        parts_resp.append(TextPart(content=p.content or ""))
                    elif p.part_kind == "tool-call":
                        parts_resp.append(ToolCallPart(
                            tool_name=p.tool_name or "",
                            args=p.args,
                            tool_call_id=p.tool_call_id or "",
                        ))
                if parts_resp:
                    reconstructed.append(ModelResponse(
                        parts=parts_resp,
                        model_name=self.model.model_name,
                        timestamp=now_utc(),
                    ))

        return reconstructed

    async def _summarize_old_messages(
        self,
        messages: List["ModelMessage"],
    ) -> List["ModelMessage"]:
        """Summarize old messages via the LLM into structured ModelRequest /
        ModelResponse objects, keeping the last ``self.keep_recent_count``
        messages verbatim.

        Uses the ``context_compression_model`` if set, otherwise falls
        back to the primary agent model.

        Args:
            messages: The full message list.

        Returns:
            A new list with old messages replaced by LLM-summarized messages.
        """
        from upsonic.messages import (
            ModelRequest,
            SystemPromptPart,
            UserPromptPart,
        )
        from upsonic.utils.printing import info_log

        if len(messages) <= self.keep_recent_count:
            return list(messages)

        system_messages: List["ModelMessage"] = []
        non_system_messages: List["ModelMessage"] = []

        for i, msg in enumerate(messages):
            if i == 0 and isinstance(msg, ModelRequest):
                if any(isinstance(p, SystemPromptPart) for p in msg.parts):
                    system_messages.append(msg)
                    continue
            non_system_messages.append(msg)

        if len(non_system_messages) <= self.keep_recent_count:
            return list(messages)

        old_messages: List["ModelMessage"] = non_system_messages[:-self.keep_recent_count]
        recent_messages: List["ModelMessage"] = non_system_messages[-self.keep_recent_count:]

        serialized_conversation: str = self._serialize_messages_for_prompt(old_messages)

        if not serialized_conversation.strip():
            return system_messages + recent_messages

        schema_json: str = json.dumps(
            ConversationSummary.model_json_schema(), indent=2
        )
        system_instruction: str = SUMMARY_SYSTEM_PROMPT.format(schema=schema_json)

        summary_prompt: str = (
            f"<conversation>\n{serialized_conversation}\n</conversation>"
        )

        from upsonic.models import ModelRequestParameters

        request_msg = ModelRequest(parts=[
            SystemPromptPart(content=system_instruction),
            UserPromptPart(content=summary_prompt),
        ])
        model_params = ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        )

        from upsonic.messages import TextPart

        summarization_model: "Model" = self._get_summarization_model()

        info_log(
            f"Summarizing {len(old_messages)} old messages using model "
            f"'{summarization_model.model_name}' (keeping {len(recent_messages)} recent)",
            context=_LOG_CONTEXT,
        )

        try:
            llm_response: "ModelResponse" = await summarization_model.request(
                messages=[request_msg],
                model_settings=summarization_model.settings,
                model_request_parameters=model_params,
            )
        except Exception as exc:
            from upsonic.utils.printing import warning_log
            warning_log(
                f"Summarization LLM call failed ({type(exc).__name__}: {exc}). "
                f"Returning original messages without summarization.",
                context=_LOG_CONTEXT,
            )
            return list(messages)

        raw_text: str = ""
        for part in llm_response.parts:
            if isinstance(part, TextPart):
                raw_text += part.content

        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            first_newline = raw_text.find("\n")
            if first_newline != -1:
                raw_text = raw_text[first_newline + 1:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

        try:
            summary: ConversationSummary = ConversationSummary.model_validate_json(raw_text)
        except Exception as exc:
            from upsonic.utils.printing import warning_log
            warning_log(
                f"Failed to parse summarization response ({type(exc).__name__}: {exc}). "
                f"Returning original messages without summarization.",
                context=_LOG_CONTEXT,
            )
            return list(messages)

        summarized_messages: List["ModelMessage"] = self._reconstruct_messages(summary)

        if not summarized_messages:
            info_log(
                "Summarization produced no messages; falling back to recent messages only",
                context=_LOG_CONTEXT,
            )
            return system_messages + recent_messages

        info_log(
            f"Summarization complete: {len(old_messages)} old messages → "
            f"{len(summarized_messages)} summarized messages",
            context=_LOG_CONTEXT,
        )

        return system_messages + summarized_messages + recent_messages



    def _build_context_full_response(
        self,
        model_name: Optional[str] = None,
    ) -> "ModelResponse":
        """Build a fixed ModelResponse indicating the context window is full."""
        from upsonic._utils import now_utc
        from upsonic.messages import ModelResponse, TextPart

        return ModelResponse(
            parts=[TextPart(content=CONTEXT_FULL_MESSAGE)],
            model_name=model_name,
            timestamp=now_utc(),
            finish_reason="length",
        )

    async def apply(
        self,
        messages: List["ModelMessage"],
    ) -> tuple[List["ModelMessage"], bool]:
        """Apply context management strategies to messages.

        Checks if the context window is exceeded and applies strategies in order:
        1. If tool calls exist, prune old tool call history first.
        2. Summarize old messages via LLM (independent of whether pruning occurred).
        3. If still exceeded, return a context_full flag.

        Args:
            messages: The current message list (will NOT be mutated).

        Returns:
            A tuple of (processed_messages, context_full).
            If context_full is True, the caller should stop processing and
            return a context-full response.
        """
        from upsonic.utils.printing import info_log

        if not self._is_context_exceeded(messages):
            return list(messages), False

        estimated_tokens: int = self._estimate_message_tokens(messages)
        max_window: Optional[int] = self._get_max_context_window()
        info_log(
            f"Context window exceeded: ~{estimated_tokens:,} tokens estimated, "
            f"limit {int(max_window * self.safety_margin_ratio):,} "
            f"(model window {max_window:,} × {self.safety_margin_ratio}). "
            f"Applying compression strategies...",
            context=_LOG_CONTEXT,
        )

        current_messages: List["ModelMessage"] = list(messages)

        # Step 1: Prune tool call history (only if tool calls exist)
        has_tools: bool = self._has_tool_related_messages(current_messages)
        if has_tools:
            pruned: List["ModelMessage"] = self._prune_tool_call_history(current_messages)
            pruned_count: int = len(current_messages) - len(pruned)
            if pruned_count > 0:
                info_log(
                    f"Step 1 — Tool pruning: removed {pruned_count} old tool-related "
                    f"messages (kept {self.keep_recent_count} recent)",
                    context=_LOG_CONTEXT,
                )
                current_messages = pruned

                if not self._is_context_exceeded(current_messages):
                    info_log(
                        "Context within limits after tool pruning. No further action needed.",
                        context=_LOG_CONTEXT,
                    )
                    return current_messages, False
            else:
                info_log(
                    f"Step 1 — Tool pruning: no old tool messages to remove "
                    f"(all {len(current_messages)} messages within keep_recent_count={self.keep_recent_count})",
                    context=_LOG_CONTEXT,
                )
        else:
            info_log(
                "Step 1 — Tool pruning: skipped (no tool call/return parts in messages)",
                context=_LOG_CONTEXT,
            )

        # Step 2: Summarize old messages via LLM
        info_log(
            "Step 2 — Summarizing old messages via LLM...",
            context=_LOG_CONTEXT,
        )
        summarized: List["ModelMessage"] = await self._summarize_old_messages(current_messages)

        if not self._is_context_exceeded(summarized):
            info_log(
                "Context within limits after summarization. Compression successful.",
                context=_LOG_CONTEXT,
            )
            return summarized, False

        # Step 3: Context is still full — signal to caller
        remaining_tokens: int = self._estimate_message_tokens(summarized)
        info_log(
            f"Step 3 — Context still exceeded after all strategies: "
            f"~{remaining_tokens:,} tokens remaining vs limit "
            f"{int(max_window * self.safety_margin_ratio):,}. Signaling context_full.",
            context=_LOG_CONTEXT,
        )
        return summarized, True
