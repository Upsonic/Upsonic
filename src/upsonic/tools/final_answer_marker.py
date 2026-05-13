"""Framework-injected sentinel tool that marks the start of the final answer.

Used by the ``stream_final_answer=True`` opt-in flag on ``Agent.astream`` /
``Agent.stream`` (and the corresponding ``Chat`` streaming methods). When
the flag is active and the agent has at least one user tool registered,
this tool's ``ToolDefinition`` is injected per-run into the model request
and a one-sentence directive is appended to the system prompt instructing
the model to call it once, immediately before producing the final
user-visible response.

In the streaming pipeline (``upsonic.agent.pipeline.steps``), the first
``ToolCallDeltaEvent`` whose ``tool_name == FINAL_ANSWER_MARKER_TOOL_NAME``
triggers a single prospective ``FinalAnswerStartEvent(triggered_by='sentinel')``.
The sentinel call itself is silently dropped — no ``ToolCallEvent`` or
``ToolResultEvent`` is emitted for it, and it does not count against
``tool_call_limit`` (see :mod:`upsonic.tools.framework_tools`).
"""

from __future__ import annotations

from typing import Any, Dict

from upsonic.tools.base import ToolDefinition
from upsonic.tools.framework_tools import FINAL_ANSWER_MARKER_TOOL_NAME

# Single source of truth for the directive appended to the system prompt
# when the flag is active. The wording is intentionally explicit about
# the two-stage behavior the consumer relies on (call once, then emit
# the final answer as continuous text with no further tool calls).
FINAL_ANSWER_MARKER_DIRECTIVE: str = (
    "\n\nIMPORTANT — final-answer streaming protocol:\n"
    "Before producing the final user-visible response, you MUST call the "
    f"`{FINAL_ANSWER_MARKER_TOOL_NAME}` tool exactly once with no arguments. "
    "Everything you emit AFTER that call is the final answer; emit it as a "
    "single continuous text response and make no further tool calls. "
    "If you still need more information or more tool calls, do NOT call "
    f"`{FINAL_ANSWER_MARKER_TOOL_NAME}` yet — only call it once you are "
    "ready to produce the definitive final answer."
)


# JSON schema for the tool — empty object, no properties, no required keys.
# Models can call it with `{}`.
_FINAL_ANSWER_MARKER_PARAMETERS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def _final_answer_marker_executor(**_kwargs: Any) -> str:
    """No-op executor.

    The pipeline never actually invokes this — sentinel calls are detected
    in ``_stream_with_tool_calls`` and short-circuited. This implementation
    is a safety net: if a code path ever does reach the executor (for
    example, a future contributor disables the suppression branch), the
    function returns an empty string and the model can continue to its
    next turn.
    """
    return ""


def build_final_answer_marker_tool_definition() -> ToolDefinition:
    """Build the per-run ``ToolDefinition`` for the sentinel tool.

    Called from ``Agent._build_tool_definitions`` when
    ``run-context.stream_final_answer`` is truthy and the agent has at
    least one user tool registered. Never persisted in ``ToolRegistry``;
    the lifecycle is bounded by a single ``astream``/``stream`` call.
    """
    return ToolDefinition(
        name=FINAL_ANSWER_MARKER_TOOL_NAME,
        description=(
            "Internal marker. Call this tool exactly once, with no "
            "arguments, immediately before you produce the final "
            "user-visible response. After calling it, emit the final "
            "answer as continuous text and make no further tool calls."
        ),
        parameters_json_schema=_FINAL_ANSWER_MARKER_PARAMETERS_SCHEMA,
    )


__all__ = [
    "FINAL_ANSWER_MARKER_TOOL_NAME",
    "FINAL_ANSWER_MARKER_DIRECTIVE",
    "build_final_answer_marker_tool_definition",
]
