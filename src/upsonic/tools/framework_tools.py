"""Framework-injected tool catalog and shared helpers.

Single source of truth for tools that the framework adds to a run on the
agent's behalf (rather than as user-defined tools). These tools should
NOT count against ``tool_call_limit`` or appear in
``ExecutionCompleteEvent.total_tool_calls`` — they are framework
plumbing, not user-attributable work.

To add a new framework-injected count-exempt tool, append its canonical
name to ``FRAMEWORK_INJECTED_TOOL_NAMES`` and the three
``tool_call_count`` increment sites in ``upsonic.agent.agent`` will
automatically skip it via :func:`is_count_exempt_tool`.

Note: The structured-output tool (``upsonic.output.DEFAULT_OUTPUT_TOOL_NAME``,
currently ``'final_result'``) is intentionally NOT in this set. It is a
real, count-bearing call that the agent uses to deliver structured output
to the user.
"""

from __future__ import annotations

# Canonical tool name for the streaming-final-answer marker.
FINAL_ANSWER_MARKER_TOOL_NAME: str = "__final_answer_marker__"

# All framework-injected tools whose calls should be exempt from
# tool_call_limit accounting and from total_tool_calls metrics.
FRAMEWORK_INJECTED_TOOL_NAMES: frozenset[str] = frozenset({
    FINAL_ANSWER_MARKER_TOOL_NAME,
})


def is_count_exempt_tool(tool_name: str) -> bool:
    """Return True if ``tool_name`` is framework-injected and must not be counted.

    Called from every ``self._tool_call_count`` increment site in
    ``upsonic.agent.agent`` to guard the increment.
    """
    return tool_name in FRAMEWORK_INJECTED_TOOL_NAMES


__all__ = [
    "FINAL_ANSWER_MARKER_TOOL_NAME",
    "FRAMEWORK_INJECTED_TOOL_NAMES",
    "is_count_exempt_tool",
]
