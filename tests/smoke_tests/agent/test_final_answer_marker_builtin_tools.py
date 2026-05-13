"""Builtin-tool smoke tests for ``stream_final_answer``.

Provider-side builtin tools (``WebSearchTool``, ``CodeExecutionTool``, etc.)
are passed to the model on a separate channel from function tools — they
do NOT appear in ``tool_definitions``. Initially the skip-when-no-user-tools
rule only consulted ``tool_definitions`` and incorrectly skipped sentinel
injection when the agent had only builtin tools.

This module exercises that scenario end-to-end to lock in the fix.
"""

from __future__ import annotations

import pytest

from _final_answer_marker_helpers import (
    Task,
    build_agent_anthropic,
    collect_events,
    has_anthropic,
    skip_if_no_provider,
)

pytestmark = skip_if_no_provider


@pytest.mark.asyncio
@pytest.mark.skipif(
    not has_anthropic(),
    reason="WebSearchTool is best supported by Anthropic Sonnet",
)
async def test_agent_with_only_builtin_tools_still_emits_marker():
    """An agent configured with ONLY builtin tools (e.g. WebSearchTool) and
    no regular function tools MUST still emit FinalAnswerStartEvent when
    stream_final_answer=True. Builtin tools are user-attributable work and
    should count toward the skip-when-no-user-tools rule.

    Originally this scenario produced 0 markers because:
      - ``agent.py:_build_model_request_parameters`` counted only
        ``tool_definitions`` (function tools)
      - ``system_prompt_manager.py:aprepare`` had the same gap

    The fix counts ``len(builtin_tools)`` in both sites.
    """
    from upsonic.run.events.events import FinalAnswerStartEvent, FinalOutputEvent
    from upsonic.tools.builtin_tools import WebSearchTool

    agent = build_agent_anthropic(
        name="BuiltinOnlyAgent",
        tools=[WebSearchTool()],
    )

    assert hasattr(agent, "agent_builtin_tools"), (
        "Agent must expose agent_builtin_tools attribute"
    )
    assert len(agent.agent_builtin_tools) >= 1, (
        "Setup failure: WebSearchTool not registered as a builtin tool"
    )

    task = Task(
        description="Briefly tell me about Upsonic AI framework in one sentence. "
                    "Use web search if needed."
    )

    events = await collect_events(agent, task, stream_final_answer=True)

    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1, "Run did not complete"

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Builtin-only agent must emit exactly one marker. "
        f"Got {len(fa_starts)}. If 0, the skip-when-no-user-tools rule is "
        f"again excluding builtin tools — fix in agent.py and "
        f"system_prompt_manager.py."
    )
