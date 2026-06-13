"""Per-run mutable agent state, isolated per asyncio task via a ContextVar.

A single :class:`~upsonic.agent.agent.Agent` instance can drive many runs at
once (e.g. ``asyncio.gather(agent.do_async(a), agent.do_async(b))``). The
run-scoped fields below used to live directly on the agent instance, so
concurrent runs clobbered each other's ``run_id`` / output / tool counters.

The fix keeps those fields addressable as ``self.<attr>`` (the agent exposes
properties that delegate here) but stores the actual values in an
:class:`AgentRunState` held by a :class:`contextvars.ContextVar`. Because
:func:`asyncio.ensure_future` copies the current context for each task, every
run gets its own isolated state object without locks or throughput loss.

When no run scope is active (before a run, or after it returns), the agent
delegates to a per-instance *fallback* state so post-run reads such as
``agent.get_run_output()`` keep working — the run's ``finally`` snapshots its
state into that fallback before the scope is torn down.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AgentRunState:
    """Mutable state that belongs to a single agent execution."""

    agent_run_output: Optional[Any] = None
    run_id: Optional[str] = None
    tool_call_count: int = 0
    tool_limit_reached: bool = False


# Holds the state of the run executing in the current async context, or None
# when no run scope is active (delegate to the agent's fallback state).
_current_run_state: ContextVar[Optional[AgentRunState]] = ContextVar(
    "upsonic_agent_run_state", default=None
)


def get_run_state() -> Optional[AgentRunState]:
    """Return the run state for the current async context, or None."""
    return _current_run_state.get()


def set_run_state(state: AgentRunState) -> Token:
    """Bind a fresh run state to the current context; returns a reset token."""
    return _current_run_state.set(state)


def reset_run_state(token: Token) -> None:
    """Restore the previous run state using a token from :func:`set_run_state`."""
    _current_run_state.reset(token)
