"""Process-wide scope tag plumbing for the usage registry.

Phase 1c lays the rails the Phase-2 emission point will ride on. When a
Chat / Agent / Task / Team enters its execution body, it pushes its own
``*_usage_id`` onto the matching :class:`contextvars.ContextVar`. The
model-call site at the bottom of the pipeline then reads
:func:`current_scope_tags` to learn every active scope.

ContextVars are inherited across ``asyncio`` tasks and threads via
``copy_context()`` automatically, so a sub-agent / tool call spawned
underneath ``Agent.do_async`` inherits the caller's chat / task scope
without any explicit plumbing — except where it should NOT (e.g. a
nested sub-agent run that wants its own agent scope just calls
:func:`scope` again and the previous value is restored when the block
exits).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional


# One ContextVar per scope. Default ``None`` means "no such scope active".
_chat_usage_id: ContextVar[Optional[str]] = ContextVar(
    "upsonic_chat_usage_id", default=None
)
_agent_usage_id: ContextVar[Optional[str]] = ContextVar(
    "upsonic_agent_usage_id", default=None
)
_task_usage_id: ContextVar[Optional[str]] = ContextVar(
    "upsonic_task_usage_id", default=None
)
_team_usage_id: ContextVar[Optional[str]] = ContextVar(
    "upsonic_team_usage_id", default=None
)
_workflow_usage_id: ContextVar[Optional[str]] = ContextVar(
    "upsonic_workflow_usage_id", default=None
)
_system_usage_id: ContextVar[Optional[str]] = ContextVar(
    "upsonic_system_usage_id", default=None
)

_run_id: ContextVar[Optional[str]] = ContextVar(
    "upsonic_run_id", default=None
)
_user_id: ContextVar[Optional[str]] = ContextVar(
    "upsonic_user_id", default=None
)


def current_scope_tags() -> Dict[str, Optional[str]]:
    """Snapshot every active scope id at the call site.

    Returned dict is safe to pass directly as ``**kwargs`` into
    :class:`upsonic.usage_registry.UsageEntry` — keys match.
    Values that are ``None`` mean the corresponding scope is not
    currently active and will be left as ``None`` on the entry.
    """
    return {
        "chat_usage_id": _chat_usage_id.get(),
        "agent_usage_id": _agent_usage_id.get(),
        "task_usage_id": _task_usage_id.get(),
        "team_usage_id": _team_usage_id.get(),
        "workflow_usage_id": _workflow_usage_id.get(),
        "system_usage_id": _system_usage_id.get(),
        "run_id": _run_id.get(),
        "user_id": _user_id.get(),
    }


@contextmanager
def scope(
    *,
    chat_usage_id: Optional[str] = None,
    agent_usage_id: Optional[str] = None,
    task_usage_id: Optional[str] = None,
    team_usage_id: Optional[str] = None,
    workflow_usage_id: Optional[str] = None,
    system_usage_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Iterator[None]:
    """Push the given non-``None`` scope ids for the duration of the block.

    Any kwarg left as ``None`` keeps the previously-active value (it is
    NOT cleared). Existing values are restored on exit even if an
    exception propagates.

    Typical use::

        with scope(agent_usage_id=self.agent_usage_id):
            ...  # every UsageEntry recorded here inherits agent_usage_id

    Nesting works the way contextvars do — the innermost push wins, and
    when the block exits the previous value is reinstated.
    """
    tokens = []
    if chat_usage_id is not None:
        tokens.append((_chat_usage_id, _chat_usage_id.set(chat_usage_id)))
    if agent_usage_id is not None:
        tokens.append((_agent_usage_id, _agent_usage_id.set(agent_usage_id)))
    if task_usage_id is not None:
        tokens.append((_task_usage_id, _task_usage_id.set(task_usage_id)))
    if team_usage_id is not None:
        tokens.append((_team_usage_id, _team_usage_id.set(team_usage_id)))
    if workflow_usage_id is not None:
        tokens.append((_workflow_usage_id, _workflow_usage_id.set(workflow_usage_id)))
    if system_usage_id is not None:
        tokens.append((_system_usage_id, _system_usage_id.set(system_usage_id)))
    if run_id is not None:
        tokens.append((_run_id, _run_id.set(run_id)))
    if user_id is not None:
        tokens.append((_user_id, _user_id.set(user_id)))

    try:
        yield
    finally:
        # Restore in reverse order so nested scope() calls unwind cleanly.
        for var, token in reversed(tokens):
            var.reset(token)


def push_scope_tags(
    *,
    chat_usage_id: Optional[str] = None,
    agent_usage_id: Optional[str] = None,
    task_usage_id: Optional[str] = None,
    team_usage_id: Optional[str] = None,
    workflow_usage_id: Optional[str] = None,
    system_usage_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    inherit: bool = False,
) -> list:
    """Imperative variant of :func:`scope` for entry / exit blocks that
    span more code than a ``with`` block can wrap cleanly (e.g. the
    long body of ``Agent.do_async`` / ``Agent.astream``).

    Pushes every non-``None`` tag onto the matching contextvar and
    returns a list of ``(ContextVar, Token)`` pairs that the caller
    feeds back to :func:`reset_scope_tags` in a ``finally`` block.

    Args:
        inherit: When ``True``, a tag that's already active in the
            current context is left untouched (the contextvar is
            NOT re-set). This is the "sub-agent inherits parent's
            scope" pattern — caller passes its own id, but a nested
            run sees the existing value and keeps it.
    """
    pairs = [
        (_chat_usage_id, chat_usage_id),
        (_agent_usage_id, agent_usage_id),
        (_task_usage_id, task_usage_id),
        (_team_usage_id, team_usage_id),
        (_workflow_usage_id, workflow_usage_id),
        (_system_usage_id, system_usage_id),
        (_run_id, run_id),
        (_user_id, user_id),
    ]
    tokens: list = []
    for var, value in pairs:
        if value is None:
            continue
        if inherit and var.get() is not None:
            continue
        tokens.append((var, var.set(value)))
    return tokens


def reset_scope_tags(tokens: list) -> None:
    """Reset every token returned by :func:`push_scope_tags`. Safe to
    call with a partial / interleaved list; failures on individual
    tokens are swallowed so a finally block can run to completion."""
    for var, token in reversed(tokens):
        try:
            var.reset(token)
        except Exception:
            pass
