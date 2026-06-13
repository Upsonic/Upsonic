"""num_last_messages history-limiting (QA bug batch F9).

The old grouping paired messages by even/odd index, which silently broke the
moment a turn contained tool-call round-trips (a turn is then >2 messages):
all_runs undercounted, fell under the limit, and the full history leaked back —
so the agent still "remembered" the first message. The fix groups by user-turn
boundary so the limit counts actual user turns regardless of tool round-trips.
"""
from types import SimpleNamespace
from typing import Any, List

from upsonic.storage.memory.session.agent import AgentSessionMemory
from upsonic.messages.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)


def _limit(history: List[Any], n: int) -> List[Any]:
    stub = SimpleNamespace(num_last_messages=n, debug=False)
    return AgentSessionMemory._limit_message_history(stub, history)


def _user_req(text: str, with_system: bool = False) -> ModelRequest:
    parts: List[Any] = []
    if with_system:
        parts.append(SystemPromptPart(content="SYSTEM"))
    parts.append(UserPromptPart(content=text))
    return ModelRequest(parts=parts)


def _text_resp(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _tool_call_resp() -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name="calc", args={"a": 1})])


def _tool_return_req() -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name="calc", content="2")])


def _user_contents(msgs: List[Any]) -> List[Any]:
    out = []
    for m in msgs:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart):
                    out.append(p.content)
    return out


def _has_system(msgs: List[Any]) -> bool:
    return any(
        isinstance(m, ModelRequest) and any(isinstance(p, SystemPromptPart) for p in m.parts)
        for m in msgs
    )


def test_plain_history_keeps_last_n_turns():
    history: List[Any] = []
    for i in range(10):
        history.append(_user_req(f"msg-{i}", with_system=(i == 0)))
        history.append(_text_resp(f"resp-{i}"))

    result = _limit(history, 3)
    contents = _user_contents(result)
    assert "msg-0" not in contents
    assert contents == ["msg-7", "msg-8", "msg-9"]
    # The system prompt is re-injected into the first kept turn.
    assert _has_system(result)


def test_tool_round_trip_history_still_limits():
    # The bug case: each turn has a tool call + tool return (4 messages/turn).
    history: List[Any] = []
    for i in range(5):
        history.append(_user_req(f"msg-{i}", with_system=(i == 0)))
        history.append(_tool_call_resp())
        history.append(_tool_return_req())
        history.append(_text_resp(f"resp-{i}"))

    result = _limit(history, 2)
    contents = _user_contents(result)
    assert "msg-0" not in contents, "first turn leaked despite num_last_messages=2"
    assert contents == ["msg-3", "msg-4"]


def test_under_limit_returns_full_history_unchanged():
    history: List[Any] = []
    for i in range(2):
        history.append(_user_req(f"msg-{i}", with_system=(i == 0)))
        history.append(_text_resp(f"resp-{i}"))

    result = _limit(history, 5)
    assert result is history  # early return, untouched


def test_leading_orphan_preamble_attaches_to_first_run():
    # A leading non-user-prompt request (e.g. a summary/tool preamble) must not
    # become its own counted run; it belongs to the first real turn and is
    # dropped with it under trimming.
    history: List[Any] = [_tool_return_req()]  # orphan preamble, no user prompt
    for i in range(4):
        history.append(_user_req(f"msg-{i}", with_system=(i == 0)))
        history.append(_text_resp(f"resp-{i}"))

    result = _limit(history, 2)
    contents = _user_contents(result)
    assert contents == ["msg-2", "msg-3"]
