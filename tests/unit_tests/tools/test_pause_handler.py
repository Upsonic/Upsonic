"""Unit tests for ``PauseHandler``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from upsonic.tools.config import ToolConfig
from upsonic.tools.hitl import (
    ConfirmationPause,
    ExternalExecutionPause,
    PauseHandler,
    PausedToolCall,
    UserInputPause,
)
from upsonic.tools.user_input import UserInputField


def _make_tool_obj(user_input_fields=None, func=None):
    tool = MagicMock()
    tool.config = ToolConfig(
        requires_user_input=True,
        user_input_fields=user_input_fields or [],
    )
    tool.function = func
    return tool


class TestPauseHandler:
    """Coverage for ``attach_paused_call`` over each pause type."""

    @pytest.fixture
    def handler(self) -> PauseHandler:
        return PauseHandler()

    # ------------------------------------------------------------------
    # ConfirmationPause
    # ------------------------------------------------------------------
    def test_attach_for_confirmation(self, handler):
        exc = ConfirmationPause()
        handler.attach_paused_call(
            exc,
            tool_name="t",
            args={"x": 1},
            tool_call_id="cid",
            tool_obj=MagicMock(),
        )
        assert len(exc.paused_calls) == 1
        pc = exc.paused_calls[0]
        assert isinstance(pc, PausedToolCall)
        assert pc.tool_name == "t"
        assert pc.tool_args == {"x": 1}
        assert pc.tool_call_id == "cid"
        assert pc.requires_confirmation is True

    # ------------------------------------------------------------------
    # ExternalExecutionPause
    # ------------------------------------------------------------------
    def test_attach_for_external_execution(self, handler):
        exc = ExternalExecutionPause()
        handler.attach_paused_call(
            exc,
            tool_name="ext",
            args={},
            tool_call_id="cid2",
            tool_obj=MagicMock(),
        )
        assert len(exc.paused_calls) == 1
        pc = exc.paused_calls[0]
        assert pc.tool_name == "ext"
        assert pc.tool_call_id == "cid2"
        assert pc.requires_confirmation is False
        assert pc.requires_user_input is False

    # ------------------------------------------------------------------
    # UserInputPause — branch (a): exc already carries schema
    # ------------------------------------------------------------------
    def test_attach_for_user_input_with_existing_schema(self, handler):
        schema = [UserInputField(name="email", field_type="str")]
        exc = UserInputPause(user_input_schema=schema)
        handler.attach_paused_call(
            exc,
            tool_name="ui",
            args={},
            tool_call_id="cid3",
            tool_obj=MagicMock(),
        )
        assert len(exc.paused_calls) == 1
        pc = exc.paused_calls[0]
        assert pc.requires_user_input is True
        # schema is preserved as field-list on exc and serialized to dicts on pc
        assert exc.user_input_schema == schema
        assert pc.user_input_schema == [{"name": "email", "field_type": "str", "description": None, "value": None}]

    # ------------------------------------------------------------------
    # UserInputPause — branch (b): build schema from tool function
    # ------------------------------------------------------------------
    def test_attach_for_user_input_builds_schema_from_function(self, handler):
        def some_tool(name: str, age: int) -> str:
            """Some tool."""
            return f"{name}/{age}"

        tool_obj = _make_tool_obj(user_input_fields=["age"], func=some_tool)

        exc = UserInputPause()  # no schema
        handler.attach_paused_call(
            exc,
            tool_name="some_tool",
            args={"name": "alice"},
            tool_call_id="cid4",
            tool_obj=tool_obj,
        )

        # exc.user_input_schema must now be a list of UserInputField
        assert exc.user_input_schema
        assert isinstance(exc.user_input_schema[0], UserInputField)

        pc = exc.paused_calls[0]
        # serialized form on the paused call
        names = [d["name"] for d in pc.user_input_schema]
        assert "name" in names
        assert "age" in names

    def test_does_not_re_raise(self, handler):
        """``attach_paused_call`` MUST NOT re-raise the exception."""
        exc = ConfirmationPause()
        # If this raises, the test fails by surfacing the exception.
        handler.attach_paused_call(
            exc,
            tool_name="t",
            args={},
            tool_call_id="cid",
            tool_obj=MagicMock(),
        )
