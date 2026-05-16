"""Unit tests for the framework-injected tool catalog and count-exemption helper.

Covers Scenario 9a (count-exemption math) at the helper level. The integrated
behavior (where the increment sites use the helper) is exercised by the
streaming smoke tests.
"""

from __future__ import annotations

import pytest


def test_sentinel_tool_is_exempt():
    from upsonic.tools.framework_tools import (
        FINAL_ANSWER_MARKER_TOOL_NAME,
        is_count_exempt_tool,
    )

    assert is_count_exempt_tool(FINAL_ANSWER_MARKER_TOOL_NAME) is True


def test_regular_tool_is_not_exempt():
    from upsonic.tools.framework_tools import is_count_exempt_tool

    assert is_count_exempt_tool("my_user_tool") is False
    assert is_count_exempt_tool("calculate_discount") is False


def test_structured_output_tool_is_not_exempt():
    """DEFAULT_OUTPUT_TOOL_NAME (currently 'final_result') is a real call.

    It must NOT be in the exempt set — structured output is user-attributable
    work and should count toward tool_call_limit and total_tool_calls.
    """
    from upsonic.output import DEFAULT_OUTPUT_TOOL_NAME
    from upsonic.tools.framework_tools import is_count_exempt_tool

    assert is_count_exempt_tool(DEFAULT_OUTPUT_TOOL_NAME) is False


def test_empty_string_is_not_exempt():
    from upsonic.tools.framework_tools import is_count_exempt_tool

    assert is_count_exempt_tool("") is False


def test_sentinel_name_constant_value():
    from upsonic.tools.framework_tools import FINAL_ANSWER_MARKER_TOOL_NAME

    assert FINAL_ANSWER_MARKER_TOOL_NAME == "__final_answer_marker__"


def test_framework_set_is_frozen():
    from upsonic.tools.framework_tools import FRAMEWORK_INJECTED_TOOL_NAMES

    assert isinstance(FRAMEWORK_INJECTED_TOOL_NAMES, frozenset)
    with pytest.raises(AttributeError):
        # frozenset has no add method
        FRAMEWORK_INJECTED_TOOL_NAMES.add("foo")  # type: ignore[attr-defined]


def test_sentinel_in_framework_set():
    from upsonic.tools.framework_tools import (
        FINAL_ANSWER_MARKER_TOOL_NAME,
        FRAMEWORK_INJECTED_TOOL_NAMES,
    )

    assert FINAL_ANSWER_MARKER_TOOL_NAME in FRAMEWORK_INJECTED_TOOL_NAMES


# ---------------------------------------------------------------------------
# build_final_answer_marker_tool_definition
# ---------------------------------------------------------------------------


def test_tool_definition_has_correct_name():
    from upsonic.tools.final_answer_marker import (
        FINAL_ANSWER_MARKER_TOOL_NAME,
        build_final_answer_marker_tool_definition,
    )

    td = build_final_answer_marker_tool_definition()
    assert td.name == FINAL_ANSWER_MARKER_TOOL_NAME


def test_tool_definition_has_argless_schema():
    from upsonic.tools.final_answer_marker import (
        build_final_answer_marker_tool_definition,
    )

    td = build_final_answer_marker_tool_definition()
    schema = td.parameters_json_schema
    assert schema["type"] == "object"
    assert schema["properties"] == {}
    # No properties → empty `required` (or absent)
    assert schema.get("required", []) == []


def test_directive_mentions_tool_name():
    """The directive must reference the canonical tool name verbatim so the
    model can call it correctly when reading the system prompt.
    """
    from upsonic.tools.final_answer_marker import (
        FINAL_ANSWER_MARKER_DIRECTIVE,
        FINAL_ANSWER_MARKER_TOOL_NAME,
    )

    assert FINAL_ANSWER_MARKER_TOOL_NAME in FINAL_ANSWER_MARKER_DIRECTIVE
