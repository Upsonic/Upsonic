"""Unit tests for FinalAnswerStartEvent shape and registration.

Covers the event-scaffolding-tier guarantees:
  - dataclass shape and defaults
  - AgentRunEvent enum membership
  - _EVENT_CLASS_REGISTRY entry
  - StepSpecificEvent / AgentStreamEvent / LLMStreamEvent union membership
  - public re-exports from upsonic.agent and upsonic.run.events
"""

from __future__ import annotations

import pytest


def test_event_kind_is_final_answer_start():
    from upsonic.run.events.events import FinalAnswerStartEvent

    event = FinalAnswerStartEvent(run_id="r1")
    assert event.event_kind == "final_answer_start"


def test_event_default_trigger_is_sentinel():
    from upsonic.run.events.events import FinalAnswerStartEvent

    event = FinalAnswerStartEvent(run_id="r1")
    assert event.triggered_by == "sentinel"


@pytest.mark.parametrize("trigger", ["sentinel", "cache_hit", "output_tool"])
def test_event_accepts_all_triggers(trigger):
    from upsonic.run.events.events import FinalAnswerStartEvent

    event = FinalAnswerStartEvent(run_id="r1", triggered_by=trigger)
    assert event.triggered_by == trigger


def test_event_type_property_reflects_class_name():
    from upsonic.run.events.events import FinalAnswerStartEvent

    event = FinalAnswerStartEvent(run_id="r1")
    # AgentEvent.event_type is a property returning the class name.
    assert event.event_type == "FinalAnswerStartEvent"


def test_event_inherits_base_agent_event_fields():
    from upsonic.run.events.events import AgentEvent, FinalAnswerStartEvent

    event = FinalAnswerStartEvent(run_id="r1")
    assert isinstance(event, AgentEvent)
    assert event.run_id == "r1"
    assert event.event_id  # auto-populated
    assert event.timestamp is not None


# ---------------------------------------------------------------------------
# Enum membership
# ---------------------------------------------------------------------------


def test_enum_has_final_answer_start_member():
    from upsonic.run.events.events import AgentRunEvent

    assert AgentRunEvent.FINAL_ANSWER_START.value == "final_answer_start"


def test_enum_member_matches_event_kind():
    from upsonic.run.events.events import AgentRunEvent, FinalAnswerStartEvent

    event = FinalAnswerStartEvent(run_id="r1")
    assert event.event_kind == AgentRunEvent.FINAL_ANSWER_START.value


# ---------------------------------------------------------------------------
# Registry membership (used by deserialization)
# ---------------------------------------------------------------------------


def test_registered_in_event_class_registry():
    from upsonic.run.events.events import _EVENT_CLASS_REGISTRY, FinalAnswerStartEvent

    assert "FinalAnswerStartEvent" in _EVENT_CLASS_REGISTRY
    assert _EVENT_CLASS_REGISTRY["FinalAnswerStartEvent"] is FinalAnswerStartEvent


# ---------------------------------------------------------------------------
# Union membership (StepSpecificEvent / AgentStreamEvent / LLMStreamEvent)
# ---------------------------------------------------------------------------


def test_member_of_agent_stream_event_union():
    """Ensure FinalAnswerStartEvent dispatches correctly through the public union."""
    from upsonic.run.events.events import (
        AgentStreamEventTypeAdapter,
        FinalAnswerStartEvent,
    )

    event = FinalAnswerStartEvent(run_id="r1", triggered_by="sentinel")
    dumped = event.to_dict()
    # Round-trip through the discriminated-union adapter
    adapter_result = AgentStreamEventTypeAdapter.validate_python(dumped)
    assert isinstance(adapter_result, FinalAnswerStartEvent)
    assert adapter_result.triggered_by == "sentinel"


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


def test_reexported_from_upsonic_run_events():
    from upsonic.run.events import FinalAnswerStartEvent as FAE

    event = FAE(run_id="r1")
    assert event.event_kind == "final_answer_start"


def test_reexported_from_upsonic_agent():
    from upsonic.agent import FinalAnswerStartEvent as FAE

    event = FAE(run_id="r1")
    assert event.event_kind == "final_answer_start"


def test_same_class_through_all_public_paths():
    from upsonic.agent import FinalAnswerStartEvent as A
    from upsonic.run.events import FinalAnswerStartEvent as B
    from upsonic.run.events.events import FinalAnswerStartEvent as C

    assert A is B is C


# ---------------------------------------------------------------------------
# Helper functions in utils.agent.events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ayield_final_answer_start_event_helper():
    from upsonic.utils.agent.events import ayield_final_answer_start_event
    from upsonic.run.events.events import FinalAnswerStartEvent

    events = []
    async for event in ayield_final_answer_start_event("r1", triggered_by="cache_hit"):
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], FinalAnswerStartEvent)
    assert events[0].triggered_by == "cache_hit"
    assert events[0].run_id == "r1"


def test_yield_final_answer_start_event_sync_helper():
    from upsonic.utils.agent.events import yield_final_answer_start_event
    from upsonic.run.events.events import FinalAnswerStartEvent

    events = list(yield_final_answer_start_event("r1", triggered_by="output_tool"))
    assert len(events) == 1
    assert isinstance(events[0], FinalAnswerStartEvent)
    assert events[0].triggered_by == "output_tool"
