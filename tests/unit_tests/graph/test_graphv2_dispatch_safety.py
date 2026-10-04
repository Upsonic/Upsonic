"""Execution receipts must survive recovery rather than becoming replay permission."""

import asyncio
import operator
import threading
from typing import Annotated, TypedDict

import pytest
from upsonic.graphv2 import (
    END,
    START,
    Command,
    MemorySaver,
    Send,
    StateGraph,
    interrupt,
)
from upsonic.graphv2.checkpoint import Checkpoint
from upsonic.graphv2.errors import GraphExecutionUncertainError
from upsonic.graphv2.task import RetryPolicy


class State(TypedDict):
    count: int
    results: Annotated[list, operator.add]


CONFIG = {"configurable": {"thread_id": "dispatch-test"}}


def graph_for(body, saver=None, **node_options):
    builder = StateGraph(State)
    builder.add_node("effect", body, **node_options)
    builder.add_edge(START, "effect")
    builder.add_edge("effect", END)
    return builder.compile(checkpointer=saver)


def test_normal_execution_has_distinct_durable_receipt():
    saver = MemorySaver()
    calls = []
    def effect(state, config):
        calls.append(config["configurable"]["node_execution_id"])
        return {"count": 1}
    graph = graph_for(effect, saver)
    assert graph.invoke({"count": 0}, CONFIG)["count"] == 1
    assert graph.invoke({"count": 0}, CONFIG)["count"] == 1
    assert len(set(calls)) == 2
    assert saver.get("dispatch-test").pending_tasks == []


@pytest.mark.asyncio
async def test_async_execution_records_identity():
    ids = []
    saver = MemorySaver()
    async def effect(state, config):
        ids.append(config["configurable"]["node_execution_id"])
        pending = saver.get("dispatch-test").pending_tasks
        assert pending[0]["id"] == ids[0]
        assert pending[0]["status"] == "DISPATCH_RECORDED"
        return {"count": 1}
    graph = graph_for(effect, saver)
    assert (await graph.ainvoke({"count": 0}, CONFIG))["count"] == 1
    assert len(ids) == 1


def test_unsafe_node_local_retry_stops_after_first_attempt():
    effects = []
    def effect(state):
        effects.append(1)
        raise TimeoutError("ack lost")
    graph = graph_for(effect, MemorySaver(), retry_policy=RetryPolicy(max_attempts=3, initial_interval=0))
    with pytest.raises(GraphExecutionUncertainError) as failure:
        graph.invoke({"count": 0}, CONFIG)
    assert type(failure.value).__name__ == "GraphExecutionUncertainError"
    assert effects == [1]
    with pytest.raises(GraphExecutionUncertainError) as recovery:
        graph.invoke(None, CONFIG)
    assert type(recovery.value).__name__ == "GraphExecutionUncertainError"
    assert effects == [1]


def test_safe_node_retry_is_explicit():
    calls = []
    def lookup(state, config):
        calls.append(config["configurable"]["node_execution_id"])
        if len(calls) == 1:
            raise TimeoutError("read failed")
        return {"count": 1}
    graph = graph_for(lookup, MemorySaver(), retry_safe=True,
                      retry_policy=RetryPolicy(max_attempts=2, initial_interval=0))
    assert graph.invoke({"count": 0}, CONFIG)["count"] == 1
    assert len(calls) == 2
    assert len(set(calls)) == 1


@pytest.mark.asyncio
async def test_completed_parallel_sibling_is_not_replayed():
    calls = {"good": 0, "bad": 0}
    saver = MemorySaver()
    builder = StateGraph(State)
    async def good(state):
        calls["good"] += 1
        return {"results": ["receipt"]}
    async def bad(state):
        calls["bad"] += 1
        await asyncio.sleep(0.01)
        raise TimeoutError("lost ack")
    builder.add_node("good", good)
    builder.add_node("bad", bad)
    for name in calls:
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    graph = builder.compile(checkpointer=saver)
    with pytest.raises(GraphExecutionUncertainError):
        await graph.ainvoke({"results": []}, CONFIG)
    checkpoint = saver.get("dispatch-test")
    assert len({t["id"] for t in checkpoint.pending_tasks}) == 2
    assert next(t for t in checkpoint.pending_tasks if t["node"] == "good")["status"] == "COMPLETED"
    with pytest.raises(GraphExecutionUncertainError) as recovery:
        await graph.ainvoke(None, CONFIG)
    assert type(recovery.value).__name__ == "GraphExecutionUncertainError"
    assert calls == {"good": 1, "bad": 1}


def test_send_identical_targets_have_distinct_identities():
    ids = []
    builder = StateGraph(State)
    builder.add_node("route", lambda state: {})
    def worker(state, config):
        ids.append(config["configurable"]["node_execution_id"])
        return {"results": [state["item"]]}
    builder.add_node("worker", worker)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", lambda state: [Send("worker", {"item": "same"})] * 2, ["worker"])
    builder.add_edge("worker", END)
    result = builder.compile(checkpointer=MemorySaver()).invoke({"results": []}, CONFIG)
    assert result["results"] == ["same", "same"]
    assert len(set(ids)) == 2


def test_dispatch_write_failure_prevents_body():
    effects = []
    class BrokenSaver(MemorySaver):
        def put(self, checkpoint):
            if any(t["status"] == "DISPATCH_RECORDED" for t in checkpoint.pending_tasks or []):
                raise OSError("disk full")
            super().put(checkpoint)
    graph = graph_for(lambda state: effects.append(1) or {}, BrokenSaver())
    with pytest.raises(OSError):
        graph.invoke({}, CONFIG)
    assert effects == []


def test_inside_interrupt_retains_uncertainty():
    effects = []
    saver = MemorySaver()
    def effect(state):
        effects.append(1)
        interrupt("approval")
        return {}
    graph = graph_for(effect, saver)
    assert "__interrupt__" in graph.invoke({}, CONFIG)
    ids = [t["id"] for t in saver.get("dispatch-test").pending_tasks]
    with pytest.raises(GraphExecutionUncertainError) as recovery:
        graph.invoke(Command(resume=True), CONFIG)
    assert type(recovery.value).__name__ == "GraphExecutionUncertainError"
    assert effects == [1]
    assert [t["id"] for t in saver.get("dispatch-test").pending_tasks] == ids


def test_legacy_checkpoint_does_not_prove_no_dispatch():
    saver = MemorySaver()
    saver.put(Checkpoint("legacy", "dispatch-test", {}, ["effect"]))
    effects = []
    graph = graph_for(lambda state: effects.append(1) or {}, saver)
    with pytest.raises(GraphExecutionUncertainError) as failure:
        graph.invoke(None, CONFIG)
    assert type(failure.value).__name__ == "GraphExecutionUncertainError"
    assert effects == []


def test_persisted_not_started_can_safely_dispatch():
    saver = MemorySaver()
    saver.put(Checkpoint("planned", "dispatch-test", {}, ["effect"], pending_tasks=[
        {"id": "invocation-1", "node": "effect", "state": {}, "status": "NOT_STARTED"}
    ]))
    ids = []
    def effect(state, config):
        ids.append(config["configurable"]["node_execution_id"])
        return {"count": 1}
    assert graph_for(effect, saver).invoke(None, CONFIG)["count"] == 1
    assert ids == ["invocation-1"]


def test_explicit_safe_recovery_preserves_invocation_id():
    saver = MemorySaver()
    saver.put(Checkpoint("uncertain", "dispatch-test", {}, ["effect"], pending_tasks=[
        {"id": "invocation-1", "node": "effect", "state": {}, "status": "DISPATCH_RECORDED"}
    ]))
    ids = []
    def lookup(state, config):
        ids.append(config["configurable"]["node_execution_id"])
        return {"count": 1}
    assert graph_for(lookup, saver, retry_safe=True).invoke(None, CONFIG)["count"] == 1
    assert ids == ["invocation-1"]


def test_cache_hit_does_not_dispatch_node_body():
    from upsonic.graphv2.cache import CachePolicy, InMemoryCache
    effects = []
    saver = MemorySaver()
    cache = InMemoryCache()
    builder = StateGraph(State)
    builder.add_node("effect", lambda state: effects.append(1) or {"count": 1},
                     cache_policy=CachePolicy(key_func=lambda state: "constant"))
    builder.add_edge(START, "effect")
    builder.add_edge("effect", END)
    graph = builder.compile(checkpointer=saver, cache=cache)
    graph.invoke({}, CONFIG)
    graph.invoke({}, CONFIG)
    assert effects == [1]
    assert saver.get("dispatch-test").pending_tasks == []


def test_completion_write_failure_keeps_dispatch_uncertain():
    effects = []
    class BrokenSaver(MemorySaver):
        def put(self, checkpoint):
            if any(t["status"] == "COMPLETED" for t in checkpoint.pending_tasks or []):
                raise OSError("receipt write failed")
            super().put(checkpoint)
    saver = BrokenSaver()
    graph = graph_for(lambda state: effects.append(1) or {}, saver)
    with pytest.raises(OSError):
        graph.invoke({}, CONFIG)
    assert saver.get("dispatch-test").pending_tasks[0]["status"] in {"DISPATCH_RECORDED", "UNKNOWN"}
    with pytest.raises(GraphExecutionUncertainError) as recovery:
        graph.invoke(None, CONFIG)
    assert type(recovery.value).__name__ == "GraphExecutionUncertainError"
    assert effects == [1]


def test_update_state_cannot_erase_pending_execution():
    saver = MemorySaver()
    effects = []
    def effect(state):
        effects.append(1)
        raise TimeoutError("ack lost")
    graph = graph_for(effect, saver)
    with pytest.raises(GraphExecutionUncertainError):
        graph.invoke({}, CONFIG)
    original_ids = [t["id"] for t in saver.get("dispatch-test").pending_tasks]
    try:
        graph.update_state(CONFIG, {"count": 1})
    except GraphExecutionUncertainError as error:
        assert type(error).__name__ == "GraphExecutionUncertainError"
    assert [t["id"] for t in saver.get("dispatch-test").pending_tasks] == original_ids
    with pytest.raises(GraphExecutionUncertainError):
        graph.invoke(None, CONFIG)
    assert effects == [1]


def test_historical_checkpoint_cannot_bypass_latest_uncertainty():
    saver = MemorySaver()
    effects = []
    builder = StateGraph(State)
    builder.add_node("prepare", lambda state: {})
    def effect(state):
        effects.append(1)
        raise TimeoutError("ack lost")
    builder.add_node("effect", effect)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "effect")
    builder.add_edge("effect", END)
    graph = builder.compile(checkpointer=saver)
    with pytest.raises(GraphExecutionUncertainError):
        graph.invoke({}, CONFIG)
    checkpoint = next(c for c in saver.list("dispatch-test") if c.next_nodes == ["effect"]
                      and c.pending_tasks[0]["status"] == "NOT_STARTED")
    assert checkpoint.checkpoint_id != saver.get("dispatch-test").checkpoint_id
    history_config = {"configurable": {"thread_id": "dispatch-test", "checkpoint_id": checkpoint.checkpoint_id}}
    with pytest.raises(GraphExecutionUncertainError) as recovery:
        graph.invoke(None, history_config)
    assert type(recovery.value).__name__ == "GraphExecutionUncertainError"
    assert effects == [1]


@pytest.mark.asyncio
async def test_parallel_recovery_reuses_completed_sibling_result_once():
    calls = {"good": 0, "read": 0}
    saver = MemorySaver()
    builder = StateGraph(State)
    async def good(state):
        calls["good"] += 1
        return {"results": ["receipt"]}
    async def read(state):
        calls["read"] += 1
        await asyncio.sleep(0.01)
        if calls["read"] == 1:
            raise TimeoutError("read failed")
        return {"results": ["lookup"]}
    builder.add_node("good", good)
    builder.add_node("read", read, retry_safe=True)
    for name in calls:
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    graph = builder.compile(checkpointer=saver)
    with pytest.raises(TimeoutError):
        await graph.ainvoke({"results": []}, CONFIG)
    ids = [t["id"] for t in saver.get("dispatch-test").pending_tasks]
    result = await graph.ainvoke(None, CONFIG)
    assert result["results"] == ["receipt", "lookup"]
    assert calls == {"good": 1, "read": 2}
    assert any([t["id"] for t in c.pending_tasks or []] == ids for c in saver.list("dispatch-test"))


def test_send_recovery_preserves_each_target_input_and_identity():
    calls = []
    saver = MemorySaver()
    builder = StateGraph(State)
    builder.add_node("route", lambda state: {})
    def worker(state, config):
        calls.append((state["item"], config["configurable"]["node_execution_id"]))
        if state["item"] == "bad" and sum(item == "bad" for item, _ in calls) == 1:
            raise TimeoutError("safe read failure")
        return {"results": [state["item"]]}
    builder.add_node("worker", worker, retry_safe=True)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", lambda state: [Send("worker", {"item": "good"}),
                                                            Send("worker", {"item": "bad"})], ["worker"])
    builder.add_edge("worker", END)
    graph = builder.compile(checkpointer=saver)
    with pytest.raises(TimeoutError):
        graph.invoke({"results": []}, CONFIG)
    assert graph.invoke(None, CONFIG)["results"] == ["good", "bad"]
    assert [item for item, _ in calls] == ["good", "bad", "bad"]
    assert calls[1][1] == calls[2][1]
    assert calls[0][1] != calls[1][1]


@pytest.mark.parametrize("send", [False, True])
def test_multiple_before_interrupts_preserve_each_approval(send):
    effects = []
    builder = StateGraph(State)
    builder.add_node("a", lambda state: effects.append(state.get("item", "a")) or {})
    builder.add_node("b", lambda state: effects.append("b") or {})
    if send:
        builder.add_node("route", lambda state: {})
        builder.add_edge(START, "route")
        builder.add_conditional_edges("route", lambda state: [Send("a", {"item": "first"}),
                                                               Send("a", {"item": "second"})], ["a"])
    else:
        builder.add_edge(START, "a")
        builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["a", "b"])
    assert "__interrupt__" in graph.invoke({}, CONFIG)
    assert "__interrupt__" in graph.invoke(Command(resume=True), CONFIG)
    assert "__interrupt__" not in graph.invoke(Command(resume=True), CONFIG)
    assert effects == (["first", "second"] if send else ["a", "b"])


@pytest.mark.asyncio
async def test_cancelled_async_body_keeps_durable_uncertainty():
    saver = MemorySaver()
    committed = asyncio.Event()
    effects = []
    async def effect(state, config):
        effects.append(config["configurable"]["node_execution_id"])
        committed.set()
        await asyncio.Event().wait()
        return {}
    graph = graph_for(effect, saver)
    running = asyncio.create_task(graph.ainvoke({}, CONFIG))
    await asyncio.wait_for(committed.wait(), timeout=2)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    pending = saver.get("dispatch-test").pending_tasks
    assert pending[0]["id"] == effects[0]
    assert pending[0]["status"] in {"DISPATCH_RECORDED", "UNKNOWN"}
    with pytest.raises(GraphExecutionUncertainError):
        await graph.ainvoke(None, CONFIG)
    assert len(effects) == 1


@pytest.mark.asyncio
async def test_parallel_inside_interrupt_reuses_completed_receipt():
    effects = []
    ids = []
    builder = StateGraph(State)
    builder.add_node("effect", lambda state: effects.append(1) or {"results": ["receipt"]})
    async def approval(state, config):
        ids.append(config["configurable"]["node_execution_id"])
        await asyncio.sleep(0.01)
        answer = interrupt("approve read")
        return {"results": [answer]}
    builder.add_node("approval", approval, retry_safe=True)
    for name in ["effect", "approval"]:
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    graph = builder.compile(checkpointer=MemorySaver())
    assert "__interrupt__" in await graph.ainvoke({"results": []}, CONFIG)
    assert (await graph.ainvoke(Command(resume="approved"), CONFIG))["results"] == ["receipt", "approved"]
    assert effects == [1]
    assert ids[0] == ids[1]


def test_terminal_legacy_checkpoint_allows_new_legitimate_input():
    saver = MemorySaver()
    saver.put(Checkpoint("legacy-finished", "dispatch-test", {"count": 1}, [END]))
    ids = []
    def effect(state, config):
        ids.append(config["configurable"]["node_execution_id"])
        return {"count": state["count"] + 1}
    result = graph_for(effect, saver).invoke({"count": 2}, CONFIG)
    assert result["count"] == 3
    assert len(ids) == 1
    assert ids[0] != "legacy-finished"


@pytest.mark.asyncio
async def test_async_mode_waits_for_older_write_before_dispatch():
    entered_write = threading.Event()
    release_write = threading.Event()
    effects = []
    class DelayedSaver(MemorySaver):
        def put(self, checkpoint):
            if not entered_write.is_set():
                entered_write.set()
                if not release_write.wait(3):
                    raise TimeoutError("test did not release checkpoint")
            super().put(checkpoint)
    saver = DelayedSaver()
    graph = graph_for(lambda state: effects.append(1) or {"count": 1}, saver)
    running = asyncio.create_task(asyncio.to_thread(graph.invoke, {}, CONFIG))
    try:
        assert await asyncio.to_thread(entered_write.wait, 2)
        assert effects == []
    finally:
        release_write.set()
    assert (await running)["count"] == 1
    await asyncio.sleep(0.01)
    assert saver.get("dispatch-test").pending_tasks == []
    assert effects == [1]
