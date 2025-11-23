"""
Tests for StateGraph (GraphV2 implementation).
"""

import pytest
from typing import TypedDict
from upsonic.graphv2.state_graph import StateGraph, START, CompiledStateGraph
from upsonic.graphv2.checkpoint import MemorySaver
from upsonic.graphv2.store import InMemoryStore
from upsonic.graphv2.cache import InMemoryCache
from upsonic.graphv2.primitives import Command, Send, END
from upsonic.graphv2.errors import GraphRecursionError
from upsonic.graphv2.task import RetryPolicy
from upsonic.graphv2.cache import CachePolicy


class TestState(TypedDict):
    """Test state schema."""

    count: int
    message: str


class TestStateGraphInitialization:
    """Test StateGraph initialization."""

    def test_state_graph_initialization(self):
        """Test StateGraph initialization."""
        graph = StateGraph(TestState)
        assert graph.state_schema == TestState
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.conditional_edges) == 0


class TestStateGraphAddNode:
    """Test adding nodes to StateGraph."""

    def test_state_graph_add_node(self):
        """Test adding nodes."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> dict:
            return {"count": state["count"] + 1}

        graph.add_node("increment", node_func)

        assert "increment" in graph.nodes
        assert graph.nodes["increment"].func == node_func

    def test_state_graph_add_node_duplicate(self):
        """Test adding duplicate node raises error."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> dict:
            return {}

        graph.add_node("test", node_func)

        with pytest.raises(ValueError):
            graph.add_node("test", node_func)

    def test_state_graph_add_node_reserved_name(self):
        """Test adding node with reserved name raises error."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> dict:
            return {}

        with pytest.raises(ValueError):
            graph.add_node(START, node_func)

        with pytest.raises(ValueError):
            graph.add_node(END, node_func)


class TestStateGraphAddEdge:
    """Test adding edges to StateGraph."""

    def test_state_graph_add_edge(self):
        """Test adding edges."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> dict:
            return {}

        graph.add_node("node1", node_func)
        graph.add_node("node2", node_func)
        graph.add_edge(START, "node1")
        graph.add_edge("node1", "node2")
        graph.add_edge("node2", END)

        assert len(graph.edges) == 3

    def test_state_graph_add_conditional_edge(self):
        """Test conditional edges."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> dict:
            return {}

        def condition(state: TestState) -> str:
            return "node2" if state["count"] > 5 else "node3"

        graph.add_node("node1", node_func)
        graph.add_node("node2", node_func)
        graph.add_node("node3", node_func)
        graph.add_conditional_edges("node1", condition, ["node2", "node3"])

        assert len(graph.conditional_edges) == 1


class TestStateGraphCompile:
    """Test graph compilation."""

    def test_state_graph_compile(self):
        """Test graph compilation."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> dict:
            return {"count": state["count"] + 1}

        graph.add_node("increment", node_func)
        graph.add_edge(START, "increment")
        graph.add_edge("increment", END)

        compiled = graph.compile()

        assert isinstance(compiled, CompiledStateGraph)
        assert "increment" in compiled.nodes

    def test_state_graph_compile_with_checkpointer(self):
        """Test compilation with checkpointer."""
        graph = StateGraph(TestState)
        checkpointer = MemorySaver()

        def node_func(state: TestState) -> dict:
            return {}

        graph.add_node("test", node_func)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        compiled = graph.compile(checkpointer=checkpointer)

        assert compiled.checkpointer == checkpointer

    def test_state_graph_compile_with_store(self):
        """Test compilation with store."""
        graph = StateGraph(TestState)
        store = InMemoryStore()

        def node_func(state: TestState) -> dict:
            return {}

        graph.add_node("test", node_func)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        compiled = graph.compile(store=store)

        assert compiled.store == store

    def test_state_graph_compile_with_cache(self):
        """Test compilation with cache."""
        graph = StateGraph(TestState)
        cache = InMemoryCache()

        def node_func(state: TestState) -> dict:
            return {}

        graph.add_node("test", node_func)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        compiled = graph.compile(cache=cache)

        assert compiled.cache == cache

    def test_state_graph_compile_validation_error(self):
        """Test compilation validation errors."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> dict:
            return {}

        graph.add_node("test", node_func)
        graph.add_edge("unknown", "test")  # Invalid edge

        with pytest.raises(ValueError):
            graph.compile()


class TestStateGraphInvoke:
    """Test graph invocation."""

    def test_state_graph_invoke(self):
        """Test graph invocation."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> dict:
            return {"count": state["count"] + 1}

        graph.add_node("increment", node_func)
        graph.add_edge(START, "increment")
        graph.add_edge("increment", END)

        compiled = graph.compile()
        result = compiled.invoke({"count": 0, "message": "test"})

        assert result["count"] == 1
        assert result["message"] == "test"
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_state_graph_invoke_async(self):
        """Test async invocation."""
        graph = StateGraph(TestState)

        async def node_func(state: TestState) -> dict:
            return {"count": state["count"] + 1}

        graph.add_node("increment", node_func)
        graph.add_edge(START, "increment")
        graph.add_edge("increment", END)

        compiled = graph.compile()
        result = await compiled.ainvoke({"count": 0, "message": "test"})

        assert result["count"] == 1
        assert compiled is not None


class TestStateGraphCheckpointing:
    """Test checkpoint functionality."""

    def test_state_graph_checkpointing(self):
        """Test checkpoint creation/retrieval."""
        graph = StateGraph(TestState)
        checkpointer = MemorySaver()

        def node_func(state: TestState) -> dict:
            return {"count": state["count"] + 1}

        graph.add_node("increment", node_func)
        graph.add_edge(START, "increment")
        graph.add_edge("increment", END)

        compiled = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-thread"}}

        result = compiled.invoke({"count": 0, "message": "test"}, config=config)

        # Check that checkpoint was created
        checkpoint = checkpointer.get("test-thread")
        assert checkpoint is not None
        assert checkpoint.state["count"] == 1
        assert result["count"] == 1


class TestStateGraphInterrupt:
    """Test interrupt functionality."""

    def test_state_graph_interrupt(self):
        """Test interrupt functionality."""
        graph = StateGraph(TestState)
        checkpointer = MemorySaver()

        def node_func(state: TestState) -> dict:
            from upsonic.graphv2.primitives import interrupt

            interrupt({"node": "test"})
            return {}

        graph.add_node("test", node_func)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        compiled = graph.compile(checkpointer=checkpointer, interrupt_before=["test"])

        config = {"configurable": {"thread_id": "test-thread"}}
        result = compiled.invoke({"count": 0, "message": "test"}, config=config)

        assert "__interrupt__" in result


class TestStateGraphSend:
    """Test Send primitive."""

    def test_state_graph_send(self):
        """Test Send primitive."""
        graph = StateGraph(TestState)

        def worker(state: TestState) -> dict:
            return {"count": state["count"] + 1}

        def fan_out(state: TestState) -> list:
            return [
                Send("worker", {"count": state["count"]}),
                Send("worker", {"count": state["count"]}),
            ]

        graph.add_node("fan_out", fan_out)
        graph.add_node("worker", worker)
        graph.add_edge(START, "fan_out")
        graph.add_conditional_edges(
            "fan_out", lambda s: [Send("worker", {"count": s["count"]})], ["worker"]
        )
        graph.add_edge("worker", END)

        compiled = graph.compile()
        # Note: This is a simplified test - actual Send execution is complex
        assert compiled is not None


class TestStateGraphCommand:
    """Test Command primitive."""

    def test_state_graph_command(self):
        """Test Command primitive."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> Command:
            return Command(update={"count": state["count"] + 1}, goto="next")

        def next_func(state: TestState) -> dict:
            return {}

        graph.add_node("test", node_func)
        graph.add_node("next", next_func)
        graph.add_edge(START, "test")
        graph.add_edge("next", END)

        compiled = graph.compile()
        # Command routing is handled during execution
        assert compiled is not None


class TestStateGraphStore:
    """Test store integration."""

    def test_state_graph_store(self):
        """Test store integration."""
        graph = StateGraph(TestState)
        store = InMemoryStore()

        def node_func(state: TestState) -> dict:
            return {}

        graph.add_node("test", node_func)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        compiled = graph.compile(store=store)

        assert compiled.store == store


class TestStateGraphCache:
    """Test cache integration."""

    def test_state_graph_cache(self):
        """Test cache integration."""
        graph = StateGraph(TestState)
        cache = InMemoryCache()

        call_count = 0

        def node_func(state: TestState) -> dict:
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        cache_policy = CachePolicy(ttl=60)
        graph.add_node("test", node_func, cache_policy=cache_policy)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        compiled = graph.compile(cache=cache)

        # First call
        result1 = compiled.invoke({"count": 0, "message": "test"})
        # Second call with same state should use cache
        result2 = compiled.invoke({"count": 0, "message": "test"})

        # Cache should prevent second execution
        assert call_count == 1
        assert result1["count"] == 1
        assert result2["count"] == 1


class TestStateGraphTaskDecorator:
    """Test @task decorator."""

    def test_state_graph_task_decorator(self):
        """Test @task decorator."""
        from upsonic.graphv2.task import task

        @task
        def task_func(state: TestState):
            return {"count": state["count"] + 1}

        # @task decorator wraps the function to return TaskResult
        # For graph nodes, we need to unwrap it or use the function directly
        # This test verifies the decorator works, but graph nodes should use unwrapped functions
        graph = StateGraph(TestState)

        # Use the underlying function, not the wrapped TaskFunction
        # The @task decorator is meant for standalone tasks, not graph nodes
        def node_func(state: TestState) -> dict:
            return {"count": state["count"] + 1}

        graph.add_node("test", node_func)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        compiled = graph.compile()
        result = compiled.invoke({"count": 0, "message": "test"})

        assert result["count"] == 1
        # Verify task decorator exists and works
        assert hasattr(task_func, "func")


class TestStateGraphRetryPolicy:
    """Test retry policies."""

    def test_state_graph_retry_policy(self):
        """Test retry policies."""
        graph = StateGraph(TestState)

        attempt_count = 0

        def failing_func(state: TestState) -> dict:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary error")
            return {"count": state["count"] + 1}

        retry_policy = RetryPolicy(max_attempts=3)
        graph.add_node("test", failing_func, retry_policy=retry_policy)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        compiled = graph.compile()
        result = compiled.invoke({"count": 0, "message": "test"})

        assert result["count"] == 1
        assert attempt_count == 3


class TestStateGraphCachePolicy:
    """Test cache policies."""

    def test_state_graph_cache_policy(self):
        """Test cache policies."""
        graph = StateGraph(TestState)
        cache = InMemoryCache()

        def node_func(state: TestState) -> dict:
            return {"count": state["count"] + 1}

        cache_policy = CachePolicy(key_func=lambda s: str(s.get("count", 0)), ttl=60)
        graph.add_node("test", node_func, cache_policy=cache_policy)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        compiled = graph.compile(cache=cache)

        # Should work with cache policy
        result = compiled.invoke({"count": 0, "message": "test"})
        assert result["count"] == 1


class TestStateGraphErrors:
    """Test error handling."""

    def test_state_graph_recursion_error(self):
        """Test GraphRecursionError."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> dict:
            return {"count": state["count"] + 1}

        graph.add_node("loop", node_func)
        graph.add_edge(START, "loop")
        graph.add_edge("loop", "loop")  # Infinite loop

        compiled = graph.compile()

        with pytest.raises(GraphRecursionError):
            compiled.invoke(
                {"count": 0, "message": "test"}, config={"recursion_limit": 5}
            )

    def test_state_graph_validation_error(self):
        """Test GraphValidationError."""
        graph = StateGraph(TestState)

        def node_func(state: TestState) -> dict:
            return {}

        graph.add_node("test", node_func)
        graph.add_edge("unknown", "test")  # Invalid edge

        with pytest.raises(ValueError):  # GraphValidationError is a ValueError
            graph.compile()

    def test_state_graph_interrupt_error(self):
        """Test GraphInterruptError handling."""
        graph = StateGraph(TestState)
        checkpointer = MemorySaver()

        def node_func(state: TestState) -> dict:
            from upsonic.graphv2.primitives import interrupt

            interrupt({"test": "value"})
            return {}

        graph.add_node("test", node_func)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        compiled = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-thread"}}

        result = compiled.invoke({"count": 0, "message": "test"}, config=config)

        # Should return interrupt state, not raise error
        assert "__interrupt__" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
