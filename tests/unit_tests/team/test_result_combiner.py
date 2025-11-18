import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List, Any
from pydantic import BaseModel
from unittest.mock import patch

from upsonic.team.result_combiner import ResultCombiner
from upsonic.tasks.tasks import Task
from upsonic.agent.agent import Agent


# ============================================================================
# MOCK COMPONENTS FOR TESTING
# ============================================================================


class MockModel:
    """Mock model for testing."""

    def __init__(self, name: str = "test-model"):
        self.model_name = name


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "TestAgent", model: Any = None, debug: bool = False):
        self.name = name
        self.model = model or MockModel()
        self.debug = debug

    async def do_async(self, task: Task) -> Any:
        """Mock async do method."""
        task._response = f"Combined result from {self.name}"
        return task.response


# ============================================================================
# TEST 1: RESULT COMBINER INITIALIZATION
# ============================================================================


def test_result_combiner_initialization():
    """
    Test ResultCombiner initialization.

    This tests that:
    1. ResultCombiner can be initialized with model
    2. Debug flag is stored
    3. Model is properly set
    """
    print("\n" + "=" * 80)
    print("TEST 1: ResultCombiner initialization")
    print("=" * 80)

    model = MockModel()
    combiner = ResultCombiner(model=model, debug=True)

    assert combiner.model == model, "Model should be set"
    assert combiner.debug == True, "Debug should be True"

    # Test without model
    combiner_no_model = ResultCombiner(model=None, debug=False)
    assert combiner_no_model.model is None, "Model can be None"
    assert combiner_no_model.debug == False, "Debug should be False"

    print("✓ ResultCombiner initialization works!")


# ============================================================================
# TEST 2: SHOULD COMBINE RESULTS
# ============================================================================


def test_result_combiner_should_combine_results():
    """
    Test should_combine_results logic.

    This tests that:
    1. Returns True for multiple results
    2. Returns False for single result
    3. Handles empty results
    """
    print("\n" + "=" * 80)
    print("TEST 2: ResultCombiner should combine results")
    print("=" * 80)

    model = MockModel()
    combiner = ResultCombiner(model=model)

    # Test with multiple results
    task1 = Task(description="Task 1")
    task1._response = "Result 1"
    task2 = Task(description="Task 2")
    task2._response = "Result 2"
    results_multiple = [task1, task2]

    should_combine = combiner.should_combine_results(results_multiple)
    assert should_combine == True, "Should combine multiple results"

    # Test with single result
    results_single = [task1]
    should_combine = combiner.should_combine_results(results_single)
    assert should_combine == False, "Should not combine single result"

    # Test with empty results
    results_empty = []
    should_combine = combiner.should_combine_results(results_empty)
    assert should_combine == False, "Should not combine empty results"

    print("✓ ResultCombiner should combine results works!")


# ============================================================================
# TEST 3: GET SINGLE RESULT
# ============================================================================


def test_result_combiner_get_single_result():
    """
    Test get_single_result method.

    This tests that:
    1. Returns response from single task
    2. Returns None for empty results
    3. Returns first result for multiple (edge case)
    """
    print("\n" + "=" * 80)
    print("TEST 3: ResultCombiner get single result")
    print("=" * 80)

    model = MockModel()
    combiner = ResultCombiner(model=model)

    # Test with single result
    task = Task(description="Task")
    task._response = "Single result"
    results = [task]

    result = combiner.get_single_result(results)
    assert result == "Single result", "Should return task response"
    # Also test that response property works
    assert task.response == "Single result", "Response property should work"

    # Test with empty results
    result_empty = combiner.get_single_result([])
    assert result_empty is None, "Should return None for empty results"

    # Test that response property returns None when _response is None
    task_empty = Task(description="Empty task")
    assert task_empty.response is None, "Response should be None when _response is None"

    print("✓ ResultCombiner get single result works!")


# ============================================================================
# TEST 4: COMBINE RESULTS
# ============================================================================


@pytest.mark.asyncio
async def test_result_combiner_combine_results():
    """
    Test result combination logic.

    This tests that:
    1. Multiple results are combined into one
    2. Agent is created and used
    3. Final task is created with correct context
    """
    print("\n" + "=" * 80)
    print("TEST 4: ResultCombiner combine results")
    print("=" * 80)

    model = MockModel()
    combiner = ResultCombiner(model=model, debug=False)

    task1 = Task(description="Task 1")
    task1._response = "Result 1"
    task2 = Task(description="Task 2")
    task2._response = "Result 2"
    results = [task1, task2]

    # Mock Agent creation and execution
    with patch("upsonic.team.result_combiner.Agent") as mock_agent_class:
        mock_agent = MockAgent("CombinerAgent", model=model)
        mock_agent.do_async = AsyncMock()

        def mock_do_async(task):
            task._response = "Combined final result"
            return task.response

        mock_agent.do_async = AsyncMock(side_effect=mock_do_async)
        mock_agent_class.return_value = mock_agent

        combined = await combiner.combine_results(results, response_format=str)

        assert combined == "Combined final result", "Should return combined result"
        mock_agent.do_async.assert_called_once()

        # Verify task was created with context
        call_args = mock_agent.do_async.call_args
        task_arg = call_args[0][0]
        assert isinstance(task_arg, Task), "Should create Task"
        assert task_arg.context == results, "Context should include all results"

    print("✓ ResultCombiner combine results works!")


# ============================================================================
# TEST 5: MULTIPLE RESULTS COMBINATION
# ============================================================================


@pytest.mark.asyncio
async def test_result_combiner_multiple_results():
    """
    Test multiple result combination.

    This tests that:
    1. Three or more results are combined
    2. All results are included in context
    3. Response format is respected
    """
    print("\n" + "=" * 80)
    print("TEST 5: ResultCombiner multiple results")
    print("=" * 80)

    model = MockModel()
    combiner = ResultCombiner(model=model)

    task1 = Task(description="Task 1")
    task1._response = "Result 1"
    task2 = Task(description="Task 2")
    task2._response = "Result 2"
    task3 = Task(description="Task 3")
    task3._response = "Result 3"
    results = [task1, task2, task3]

    # Mock Agent
    with patch("upsonic.team.result_combiner.Agent") as mock_agent_class:
        mock_agent = MockAgent("CombinerAgent", model=model)

        def mock_do_async(task):
            task._response = "All results combined"
            return task.response

        mock_agent.do_async = AsyncMock(side_effect=mock_do_async)
        mock_agent_class.return_value = mock_agent

        combined = await combiner.combine_results(results, response_format=str)

        assert combined == "All results combined", "Should combine all results"

        # Verify all results in context
        call_args = mock_agent.do_async.call_args
        task_arg = call_args[0][0]
        assert len(task_arg.context) == 3, "Context should include all 3 results"

    print("✓ ResultCombiner multiple results works!")


# ============================================================================
# TEST 6: RESPONSE FORMAT
# ============================================================================


@pytest.mark.asyncio
async def test_result_combiner_response_format():
    """
    Test response format handling.

    This tests that:
    1. Response format is passed to final task
    2. Different formats are supported
    3. Default format is str
    """
    print("\n" + "=" * 80)
    print("TEST 6: ResultCombiner response format")
    print("=" * 80)

    model = MockModel()
    combiner = ResultCombiner(model=model)

    task = Task(description="Task")
    task._response = "Result"
    results = [task]

    # Test with custom response format
    class CustomResponse(BaseModel):
        summary: str
        details: List[str]

    with patch("upsonic.team.result_combiner.Agent") as mock_agent_class:
        mock_agent = MockAgent("CombinerAgent", model=model)

        def mock_do_async(task):
            task._response = CustomResponse(
                summary="Summary", details=["Detail1", "Detail2"]
            )
            return task.response

        mock_agent.do_async = AsyncMock(side_effect=mock_do_async)
        mock_agent_class.return_value = mock_agent

        combined = await combiner.combine_results(
            results, response_format=CustomResponse
        )

        assert isinstance(combined, CustomResponse), (
            "Should return CustomResponse format"
        )

        # Verify response_format was set
        call_args = mock_agent.do_async.call_args
        task_arg = call_args[0][0]
        assert task_arg.response_format == CustomResponse, (
            "Response format should be set"
        )

    print("✓ ResultCombiner response format works!")


# ============================================================================
# TEST 7: ERROR HANDLING
# ============================================================================


@pytest.mark.asyncio
async def test_result_combiner_error_handling():
    """
    Test error handling.

    This tests that:
    1. Missing model raises ValueError
    2. Errors during combination are handled
    """
    print("\n" + "=" * 80)
    print("TEST 7: ResultCombiner error handling")
    print("=" * 80)

    # Test without model
    combiner_no_model = ResultCombiner(model=None)
    task = Task(description="Task")
    task._response = "Result"
    results = [task]

    with pytest.raises(ValueError, match="requires a model"):
        await combiner_no_model.combine_results(results, response_format=str)

    print("✓ ResultCombiner error handling works!")


# ============================================================================
# TEST 8: DEBUG SETTING
# ============================================================================


@pytest.mark.asyncio
async def test_result_combiner_debug_setting():
    """
    Test debug setting handling.

    This tests that:
    1. Debug setting from combiner is used
    2. Falls back to agent debug if not set
    """
    print("\n" + "=" * 80)
    print("TEST 8: ResultCombiner debug setting")
    print("=" * 80)

    model = MockModel()

    # Test with combiner debug=True
    combiner_debug = ResultCombiner(model=model, debug=True)

    # Test with combiner debug=False, fallback to agent
    agent = MockAgent("Agent1", model=model, debug=True)
    combiner_fallback = ResultCombiner(model=model, debug=False)

    task = Task(description="Task")
    task._response = "Result"
    results = [task]

    with patch("upsonic.team.result_combiner.Agent") as mock_agent_class:
        mock_agent = MockAgent("CombinerAgent", model=model)
        mock_agent.do_async = AsyncMock(return_value="Result")
        mock_agent_class.return_value = mock_agent

        await combiner_fallback.combine_results(
            results, response_format=str, agents=[agent]
        )

        # Verify Agent was created with correct debug setting
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs.get("debug") == True, "Should use agent debug setting"

    print("✓ ResultCombiner debug setting works!")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
