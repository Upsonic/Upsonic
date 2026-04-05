import unittest
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager
from upsonic import Task, Agent
from upsonic.run.agent.output import AgentRunOutput
from upsonic.models import ModelResponse, TextPart


class TestDo(unittest.TestCase):
    """Test suite for Task, Agent, and do functionality"""
    
    @patch('upsonic.models.infer_model')
    def test_agent_do_basic(self, mock_infer_model):
        """Test basic functionality of Agent.do with a Task"""
        # Mock the model inference
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        
        # Mock the model request to return a proper ModelResponse
        mock_response = ModelResponse(
            parts=[TextPart(content="I was developed by Upsonic, an AI agent framework designed for building reliable AI applications.")],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        )
        mock_model.request = AsyncMock(return_value=mock_response)
        
        # Create a task
        task = Task("Who developed you?")
        
        # Create an agent
        agent = Agent(name="Coder", model=mock_model)
        
        result = agent.do(task)

        # Check that task has a response
        self.assertNotEqual(task.response, None)
        self.assertNotEqual(task.response, "")
        self.assertIsInstance(task.response, str)

        # Check that result is a string (the actual output)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, None)
        self.assertNotEqual(result, "")


class TestCallEndPriceIdTracking(unittest.TestCase):
    """Test that call_end and agent_end track price_id regardless of print_output."""

    def setUp(self) -> None:
        from upsonic.utils.printing import price_id_summary
        price_id_summary.clear()

    def tearDown(self) -> None:
        from upsonic.utils.printing import price_id_summary
        price_id_summary.clear()

    def test_call_end_tracks_price_id_when_print_output_false(self) -> None:
        """call_end must populate price_id_summary even when print_output=False."""
        from upsonic.utils.printing import call_end, price_id_summary

        mock_model = MagicMock()
        mock_model.model_name = "openai/gpt-4o-mini"

        usage = {"input_tokens": 100, "output_tokens": 50}
        price_id = "test-price-id-001"

        call_end(
            result="test result",
            model=mock_model,
            response_format=str,
            start_time=1000.0,
            end_time=1001.0,
            usage=usage,
            tool_usage=None,
            debug=False,
            price_id=price_id,
            print_output=False,
        )

        self.assertIn(price_id, price_id_summary)
        self.assertEqual(price_id_summary[price_id]["input_tokens"], 100)
        self.assertEqual(price_id_summary[price_id]["output_tokens"], 50)

    def test_call_end_tracks_price_id_when_print_output_true(self) -> None:
        """call_end must populate price_id_summary when print_output=True."""
        from upsonic.utils.printing import call_end, price_id_summary

        mock_model = MagicMock()
        mock_model.model_name = "openai/gpt-4o-mini"

        usage = {"input_tokens": 200, "output_tokens": 80}
        price_id = "test-price-id-002"

        call_end(
            result="test result",
            model=mock_model,
            response_format=str,
            start_time=1000.0,
            end_time=1001.0,
            usage=usage,
            tool_usage=None,
            debug=False,
            price_id=price_id,
            print_output=True,
        )

        self.assertIn(price_id, price_id_summary)
        self.assertEqual(price_id_summary[price_id]["input_tokens"], 200)
        self.assertEqual(price_id_summary[price_id]["output_tokens"], 80)

    def test_call_end_accumulates_across_multiple_calls(self) -> None:
        """Multiple call_end invocations with same price_id accumulate tokens."""
        from upsonic.utils.printing import call_end, price_id_summary

        mock_model = MagicMock()
        mock_model.model_name = "openai/gpt-4o-mini"
        price_id = "test-price-id-003"

        call_end(
            result="r1", model=mock_model, response_format=str,
            start_time=1000.0, end_time=1001.0,
            usage={"input_tokens": 50, "output_tokens": 20},
            tool_usage=None, debug=False, price_id=price_id,
            print_output=False,
        )
        call_end(
            result="r2", model=mock_model, response_format=str,
            start_time=1001.0, end_time=1002.0,
            usage={"input_tokens": 30, "output_tokens": 10},
            tool_usage=None, debug=False, price_id=price_id,
            print_output=False,
        )

        self.assertEqual(price_id_summary[price_id]["input_tokens"], 80)
        self.assertEqual(price_id_summary[price_id]["output_tokens"], 30)

    def test_agent_end_tracks_price_id_when_print_output_false(self) -> None:
        """agent_end must populate price_id_summary even when print_output=False."""
        from upsonic.utils.printing import agent_end, price_id_summary

        mock_model = MagicMock()
        mock_model.model_name = "openai/gpt-4o-mini"

        usage = {"input_tokens": 150, "output_tokens": 60}
        price_id = "test-price-id-004"

        agent_end(
            result="test result",
            model=mock_model,
            response_format=str,
            start_time=1000.0,
            end_time=1001.0,
            usage=usage,
            tool_usage=None,
            tool_count=0,
            context_count=0,
            debug=False,
            price_id=price_id,
            print_output=False,
        )

        self.assertIn(price_id, price_id_summary)
        self.assertEqual(price_id_summary[price_id]["input_tokens"], 150)
        self.assertEqual(price_id_summary[price_id]["output_tokens"], 60)

    def test_no_price_id_does_not_add_entry(self) -> None:
        """call_end with price_id=None does not add entry to price_id_summary."""
        from upsonic.utils.printing import call_end, price_id_summary

        mock_model = MagicMock()
        mock_model.model_name = "openai/gpt-4o-mini"

        call_end(
            result="test result",
            model=mock_model,
            response_format=str,
            start_time=1000.0,
            end_time=1001.0,
            usage={"input_tokens": 100, "output_tokens": 50},
            tool_usage=None,
            debug=False,
            price_id=None,
            print_output=False,
        )

        self.assertEqual(len(price_id_summary), 0)


class TestTaskMetricsAfterExecution(unittest.TestCase):
    """Test that task metrics (total_cost, total_input_token, etc.) are properly set
    after execution via get_total_cost / price_id_summary integration."""

    def setUp(self) -> None:
        from upsonic.utils.printing import price_id_summary
        price_id_summary.clear()

    def tearDown(self) -> None:
        from upsonic.utils.printing import price_id_summary
        price_id_summary.clear()

    def test_task_total_cost_returns_value_when_price_tracked(self) -> None:
        """After price_id_summary is populated, task.total_cost should return a float."""
        from upsonic.utils.printing import price_id_summary

        task = Task("Test task")
        price_id = task.price_id  # triggers auto-generation

        price_id_summary[price_id] = {
            "input_tokens": 500,
            "output_tokens": 200,
            "estimated_cost": 0.0035,
        }

        self.assertIsNotNone(task.total_cost)
        self.assertIsInstance(task.total_cost, float)
        self.assertAlmostEqual(task.total_cost, 0.0035)

    def test_task_total_input_token_returns_value_when_price_tracked(self) -> None:
        """After price_id_summary is populated, task.total_input_token should return an int."""
        from upsonic.utils.printing import price_id_summary

        task = Task("Test task")
        price_id = task.price_id

        price_id_summary[price_id] = {
            "input_tokens": 500,
            "output_tokens": 200,
            "estimated_cost": 0.0035,
        }

        self.assertIsNotNone(task.total_input_token)
        self.assertEqual(task.total_input_token, 500)

    def test_task_total_output_token_returns_value_when_price_tracked(self) -> None:
        """After price_id_summary is populated, task.total_output_token should return an int."""
        from upsonic.utils.printing import price_id_summary

        task = Task("Test task")
        price_id = task.price_id

        price_id_summary[price_id] = {
            "input_tokens": 500,
            "output_tokens": 200,
            "estimated_cost": 0.0035,
        }

        self.assertIsNotNone(task.total_output_token)
        self.assertEqual(task.total_output_token, 200)

    def test_task_metrics_none_when_no_price_data(self) -> None:
        """Task metrics should return None when price_id is not in price_id_summary."""
        task = Task("Test task")
        _ = task.price_id  # triggers auto-generation

        self.assertIsNone(task.total_cost)
        self.assertIsNone(task.total_input_token)
        self.assertIsNone(task.total_output_token)

    def test_task_duration_set_after_start_and_end(self) -> None:
        """Task.duration should compute correctly when start_time and end_time are set."""
        import time

        task = Task("Test task")
        task.start_time = time.time()
        task.end_time = task.start_time + 2.5

        self.assertIsNotNone(task.duration)
        self.assertAlmostEqual(task.duration, 2.5, places=1)

    def test_task_duration_none_without_times(self) -> None:
        """Task.duration should be None when start_time or end_time is not set."""
        task = Task("Test task")

        self.assertIsNone(task.duration)

    def test_task_tool_calls_initially_empty(self) -> None:
        """Task.tool_calls should be an empty list initially."""
        task = Task("Test task")

        self.assertIsInstance(task.tool_calls, list)
        self.assertEqual(len(task.tool_calls), 0)

    def test_task_add_tool_call(self) -> None:
        """Task.add_tool_call properly appends to tool_calls."""
        task = Task("Test task")
        task.add_tool_call({"tool_name": "search", "params": {"q": "test"}, "tool_result": "ok"})

        self.assertEqual(len(task.tool_calls), 1)
        self.assertEqual(task.tool_calls[0]["tool_name"], "search")

    def test_price_id_reset_generates_new_id(self) -> None:
        """Setting a new price_id_ generates a different UUID."""
        import uuid
        task = Task("Test task")
        first_id = task.price_id

        task.price_id_ = str(uuid.uuid4())
        second_id = task.price_id

        self.assertNotEqual(first_id, second_id)

    def test_call_end_then_task_metrics_end_to_end(self) -> None:
        """End-to-end: call_end populates price_id_summary, then task.total_cost works."""
        from upsonic.utils.printing import call_end

        task = Task("Test task")
        price_id = task.price_id

        mock_model = MagicMock()
        mock_model.model_name = "openai/gpt-4o-mini"

        call_end(
            result="response",
            model=mock_model,
            response_format=str,
            start_time=1000.0,
            end_time=1001.0,
            usage={"input_tokens": 300, "output_tokens": 100},
            tool_usage=None,
            debug=False,
            price_id=price_id,
            print_output=False,
        )

        self.assertIsNotNone(task.total_cost)
        self.assertIsInstance(task.total_cost, float)
        self.assertGreaterEqual(task.total_cost, 0)
        self.assertIsNotNone(task.total_input_token)
        self.assertEqual(task.total_input_token, 300)
        self.assertIsNotNone(task.total_output_token)
        self.assertEqual(task.total_output_token, 100)

        

