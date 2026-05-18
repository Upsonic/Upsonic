import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from upsonic import Task, Agent
from upsonic.models import ModelResponse, TextPart


class TestDo(unittest.TestCase):
    """Smoke coverage for Task + Agent.do wiring."""

    @patch('upsonic.models.infer_model')
    def test_agent_do_basic(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model

        mock_response = ModelResponse(
            parts=[TextPart(content="I was developed by Upsonic.")],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop",
        )
        mock_model.request = AsyncMock(return_value=mock_response)

        task = Task("Who developed you?")
        agent = Agent(name="Coder", model=mock_model)
        result = agent.do(task)

        self.assertNotEqual(task.response, None)
        self.assertIsInstance(task.response, str)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")


class TestTaskTimingAndToolCalls(unittest.TestCase):
    """Task-level metric scaffolding that does not require a live run."""

    def test_task_duration_set_after_start_and_end(self) -> None:
        import time

        task = Task("Test task")
        task.start_time = time.time()
        task.end_time = task.start_time + 2.5

        self.assertIsNotNone(task.duration)
        self.assertAlmostEqual(task.duration, 2.5, places=1)

    def test_task_duration_none_without_times(self) -> None:
        task = Task("Test task")
        self.assertIsNone(task.duration)

    def test_task_tool_calls_initially_empty(self) -> None:
        task = Task("Test task")
        self.assertIsInstance(task.tool_calls, list)
        self.assertEqual(len(task.tool_calls), 0)

    def test_task_add_tool_call(self) -> None:
        task = Task("Test task")
        task.add_tool_call({"tool_name": "search", "params": {"q": "test"}, "tool_result": "ok"})

        self.assertEqual(len(task.tool_calls), 1)
        self.assertEqual(task.tool_calls[0]["tool_name"], "search")
