"""Phase 5/2 read-through: agent.usage and task.usage query the registry."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from upsonic import Agent, Task
from upsonic.models import ModelResponse, TextPart
from upsonic.usage import RequestUsage
from upsonic.usage_registry import (
    get_default_registry,
    record_request_usage,
    scope,
)


def _response(input_tokens=70, output_tokens=30):
    return ModelResponse(
        parts=[TextPart(content="ok")],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        provider_name="test-provider",
        provider_response_id="r-1",
        provider_details={},
        finish_reason="stop",
    )


class TestAgentUsageRegistryView(unittest.TestCase):
    def setUp(self):
        get_default_registry().clear()

    @patch("upsonic.models.infer_model")
    def test_agent_usage_reflects_registry_entries(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response(70, 30))

        agent = Agent(name="A", model=mock_model)
        agent.do(Task("hi"))

        usage = agent.usage
        self.assertIsNotNone(usage)
        self.assertEqual(usage.input_tokens, 70)
        self.assertEqual(usage.output_tokens, 30)
        self.assertEqual(usage.requests, 1)

    @patch("upsonic.models.infer_model")
    def test_agent_usage_picks_up_external_registry_writes(self, mock_infer_model):
        """A memory / reliability sub-agent that writes a registry entry
        under the parent agent's scope must show up in agent.usage."""
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response(70, 30))

        agent = Agent(name="A", model=mock_model)
        agent.do(Task("hi"))

        with scope(agent_usage_id=agent.agent_usage_id):
            record_request_usage(
                RequestUsage(input_tokens=11, output_tokens=22),
                model="memory-model",
                pipeline_step="summary",
                cost_usd=0.0042,
            )

        usage = agent.usage
        self.assertEqual(usage.input_tokens, 70 + 11)
        self.assertEqual(usage.output_tokens, 30 + 22)


class TestTaskUsageRegistryView(unittest.TestCase):
    def setUp(self):
        get_default_registry().clear()

    @patch("upsonic.models.infer_model")
    def test_task_usage_reflects_registry_entries(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response(50, 25))

        agent = Agent(name="A", model=mock_model)
        task = Task("hi")
        agent.do(task)

        usage = task.usage
        self.assertEqual(usage.input_tokens, 50)
        self.assertEqual(usage.output_tokens, 25)

    @patch("upsonic.models.infer_model")
    def test_task_usage_per_task_isolation(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response(40, 20))

        agent = Agent(name="A", model=mock_model)
        t1, t2 = Task("a"), Task("b")
        agent.do(t1)
        agent.do(t2)

        # Each task only sees its own spend.
        self.assertEqual(t1.usage.input_tokens, 40)
        self.assertEqual(t2.usage.input_tokens, 40)
        self.assertNotEqual(t1.task_usage_id, t2.task_usage_id)


if __name__ == "__main__":
    unittest.main()
