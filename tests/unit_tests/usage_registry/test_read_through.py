"""Phase 3b read-through: Chat public usage properties query the registry."""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from upsonic import Agent, Task
from upsonic.chat.chat import Chat
from upsonic.models import ModelResponse, TextPart
from upsonic.storage.in_memory.in_memory import InMemoryStorage
from upsonic.usage import RequestUsage
from upsonic.usage_registry import (
    UsageEntry,
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


class TestChatReadThrough(unittest.TestCase):
    def setUp(self):
        get_default_registry().clear()

    @patch("upsonic.models.infer_model")
    def test_chat_total_tokens_reads_from_registry(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response(70, 30))

        agent = Agent(name="A", model=mock_model)
        chat = Chat(session_id="s", user_id="u", agent=agent, storage=InMemoryStorage())
        asyncio.run(chat.invoke("hi"))

        # Inject a single extra entry directly via the registry (e.g. simulating
        # a memory-summary call). The chat surface should pick it up — the
        # legacy session.usage would NOT.
        with scope(chat_usage_id=chat.chat_usage_id):
            record_request_usage(
                RequestUsage(input_tokens=11, output_tokens=22),
                model="memory-model",
                pipeline_step="summary",
                cost_usd=0.003,
            )

        # Chat properties should reflect both rows.
        self.assertEqual(chat.usage.input_tokens, 70 + 11)
        self.assertEqual(chat.usage.output_tokens, 30 + 22)
        self.assertEqual(chat.usage.total_tokens, 70 + 30 + 11 + 22)
        # Cost sums every priced entry, including whatever genai_prices
        # assigned to the initial invoke — we just need the 0.003 injection
        # to be reflected.
        self.assertGreaterEqual((chat.usage.cost or 0.0), 0.003)
        self.assertEqual(chat.usage.requests, 2)

    @patch("upsonic.models.infer_model")
    def test_get_usage_returns_aggregated_view_by_default(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response(50, 25))

        agent = Agent(name="A", model=mock_model)
        chat = Chat(session_id="s", user_id="u", agent=agent, storage=InMemoryStorage())
        asyncio.run(chat.invoke("hi"))

        view = chat.usage
        # Returns AggregatedUsage; must have the same shape interface
        # callers used to get from RunUsage (input_tokens, output_tokens,
        # total_tokens, requests, cost).
        self.assertEqual(view.input_tokens, 50)
        self.assertEqual(view.output_tokens, 25)
        self.assertEqual(view.total_tokens, 75)
        self.assertEqual(view.requests, 1)


if __name__ == "__main__":
    unittest.main()
