"""
Tests for Direct Agent

This module contains comprehensive tests for the Direct agent class,
including initialization, execution methods, builder methods, and error handling.
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from pydantic import BaseModel

import os

from upsonic import Direct, Task
from upsonic.models import ModelResponse, TextPart, ModelRequestParameters
from upsonic.models.settings import ModelSettings
from upsonic.profiles import ModelProfileSpec
from upsonic.providers import Provider


def _real_test_model():
    """Build a real model object for execution tests; the caller patches its
    ``request`` so no network occurs. A genuine ``Model`` instance is required
    because model inference rejects bare mocks. A placeholder key suffices for
    client construction."""
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-placeholder-for-construction")
    from upsonic.models import infer_model
    return infer_model("anthropic/claude-haiku-4-5")


# Mock Pydantic models for testing structured outputs
class MockResponse(BaseModel):
    """Mock response model for testing."""

    name: str
    age: int
    city: str


class MockUsage:
    """Mock usage object."""

    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class TestDirectInitialization(unittest.TestCase):
    """Test suite for Direct class initialization."""

    def test_direct_initialization(self):
        """Test Direct class initialization without parameters."""
        direct = Direct()
        self.assertIsNone(direct._model)
        self.assertIsNone(direct._settings)
        self.assertIsNone(direct._profile)
        self.assertIsNone(direct._provider)

    @patch("upsonic.models.infer_model")
    def test_direct_initialization_with_model(self, mock_infer_model):
        """Test init with model string."""
        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_infer_model.return_value = mock_model

        direct = Direct(model="openai/gpt-4o")

        self.assertIsNotNone(direct._model)
        self.assertEqual(direct._model, mock_model)
        mock_infer_model.assert_called_once_with("openai/gpt-4o")

    def test_direct_initialization_with_model_instance(self):
        """Test init with Model instance."""
        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.request = AsyncMock()

        direct = Direct(model=mock_model)

        self.assertIsNotNone(direct._model)
        self.assertEqual(direct._model, mock_model)

    def test_direct_initialization_with_settings(self):
        """Test init with ModelSettings."""
        mock_settings = MagicMock(spec=ModelSettings)

        direct = Direct(settings=mock_settings)

        self.assertEqual(direct._settings, mock_settings)

    def test_direct_initialization_with_profile(self):
        """Test init with ModelProfileSpec."""
        mock_profile = MagicMock(spec=ModelProfileSpec)

        direct = Direct(profile=mock_profile)

        self.assertEqual(direct._profile, mock_profile)

    def test_direct_initialization_with_provider(self):
        """Test init with Provider."""
        mock_provider = MagicMock(spec=Provider)

        direct = Direct(provider=mock_provider)

        self.assertEqual(direct._provider, mock_provider)

    def test_direct_initialization_with_all_params(self):
        """Test init with all parameters."""
        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.request = AsyncMock()
        mock_settings = MagicMock(spec=ModelSettings)
        mock_profile = MagicMock(spec=ModelProfileSpec)
        mock_provider = MagicMock(spec=Provider)

        direct = Direct(
            model=mock_model,
            settings=mock_settings,
            profile=mock_profile,
            provider=mock_provider,
        )

        self.assertEqual(direct._model, mock_model)
        self.assertEqual(direct._settings, mock_settings)
        self.assertEqual(direct._profile, mock_profile)
        self.assertEqual(direct._provider, mock_provider)


class TestDirectBuilderMethods(unittest.TestCase):
    """Test suite for Direct builder methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MagicMock()
        self.mock_model.model_name = "test-model"
        self.mock_model.request = AsyncMock()
        self.mock_settings = MagicMock(spec=ModelSettings)
        self.mock_profile = MagicMock(spec=ModelProfileSpec)
        self.mock_provider = MagicMock(spec=Provider)

    @patch("upsonic.models.infer_model")
    def test_direct_with_model(self, mock_infer_model):
        """Test with_model() builder method."""
        new_mock_model = MagicMock()
        new_mock_model.model_name = "new-model"
        mock_infer_model.return_value = new_mock_model

        direct = Direct(model=self.mock_model)
        new_direct = direct.with_model("openai/gpt-4o")

        self.assertIsNotNone(new_direct._model)
        self.assertEqual(new_direct._model.model_name, "new-model")
        self.assertIsNot(new_direct, direct)  # Should be a new instance

    def test_direct_with_settings(self):
        """Test with_settings() builder method."""
        new_settings = MagicMock(spec=ModelSettings)

        direct = Direct(model=self.mock_model)
        new_direct = direct.with_settings(new_settings)

        self.assertEqual(new_direct._settings, new_settings)
        self.assertEqual(new_direct._model, self.mock_model)
        self.assertIsNot(new_direct, direct)

    def test_direct_with_profile(self):
        """Test with_profile() builder method."""
        new_profile = MagicMock(spec=ModelProfileSpec)

        direct = Direct(model=self.mock_model)
        new_direct = direct.with_profile(new_profile)

        self.assertEqual(new_direct._profile, new_profile)
        self.assertEqual(new_direct._model, self.mock_model)
        self.assertIsNot(new_direct, direct)

    def test_direct_with_provider(self):
        """Test with_provider() builder method."""
        new_provider = MagicMock(spec=Provider)

        direct = Direct(model=self.mock_model)
        new_direct = direct.with_provider(new_provider)

        self.assertEqual(new_direct._provider, new_provider)
        self.assertEqual(new_direct._model, self.mock_model)
        self.assertIsNot(new_direct, direct)


class TestDirectProperties(unittest.TestCase):
    """Test suite for Direct property accessors."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MagicMock()
        self.mock_settings = MagicMock(spec=ModelSettings)
        self.mock_profile = MagicMock(spec=ModelProfileSpec)
        self.mock_provider = MagicMock(spec=Provider)

    def test_direct_model_property(self):
        """Test model property accessor."""
        direct = Direct(model=self.mock_model)
        self.assertEqual(direct.model, self.mock_model)

    def test_direct_settings_property(self):
        """Test settings property accessor."""
        direct = Direct(settings=self.mock_settings)
        self.assertEqual(direct.settings, self.mock_settings)

    def test_direct_profile_property(self):
        """Test profile property accessor."""
        direct = Direct(profile=self.mock_profile)
        self.assertEqual(direct.profile, self.mock_profile)

    def test_direct_provider_property(self):
        """Test provider property accessor."""
        direct = Direct(provider=self.mock_provider)
        self.assertEqual(direct.provider, self.mock_provider)

    def test_direct_model_name_property(self):
        """model_name returns the model's name, or None when no model is set."""
        self.mock_model.model_name = "test-model"
        direct = Direct(model=self.mock_model)
        self.assertEqual(direct.model_name, "test-model")
        self.assertIsNone(Direct().model_name)


class TestDirectDoMethods(unittest.TestCase):
    """Test suite for Direct do() and related execution methods."""

    def setUp(self):
        """Set up test fixtures."""
        # Real model instance; only `.request` is patched (no network).
        self.mock_model = _real_test_model()

        # Mock response
        self.mock_response = ModelResponse(
            parts=[TextPart(content="Hello, this is a test response.")],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=MockUsage(input_tokens=100, output_tokens=50),
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop",
        )
        self.mock_model.request = AsyncMock(return_value=self.mock_response)

    @patch("upsonic.models.infer_model")
    def test_direct_do_basic(self, mock_infer_model):
        """do() returns the delegated output for a string-specified model."""
        mock_infer_model.return_value = self.mock_model

        direct = Direct(model="openai/gpt-4o")
        result = direct.do(Task("What is 2+2?"))

        self.assertEqual(result, "Hello, this is a test response.")
        self.mock_model.request.assert_called_once()

    def test_direct_do_with_text_task(self):
        """do() returns a non-empty string for a text task."""
        direct = Direct(model=self.mock_model)
        result = direct.do(Task("Tell me a joke"))
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")

    def test_direct_do_with_structured_output(self):
        """do() returns a validated Pydantic model for a structured task."""
        json_response = ModelResponse(
            parts=[TextPart(content='{"name": "John", "age": 30, "city": "New York"}')],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=MockUsage(),
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop",
        )
        self.mock_model.request = AsyncMock(return_value=json_response)

        direct = Direct(model=self.mock_model)
        result = direct.do(Task("Extract user information", response_format=MockResponse))

        self.assertIsInstance(result, MockResponse)
        self.assertEqual(result.name, "John")
        self.assertEqual(result.age, 30)

    def test_direct_do_with_context(self):
        """do() runs with task context and returns the delegated output."""
        direct = Direct(model=self.mock_model)
        result = direct.do(Task("Summarize this", context=["Some context text"]))
        self.assertIsInstance(result, str)
        self.mock_model.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_direct_do_async(self):
        """do_async() returns the delegated output."""
        direct = Direct(model=self.mock_model)
        result = await direct.do_async(Task("Async test task"))
        self.assertEqual(result, "Hello, this is a test response.")

    @pytest.mark.asyncio
    async def test_direct_forwards_agent_print_semantics(self):
        """The facade uses Agent's native print semantics: do()/do_async pass
        ``_print_method_default=False``; print_do()/print_do_async pass ``True``.
        Direct does no print resolution of its own — the Agent owns it."""
        direct = Direct(model=self.mock_model)
        agent = direct._build_internal_agent()
        with patch.object(agent, "do_async", new=AsyncMock(return_value="r")) as mock_do:
            await direct.do_async(Task("quiet"))
            self.assertIs(mock_do.call_args.kwargs.get("_print_method_default"), False)

            await direct.print_do_async(Task("loud"))
            self.assertIs(mock_do.call_args.kwargs.get("_print_method_default"), True)

    def test_direct_print_do_prints_via_agent_pipeline(self):
        """print_do() ⇒ the Agent pipeline's completion printer (call_end) fires."""
        with patch("upsonic.utils.printing.call_end") as mock_call_end:
            direct = Direct(model=self.mock_model)
            result = direct.print_do(Task("Print test"))
        self.assertIsInstance(result, str)
        mock_call_end.assert_called_once()


class TestDirectErrorHandling(unittest.TestCase):
    """Test suite for Direct error handling."""

    def setUp(self):
        """Set up test fixtures."""
        # Real model instance; tests patch `.request` to raise (no network).
        self.mock_model = _real_test_model()

    def test_direct_error_handling(self):
        """A failing run re-raises the original exception via print_do (the
        pipeline does not swallow it; no error panel in non-debug, like Agent)."""
        self.mock_model.request = AsyncMock(side_effect=Exception("Model request failed"))

        direct = Direct(model=self.mock_model)

        with self.assertRaises(Exception) as context:
            direct.print_do(Task("Test task that will fail"))

        self.assertEqual(str(context.exception), "Model request failed")

    def test_direct_error_handling_no_output(self):
        """Test error handling in do() method without output."""
        # Mock model to raise an exception
        error_response = Exception("Model request failed")
        self.mock_model.request = AsyncMock(side_effect=error_response)

        direct = Direct(model=self.mock_model)
        task = Task("Test task that will fail")

        # Test that the exception propagates even when not printing.
        with self.assertRaises(Exception) as context:
            direct.do(task)

        self.assertEqual(str(context.exception), "Model request failed")

    def test_direct_invalid_model_type(self):
        """Test Direct raises error for invalid model type."""
        direct = Direct()

        with self.assertRaises(ValueError):
            direct._set_model(123)  # Invalid type

    @patch("upsonic.models.infer_model")
    def test_direct_set_model_with_string(self, mock_infer_model):
        """Test _set_model() with string."""
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model

        direct = Direct()
        direct._set_model("openai/gpt-4o")

        self.assertEqual(direct._model, mock_model)
        mock_infer_model.assert_called_once_with("openai/gpt-4o")

    def test_direct_set_model_with_model_instance(self):
        """Test _set_model() with Model instance."""
        mock_model = MagicMock()
        mock_model.request = AsyncMock()

        direct = Direct()
        direct._set_model(mock_model)

        self.assertEqual(direct._model, mock_model)


if __name__ == "__main__":
    unittest.main()
