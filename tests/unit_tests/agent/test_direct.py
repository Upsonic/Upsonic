"""
Tests for Direct Agent

This module contains comprehensive tests for the Direct agent class,
including initialization, execution methods, builder methods, and error handling.
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock, mock_open
import pytest
from pydantic import BaseModel

from upsonic import Direct, Task
from upsonic.models import ModelResponse, TextPart, ModelRequestParameters
from upsonic.models.settings import ModelSettings
from upsonic.profiles import ModelProfileSpec
from upsonic.providers import Provider


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


class TestDirectDoMethods(unittest.TestCase):
    """Test suite for Direct do() and related execution methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MagicMock()
        self.mock_model.model_name = "test-model"
        self.mock_model.settings = MagicMock()
        self.mock_model.customize_request_parameters = MagicMock(
            side_effect=lambda x: x
        )

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

    @patch("upsonic.utils.printing.direct_started")
    @patch("upsonic.utils.printing.direct_completed")
    @patch("upsonic.models.infer_model")
    def test_direct_do_basic(self, mock_infer_model, mock_completed, mock_started):
        """Test basic do() method."""
        mock_infer_model.return_value = self.mock_model

        direct = Direct(model="openai/gpt-4o")
        task = Task("What is 2+2?")

        result = direct.do(task, show_output=False)

        self.assertIsInstance(result, str)
        self.assertEqual(result, "Hello, this is a test response.")
        self.mock_model.request.assert_called_once()

    @patch("upsonic.utils.printing.direct_started")
    @patch("upsonic.utils.printing.direct_completed")
    def test_direct_do_with_text_task(self, mock_completed, mock_started):
        """Test do() with simple text task."""
        direct = Direct(model=self.mock_model)
        task = Task("Tell me a joke")

        result = direct.do(task, show_output=False)

        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")

    @patch("upsonic.utils.printing.direct_started")
    @patch("upsonic.utils.printing.direct_completed")
    def test_direct_do_with_structured_output(self, mock_completed, mock_started):
        """Test do() with Pydantic response format."""
        # Mock JSON response
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
        task = Task("Extract user information", response_format=MockResponse)

        result = direct.do(task, show_output=False)

        self.assertIsInstance(result, MockResponse)
        self.assertEqual(result.name, "John")
        self.assertEqual(result.age, 30)
        self.assertEqual(result.city, "New York")

    @patch("upsonic.utils.printing.direct_started")
    @patch("upsonic.utils.printing.direct_completed")
    def test_direct_do_with_context(self, mock_completed, mock_started):
        """Test do() with task context."""
        direct = Direct(model=self.mock_model)
        task = Task("Summarize this", context=["Some context text"])

        result = direct.do(task, show_output=False)

        self.assertIsInstance(result, str)
        # Verify context was included in message building
        self.mock_model.request.assert_called_once()

    @patch("upsonic.utils.printing.direct_started")
    @patch("upsonic.utils.printing.direct_completed")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake image data")
    @patch("mimetypes.guess_type", return_value=("image/png", None))
    def test_direct_do_with_attachments(
        self, mock_guess_type, mock_file, mock_completed, mock_started
    ):
        """Test do() with file attachments."""
        direct = Direct(model=self.mock_model)
        task = Task("Analyze this image", attachments=["/path/to/image.png"])

        result = direct.do(task, show_output=False)

        self.assertIsInstance(result, str)
        mock_file.assert_called_once_with("/path/to/image.png", "rb")

    @patch("upsonic.utils.printing.direct_started")
    @patch("upsonic.utils.printing.direct_completed")
    @pytest.mark.asyncio
    async def test_direct_do_async(self, mock_completed, mock_started):
        """Test async execution."""
        direct = Direct(model=self.mock_model)
        task = Task("Async test task")

        result = await direct.do_async(task, show_output=False)

        self.assertIsInstance(result, str)
        self.assertEqual(result, "Hello, this is a test response.")

    @patch("upsonic.utils.printing.direct_started")
    @patch("upsonic.utils.printing.direct_completed")
    def test_direct_print_do(self, mock_completed, mock_started):
        """Test print_do() method."""
        direct = Direct(model=self.mock_model)
        task = Task("Print test")

        result = direct.print_do(task)

        self.assertIsInstance(result, str)
        # Verify print functions were called
        mock_started.assert_called_once()
        mock_completed.assert_called_once()

    @patch("upsonic.utils.printing.direct_started")
    @patch("upsonic.utils.printing.direct_completed")
    @pytest.mark.asyncio
    async def test_direct_print_do_async(self, mock_completed, mock_started):
        """Test async print_do_async()."""
        direct = Direct(model=self.mock_model)
        task = Task("Async print test")

        result = await direct.print_do_async(task)

        self.assertIsInstance(result, str)
        mock_started.assert_called_once()
        mock_completed.assert_called_once()


class TestDirectInternalMethods(unittest.TestCase):
    """Test suite for Direct internal methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MagicMock()
        self.mock_model.model_name = "test-model"
        self.mock_model.request = AsyncMock()

    @patch("upsonic.models.infer_model")
    def test_direct_model_preparation_with_model(self, mock_infer_model):
        """Test _prepare_model() when model is already set."""
        direct = Direct(model=self.mock_model)

        model = direct._prepare_model()

        self.assertEqual(model, self.mock_model)
        mock_infer_model.assert_not_called()

    @patch("upsonic.models.infer_model")
    def test_direct_model_preparation_without_model(self, mock_infer_model):
        """Test _prepare_model() when model is None."""
        default_model = MagicMock()
        default_model.model_name = "default-model"
        mock_infer_model.return_value = default_model

        direct = Direct()
        model = direct._prepare_model()

        self.assertIsNotNone(model)
        mock_infer_model.assert_called_once_with("openai/gpt-4o")

    @patch("upsonic.models.infer_model")
    def test_direct_model_preparation_with_settings(self, mock_infer_model):
        """Test _prepare_model() with settings applied."""
        mock_settings = MagicMock(spec=ModelSettings)
        default_model = MagicMock()
        default_model.model_name = "default-model"
        default_model._settings = None
        mock_infer_model.return_value = default_model

        direct = Direct(settings=mock_settings)
        model = direct._prepare_model()

        self.assertEqual(model._settings, mock_settings)

    def test_direct_build_messages_from_task(self):
        """Test _build_messages_from_task() method."""
        direct = Direct(model=self.mock_model)
        task = Task("Test task description")

        messages = direct._build_messages_from_task(task)

        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].parts[0].content, "Test task description")

    def test_direct_build_messages_from_task_with_attachments(self):
        """Test _build_messages_from_task() with attachments."""
        direct = Direct(model=self.mock_model)
        task = Task("Test task", attachments=["/path/to/file.txt"])

        with patch("builtins.open", mock_open(read_data=b"file content")):
            with patch("mimetypes.guess_type", return_value=("text/plain", None)):
                messages = direct._build_messages_from_task(task)

        self.assertEqual(len(messages), 1)
        self.assertEqual(len(messages[0].parts), 2)  # UserPromptPart + BinaryContent

    def test_direct_build_request_parameters_text(self):
        """Test _build_request_parameters() for text output."""
        direct = Direct(model=self.mock_model)
        task = Task("Test task", response_format=str)

        params = direct._build_request_parameters(task)

        self.assertIsInstance(params, ModelRequestParameters)
        self.assertEqual(params.output_mode, "text")
        self.assertIsNone(params.output_object)

    def test_direct_build_request_parameters_structured(self):
        """Test _build_request_parameters() for structured output."""
        direct = Direct(model=self.mock_model)
        task = Task("Test task", response_format=MockResponse)

        params = direct._build_request_parameters(task)

        self.assertIsInstance(params, ModelRequestParameters)
        self.assertEqual(params.output_mode, "native")
        self.assertIsNotNone(params.output_object)
        self.assertEqual(params.output_object.name, "MockResponse")

    def test_direct_extract_output_text(self):
        """Test _extract_output() for text response."""
        direct = Direct(model=self.mock_model)
        task = Task("Test task", response_format=str)

        response = ModelResponse(
            parts=[TextPart(content="Test response text")],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop",
        )

        result = direct._extract_output(response, task)

        self.assertEqual(result, "Test response text")

    def test_direct_extract_output_structured(self):
        """Test _extract_output() for structured response."""
        direct = Direct(model=self.mock_model)
        task = Task("Test task", response_format=MockResponse)

        response = ModelResponse(
            parts=[TextPart(content='{"name": "Alice", "age": 25, "city": "Boston"}')],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop",
        )

        result = direct._extract_output(response, task)

        self.assertIsInstance(result, MockResponse)
        self.assertEqual(result.name, "Alice")
        self.assertEqual(result.age, 25)
        self.assertEqual(result.city, "Boston")

    def test_direct_extract_output_invalid_json(self):
        """Test _extract_output() with invalid JSON falls back to text."""
        direct = Direct(model=self.mock_model)
        task = Task("Test task", response_format=MockResponse)

        response = ModelResponse(
            parts=[TextPart(content="Not valid JSON")],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop",
        )

        result = direct._extract_output(response, task)

        # Should return text content when JSON parsing fails
        self.assertEqual(result, "Not valid JSON")


class TestDirectErrorHandling(unittest.TestCase):
    """Test suite for Direct error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MagicMock()
        self.mock_model.model_name = "test-model"
        self.mock_model.settings = MagicMock()
        self.mock_model.customize_request_parameters = MagicMock(
            side_effect=lambda x: x
        )

    @patch("upsonic.utils.printing.direct_error")
    @patch("upsonic.utils.printing.direct_started")
    def test_direct_error_handling(self, mock_started, mock_error):
        """Test error handling in do() method."""
        # Mock model to raise an exception
        error_response = Exception("Model request failed")
        self.mock_model.request = AsyncMock(side_effect=error_response)

        direct = Direct(model=self.mock_model)
        task = Task("Test task that will fail")

        # Test that exception is raised and direct_error is called when show_output=True
        with self.assertRaises(Exception):
            direct.do(task, show_output=True)

        mock_error.assert_called_once()
        mock_started.assert_called_once()

    def test_direct_error_handling_no_output(self):
        """Test error handling in do() method without output."""
        # Mock model to raise an exception
        error_response = Exception("Model request failed")
        self.mock_model.request = AsyncMock(side_effect=error_response)

        direct = Direct(model=self.mock_model)
        task = Task("Test task that will fail")

        # Test that exception is raised even when show_output=False
        with self.assertRaises(Exception) as context:
            direct.do(task, show_output=False)

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
