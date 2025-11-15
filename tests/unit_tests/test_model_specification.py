import unittest
from unittest.mock import patch, MagicMock

# Patch GoogleFinishReason.NO_IMAGE before any imports that might use it
try:
    from google.genai.types import FinishReason as GoogleFinishReason
    if not hasattr(GoogleFinishReason, 'NO_IMAGE'):
        GoogleFinishReason.NO_IMAGE = MagicMock()
except (ImportError, AttributeError):
    pass

from upsonic import Agent
from upsonic.models import infer_model
from upsonic.providers.openai import OpenAIProvider
from upsonic.providers.anthropic import AnthropicProvider
from upsonic.providers.google import GoogleProvider

class TestModelSpecification(unittest.TestCase):
    @patch('upsonic.providers.openai.AsyncOpenAI')
    def test_string_based_specifications(self, mock_openai):
        # Mock the OpenAI client
        mock_openai.return_value = MagicMock()
        
        # Test string-based model specifications
        agent1 = Agent(name="String Agent 1", model="openai/gpt-4o")
        self.assertEqual(agent1.model.model_name, "gpt-4o")
        self.assertIsInstance(agent1.model._provider, OpenAIProvider)

    @patch('upsonic.providers.anthropic.AsyncAnthropic')
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    def test_anthropic_specification(self, mock_anthropic):
        # Mock the Anthropic client
        mock_anthropic.return_value = MagicMock()
        
        agent2 = Agent(name="String Agent 2", model="anthropic/claude-3-5-sonnet-latest")
        self.assertEqual(agent2.model.model_name, "claude-3-5-sonnet-latest")
        self.assertIsInstance(agent2.model._provider, AnthropicProvider)

    @patch('upsonic.providers.google.HttpOptions')
    @patch('upsonic.providers.google.Client')
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    def test_google_specification(self, mock_google, mock_http_options):
        # Mock HttpOptions to accept any parameters (including httpx_async_client)
        mock_http_options_instance = MagicMock()
        mock_http_options.return_value = mock_http_options_instance
        
        # Mock the Google client
        mock_google.return_value = MagicMock()
        
        agent3 = Agent(name="String Agent 3", model="google-gla/gemini-2.5-pro")
        self.assertEqual(agent3.model.model_name, "gemini-2.5-pro")
        self.assertIsInstance(agent3.model._provider, GoogleProvider)

    @patch('upsonic.providers.openai.AsyncOpenAI')
    def test_model_inference_direct(self, mock_openai):
        # Mock the OpenAI client
        mock_openai.return_value = MagicMock()
        
        # Test direct model inference
        openai_model = infer_model("openai/gpt-4o")
        
        self.assertEqual(openai_model.model_name, "gpt-4o")
        self.assertIsInstance(openai_model._provider, OpenAIProvider)

    def test_error_handling(self):
        # Test cases that should raise exceptions
        error_cases = [
            ("invalid/gpt-4o", "unknown provider"),
            ("just-a-model-name", "unknown model"),
        ]
        for model_spec, expected_error in error_cases:
            with self.assertRaises(Exception) as excinfo:
                Agent(name="Invalid Agent", model=model_spec)
            self.assertIn(expected_error, str(excinfo.exception).lower())
        
        # Test case that should only show a warning (not raise exception)
        # This tests that invalid model names are handled gracefully
        with patch('upsonic.providers.openai.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = MagicMock()
            try:
                agent = Agent(name="Warning Agent", model="openai/invalid-model")
                # Should not raise exception, just show warning
                self.assertIsNotNone(agent)
                self.assertEqual(agent.model.model_name, "invalid-model")
            except Exception as e:
                self.fail(f"openai/invalid-model should not raise exception, but got: {e}")

if __name__ == "__main__":
    unittest.main()
