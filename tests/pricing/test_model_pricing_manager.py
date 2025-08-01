import pytest
import requests
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import sys
import os

# Add src to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from upsonic.models.model_pricing import (
    ModelPricingManager,
    get_dynamic_pricing_for_model,
    get_all_available_openrouter_models,
    refresh_dynamic_pricing,
    is_openrouter_model_available,
    get_estimated_cost_for_tokens,
    _pricing_manager,
)
from upsonic.models.model_registry import (
    get_pricing,
    get_estimated_cost,
)


class TestModelPricingManager:
    """Test cases for ModelPricingManager class."""

    def setup_method(self):
        """Setup method run before each test."""
        self.pricing_manager = ModelPricingManager(cache_duration_hours=1)

    def test_init(self):
        """Test ModelPricingManager initialization."""
        manager = ModelPricingManager(cache_duration_hours=12)
        assert manager._cache == {}
        assert manager._cache_timestamp is None
        assert manager._cache_duration == timedelta(hours=12)
        assert manager._openrouter_base_url == "https://openrouter.ai/api/v1/models"

    def test_cache_validity(self):
        """Test cache validity checking."""
        # Initially invalid (no timestamp)
        assert not self.pricing_manager._is_cache_valid()

        # Set timestamp to now - should be valid
        self.pricing_manager._cache_timestamp = datetime.now()
        assert self.pricing_manager._is_cache_valid()

        # Set timestamp to past beyond cache duration - should be invalid
        self.pricing_manager._cache_timestamp = datetime.now() - timedelta(hours=2)
        assert not self.pricing_manager._is_cache_valid()

    @patch("requests.get")
    def test_fetch_openrouter_models_success(self, mock_get):
        """Test successful fetching of OpenRouter models."""
        # Mock API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "qwen/qwen3-235b-a22b-thinking-2507",
                    "name": "Qwen: Qwen3 235B A22B Thinking 2507",
                    "pricing": {"prompt": "0.0000007", "completion": "0.0000084"},
                    "context_length": 131072,
                    "description": "Test model description",
                }
            ]
        }
        mock_get.return_value = mock_response

        result = self.pricing_manager._fetch_openrouter_models()

        assert result is not None
        assert "qwen/qwen3-235b-a22b-thinking-2507" in result
        model_info = result["qwen/qwen3-235b-a22b-thinking-2507"]
        assert model_info["pricing"]["input"] == 0.7  # 0.0000007 * 1,000,000
        assert (
            abs(model_info["pricing"]["output"] - 8.4) < 0.01
        )  # 0.0000084 * 1,000,000
        assert model_info["context_length"] == 131072
        assert model_info["name"] == "Qwen: Qwen3 235B A22B Thinking 2507"

    @patch("requests.get")
    def test_fetch_openrouter_models_failure(self, mock_get):
        """Test handling of API fetch failures."""
        mock_get.side_effect = requests.RequestException("API error")

        result = self.pricing_manager._fetch_openrouter_models()

        assert result is None

    @patch("requests.get")
    def test_get_dynamic_pricing(self, mock_get):
        """Test getting dynamic pricing with caching."""
        # Mock API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test/model",
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                }
            ]
        }
        mock_get.return_value = mock_response

        # First call should fetch from API
        pricing = self.pricing_manager.get_dynamic_pricing("test/model")
        assert pricing is not None
        assert pricing["input"] == 1.0
        assert pricing["output"] == 2.0
        assert mock_get.call_count == 1

        # Second call should use cache
        pricing2 = self.pricing_manager.get_dynamic_pricing("test/model")
        assert pricing2 == pricing
        assert mock_get.call_count == 1  # No additional API call

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Add some data to cache
        self.pricing_manager._cache = {
            "test": {"pricing": {"input": 1.0, "output": 2.0}}
        }
        self.pricing_manager._cache_timestamp = datetime.now()

        # Clear cache
        self.pricing_manager.clear_cache()

        assert self.pricing_manager._cache == {}
        assert self.pricing_manager._cache_timestamp is None


class TestDynamicPricingIntegration:
    """Integration tests for dynamic pricing functionality."""

    def setup_method(self):
        """Setup method run before each test."""
        # Clear global pricing manager cache
        _pricing_manager.clear_cache()

    @patch("upsonic.models.model_registry.get_dynamic_pricing_for_model")
    def test_get_pricing_with_dynamic_openrouter(self, mock_dynamic_pricing):
        """Test get_pricing function with dynamic OpenRouter pricing."""
        # Mock dynamic pricing return
        mock_dynamic_pricing.return_value = {"input": 1.5, "output": 3.0}

        result = get_pricing("openrouter/test/model")

        assert result == {"input": 1.5, "output": 3.0}
        mock_dynamic_pricing.assert_called_once_with("openrouter/test/model")

    @patch("upsonic.models.model_registry.get_dynamic_pricing_for_model")
    def test_get_pricing_fallback_to_static(self, mock_dynamic_pricing):
        """Test get_pricing falls back to static pricing when dynamic fails."""
        # Mock dynamic pricing failure
        mock_dynamic_pricing.return_value = None

        # Test with a known static model
        result = get_pricing("openai/gpt-4o")

        assert result is not None
        assert "input" in result
        assert "output" in result

    def test_get_pricing_non_openrouter_model(self):
        """Test get_pricing with non-OpenRouter model (should use static pricing)."""
        result = get_pricing("openai/gpt-4o")

        assert result is not None
        assert result == {"input": 2.50, "output": 10.00}

    @patch("upsonic.models.model_registry.get_dynamic_pricing_for_model")
    def test_get_estimated_cost_with_dynamic_pricing(self, mock_dynamic_pricing):
        """Test cost estimation with dynamic pricing."""
        mock_dynamic_pricing.return_value = {"input": 1.0, "output": 2.0}

        cost = get_estimated_cost(100000, 50000, "openrouter/test/model")

        # 100,000 input tokens = 0.1M * $1.0 = $0.10
        # 50,000 output tokens = 0.05M * $2.0 = $0.10
        # Total = $0.20
        assert cost == "~$0.2"

    def test_get_estimated_cost_unknown_model(self):
        """Test cost estimation with unknown model."""
        cost = get_estimated_cost(100000, 50000, "unknown/model")
        assert cost == "Unknown"

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_get_all_available_openrouter_models(self, mock_get_all):
        """Test getting all available OpenRouter models."""
        mock_models = {
            "test/model1": {"pricing": {"input": 1.0, "output": 2.0}},
            "test/model2": {"pricing": {"input": 1.5, "output": 3.0}},
        }
        mock_get_all.return_value = mock_models

        result = get_all_available_openrouter_models()

        assert result == mock_models
        mock_get_all.assert_called_once()

    @patch("upsonic.models.model_pricing._pricing_manager.clear_cache")
    def test_refresh_dynamic_pricing(self, mock_clear_cache):
        """Test manual refresh of dynamic pricing."""
        refresh_dynamic_pricing()
        mock_clear_cache.assert_called_once()

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_is_openrouter_model_available(self, mock_get_all):
        """Test checking if OpenRouter model is available."""
        mock_get_all.return_value = {
            "test/available": {"pricing": {"input": 1.0, "output": 2.0}},
            "test/another": {"pricing": {"input": 1.5, "output": 3.0}},
        }

        assert is_openrouter_model_available("test/available") is True
        assert is_openrouter_model_available("test/unavailable") is False


class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch("requests.get")
    def test_api_timeout_handling(self, mock_get):
        """Test handling of API timeouts."""
        mock_get.side_effect = requests.Timeout("Request timeout")

        manager = ModelPricingManager()
        result = manager.get_dynamic_pricing("test/model")

        assert result is None

    @patch("requests.get")
    def test_invalid_json_response(self, mock_get):
        """Test handling of invalid JSON responses."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        manager = ModelPricingManager()
        result = manager.get_dynamic_pricing("test/model")

        assert result is None

    @patch("requests.get")
    def test_missing_pricing_data(self, mock_get):
        """Test handling of models without pricing data."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test/model",
                    "name": "Test Model",
                    # Missing pricing data
                }
            ]
        }
        mock_get.return_value = mock_response

        manager = ModelPricingManager()
        result = manager._fetch_openrouter_models()

        assert result is not None
        assert "test/model" in result
        # Should have default pricing values
        assert result["test/model"]["pricing"]["input"] == 0.0
        assert result["test/model"]["pricing"]["output"] == 0.0


class TestComprehensiveDynamicPricing:
    """Test the new comprehensive dynamic pricing functionality."""

    def setup_method(self):
        """Setup method run before each test."""
        # Clear global pricing manager cache
        _pricing_manager.clear_cache()

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_claude_model_dynamic_pricing(self, mock_get_all_models):
        """Test that Claude models can now use dynamic pricing."""
        # Mock OpenRouter models data
        mock_get_all_models.return_value = {
            "anthropic/claude-3-sonnet": {
                "pricing": {"input": 3.0, "output": 15.0},
                "name": "Anthropic Claude 3 Sonnet",
            },
            "anthropic/claude-3-5-sonnet-20241022": {
                "pricing": {"input": 3.0, "output": 15.0},
                "name": "Anthropic Claude 3.5 Sonnet",
            },
        }

        # Test claude models that should now get dynamic pricing
        claude_pricing = get_pricing("claude/claude-3-5-sonnet")
        assert claude_pricing is not None
        assert claude_pricing["input"] == 3.0
        assert claude_pricing["output"] == 15.0

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_openai_model_dynamic_pricing(self, mock_get_all_models):
        """Test that OpenAI models can use dynamic pricing when available."""
        mock_get_all_models.return_value = {
            "openai/gpt-4o": {
                "pricing": {"input": 2.5, "output": 10.0},
                "name": "OpenAI GPT-4o",
            },
            "openai/gpt-4o-mini": {
                "pricing": {"input": 0.15, "output": 0.6},
                "name": "OpenAI GPT-4o Mini",
            },
        }

        # Test that OpenAI models now use dynamic pricing when available
        gpt4o_pricing = get_pricing("openai/gpt-4o")
        assert gpt4o_pricing is not None
        assert gpt4o_pricing["input"] == 2.5
        assert gpt4o_pricing["output"] == 10.0

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_gemini_model_dynamic_pricing(self, mock_get_all_models):
        """Test that Gemini models can use dynamic pricing."""
        mock_get_all_models.return_value = {
            "google/gemini-2.0-flash-001": {
                "pricing": {"input": 0.1, "output": 0.4},
                "name": "Google Gemini 2.0 Flash",
            }
        }

        gemini_pricing = get_pricing("gemini/gemini-2.0-flash")
        assert gemini_pricing is not None
        assert gemini_pricing["input"] == 0.1
        assert gemini_pricing["output"] == 0.4

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_fallback_to_static_when_no_match(self, mock_get_all_models):
        """Test that we fall back to static pricing when no OpenRouter match is found."""
        # Mock empty OpenRouter response
        mock_get_all_models.return_value = {}

        # Should fall back to static pricing
        openai_pricing = get_pricing("openai/gpt-4o")
        assert openai_pricing is not None
        assert openai_pricing["input"] == 2.50  # Static pricing
        assert openai_pricing["output"] == 10.00

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_fuzzy_matching(self, mock_get_all_models):
        """Test that fuzzy matching works for similar model names."""
        mock_get_all_models.return_value = {
            "anthropic/claude-3-5-sonnet-20241022": {
                "pricing": {"input": 3.0, "output": 15.0},
                "name": "Anthropic Claude 3.5 Sonnet",
            },
            "some/unrelated-model": {
                "pricing": {"input": 1.0, "output": 2.0},
                "name": "Unrelated Model",
            },
        }

        # Should match claude-3-5-sonnet to claude-3-5-sonnet-20241022
        claude_pricing = get_dynamic_pricing_for_model("claude/claude-3-5-sonnet")
        assert claude_pricing is not None
        assert claude_pricing["input"] == 3.0
        assert claude_pricing["output"] == 15.0

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_exact_match_priority(self, mock_get_all_models):
        """Test that exact matches are prioritized over fuzzy matches."""
        mock_get_all_models.return_value = {
            "openai/gpt-4o": {
                "pricing": {"input": 2.5, "output": 10.0},
                "name": "Exact Match",
            },
            "openai/gpt-4o-extended-version": {
                "pricing": {"input": 5.0, "output": 20.0},
                "name": "Similar Match",
            },
        }

        # Should get the exact match, not the similar one
        pricing = get_dynamic_pricing_for_model("openai/gpt-4o")
        assert pricing is not None
        assert pricing["input"] == 2.5  # Exact match pricing
        assert pricing["output"] == 10.0

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_create_model_mapping(self, mock_get_all_models):
        """Test the model mapping creation functionality."""
        mock_get_all_models.return_value = {
            "openai/gpt-4o": {
                "pricing": {"input": 2.5, "output": 10.0},
                "name": "OpenAI GPT-4o",
            },
            "anthropic/claude-3-sonnet": {
                "pricing": {"input": 3.0, "output": 15.0},
                "name": "Anthropic Claude 3 Sonnet",
            },
        }

        from upsonic.models.model_pricing import create_model_mapping

        mapping = create_model_mapping()

        assert isinstance(mapping, dict)
        # Should find mappings for our test models
        assert len(mapping) > 0


class TestComprehensiveDynamicPricing:
    """Test the new comprehensive dynamic pricing functionality."""

    def setup_method(self):
        """Setup method run before each test."""
        # Clear global pricing manager cache
        _pricing_manager.clear_cache()

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_claude_model_dynamic_pricing(self, mock_get_all_models):
        """Test that Claude models can now use dynamic pricing."""
        # Mock OpenRouter models data
        mock_get_all_models.return_value = {
            "anthropic/claude-3-sonnet": {
                "pricing": {"input": 3.0, "output": 15.0},
                "name": "Anthropic Claude 3 Sonnet",
            },
            "anthropic/claude-3-5-sonnet-20241022": {
                "pricing": {"input": 3.0, "output": 15.0},
                "name": "Anthropic Claude 3.5 Sonnet",
            },
        }

        # Test claude models that should now get dynamic pricing
        claude_pricing = get_pricing("claude/claude-3-5-sonnet")
        assert claude_pricing is not None
        assert claude_pricing["input"] == 3.0
        assert claude_pricing["output"] == 15.0

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_openai_model_dynamic_pricing(self, mock_get_all_models):
        """Test that OpenAI models can use dynamic pricing when available."""
        mock_get_all_models.return_value = {
            "openai/gpt-4o": {
                "pricing": {"input": 2.5, "output": 10.0},
                "name": "OpenAI GPT-4o",
            },
            "openai/gpt-4o-mini": {
                "pricing": {"input": 0.15, "output": 0.6},
                "name": "OpenAI GPT-4o Mini",
            },
        }

        # Test that OpenAI models now use dynamic pricing when available
        gpt4o_pricing = get_pricing("openai/gpt-4o")
        assert gpt4o_pricing is not None
        assert gpt4o_pricing["input"] == 2.5
        assert gpt4o_pricing["output"] == 10.0

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_gemini_model_dynamic_pricing(self, mock_get_all_models):
        """Test that Gemini models can use dynamic pricing."""
        mock_get_all_models.return_value = {
            "google/gemini-2.0-flash-001": {
                "pricing": {"input": 0.1, "output": 0.4},
                "name": "Google Gemini 2.0 Flash",
            }
        }

        gemini_pricing = get_pricing("gemini/gemini-2.0-flash")
        assert gemini_pricing is not None
        assert gemini_pricing["input"] == 0.1
        assert gemini_pricing["output"] == 0.4

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_fallback_to_static_when_no_match(self, mock_get_all_models):
        """Test that we fall back to static pricing when no OpenRouter match is found."""
        # Mock empty OpenRouter response
        mock_get_all_models.return_value = {}

        # Should fall back to static pricing
        openai_pricing = get_pricing("openai/gpt-4o")
        assert openai_pricing is not None
        assert openai_pricing["input"] == 2.50  # Static pricing
        assert openai_pricing["output"] == 10.00

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_fuzzy_matching(self, mock_get_all_models):
        """Test that fuzzy matching works for similar model names."""
        mock_get_all_models.return_value = {
            "anthropic/claude-3-5-sonnet-20241022": {
                "pricing": {"input": 3.0, "output": 15.0},
                "name": "Anthropic Claude 3.5 Sonnet",
            },
            "some/unrelated-model": {
                "pricing": {"input": 1.0, "output": 2.0},
                "name": "Unrelated Model",
            },
        }

        # Should match claude-3-5-sonnet to claude-3-5-sonnet-20241022
        claude_pricing = get_dynamic_pricing_for_model("claude/claude-3-5-sonnet")
        assert claude_pricing is not None
        assert claude_pricing["input"] == 3.0
        assert claude_pricing["output"] == 15.0

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_exact_match_priority(self, mock_get_all_models):
        """Test that exact matches are prioritized over fuzzy matches."""
        mock_get_all_models.return_value = {
            "openai/gpt-4o": {
                "pricing": {"input": 2.5, "output": 10.0},
                "name": "Exact Match",
            },
            "openai/gpt-4o-extended-version": {
                "pricing": {"input": 5.0, "output": 20.0},
                "name": "Similar Match",
            },
        }

        # Should get the exact match, not the similar one
        pricing = get_dynamic_pricing_for_model("openai/gpt-4o")
        assert pricing is not None
        assert pricing["input"] == 2.5  # Exact match pricing
        assert pricing["output"] == 10.0

    @patch("upsonic.models.model_pricing._pricing_manager.get_all_openrouter_models")
    def test_create_model_mapping(self, mock_get_all_models):
        """Test the model mapping creation functionality."""
        mock_get_all_models.return_value = {
            "openai/gpt-4o": {
                "pricing": {"input": 2.5, "output": 10.0},
                "name": "OpenAI GPT-4o",
            },
            "anthropic/claude-3-sonnet": {
                "pricing": {"input": 3.0, "output": 15.0},
                "name": "Anthropic Claude 3 Sonnet",
            },
        }

        from upsonic.models.model_pricing import create_model_mapping

        mapping = create_model_mapping()

        assert isinstance(mapping, dict)
        # Should find mappings for our test models
        assert len(mapping) > 0


if __name__ == "__main__":
    pytest.main([__file__])
