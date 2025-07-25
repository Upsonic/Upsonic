import os
import requests
import time
from datetime import datetime, timedelta
from threading import Lock, Timer
from decimal import Decimal
from typing import Dict, Optional, Any


class ModelPricingManager:
    """
    Manages dynamic pricing information for models with caching and fallback mechanisms.
    Follows OOP principles with proper encapsulation and responsibility separation.
    """

    def __init__(self, cache_duration_hours: int = 24, auto_refresh: bool = False):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(hours=cache_duration_hours)
        self._lock = Lock()
        self._openrouter_base_url = "https://openrouter.ai/api/v1/models"
        self._auto_refresh = auto_refresh
        self._refresh_timer: Optional[Timer] = None

    def _is_cache_valid(self) -> bool:
        """Check if the current cache is still valid."""
        if not self._cache_timestamp:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_duration

    def _schedule_auto_refresh(self):
        """Schedule automatic cache refresh for long-running agents."""
        if not self._auto_refresh:
            return

        # Cancel any existing timer
        if self._refresh_timer:
            self._refresh_timer.cancel()

        # Schedule refresh for half the cache duration (e.g., 12 hours for 24h cache)
        refresh_interval = self._cache_duration.total_seconds() / 2

        def auto_refresh():
            try:
                self._update_cache()
                # Schedule next refresh
                self._schedule_auto_refresh()
            except Exception as e:
                pass  # Silent fail for auto-refresh

        self._refresh_timer = Timer(refresh_interval, auto_refresh)
        self._refresh_timer.daemon = True  # Don't prevent program exit
        self._refresh_timer.start()

    def enable_auto_refresh(self, enabled: bool = True):
        """Enable or disable automatic cache refresh for long-running agents."""
        self._auto_refresh = enabled
        if enabled:
            self._schedule_auto_refresh()
        else:
            if self._refresh_timer:
                self._refresh_timer.cancel()
                self._refresh_timer = None

    def _fetch_openrouter_models(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Fetch all available models from OpenRouter API with pricing and extended information.
        Returns a dictionary mapping model IDs to their information.
        """
        try:
            # 3 second timeout for API requests
            response = requests.get(self._openrouter_base_url, timeout=3)
            response.raise_for_status()

            data = response.json()
            models_dict = {}

            for model in data.get("data", []):
                model_id = model.get("id", "")
                if not model_id:
                    continue

                # Extract pricing information
                pricing = model.get("pricing", {})
                top_provider = model.get("top_provider", {})
                architecture = model.get("architecture", {})

                # Basic pricing per million tokens
                pricing_info = {
                    "input": float(pricing.get("prompt", "0")) * 1000000,
                    "output": float(pricing.get("completion", "0")) * 1000000,
                }

                # Extended pricing information
                extended_pricing = {}
                if pricing.get("image"):
                    extended_pricing["image"] = float(pricing.get("image", "0"))
                if pricing.get("web_search"):
                    extended_pricing["web_search"] = float(
                        pricing.get("web_search", "0")
                    )
                if pricing.get("input_cache_read"):
                    extended_pricing["input_cache_read"] = (
                        float(pricing.get("input_cache_read", "0")) * 1000000
                    )
                if pricing.get("input_cache_write"):
                    extended_pricing["input_cache_write"] = (
                        float(pricing.get("input_cache_write", "0")) * 1000000
                    )
                if pricing.get("internal_reasoning"):
                    extended_pricing["internal_reasoning"] = float(
                        pricing.get("internal_reasoning", "0")
                    )
                if pricing.get("request"):
                    extended_pricing["request"] = float(pricing.get("request", "0"))

                models_dict[model_id] = {
                    "pricing": pricing_info,
                    "extended_pricing": extended_pricing,
                    "context_length": model.get("context_length", 0),
                    "max_completion_tokens": top_provider.get(
                        "max_completion_tokens", 0
                    ),
                    "description": model.get("description", ""),
                    "name": model.get("name", ""),
                    "modality": architecture.get("modality", "text->text"),
                    "input_modalities": architecture.get("input_modalities", ["text"]),
                    "output_modalities": architecture.get(
                        "output_modalities", ["text"]
                    ),
                    "supported_parameters": model.get("supported_parameters", []),
                    "created": model.get("created", 0),
                    "canonical_slug": model.get("canonical_slug", ""),
                    "is_moderated": top_provider.get("is_moderated", False),
                }

            return models_dict

        except (requests.RequestException, requests.Timeout, ValueError, KeyError) as e:
            return None

    def _update_cache(self) -> None:
        """Update the pricing cache from OpenRouter API."""
        openrouter_models = self._fetch_openrouter_models()
        if openrouter_models:
            with self._lock:
                self._cache.update(openrouter_models)
                self._cache_timestamp = datetime.now()

            # Schedule auto-refresh if enabled
            if self._auto_refresh:
                self._schedule_auto_refresh()

    def get_dynamic_pricing(self, model_id: str) -> Optional[Dict[str, float]]:
        """
        Get dynamic pricing for a model. Returns cached data if available and valid,
        otherwise fetches fresh data from OpenRouter API.
        """
        # Check if cache needs refresh
        if not self._is_cache_valid():
            self._update_cache()

        # Return pricing if available in cache
        with self._lock:
            model_info = self._cache.get(model_id)
            if model_info and "pricing" in model_info:
                return model_info["pricing"]

        return None

    def get_all_openrouter_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available OpenRouter models with their information."""
        if not self._is_cache_valid():
            self._update_cache()

        with self._lock:
            return self._cache.copy()

    def clear_cache(self) -> None:
        """Manually clear the pricing cache."""
        with self._lock:
            self._cache.clear()
            self._cache_timestamp = None


# Global pricing manager instance
_pricing_manager = ModelPricingManager()


def initialize_agent_pricing(enable_auto_refresh: bool = True):
    """
    Initialize pricing system for agent usage.
    Always attempts dynamic pricing first, falls back to static if OpenRouter Models API fails.

    Args:
        enable_auto_refresh: Whether to enable automatic cache refresh for long-running agents
    """
    global _pricing_manager

    # Clear existing cache to force fresh data
    _pricing_manager.clear_cache()

    # Enable auto-refresh for long-running agents
    if enable_auto_refresh:
        _pricing_manager.enable_auto_refresh(True)

    # Try to pre-load the cache with fresh OpenRouter data (3s timeout)
    try:
        models = _pricing_manager.get_all_openrouter_models()
        return True
    except Exception as e:
        return False


def refresh_pricing_cache():
    """
    Manually refresh the pricing cache with latest OpenRouter data.
    Useful for long-running agents.
    """
    global _pricing_manager
    _pricing_manager.clear_cache()
    try:
        models = _pricing_manager.get_all_openrouter_models()
        return True
    except Exception as e:
        return False


def get_dynamic_pricing_for_model(llm_model: str) -> Optional[Dict[str, float]]:
    """
    Get dynamic pricing for a model with comprehensive OpenRouter matching.
    This function now checks ALL models against OpenRouter, not just openrouter/ prefixed ones.
    """
    # Get all available OpenRouter models
    all_openrouter_models = _pricing_manager.get_all_openrouter_models()

    if not all_openrouter_models:
        return None

    # Strategy 1: Handle openrouter/ prefixed models (existing behavior)
    if llm_model.startswith("openrouter/"):
        model_name = llm_model.split("openrouter/", 1)[1]
        dynamic_pricing = _pricing_manager.get_dynamic_pricing(model_name)
        if dynamic_pricing:
            return dynamic_pricing

    # Strategy 2: Extract model name and search for exact matches
    if "/" in llm_model:
        provider, model_name = llm_model.split("/", 1)

        # Try exact match first
        if model_name in all_openrouter_models:
            return all_openrouter_models[model_name]["pricing"]

        # Try with provider prefix
        full_name = f"{provider}/{model_name}"
        if full_name in all_openrouter_models:
            return all_openrouter_models[full_name]["pricing"]
    else:
        model_name = llm_model

    # Strategy 3: Fuzzy matching - find best match
    best_match = None
    best_score = 0

    for openrouter_id, openrouter_info in all_openrouter_models.items():
        # Calculate similarity score
        score = 0

        # Exact model name match gets highest score
        if model_name.lower() in openrouter_id.lower():
            score = len(model_name) / len(openrouter_id) * 100
        elif openrouter_id.lower() in model_name.lower():
            score = len(openrouter_id) / len(model_name) * 100

        # Boost score for similar names
        model_parts = model_name.lower().replace("-", " ").split()
        openrouter_parts = (
            openrouter_id.lower().replace("-", " ").replace("/", " ").split()
        )

        common_parts = set(model_parts) & set(openrouter_parts)
        if common_parts:
            score += len(common_parts) * 10

        # Prefer non-free versions if available (unless specifically looking for free)
        if ":free" not in openrouter_id and "free" not in model_name.lower():
            score += 5
        elif ":free" in openrouter_id and "free" in model_name.lower():
            score += 10

        if score > best_score and score > 20:  # Minimum threshold for matching
            best_score = score
            best_match = openrouter_info["pricing"]

    return best_match


def get_extended_pricing_for_model(llm_model: str) -> Optional[Dict[str, Any]]:
    """
    Get extended pricing information for a model including image, cache, etc.
    """
    all_openrouter_models = _pricing_manager.get_all_openrouter_models()

    if not all_openrouter_models:
        return None

    # Use the same matching logic as get_dynamic_pricing_for_model
    if llm_model.startswith("openrouter/"):
        model_name = llm_model.split("openrouter/", 1)[1]
        if model_name in all_openrouter_models:
            return all_openrouter_models[model_name].get("extended_pricing", {})

    # Try exact matches and fuzzy matching
    if "/" in llm_model:
        provider, model_name = llm_model.split("/", 1)

        # Try exact match first
        if model_name in all_openrouter_models:
            return all_openrouter_models[model_name].get("extended_pricing", {})

        # Try with provider prefix
        full_name = f"{provider}/{model_name}"
        if full_name in all_openrouter_models:
            return all_openrouter_models[full_name].get("extended_pricing", {})

    # Fuzzy matching for extended pricing
    best_match = None
    best_score = 0
    model_name = llm_model.split("/", 1)[-1] if "/" in llm_model else llm_model

    for openrouter_id, openrouter_info in all_openrouter_models.items():
        score = 0

        if model_name.lower() in openrouter_id.lower():
            score = len(model_name) / len(openrouter_id) * 100
        elif openrouter_id.lower() in model_name.lower():
            score = len(openrouter_id) / len(model_name) * 100

        if score > best_score and score > 20:
            best_score = score
            best_match = openrouter_info.get("extended_pricing", {})

    return best_match


def get_model_capabilities(llm_model: str) -> Optional[Dict[str, Any]]:
    """
    Get model capabilities including context length, modalities, supported parameters, etc.
    """
    all_openrouter_models = _pricing_manager.get_all_openrouter_models()

    if not all_openrouter_models:
        return None

    # Use the same matching logic
    if llm_model.startswith("openrouter/"):
        model_name = llm_model.split("openrouter/", 1)[1]
        if model_name in all_openrouter_models:
            model_info = all_openrouter_models[model_name]
            return {
                "context_length": model_info.get("context_length", 0),
                "max_completion_tokens": model_info.get("max_completion_tokens", 0),
                "modality": model_info.get("modality", "text->text"),
                "input_modalities": model_info.get("input_modalities", ["text"]),
                "output_modalities": model_info.get("output_modalities", ["text"]),
                "supported_parameters": model_info.get("supported_parameters", []),
                "is_moderated": model_info.get("is_moderated", False),
                "name": model_info.get("name", ""),
                "description": model_info.get("description", ""),
            }

    # Fuzzy matching for capabilities
    best_match = None
    best_score = 0
    model_name = llm_model.split("/", 1)[-1] if "/" in llm_model else llm_model

    for openrouter_id, openrouter_info in all_openrouter_models.items():
        score = 0

        if model_name.lower() in openrouter_id.lower():
            score = len(model_name) / len(openrouter_id) * 100
        elif openrouter_id.lower() in model_name.lower():
            score = len(openrouter_id) / len(model_name) * 100

        if score > best_score and score > 20:
            best_score = score
            best_match = {
                "context_length": openrouter_info.get("context_length", 0),
                "max_completion_tokens": openrouter_info.get(
                    "max_completion_tokens", 0
                ),
                "modality": openrouter_info.get("modality", "text->text"),
                "input_modalities": openrouter_info.get("input_modalities", ["text"]),
                "output_modalities": openrouter_info.get("output_modalities", ["text"]),
                "supported_parameters": openrouter_info.get("supported_parameters", []),
                "is_moderated": openrouter_info.get("is_moderated", False),
                "name": openrouter_info.get("name", ""),
                "description": openrouter_info.get("description", ""),
            }

    return best_match


def search_models_by_capability(
    capability: str, provider: str = None
) -> Dict[str, Dict[str, Any]]:
    """
    Search for models that support specific capabilities.

    Args:
        capability: Can be 'multimodal', 'image', 'reasoning', 'tools', etc.
        provider: Optional provider filter like 'anthropic', 'openai', etc.

    Returns:
        Dictionary of matching models with their information
    """
    all_openrouter_models = _pricing_manager.get_all_openrouter_models()

    if not all_openrouter_models:
        return {}

    matching_models = {}

    for model_id, model_info in all_openrouter_models.items():
        # Provider filter
        if provider and not model_id.lower().startswith(provider.lower()):
            continue

        # Capability matching
        match = False

        if capability.lower() == "multimodal":
            if "image" in model_info.get("input_modalities", []):
                match = True
        elif capability.lower() == "image":
            if "image" in model_info.get("input_modalities", []):
                match = True
        elif capability.lower() == "reasoning":
            if "reasoning" in model_info.get("supported_parameters", []):
                match = True
        elif capability.lower() == "tools":
            if "tools" in model_info.get("supported_parameters", []):
                match = True
        elif capability.lower() == "cache":
            if model_info.get("extended_pricing", {}).get("input_cache_read"):
                match = True
        else:
            # Generic search in supported parameters
            if capability in model_info.get("supported_parameters", []):
                match = True

        if match:
            matching_models[model_id] = model_info

    return matching_models


def create_model_mapping() -> Dict[str, str]:
    """
    Create a mapping of static model IDs to their best OpenRouter equivalents.
    This helps with faster lookups and debugging.
    """
    all_openrouter_models = _pricing_manager.get_all_openrouter_models()
    if not all_openrouter_models:
        return {}

    # Import here to avoid circular imports
    from .model_registry import MODEL_REGISTRY

    mapping = {}
    for static_model_id in MODEL_REGISTRY.keys():
        if "/" in static_model_id:
            provider, model_name = static_model_id.split("/", 1)

            # Find best match
            best_match = None
            best_score = 0

            for openrouter_id in all_openrouter_models.keys():
                score = 0

                if model_name.lower() in openrouter_id.lower():
                    score = len(model_name) / len(openrouter_id) * 100
                elif openrouter_id.lower() in model_name.lower():
                    score = len(openrouter_id) / len(model_name) * 100

                # Add bonus for exact provider match
                if provider.lower() in openrouter_id.lower():
                    score += 20

                if score > best_score and score > 30:
                    best_score = score
                    best_match = openrouter_id

            if best_match:
                mapping[static_model_id] = best_match

    return mapping


def get_all_available_openrouter_models() -> Dict[str, Dict[str, Any]]:
    """
    Get all available models from OpenRouter API.
    This provides access to models not in the static registry.
    """
    return _pricing_manager.get_all_openrouter_models()


def refresh_dynamic_pricing():
    """
    Manually refresh the dynamic pricing cache.
    Useful for testing or when immediate price updates are needed.
    """
    _pricing_manager.clear_cache()


def is_openrouter_model_available(model_name: str) -> bool:
    """
    Check if a specific OpenRouter model is available via the API.
    """
    all_models = _pricing_manager.get_all_openrouter_models()
    return model_name in all_models


def get_estimated_cost_for_tokens(
    input_tokens: int, output_tokens: int, pricing: Dict[str, float]
) -> str:
    """
    Calculate estimated cost for token usage with given pricing.
    Separated for better testability and reusability.
    """
    if not pricing:
        return "Unknown"

    # Convert token counts to millions for pricing calculation using Decimal
    input_tokens_millions = Decimal(str(input_tokens)) / Decimal("1000000")
    output_tokens_millions = Decimal(str(output_tokens)) / Decimal("1000000")

    input_cost = Decimal(str(pricing["input"])) * input_tokens_millions
    output_cost = Decimal(str(pricing["output"])) * output_tokens_millions
    total = input_cost + output_cost

    return f"~${float(round(total, 4))}"
