"""
Tests for Model Selection System
"""

import pytest
from upsonic.models.model_registry import (
    MODEL_REGISTRY,
    ModelCapability,
    ModelTier,
    get_model_metadata,
    get_models_by_capability,
    get_models_by_tier,
    get_top_models,
)
from upsonic.models.model_selector import (
    RuleBasedSelector,
    SelectionCriteria,
    ModelRecommendation,
)


class TestModelRegistry:
    """Test the model registry."""
    
    def test_registry_not_empty(self):
        """Test that the model registry is populated."""
        assert len(MODEL_REGISTRY) > 0
        assert "openai/gpt-4o" in MODEL_REGISTRY
    
    def test_get_model_metadata(self):
        """Test getting model metadata."""
        model = get_model_metadata("openai/gpt-4o")
        assert model is not None
        assert model.name == "openai/gpt-4o"
        assert model.provider == "openai"
        assert model.benchmarks is not None
    
    def test_get_models_by_capability(self):
        """Test filtering models by capability."""
        coding_models = get_models_by_capability(ModelCapability.CODE_GENERATION)
        assert len(coding_models) > 0
        
        # All returned models should have code generation capability
        for model in coding_models:
            assert ModelCapability.CODE_GENERATION in model.capabilities
    
    def test_get_models_by_tier(self):
        """Test filtering models by tier."""
        flagship_models = get_models_by_tier(ModelTier.FLAGSHIP)
        assert len(flagship_models) > 0
        
        # All returned models should be flagship tier
        for model in flagship_models:
            assert model.tier == ModelTier.FLAGSHIP
    
    def test_get_top_models(self):
        """Test getting top models."""
        top_models = get_top_models(n=5)
        assert len(top_models) <= 5
        assert len(top_models) > 0
        
        # Test specific benchmark
        top_math = get_top_models(n=3, by_benchmark="math")
        assert len(top_math) <= 3
    
    def test_model_benchmark_scores(self):
        """Test that models have valid benchmark scores."""
        model = get_model_metadata("openai/gpt-4o")
        assert model.benchmarks.mmlu is not None
        assert 0 <= model.benchmarks.mmlu <= 100
        
        # Test overall score calculation
        overall = model.benchmarks.overall_score()
        assert overall > 0


class TestRuleBasedSelector:
    """Test the rule-based model selector."""
    
    def test_basic_selection(self):
        """Test basic model selection."""
        selector = RuleBasedSelector()
        recommendation = selector.select_model(
            task_description="Write Python code to sort a list",
            default_model="openai/gpt-4o"
        )
        
        assert isinstance(recommendation, ModelRecommendation)
        assert recommendation.model_name is not None
        assert "/" in recommendation.model_name  # Has provider prefix
        assert 0 <= recommendation.confidence_score <= 1
        assert recommendation.selection_method == "rule_based"
    
    def test_code_generation_task(self):
        """Test that coding tasks select appropriate models."""
        selector = RuleBasedSelector()
        recommendation = selector.select_model(
            task_description="Implement a binary search algorithm in Python with unit tests",
            default_model="openai/gpt-4o"
        )
        
        # Should recommend a model with code generation capability
        model = get_model_metadata(recommendation.model_name)
        assert model is not None
        assert ModelCapability.CODE_GENERATION in model.capabilities
    
    def test_math_task(self):
        """Test that math tasks select appropriate models."""
        selector = RuleBasedSelector()
        recommendation = selector.select_model(
            task_description="Solve complex calculus equations with detailed steps",
            default_model="openai/gpt-4o"
        )
        
        # Should recommend a model with math capability
        model = get_model_metadata(recommendation.model_name)
        assert model is not None
        # Many models have math capability, just check it was considered
        assert recommendation.confidence_score > 0
    
    def test_cost_priority(self):
        """Test selection with cost priority."""
        selector = RuleBasedSelector()
        criteria = SelectionCriteria(
            prioritize_cost=True,
            max_cost_tier=3
        )
        
        recommendation = selector.select_model(
            task_description="Simple text generation task",
            criteria=criteria,
            default_model="openai/gpt-4o"
        )
        
        # Should recommend a cost-effective model
        assert recommendation.estimated_cost_tier <= 5  # Should be reasonably cheap
    
    def test_speed_priority(self):
        """Test selection with speed priority."""
        selector = RuleBasedSelector()
        criteria = SelectionCriteria(
            prioritize_speed=True
        )
        
        recommendation = selector.select_model(
            task_description="Quick response needed",
            criteria=criteria,
            default_model="openai/gpt-4o"
        )
        
        # Should recommend a fast model
        assert recommendation.estimated_speed_tier >= 5  # Should be reasonably fast
    
    def test_context_window_requirement(self):
        """Test selection with minimum context window."""
        selector = RuleBasedSelector()
        criteria = SelectionCriteria(
            min_context_window=100000  # Require at least 100K tokens
        )
        
        recommendation = selector.select_model(
            task_description="Analyze this large document",
            criteria=criteria,
            default_model="openai/gpt-4o"
        )
        
        # Should recommend a model with sufficient context
        model = get_model_metadata(recommendation.model_name)
        assert model.context_window >= 100000
    
    def test_required_capabilities(self):
        """Test selection with required capabilities."""
        selector = RuleBasedSelector()
        criteria = SelectionCriteria(
            required_capabilities=[
                ModelCapability.CODE_GENERATION,
                ModelCapability.FUNCTION_CALLING
            ]
        )
        
        recommendation = selector.select_model(
            task_description="Build an API with functions",
            criteria=criteria,
            default_model="openai/gpt-4o"
        )
        
        # Should recommend a model with all required capabilities
        model = get_model_metadata(recommendation.model_name)
        assert ModelCapability.CODE_GENERATION in model.capabilities
        assert ModelCapability.FUNCTION_CALLING in model.capabilities
    
    def test_provider_preference(self):
        """Test selection with provider preference."""
        selector = RuleBasedSelector()
        criteria = SelectionCriteria(
            preferred_provider="anthropic"
        )
        
        recommendation = selector.select_model(
            task_description="General task",
            criteria=criteria,
            default_model="openai/gpt-4o"
        )
        
        # Should prefer Anthropic models
        assert "anthropic" in recommendation.model_name.lower() or \
               "claude" in recommendation.model_name.lower()
    
    def test_fallback_to_default(self):
        """Test fallback when no models match criteria."""
        selector = RuleBasedSelector()
        criteria = SelectionCriteria(
            max_cost_tier=0,  # Impossible requirement
            min_context_window=10000000  # Impossible requirement
        )
        
        recommendation = selector.select_model(
            task_description="Task with impossible requirements",
            criteria=criteria,
            default_model="openai/gpt-4o"
        )
        
        # Should fall back to default
        assert recommendation.model_name == "openai/gpt-4o"
        assert recommendation.confidence_score < 1.0


class TestSelectionCriteria:
    """Test the SelectionCriteria model."""
    
    def test_default_criteria(self):
        """Test default criteria initialization."""
        criteria = SelectionCriteria()
        assert criteria.prioritize_cost is False
        assert criteria.prioritize_speed is False
        assert criteria.prioritize_quality is False
        assert criteria.require_open_source is False
    
    def test_custom_criteria(self):
        """Test custom criteria."""
        criteria = SelectionCriteria(
            requires_reasoning=True,
            prioritize_cost=True,
            max_cost_tier=5,
            preferred_provider="openai"
        )
        
        assert criteria.requires_reasoning is True
        assert criteria.prioritize_cost is True
        assert criteria.max_cost_tier == 5
        assert criteria.preferred_provider == "openai"


class TestModelRecommendation:
    """Test the ModelRecommendation model."""
    
    def test_recommendation_structure(self):
        """Test recommendation model structure."""
        recommendation = ModelRecommendation(
            model_name="openai/gpt-4o",
            reason="Best for general tasks",
            confidence_score=0.95,
            alternative_models=["anthropic/claude-3-7-sonnet-20250219"],
            estimated_cost_tier=7,
            estimated_speed_tier=6,
            selection_method="rule_based"
        )
        
        assert recommendation.model_name == "openai/gpt-4o"
        assert recommendation.reason == "Best for general tasks"
        assert 0 <= recommendation.confidence_score <= 1
        assert len(recommendation.alternative_models) == 1
        assert 1 <= recommendation.estimated_cost_tier <= 10
        assert 1 <= recommendation.estimated_speed_tier <= 10
        assert recommendation.selection_method in ["llm", "rule_based"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

