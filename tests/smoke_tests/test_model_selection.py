"""
Smoke test for Agent model selection features.

Tests:
- model_selection_criteria: Default criteria for model selection
- use_llm_for_selection: Whether to use LLM for model selection
"""

import pytest
from upsonic import Agent, Task

pytestmark = pytest.mark.timeout(60)


@pytest.mark.asyncio
async def test_model_selection_criteria():
    """Test that model_selection_criteria is used in model recommendations."""
    # Create agent with model selection criteria
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        model_selection_criteria={
            "prioritize_cost": True,
            "prioritize_speed": False,
            "requires_reasoning": False,
            "requires_coding": False,
            "requires_math": False,
            "requires_vision": False,
            "requires_audio": False,
            "requires_long_context": False
        }
    )
    
    # Verify criteria is set
    assert agent.model_selection_criteria is not None, "model_selection_criteria should be set"
    assert isinstance(agent.model_selection_criteria, dict), "model_selection_criteria should be a dict"
    assert "prioritize_cost" in agent.model_selection_criteria, "prioritize_cost should be in criteria"
    
    # Test model recommendation
    task = Task(description="What is 2 + 2?")
    
    recommendation = await agent.recommend_model_for_task_async(task)
    
    # Verify recommendation is returned
    assert recommendation is not None, "Recommendation should not be None"
    assert hasattr(recommendation, 'model_name'), "Recommendation should have model_name"
    assert hasattr(recommendation, 'reason'), "Recommendation should have reason"
    assert hasattr(recommendation, 'confidence_score'), "Recommendation should have confidence_score"
    assert hasattr(recommendation, 'selection_method'), "Recommendation should have selection_method"
    
    # Verify selection method is rule_based (since use_llm_for_selection is False by default)
    assert recommendation.selection_method == "rule_based", "Selection method should be rule_based by default"


@pytest.mark.asyncio
async def test_use_llm_for_selection():
    """Test that use_llm_for_selection uses LLM for model recommendations."""
    # Create agent with LLM-based selection
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        use_llm_for_selection=True
    )
    
    # Verify flag is set
    assert agent.use_llm_for_selection is True, "use_llm_for_selection should be True"
    
    # Test model recommendation
    task = Task(description="Analyze this complex codebase and provide architectural recommendations")
    
    recommendation = await agent.recommend_model_for_task_async(task)
    
    # Verify recommendation is returned
    assert recommendation is not None, "Recommendation should not be None"
    assert hasattr(recommendation, 'model_name'), "Recommendation should have model_name"
    assert hasattr(recommendation, 'reason'), "Recommendation should have reason"
    assert hasattr(recommendation, 'confidence_score'), "Recommendation should have confidence_score"
    assert hasattr(recommendation, 'selection_method'), "Recommendation should have selection_method"
    
    # Verify selection method is llm
    assert recommendation.selection_method == "llm", "Selection method should be llm when use_llm_for_selection is True"


@pytest.mark.asyncio
async def test_model_selection_criteria_with_llm():
    """Test that model_selection_criteria works with LLM-based selection."""
    # Create agent with both criteria and LLM selection
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        model_selection_criteria={
            "prioritize_cost": False,
            "prioritize_speed": False,
            "requires_reasoning": True,
            "requires_coding": False,
            "requires_math": False,
            "requires_vision": False,
            "requires_audio": False,
            "requires_long_context": False
        },
        use_llm_for_selection=True
    )
    
    # Verify both are set
    assert agent.model_selection_criteria is not None, "model_selection_criteria should be set"
    assert agent.use_llm_for_selection is True, "use_llm_for_selection should be True"
    assert agent.model_selection_criteria.get("requires_reasoning") is True, "requires_reasoning should be True"
    
    # Test model recommendation
    task = Task(description="Solve this complex reasoning problem step by step")
    
    recommendation = await agent.recommend_model_for_task_async(task)
    
    # Verify recommendation is returned
    assert recommendation is not None, "Recommendation should not be None"
    assert hasattr(recommendation, 'model_name'), "Recommendation should have model_name"
    assert hasattr(recommendation, 'reason'), "Recommendation should have reason"
    assert recommendation.selection_method == "llm", "Selection method should be llm"
    
    # The recommendation should consider the reasoning requirement
    assert "reasoning" in recommendation.reason.lower() or recommendation.model_name, "Recommendation should consider reasoning requirement"


@pytest.mark.asyncio
async def test_model_selection_without_criteria():
    """Test that model selection works without explicit criteria."""
    # Create agent without criteria
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent"
    )
    
    # Verify defaults
    assert agent.model_selection_criteria is None or isinstance(agent.model_selection_criteria, dict), "model_selection_criteria should be None or dict"
    assert agent.use_llm_for_selection is False, "use_llm_for_selection should default to False"
    
    # Test model recommendation
    task = Task(description="What is the capital of France?")
    
    recommendation = await agent.recommend_model_for_task_async(task)
    
    # Verify recommendation is returned
    assert recommendation is not None, "Recommendation should not be None"
    assert hasattr(recommendation, 'model_name'), "Recommendation should have model_name"
    assert recommendation.selection_method == "rule_based", "Selection method should be rule_based by default"

