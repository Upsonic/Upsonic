"""
Smoke test for Structured Task Output.

Tests that Task objects with response_format return structured output (Pydantic models).
"""

import pytest
from pydantic import BaseModel
from upsonic import Agent, Task

pytestmark = pytest.mark.timeout(60)


class AnalysisResult(BaseModel):
    """Structured output model for testing."""
    summary: str
    confidence: float
    recommendations: list[str]
    key_metrics: dict[str, float]


@pytest.mark.asyncio
async def test_structured_task_output():
    """Test that Task with response_format returns structured Pydantic output."""
    agent = Agent(
        model="openai/gpt-4o",
        name="Analysis Agent"
    )
    
    task = Task(
        description="Analyze the benefits of renewable energy and provide structured results with summary, confidence score, recommendations, and key metrics.",
        response_format=AnalysisResult
    )
    
    result = await agent.do_async(task)
    
    # Verify result is structured (Pydantic model instance)
    assert result is not None, "Result should not be None"
    assert isinstance(result, AnalysisResult), f"Result should be AnalysisResult instance, got {type(result)}"
    
    # Verify all required fields are present and have correct types
    assert isinstance(result.summary, str), "summary should be a string"
    assert isinstance(result.confidence, float), "confidence should be a float"
    assert 0.0 <= result.confidence <= 1.0, "confidence should be between 0.0 and 1.0"
    assert isinstance(result.recommendations, list), "recommendations should be a list"
    assert all(isinstance(rec, str) for rec in result.recommendations), "all recommendations should be strings"
    assert isinstance(result.key_metrics, dict), "key_metrics should be a dict"
    assert all(isinstance(k, str) for k in result.key_metrics.keys()), "all key_metrics keys should be strings"
    assert all(isinstance(v, (int, float)) for v in result.key_metrics.values()), "all key_metrics values should be numbers"
    
    # Verify task.response is also set correctly
    assert hasattr(task, 'response'), "Task should have response attribute"
    assert isinstance(task.response, AnalysisResult), "Task.response should be AnalysisResult instance"

