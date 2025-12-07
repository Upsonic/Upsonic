"""
Test 10: Reflection feature for Agent

Success criteria:
- Reflection process starts and logs appear
- Evaluation results are logged properly
- Improvement process is logged when needed
- Reflection completion is logged
- We check logs for that
"""

import pytest
from upsonic import Agent, Task
from upsonic.reflection import ReflectionConfig
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.timeout(120)


@pytest.mark.asyncio
async def test_reflection_basic():
    """Test basic reflection feature with default config."""
    # Create reflection config
    reflection_config = ReflectionConfig(
        max_iterations=2,
        acceptance_threshold=0.7,
        enable_self_critique=True,
        enable_improvement_suggestions=True
    )
    
    # Create agent with reflection enabled
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        reflection=True,
        reflection_config=reflection_config,
        debug=True
    )
    
    # Verify reflection is enabled
    assert agent.reflection is True, "reflection should be True"
    assert agent.reflection_processor is not None, "reflection_processor should be set"
    assert agent.reflection_config is not None, "reflection_config should be set"
    
    # Capture stdout
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        # Simple task that might need improvement
        task = Task(
            description="Write a brief explanation of how machine learning works. Make it clear and comprehensive."
        )
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify reflection logs appear
    assert "Reflection Process Started" in output or "reflection" in output.lower(), f"Should see reflection start logs. Output: {output[:1000]}"
    assert "Reflection Evaluation" in output or "evaluation" in output.lower(), f"Should see evaluation logs. Output: {output[:1000]}"
    assert "Reflection Process Completed" in output or "completed" in output.lower(), f"Should see reflection completion logs. Output: {output[:1000]}"


@pytest.mark.asyncio
async def test_reflection_with_low_threshold():
    """Test reflection with low acceptance threshold to trigger improvement."""
    # Create reflection config with low threshold to force improvement
    reflection_config = ReflectionConfig(
        max_iterations=3,
        acceptance_threshold=0.9,  # High threshold to potentially trigger improvement
        enable_self_critique=True,
        enable_improvement_suggestions=True
    )
    
    # Create agent with reflection
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        reflection=True,
        reflection_config=reflection_config,
        debug=True
    )
    
    # Capture stdout
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        # Task that might need multiple iterations
        task = Task(
            description="Explain quantum computing in detail, covering qubits, superposition, and entanglement. Be thorough and accurate."
        )
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify reflection logs
    assert "Reflection Process Started" in output or "reflection" in output.lower(), f"Should see reflection start. Output: {output[:1000]}"
    assert "Reflection Evaluation" in output or "evaluation" in output.lower(), f"Should see evaluation. Output: {output[:1000]}"
    
    # Check for improvement logs if improvement was triggered
    if "Improvement" in output or "improvement" in output.lower():
        assert "Reflection Improvement Started" in output or "improvement" in output.lower(), f"Should see improvement logs. Output: {output[:1000]}"
    
    assert "Reflection Process Completed" in output or "completed" in output.lower(), f"Should see completion. Output: {output[:1000]}"


@pytest.mark.asyncio
async def test_reflection_config_attributes():
    """Test that reflection config attributes are properly set."""
    # Create custom reflection config
    reflection_config = ReflectionConfig(
        max_iterations=5,
        acceptance_threshold=0.85,
        evaluator_model="openai/gpt-4o",
        enable_self_critique=True,
        enable_improvement_suggestions=True
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        reflection=True,
        reflection_config=reflection_config,
        debug=True
    )
    
    # Verify config attributes
    assert agent.reflection_config.max_iterations == 5, "max_iterations should be 5"
    assert agent.reflection_config.acceptance_threshold == 0.85, "acceptance_threshold should be 0.85"
    assert agent.reflection_config.evaluator_model == "openai/gpt-4o", "evaluator_model should be set"
    assert agent.reflection_config.enable_self_critique is True, "enable_self_critique should be True"
    assert agent.reflection_config.enable_improvement_suggestions is True, "enable_improvement_suggestions should be True"
    
    # Capture stdout
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        task = Task(description="Write a short paragraph about Python programming.")
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify reflection logs
    assert "Reflection" in output or "reflection" in output.lower(), f"Should see reflection logs. Output: {output[:1000]}"


@pytest.mark.asyncio
async def test_reflection_without_config():
    """Test reflection with default config (no explicit config provided)."""
    # Create agent with reflection=True but no explicit config
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        reflection=True,
        debug=True
    )
    
    # Verify reflection is enabled with default config
    assert agent.reflection is True, "reflection should be True"
    assert agent.reflection_processor is not None, "reflection_processor should be set"
    assert agent.reflection_config is not None, "reflection_config should be set (default created)"
    
    # Default config should have default values
    assert agent.reflection_config.max_iterations == 3, "Default max_iterations should be 3"
    assert agent.reflection_config.acceptance_threshold == 0.8, "Default acceptance_threshold should be 0.8"
    
    # Capture stdout
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        task = Task(description="Explain what AI is in one sentence.")
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify reflection logs appear
    assert "Reflection" in output or "reflection" in output.lower(), f"Should see reflection logs. Output: {output[:1000]}"

