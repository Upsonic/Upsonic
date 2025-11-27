from unittest.mock import Mock
from upsonic.context.agent import turn_agent_to_string


def test_context_agent_initialization():
    """Test context agent."""
    mock_agent = Mock()
    mock_agent.agent_id = "agent-123"
    mock_agent.name = "TestAgent"
    mock_agent.company_url = "https://example.com"
    mock_agent.company_objective = "Test objective"
    mock_agent.company_description = "Test description"
    mock_agent.company_name = "Test Company"
    mock_agent.system_prompt = "Test system prompt"

    # Test conversion to string
    result = turn_agent_to_string(mock_agent)

    assert isinstance(result, str)
    assert "agent-123" in result
    assert "TestAgent" in result
    assert "Test Company" in result


def test_context_agent_build_context():
    """Test context building."""
    mock_agent = Mock()
    mock_agent.agent_id = "agent-456"
    mock_agent.name = "ContextAgent"
    mock_agent.company_url = "https://test.com"
    mock_agent.company_objective = "Build context"
    mock_agent.company_description = "Agent for context building"
    mock_agent.company_name = "Context Company"
    mock_agent.system_prompt = "System prompt for context"

    # Convert to string
    context_string = turn_agent_to_string(mock_agent)

    # Verify all fields are present
    assert "agent-456" in context_string
    assert "ContextAgent" in context_string
    assert "Build context" in context_string
    assert "Context Company" in context_string
    assert "System prompt for context" in context_string
