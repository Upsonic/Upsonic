import pytest
import asyncio
from upsonic.agent.agent import Direct
from upsonic.tasks.tasks import Task
from upsonic.eval.reliability import ReliabilityEvaluator
from upsonic.models.providers import OpenAI


@pytest.fixture
def test_agent():
    return Direct(
        name="Test Agent",
        model=OpenAI(model_name="gpt-4o-mini")
    )


@pytest.fixture
def mock_tools():
    def search_web(query: str) -> str:
        return "Mock search result"
    
    def translate_to_french(text: str) -> str:
        return "Mock French translation"
    
    def send_email(to: str, subject: str) -> str:
        return "Mock email sent"
    
    return [search_web, translate_to_french, send_email]


class TestReliabilityEvaluator:
    """Core reliability evaluator tests."""

    @pytest.mark.asyncio
    async def test_presence_check_pass(self, test_agent, mock_tools):
        """Test that evaluator passes when expected tools are present."""
        task = Task(
            description="Search and translate",
            tools=mock_tools[:2]
        )
        # Create mock tool calls since we can't make actual API calls without keys
        task._tool_calls = [
            {"tool_name": "search_web"},
            {"tool_name": "translate_to_french"}
        ]
        
        evaluator = ReliabilityEvaluator(
            expected_tool_calls=["search_web", "translate_to_french"]
        )
        result = evaluator.run(task)
        
        assert result.passed is True
        result.assert_passed()

    @pytest.mark.asyncio
    async def test_presence_check_fail(self, test_agent, mock_tools):
        """Test that evaluator fails when expected tools are missing."""
        task = Task(
            description="Search and translate",
            tools=mock_tools[:2]
        )
        # Mock only 2 tool calls but expect 3
        task._tool_calls = [
            {"tool_name": "search_web"},
            {"tool_name": "translate_to_french"}
        ]
        
        evaluator = ReliabilityEvaluator(
            expected_tool_calls=["search_web", "translate_to_french", "send_email"]
        )
        result = evaluator.run(task)
        
        assert result.passed is False
        with pytest.raises(AssertionError):
            result.assert_passed()

    @pytest.mark.asyncio
    async def test_order_matters_pass(self, test_agent, mock_tools):
        """Test that evaluator passes when tools are in correct order."""
        task = Task(
            description="Search and translate",
            tools=mock_tools[:2]
        )
        # Mock tool calls in correct order
        task._tool_calls = [
            {"tool_name": "search_web"},
            {"tool_name": "translate_to_french"}
        ]
        
        evaluator = ReliabilityEvaluator(
            expected_tool_calls=["search_web", "translate_to_french"],
            order_matters=True
        )
        result = evaluator.run(task)
        
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_exact_match_pass(self, test_agent, mock_tools):
        """Test that evaluator passes with exact match when appropriate."""
        task = Task(
            description="Search and translate",
            tools=mock_tools[:2]
        )
        # Mock exactly the expected tool calls
        task._tool_calls = [
            {"tool_name": "search_web"},
            {"tool_name": "translate_to_french"}
        ]
        
        evaluator = ReliabilityEvaluator(
            expected_tool_calls=["search_web", "translate_to_french"],
            exact_match=True
        )
        result = evaluator.run(task)
        
        assert result.passed is True

    def test_exact_match_fail(self, test_agent, mock_tools):
        """Test that evaluator fails with exact match when extra tools used."""
        task = Task(
            description="Use extra tools",
            tools=mock_tools  # All 3 tools
        )
        task._tool_calls = [
            {"tool_name": "search_web"}, 
            {"tool_name": "translate_to_french"}, 
            {"tool_name": "send_email"}
        ]
        
        evaluator = ReliabilityEvaluator(
            expected_tool_calls=["search_web", "translate_to_french"],
            exact_match=True
        )
        result = evaluator.run(task)
        
        assert result.passed is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])