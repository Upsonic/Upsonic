"""
Test 26: Eval features testing for Team class
Success criteria: We check the attributes, what we log and results
"""
import pytest
from io import StringIO
from contextlib import redirect_stdout

from upsonic import Agent, Task, Team
from upsonic.eval import AccuracyEvaluator, PerformanceEvaluator

pytestmark = pytest.mark.timeout(300)


@pytest.mark.asyncio
async def test_team_accuracy_evaluation(capsys):
    """Test Team with AccuracyEvaluator - verify all attributes and logging."""
    
    # Create team agents
    researcher = Agent(
        model="openai/gpt-4o-mini",
        name="Researcher",
        role="Research Specialist",
        goal="Find accurate information"
    )
    writer = Agent(
        model="openai/gpt-4o-mini",
        name="Writer",
        role="Content Writer",
        goal="Create concise summaries"
    )
    
    # Create team
    team = Team(
        agents=[researcher, writer],
        mode="sequential"
    )
    
    # Create judge agent
    judge_agent = Agent(
        model="openai/gpt-4o-mini",
        name="Judge"
    )
    
    # Create evaluator
    evaluator = AccuracyEvaluator(
        judge_agent=judge_agent,
        agent_under_test=team,
        query="What is the capital of France?",
        expected_output="Paris is the capital of France.",
        additional_guidelines="Check if the answer correctly identifies Paris as the capital.",
        num_iterations=1
    )
    
    # Verify evaluator attributes
    assert evaluator.judge_agent == judge_agent, "Judge agent should be set correctly"
    assert evaluator.agent_under_test == team, "Agent under test should be the team"
    assert evaluator.query == "What is the capital of France?", "Query should be set"
    assert evaluator.expected_output == "Paris is the capital of France.", "Expected output should be set"
    assert evaluator.num_iterations == 1, "Iterations should be 1"
    assert len(evaluator._results) == 0, "Results should be empty initially"
    
    # Run evaluation
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await evaluator.run(print_results=True)
    
    output = output_buffer.getvalue()
    
    # Verify result attributes
    assert result is not None, "Evaluation result should not be None"
    assert hasattr(result, 'average_score'), "Result should have average_score"
    assert hasattr(result, 'evaluation_scores'), "Result should have evaluation_scores list"
    assert hasattr(result, 'generated_output'), "Result should have generated_output"
    
    # Verify scores
    assert isinstance(result.evaluation_scores, list), "Scores should be a list"
    assert len(result.evaluation_scores) == 1, "Should have 1 score for 1 iteration"
    
    score = result.evaluation_scores[0]
    assert hasattr(score, 'score'), "Score should have score attribute"
    assert hasattr(score, 'reasoning'), "Score should have reasoning attribute"
    assert hasattr(score, 'is_met'), "Score should have is_met attribute"
    assert hasattr(score, 'critique'), "Score should have critique attribute"
    
    assert isinstance(score.score, (int, float)), "Score value should be numeric"
    assert 1 <= score.score <= 10, "Score should be between 1 and 10"
    assert isinstance(score.reasoning, str), "Reasoning should be a string"
    assert len(score.reasoning) > 0, "Reasoning should not be empty"
    assert isinstance(score.is_met, bool), "is_met should be boolean"
    assert isinstance(score.critique, str), "critique should be a string"
    
    # Verify average score
    assert isinstance(result.average_score, (int, float)), "Average score should be numeric"
    assert 1 <= result.average_score <= 10, "Average score should be between 1 and 10"
    
    # Verify generated output
    assert isinstance(result.generated_output, str), "Generated output should be a string"
    assert len(result.generated_output) > 0, "Generated output should not be empty"
    assert "paris" in result.generated_output.lower(), "Output should mention Paris"
    
    # Verify logging output
    assert len(output) > 0, "Should have logging output"
    assert "Evaluation Results" in output or "Score" in output or "score" in output.lower(), \
        "Should log evaluation results"


@pytest.mark.asyncio
async def test_team_performance_evaluation(capsys):
    """Test Team with PerformanceEvaluator - verify all attributes and logging."""
    
    # Create team agents
    analyst = Agent(
        model="openai/gpt-4o-mini",
        name="Analyst",
        role="Data Analyst",
        goal="Analyze data"
    )
    
    # Create simple team
    team = Team(
        agents=[analyst],
        mode="sequential"
    )
    
    # Create task
    task = Task(description="Calculate 5 + 5")
    
    # Create evaluator
    evaluator = PerformanceEvaluator(
        agent_under_test=team,
        task=task,
        num_iterations=2,
        warmup_runs=1
    )
    
    # Verify evaluator attributes
    assert evaluator.agent_under_test == team, "Agent under test should be the team"
    assert evaluator.task == task, "Task should be set correctly"
    assert evaluator.num_iterations == 2, "Iterations should be 2"
    assert evaluator.warmup_runs == 1, "Warmup runs should be 1"
    
    # Run evaluation
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await evaluator.run(print_results=True)
    
    output = output_buffer.getvalue()
    
    # Verify result attributes
    assert result is not None, "Performance result should not be None"
    assert hasattr(result, 'all_runs'), "Result should have all_runs"
    assert hasattr(result, 'num_iterations'), "Result should have num_iterations"
    assert hasattr(result, 'warmup_runs'), "Result should have warmup_runs"
    assert hasattr(result, 'latency_stats'), "Result should have latency_stats"
    assert hasattr(result, 'memory_increase_stats'), "Result should have memory_increase_stats"
    assert hasattr(result, 'memory_peak_stats'), "Result should have memory_peak_stats"
    
    # Verify runs
    assert isinstance(result.all_runs, list), "All runs should be a list"
    assert len(result.all_runs) == 2, "Should have 2 runs for 2 iterations"
    assert result.num_iterations == 2, "num_iterations should be 2"
    assert result.warmup_runs == 1, "warmup_runs should be 1"
    
    # Verify each run
    for run in result.all_runs:
        assert hasattr(run, 'latency_seconds'), "Run should have latency_seconds"
        assert hasattr(run, 'memory_increase_bytes'), "Run should have memory_increase_bytes"
        assert hasattr(run, 'memory_peak_bytes'), "Run should have memory_peak_bytes"
        
        assert isinstance(run.latency_seconds, (int, float)), "Latency should be numeric"
        assert run.latency_seconds > 0, "Latency should be positive"
        assert isinstance(run.memory_increase_bytes, int), "Memory increase should be int"
        assert isinstance(run.memory_peak_bytes, int), "Memory peak should be int"
    
    # Verify statistics
    assert isinstance(result.latency_stats, dict), "Latency stats should be a dict"
    assert 'average' in result.latency_stats, "Should have average latency"
    assert 'median' in result.latency_stats, "Should have median latency"
    assert 'min' in result.latency_stats, "Should have min latency"
    assert 'max' in result.latency_stats, "Should have max latency"
    assert 'std_dev' in result.latency_stats, "Should have std_dev latency"
    
    assert result.latency_stats['average'] > 0, "Avg latency should be positive"
    assert result.latency_stats['min'] <= result.latency_stats['average'] <= result.latency_stats['max'], \
        "Latency stats should be in order"
    
    assert isinstance(result.memory_increase_stats, dict), "Memory increase stats should be a dict"
    assert isinstance(result.memory_peak_stats, dict), "Memory peak stats should be a dict"
    
    # Verify logging output
    assert len(output) > 0, "Should have logging output"
    assert "Performance" in output or "Latency" in output or "latency" in output.lower(), \
        "Should log performance metrics"


def test_team_evaluator_validation():
    """Test Team evaluator parameter validation."""
    
    team = Team(
        agents=[Agent(model="openai/gpt-4o-mini", name="Agent1")],
        mode="sequential"
    )
    
    judge = Agent(model="openai/gpt-4o-mini", name="Judge")
    
    # Test AccuracyEvaluator validation
    with pytest.raises(TypeError, match="judge_agent.*Agent"):
        AccuracyEvaluator(
            judge_agent="not_an_agent",
            agent_under_test=team,
            query="test",
            expected_output="test"
        )
    
    with pytest.raises(TypeError, match="agent_under_test"):
        AccuracyEvaluator(
            judge_agent=judge,
            agent_under_test="not_a_team",
            query="test",
            expected_output="test"
        )
    
    with pytest.raises(ValueError, match="num_iterations"):
        AccuracyEvaluator(
            judge_agent=judge,
            agent_under_test=team,
            query="test",
            expected_output="test",
            num_iterations=0
        )
    
    # Test PerformanceEvaluator validation
    task = Task(description="test")
    
    with pytest.raises(TypeError, match="agent_under_test"):
        PerformanceEvaluator(
            agent_under_test="not_a_team",
            task=task
        )
    
    with pytest.raises(TypeError, match="task.*Task"):
        PerformanceEvaluator(
            agent_under_test=team,
            task="not_a_task"
        )
    
    with pytest.raises(ValueError, match="num_iterations"):
        PerformanceEvaluator(
            agent_under_test=team,
            task=task,
            num_iterations=0
        )
    
    with pytest.raises(ValueError, match="warmup_runs"):
        PerformanceEvaluator(
            agent_under_test=team,
            task=task,
            warmup_runs=-1
        )

