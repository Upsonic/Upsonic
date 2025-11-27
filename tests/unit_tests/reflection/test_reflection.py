import asyncio
from unittest.mock import Mock, AsyncMock, patch
from upsonic.reflection.processor import ReflectionProcessor
from upsonic.reflection.models import (
    ReflectionConfig,
    ReflectionState,
    EvaluationResult,
    EvaluationCriteria,
    ReflectionAction,
)


def test_reflection_processor_initialization():
    """Test ReflectionProcessor init."""
    config = ReflectionConfig(
        max_iterations=3, acceptance_threshold=0.8, enable_self_critique=True
    )

    processor = ReflectionProcessor(config)

    assert processor.config == config
    assert processor.config.max_iterations == 3
    assert processor.config.acceptance_threshold == 0.8


def test_reflection_processor_process():
    """Test reflection processing."""
    config = ReflectionConfig(max_iterations=2, acceptance_threshold=0.7)
    processor = ReflectionProcessor(config)

    mock_agent = Mock()
    mock_agent.model = Mock()
    mock_agent.name = "TestAgent"
    mock_agent.debug = False

    mock_task = Mock()
    mock_task.description = "Test task"
    mock_task.context = []
    mock_task.response_format = str
    mock_task.tools = []
    mock_task.attachments = []

    initial_response = "Test response"

    async def run_test():
        # Mock the evaluator agent
        with (
            patch("upsonic.agent.agent.Agent") as mock_agent_class,
            patch("upsonic.tasks.tasks.Task") as mock_task_class,
        ):
            mock_evaluator = Mock()
            evaluation_result = EvaluationResult(
                criteria=EvaluationCriteria(
                    accuracy=0.8, completeness=0.8, relevance=0.8, clarity=0.8
                ),
                overall_score=0.8,
                feedback="Good response",
                suggested_improvements=[],
                action=ReflectionAction.ACCEPT,
                confidence=0.9,
            )
            mock_eval_task = Mock()
            mock_eval_task.response = evaluation_result
            mock_task_class.return_value = mock_eval_task
            mock_evaluator.do_async = AsyncMock(return_value=evaluation_result)
            mock_agent_class.return_value = mock_evaluator

            # Process with reflection
            result = await processor.process_with_reflection(
                agent=mock_agent, task=mock_task, initial_response=initial_response
            )

            assert result is not None

    asyncio.run(run_test())


def test_reflection_models():
    """Test reflection models."""
    # Test EvaluationCriteria
    criteria = EvaluationCriteria(
        accuracy=0.9, completeness=0.8, relevance=0.85, clarity=0.9
    )

    overall = criteria.overall_score()
    assert 0 <= overall <= 1

    # Test EvaluationResult
    result = EvaluationResult(
        criteria=criteria,
        overall_score=overall,
        feedback="Test feedback",
        suggested_improvements=["Improvement 1", "Improvement 2"],
        action=ReflectionAction.REVISE,
        confidence=0.85,
    )

    assert result.overall_score == overall
    assert result.action == ReflectionAction.REVISE
    assert len(result.suggested_improvements) == 2

    # Test ReflectionState
    state = ReflectionState()
    assert state.iteration == 0
    assert len(state.evaluations) == 0

    state.add_evaluation("Response 1", result)
    assert state.iteration == 1
    assert len(state.evaluations) == 1

    # Test should_continue
    # With overall_score 0.85 and threshold 0.8, should_continue returns False
    # because 0.85 >= 0.8 (meets acceptance threshold)
    config = ReflectionConfig(max_iterations=3, acceptance_threshold=0.8)
    assert state.should_continue(config) is False

    # Test with lower score that should continue
    low_score_result = EvaluationResult(
        criteria=EvaluationCriteria(
            accuracy=0.5, completeness=0.5, relevance=0.5, clarity=0.5
        ),
        overall_score=0.5,
        feedback="Needs improvement",
        suggested_improvements=["Improve accuracy"],
        action=ReflectionAction.REVISE,
        confidence=0.5,
    )
    state2 = ReflectionState()
    state2.add_evaluation("Response 2", low_score_result)
    assert state2.should_continue(config) is True  # 0.5 < 0.8, should continue

    # Test ReflectionConfig
    config = ReflectionConfig(
        max_iterations=5,
        acceptance_threshold=0.9,
        evaluator_model="gpt-4",
        enable_self_critique=True,
        enable_improvement_suggestions=True,
    )

    assert config.max_iterations == 5
    assert config.acceptance_threshold == 0.9
    assert config.enable_self_critique is True
