import asyncio
from unittest.mock import Mock, AsyncMock, patch
from upsonic.reliability_layer.reliability_layer import (
    ReliabilityProcessor,
    ValidationResult,
    ValidationPoint,
    SourceReliability,
)


def test_reliability_layer_initialization():
    """Test ReliabilityLayer init."""
    processor = ReliabilityProcessor(confidence_threshold=0.7)

    assert processor.confidence_threshold == 0.7
    assert processor is not None


def test_reliability_layer_verify():
    """Test verification."""
    # Test validation result creation
    validation_result = ValidationResult(
        url_validation=ValidationPoint(
            is_suspicious=False,
            feedback="No URLs found",
            suspicious_points=[],
            source_reliability=SourceReliability.HIGH,
            verification_method="regex",
            confidence_score=1.0,
        ),
        number_validation=ValidationPoint(
            is_suspicious=False,
            feedback="Numbers verified",
            suspicious_points=[],
            source_reliability=SourceReliability.HIGH,
            verification_method="context_check",
            confidence_score=0.9,
        ),
        information_validation=ValidationPoint(
            is_suspicious=False,
            feedback="Information verified",
            suspicious_points=[],
            source_reliability=SourceReliability.MEDIUM,
            verification_method="context_check",
            confidence_score=0.8,
        ),
        code_validation=ValidationPoint(
            is_suspicious=False,
            feedback="No code found",
            suspicious_points=[],
            source_reliability=SourceReliability.UNKNOWN,
            verification_method="regex",
            confidence_score=1.0,
        ),
        any_suspicion=False,
        suspicious_points=[],
        overall_feedback="",
    )

    # Calculate suspicion
    summary = validation_result.calculate_suspicion()

    assert validation_result.any_suspicion is False
    assert validation_result.overall_confidence > 0
    assert isinstance(summary, str)


def test_reliability_layer_edit():
    """Test editing."""
    mock_task = Mock()
    mock_task.response = "Test response"
    mock_task.description = "Test task"
    mock_task.context = []
    mock_task.response_format = str
    mock_task.tools = []
    mock_task.attachments = []
    mock_task.images = []
    mock_task.price_id = None

    mock_reliability_layer = Mock()
    mock_reliability_layer.prevent_hallucination = 10

    async def run_test():
        # Test process_task with prevent_hallucination = 10
        # Since the code imports Task and Agent inside the method, we need to patch them
        with (
            patch("upsonic.agent.agent.Agent") as mock_agent_class,
            patch(
                "upsonic.reliability_layer.reliability_layer.Task", create=True
            ) as mock_task_class,
        ):
            mock_agent = Mock()
            mock_agent.do_async = AsyncMock(return_value=None)
            mock_agent_class.return_value = mock_agent

            # Mock Task class to return a task with proper response
            mock_validation_task = Mock()
            mock_validation_task.response = ValidationPoint(
                is_suspicious=False,
                feedback="Validated",
                suspicious_points=[],
                source_reliability=SourceReliability.HIGH,
                verification_method="test",
                confidence_score=0.9,
            )
            mock_task_class.return_value = mock_validation_task

            # The validation will skip if no URLs/numbers/code are found, so this should work
            result = await ReliabilityProcessor.process_task(
                task=mock_task, reliability_layer=mock_reliability_layer, model=None
            )

            assert result is not None

    asyncio.run(run_test())


def test_reliability_layer_iterative_improvement():
    """Test iterative improvement."""
    # Test with suspicious content
    validation_result = ValidationResult(
        url_validation=ValidationPoint(
            is_suspicious=True,
            feedback="Suspicious URL found",
            suspicious_points=["url1"],
            source_reliability=SourceReliability.LOW,
            verification_method="validation",
            confidence_score=0.3,
        ),
        number_validation=ValidationPoint(
            is_suspicious=False,
            feedback="Numbers verified",
            suspicious_points=[],
            source_reliability=SourceReliability.HIGH,
            verification_method="context_check",
            confidence_score=0.9,
        ),
        information_validation=ValidationPoint(
            is_suspicious=False,
            feedback="Information verified",
            suspicious_points=[],
            source_reliability=SourceReliability.MEDIUM,
            verification_method="context_check",
            confidence_score=0.8,
        ),
        code_validation=ValidationPoint(
            is_suspicious=False,
            feedback="No code found",
            suspicious_points=[],
            source_reliability=SourceReliability.UNKNOWN,
            verification_method="regex",
            confidence_score=1.0,
        ),
        any_suspicion=False,
        suspicious_points=[],
        overall_feedback="",
    )

    # Calculate suspicion
    summary = validation_result.calculate_suspicion()

    assert validation_result.url_validation.is_suspicious is True
    assert len(validation_result.url_validation.suspicious_points) == 1
    assert isinstance(summary, str)
