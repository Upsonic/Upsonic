from unittest.mock import Mock, patch
from upsonic.chat.cost_calculator import CostTracker
from upsonic.usage import RequestUsage


def test_cost_calculator_initialization():
    """Test CostCalculator initialization."""
    tracker = CostTracker()

    assert tracker._cost_history == []
    assert tracker.input_tokens == 0
    assert tracker.output_tokens == 0
    assert tracker.total_cost == 0.0


def test_cost_calculator_calculate_cost():
    """Test cost calculation."""
    tracker = CostTracker()

    # Mock usage
    usage = RequestUsage(input_tokens=100, output_tokens=50)
    mock_model = Mock()
    mock_model.model_name = "gpt-4"

    # Mock the cost calculation functions
    with patch(
        "upsonic.chat.cost_calculator.get_estimated_cost_from_usage"
    ) as mock_cost:
        mock_cost.return_value = "~$0.0123"

        tracker.add_usage(usage, mock_model)

        assert len(tracker._cost_history) == 1
        assert tracker.input_tokens == 100
        assert tracker.output_tokens == 50


def test_cost_calculator_different_models():
    """Test different models."""
    tracker = CostTracker()

    usage1 = RequestUsage(input_tokens=100, output_tokens=50)
    usage2 = RequestUsage(input_tokens=200, output_tokens=100)

    mock_model1 = Mock()
    mock_model1.model_name = "gpt-4"

    mock_model2 = Mock()
    mock_model2.model_name = "gpt-3.5-turbo"

    with patch(
        "upsonic.chat.cost_calculator.get_estimated_cost_from_usage"
    ) as mock_cost:
        mock_cost.return_value = "~$0.01"

        tracker.add_usage(usage1, mock_model1)
        tracker.add_usage(usage2, mock_model2)

        assert len(tracker._cost_history) == 2
        assert tracker.input_tokens == 300
        assert tracker.output_tokens == 150


def test_cost_calculator_usage_tracking():
    """Test usage tracking."""
    tracker = CostTracker()

    usage = RequestUsage(input_tokens=100, output_tokens=50)
    mock_model = Mock()

    with patch(
        "upsonic.chat.cost_calculator.get_estimated_cost_from_usage"
    ) as mock_cost:
        mock_cost.return_value = "~$0.0123"

        tracker.add_usage(usage, mock_model)

        # Test cost history
        history = tracker.get_cost_history()
        assert len(history) == 1
        assert history[0]["input_tokens"] == 100
        assert history[0]["output_tokens"] == 50

        # Test session summary
        summary = tracker.get_session_summary()
        assert summary["total_input_tokens"] == 100
        assert summary["total_output_tokens"] == 50
        assert "total_cost" in summary
