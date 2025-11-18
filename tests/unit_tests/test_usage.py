from upsonic.usage import RequestUsage, RunUsage, UsageLimits, Usage


def test_usage_tracking():
    """Test usage tracking."""
    # Test RequestUsage
    request_usage = RequestUsage(
        input_tokens=100, output_tokens=50, cache_write_tokens=10, cache_read_tokens=5
    )

    assert request_usage.input_tokens == 100
    assert request_usage.output_tokens == 50
    assert request_usage.total_tokens == 150
    assert request_usage.requests == 1

    # Test increment
    other_usage = RequestUsage(input_tokens=50, output_tokens=25)
    request_usage.incr(other_usage)

    assert request_usage.input_tokens == 150
    assert request_usage.output_tokens == 75

    # Test addition
    usage1 = RequestUsage(input_tokens=10, output_tokens=5)
    usage2 = RequestUsage(input_tokens=20, output_tokens=10)
    combined = usage1 + usage2

    assert combined.input_tokens == 30
    assert combined.output_tokens == 15

    # Test RunUsage
    run_usage = RunUsage(requests=5, tool_calls=3, input_tokens=500, output_tokens=250)

    assert run_usage.requests == 5
    assert run_usage.tool_calls == 3
    assert run_usage.total_tokens == 750

    # Test increment with RequestUsage
    run_usage.incr(RequestUsage(input_tokens=100, output_tokens=50))
    assert run_usage.input_tokens == 600
    assert run_usage.output_tokens == 300


def test_usage_metrics():
    """Test usage metrics."""
    # Test UsageLimits
    limits = UsageLimits(
        request_limit=100,
        tool_calls_limit=50,
        input_tokens_limit=10000,
        output_tokens_limit=5000,
        total_tokens_limit=15000,
        count_tokens_before_request=False,
    )

    assert limits.request_limit == 100
    assert limits.tool_calls_limit == 50
    assert limits.input_tokens_limit == 10000
    assert limits.output_tokens_limit == 5000
    assert limits.total_tokens_limit == 15000

    # Test has_token_limits
    assert limits.has_token_limits() is True

    limits_no_tokens = UsageLimits(request_limit=100)
    assert limits_no_tokens.has_token_limits() is False

    # Test check_before_request
    usage = RunUsage(requests=50, input_tokens=5000, output_tokens=2500)
    limits.check_before_request(usage)  # Should not raise

    # Test check_tokens
    usage_within_limits = RunUsage(input_tokens=5000, output_tokens=2500)
    limits.check_tokens(usage_within_limits)  # Should not raise

    # Test check_before_tool_call
    usage_with_tools = RunUsage(tool_calls=25)
    limits.check_before_tool_call(usage_with_tools)  # Should not raise

    # Test Usage (deprecated alias)
    usage = Usage(requests=10, input_tokens=1000, output_tokens=500)
    assert usage.requests == 10
    assert usage.total_tokens == 1500
