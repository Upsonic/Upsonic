#!/usr/bin/env python3
"""
Overhead Analysis: Direct vs Agent Benchmark

This benchmark compares the performance overhead of Upsonic's Direct LLM Call
(minimal overhead) versus the full-featured Agent approach.

Usage:
    python -m benchmarks.overhead_analysis.benchmark
    python -m benchmarks.overhead_analysis.benchmark --test-case "Simple Text Query"
    python -m benchmarks.overhead_analysis.benchmark --model "gpt-5-mini-2025-08-07"
    python -m benchmarks.overhead_analysis.benchmark --model "gpt-5-mini-2025-08-07,anthropic/claude-3-5-haiku-20241022"
    python -m benchmarks.overhead_analysis.benchmark --iterations 20 --all-tests
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import os
# Disable telemetry
os.environ["UPSONIC_TELEMETRY"] = "False"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from upsonic import Agent, Direct, Task
from benchmarks.overhead_analysis.test_cases import TestCases
from benchmarks.utils import (
    BenchmarkReporter,
    BenchmarkResult,
    CostMetrics,
    MemoryMetrics,
    MemoryProfiler,
    PerformanceMetrics,
    PerformanceProfiler,
)


def _create_task_from_test_case(test_case: Dict[str, Any]) -> Task:
    """Create a Task object from a test case dictionary."""
    return Task(
        description=test_case["description"],
        response_format=test_case.get("response_format", str),
        attachments=test_case.get("attachments"),
        context=test_case.get("context"),
    )


def _extract_task_metadata(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Extract task metadata for reporting."""
    response_format = test_case.get("response_format", str)
    format_name = (
        response_format.__name__
        if hasattr(response_format, "__name__")
        else str(response_format)
    )

    return {
        "description": test_case["description"],
        "response_format": format_name,
        "attachments": test_case.get("attachments"),
        "context": test_case.get("context"),
    }


def validate_model(model_str: str) -> Tuple[bool, str]:
    """
    Validate a model by making a minimal test call.

    Args:
        model_str: Model identifier (e.g., "gpt-5-mini-2025-08-07")

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        direct = Direct(model=model_str)
        task = Task(description="test", response_format=str)

        # Make minimal test call
        try:
            direct.do(task)
            return True, ""
        except Exception as e:
            error_msg = str(e)
            # Parse common error types
            if "404" in error_msg or "not_found" in error_msg.lower():
                return False, f"Model not found: {model_str.split('/')[-1]}"
            elif "401" in error_msg or "authentication" in error_msg.lower():
                return False, "Authentication failed. Check your API key."
            elif "400" in error_msg:
                return False, f"Bad request: {error_msg[:100]}"
            else:
                return False, error_msg[:150]

    except Exception as e:
        error_msg = str(e)
        if "Unknown provider" in error_msg:
            return False, f"Unknown provider: {model_str.split('/')[0]}"
        return False, error_msg[:150]


def benchmark_direct(
    test_case: Dict[str, Any], model: str = "gpt-5-mini-2025-08-07", iterations: int = 5
) -> BenchmarkResult:
    """
    Benchmark Direct LLM Call approach.

    Args:
        test_case: Test case dictionary from TestCases
        model: Model identifier
        iterations: Number of iterations for performance measurement

    Returns:
        BenchmarkResult with metrics
    """
    print(f"\n🔍 Benchmarking Direct LLM Call...")

    memory_profiler = MemoryProfiler()

    # Measure initialization time and memory
    memory_profiler.start_tracking()
    init_start = time.perf_counter()

    direct = Direct(model=model)

    init_end = time.perf_counter()
    init_current_mb, init_peak_mb = memory_profiler.stop_tracking()
    init_time_ms = (init_end - init_start) * 1000

    # Measure object size
    shallow_size, deep_size = memory_profiler.measure_object_size(direct)

    print(f"  ✓ Initialization: {init_time_ms:.2f} ms")
    print(f"  ✓ Object size: {deep_size:,} bytes")

    # Track cost metrics
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    sample_output = None

    # Execution function
    def execute_task() -> Any:
        nonlocal total_cost, total_input_tokens, total_output_tokens

        task = _create_task_from_test_case(test_case)
        result = direct.do(task, show_output=False)

        # Collect cost metrics from task
        if task.total_cost:
            total_cost += task.total_cost
        if task.total_input_token:
            total_input_tokens += task.total_input_token
        if task.total_output_token:
            total_output_tokens += task.total_output_token

        return result

    # Warmup run and capture sample output
    print(f"  ⏳ Running warmup...")
    try:
        sample_output = str(execute_task())
    except Exception as e:
        print(f"\n  ❌ Warmup failed: {str(e)[:200]}")
        print(f"  ⚠️  Skipping this benchmark due to model error\n")
        raise

    # Measure multiple runs for statistics
    print(f"  ⏳ Running {iterations} iterations...")
    memory_profiler.start_tracking()

    perf_metrics = PerformanceProfiler.measure_multiple_runs(
        execute_task, iterations=iterations, warmup=0  # Already did warmup
    )

    exec_current_mb, exec_peak_mb = memory_profiler.stop_tracking()

    # Update performance metrics with init time
    perf_metrics.init_time_ms = init_time_ms
    perf_metrics.total_time_ms = init_time_ms + perf_metrics.execution_time_ms

    # Create memory metrics
    memory_metrics = MemoryMetrics(
        shallow_size_bytes=shallow_size,
        deep_size_bytes=deep_size,
        peak_memory_mb=max(init_peak_mb, exec_peak_mb),
        current_memory_mb=exec_current_mb,
    )

    print(f"  ✓ Average execution: {perf_metrics.mean_time_ms:.2f} ms")
    print(f"  ✓ Peak memory: {memory_metrics.peak_memory_mb:.2f} MB")

    # Create cost metrics
    total_tokens = total_input_tokens + total_output_tokens
    cost_per_1k = (total_cost / (total_tokens / 1000)) if total_tokens > 0 else 0.0

    cost_metrics = CostMetrics(
        total_cost=total_cost,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        cost_per_1k_tokens=cost_per_1k,
    )

    print(f"  ✓ Total cost: ${total_cost:.6f}")
    print(f"  ✓ Total tokens: {total_tokens:,}")

    return BenchmarkResult(
        name="Direct",
        memory=memory_metrics,
        performance=perf_metrics,
        cost=cost_metrics,
        metadata={
            "model": model,
            "test_case": test_case["name"],
            "task_details": _extract_task_metadata(test_case),
        },
        sample_output=sample_output,
    )


def benchmark_agent(
    test_case: Dict[str, Any], 
    model: str = "gpt-5-mini-2025-08-07", 
    iterations: int = 5,
    with_system_prompt: bool = False
) -> BenchmarkResult:
    """
    Benchmark Agent approach.

    Args:
        test_case: Test case dictionary from TestCases
        model: Model identifier
        iterations: Number of iterations for performance measurement
        with_system_prompt: Whether to use default system prompt (default: False)

    Returns:
        BenchmarkResult with metrics
    """
    prompt_type = "with system prompt" if with_system_prompt else "without system prompt"
    print(f"\n🤖 Benchmarking Agent ({prompt_type})...")

    memory_profiler = MemoryProfiler()

    # Measure initialization time and memory
    memory_profiler.start_tracking()
    init_start = time.perf_counter()

    if with_system_prompt:
        # Use default system prompt
        agent = Agent(model=model, name="BenchmarkAgent")
    else:
        # Empty system prompt for fair comparison with Direct
        agent = Agent(model=model, name="BenchmarkAgent", system_prompt="")

    init_end = time.perf_counter()
    init_current_mb, init_peak_mb = memory_profiler.stop_tracking()
    init_time_ms = (init_end - init_start) * 1000

    # Measure object size
    shallow_size, deep_size = memory_profiler.measure_object_size(agent)

    print(f"  ✓ Initialization: {init_time_ms:.2f} ms")
    print(f"  ✓ Object size: {deep_size:,} bytes")

    # Track cost metrics
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    sample_output = None

    # Execution function
    def execute_task() -> Any:
        nonlocal total_cost, total_input_tokens, total_output_tokens

        task = _create_task_from_test_case(test_case)
        result = agent.do(task, debug=False)

        # Collect cost metrics from task
        if task.total_cost:
            total_cost += task.total_cost
        if task.total_input_token:
            total_input_tokens += task.total_input_token
        if task.total_output_token:
            total_output_tokens += task.total_output_token

        return result

    # Warmup run and capture sample output
    print(f"  ⏳ Running warmup...")
    try:
        sample_output = str(execute_task())
    except Exception as e:
        print(f"\n  ❌ Warmup failed: {str(e)[:200]}")
        print(f"  ⚠️  Skipping this benchmark due to model error\n")
        raise

    # Measure multiple runs for statistics
    print(f"  ⏳ Running {iterations} iterations...")
    memory_profiler.start_tracking()

    perf_metrics = PerformanceProfiler.measure_multiple_runs(
        execute_task, iterations=iterations, warmup=0  # Already did warmup
    )

    exec_current_mb, exec_peak_mb = memory_profiler.stop_tracking()

    # Update performance metrics with init time
    perf_metrics.init_time_ms = init_time_ms
    perf_metrics.total_time_ms = init_time_ms + perf_metrics.execution_time_ms

    # Create memory metrics
    memory_metrics = MemoryMetrics(
        shallow_size_bytes=shallow_size,
        deep_size_bytes=deep_size,
        peak_memory_mb=max(init_peak_mb, exec_peak_mb),
        current_memory_mb=exec_current_mb,
    )

    print(f"  ✓ Average execution: {perf_metrics.mean_time_ms:.2f} ms")
    print(f"  ✓ Peak memory: {memory_metrics.peak_memory_mb:.2f} MB")

    # Create cost metrics
    total_tokens = total_input_tokens + total_output_tokens
    cost_per_1k = (total_cost / (total_tokens / 1000)) if total_tokens > 0 else 0.0

    cost_metrics = CostMetrics(
        total_cost=total_cost,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        cost_per_1k_tokens=cost_per_1k,
    )

    print(f"  ✓ Total cost: ${total_cost:.6f}")
    print(f"  ✓ Total tokens: {total_tokens:,}")

    name = "Agent (with prompt)" if with_system_prompt else "Agent (no prompt)"
    
    return BenchmarkResult(
        name=name,
        memory=memory_metrics,
        performance=perf_metrics,
        cost=cost_metrics,
        metadata={
            "model": model,
            "test_case": test_case["name"],
            "with_system_prompt": with_system_prompt,
            "task_details": _extract_task_metadata(test_case),
        },
        sample_output=sample_output,
    )


def run_benchmark(
    test_case_name: Optional[str] = None,
    model: str = "gpt-5-mini-2025-08-07",
    iterations: int = 5,
    all_tests: bool = False,
) -> None:
    """
    Run the benchmark comparison.

    Args:
        test_case_name: Name of test case to run (None for default)
        model: Model identifier or comma-separated list of models
        iterations: Number of iterations
        all_tests: Run all available test cases
    """
    # Load environment variables
    load_dotenv()

    # Parse model string (comma-separated)
    models = [m.strip() for m in model.split(",")]

    # Validate all models before running benchmarks
    print("\n🔍 Validating models (this may take a moment)...")
    valid_models = []
    for model_str in models:
        print(f"  Testing '{model_str}'...", end=" ", flush=True)
        is_valid, error_msg = validate_model(model_str)
        if is_valid:
            print("✓")
            valid_models.append(model_str)
        else:
            print(f"✗ ({error_msg})")

    if not valid_models:
        _print_model_validation_help()
        return

    models = valid_models
    print(f"\n✅ {len(valid_models)} model(s) validated successfully\n")

    # Determine which test cases to run
    if all_tests:
        test_cases = TestCases.get_all_test_cases()
    elif test_case_name:
        test_cases = [TestCases.get_test_case_by_name(test_case_name)]
    else:
        # Default: simple text query
        test_cases = [TestCases.get_simple_text()]

    # Results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Run benchmarks for each test case and each model
    for test_case in test_cases:
        # Store all results for cross-model comparison
        all_model_results = []

        for current_model in models:
            print("\n" + "=" * 80)
            print(f"TEST CASE: {test_case['name']}")
            print("=" * 80)
            print(f"Description: {test_case['description'][:100]}...")
            print(f"Model: {current_model}")
            print(f"Iterations: {iterations}")

            try:
                # Benchmark Direct
                direct_result = benchmark_direct(test_case, current_model, iterations)

                # Benchmark Agent without system prompt
                agent_no_prompt_result = benchmark_agent(
                    test_case, current_model, iterations, with_system_prompt=False
                )

                # Benchmark Agent with default system prompt
                agent_with_prompt_result = benchmark_agent(
                    test_case, current_model, iterations, with_system_prompt=True
                )

                # Store results for later comparison
                all_model_results.append(
                    {
                        "model": current_model,
                        "direct": direct_result,
                        "agent_no_prompt": agent_no_prompt_result,
                        "agent_with_prompt": agent_with_prompt_result,
                    }
                )

                # Create comparison report for this model
                report = BenchmarkReporter.create_comparison_report(
                    results=[direct_result, agent_no_prompt_result, agent_with_prompt_result],
                    test_name=f"{test_case['name']} ({current_model.split('/')[-1]})",
                )

                # Save JSON report (for programmatic use)
                json_path = BenchmarkReporter.save_report(report, results_dir)

                # Save Markdown report (for human reading)
                md_path = BenchmarkReporter.save_markdown_report(report, results_dir)

                # Print summary
                BenchmarkReporter.print_summary(report)

                print(f"📊 JSON Report: {json_path}")
                print(f"📄 Markdown Report: {md_path}")

            except Exception as e:
                print(f"\n❌ Benchmark failed for {current_model}")
                print(f"   Error: {str(e)[:200]}")
                print(f"   This model will be skipped. Continuing with remaining models...\n")

        # If multiple models were tested, create cross-model comparison
        if len(all_model_results) > 1:
            _print_cross_model_comparison(test_case, all_model_results)


def _print_model_validation_help() -> None:
    """Print helpful information when model validation fails."""
    print("\n❌ No valid models provided. Please check your model names and API keys.")
    print("\n💡 Examples of valid model formats:")
    print("   gpt-5-mini-2025-08-07")
    print("   openai/gpt-4o-mini")
    print("   anthropic/claude-3-5-haiku-20241022")
    print("   anthropic/claude-3-5-sonnet-20241022")
    print("\n💡 Make sure you have the required API keys in your .env file:")
    print("   OPENAI_API_KEY for OpenAI models")
    print("   ANTHROPIC_API_KEY for Anthropic models")
    print("\n💡 Common issues:")
    print("   - Model name typo (check the exact model name)")
    print("   - Missing or invalid API key")
    print("   - Insufficient API credits")


def _print_cross_model_comparison(
    test_case: Dict[str, Any], all_model_results: List[Dict[str, Any]]
) -> None:
    """Print cross-model comparison results."""
    print("\n" + "=" * 80)
    print(f"CROSS-MODEL COMPARISON: {test_case['name']}")
    print("=" * 80)

    # Compare Direct implementations across models
    print("\n🔍 DIRECT COMPARISON:")
    for result in all_model_results:
        model_name = result["model"].split("/")[-1]
        direct = result["direct"]
        mean_cost = direct.cost.total_cost / direct.performance.iterations
        print(f"\n  {model_name}:")
        print(f"    Mean Time: {direct.performance.mean_time_ms:.2f} ms (±{direct.performance.stdev_time_ms:.2f})")
        print(f"    Median Time: {direct.performance.median_time_ms:.2f} ms")
        print(f"    Memory: {direct.memory.deep_size_bytes:,} bytes")
        print(f"    Mean Cost: ${mean_cost:.6f} (${direct.cost.total_cost:.6f} total)")

    # Compare Agent (no prompt) implementations across models
    print("\n🤖 AGENT (NO PROMPT) COMPARISON:")
    for result in all_model_results:
        model_name = result["model"].split("/")[-1]
        agent = result["agent_no_prompt"]
        mean_cost = agent.cost.total_cost / agent.performance.iterations
        print(f"\n  {model_name}:")
        print(f"    Mean Time: {agent.performance.mean_time_ms:.2f} ms (±{agent.performance.stdev_time_ms:.2f})")
        print(f"    Median Time: {agent.performance.median_time_ms:.2f} ms")
        print(f"    Memory: {agent.memory.deep_size_bytes:,} bytes")
        print(f"    Mean Cost: ${mean_cost:.6f} (${agent.cost.total_cost:.6f} total)")

    # Compare Agent (with prompt) implementations across models
    print("\n🤖 AGENT (WITH PROMPT) COMPARISON:")
    for result in all_model_results:
        model_name = result["model"].split("/")[-1]
        agent = result["agent_with_prompt"]
        mean_cost = agent.cost.total_cost / agent.performance.iterations
        print(f"\n  {model_name}:")
        print(f"    Mean Time: {agent.performance.mean_time_ms:.2f} ms (±{agent.performance.stdev_time_ms:.2f})")
        print(f"    Median Time: {agent.performance.median_time_ms:.2f} ms")
        print(f"    Memory: {agent.memory.deep_size_bytes:,} bytes")
        print(f"    Mean Cost: ${mean_cost:.6f} (${agent.cost.total_cost:.6f} total)")

    # Find fastest based on mean time (more reliable than single execution)
    fastest_direct = min(
        all_model_results, key=lambda x: x["direct"].performance.mean_time_ms
    )
    fastest_agent_no_prompt = min(
        all_model_results, key=lambda x: x["agent_no_prompt"].performance.mean_time_ms
    )
    fastest_agent_with_prompt = min(
        all_model_results, key=lambda x: x["agent_with_prompt"].performance.mean_time_ms
    )
    
    # Find most cost-efficient (using mean cost per iteration)
    cheapest_direct = min(
        all_model_results, 
        key=lambda x: x["direct"].cost.total_cost / x["direct"].performance.iterations
    )
    cheapest_agent_no_prompt = min(
        all_model_results, 
        key=lambda x: x["agent_no_prompt"].cost.total_cost / x["agent_no_prompt"].performance.iterations
    )
    cheapest_agent_with_prompt = min(
        all_model_results, 
        key=lambda x: x["agent_with_prompt"].cost.total_cost / x["agent_with_prompt"].performance.iterations
    )

    print("\n🏆 WINNERS:")
    print(
        f"  Fastest Direct: {fastest_direct['model'].split('/')[-1]} "
        f"(mean: {fastest_direct['direct'].performance.mean_time_ms:.2f} ms, "
        f"±{fastest_direct['direct'].performance.stdev_time_ms:.2f} ms)"
    )
    print(
        f"  Fastest Agent (no prompt): {fastest_agent_no_prompt['model'].split('/')[-1]} "
        f"(mean: {fastest_agent_no_prompt['agent_no_prompt'].performance.mean_time_ms:.2f} ms, "
        f"±{fastest_agent_no_prompt['agent_no_prompt'].performance.stdev_time_ms:.2f} ms)"
    )
    print(
        f"  Fastest Agent (with prompt): {fastest_agent_with_prompt['model'].split('/')[-1]} "
        f"(mean: {fastest_agent_with_prompt['agent_with_prompt'].performance.mean_time_ms:.2f} ms, "
        f"±{fastest_agent_with_prompt['agent_with_prompt'].performance.stdev_time_ms:.2f} ms)"
    )
    
    # Calculate mean costs for display
    cheapest_direct_mean = (
        cheapest_direct['direct'].cost.total_cost / 
        cheapest_direct['direct'].performance.iterations
    )
    cheapest_agent_no_prompt_mean = (
        cheapest_agent_no_prompt['agent_no_prompt'].cost.total_cost / 
        cheapest_agent_no_prompt['agent_no_prompt'].performance.iterations
    )
    cheapest_agent_with_prompt_mean = (
        cheapest_agent_with_prompt['agent_with_prompt'].cost.total_cost / 
        cheapest_agent_with_prompt['agent_with_prompt'].performance.iterations
    )
    
    print(
        f"  Cheapest Direct: {cheapest_direct['model'].split('/')[-1]} "
        f"(mean: ${cheapest_direct_mean:.6f}/iter)"
    )
    print(
        f"  Cheapest Agent (no prompt): {cheapest_agent_no_prompt['model'].split('/')[-1]} "
        f"(mean: ${cheapest_agent_no_prompt_mean:.6f}/iter)"
    )
    print(
        f"  Cheapest Agent (with prompt): {cheapest_agent_with_prompt['model'].split('/')[-1]} "
        f"(mean: ${cheapest_agent_with_prompt_mean:.6f}/iter)"
    )
    print("=" * 80)


def main() -> None:
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark Direct vs Agent in Upsonic Framework"
    )
    parser.add_argument(
        "--test-case", type=str, help="Name of test case to run (default: Simple Text Query)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Model identifier or comma-separated list (e.g., 'gpt-5-mini-2025-08-07' or "
        "'gpt-5-mini-2025-08-07,anthropic/claude-3-5-haiku-20241022')",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations for performance measurement (default: 5)",
    )
    parser.add_argument(
        "--all-tests", action="store_true", help="Run all available test cases"
    )
    parser.add_argument(
        "--list-tests", action="store_true", help="List available test cases and exit"
    )

    args = parser.parse_args()

    # List test cases if requested
    if args.list_tests:
        print("\nAvailable Test Cases:")
        print("-" * 40)
        for case in TestCases.get_all_test_cases():
            print(f"  • {case['name']}")
        print()
        return

    # Run benchmark
    run_benchmark(
        test_case_name=args.test_case,
        model=args.model,
        iterations=args.iterations,
        all_tests=args.all_tests,
    )


if __name__ == "__main__":
    main()
