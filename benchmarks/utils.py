"""
Benchmark utilities for memory and performance profiling.
"""

import sys
import time
import tracemalloc
import statistics
import platform
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    shallow_size_bytes: int
    deep_size_bytes: int
    peak_memory_mb: float
    current_memory_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CostMetrics:
    """Cost and token usage metrics."""
    total_cost: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_per_1k_tokens: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance timing metrics."""
    init_time_ms: float
    execution_time_ms: float
    total_time_ms: float
    iterations: int
    mean_time_ms: Optional[float] = None
    median_time_ms: Optional[float] = None
    stdev_time_ms: Optional[float] = None
    min_time_ms: Optional[float] = None
    max_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single approach."""
    name: str
    memory: MemoryMetrics
    performance: PerformanceMetrics
    cost: CostMetrics
    metadata: Dict[str, Any]
    sample_output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "memory": self.memory.to_dict(),
            "performance": self.performance.to_dict(),
            "cost": self.cost.to_dict(),
            "metadata": self.metadata,
            "sample_output": self.sample_output
        }


class MemoryProfiler:
    """Profile memory usage of objects and operations."""
    
    def __init__(self):
        self.current_memory = 0.0
        self.peak_memory = 0.0
    
    def start_tracking(self) -> None:
        """Start tracking memory allocations."""
        tracemalloc.start()
    
    def stop_tracking(self) -> tuple[float, float]:
        """Stop tracking and return (current_mb, peak_mb)."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        current_mb = current / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)
        
        self.current_memory = current_mb
        self.peak_memory = peak_mb
        
        return current_mb, peak_mb
    
    @staticmethod
    def measure_object_size(obj: Any) -> tuple[int, int]:
        """
        Measure object size in bytes.
        
        Returns:
            tuple: (shallow_size, deep_size)
        """
        shallow_size = sys.getsizeof(obj)
        
        # Try to get deep size using pympler if available
        try:
            from pympler import asizeof
            deep_size = asizeof.asizeof(obj)
        except ImportError:
            # Fallback: use shallow size
            deep_size = shallow_size
        
        return shallow_size, deep_size
    
    def profile_operation(self, operation: Callable, *args, **kwargs) -> MemoryMetrics:
        """
        Profile memory usage of an operation.
        
        Args:
            operation: Callable to execute
            *args: Arguments for the callable
            **kwargs: Keyword arguments for the callable
            
        Returns:
            MemoryMetrics with memory usage information
        """
        self.start_tracking()
        
        result = operation(*args, **kwargs)
        
        current_mb, peak_mb = self.stop_tracking()
        
        shallow_size, deep_size = self.measure_object_size(result)
        
        return MemoryMetrics(
            shallow_size_bytes=shallow_size,
            deep_size_bytes=deep_size,
            peak_memory_mb=peak_mb,
            current_memory_mb=current_mb
        )


class PerformanceProfiler:
    """Profile execution time and performance metrics."""
    
    @staticmethod
    def measure_time(operation: Callable, *args, **kwargs) -> tuple[Any, float]:
        """
        Measure execution time of an operation.
        
        Returns:
            tuple: (result, elapsed_time_ms)
        """
        start = time.perf_counter()
        result = operation(*args, **kwargs)
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        return result, elapsed_ms
    
    @staticmethod
    def measure_multiple_runs(
        operation: Callable,
        iterations: int = 10,
        warmup: int = 1,
        *args,
        **kwargs
    ) -> PerformanceMetrics:
        """
        Measure performance across multiple runs with statistics.
        
        Args:
            operation: Callable to execute
            iterations: Number of iterations to run
            warmup: Number of warmup runs (not counted)
            *args: Arguments for the callable
            **kwargs: Keyword arguments for the callable
            
        Returns:
            PerformanceMetrics with detailed timing statistics
        """
        # Warmup runs
        for _ in range(warmup):
            operation(*args, **kwargs)
        
        # Actual measurement runs
        times = []
        for _ in range(iterations):
            _, elapsed = PerformanceProfiler.measure_time(operation, *args, **kwargs)
            times.append(elapsed)
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)
        
        return PerformanceMetrics(
            init_time_ms=0.0,  # To be set separately if needed
            execution_time_ms=mean_time,
            total_time_ms=mean_time,
            iterations=iterations,
            mean_time_ms=mean_time,
            median_time_ms=median_time,
            stdev_time_ms=stdev_time,
            min_time_ms=min_time,
            max_time_ms=max_time
        )


class BenchmarkReporter:
    """Generate and save benchmark reports."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information for the report."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        }
    
    @staticmethod
    def create_comparison_report(
        results: List[BenchmarkResult],
        test_name: str,
        upsonic_version: str = "0.1.0"
    ) -> Dict[str, Any]:
        """
        Create a comprehensive comparison report.
        
        Args:
            results: List of BenchmarkResult objects
            test_name: Name of the test scenario
            upsonic_version: Version of Upsonic framework
            
        Returns:
            Dictionary with complete report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "upsonic_version": upsonic_version,
            "system_info": BenchmarkReporter.get_system_info(),
            "results": {result.name: result.to_dict() for result in results}
        }
        
        # Add comparison based on number of results
        direct_result = next((r for r in results if r.name == "Direct"), None)
        agent_no_prompt = next((r for r in results if "no prompt" in r.name), None)
        agent_with_prompt = next((r for r in results if "with prompt" in r.name), None)
        
        # Handle 3-way comparison (Direct, Agent no prompt, Agent with prompt)
        if len(results) == 3 and direct_result and agent_no_prompt and agent_with_prompt:
            comparisons = {}
            
            # Direct vs Agent (no prompt)
            comparisons["direct_vs_agent_no_prompt"] = BenchmarkReporter._create_pair_comparison(
                direct_result, agent_no_prompt, "Direct", "Agent (no prompt)"
            )
            
            # Direct vs Agent (with prompt)
            comparisons["direct_vs_agent_with_prompt"] = BenchmarkReporter._create_pair_comparison(
                direct_result, agent_with_prompt, "Direct", "Agent (with prompt)"
            )
            
            # Agent (no prompt) vs Agent (with prompt)
            comparisons["agent_no_prompt_vs_with_prompt"] = BenchmarkReporter._create_pair_comparison(
                agent_no_prompt, agent_with_prompt, "Agent (no prompt)", "Agent (with prompt)"
            )
            
            report["comparison"] = comparisons
            
        # Handle 2-way comparison (backward compatibility)
        elif len(results) == 2 and direct_result:
            agent_result = next((r for r in results if r.name != "Direct"), None)
            if agent_result:
                comparison = BenchmarkReporter._create_pair_comparison(
                    direct_result, agent_result, "Direct", agent_result.name
                )
                report["comparison"] = comparison
        
        return report
    
    @staticmethod
    def _add_pair_comparison_to_markdown(md_lines: List[str], comp: Dict[str, Any]) -> None:
        """Add a pair comparison section to markdown lines."""
        name_a = comp.get('name_a', 'A')
        name_b = comp.get('name_b', 'B')
        
        # Speed comparison
        speed_ratio = comp['speed_improvement_ratio']
        speed_diff = comp['faster_by_ms']
        
        md_lines.append(f"**⚡ Speed**")
        md_lines.append("")
        if speed_ratio > 1:
            md_lines.append(f"- {name_b} is **{speed_ratio:.2f}x slower** than {name_a}")
            md_lines.append(f"- {name_a} completes tasks **{abs(speed_diff):.2f} ms faster**")
        else:
            md_lines.append(f"- {name_b} is **{1/speed_ratio:.2f}x faster** than {name_a}")
            md_lines.append(f"- {name_b} completes tasks **{abs(speed_diff):.2f} ms faster**")
        md_lines.append("")
        
        # Memory comparison
        mem_ratio = comp['memory_overhead_ratio']
        mem_diff = comp['memory_overhead_mb']
        md_lines.append(f"**💾 Memory**")
        md_lines.append("")
        if mem_ratio > 1:
            md_lines.append(f"- {name_b} uses **{mem_ratio:.2f}x more memory** than {name_a}")
            md_lines.append(f"- Memory overhead: **{mem_diff:.2f} MB** ({mem_diff*1024:.2f} KB)")
        else:
            md_lines.append(f"- {name_b} uses **{mem_ratio:.2f}x less memory** than {name_a}")
            md_lines.append(f"- Memory saved: **{abs(mem_diff):.2f} MB** ({abs(mem_diff)*1024:.2f} KB)")
        md_lines.append("")
        
        # Cost comparison (if available)
        if "cost_ratio" in comp:
            cost_ratio = comp['cost_ratio']
            cost_diff = comp.get('cost_difference', 0.0)
            md_lines.append(f"**💰 Cost**")
            md_lines.append("")
            if cost_ratio > 1:
                md_lines.append(f"- {name_b} costs **{cost_ratio:.2f}x more** than {name_a}")
                md_lines.append(f"- Cost difference: **${cost_diff:.6f}**")
                md_lines.append(f"- {name_a} total cost: **${comp['cost_a']:.6f}**")
                md_lines.append(f"- {name_b} total cost: **${comp['cost_b']:.6f}**")
            else:
                md_lines.append(f"- {name_b} costs **{1/cost_ratio:.2f}x less** than {name_a}")
                md_lines.append(f"- Cost savings: **${abs(cost_diff):.6f}**")
                md_lines.append(f"- {name_a} total cost: **${comp['cost_a']:.6f}**")
                md_lines.append(f"- {name_b} total cost: **${comp['cost_b']:.6f}**")
            md_lines.append("")
    
    @staticmethod
    def _add_detailed_comparison_table(md_lines: List[str], report: Dict[str, Any]) -> None:
        """Add a detailed comparison table showing all three approaches side-by-side."""
        results = report["results"]
        
        # Extract the three results
        direct = results.get("Direct")
        agent_no = results.get("Agent (no prompt)")
        agent_with = results.get("Agent (with prompt)")
        
        if not (direct and agent_no and agent_with):
            return
        
        # Calculate mean values per iteration
        direct_perf = direct["performance"]
        agent_no_perf = agent_no["performance"]
        agent_with_perf = agent_with["performance"]
        
        direct_cost = direct["cost"]
        agent_no_cost = agent_no["cost"]
        agent_with_cost = agent_with["cost"]
        
        # Mean costs per iteration
        direct_mean_cost = direct_cost["total_cost"] / direct_perf["iterations"]
        agent_no_mean_cost = agent_no_cost["total_cost"] / agent_no_perf["iterations"]
        agent_with_mean_cost = agent_with_cost["total_cost"] / agent_with_perf["iterations"]
        
        # Mean tokens per iteration
        direct_mean_input = direct_cost["input_tokens"] / direct_perf["iterations"]
        agent_no_mean_input = agent_no_cost["input_tokens"] / agent_no_perf["iterations"]
        agent_with_mean_input = agent_with_cost["input_tokens"] / agent_with_perf["iterations"]
        
        direct_mean_output = direct_cost["output_tokens"] / direct_perf["iterations"]
        agent_no_mean_output = agent_no_cost["output_tokens"] / agent_no_perf["iterations"]
        agent_with_mean_output = agent_with_cost["output_tokens"] / agent_with_perf["iterations"]
        
        # Create markdown table
        md_lines.append("| Metric | Direct | Agent (No Prompt) | Agent (With Prompt) |")
        md_lines.append("|--------|--------|-------------------|---------------------|")
        
        # Speed metrics
        md_lines.append(f"| **Mean Time (ms)** | {direct_perf['mean_time_ms']:.2f} | {agent_no_perf['mean_time_ms']:.2f} | {agent_with_perf['mean_time_ms']:.2f} |")
        md_lines.append(f"| **Stdev Time (ms)** | {direct_perf['stdev_time_ms']:.2f} | {agent_no_perf['stdev_time_ms']:.2f} | {agent_with_perf['stdev_time_ms']:.2f} |")
        md_lines.append(f"| **Median Time (ms)** | {direct_perf['median_time_ms']:.2f} | {agent_no_perf['median_time_ms']:.2f} | {agent_with_perf['median_time_ms']:.2f} |")
        md_lines.append(f"| **Min Time (ms)** | {direct_perf['min_time_ms']:.2f} | {agent_no_perf['min_time_ms']:.2f} | {agent_with_perf['min_time_ms']:.2f} |")
        md_lines.append(f"| **Max Time (ms)** | {direct_perf['max_time_ms']:.2f} | {agent_no_perf['max_time_ms']:.2f} | {agent_with_perf['max_time_ms']:.2f} |")
        
        # Memory metrics
        md_lines.append(f"| **Memory (bytes)** | {direct['memory']['deep_size_bytes']:,} | {agent_no['memory']['deep_size_bytes']:,} | {agent_with['memory']['deep_size_bytes']:,} |")
        
        # Cost metrics
        md_lines.append(f"| **Mean Cost ($/iter)** | ${direct_mean_cost:.8f} | ${agent_no_mean_cost:.8f} | ${agent_with_mean_cost:.8f} |")
        md_lines.append(f"| **Total Cost ($)** | ${direct_cost['total_cost']:.8f} | ${agent_no_cost['total_cost']:.8f} | ${agent_with_cost['total_cost']:.8f} |")
        
        # Token metrics (mean per iteration)
        md_lines.append(f"| **Mean Input Tokens** | {direct_mean_input:.1f} | {agent_no_mean_input:.1f} | {agent_with_mean_input:.1f} |")
        md_lines.append(f"| **Mean Output Tokens** | {direct_mean_output:.1f} | {agent_no_mean_output:.1f} | {agent_with_mean_output:.1f} |")
        md_lines.append(f"| **Mean Total Tokens** | {direct_mean_input + direct_mean_output:.1f} | {agent_no_mean_input + agent_no_mean_output:.1f} | {agent_with_mean_input + agent_with_mean_output:.1f} |")
        
        # Token metrics (total)
        md_lines.append(f"| **Total Input Tokens** | {direct_cost['input_tokens']:,} | {agent_no_cost['input_tokens']:,} | {agent_with_cost['input_tokens']:,} |")
        md_lines.append(f"| **Total Output Tokens** | {direct_cost['output_tokens']:,} | {agent_no_cost['output_tokens']:,} | {agent_with_cost['output_tokens']:,} |")
        md_lines.append(f"| **Total Tokens** | {direct_cost['total_tokens']:,} | {agent_no_cost['total_tokens']:,} | {agent_with_cost['total_tokens']:,} |")
    
    @staticmethod
    def _add_sample_outputs(md_lines: List[str], report: Dict[str, Any]) -> None:
        """Add sample outputs section showing how each approach answered the task."""
        results = report["results"]
        
        # Extract the three results
        direct = results.get("Direct")
        agent_no = results.get("Agent (no prompt)")
        agent_with = results.get("Agent (with prompt)")
        
        if not (direct and agent_no and agent_with):
            return
        
        # Direct output
        if direct.get("sample_output"):
            md_lines.append("### Direct")
            md_lines.append("")
            md_lines.append("> " + direct["sample_output"].replace("\n", "\n> "))
            md_lines.append("")
        
        # Agent (no prompt) output
        if agent_no.get("sample_output"):
            md_lines.append("### Agent (No Prompt)")
            md_lines.append("")
            md_lines.append("> " + agent_no["sample_output"].replace("\n", "\n> "))
            md_lines.append("")
        
        # Agent (with prompt) output
        if agent_with.get("sample_output"):
            md_lines.append("### Agent (With Prompt)")
            md_lines.append("")
            md_lines.append("> " + agent_with["sample_output"].replace("\n", "\n> "))
            md_lines.append("")
    
    @staticmethod
    def _create_pair_comparison(
        result_a: BenchmarkResult,
        result_b: BenchmarkResult,
        name_a: str,
        name_b: str
    ) -> Dict[str, Any]:
        """Create comparison metrics between two results."""
        comparison = {
            "name_a": name_a,
            "name_b": name_b,
            "memory_overhead_ratio": (
                result_b.memory.deep_size_bytes / result_a.memory.deep_size_bytes
            ),
            "speed_improvement_ratio": (
                result_b.performance.execution_time_ms / result_a.performance.execution_time_ms
            ),
            "faster_by_ms": (
                result_b.performance.execution_time_ms - result_a.performance.execution_time_ms
            ),
            "memory_overhead_mb": (
                result_b.memory.peak_memory_mb - result_a.memory.peak_memory_mb
            )
        }
        
        # Add cost comparison if both have cost data
        if result_a.cost.total_cost > 0 and result_b.cost.total_cost > 0:
            comparison["cost_ratio"] = result_b.cost.total_cost / result_a.cost.total_cost
            comparison["cost_difference"] = result_b.cost.total_cost - result_a.cost.total_cost
            comparison["cost_a"] = result_a.cost.total_cost
            comparison["cost_b"] = result_b.cost.total_cost
        
        return comparison
    
    @staticmethod
    def save_report(report: Dict[str, Any], output_dir: Path) -> Path:
        """
        Save benchmark report to JSON file organized by test case.
        
        Args:
            report: Report dictionary
            output_dir: Directory to save the report
            
        Returns:
            Path to the saved report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = report.get("test_name", "benchmark")
        
        # Extract base test name (without model info in parentheses)
        base_test_name = test_name.split("(")[0].strip().replace(" ", "_")
        
        # Create subdirectory for this test case
        test_dir = output_dir / base_test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with full test name
        filename = f"{test_name.replace(' ', '_')}_{timestamp}.json"
        filepath = test_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    @staticmethod
    def save_markdown_report(report: Dict[str, Any], output_dir: Path) -> Path:
        """
        Save benchmark report as readable Markdown file organized by test case.
        
        Args:
            report: Report dictionary
            output_dir: Directory to save the report
            
        Returns:
            Path to the saved markdown file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = report.get("test_name", "benchmark")
        
        # Extract base test name (without model info in parentheses)
        base_test_name = test_name.split("(")[0].strip().replace(" ", "_")
        
        # Create subdirectory for this test case
        test_dir = output_dir / base_test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with full test name
        filename = f"{test_name.replace(' ', '_')}_{timestamp}.md"
        filepath = test_dir / filename
        
        # Build markdown content
        md_lines = []
        
        # Header
        md_lines.append(f"# {report['test_name']} - Benchmark Report")
        md_lines.append("")
        md_lines.append(f"**Date:** {report['timestamp']}")
        md_lines.append(f"**Upsonic Version:** {report['upsonic_version']}")
        md_lines.append(f"**Python:** {report['system_info']['python_version']}")
        md_lines.append(f"**Platform:** {report['system_info']['platform']}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # Task Information (if available in metadata)
        if report["results"]:
            first_result = list(report["results"].values())[0]
            if "metadata" in first_result and "task_details" in first_result["metadata"]:
                task_details = first_result["metadata"]["task_details"]
                md_lines.append("## 📋 Task Information")
                md_lines.append("")
                md_lines.append(f"**Description:**")
                md_lines.append("")
                md_lines.append(f"> {task_details['description']}")
                md_lines.append("")
                
                if task_details.get('response_format'):
                    response_fmt = task_details['response_format']
                    if isinstance(response_fmt, str):
                        md_lines.append(f"**Response Format:** `{response_fmt}`")
                    else:
                        md_lines.append(f"**Response Format:** `{response_fmt}`")
                    md_lines.append("")
                
                if task_details.get('context'):
                    md_lines.append(f"**Context:** Yes ({len(task_details['context'])} item(s))")
                    md_lines.append("")
                
                if task_details.get('attachments'):
                    md_lines.append(f"**Attachments:** {', '.join(task_details['attachments'])}")
                    md_lines.append("")
                
        md_lines.append("---")
        md_lines.append("")
        
        # For 3-way comparison, only show the detailed comparison table
        if "comparison" in report:
            comp = report["comparison"]
            # Check if we have multiple comparisons (3-way)
            if isinstance(comp, dict) and "direct_vs_agent_no_prompt" in comp:
                # Add detailed comparison table for 3-way comparison
                md_lines.append("## 📊 Detailed Comparison Table")
                md_lines.append("")
                BenchmarkReporter._add_detailed_comparison_table(md_lines, report)
                md_lines.append("")
                
                # Add sample outputs section
                md_lines.append("## 📝 Sample Outputs")
                md_lines.append("")
                BenchmarkReporter._add_sample_outputs(md_lines, report)
                md_lines.append("")
            else:
                # 2-way comparison (backward compatibility) - show detailed results
                # Results for each approach
                for name, result in report["results"].items():
                    md_lines.append(f"## {name}")
                    md_lines.append("")
                    
                    # Metadata
                    if result.get('metadata'):
                        md_lines.append("### Configuration")
                        md_lines.append("")
                        for key, value in result['metadata'].items():
                            # Skip task_details as it's already shown at the top
                            if key == 'task_details':
                                continue
                            md_lines.append(f"- **{key.replace('_', ' ').title()}:** `{value}`")
                        md_lines.append("")
                    
                    # Memory metrics
                    mem = result['memory']
                    md_lines.append("### Memory Usage")
                    md_lines.append("")
                    md_lines.append(f"- **Object Size:** {mem['deep_size_bytes']:,} bytes ({mem['deep_size_bytes']/1024:.2f} KB)")
                    md_lines.append(f"- **Peak Memory:** {mem['peak_memory_mb']:.2f} MB")
                    md_lines.append(f"- **Current Memory:** {mem['current_memory_mb']:.2f} MB")
                    md_lines.append("")
                    
                    # Performance metrics
                    perf = result['performance']
                    md_lines.append("### Performance Metrics")
                    md_lines.append("")
                    md_lines.append(f"- **Initialization Time:** {perf['init_time_ms']:.2f} ms")
                    md_lines.append(f"- **Execution Time:** {perf['execution_time_ms']:.2f} ms")
                    md_lines.append(f"- **Total Time:** {perf['total_time_ms']:.2f} ms")
                    
                    if perf.get('iterations') and perf['iterations'] > 1:
                        md_lines.append("")
                        md_lines.append("**Statistical Analysis:**")
                        md_lines.append("")
                        md_lines.append(f"- Iterations: {perf['iterations']}")
                        md_lines.append(f"- Mean: {perf.get('mean_time_ms', 0):.2f} ms")
                        md_lines.append(f"- Median: {perf.get('median_time_ms', 0):.2f} ms")
                        md_lines.append(f"- Std Dev: {perf.get('stdev_time_ms', 0):.2f} ms")
                        md_lines.append(f"- Min: {perf.get('min_time_ms', 0):.2f} ms")
                        md_lines.append(f"- Max: {perf.get('max_time_ms', 0):.2f} ms")
                    
                    md_lines.append("")
                    
                    # Cost metrics
                    cost = result['cost']
                    md_lines.append("### Cost Metrics")
                    md_lines.append("")
                    md_lines.append(f"- **Total Cost:** ${cost['total_cost']:.6f}")
                    md_lines.append(f"- **Input Tokens:** {cost['input_tokens']:,}")
                    md_lines.append(f"- **Output Tokens:** {cost['output_tokens']:,}")
                    md_lines.append(f"- **Total Tokens:** {cost['total_tokens']:,}")
                    md_lines.append(f"- **Cost per 1K tokens:** ${cost['cost_per_1k_tokens']:.6f}")
                    
                    md_lines.append("")
                    md_lines.append("---")
                    md_lines.append("")
                
                # 2-way comparison
                name_a = comp.get('name_a', 'Direct')
                name_b = comp.get('name_b', 'Agent')
                md_lines.append(f"## Comparison: {name_a} vs {name_b}")
                md_lines.append("")
                BenchmarkReporter._add_pair_comparison_to_markdown(md_lines, comp)
                
                # Summary (only for 2-way comparison)
                md_lines.append("### 📊 Summary")
                md_lines.append("")
                
                # Get speed ratio from the comparison
                speed_ratio = comp.get('speed_improvement_ratio', 1.0)
                
                # Determine winner for different use cases
                if speed_ratio > 1.1:
                    md_lines.append("✅ **Use Direct when:**")
                    md_lines.append("- Speed is critical")
                    md_lines.append("- Simple query/response tasks")
                    md_lines.append("- Memory efficiency matters")
                    md_lines.append("- No need for tools or memory management")
                    md_lines.append("")
                
                md_lines.append("✅ **Use Agent when:**")
                md_lines.append("- Complex workflows required")
                md_lines.append("- Tool orchestration needed")
                md_lines.append("- Memory management important")
                md_lines.append("- Safety policies and guardrails needed")
                md_lines.append("")
        
        # Footer
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("*Generated by Upsonic Benchmark System*")
        
        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        
        return filepath
    
    @staticmethod
    def print_summary(report: Dict[str, Any]) -> None:
        """Print a human-readable summary of the benchmark results."""
        print("\n" + "=" * 80)
        print(f"BENCHMARK RESULTS: {report['test_name']}")
        print("=" * 80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Upsonic Version: {report['upsonic_version']}")
        print(f"Python: {report['system_info']['python_version']}")
        print(f"Platform: {report['system_info']['platform']}")
        print("-" * 80)
        
        # Print task details if available
        if report["results"]:
            first_result = list(report["results"].values())[0]
            if "metadata" in first_result and "task_details" in first_result["metadata"]:
                task_details = first_result["metadata"]["task_details"]
                print("\nTASK:")
                print(f"  Description: {task_details['description'][:100]}...")
                print(f"  Response Format: {task_details['response_format']}")
                if task_details.get('context'):
                    print(f"  Context: Yes")
                if task_details.get('attachments'):
                    print(f"  Attachments: {', '.join(task_details['attachments'])}")
        
        for name, result in report["results"].items():
            print(f"\n{name.upper()}:")
            print(f"  Memory:")
            print(f"    Object Size: {result['memory']['deep_size_bytes']:,} bytes")
            print(f"    Peak Memory: {result['memory']['peak_memory_mb']:.2f} MB")
            print(f"  Performance:")
            print(f"    Execution Time: {result['performance']['execution_time_ms']:.2f} ms")
            if result['performance'].get('mean_time_ms'):
                print(f"    Mean Time: {result['performance']['mean_time_ms']:.2f} ms")
                print(f"    Median Time: {result['performance']['median_time_ms']:.2f} ms")
                print(f"    Std Dev: {result['performance']['stdev_time_ms']:.2f} ms")
            print(f"  Cost:")
            print(f"    Total Cost: ${result['cost']['total_cost']:.6f}")
            print(f"    Input Tokens: {result['cost']['input_tokens']:,}")
            print(f"    Output Tokens: {result['cost']['output_tokens']:,}")
            print(f"    Cost per 1K tokens: ${result['cost']['cost_per_1k_tokens']:.6f}")
        
        if "comparison" in report:
            print("\n" + "-" * 80)
            comp = report["comparison"]
            
            # Check if we have 3-way comparison or 2-way
            if isinstance(comp, dict) and "direct_vs_agent_no_prompt" in comp:
                # 3-way comparison - print summary
                print("COMPARISONS:")
                
                # Direct vs Agent (no prompt)
                dvnp = comp["direct_vs_agent_no_prompt"]
                print(f"\n  Direct vs Agent (no prompt):")
                if dvnp['speed_improvement_ratio'] > 1:
                    print(f"    Speed: Direct is {dvnp['speed_improvement_ratio']:.2f}x faster")
                else:
                    print(f"    Speed: Agent is {1/dvnp['speed_improvement_ratio']:.2f}x faster")
                print(f"    Memory: Agent uses {dvnp['memory_overhead_ratio']:.2f}x more")
                if 'cost_ratio' in dvnp:
                    print(f"    Cost: Agent costs {dvnp['cost_ratio']:.2f}x more" if dvnp['cost_ratio'] > 1 
                          else f"    Cost: Agent costs {1/dvnp['cost_ratio']:.2f}x less")
                
                # Direct vs Agent (with prompt)
                dvwp = comp["direct_vs_agent_with_prompt"]
                print(f"\n  Direct vs Agent (with prompt):")
                if dvwp['speed_improvement_ratio'] > 1:
                    print(f"    Speed: Direct is {dvwp['speed_improvement_ratio']:.2f}x faster")
                else:
                    print(f"    Speed: Agent is {1/dvwp['speed_improvement_ratio']:.2f}x faster")
                print(f"    Memory: Agent uses {dvwp['memory_overhead_ratio']:.2f}x more")
                if 'cost_ratio' in dvwp:
                    print(f"    Cost: Agent costs {dvwp['cost_ratio']:.2f}x more" if dvwp['cost_ratio'] > 1 
                          else f"    Cost: Agent costs {1/dvwp['cost_ratio']:.2f}x less")
                
                # Agent (no prompt) vs Agent (with prompt)
                npvwp = comp["agent_no_prompt_vs_with_prompt"]
                print(f"\n  Agent (no prompt) vs Agent (with prompt):")
                if npvwp['speed_improvement_ratio'] > 1:
                    print(f"    Speed: No prompt is {npvwp['speed_improvement_ratio']:.2f}x faster")
                else:
                    print(f"    Speed: With prompt is {1/npvwp['speed_improvement_ratio']:.2f}x faster")
                print(f"    Memory: With prompt uses {npvwp['memory_overhead_ratio']:.2f}x more")
                if 'cost_ratio' in npvwp:
                    print(f"    Cost: With prompt costs {npvwp['cost_ratio']:.2f}x more" if npvwp['cost_ratio'] > 1 
                          else f"    Cost: With prompt costs {1/npvwp['cost_ratio']:.2f}x less")
            else:
                # 2-way comparison (backward compatibility)
                print("COMPARISON:")
                name_a = comp.get('name_a', 'Direct')
                name_b = comp.get('name_b', 'Agent')
                speed_ratio = comp['speed_improvement_ratio']
                faster_by = comp['faster_by_ms']
                mem_ratio = comp['memory_overhead_ratio']
                mem_overhead = comp['memory_overhead_mb']
                
                if speed_ratio > 1:
                    print(f"  {name_a} is {speed_ratio:.2f}x faster than {name_b}")
                    print(f"  {name_a} is faster by {abs(faster_by):.2f} ms")
                else:
                    print(f"  {name_b} is {1/speed_ratio:.2f}x faster than {name_a}")
                    print(f"  {name_b} is faster by {abs(faster_by):.2f} ms")
                print(f"  {name_b} has {mem_ratio:.2f}x more memory overhead")
                print(f"  {name_b} uses {mem_overhead:.2f} MB more memory")
        
        print("=" * 80 + "\n")

