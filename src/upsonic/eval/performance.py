from __future__ import annotations
import asyncio
import copy
import time
import tracemalloc
import statistics
from typing import Union, List, Dict

from upsonic.agent.agent import Agent
from upsonic.graph.graph import Graph
from upsonic.team.team import Team
from upsonic.tasks.tasks import Task
from upsonic.eval.models import PerformanceRunResult, PerformanceEvaluationResult
from upsonic.utils.printing import console, debug_log

from rich.table import Table


class PerformanceEvaluator:
    """
    The main user-facing profiler for measuring the latency and memory
    footprint of Upsonic agents, graphs, or teams.
    """

    def __init__(
        self,
        agent_under_test: Union[Agent, Graph, Team],
        task: Union[Task, List[Task]],
        num_iterations: int = 10,
        warmup_runs: int = 2
    ):
        if not isinstance(agent_under_test, (Agent, Graph, Team)):
            raise TypeError("The `agent_under_test` must be an instance of `Agent`, `Graph`, or `Team`.")
        if not isinstance(task, (Task, list)):
            raise TypeError("The `task` must be an instance of `Task` or a list of `Task` objects.")
        if not isinstance(num_iterations, int) or num_iterations < 1:
            raise ValueError("`num_iterations` must be a positive integer.")
        if not isinstance(warmup_runs, int) or warmup_runs < 0:
            raise ValueError("`warmup_runs` must be a non-negative integer.")

        self.agent_under_test = agent_under_test
        self.task = task
        self.num_iterations = num_iterations
        self.warmup_runs = warmup_runs

    async def run(self, print_results: bool = True) -> PerformanceEvaluationResult:
        """
        Executes the end-to-end performance profiling workflow.

        This method will:
        1. Perform warmup runs to ensure steady-state measurements.
        2. Execute the component for a set number of iterations, capturing
           high-precision latency and memory metrics for each run.
        3. Aggregate the metrics into detailed statistics.
        4. Return a final `PerformanceEvaluationResult` object.

        Args:
            print_results: If True, prints a formatted summary of the results to the console.

        Returns:
            A `PerformanceEvaluationResult` object with detailed statistics.
        """
        tracemalloc.start()

        if self.warmup_runs > 0:
            console.print(f"[bold dim]Running {self.warmup_runs} warmup iteration(s)...[/bold dim]")
            for _ in range(self.warmup_runs):
                # Don't deepcopy agent_under_test - it contains unpicklable objects like httpx.AsyncClient
                # Each task execution should be stateless anyway
                task_for_this_run = copy.deepcopy(self.task)
                await self._execute_component(self.agent_under_test, task_for_this_run)

        all_run_results: List[PerformanceRunResult] = []
        console.print(f"[bold blue]Running {self.num_iterations} measurement iteration(s)...[/bold blue]")
        for _ in range(self.num_iterations):
            # Don't deepcopy agent_under_test - it contains unpicklable objects like httpx.AsyncClient
            # Each task execution should be stateless anyway
            task_for_this_run = copy.deepcopy(self.task)

            tracemalloc.clear_traces()
            start_mem, _ = tracemalloc.get_traced_memory()
            debug_log(f"start_mem: {start_mem}", context="PerformanceEvaluator")
            
            start_time = time.perf_counter()

            await self._execute_component(self.agent_under_test, task_for_this_run)

            end_time = time.perf_counter()
            latency = end_time - start_time
            
            end_mem, peak_mem = tracemalloc.get_traced_memory()
            debug_log(f"end_mem: {end_mem}, peak_mem: {peak_mem}", context="PerformanceEvaluator")
            
            run_result = PerformanceRunResult(
                latency_seconds=latency,
                memory_increase_bytes=end_mem - start_mem,
                memory_peak_bytes=peak_mem - start_mem
            )
            all_run_results.append(run_result)

        tracemalloc.stop()

        final_result = self._aggregate_results(all_run_results)

        if print_results:
            self._print_formatted_results(final_result)
        
        return final_result

    async def _execute_component(self, agent: Union[Agent, Graph, Team], task: Union[Task, List[Task]]) -> None:
        """Internal helper now accepts the components to execute."""
        if isinstance(agent, Agent):
            task_to_run = task[0] if isinstance(task, list) else task
            await agent.do_async(task_to_run)
        elif isinstance(agent, Graph):
            await agent.run_async(verbose=False, show_progress=False)
        elif isinstance(agent, Team):
            await agent.multi_agent_async(
                agent_configurations=agent.agents,
                tasks=task,
            )

    def _calculate_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculates and returns a dictionary of statistics for a list of numbers."""
        if not data:
            return {}
        return {
            "average": statistics.mean(data),
            "median": statistics.median(data),
            "min": min(data),
            "max": max(data),
            "std_dev": statistics.stdev(data) if len(data) > 1 else 0.0,
        }

    def _aggregate_results(self, run_results: List[PerformanceRunResult]) -> PerformanceEvaluationResult:
        """Aggregates raw run data into the final result object."""
        latencies = [r.latency_seconds for r in run_results]
        mem_increases = [float(r.memory_increase_bytes) for r in run_results]
        mem_peaks = [float(r.memory_peak_bytes) for r in run_results]

        return PerformanceEvaluationResult(
            all_runs=run_results,
            num_iterations=self.num_iterations,
            warmup_runs=self.warmup_runs,
            latency_stats=self._calculate_stats(latencies),
            memory_increase_stats=self._calculate_stats(mem_increases),
            memory_peak_stats=self._calculate_stats(mem_peaks),
        )

    def _print_formatted_results(self, result: PerformanceEvaluationResult) -> None:
        """Prints a rich, formatted table of the performance results."""
        table = Table(title=f"[bold]Performance Evaluation Results[/bold]\n({result.num_iterations} iterations, {result.warmup_runs} warmups)")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Average", style="magenta")
        table.add_column("Median", style="green")
        table.add_column("Min", style="blue")
        table.add_column("Max", style="red")
        table.add_column("Std. Dev.", style="yellow")

        def format_mem(byte_val: float) -> str:
            if abs(byte_val) < 1024:
                return f"{byte_val:.2f} B"
            elif abs(byte_val) < 1024**2:
                return f"{byte_val / 1024:.2f} KB"
            else:
                return f"{byte_val / 1024**2:.2f} MB"

        ls = result.latency_stats
        table.add_row(
            "Latency",
            f"{ls['average'] * 1000:.2f} ms",
            f"{ls['median'] * 1000:.2f} ms",
            f"{ls['min'] * 1000:.2f} ms",
            f"{ls['max'] * 1000:.2f} ms",
            f"{ls['std_dev'] * 1000:.2f} ms",
        )

        mis = result.memory_increase_stats
        table.add_row(
            "Memory Increase",
            format_mem(mis['average']),
            format_mem(mis['median']),
            format_mem(mis['min']),
            format_mem(mis['max']),
            format_mem(mis['std_dev']),
        )

        mps = result.memory_peak_stats
        table.add_row(
            "Memory Peak",
            format_mem(mps['average']),
            format_mem(mps['median']),
            format_mem(mps['min']),
            format_mem(mps['max']),
            format_mem(mps['std_dev']),
        )

        console.print(table)