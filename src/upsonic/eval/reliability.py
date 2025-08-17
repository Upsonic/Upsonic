from __future__ import annotations
from typing import List, Union

from upsonic.tasks.tasks import Task
from upsonic.graph.graph import Graph, TaskNode
from upsonic.eval.models import ReliabilityEvaluationResult, ToolCallCheck
from upsonic.utils.printing import console

from rich.table import Table
from rich.panel import Panel

class ReliabilityEvaluator:
    """
    A post-execution assertion and verification engine for an agent's tool usage.
    """

    def __init__(
        self,
        expected_tool_calls: List[str],
        order_matters: bool = False,
        exact_match: bool = False,
    ):
        if not isinstance(expected_tool_calls, list) or not all(isinstance(i, str) for i in expected_tool_calls):
            raise TypeError("`expected_tool_calls` must be a list of strings.")
        if not expected_tool_calls:
            raise ValueError("`expected_tool_calls` cannot be an empty list.")

        self.expected_tool_calls = expected_tool_calls
        self.order_matters = order_matters
        self.exact_match = exact_match

    def run(
        self, 
        run_result: Union[Task, List[Task], Graph],
        print_results: bool = True
    ) -> ReliabilityEvaluationResult:
        """
        Analyzes the result of an agent, team, or graph run and verifies its
        tool-calling behavior against the configured rules.

        Args:
            run_result: The completed result object from an execution. This can be
                        a single `Task`, a list of `Task`s (from a Team), or a
                        `Graph` object after its `run()` method has completed.
            print_results: If True, prints a formatted summary of the results.

        Returns:
            A `ReliabilityEvaluationResult` object with the detailed outcome.
        """
        actual_tool_calls = self._normalize_tool_call_history(run_result)

        passed = True
        summary_messages = []
        
        checks: List[ToolCallCheck] = []
        missing_tool_calls: List[str] = []
        for expected_tool in self.expected_tool_calls:
            count = actual_tool_calls.count(expected_tool)
            was_called = count > 0
            checks.append(ToolCallCheck(tool_name=expected_tool, was_called=was_called, times_called=count))
            if not was_called:
                passed = False
                missing_tool_calls.append(expected_tool)
        
        if missing_tool_calls:
            summary_messages.append(f"Missing expected tool calls: {', '.join(missing_tool_calls)}.")

        if self.order_matters:
            it = iter(actual_tool_calls)
            if not all(tool in it for tool in self.expected_tool_calls):
                passed = False
                summary_messages.append("Tools were not called in the expected order.")

        unexpected_tool_calls: List[str] = []
        if self.exact_match:
            unexpected_set = set(actual_tool_calls) - set(self.expected_tool_calls)
            if unexpected_set:
                passed = False
                unexpected_tool_calls = sorted(list(unexpected_set))
                summary_messages.append(f"Unexpected tools were called: {', '.join(unexpected_tool_calls)}.")

        if passed:
            summary_messages.append("All reliability checks passed.")
        
        final_summary = " ".join(summary_messages)

        final_result = ReliabilityEvaluationResult(
            passed=passed,
            summary=final_summary,
            expected_tool_calls=self.expected_tool_calls,
            actual_tool_calls=actual_tool_calls,
            checks=checks,
            missing_tool_calls=missing_tool_calls,
            unexpected_tool_calls=unexpected_tool_calls
        )

        if print_results:
            self._print_formatted_results(final_result)
        
        return final_result

    def _normalize_tool_call_history(self, run_result: Union[Task, List[Task], Graph]) -> List[str]:
        """Extracts a single, flat list of tool call names from the run result."""
        actual_tool_calls = []
        
        if isinstance(run_result, Task):
            actual_tool_calls.extend(call['tool_name'] for call in run_result.tool_calls)
        
        elif isinstance(run_result, list) and all(isinstance(t, Task) for t in run_result):
            for task in run_result:
                actual_tool_calls.extend(call['tool_name'] for call in task.tool_calls)

        elif isinstance(run_result, Graph):
            executed_node_ids = set(run_result.state.task_outputs.keys())
            for node in run_result.nodes:
                if isinstance(node, TaskNode) and node.id in executed_node_ids:
                    actual_tool_calls.extend(call['tool_name'] for call in node.task.tool_calls)
        else:
            raise TypeError(
                f"Unsupported `run_result` type for reliability evaluation: {type(run_result).__name__}. "
                "Expected Task, List[Task], or Graph."
            )
            
        return actual_tool_calls

    def _print_formatted_results(self, result: ReliabilityEvaluationResult) -> None:
        """Prints a rich, formatted summary of the reliability results."""
        if result.passed:
            color = "green"
            title = "[bold green]✅ Reliability Check Passed[/bold green]"
        else:
            color = "red"
            title = "[bold red]❌ Reliability Check Failed[/bold red]"

        panel_content = [result.summary, ""]
        
        table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
        table.add_column("Status", style=color)
        table.add_column("Tool Name", style="cyan")
        table.add_column("Times Called", style="magenta")

        for check in result.checks:
            status_icon = "✅" if check.was_called else "❌"
            table.add_row(status_icon, check.tool_name, str(check.times_called))
        
        panel = Panel(
            table,
            title=title,
            border_style=color,
            subtitle=f"[dim]Expected: {result.expected_tool_calls} | Actual: {result.actual_tool_calls}[/dim]"
        )

        console.print(panel)