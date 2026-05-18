from __future__ import annotations
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from upsonic.agent.agent import Agent
from upsonic.tasks.tasks import Task
from upsonic.graph.graph import Graph, TaskNode
from upsonic.eval.models import ReliabilityEvaluationResult, ToolCallCheck
from upsonic.eval._pl_helpers import extract_model_parameters
from upsonic.utils.printing import console

_logger = logging.getLogger(__name__)

from rich.table import Table
from rich.panel import Panel

if TYPE_CHECKING:
    from upsonic.integrations.promptlayer import PromptLayer


class ReliabilityEvaluator:
    """
    A post-execution assertion and verification engine for an agent's tool usage.
    """

    def __init__(
        self,
        expected_tool_calls: List[str],
        order_matters: bool = False,
        exact_match: bool = False,
        promptlayer: Optional["PromptLayer"] = None,
        agent_under_test: Optional[Agent] = None,
    ):
        if not isinstance(expected_tool_calls, list) or not all(isinstance(i, str) for i in expected_tool_calls):
            raise TypeError("`expected_tool_calls` must be a list of strings.")
        if not expected_tool_calls:
            raise ValueError("`expected_tool_calls` cannot be an empty list.")

        self.expected_tool_calls: List[str] = expected_tool_calls
        self.order_matters: bool = order_matters
        self.exact_match: bool = exact_match
        self.promptlayer: Optional["PromptLayer"] = promptlayer
        self.agent_under_test: Optional[Agent] = agent_under_test

    def run(
        self, 
        run_result: Union[Task, List[Task], Graph],
        print_results: bool = True
    ) -> ReliabilityEvaluationResult:
        eval_start_time: float = time.time()
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

        final_result: ReliabilityEvaluationResult = ReliabilityEvaluationResult(
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

        eval_end_time: float = time.time()

        if self.promptlayer is not None:
            input_tokens: int
            output_tokens: int
            price: float
            input_tokens, output_tokens, price = self._extract_task_usage(run_result)
            self._log_eval_to_promptlayer_background(
                final_result,
                start_time=eval_start_time,
                end_time=eval_end_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                price=price,
            )

        return final_result

    def _extract_task_usage(self, run_result: Union[Task, List[Task], Graph]) -> tuple[int, int, float]:
        """Extract aggregated token usage and cost from Task objects in the run result."""
        tasks: List[Task] = []

        if isinstance(run_result, Task):
            tasks.append(run_result)
        elif isinstance(run_result, list):
            tasks.extend(t for t in run_result if isinstance(t, Task))
        elif isinstance(run_result, Graph):
            executed_node_ids = set(run_result.state.task_outputs.keys())
            for node in run_result.nodes:
                if isinstance(node, TaskNode) and node.id in executed_node_ids:
                    tasks.append(node.task)

        total_input: int = 0
        total_output: int = 0
        total_cost: float = 0.0

        for task in tasks:
            usage = getattr(task, "_usage", None)
            if usage is None:
                continue
            total_input += getattr(usage, "input_tokens", 0) or 0
            total_output += getattr(usage, "output_tokens", 0) or 0
            cost = getattr(usage, "cost", None)
            if cost is not None:
                total_cost += cost

        return total_input, total_output, total_cost

    def _log_eval_to_promptlayer_background(
        self,
        result: ReliabilityEvaluationResult,
        *,
        start_time: float,
        end_time: float,
        input_tokens: int,
        output_tokens: int,
        price: float,
    ) -> None:
        """Fire-and-forget: launches PromptLayer eval logging in a background thread."""
        def _run():
            try:
                self._log_eval_to_promptlayer(
                    result,
                    start_time=start_time,
                    end_time=end_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    price=price,
                )
            except Exception as e:
                _logger.warning("Background PromptLayer eval logging failed: %s", e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _log_eval_to_promptlayer(
        self,
        result: ReliabilityEvaluationResult,
        *,
        start_time: float,
        end_time: float,
        input_tokens: int,
        output_tokens: int,
        price: float,
    ) -> None:
        if self.promptlayer is None:
            return

        per_tool_scores: dict[str, int] = {
            f"tool_{check.tool_name}_called": 100 if check.was_called else 0
            for check in result.checks
        }

        provider: str = "upsonic"
        model: str = "reliability_eval"
        model_params: Optional[Dict[str, Any]] = None

        if self.agent_under_test is not None:
            agent_model: str = getattr(self.agent_under_test, "model_name", "unknown")
            provider, model = self.promptlayer._parse_provider_model(str(agent_model))
            model_params = extract_model_parameters(self.agent_under_test)

        self.promptlayer.log(
            provider=provider,
            model=model,
            input_text=", ".join(result.expected_tool_calls),
            output_text=", ".join(result.actual_tool_calls),
            start_time=start_time,
            end_time=end_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            price=price,
            parameters=model_params,
            tags=["upsonic-eval", "reliability-eval"],
            metadata={
                "eval_type": "reliability",
                "passed": result.passed,
                "summary": result.summary,
                "expected_tool_calls": result.expected_tool_calls,
                "actual_tool_calls": result.actual_tool_calls,
                "missing_tool_calls": result.missing_tool_calls,
                "unexpected_tool_calls": result.unexpected_tool_calls,
            },
            score=100 if result.passed else 0,
            status="SUCCESS",
            function_name="reliability_eval",
            scores=per_tool_scores,
        )

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
        if result.passed:
            color = "green"
            title = "[bold green]✅ Reliability Check Passed[/bold green]"
        else:
            color = "red"
            title = "[bold red]❌ Reliability Check Failed[/bold red]"

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
