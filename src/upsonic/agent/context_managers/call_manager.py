import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.models import Model
    from upsonic.run.agent.output import AgentRunOutput
    from upsonic.tasks.tasks import Task


class CallManager:
    def __init__(
        self,
        model: "Model",
        task: "Task",
        debug: bool = False,
        show_tool_calls: bool = True,
        print_output: bool = False,
    ):
        """
        Initializes the CallManager.

        Args:
            model: The instantiated model object for this call.
            task: The task being executed.
            debug: Whether debug mode is enabled.
            show_tool_calls: Whether to show tool calls.
            print_output: Whether to print output to console.
        """
        self.model: "Model" = model
        self.task: "Task" = task
        self.show_tool_calls: bool = show_tool_calls
        self.debug: bool = debug
        self.print_output: bool = print_output
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.model_response: Optional[Any] = None

    def process_response(self, model_response: Any) -> Any:
        self.model_response = model_response
        return self.model_response

    async def aprepare(self) -> None:
        """Prepare the call by recording start time."""
        self.start_time = time.time()

    async def afinalize(self) -> None:
        """Finalize the call by setting end time."""
        if self.end_time is None:
            self.end_time = time.time()
        if self.start_time is None:
            self.start_time = self.end_time

    async def alog_completion(self, context: "AgentRunOutput") -> None:
        """Log the completion with usage tracking, Tool Calls, LLM Result, and Task Metrics.

        This is the single printing entry-point for completed runs.
        It calls ``call_end`` (which prints Tool Calls + LLM Result
        and tracks price_id) and then ``print_price_id_summary`` (Task Metrics).

        Args:
            context: AgentRunOutput object containing messages and output.
        """
        if context is None:
            return

        from upsonic.utils.tool_usage import tool_usage
        from upsonic.utils.printing import call_end

        task_usage: Optional[Any] = getattr(self.task, '_usage', None)
        if task_usage is not None and (task_usage.input_tokens or task_usage.output_tokens):
            usage: Dict[str, int] = {
                "input_tokens": task_usage.input_tokens,
                "output_tokens": task_usage.output_tokens,
            }
        elif context.usage is not None and hasattr(context.usage, 'input_tokens'):
            usage = {
                "input_tokens": context.usage.input_tokens,
                "output_tokens": context.usage.output_tokens,
            }
        else:
            from upsonic.utils.llm_usage import llm_usage
            usage = llm_usage(context)

        # Always populate task._tool_calls; display is gated by show_tool_calls
        tool_usage_result: Optional[List[Dict[str, Any]]] = tool_usage(context, self.task)

        has_output: bool = context.output is not None
        has_tool_calls: bool = bool(tool_usage_result and len(tool_usage_result) > 0)
        if not has_output and not has_tool_calls:
            return

        # Use task._usage.model_execution_time — the cumulative wall-clock time
        # of ALL model.request() calls (including tool call rounds).
        task_usage = getattr(self.task, '_usage', None)
        model_exec_time = getattr(task_usage, 'model_execution_time', None) if task_usage else None
        if model_exec_time is not None:
            effective_end = time.time()
            effective_start = effective_end - model_exec_time
        else:
            effective_end = self.end_time if self.end_time is not None else time.time()
            effective_start = self.start_time if self.start_time is not None else effective_end

        call_end(
            context.output,
            self.model,
            self.task.response_format if hasattr(self.task, 'response_format') else str,
            effective_start,
            effective_end,
            usage,
            tool_usage_result,
            self.debug,
            getattr(self.task, 'price_id', None),
            print_output=self.print_output,
            show_tool_calls=self.show_tool_calls,
        )

        if self.task and not getattr(self.task, 'not_main_task', False):
            from upsonic.utils.printing import print_price_id_summary, price_id_summary
            price_id: Optional[str] = getattr(self.task, 'price_id', None)
            if price_id and price_id in price_id_summary:
                print_price_id_summary(price_id, self.task, print_output=self.print_output)

    def prepare(self) -> None:
        """Synchronous version of aprepare."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.aprepare())

    def finalize(self) -> None:
        """Synchronous version of afinalize."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.afinalize())

    def log_completion(self, context: "AgentRunOutput") -> None:
        """Synchronous version of alog_completion."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.alog_completion(context))

    @asynccontextmanager
    async def manage_call(self):
        """
        Async context manager for call lifecycle.

        Note: This context manager is kept for backward compatibility.
        For step-based architecture, use aprepare() and afinalize() directly.
        """
        await self.aprepare()

        try:
            yield self
        finally:
            await self.afinalize()
