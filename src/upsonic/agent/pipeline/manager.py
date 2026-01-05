"""
Pipeline Manager - Orchestrates step execution.

The PipelineManager is responsible for executing a sequence of steps
in order, managing the context flow, and handling the overall execution.
It holds references to task, agent, and model, passing them to each step.
It also supports comprehensive event streaming for full visibility into
the execution pipeline.
"""

import time
from typing import List, Optional, Dict, Any, AsyncIterator, TYPE_CHECKING
from upsonic.agent.pipeline.step import Step
from .step import Step, StepResult, StepStatus

if TYPE_CHECKING:
    from upsonic.run.agent.context import AgentRunContext
    from upsonic.tasks.tasks import Task
    from upsonic.models import Model
    from upsonic.agent.agent import Agent
    from upsonic.tools.processor import ExternalExecutionPause
    from upsonic.run.events.events import AgentEvent
else:
    AgentRunContext = "AgentRunContext"
    Task = "Task"
    Model = "Model"
    Agent = "Agent"
    AgentEvent = "AgentEvent"

from upsonic.utils.logging_config import sentry_sdk, get_logger

_sentry_logger = get_logger("upsonic.sentry.pipeline")


class PipelineManager:
    """
    Manages the execution of a pipeline of steps.
    
    The PipelineManager orchestrates the agent execution by:
    - Running steps in sequence
    - Managing the shared context
    - Handling errors and early termination
    - Providing execution statistics
    - Emitting events for streaming visibility
    
    Usage:
        ```python
        manager = PipelineManager(steps=[
            CacheCheckStep(),
            ModelExecutionStep(),
            FinalizationStep()
        ])
        
        context = StepContext(task=task, agent=agent)
        result = await manager.execute(context)
        
        # Or for streaming with events:
        async for event in manager.execute_stream(context):
            if isinstance(event, StepStartEvent):
                print(f"Starting: {event.step_name}")
            elif isinstance(event, TextDeltaEvent):
                print(event.content, end='')
        ```
    """
    
    def __init__(
        self,
        steps: Optional[List[Step]] = None,
        task: Optional["Task"] = None,
        agent: Optional["Agent"] = None,
        model: Optional["Model"] = None,
        debug: bool = False
    ):
        """
        Initialize the pipeline manager.
        
        Args:
            steps: List of steps to execute
            task: The task being executed
            agent: The agent instance
            model: The model being used
            debug: Enable debug logging
        """
        self.steps: List[Step] = steps or []
        self.task = task
        self.agent = agent
        self.model = model
        self.debug = debug
        self._execution_stats: Dict[str, Any] = {}
    
    def add_step(self, step: Step) -> None:
        """Add a step to the pipeline."""
        self.steps.append(step)
    
    def insert_step(self, index: int, step: Step) -> None:
        """Insert a step at a specific position."""
        self.steps.insert(index, step)
    
    def remove_step(self, step_name: str) -> bool:
        """Remove a step by name."""
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                self.steps.pop(i)
                return True
        return False
    
    def get_step(self, step_name: str) -> Optional[Step]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    async def execute(
        self, 
        context: "AgentRunContext",
        start_step_index: int = 0
    ) -> "AgentRunContext":
        """
        Execute the pipeline (non-streaming).

        Runs all steps in sequence, passing the context through each one.
        All steps must execute. If any step raises an error, the pipeline stops,
        logs the error properly, and raises it to the caller.

        Args:
            context: The initial context
            start_step_index: Index to start execution from (0-based). Used for HITL resumption.

        Returns:
            AgentRunContext: The final context after all steps

        Raises:
            Exception: Any exception from step execution is raised with proper error message
        """
        from upsonic.run.pipeline.stats import PipelineExecutionStats
        if not context.execution_stats:
            context.execution_stats = PipelineExecutionStats(total_steps=len(self.steps))
        
        if start_step_index > 0:
            context.execution_stats.resumed_from = start_step_index
            if self.debug:
                from upsonic.utils.printing import info_log
                step_name = self.steps[start_step_index].name if start_step_index < len(self.steps) else "unknown"
                info_log(f"Resuming pipeline from step {start_step_index} ({step_name})", "PipelineManager")
        
        self._execution_stats = {
            "total_steps": len(self.steps),
            "executed_steps": 0,
            "step_results": {}
        }

        with sentry_sdk.start_transaction(
            op="agent.pipeline.execute",
            name=f"Agent Pipeline ({len(self.steps)} steps)"
        ) as transaction:
            transaction.set_tag("pipeline.total_steps", len(self.steps))
            transaction.set_tag("pipeline.debug", self.debug)
            transaction.set_tag("pipeline.streaming", context.is_streaming if hasattr(context, 'is_streaming') else False)

            if self.task:
                transaction.set_tag("task.type", type(self.task).__name__)
                if hasattr(self.task, 'description'):
                    transaction.set_data("task.description", str(self.task.description)[:200])

            _sentry_logger.info(
                "Pipeline started: %d steps",
                len(self.steps),
                extra={
                    "total_steps": len(self.steps),
                    "debug": self.debug,
                    "streaming": context.is_streaming if hasattr(context, 'is_streaming') else False
                }
            )

            if self.debug:
                from upsonic.utils.printing import pipeline_started
                pipeline_started(len(self.steps))

            try:
                for step_index in range(start_step_index, len(self.steps)):
                    step = self.steps[step_index]
                    self._current_step_index = step_index
                    self._current_step = step
                    
                    with sentry_sdk.start_span(
                        op=f"pipeline.step.{step.name}",
                        name=step.description
                    ) as span:
                        span.set_tag("step.name", step.name)
                        span.set_data("step.description", step.description)

                        if self.debug:
                            from upsonic.utils.printing import pipeline_step_started, debug_log_level2
                            pipeline_step_started(step.name, step.description)
                            
                            debug_level = getattr(self.agent, 'debug_level', 1) if self.agent else 1
                            if debug_level >= 2:
                                debug_log_level2(
                                    f"Pipeline step starting: {step.name}",
                                    "PipelineManager",
                                    debug=self.debug,
                                    debug_level=debug_level,
                                    step_name=step.name,
                                    step_description=step.description,
                                    step_index=step_index,
                                    total_steps=len(self.steps),
                                    task_description=self.task.description[:200] if self.task else None,
                                    agent_name=getattr(self.agent, 'name', 'Unknown') if self.agent else None,
                                    model_name=getattr(self.model, 'model_name', 'Unknown') if self.model else None,
                                    is_streaming=context.is_streaming
                                )

                        # Execute step - run() now returns StepResult directly
                        result = await step.run(context, self.task, self.agent, self.model, step_index)
                        
                        # Update execution stats
                        context.execution_stats.executed_steps += 1
                        context.execution_stats.step_timing[step.name] = result.execution_time
                        context.execution_stats.step_statuses[step.name] = result.status.value

                        span.set_tag("step.status", result.status.value)
                        span.set_data("step.message", result.message)
                        span.set_data("step.execution_time", result.execution_time)

                        self._execution_stats["step_results"][step.name] = {
                            "status": result.status.value,
                            "message": result.message,
                            "execution_time": result.execution_time
                        }

                        self._execution_stats["executed_steps"] += 1

                        if self.debug:
                            from upsonic.utils.printing import pipeline_step_completed, debug_log_level2
                            pipeline_step_completed(
                                step.name,
                                result.status.value,
                                result.execution_time,
                                result.message
                            )
                            
                            debug_level = getattr(self.agent, 'debug_level', 1) if self.agent else 1
                            if debug_level >= 2:
                                debug_log_level2(
                                    f"Pipeline step completed: {step.name}",
                                    "PipelineManager",
                                    debug=self.debug,
                                    debug_level=debug_level,
                                    step_name=step.name,
                                    step_status=result.status.value,
                                    execution_time=result.execution_time,
                                    step_message=result.message,
                                    cumulative_execution_time=sum(s.get('execution_time', 0) for s in self._execution_stats["step_results"].values()),
                                    steps_completed=self._execution_stats["executed_steps"]
                                )
                        
                        if result.status == StepStatus.COMPLETED:
                            pass

                total_time = sum(r["execution_time"] for r in self._execution_stats["step_results"].values())
                
                transaction.set_tag("pipeline.status", "success")
                transaction.set_data("pipeline.executed_steps", self._execution_stats['executed_steps'])
                transaction.set_data("pipeline.total_time", total_time)

                _sentry_logger.info(
                    "Pipeline completed: %d/%d steps, %.3fs",
                    self._execution_stats['executed_steps'],
                    len(self.steps),
                    total_time,
                    extra={
                        "executed_steps": self._execution_stats['executed_steps'],
                        "total_steps": len(self.steps),
                        "total_time": total_time,
                        "status": "success"
                    }
                )

                if self.debug:
                    from upsonic.utils.printing import pipeline_completed, pipeline_timeline
                    pipeline_completed(
                        self._execution_stats['executed_steps'],
                        len(self.steps),
                        total_time
                    )
                    pipeline_timeline(
                        self._execution_stats["step_results"],
                        total_time
                    )
                
                # Save on successful pipeline completion
                last_step_result = context.step_results[-1] if context.step_results else None
                if last_step_result and last_step_result.status == StepStatus.COMPLETED:
                    if self.agent and self.agent._agent_run_output:
                        from upsonic.run.base import RunStatus
                        self.agent._agent_run_output.mark_completed()
                        self.agent._agent_run_context = context
                        self.agent._agent_run_output.sync_from_context(context)
                        
                        # Save completed run to storage
                        if hasattr(self.agent, '_save_checkpoint_to_storage'):
                            try:
                                await self.agent._save_checkpoint_to_storage(context)
                            except Exception:
                                pass

                return context

            except Exception as e:
                from upsonic.run.requirements import RunRequirement
                from upsonic.exceptions import RunCancelledException
                from upsonic.tools.processor import ExternalExecutionPause
                from upsonic.run.base import RunStatus
                
                # External tool pause - return normally with paused status
                if isinstance(e, ExternalExecutionPause):
                    await self._handle_external_tool_pause(context, e, self._execution_stats.get('executed_steps', 0))
                    return context
                
                # Cancel run - return normally with cancelled status (don't re-raise)
                if isinstance(e, RunCancelledException):
                    cancelled_step_result = context.get_cancelled_step()
                    
                    if cancelled_step_result is None and hasattr(self, '_current_step') and self._current_step is not None:
                        cancelled_step_result = StepResult(
                            name=self._current_step.name,
                            step_number=getattr(self, '_current_step_index', 0),
                            status=StepStatus.CANCELLED,
                            message=str(e)[:500],
                            execution_time=0.0
                        )
                        context.step_results.append(cancelled_step_result)
                    
                    if cancelled_step_result:
                        if self.agent:
                            context.tool_call_count = getattr(self.agent, '_tool_call_count', 0)
                            context.tool_limit_reached = getattr(self.agent, '_tool_limit_reached', False)
                        
                        requirement = RunRequirement(pause_type='cancel')
                        requirement.set_continuation_data(
                            messages=list(context.chat_history),
                            response=context.response,
                            agent_state={
                                'tool_call_count': context.tool_call_count,
                                'tool_limit_reached': context.tool_limit_reached,
                            },
                            step_result=cancelled_step_result,
                            execution_stats=context.execution_stats
                        )
                        context.add_requirement(requirement)
                        
                        if self.agent and hasattr(self.agent, '_save_checkpoint_to_storage'):
                            try:
                                await self.agent._save_checkpoint_to_storage(context)
                            except Exception:
                                pass
                    
                    if self.agent and self.agent._agent_run_output:
                        self.agent._agent_run_output.status = RunStatus.cancelled
                        self.agent._agent_run_context = context
                        self.agent._agent_run_output.sync_from_context(context)
                    
                    if self.debug:
                        from upsonic.utils.printing import warning_log
                        step_name = cancelled_step_result.name if cancelled_step_result else 'unknown'
                        warning_log(f"Run cancelled at step {step_name}", "PipelineManager")
                    
                    return context
                
                # Durable execution (error recovery) - save checkpoint and re-raise
                failed_step_result = context.get_error_step()
                
                if failed_step_result is None and hasattr(self, '_current_step') and self._current_step is not None:
                    failed_step_result = StepResult(
                        name=self._current_step.name,
                        step_number=getattr(self, '_current_step_index', 0),
                        status=StepStatus.ERROR,
                        message=str(e)[:500],
                        execution_time=0.0
                    )
                    context.step_results.append(failed_step_result)
                
                if failed_step_result:
                    if self.agent:
                        context.tool_call_count = getattr(self.agent, '_tool_call_count', 0)
                        context.tool_limit_reached = getattr(self.agent, '_tool_limit_reached', False)
                    
                    requirement = RunRequirement(pause_type='durable_execution')
                    requirement.set_continuation_data(
                        messages=list(context.chat_history),
                        response=context.response,
                        agent_state={
                            'tool_call_count': context.tool_call_count,
                            'tool_limit_reached': context.tool_limit_reached,
                        },
                        step_result=failed_step_result,
                        execution_stats=context.execution_stats
                    )
                    context.add_requirement(requirement)
                    
                    if self.agent and hasattr(self.agent, '_save_checkpoint_to_storage'):
                        try:
                            await self.agent._save_checkpoint_to_storage(context)
                        except Exception as save_error:
                            if self.debug:
                                from upsonic.utils.printing import warning_log
                                warning_log(f"Failed to save checkpoint: {save_error}", "PipelineManager")
                
                if self.agent and self.agent._agent_run_output:
                    self.agent._agent_run_output.status = RunStatus.error
                    self.agent._agent_run_context = context
                    self.agent._agent_run_output.sync_from_context(context)
                
                if self.debug:
                    from upsonic.utils.printing import warning_log, pipeline_failed, debug_log_level2
                    import traceback
                    step_name = failed_step_result.name if failed_step_result else 'unknown'
                    step_number = failed_step_result.step_number if failed_step_result else 0
                    warning_log(
                        f"❌ ERROR at step {step_number} ({step_name}): {str(e)[:100]}",
                        "PipelineManager"
                    )
                    pipeline_failed(
                        str(e),
                        self._execution_stats['executed_steps'],
                        self._execution_stats['total_steps'],
                        failed_step_result.message if failed_step_result else None,
                        failed_step_result.execution_time if failed_step_result else None
                    )
                    
                    debug_level = getattr(self.agent, 'debug_level', 1) if self.agent else 1
                    if debug_level >= 2:
                        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                        debug_log_level2(
                            "Pipeline execution error",
                            "PipelineManager",
                            debug=self.debug,
                            debug_level=debug_level,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            error_traceback=error_traceback[-2000:],
                            failed_step_index=failed_step_result.step_number if failed_step_result else 0,
                            failed_step_name=failed_step_result.name if failed_step_result else "Unknown",
                            executed_steps=self._execution_stats['executed_steps'],
                            total_steps=self._execution_stats['total_steps'],
                            step_results=self._execution_stats["step_results"],
                            task_description=self.task.description[:300] if self.task else None,
                            agent_name=getattr(self.agent, 'name', 'Unknown') if self.agent else None,
                            model_name=getattr(self.model, 'model_name', 'Unknown') if self.model else None
                        )
                
                transaction.set_tag("pipeline.status", "error")
                transaction.set_data("error.message", str(e))
                transaction.set_data("error.type", type(e).__name__)

                sentry_sdk.capture_exception(e)
                raise
    
    async def execute_stream(
        self, 
        context: "AgentRunContext",
        start_step_index: int = 0
    ) -> AsyncIterator["AgentEvent"]:
        """
        Execute the pipeline in streaming mode with comprehensive event emission.
        
        Runs all steps in sequence, yielding events for:
        - Pipeline start/end
        - Step start/end
        - Step-specific events (cache, policy, tools, etc.)
        - LLM streaming events (text deltas, tool calls)
        
        Events are collected in context.events and yielded after each step,
        as well as during streaming steps that emit events.
        
        Args:
            context: The initial context with is_streaming=True
            start_step_index: Index to start execution from (0-based). Used for HITL resumption.
            
        Yields:
            AgentEvent: Various event types providing visibility into execution
            
        Raises:
            Exception: Any exception from step execution is raised with proper error message
        """
        from upsonic.run.events.events import PipelineStartEvent, PipelineEndEvent, RunStartedEvent, RunCompletedEvent, StepStartEvent, StepEndEvent
        
        if not context.is_streaming:
            await self.execute(context, start_step_index)
            return
        
        if start_step_index > 0:
            if self.debug:
                from upsonic.utils.printing import info_log
                step_name = self.steps[start_step_index].name if start_step_index < len(self.steps) else "unknown"
                info_log(f"Resuming streaming pipeline from step {start_step_index} ({step_name})", "PipelineManager")
        
        from upsonic.run.pipeline.stats import PipelineExecutionStats
        if not context.execution_stats:
            context.execution_stats = PipelineExecutionStats(total_steps=len(self.steps))
        
        self._execution_stats = {
            "total_steps": len(self.steps),
            "executed_steps": 0,
            "step_results": {}
        }
        
        pipeline_start_time = time.time()
        
        task_description = None
        if self.task:
            task_description = str(getattr(self.task, 'description', ''))[:200]
        
        run_id = None
        if self.agent and hasattr(self.agent, 'run_id'):
            run_id = self.agent.run_id
        
        agent_id = None
        if self.agent and hasattr(self.agent, 'agent_id'):
            agent_id = self.agent.agent_id
        
        # Emit RunStartedEvent FIRST - before pipeline
        yield RunStartedEvent(
            run_id=run_id or "",
            agent_id=agent_id or "",
            task_description=task_description
        )
        
        # Emit pipeline start event
        yield PipelineStartEvent(
            run_id=run_id or "",
            total_steps=len(self.steps),
            is_streaming=True,
            task_description=task_description
        )
        
        if self.debug:
            from upsonic.utils.printing import pipeline_started
            pipeline_started(len(self.steps))
        
        error_message = None
        
        try:
            for step_index in range(start_step_index, len(self.steps)):
                step = self.steps[step_index]
                self._current_step_index = step_index
                self._current_step = step
                
                if self.debug:
                    from upsonic.utils.printing import pipeline_step_started
                    pipeline_step_started(step.name, step.description)
                
                step_start_time = time.time()
                
                # Clear events before step execution to track new events
                events_before = len(context.events)
                
                # Use run_stream() for streaming execution - yields events including step start/end
                if step.supports_streaming and context.is_streaming:
                    async for event in step.run_stream(context, self.task, self.agent, self.model, step_index):
                        yield event
                        # Also collect in context.events if not already there
                        if event not in context.events:
                            context.events.append(event)
                else:
                    # Non-streaming step - run() returns StepResult directly
                    # Emit StepStartEvent for non-streaming steps
                    yield StepStartEvent(
                        run_id=run_id or "",
                        step_name=step.name,
                        step_index=step_index,
                        total_steps=len(self.steps),
                        step_description=step.description
                    )
                    
                    result = await step.run(context, self.task, self.agent, self.model, step_index)
                    
                    # Yield any events that were added to context during execution
                    for event in context.events[events_before:]:
                        yield event
                    
                    # Emit StepEndEvent for non-streaming steps
                    step_result = context.step_results[-1] if context.step_results else None
                    yield StepEndEvent(
                        run_id=run_id or "",
                        step_name=step.name,
                        step_index=step_index,
                        status=step_result.status.value if step_result else "unknown",
                        execution_time=step_result.execution_time if step_result else time.time() - step_start_time,
                        message=step_result.message if step_result else None
                    )
                
                # Get result from context (last step result)
                result = context.step_results[-1] if context.step_results else None
                
                # Update execution stats
                if result:
                    context.execution_stats.executed_steps += 1
                    context.execution_stats.step_timing[step.name] = result.execution_time
                    context.execution_stats.step_statuses[step.name] = result.status.value
                    
                    self._execution_stats["step_results"][step.name] = {
                        "status": result.status.value,
                        "message": result.message,
                        "execution_time": result.execution_time
                    }
                    self._execution_stats["executed_steps"] += 1
                
                if self.debug and result:
                    from upsonic.utils.printing import pipeline_step_completed
                    pipeline_step_completed(
                        step.name, 
                        result.status.value, 
                        result.execution_time, 
                        result.message
                    )
                
                # Handle PAUSED status (simple break for streaming - no HITL in stream mode)
                if result and result.status == StepStatus.PAUSED:
                    from upsonic.run.events.events import RunPausedEvent
                    reason = "Execution paused"
                    yield RunPausedEvent(
                        run_id=run_id or "",
                        reason=reason,
                        requirements=None
                    )
                    
                    if self.debug:
                        from upsonic.utils.printing import pipeline_paused
                        pipeline_paused(step.name)
                    break
            
            total_time = time.time() - pipeline_start_time
            
            last_step_status = context.step_results[-1].status if context.step_results else StepStatus.COMPLETED
            
            # Note: RunCompletedEvent is already yielded by FinalizationStep/StreamFinalizationStep
            # so we don't yield it here to avoid duplication
            
            if self.debug:
                from upsonic.utils.printing import pipeline_completed, pipeline_timeline
                pipeline_completed(
                    context.execution_stats.executed_steps if context.execution_stats else 0,
                    len(self.steps),
                    total_time
                )
                step_results_dict = {sr.name: {"status": sr.status.value, "message": sr.message, "execution_time": sr.execution_time} for sr in context.step_results}
                pipeline_timeline(step_results_dict, total_time)
            
        except Exception as e:
            from upsonic.exceptions import RunCancelledException
            from upsonic.run.events.events import RunCancelledEvent
            
            if isinstance(e, RunCancelledException):
                step_name = self._current_step.name if hasattr(self, '_current_step') and self._current_step else None
                yield RunCancelledEvent(
                    run_id=run_id or "",
                    message=str(e),
                    step_name=step_name
                )
            
            error_message = str(e)
            
            if self.agent and self.agent._agent_run_output:
                from upsonic.run.base import RunStatus
                if isinstance(e, RunCancelledException):
                    self.agent._agent_run_output.status = RunStatus.cancelled
                else:
                    self.agent._agent_run_output.status = RunStatus.error
                
                self.agent._agent_run_context = context
                self.agent._agent_run_output.sync_from_context(context)
            
            if self.debug:
                from upsonic.utils.printing import pipeline_failed
                pipeline_failed(
                    str(e),
                    context.execution_stats.executed_steps if context.execution_stats else 0,
                    context.execution_stats.total_steps if context.execution_stats else len(self.steps),
                    None,
                    None
                )
            
            raise
        
        finally:
            total_time = time.time() - pipeline_start_time
            
            derived_status: str = 'success'
            if context.step_results:
                last_status = context.step_results[-1].status
                if last_status == StepStatus.ERROR:
                    derived_status = 'error'
                elif last_status == StepStatus.CANCELLED:
                    derived_status = 'cancelled'
                elif last_status == StepStatus.PAUSED:
                    derived_status = 'paused'
            
            # Emit pipeline end event
            yield PipelineEndEvent(
                run_id=run_id or "",
                status=derived_status,
                total_duration=total_time,
                total_steps=len(self.steps),
                executed_steps=self._execution_stats.get('executed_steps', 0),
                error_message=error_message
            )
            
            # Emit RunCompletedEvent LAST - after pipeline (only on success/completion)
            if derived_status in ('success', 'completed'):
                output_preview = str(context.final_output)[:100] if context.final_output else None
                yield RunCompletedEvent(
                    run_id=run_id or "",
                    agent_id=agent_id or "",
                    output_preview=output_preview
                )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about the last execution."""
        return self._execution_stats.copy()
    
    def find_step_index_by_name(self, step_name: str) -> int:
        """Find step index by step name."""
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                return i
        raise ValueError(f"Step '{step_name}' not found in pipeline")
    
    async def _handle_external_tool_pause(
        self, 
        context: "AgentRunContext", 
        e: "ExternalExecutionPause", 
        step_index: int
    ) -> None:
        """Handle ExternalExecutionPause exception (non-streaming)."""
        from upsonic.tools.processor import ExternalExecutionPause
        from upsonic.run.requirements import RunRequirement
        from upsonic.run.tools.tools import ToolExecution
        from upsonic.run.base import RunStatus
        
        if self.task:
            self.task.is_paused = True
        
        external_calls = e.external_calls or []
        
        if not external_calls:
            raise RuntimeError("ExternalExecutionPause must have external_calls attached by ToolManager")
        
        if self.agent:
            context.tool_call_count = getattr(self.agent, '_tool_call_count', 0)
            context.tool_limit_reached = getattr(self.agent, '_tool_limit_reached', False)
        
        for external_call in external_calls:
            tool_execution = ToolExecution(
                tool_call_id=external_call.tool_call_id,
                tool_name=external_call.tool_name,
                tool_args=external_call.tool_args,
                result=external_call.result,
                external_execution_required=True,
            )
            
            requirement = RunRequirement(
                tool_execution=tool_execution,
                pause_type='external_tool'
            )
            
            requirement.set_continuation_data(
                messages=list(context.chat_history),
                response=context.response,
                agent_state={
                    'tool_call_count': context.tool_call_count,
                    'tool_limit_reached': context.tool_limit_reached,
                },
                step_result=StepResult(
                    name=self.steps[step_index].name if step_index < len(self.steps) else "unknown",
                    step_number=step_index,
                    status=StepStatus.PAUSED,
                    message=f"Paused for external tool: {external_call.tool_name}",
                    execution_time=0.0,
                    extra_data={
                        'tool_call_id': external_call.tool_call_id,
                        'tool_name': external_call.tool_name,
                        'tool_args': external_call.tool_args,
                    }
                ),
                execution_stats=context.execution_stats
            )
            context.add_requirement(requirement)
        
        if self.task:
            context.final_output = self.task.response
        if self.agent and self.agent._agent_run_output:
            self.agent._agent_run_output.content = context.final_output
        
        paused_step_result = StepResult(
            name=self.steps[step_index].name if step_index < len(self.steps) else "model_execution",
            step_number=step_index,
            status=StepStatus.PAUSED,
            message=f"Execution paused for {len(external_calls)} external tool(s)",
            execution_time=0.0,
            extra_data={'external_tool_count': len(external_calls)}
        )
        context.step_results.append(paused_step_result)
        
        if self.agent and self.agent._agent_run_output:
            self.agent._agent_run_output.status = RunStatus.paused
            self.agent._agent_run_context = context
            self.agent._agent_run_output.sync_from_context(context)
        
        if self.agent and hasattr(self.agent, '_save_checkpoint_to_storage'):
            try:
                await self.agent._save_checkpoint_to_storage(context)
            except Exception:
                pass
        
        if self.debug:
            from upsonic.utils.printing import info_log
            info_log(f"External tool pause: {len(external_calls)} tool(s) waiting", "PipelineManager")
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        step_names = [step.name for step in self.steps]
        return f"PipelineManager(steps={step_names})"
