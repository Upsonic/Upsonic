"""
Pipeline Manager - Orchestrates step execution.

The PipelineManager is responsible for executing a sequence of steps
in order, managing the context flow, and handling the overall execution.
It also supports comprehensive event streaming for full visibility into
the execution pipeline.
"""

import time
from typing import List, Optional, Dict, Any, AsyncIterator, Union
from upsonic.agent.pipeline.step import Step
from .step import Step, StepResult, StepStatus
from .context import StepContext

# Sentry tracing for pipeline execution
from upsonic.utils.logging_config import sentry_sdk, get_logger

# Sentry event logger (debug flag'inden bağımsız)
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
        debug: bool = False
    ):
        """
        Initialize the pipeline manager.
        
        Args:
            steps: List of steps to execute
            debug: Enable debug logging
        """
        self.steps: List[Step] = steps or []
        self.debug = debug
        self._execution_stats: Dict[str, Any] = {}
    
    def add_step(self, step: Step) -> None:
        """
        Add a step to the pipeline.
        
        Args:
            step: Step to add
        """
        self.steps.append(step)
    
    def insert_step(self, index: int, step: Step) -> None:
        """
        Insert a step at a specific position.
        
        Args:
            index: Position to insert at
            step: Step to insert
        """
        self.steps.insert(index, step)
    
    def remove_step(self, step_name: str) -> bool:
        """
        Remove a step by name.
        
        Args:
            step_name: Name of the step to remove
            
        Returns:
            bool: True if step was removed, False if not found
        """
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                self.steps.pop(i)
                return True
        return False
    
    def get_step(self, step_name: str) -> Optional[Step]:
        """
        Get a step by name.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Optional[Step]: The step if found, None otherwise
        """
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    async def execute(self, context: StepContext) -> StepContext:
        """
        Execute the pipeline.

        Runs all steps in sequence, passing the context through each one.
        All steps must execute. If any step raises an error, the pipeline stops,
        logs the error properly, and raises it to the caller.

        Args:
            context: The initial context

        Returns:
            StepContext: The final context after all steps

        Raises:
            Exception: Any exception from step execution is raised with proper error message
        """
        self._execution_stats = {
            "total_steps": len(self.steps),
            "executed_steps": 0,
            "step_results": {}
        }

        # Start Sentry transaction for entire pipeline
        # Using 'with' ensures transaction is set in scope and spans are attached
        with sentry_sdk.start_transaction(
            op="agent.pipeline.execute",
            name=f"Agent Pipeline ({len(self.steps)} steps)"
        ) as transaction:
            # Add pipeline metadata
            transaction.set_tag("pipeline.total_steps", len(self.steps))
            transaction.set_tag("pipeline.debug", self.debug)
            transaction.set_tag("pipeline.streaming", context.is_streaming if hasattr(context, 'is_streaming') else False)

            # Add task metadata if available
            if hasattr(context, 'task') and context.task:
                transaction.set_tag("task.type", type(context.task).__name__)
                if hasattr(context.task, 'description'):
                    transaction.set_data("task.description", str(context.task.description)[:200])

            # Sentry logging (her zaman gönder, debug flag'inden bağımsız)
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
                for step_index, step in enumerate[Step](self.steps):
                    # Track current step for error handling  
                    self._current_step_index = step_index
                    self._current_step = step
                    
                    # Start span for each step
                    with sentry_sdk.start_span(
                        op=f"pipeline.step.{step.name}",
                        name=step.description
                    ) as span:
                        # Add step metadata
                        span.set_tag("step.name", step.name)
                        span.set_data("step.description", step.description)

                        if self.debug:
                            from upsonic.utils.printing import pipeline_step_started
                            pipeline_step_started(step.name, step.description)

                        # This will raise an exception if the step fails
                        result = await step.run(context)

                        # Add result to span
                        span.set_tag("step.status", result.status.value)
                        span.set_data("step.message", result.message)
                        span.set_data("step.execution_time", result.execution_time)

                        # Record statistics
                        self._execution_stats["step_results"][step.name] = {
                            "status": result.status.value,
                            "message": result.message,
                            "execution_time": result.execution_time
                        }

                        self._execution_stats["executed_steps"] += 1

                        if self.debug:
                            from upsonic.utils.printing import pipeline_step_completed
                            pipeline_step_completed(
                                step.name,
                                result.status.value,
                                result.execution_time,
                                result.message
                            )
                        
                        # Save durable execution checkpoint after successful step
                        if result.status == StepStatus.SUCCESS:
                            await self._save_durable_checkpoint(context, step, self._execution_stats["executed_steps"] - 1, status="success")

                        # Handle PENDING status (like external execution pause)
                        if result.status == StepStatus.PENDING:
                            # Save checkpoint for pending state too
                            await self._save_durable_checkpoint(context, step, self._execution_stats["executed_steps"] - 1, status="paused")
                            
                            if self.debug:
                                from upsonic.utils.printing import pipeline_paused
                                pipeline_paused(step.name)
                            # Continue to finalization but mark as pending
                            break

                # Pipeline completed successfully
                total_time = sum(r["execution_time"] for r in self._execution_stats["step_results"].values())
                
                # Mark durable execution as completed
                await self._mark_durable_completed(context)

                # Add final metrics to transaction
                transaction.set_tag("pipeline.status", "success")
                transaction.set_data("pipeline.executed_steps", self._execution_stats['executed_steps'])
                transaction.set_data("pipeline.total_time", total_time)

                # Sentry logging (her zaman gönder, debug flag'inden bağımsız)
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

                    # Show timeline of step execution times
                    pipeline_timeline(
                        self._execution_stats["step_results"],
                        total_time
                    )

                # Transaction will be automatically finished with "ok" status by context manager
                return context

            except Exception as e:
                # Save checkpoint at the FAILED step (not the previous successful one)
                # This way the checkpoint accurately reflects which step failed
                failed_step_index = getattr(self, '_current_step_index', max(0, self._execution_stats["executed_steps"] - 1))
                failed_step = getattr(self, '_current_step', None)
                if failed_step is None and failed_step_index < len(self.steps):
                    failed_step = self.steps[failed_step_index]
                
                # Debug: Show which step failed and where we're saving checkpoint
                if self.debug:
                    from upsonic.utils.printing import warning_log
                    warning_log(
                        f"❌ ERROR at step {failed_step_index} ({failed_step.name if failed_step else 'unknown'}): {str(e)[:100]}",
                        "PipelineManager"
                    )
                    if context.task.durable_checkpoint_enabled:
                        warning_log(
                            f"💾 Saving checkpoint at step {failed_step_index} ({failed_step.name if failed_step else 'N/A'}) with status='failed'",
                            "PipelineManager"
                        )
                
                # Save checkpoint with failed status at the failed step
                await self._save_durable_checkpoint(
                    context,
                    failed_step,
                    failed_step_index,
                    status="failed"
                )
                
                # Mark durable execution as failed
                await self._mark_durable_failed(context, str(e))
                
                # Mark transaction as failed
                transaction.set_tag("pipeline.status", "error")
                transaction.set_data("error.message", str(e))
                transaction.set_data("error.type", type(e).__name__)

                # Get error result if it was stored
                error_result = getattr(context, '_error_result', None)

                if error_result:
                    # Record the error in statistics
                    self._execution_stats["step_results"]["error"] = {
                        "status": error_result.status.value,
                        "message": error_result.message,
                        "execution_time": error_result.execution_time
                    }

                if self.debug:
                    from upsonic.utils.printing import pipeline_failed
                    pipeline_failed(
                        str(e),
                        self._execution_stats['executed_steps'],
                        self._execution_stats['total_steps'],
                        error_result.message if error_result else None,
                        error_result.execution_time if error_result else None
                    )

                # Capture exception in Sentry
                sentry_sdk.capture_exception(e)

                # Transaction will be automatically finished with error status by context manager
                # Re-raise the original exception
                raise
    
    async def execute_stream(self, context: StepContext) -> AsyncIterator[Any]:
        """
        Execute the pipeline in streaming mode with comprehensive event emission.
        
        Runs all steps in sequence, yielding events for:
        - Pipeline start/end
        - Step start/end
        - Step-specific events (cache, policy, tools, etc.)
        - LLM streaming events (text deltas, tool calls)
        
        This provides complete visibility into the execution pipeline.
        
        Args:
            context: The initial context with is_streaming=True
            
        Yields:
            AgentEvent: Various event types providing visibility into execution
            
        Returns:
            StepContext: The final context after all steps (implicitly through context mutation)
            
        Raises:
            Exception: Any exception from step execution is raised with proper error message
        """
        # Safety check: Only yield events if streaming is explicitly enabled
        if not context.is_streaming:
            # If streaming is not enabled, fall back to regular execution
            # This should never happen if called correctly, but provides safety
            await self.execute(context)
            return
            
        from upsonic.agent.events import (
            PipelineStartEvent,
            PipelineEndEvent,
            StepStartEvent,
            StepEndEvent,
            convert_llm_event_to_agent_event,
        )
        
        self._execution_stats = {
            "total_steps": len(self.steps),
            "executed_steps": 0,
            "step_results": {}
        }
        
        pipeline_start_time = time.time()
        
        # Get task description for event
        task_description = None
        if hasattr(context, 'task') and context.task:
            task_description = str(getattr(context.task, 'description', ''))[:200]
        
        # Emit pipeline start event
        yield PipelineStartEvent(
            total_steps=len(self.steps),
            is_streaming=True,
            task_description=task_description
        )
        
        if self.debug:
            from upsonic.utils.printing import pipeline_started
            pipeline_started(len(self.steps))
        
        final_status = 'success'
        error_message = None
        
        try:
            for step_index, step in enumerate(self.steps):
                # Track current step for error handling
                self._current_step_index = step_index
                self._current_step = step
                
                # Emit step start event
                yield StepStartEvent(
                    step_name=step.name,
                    step_description=step.description,
                    step_index=step_index,
                    total_steps=len(self.steps)
                )
                
                if self.debug:
                    from upsonic.utils.printing import pipeline_step_started
                    pipeline_step_started(step.name, step.description)
                
                step_start_time = time.time()
                result = None
                
                # Check if this step supports streaming (has custom execute_stream implementation)
                if step.supports_streaming and context.is_streaming:
                    # Execute streaming step and yield only Agent events
                    async for event in step.execute_stream(context):
                        # Only yield AgentEvent instances - LLM events are converted inside steps
                        from upsonic.agent.events import AgentEvent, TextDeltaEvent
                        
                        if isinstance(event, AgentEvent):
                            # Update accumulated text for text events
                            if isinstance(event, TextDeltaEvent):
                                context.accumulated_text += event.content
                            yield event
                        # Skip raw LLM events - they should be converted to Agent events in the step
                    
                    # Get the result after streaming completes
                    result = getattr(step, '_last_result', StepResult(
                        status=StepStatus.SUCCESS,
                        message=f"Streaming step {step.name} completed",
                        execution_time=time.time() - step_start_time
                    ))
                else:
                    # Regular step execution
                    result = await step.run(context)
                    
                    # Yield any step-specific events from the result
                    if result.events:
                        for event in result.events:
                            yield event
                    
                    # Yield any events emitted to context during execution
                    pending_events = context.pop_pending_events()
                    for event in pending_events:
                        yield event
                
                # Record statistics
                self._execution_stats["step_results"][step.name] = {
                    "status": result.status.value if result else "success",
                    "message": result.message if result else None,
                    "execution_time": result.execution_time if result else time.time() - step_start_time
                }
                
                self._execution_stats["executed_steps"] += 1
                
                # Emit step end event
                yield StepEndEvent(
                    step_name=step.name,
                    step_index=step_index,
                    status=result.status.value if result else 'success',
                    message=result.message if result else f"{step.name} completed",
                    execution_time=result.execution_time if result else time.time() - step_start_time
                )
                
                if self.debug:
                    from upsonic.utils.printing import pipeline_step_completed
                    pipeline_step_completed(
                        step.name, 
                        result.status.value if result else 'success', 
                        result.execution_time if result else 0.0, 
                        result.message if result else None
                    )
                
                # Handle PENDING status (like external execution pause)
                if result and result.status == StepStatus.PENDING:
                    final_status = 'paused'
                    if self.debug:
                        from upsonic.utils.printing import pipeline_paused
                        pipeline_paused(step.name)
                    break
            
            total_time = time.time() - pipeline_start_time
            
            if self.debug:
                from upsonic.utils.printing import pipeline_completed, pipeline_timeline
                pipeline_completed(
                    self._execution_stats['executed_steps'],
                    len(self.steps),
                    total_time
                )

                # Show timeline of step execution times
                pipeline_timeline(
                    self._execution_stats["step_results"],
                    total_time
                )
            
        except Exception as e:
            final_status = 'error'
            error_message = str(e)
            
            # Get error result if it was stored
            error_result = getattr(context, '_error_result', None)
            
            if error_result:
                # Record the error in statistics
                self._execution_stats["step_results"]["error"] = {
                    "status": error_result.status.value,
                    "message": error_result.message,
                    "execution_time": error_result.execution_time
                }
            
            if self.debug:
                from upsonic.utils.printing import pipeline_failed
                pipeline_failed(
                    str(e),
                    self._execution_stats['executed_steps'],
                    self._execution_stats['total_steps'],
                    error_result.message if error_result else None,
                    error_result.execution_time if error_result else None
                )
            
            # Re-raise the original exception
            raise
        
        finally:
            total_time = time.time() - pipeline_start_time
            
            # Emit pipeline end event
            yield PipelineEndEvent(
                total_steps=len(self.steps),
                executed_steps=self._execution_stats['executed_steps'],
                total_duration=total_time,
                status=final_status,
                error_message=error_message
            )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the last execution.
        
        Returns:
            Dict containing execution statistics
        """
        return self._execution_stats.copy()
    
    async def _save_durable_checkpoint(
        self, 
        context: StepContext, 
        step: Step, 
        step_index: int,
        status: str = "running"
    ) -> None:
        """
        Save a durable execution checkpoint after a step completes.
        
        Args:
            context: The current step context
            step: The step that just completed
            step_index: Index of the completed step
            status: Execution status ('running', 'paused', etc.)
        """
        # Check if task has durable execution enabled
        if not hasattr(context, 'task') or not context.task:
            return
        
        if not hasattr(context.task, 'durable_execution') or not context.task.durable_execution:
            return
        
        if not context.task.durable_checkpoint_enabled:
            return
        
        try:
            agent_state = {}
            if hasattr(context, 'agent') and context.agent:
                agent_state = {
                    'tool_call_count': getattr(context.agent, '_tool_call_count', 0),
                    'tool_limit_reached': getattr(context.agent, '_tool_limit_reached', False),
                }
            
            await context.task.durable_execution.save_checkpoint_async(
                task=context.task,
                context=context,
                step_index=step_index,
                step_name=step.name,
                status=status,
                agent_state=agent_state
            )
        except Exception as e:
            if self.debug:
                from upsonic.utils.printing import warning_log
                warning_log(
                    f"Failed to save durable checkpoint: {str(e)}",
                    "PipelineManager"
                )
    
    async def _mark_durable_completed(self, context: StepContext) -> None:
        """Mark durable execution as completed."""
        if not hasattr(context, 'task') or not context.task:
            return
        
        if not hasattr(context.task, 'durable_execution') or not context.task.durable_execution:
            return
        
        try:
            await context.task.durable_execution.mark_completed_async()
        except Exception as e:
            if self.debug:
                from upsonic.utils.printing import warning_log
                warning_log(
                    f"Failed to mark durable execution as completed: {str(e)}",
                    "PipelineManager"
                )
    
    async def _mark_durable_failed(self, context: StepContext, error: str) -> None:
        """Mark durable execution as failed."""
        if not hasattr(context, 'task') or not context.task:
            return
        
        if not hasattr(context.task, 'durable_execution') or not context.task.durable_execution:
            return
        
        try:
            await context.task.durable_execution.mark_failed_async(error)
        except Exception as e:
            # Log error but don't fail
            if self.debug:
                from upsonic.utils.printing import warning_log
                warning_log(
                    f"Failed to mark durable execution as failed: {str(e)}",
                    "PipelineManager"
                )
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        step_names = [step.name for step in self.steps]
        return f"PipelineManager(steps={step_names})"
