"""
Agent Event Yielding Utilities

This module provides utility functions for yielding events during agent execution.
Each function creates and yields a specific event type.
"""

from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional

from upsonic.run.events.events import (
    AgentInitializedEvent,
    CacheCheckEvent,
    CacheHitEvent,
    CacheMissEvent,
    CacheStoredEvent,
    CultureUpdateEvent,
    ExecutionCompleteEvent,
    ExternalToolPauseEvent,
    FinalOutputEvent,
    LLMPreparedEvent,
    MemoryUpdateEvent,
    MessagesBuiltEvent,
    ModelRequestStartEvent,
    ModelResponseEvent,
    ModelSelectedEvent,
    PipelineEndEvent,
    PipelineStartEvent,
    PolicyCheckEvent,
    PolicyFeedbackEvent,
    ReflectionEvent,
    ReliabilityEvent,
    RunCancelledEvent,
    RunCompletedEvent,
    RunPausedEvent,
    RunStartedEvent,
    StepEndEvent,
    StepStartEvent,
    StorageConnectionEvent,
    TextCompleteEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolsConfiguredEvent,
    ValidationEvent,
)


# Pipeline Events

async def ayield_pipeline_start_event(
    run_id: str,
    total_steps: int,
    task_description: Optional[str] = None,
    is_streaming: bool = True,
) -> AsyncIterator[PipelineStartEvent]:
    """Yield a PipelineStartEvent."""
    yield PipelineStartEvent(
        run_id=run_id,
        total_steps=total_steps,
        task_description=task_description,
        is_streaming=is_streaming,
    )


def yield_pipeline_start_event(
    run_id: str,
    total_steps: int,
    task_description: Optional[str] = None,
    is_streaming: bool = True,
) -> Iterator[PipelineStartEvent]:
    """Yield a PipelineStartEvent (sync)."""
    yield PipelineStartEvent(
        run_id=run_id,
        total_steps=total_steps,
        task_description=task_description,
        is_streaming=is_streaming,
    )


async def ayield_pipeline_end_event(
    run_id: str,
    status: str,
    total_duration: float,
    total_steps: int,
    message: Optional[str] = None,
    executed_steps: int = 0,
) -> AsyncIterator[PipelineEndEvent]:
    """Yield a PipelineEndEvent."""
    yield PipelineEndEvent(
        run_id=run_id,
        status=status,
        total_duration=total_duration,
        total_steps=total_steps,
        executed_steps=executed_steps,
        error_message=message,
    )


def yield_pipeline_end_event(
    run_id: str,
    status: str,
    total_duration: float,
    total_steps: int,
    message: Optional[str] = None,
    executed_steps: int = 0,
) -> Iterator[PipelineEndEvent]:
    """Yield a PipelineEndEvent (sync)."""
    yield PipelineEndEvent(
        run_id=run_id,
        status=status,
        total_duration=total_duration,
        total_steps=total_steps,
        executed_steps=executed_steps,
        error_message=message,
    )


# Step Events

async def ayield_step_start_event(
    run_id: str,
    step_name: str,
    step_description: str,
    step_index: int,
    total_steps: int,
) -> AsyncIterator[StepStartEvent]:
    """Yield a StepStartEvent."""
    yield StepStartEvent(
        run_id=run_id,
        step_name=step_name,
        step_description=step_description,
        step_index=step_index,
        total_steps=total_steps,
    )


def yield_step_start_event(
    run_id: str,
    step_name: str,
    step_description: str,
    step_index: int,
    total_steps: int,
) -> Iterator[StepStartEvent]:
    """Yield a StepStartEvent (sync)."""
    yield StepStartEvent(
        run_id=run_id,
        step_name=step_name,
        step_description=step_description,
        step_index=step_index,
        total_steps=total_steps,
    )


async def ayield_step_end_event(
    run_id: str,
    step_name: str,
    step_index: int,
    status: str,
    execution_time: float,
    message: Optional[str] = None,
) -> AsyncIterator[StepEndEvent]:
    """Yield a StepEndEvent."""
    yield StepEndEvent(
        run_id=run_id,
        step_name=step_name,
        step_index=step_index,
        status=status,
        execution_time=execution_time,
        message=message,
    )


def yield_step_end_event(
    run_id: str,
    step_name: str,
    status: str,
    execution_time: float,
    message: Optional[str] = None,
) -> Iterator[StepEndEvent]:
    """Yield a StepEndEvent (sync)."""
    yield StepEndEvent(
        run_id=run_id,
        step_name=step_name,
        status=status,
        execution_time=execution_time,
        message=message,
    )


# Text Events

async def ayield_text_delta_event(
    run_id: str,
    content: str,
) -> AsyncIterator[TextDeltaEvent]:
    """Yield a TextDeltaEvent."""
    yield TextDeltaEvent(
        run_id=run_id,
        content=content,
    )


def yield_text_delta_event(
    run_id: str,
    content: str,
) -> Iterator[TextDeltaEvent]:
    """Yield a TextDeltaEvent (sync)."""
    yield TextDeltaEvent(
        run_id=run_id,
        content=content,
    )


async def ayield_text_complete_event(
    run_id: str,
    content: str,
    part_index: int = 0,
) -> AsyncIterator[TextCompleteEvent]:
    """Yield a TextCompleteEvent."""
    yield TextCompleteEvent(
        run_id=run_id,
        content=content,
        part_index=part_index,
    )


def yield_text_complete_event(
    run_id: str,
    content: str,
    part_index: int = 0,
) -> Iterator[TextCompleteEvent]:
    """Yield a TextCompleteEvent (sync)."""
    yield TextCompleteEvent(
        run_id=run_id,
        content=content,
        part_index=part_index,
    )


# Tool Events

async def ayield_tool_call_event(
    run_id: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_call_id: Optional[str] = None,
    tool_index: int = 0,
) -> AsyncIterator[ToolCallEvent]:
    """Yield a ToolCallEvent."""
    yield ToolCallEvent(
        run_id=run_id,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_call_id=tool_call_id,
        tool_index=tool_index,
    )


def yield_tool_call_event(
    run_id: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_call_id: Optional[str] = None,
    tool_index: int = 0,
) -> Iterator[ToolCallEvent]:
    """Yield a ToolCallEvent (sync)."""
    yield ToolCallEvent(
        run_id=run_id,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_call_id=tool_call_id,
        tool_index=tool_index,
    )


async def ayield_tool_result_event(
    run_id: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: Any,
    is_error: bool = False,
    tool_call_id: Optional[str] = None,
    result_preview: Optional[str] = None,
    execution_time: Optional[float] = None,
) -> AsyncIterator[ToolResultEvent]:
    """Yield a ToolResultEvent."""
    yield ToolResultEvent(
        run_id=run_id,
        tool_name=tool_name,
        tool_args=tool_args,
        result=result,
        is_error=is_error,
        tool_call_id=tool_call_id,
        result_preview=result_preview,
        execution_time=execution_time,
    )


def yield_tool_result_event(
    run_id: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: Any,
    is_error: bool = False,
    tool_call_id: Optional[str] = None,
    result_preview: Optional[str] = None,
    execution_time: Optional[float] = None,
) -> Iterator[ToolResultEvent]:
    """Yield a ToolResultEvent (sync)."""
    yield ToolResultEvent(
        run_id=run_id,
        tool_name=tool_name,
        tool_args=tool_args,
        result=result,
        is_error=is_error,
        tool_call_id=tool_call_id,
        result_preview=result_preview,
        execution_time=execution_time,
    )


# Cache Events

async def ayield_cache_check_event(
    run_id: str,
    cache_enabled: bool,
    cache_method: Optional[str] = None,
) -> AsyncIterator[CacheCheckEvent]:
    """Yield a CacheCheckEvent."""
    yield CacheCheckEvent(
        run_id=run_id,
        cache_enabled=cache_enabled,
        cache_method=cache_method,
    )


def yield_cache_check_event(
    run_id: str,
    cache_enabled: bool,
    cache_method: Optional[str] = None,
) -> Iterator[CacheCheckEvent]:
    """Yield a CacheCheckEvent (sync)."""
    yield CacheCheckEvent(
        run_id=run_id,
        cache_enabled=cache_enabled,
        cache_method=cache_method,
    )


async def ayield_cache_hit_event(
    run_id: str,
    cache_method: str,
    similarity: Optional[float] = None,
    cached_response_preview: Optional[str] = None,
) -> AsyncIterator[CacheHitEvent]:
    """Yield a CacheHitEvent."""
    yield CacheHitEvent(
        run_id=run_id,
        cache_method=cache_method,
        similarity=similarity,
        cached_response_preview=cached_response_preview,
    )


def yield_cache_hit_event(
    run_id: str,
    cache_method: str,
    similarity: Optional[float] = None,
    cached_response_preview: Optional[str] = None,
) -> Iterator[CacheHitEvent]:
    """Yield a CacheHitEvent (sync)."""
    yield CacheHitEvent(
        run_id=run_id,
        cache_method=cache_method,
        similarity=similarity,
        cached_response_preview=cached_response_preview,
    )


async def ayield_cache_miss_event(
    run_id: str,
    cache_method: str,
) -> AsyncIterator[CacheMissEvent]:
    """Yield a CacheMissEvent."""
    yield CacheMissEvent(run_id=run_id, cache_method=cache_method)


def yield_cache_miss_event(
    run_id: str,
    cache_method: str,
) -> Iterator[CacheMissEvent]:
    """Yield a CacheMissEvent (sync)."""
    yield CacheMissEvent(run_id=run_id, cache_method=cache_method)

async def ayield_external_tool_pause_event(
    run_id: str,
    tool_name: str,
    tool_call_id: str,
    tool_args: Dict[str, Any],
) -> AsyncIterator[ExternalToolPauseEvent]:
    """Yield a ExternalToolPauseEvent."""
    yield ExternalToolPauseEvent(run_id=run_id, tool_name=tool_name, tool_call_id=tool_call_id, tool_args=tool_args)
    
def yield_external_tool_pause_event(
    run_id: str,
    tool_name: str,
    tool_call_id: str,
    tool_args: Dict[str, Any],
) -> Iterator[ExternalToolPauseEvent]:
    """Yield a ExternalToolPauseEvent (sync)."""
    yield ExternalToolPauseEvent(run_id=run_id, tool_name=tool_name, tool_call_id=tool_call_id, tool_args=tool_args)

async def ayield_cache_stored_event(
    run_id: str,
    cache_method: str,
    duration_minutes: Optional[int] = None,
) -> AsyncIterator[CacheStoredEvent]:
    """Yield a CacheStoredEvent."""
    yield CacheStoredEvent(
        run_id=run_id,
        cache_method=cache_method,
        duration_minutes=duration_minutes,
    )


def yield_cache_stored_event(
    run_id: str,
    cache_method: str,
    duration_minutes: Optional[int] = None,
) -> Iterator[CacheStoredEvent]:
    """Yield a CacheStoredEvent (sync)."""
    yield CacheStoredEvent(
        run_id=run_id,
        cache_method=cache_method,
        duration_minutes=duration_minutes,
    )


# Agent Events

async def ayield_agent_initialized_event(
    run_id: str,
    agent_id: str,
    is_streaming: bool = False,
) -> AsyncIterator[AgentInitializedEvent]:
    """Yield an AgentInitializedEvent."""
    yield AgentInitializedEvent(
        run_id=run_id,
        agent_id=agent_id,
        is_streaming=is_streaming,
    )


def yield_agent_initialized_event(
    run_id: str,
    agent_id: str,
    is_streaming: bool = False,
) -> Iterator[AgentInitializedEvent]:
    """Yield an AgentInitializedEvent (sync)."""
    yield AgentInitializedEvent(
        run_id=run_id,
        agent_id=agent_id,
        is_streaming=is_streaming,
    )


# Model Events

async def ayield_model_selected_event(
    run_id: str,
    model_name: str,
    model_provider: str,
) -> AsyncIterator[ModelSelectedEvent]:
    """Yield a ModelSelectedEvent."""
    yield ModelSelectedEvent(
        run_id=run_id,
        model_name=model_name,
        provider=model_provider,
    )


def yield_model_selected_event(
    run_id: str,
    model_name: str,
    model_provider: str,
) -> Iterator[ModelSelectedEvent]:
    """Yield a ModelSelectedEvent (sync)."""
    yield ModelSelectedEvent(
        run_id=run_id,
        model_name=model_name,
        provider=model_provider,
    )


async def ayield_model_request_start_event(
    run_id: str,
    model_name: str,
    is_streaming: bool = False,
    has_tools: bool = False,
    tool_call_count: int = 0,
    tool_call_limit: Optional[int] = None,
) -> AsyncIterator[ModelRequestStartEvent]:
    """Yield a ModelRequestStartEvent."""
    yield ModelRequestStartEvent(
        run_id=run_id,
        model_name=model_name,
        is_streaming=is_streaming,
        has_tools=has_tools,
        tool_call_count=tool_call_count,
        tool_call_limit=tool_call_limit,
    )


def yield_model_request_start_event(
    run_id: str,
    model_name: str,
    is_streaming: bool = False,
    has_tools: bool = False,
    tool_call_count: int = 0,
    tool_call_limit: Optional[int] = None,
) -> Iterator[ModelRequestStartEvent]:
    """Yield a ModelRequestStartEvent (sync)."""
    yield ModelRequestStartEvent(
        run_id=run_id,
        model_name=model_name,
        is_streaming=is_streaming,
        has_tools=has_tools,
        tool_call_count=tool_call_count,
        tool_call_limit=tool_call_limit,
    )


async def ayield_model_response_event(
    run_id: str,
    model_name: str,
    has_text: bool = False,
    has_tool_calls: bool = False,
    tool_call_count: int = 0,
    finish_reason: Optional[str] = None,
) -> AsyncIterator[ModelResponseEvent]:
    """Yield a ModelResponseEvent."""
    yield ModelResponseEvent(
        run_id=run_id,
        model_name=model_name,
        has_text=has_text,
        has_tool_calls=has_tool_calls,
        tool_call_count=tool_call_count,
        finish_reason=finish_reason,
    )


def yield_model_response_event(
    run_id: str,
    model_name: str,
    has_text: bool = False,
    has_tool_calls: bool = False,
    tool_call_count: int = 0,
    finish_reason: Optional[str] = None,
) -> Iterator[ModelResponseEvent]:
    """Yield a ModelResponseEvent (sync)."""
    yield ModelResponseEvent(
        run_id=run_id,
        model_name=model_name,
        has_text=has_text,
        has_tool_calls=has_tool_calls,
        tool_call_count=tool_call_count,
        finish_reason=finish_reason,
    )


async def ayield_tools_configured_event(
    run_id: str,
    tool_count: int,
    tool_names: Optional[List[str]] = None,
    has_mcp_handlers: bool = False,
) -> AsyncIterator[ToolsConfiguredEvent]:
    """Yield a ToolsConfiguredEvent."""
    yield ToolsConfiguredEvent(
        run_id=run_id,
        tool_count=tool_count,
        tool_names=tool_names or [],
        has_mcp_handlers=has_mcp_handlers,
    )


def yield_tools_configured_event(
    run_id: str,
    tool_count: int,
    tool_names: Optional[List[str]] = None,
    has_mcp_handlers: bool = False,
) -> Iterator[ToolsConfiguredEvent]:
    """Yield a ToolsConfiguredEvent (sync)."""
    yield ToolsConfiguredEvent(
        run_id=run_id,
        tool_count=tool_count,
        tool_names=tool_names or [],
        has_mcp_handlers=has_mcp_handlers,
    )


async def ayield_messages_built_event(
    run_id: str,
    message_count: int,
    has_system_prompt: bool = False,
    has_memory_messages: bool = False,
    is_continuation: bool = False,
) -> AsyncIterator[MessagesBuiltEvent]:
    """Yield a MessagesBuiltEvent."""
    yield MessagesBuiltEvent(
        run_id=run_id,
        message_count=message_count,
        has_system_prompt=has_system_prompt,
        has_memory_messages=has_memory_messages,
        is_continuation=is_continuation,
    )


def yield_messages_built_event(
    run_id: str,
    message_count: int,
    has_system_prompt: bool = False,
    has_memory_messages: bool = False,
    is_continuation: bool = False,
) -> Iterator[MessagesBuiltEvent]:
    """Yield a MessagesBuiltEvent (sync)."""
    yield MessagesBuiltEvent(
        run_id=run_id,
        message_count=message_count,
        has_system_prompt=has_system_prompt,
        has_memory_messages=has_memory_messages,
        is_continuation=is_continuation,
    )


# Run Events

async def ayield_run_started_event(
    run_id: str,
    agent_id: Optional[str] = None,
    task_description: Optional[str] = None,
) -> AsyncIterator[RunStartedEvent]:
    """Yield a RunStartedEvent."""
    yield RunStartedEvent(
        run_id=run_id,
        agent_id=agent_id,
        task_description=task_description,
    )


def yield_run_started_event(
    run_id: str,
    agent_id: Optional[str] = None,
    task_description: Optional[str] = None,
) -> Iterator[RunStartedEvent]:
    """Yield a RunStartedEvent (sync)."""
    yield RunStartedEvent(
        run_id=run_id,
        agent_id=agent_id,
        task_description=task_description,
    )


async def ayield_run_completed_event(
    run_id: str,
    agent_id: Optional[str] = None,
    output_preview: Optional[str] = None,
) -> AsyncIterator[RunCompletedEvent]:
    """Yield a RunCompletedEvent."""
    yield RunCompletedEvent(
        run_id=run_id,
        agent_id=agent_id,
        output_preview=output_preview,
    )


def yield_run_completed_event(
    run_id: str,
    agent_id: Optional[str] = None,
    output_preview: Optional[str] = None,
) -> Iterator[RunCompletedEvent]:
    """Yield a RunCompletedEvent (sync)."""
    yield RunCompletedEvent(
        run_id=run_id,
        agent_id=agent_id,
        output_preview=output_preview,
    )


async def ayield_run_cancelled_event(
    run_id: str,
    message: Optional[str] = None,
    step_name: Optional[str] = None,
) -> AsyncIterator[RunCancelledEvent]:
    """Yield a RunCancelledEvent."""
    yield RunCancelledEvent(
        run_id=run_id,
        message=message,
        step_name=step_name,
    )


def yield_run_cancelled_event(
    run_id: str,
    message: Optional[str] = None,
    step_name: Optional[str] = None,
) -> Iterator[RunCancelledEvent]:
    """Yield a RunCancelledEvent (sync)."""
    yield RunCancelledEvent(
        run_id=run_id,
        message=message,
        step_name=step_name,
    )


async def ayield_run_paused_event(
    run_id: str,
    reason: str,
    requirements: Optional[List[Any]] = None,
) -> AsyncIterator[RunPausedEvent]:
    """Yield a RunPausedEvent."""
    yield RunPausedEvent(
        run_id=run_id,
        reason=reason,
        requirements=requirements,
    )


def yield_run_paused_event(
    run_id: str,
    reason: str,
    requirements: Optional[List[Any]] = None,
) -> Iterator[RunPausedEvent]:
    """Yield a RunPausedEvent (sync)."""
    yield RunPausedEvent(
        run_id=run_id,
        reason=reason,
        requirements=requirements,
    )


# Memory Events

async def ayield_memory_update_event(
    run_id: str,
    memory_type: Optional[str] = None,
    messages_added: int = 0,
) -> AsyncIterator[MemoryUpdateEvent]:
    """Yield a MemoryUpdateEvent."""
    yield MemoryUpdateEvent(
        run_id=run_id,
        memory_type=memory_type,
        messages_added=messages_added,
    )


def yield_memory_update_event(
    run_id: str,
    memory_type: Optional[str] = None,
    messages_added: int = 0,
) -> Iterator[MemoryUpdateEvent]:
    """Yield a MemoryUpdateEvent (sync)."""
    yield MemoryUpdateEvent(
        run_id=run_id,
        memory_type=memory_type,
        messages_added=messages_added,
    )


# Storage Connection Events

async def ayield_storage_connection_event(
    run_id: str,
    storage_type: Optional[str] = None,
    is_connected: bool = False,
    has_memory: bool = False,
    session_id: Optional[str] = None,
) -> AsyncIterator[StorageConnectionEvent]:
    """Yield a StorageConnectionEvent."""
    yield StorageConnectionEvent(
        run_id=run_id,
        storage_type=storage_type,
        is_connected=is_connected,
        has_memory=has_memory,
        session_id=session_id,
    )


def yield_storage_connection_event(
    run_id: str,
    storage_type: Optional[str] = None,
    is_connected: bool = False,
    has_memory: bool = False,
    session_id: Optional[str] = None,
) -> Iterator[StorageConnectionEvent]:
    """Yield a StorageConnectionEvent (sync)."""
    yield StorageConnectionEvent(
        run_id=run_id,
        storage_type=storage_type,
        is_connected=is_connected,
        has_memory=has_memory,
        session_id=session_id,
    )


# LLM Prepared Events

async def ayield_llm_prepared_event(
    run_id: str,
    default_model: Optional[str] = None,
    requested_model: Optional[str] = None,
    model_changed: bool = False,
) -> AsyncIterator[LLMPreparedEvent]:
    """Yield a LLMPreparedEvent."""
    yield LLMPreparedEvent(
        run_id=run_id,
        default_model=default_model,
        requested_model=requested_model,
        model_changed=model_changed,
    )


def yield_llm_prepared_event(
    run_id: str,
    default_model: Optional[str] = None,
    requested_model: Optional[str] = None,
    model_changed: bool = False,
) -> Iterator[LLMPreparedEvent]:
    """Yield a LLMPreparedEvent (sync)."""
    yield LLMPreparedEvent(
        run_id=run_id,
        default_model=default_model,
        requested_model=requested_model,
        model_changed=model_changed,
    )


# Validation Events

async def ayield_validation_event(
    run_id: str,
    attachments_validated: bool = False,
    attachment_count: int = 0,
    validation_passed: bool = True,
    warnings: Optional[List[str]] = None,
) -> AsyncIterator[ValidationEvent]:
    """Yield a ValidationEvent."""
    yield ValidationEvent(
        run_id=run_id,
        attachments_validated=attachments_validated,
        attachment_count=attachment_count,
        validation_passed=validation_passed,
        warnings=warnings,
    )


def yield_validation_event(
    run_id: str,
    attachments_validated: bool = False,
    attachment_count: int = 0,
    validation_passed: bool = True,
    warnings: Optional[List[str]] = None,
) -> Iterator[ValidationEvent]:
    """Yield a ValidationEvent (sync)."""
    yield ValidationEvent(
        run_id=run_id,
        attachments_validated=attachments_validated,
        attachment_count=attachment_count,
        validation_passed=validation_passed,
        warnings=warnings,
    )


# Policy Events

async def ayield_policy_check_event(
    run_id: str,
    policy_type: Literal['user_policy', 'agent_policy'],
    action: Literal['ALLOW', 'BLOCK', 'REPLACE', 'ANONYMIZE', 'RAISE ERROR'],
    policies_checked: int = 0,
    content_modified: bool = False,
    blocked_reason: Optional[str] = None,
) -> AsyncIterator[PolicyCheckEvent]:
    """Yield a PolicyCheckEvent."""
    yield PolicyCheckEvent(
        run_id=run_id,
        policy_type=policy_type,
        action=action,
        policies_checked=policies_checked,
        content_modified=content_modified,
        blocked_reason=blocked_reason,
    )


def yield_policy_check_event(
    run_id: str,
    policy_type: Literal['user_policy', 'agent_policy'],
    action: Literal['ALLOW', 'BLOCK', 'REPLACE', 'ANONYMIZE', 'RAISE ERROR'],
    policies_checked: int = 0,
    content_modified: bool = False,
    blocked_reason: Optional[str] = None,
) -> Iterator[PolicyCheckEvent]:
    """Yield a PolicyCheckEvent (sync)."""
    yield PolicyCheckEvent(
        run_id=run_id,
        policy_type=policy_type,
        action=action,
        policies_checked=policies_checked,
        content_modified=content_modified,
        blocked_reason=blocked_reason,
    )


async def ayield_policy_feedback_event(
    run_id: str,
    policy_type: Literal['user_policy', 'agent_policy'],
    feedback_message: str,
    retry_count: int = 0,
    max_retries: int = 1,
    violated_policy: Optional[str] = None,
) -> AsyncIterator[PolicyFeedbackEvent]:
    """Yield a PolicyFeedbackEvent."""
    yield PolicyFeedbackEvent(
        run_id=run_id,
        policy_type=policy_type,
        feedback_message=feedback_message,
        retry_count=retry_count,
        max_retries=max_retries,
        violated_policy=violated_policy,
    )


def yield_policy_feedback_event(
    run_id: str,
    policy_type: Literal['user_policy', 'agent_policy'],
    feedback_message: str,
    retry_count: int = 0,
    max_retries: int = 1,
    violated_policy: Optional[str] = None,
) -> Iterator[PolicyFeedbackEvent]:
    """Yield a PolicyFeedbackEvent (sync)."""
    yield PolicyFeedbackEvent(
        run_id=run_id,
        policy_type=policy_type,
        feedback_message=feedback_message,
        retry_count=retry_count,
        max_retries=max_retries,
        violated_policy=violated_policy,
    )

async def ayield_reflection_event(
    run_id: str,
    reflection_applied: bool = False,
    improvement_made: bool = False,
    original_preview: Optional[str] = None,
    improved_preview: Optional[str] = None,
) -> AsyncIterator[ReflectionEvent]:
    """Yield a ReflectionEvent."""
    yield ReflectionEvent(
        run_id=run_id,
        reflection_applied=reflection_applied,
        improvement_made=improvement_made,
        original_preview=original_preview,
        improved_preview=improved_preview,
    )


def yield_reflection_event(
    run_id: str,
    reflection_applied: bool = False,
    improvement_made: bool = False,
    original_preview: Optional[str] = None,
    improved_preview: Optional[str] = None,
) -> Iterator[ReflectionEvent]:
    """Yield a ReflectionEvent (sync)."""
    yield ReflectionEvent(
        run_id=run_id,
        reflection_applied=reflection_applied,
        improvement_made=improvement_made,
        original_preview=original_preview,
        improved_preview=improved_preview,
    )


async def ayield_reliability_event(
    run_id: str,
    reliability_applied: bool = False,
    modifications_made: bool = False,
) -> AsyncIterator[ReliabilityEvent]:
    """Yield a ReliabilityEvent."""
    yield ReliabilityEvent(
        run_id=run_id,
        reliability_applied=reliability_applied,
        modifications_made=modifications_made,
    )


def yield_reliability_event(
    run_id: str,
    reliability_applied: bool = False,
    modifications_made: bool = False,
) -> Iterator[ReliabilityEvent]:
    """Yield a ReliabilityEvent (sync)."""
    yield ReliabilityEvent(
        run_id=run_id,
        reliability_applied=reliability_applied,
        modifications_made=modifications_made,
    )


# Execution Events

async def ayield_execution_complete_event(
    run_id: str,
    output_type: Literal['text', 'structured', 'cached', 'blocked'],
    has_output: bool,
    output_preview: Optional[str],
    total_tool_calls: int,
    total_duration: Optional[float],
) -> AsyncIterator[ExecutionCompleteEvent]:
    """Yield an ExecutionCompleteEvent."""
    yield ExecutionCompleteEvent(
        run_id=run_id,
        output_type=output_type,
        has_output=has_output,
        output_preview=output_preview,
        total_tool_calls=total_tool_calls,
        total_duration=total_duration,
    )


def yield_execution_complete_event(
    run_id: str,
    output_type: Literal['text', 'structured', 'cached', 'blocked'],
    has_output: bool,
    output_preview: Optional[str],
    total_tool_calls: int,
    total_duration: Optional[float],
) -> Iterator[ExecutionCompleteEvent]:
    """Yield an ExecutionCompleteEvent (sync)."""
    yield ExecutionCompleteEvent(
        run_id=run_id,
        output_type=output_type,
        has_output=has_output,
        output_preview=output_preview,
        total_tool_calls=total_tool_calls,
        total_duration=total_duration,
    )


async def ayield_final_output_event(
    run_id: str,
    output: Any,
    output_type: Literal['text', 'structured', 'cached', 'blocked'] = 'text',
) -> AsyncIterator[FinalOutputEvent]:
    """Yield a FinalOutputEvent."""
    yield FinalOutputEvent(
        run_id=run_id,
        output=output,
        output_type=output_type,
    )


def yield_final_output_event(
    run_id: str,
    output: Any,
    output_type: Literal['text', 'structured', 'cached', 'blocked'] = 'text',
) -> Iterator[FinalOutputEvent]:
    """Yield a FinalOutputEvent (sync)."""
    yield FinalOutputEvent(
        run_id=run_id,
        output=output,
        output_type=output_type,
    )


__all__ = [
    # Pipeline events
    "ayield_pipeline_start_event",
    "yield_pipeline_start_event",
    "ayield_pipeline_end_event",
    "yield_pipeline_end_event",
    # Step events
    "ayield_step_start_event",
    "yield_step_start_event",
    "ayield_step_end_event",
    "yield_step_end_event",
    # Agent events (currently unused - kept for future use)
    # "ayield_agent_initialized_event",
    # "yield_agent_initialized_event",
    # Text events
    "ayield_text_delta_event",
    "yield_text_delta_event",
    "ayield_text_complete_event",
    "yield_text_complete_event",
    # Tool events
    "ayield_tool_call_event",
    "yield_tool_call_event",
    "ayield_tool_result_event",
    "yield_tool_result_event",
    "ayield_external_tool_pause_event",
    "yield_external_tool_pause_event",
    # Cache events
    "ayield_cache_check_event",
    "yield_cache_check_event",
    "ayield_cache_hit_event",
    "yield_cache_hit_event",
    "ayield_cache_miss_event",
    "yield_cache_miss_event",
    "ayield_cache_stored_event",
    "yield_cache_stored_event",
    # Model events
    "ayield_model_selected_event",
    "yield_model_selected_event",
    "ayield_model_request_start_event",
    "yield_model_request_start_event",
    "ayield_model_response_event",
    "yield_model_response_event",
    # Tools and Messages events
    "ayield_tools_configured_event",
    "yield_tools_configured_event",
    "ayield_messages_built_event",
    "yield_messages_built_event",
    # Run events
    "ayield_run_started_event",
    "yield_run_started_event",
    # Run events (currently unused - kept for future use)
    # "ayield_run_completed_event",
    # "yield_run_completed_event",
    # "ayield_run_cancelled_event",
    # "yield_run_cancelled_event",
    # "ayield_run_paused_event",
    # "yield_run_paused_event",
    # Memory events
    "ayield_memory_update_event",
    "yield_memory_update_event",
    # Storage Connection events
    "ayield_storage_connection_event",
    "yield_storage_connection_event",
    # LLM Prepared events
    "ayield_llm_prepared_event",
    "yield_llm_prepared_event",
    # Validation events
    "ayield_validation_event",
    "yield_validation_event",
    # Policy events
    "ayield_policy_check_event",
    "yield_policy_check_event",
    "ayield_policy_feedback_event",
    "yield_policy_feedback_event",
    # Reflection and Reliability events
    "ayield_reflection_event",
    "yield_reflection_event",
    "ayield_reliability_event",
    "yield_reliability_event",
    # Execution events
    "ayield_execution_complete_event",
    "yield_execution_complete_event",
    "ayield_final_output_event",
    "yield_final_output_event",
]


