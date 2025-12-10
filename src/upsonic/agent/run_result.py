"""
Run Result Module

This module provides the RunResult class that wraps agent execution results
with comprehensive message tracking and output management capabilities.
"""

import asyncio
import time
import threading
import uuid
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import List, Generic, AsyncIterator, Iterator, Optional, Any, TYPE_CHECKING, Dict, Union
from contextlib import asynccontextmanager

from upsonic.messages.messages import ModelMessage, ModelResponseStreamEvent, TextPart, PartStartEvent, PartDeltaEvent, FinalResultEvent
from upsonic.output import OutputDataT
from upsonic.agent.run_input import RunInput
from upsonic.agent.run_events import (
    RunStatus,
    BaseAgentRunEvent,
    AgentRunEvent,
    RunStartedEvent,
    RunContentEvent,
    RunContentCompletedEvent,
    RunCompletedEvent,
    RunErrorEvent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
)

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task
    from upsonic.messages.messages import ModelResponse
    from upsonic.usage import RequestUsage


@dataclass
class RunResult(Generic[OutputDataT]):
    """
    A comprehensive result wrapper for agent executions.
    
    This class encapsulates:
    - The actual output/response from the agent
    - All messages exchanged during all runs
    - Run metadata (ID, agent info, status)
    - Input capture for reproducibility
    - Methods to access message history
    
    Attributes:
        output: The actual output from the agent execution
        run_id: Unique identifier for this run
        agent_id: Unique identifier for the agent
        agent_name: Display name of the agent
        input: The RunInput that initiated this run
        status: Current status of the run
        _all_messages: Internal storage for all messages across all runs
        _run_boundaries: Indices marking the start of each run
        _events: Agent run events captured during execution
        
    Example:
        ```python
        result = agent.do(task)
        print(result.output)  # Access the actual response
        print(result.all_messages())  # Get all messages
        print(result.new_messages())  # Get last run's messages
        print(result.run_id)  # Get the unique run ID
        print(result.status)  # Check run status
        ```
    """
    
    output: OutputDataT
    """The actual output/response from the agent execution."""
    
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this run."""
    
    agent_id: str = ""
    """Unique identifier for the agent."""
    
    agent_name: str = ""
    """Display name of the agent."""
    
    input: Optional[RunInput] = None
    """The RunInput that initiated this run."""
    
    status: RunStatus = RunStatus.completed
    """Current status of the run."""
    
    _all_messages: List[ModelMessage] = field(default_factory=list)
    """Internal storage for all messages across all runs."""
    
    _run_boundaries: List[int] = field(default_factory=list)
    """Indices marking where each run starts in the message list."""
    
    _events: List[BaseAgentRunEvent] = field(default_factory=list)
    """Agent run events captured during execution."""
    
    def all_messages(self) -> List[ModelMessage]:
        """
        Get all messages from all runs of the agent.
        
        Returns:
            List of all ModelMessage objects (ModelRequest and ModelResponse) 
            from all agent runs.
        """
        return self._all_messages.copy()
    
    def new_messages(self) -> List[ModelMessage]:
        """
        Get messages from the last run only.
        
        A single run may include multiple message exchanges due to tool calls,
        resulting in sequences like:
        - ModelRequest (user prompt)
        - ModelResponse (with tool calls)
        - ModelRequest (tool results)
        - ModelResponse (final answer)
        
        Returns:
            List of all ModelMessage objects from the most recent run, including
            all intermediate tool call exchanges.
        """
        if not self._run_boundaries:
            # No run boundaries marked, return all messages (single run)
            return self._all_messages.copy()
        
        # Get messages from the start of the last run to the end
        last_run_start_idx = self._run_boundaries[-1]
        return self._all_messages[last_run_start_idx:].copy()
    
    def get_last_model_response(self) -> Optional['ModelResponse']:
        """
        Get the last ModelResponse from the messages.
        
        This method searches through the messages from the last run and returns
        the most recent ModelResponse, if any exists.
        
        Returns:
            The last ModelResponse from the messages, or None if no ModelResponse exists.
        """
        from upsonic.messages.messages import ModelResponse
        
        messages = self.new_messages()
        for msg in reversed(messages):
            if isinstance(msg, ModelResponse):
                return msg
        return None
    
    def add_messages(self, messages: List[ModelMessage]) -> None:
        """
        Add messages to the internal message store.
        
        Args:
            messages: List of ModelMessage objects to add.
        """
        self._all_messages.extend(messages)
    
    def add_message(self, message: ModelMessage) -> None:
        """
        Add a single message to the internal message store.
        
        Args:
            message: A ModelMessage object to add.
        """
        self._all_messages.append(message)
    
    def start_new_run(self) -> None:
        """
        Mark the start of a new run in the message history.
        
        This should be called before adding messages from a new agent run
        to properly track run boundaries for the new_messages() method.
        """
        self._run_boundaries.append(len(self._all_messages))
    
    def add_event(self, event: BaseAgentRunEvent) -> None:
        """
        Add an agent run event to the event history.
        
        Args:
            event: An AgentRunEvent to add.
        """
        self._events.append(event)
    
    def get_events(self) -> List[BaseAgentRunEvent]:
        """
        Get all agent run events.
        
        Returns:
            List of all agent run events captured during execution.
        """
        return self._events.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert RunResult to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the RunResult
        """
        result: Dict[str, Any] = {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
        }
        
        if self.output is not None:
            if hasattr(self.output, 'model_dump'):
                result["output"] = self.output.model_dump(exclude_none=True)
            else:
                result["output"] = self.output
        
        if self.input is not None:
            result["input"] = self.input.to_dict()
        
        if self._events:
            result["events"] = [event.to_dict() for event in self._events]
        
        return result
    
    def __str__(self) -> str:
        """String representation returns the output as string."""
        return str(self.output)
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"RunResult(output={self.output!r}, messages_count={len(self._all_messages)})"


@dataclass
class StreamRunResult(Generic[OutputDataT]):
    """
    A comprehensive result wrapper for streaming agent executions.
    
    This class provides:
    - Async context manager support for streaming operations
    - Real-time message and event tracking during streaming
    - Access to streaming events as they occur
    - Final result accumulation after streaming completes
    - Integration with agent message tracking system
    
    Usage:
        ```python
        async with agent.run_stream(task) as result:
            async for output in result.stream():
                print(output, end='', flush=True)
            final_result = result.get_final_output()
        ```
    
    Attributes:
        _agent: Reference to the agent instance
        _task: The task being executed
        _accumulated_text: Text accumulated during streaming
        _streaming_events: All streaming events received
        _final_output: The final output after streaming completes
        _all_messages: All messages from the streaming session
        _run_boundaries: Message boundaries for tracking runs
        _is_complete: Whether streaming has completed
        _context_entered: Whether we're inside the async context
        _model: Model override for streaming
        _debug: Debug mode for streaming
        _retry: Number of retries for streaming
    """
    
    _agent: Any = None
    """Reference to the agent instance for streaming operations."""
    
    _task: Optional['Task'] = None
    """The task being executed."""
    
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this run."""
    
    agent_id: str = ""
    """Unique identifier for the agent."""
    
    agent_name: str = ""
    """Display name of the agent."""
    
    input: Optional[RunInput] = None
    """The RunInput that initiated this run."""
    
    status: RunStatus = RunStatus.pending
    """Current status of the run."""
    
    _accumulated_text: str = field(default_factory=str)
    """Text content accumulated during streaming."""
    
    _streaming_events: List[ModelResponseStreamEvent] = field(default_factory=list)
    """All streaming events received during execution (model-level)."""
    
    _agent_events: List[BaseAgentRunEvent] = field(default_factory=list)
    """Agent-level run events captured during streaming."""
    
    _final_output: Optional[OutputDataT] = None
    """The final output after streaming completes."""
    
    _all_messages: List[ModelMessage] = field(default_factory=list)
    """Internal storage for all messages during streaming."""
    
    _run_boundaries: List[int] = field(default_factory=list)
    """Indices marking where each run starts in the message list."""
    
    _is_complete: bool = False
    """Whether the streaming operation has completed."""
    
    _context_entered: bool = False
    """Whether we're currently inside the async context manager."""
    
    _start_time: Optional[float] = None
    """Timestamp when streaming started."""
    
    _end_time: Optional[float] = None
    """Timestamp when streaming completed."""
    
    _first_token_time: Optional[float] = None
    """Timestamp when first token was received."""
    
    _model: Any = None
    """Model override for streaming."""
    
    _debug: bool = False
    """Debug mode for streaming."""
    
    _retry: int = 1
    """Number of retries for streaming."""
    
    _event_queue: Optional[asyncio.Queue] = field(default=None, repr=False)
    """Queue for real-time event dispatch during streaming."""
    
    _event_queue_enabled: bool = False
    """Whether real-time event queue is enabled."""
    
    def __post_init__(self):
        """Initialize the stream run result."""
        if not self._run_boundaries:
            self._run_boundaries.append(0)
    
    async def __aenter__(self) -> 'StreamRunResult[OutputDataT]':
        """Enter the async context manager."""
        if self._context_entered:
            raise RuntimeError("StreamRunResult context manager is already active")
        
        self._context_entered = True
        self._start_time = time.time()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager."""
        self._context_entered = False
        
        if self._end_time is None:
            self._end_time = time.time()
        
        return False
    
    def get_agent_events(self) -> List[BaseAgentRunEvent]:
        """
        Get all agent-level run events captured during streaming.
        
        Returns:
            List[BaseAgentRunEvent]: All captured agent events
        """
        return self._agent_events.copy()
    
    def add_agent_event(self, event: BaseAgentRunEvent) -> None:
        """
        Add an agent run event, dispatching it in real-time if streaming is active.
        
        This method stores the event and also puts it in the event queue
        if real-time streaming is enabled (full_stream=True).
        
        Args:
            event: The agent event to add
        """
        self._agent_events.append(event)
        
        # Put in queue for real-time dispatch if enabled
        if self._event_queue_enabled and self._event_queue is not None:
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Skip if queue is full
    
    async def stream_output(
        self, 
        full_stream: bool = False
    ) -> AsyncIterator[Union[str, BaseAgentRunEvent]]:
        """
        Stream content from the agent response.
        
        Args:
            full_stream: If False (default), yields only text content chunks.
                        If True, yields all agent events including tool calls,
                        thinking events, and content events.
        
        Yields:
            When full_stream=False: str - Incremental text content
            When full_stream=True: BaseAgentRunEvent - All agent events
            
        Example:
            ```python
            # Standard text streaming (backward compatible)
            async with agent.stream(task) as result:
                async for text_chunk in result.stream_output():
                    print(text_chunk, end='', flush=True)
                print()  # New line after streaming
            
            # Full event streaming
            async with agent.stream(task) as result:
                async for event in result.stream_output(full_stream=True):
                    if event.event == "ToolCallStarted":
                        print(f"Tool: {event.tool_name}")
                    elif event.event == "RunContent":
                        print(event.content, end='', flush=True)
            ```
        """
        if not self._context_entered:
            raise RuntimeError("StreamRunResult.stream_output() must be called within async context manager")
        
        if not self._agent or not self._task:
            raise RuntimeError("No agent or task available for streaming")
        
        # Update status to running
        self.status = RunStatus.running
        
        # Emit run started event
        run_started = RunStartedEvent(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            run_id=self.run_id,
            model=str(getattr(self._model, 'name', self._model)) if self._model else "",
        )
        self._agent_events.append(run_started)
        if full_stream:
            yield run_started
        
        try:
            # Get streaming parameters
            state = getattr(self, '_state', None)
            graph_execution_id = getattr(self, '_graph_execution_id', None)
            
            if full_stream:
                # Use index-based tracking to yield new events as they're added
                # This is more reliable than queue as it handles timing issues
                last_yielded_idx = len(self._agent_events)
                
                # Stream model events and yield new agent events as they appear
                async for event in self._agent._stream_events_output(
                    self._task,
                    self._model,
                    self._debug,
                    self._retry,
                    state,
                    graph_execution_id,
                    stream_result=self
                ):
                    # Yield any new agent events that were added (from _emit_event)
                    while last_yielded_idx < len(self._agent_events):
                        yield self._agent_events[last_yielded_idx]
                        last_yielded_idx += 1
                    
                    # Convert model-level events to agent-level events
                    # Skip ToolCallPart - we emit proper ToolCallStarted/Completed from _emit_event
                    agent_event = self._convert_to_agent_event(event)
                    if agent_event:
                        self._agent_events.append(agent_event)
                        yield agent_event
                        last_yielded_idx += 1
                
                # Yield any remaining events after model stream completes
                while last_yielded_idx < len(self._agent_events):
                    yield self._agent_events[last_yielded_idx]
                    last_yielded_idx += 1
            else:
                # Yield only text content (backward compatible)
                async for text_chunk in self._agent._stream_text_output(
                    self._task,
                    self._model,
                    self._debug,
                    self._retry,
                    state,
                    graph_execution_id,
                    stream_result=self
                ):
                    # Create content event and store it
                    content_event = RunContentEvent(
                        agent_id=self.agent_id,
                        agent_name=self.agent_name,
                        run_id=self.run_id,
                        content=text_chunk,
                    )
                    self._agent_events.append(content_event)
                    yield text_chunk
            
            self._end_time = time.time()
            self.status = RunStatus.completed
            
            # Emit run completed event
            run_completed = RunCompletedEvent(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                run_id=self.run_id,
                content=self._final_output,
                duration_ms=int((self._end_time - self._start_time) * 1000) if self._start_time else None,
            )
            self._agent_events.append(run_completed)
            if full_stream:
                yield run_completed
            
        except Exception as e:
            # Ensure completion state is set even on error
            self._is_complete = True
            self._end_time = time.time()
            self.status = RunStatus.error
            
            # Emit error event
            error_event = RunErrorEvent(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                run_id=self.run_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self._agent_events.append(error_event)
            if full_stream:
                yield error_event
            
            raise
    
    def _convert_to_agent_event(self, model_event: ModelResponseStreamEvent) -> Optional[BaseAgentRunEvent]:
        """
        Convert a model-level streaming event to an agent-level event.
        
        Args:
            model_event: The model-level streaming event
            
        Returns:
            BaseAgentRunEvent or None if the event doesn't map to an agent event
        """
        from upsonic.messages.messages import PartStartEvent, PartDeltaEvent, TextPart, ToolCallPart, ThinkingPart, TextPartDelta
        from upsonic.agent.run_events import ThinkingStartedEvent, ThinkingStepEvent
        
        if isinstance(model_event, PartStartEvent):
            part = model_event.part
            if isinstance(part, TextPart):
                return RunContentEvent(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    run_id=self.run_id,
                    content=part.content,
                )
            elif isinstance(part, ThinkingPart):
                # ThinkingPart starts thinking/reasoning
                return ThinkingStartedEvent(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    run_id=self.run_id,
                )
            # ToolCallPart is NOT converted here - we emit proper ToolCallStarted/Completed
            # from _emit_event in Agent._execute_tool_calls with full args and results
        elif isinstance(model_event, PartDeltaEvent):
            delta = model_event.delta
            if isinstance(delta, TextPartDelta):
                return RunContentEvent(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    run_id=self.run_id,
                    content=delta.content_delta,
                )
            # Check for thinking delta
            if hasattr(delta, 'content_delta') and hasattr(delta, '__class__') and 'Thinking' in delta.__class__.__name__:
                return ThinkingStepEvent(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    run_id=self.run_id,
                    thinking_content=getattr(delta, 'content_delta', ''),
                )
        
        return None
    
    async def stream_events(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """
        Stream raw events from the agent response.
        
        This provides access to the full stream events including tool calls,
        thinking events, and other non-text content.
        
        Yields:
            ModelResponseStreamEvent: Raw streaming events
            
        Example:
            ```python
            async with agent.run_stream(task) as result:
                async for event in result.stream_events():
                    if isinstance(event, PartStartEvent):
                        print(f"New part: {type(event.part).__name__}")
                    elif isinstance(event, FinalResultEvent):
                        print("Final result available")
            ```
        """
        if not self._context_entered:
            raise RuntimeError("StreamRunResult.stream_events() must be called within async context manager")
        
        if not self._agent or not self._task:
            raise RuntimeError("No agent or task available for streaming")
        
        try:
            state = getattr(self, '_state', None)
            graph_execution_id = getattr(self, '_graph_execution_id', None)
            
            # Always execute the stream to get real-time events
            # Events are collected in _streaming_events during iteration
            async for event in self._agent._stream_events_output(
                self._task,
                self._model,
                self._debug,
                self._retry,
                state,
                graph_execution_id,
                stream_result=self
            ):
                yield event
            
            self._end_time = time.time()
            
        except Exception as e:
            # Ensure completion state is set even on error
            self._is_complete = True
            self._end_time = time.time()
            raise
    
    def stream_output_sync(self) -> Iterator[str]:
        """
        Stream text content from the agent response synchronously.
        
        This is a synchronous wrapper around stream_output().
        
        Yields:
            str: Incremental text content as it becomes available
            
        Example:
            ```python
            result = agent.stream(task)
            for text_chunk in result.stream_output_sync():
                print(text_chunk, end='', flush=True)
            ```
        """
        queue: Queue = Queue()
        exception_holder = [None]
        done = threading.Event()
        
        async def _run_async():
            try:
                async with self:
                    async for item in self.stream_output():
                        queue.put(item)
            except Exception as e:
                exception_holder[0] = e
            finally:
                queue.put(None)
                done.set()
        
        def _run_in_thread():
            try:
                asyncio.run(_run_async())
            except Exception as e:
                exception_holder[0] = e
                queue.put(None)
                done.set()
        
        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()
        
        while True:
            try:
                item = queue.get(timeout=0.1)
                if item is None:
                    break
                yield item
            except Empty:
                if done.is_set():
                    break
                continue
        
        if exception_holder[0]:
            raise exception_holder[0]
    
    def stream_events_sync(self) -> Iterator[ModelResponseStreamEvent]:
        """
        Stream raw events from the agent response synchronously.
        
        This is a synchronous wrapper around stream_events().
        
        Yields:
            ModelResponseStreamEvent: Raw streaming events
            
        Example:
            ```python
            result = agent.stream(task)
            for event in result.stream_events_sync():
                if isinstance(event, PartStartEvent):
                    print(f"New part: {type(event.part).__name__}")
            ```
        """
        queue: Queue = Queue()
        exception_holder = [None]
        done = threading.Event()
        
        async def _run_async():
            try:
                async with self:
                    async for item in self.stream_events():
                        queue.put(item)
            except Exception as e:
                exception_holder[0] = e
            finally:
                queue.put(None)
                done.set()
        
        def _run_in_thread():
            try:
                asyncio.run(_run_async())
            except Exception as e:
                exception_holder[0] = e
                queue.put(None)
                done.set()
        
        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()
        
        while True:
            try:
                item = queue.get(timeout=0.1)
                if item is None:
                    break
                yield item
            except Empty:
                if done.is_set():
                    break
                continue
        
        if exception_holder[0]:
            raise exception_holder[0]
    
    def get_final_output(self) -> Optional[OutputDataT]:
        """
        Get the final accumulated output after streaming completes.
        
        Returns:
            The final output, or None if streaming hasn't completed
        """
        return self._final_output
    
    @property
    def output(self) -> Optional[OutputDataT]:
        """Get the current accumulated output."""
        return self._final_output if self._is_complete else self._accumulated_text
    
    def is_complete(self) -> bool:
        """Check if streaming has completed."""
        return self._is_complete
    
    def get_accumulated_text(self) -> str:
        """Get all text accumulated so far."""
        return self._accumulated_text
    
    def get_streaming_events(self) -> List[ModelResponseStreamEvent]:
        """Get all model-level streaming events received so far."""
        return self._streaming_events.copy()
    
    def get_text_events(self) -> List[ModelResponseStreamEvent]:
        """Get only text-related streaming events."""
        return [event for event in self._streaming_events 
                if isinstance(event, (PartStartEvent, PartDeltaEvent)) 
                and hasattr(event, 'part') and isinstance(event.part, TextPart) 
                or hasattr(event, 'delta') and hasattr(event.delta, 'content_delta')]
    
    def get_tool_events(self) -> List[ModelResponseStreamEvent]:
        """Get only tool-related streaming events."""
        return [event for event in self._streaming_events 
                if isinstance(event, PartStartEvent) and hasattr(event.part, 'tool_name')]
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the streaming session."""
        if not self._streaming_events:
            return {}
        
        text_events = self.get_text_events()
        tool_events = self.get_tool_events()
        
        return {
            'total_events': len(self._streaming_events),
            'text_events': len(text_events),
            'tool_events': len(tool_events),
            'accumulated_chars': len(self._accumulated_text),
            'is_complete': self._is_complete,
            'has_final_output': self._final_output is not None,
            'event_types': {
                type(event).__name__: sum(1 for e in self._streaming_events if type(e) == type(event))
                for event in self._streaming_events
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Optional[float]]:
        """Get detailed performance metrics for the streaming session."""
        metrics = {
            'start_time': self._start_time,
            'end_time': self._end_time,
            'first_token_time': self._first_token_time,
            'total_duration': None,
            'time_to_first_token': None,
            'tokens_per_second': None,
            'characters_per_second': None,
        }
        
        if self._start_time and self._end_time:
            metrics['total_duration'] = self._end_time - self._start_time
            
        if self._start_time and self._first_token_time:
            metrics['time_to_first_token'] = self._first_token_time - self._start_time
        
        if metrics['total_duration'] and metrics['total_duration'] > 0:
            estimated_tokens = len(self._accumulated_text) / 4
            metrics['tokens_per_second'] = estimated_tokens / metrics['total_duration']
            metrics['characters_per_second'] = len(self._accumulated_text) / metrics['total_duration']
        
        return metrics
    
    def all_messages(self) -> List[ModelMessage]:
        """
        Get all messages from the streaming session.
        
        Returns:
            List of all ModelMessage objects from the streaming session
        """
        return self._all_messages.copy()
    
    def new_messages(self) -> List[ModelMessage]:
        """
        Get messages from the last run only.
        
        Returns:
            List of all ModelMessage objects from the most recent run
        """
        if not self._run_boundaries:
            return self._all_messages.copy()
        
        last_run_start_idx = self._run_boundaries[-1]
        return self._all_messages[last_run_start_idx:].copy()
    
    def get_last_model_response(self) -> Optional['ModelResponse']:
        """
        Get the last ModelResponse from the messages.
        
        This method searches through the messages from the last run and returns
        the most recent ModelResponse, if any exists.
        
        Returns:
            The last ModelResponse from the messages, or None if no ModelResponse exists.
        """
        from upsonic.messages.messages import ModelResponse
        
        messages = self.new_messages()
        for msg in reversed(messages):
            if isinstance(msg, ModelResponse):
                return msg
        return None
    
    def add_messages(self, messages: List[ModelMessage]) -> None:
        """
        Add messages to the internal message store.
        
        Args:
            messages: List of ModelMessage objects to add
        """
        self._all_messages.extend(messages)
    
    def add_message(self, message: ModelMessage) -> None:
        """
        Add a single message to the internal message store.
        
        Args:
            message: A ModelMessage object to add
        """
        self._all_messages.append(message)
    
    def start_new_run(self) -> None:
        """
        Mark the start of a new run in the message history.
        
        This should be called before adding messages from a new streaming run
        to properly track run boundaries for the new_messages() method.
        """
        self._run_boundaries.append(len(self._all_messages))
    
    def __str__(self) -> str:
        """String representation returns the accumulated text."""
        return str(self._accumulated_text or "")
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        status = "complete" if self._is_complete else "streaming"
        return (f"StreamRunResult(status={status}, "
                f"accumulated_chars={len(self._accumulated_text)}, "
                f"events_count={len(self._streaming_events)}, "
                f"messages_count={len(self._all_messages)})")


# Type aliases for easier imports
AgentRunResult = RunResult[OutputDataT]
"""Type alias for the standard agent run result."""

StreamAgentRunResult = StreamRunResult[OutputDataT]  
"""Type alias for the streaming agent run result."""
