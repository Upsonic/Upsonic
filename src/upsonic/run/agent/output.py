from __future__ import annotations

from dataclasses import dataclass, field
from time import time as current_time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from upsonic.run.base import RunStatus

if TYPE_CHECKING:
    from pydantic import BaseModel
    from upsonic.run.agent.context import AgentRunContext
    from upsonic.messages.messages import (
        BinaryContent,
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ThinkingPart,
    )
    from upsonic.run.agent.input import AgentRunInput
    from upsonic.run.events.events import AgentEvent
    from upsonic.run.requirements import RunRequirement
    from upsonic.run.tools.tools import ToolExecution
    from upsonic.usage import RequestUsage
    from upsonic.profiles import ModelProfile
    from upsonic.agent.pipeline.step import StepResult
    from upsonic.run.pipeline.stats import PipelineExecutionStats


@dataclass
class AgentRunOutput:
    """Complete output from an agent run.
    
    Replaces RunResult and StreamRunResult with a unified class that handles:
    - Message tracking with run boundaries
    - Tool execution results
    - HITL requirements
    - Streaming state with async context manager
    - Event streaming
    - Text output streaming
    - Comprehensive serialization
    """
    
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Input reference
    input: Optional["AgentRunInput"] = None
    
    # Output content
    content: Optional[Any] = None
    output_schema: Optional[Union[str, Type["BaseModel"]]] = None
    
    # Thinking/reasoning
    thinking_content: Optional[str] = None
    thinking_parts: Optional[List["ThinkingPart"]] = None
    
    # Model info
    model: Optional[Union[str, Any]] = None # Model name
    model_provider: Optional[str] = None
    model_provider_profile: Optional["ModelProfile"] = None
    
    # Messages and usage
    messages: Optional[List["ModelMessage"]] = None # Must be same with the agent_run_context.messages. Messages for that run
    usage: Optional["RequestUsage"] = None
    additional_input_message: Optional[List["ModelRequest"]] = None
    
    # Tool executions
    tools: Optional[List["ToolExecution"]] = None
    
    # Media outputs
    images: Optional[List["BinaryContent"]] = None # Output image
    files: Optional[List["BinaryContent"]] = None # Output file
    
    # Status and HITL
    status: RunStatus = field(default_factory=lambda: RunStatus.running)
    requirements: Optional[List["RunRequirement"]] = None # Must be same with the agent_run_context.requirements
    step_results: List["StepResult"] = field(default_factory=list) # Step results for that run

    # Pipeline execution statistics
    execution_stats: Optional["PipelineExecutionStats"] = None
    
    # Events (for streaming)
    events: Optional[List["AgentEvent"]] = None # Must be same with the agent_run_context.events
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    session_state: Optional[Dict[str, Any]] = None # Must be same with the agent_run_context.session_state
    
    # User-facing pause information
    pause_reason: Optional[str] = None  # "durable_execution", "external_tool", "cancel". Must be same with the latest requirement.pause_type
    error_details: Optional[str] = None
    
    # Message tracking
    _run_boundaries: List[int] = field(default_factory=list)
    
    # Output tracking
    _accumulated_text: str = "" # Text accumulated during streaming
    
    
    # Timestamps
    created_at: int = field(default_factory=lambda: int(current_time()))
    updated_at: Optional[int] = None
    
    # --- Properties ---
    
    @property
    def is_paused(self) -> bool:
        """Check if the run is paused."""
        return self.status == RunStatus.paused
    
    @property
    def is_cancelled(self) -> bool:
        """Check if the run was cancelled."""
        return self.status == RunStatus.cancelled
    
    @property
    def is_complete(self) -> bool:
        """Check if the run is complete."""
        return self.status == RunStatus.completed
    
    @property
    def is_error(self) -> bool:
        """Check if the run has an error."""
        return self.status == RunStatus.error
    
    @property
    def output(self) -> Optional[Any]:
        """Get the output content."""
        return self.content
    
    @property
    def active_requirements(self) -> List["RunRequirement"]:
        """Get unresolved HITL requirements."""
        if not self.requirements:
            return []
        return [req for req in self.requirements if not req.is_resolved()]
    
    @property
    def tools_requiring_confirmation(self) -> List["ToolExecution"]:
        """Get tools that require user confirmation."""
        if not self.tools:
            return []
        return [t for t in self.tools if t.requires_confirmation and not t.confirmed]
    
    @property
    def tools_requiring_user_input(self) -> List["ToolExecution"]:
        """Get tools that require user input."""
        if not self.tools:
            return []
        return [t for t in self.tools if t.requires_user_input and not t.answered]
    
    @property
    def tools_awaiting_external_execution(self) -> List["ToolExecution"]:
        """Get tools awaiting external execution."""
        if not self.tools:
            return []
        return [t for t in self.tools if t.external_execution_required and t.result is None]
    
    # --- HITL Context Methods ---
    
    def get_step_results(self) -> List["StepResult"]:
        """Get step results from execution context."""
        if self.step_results:
            return self.step_results
        return []
    
    def get_execution_stats(self) -> Optional["PipelineExecutionStats"]:
        """Get execution statistics from context."""
        if self.execution_stats:
            return self.execution_stats
        return None
    
    # --- Message Tracking Methods ---
    
    def all_messages(self) -> List["ModelMessage"]:
        """Get all messages from the run."""
        return (self.messages or []).copy()
    
    def new_messages(self) -> List["ModelMessage"]:
        """Get messages from the last iteration only."""
        if not self._run_boundaries or not self.messages:
            return (self.messages or []).copy()
        last_start = self._run_boundaries[-1]
        return self.messages[last_start:].copy()
    
    def add_messages(self, messages: List["ModelMessage"]) -> None:
        """Add messages to the run."""
        if self.messages is None:
            self.messages = []
        self.messages.extend(messages)
    
    def add_message(self, message: "ModelMessage") -> None:
        """Add a single message to the run."""
        if self.messages is None:
            self.messages = []
        self.messages.append(message)
    
    def start_new_run(self) -> None:
        """Initialize a new run by starting a new iteration."""
        self.start_new_iteration()
    
    def start_new_iteration(self) -> None:
        """Mark the start of a new tool call iteration."""
        if self.messages:
            self._run_boundaries.append(len(self.messages))
        else:
            self._run_boundaries.append(0)
    
    def get_last_model_response(self) -> Optional["ModelResponse"]:
        """Get the last ModelResponse from the messages."""
        from upsonic.messages.messages import ModelResponse
        
        messages = self.new_messages()
        for msg in reversed(messages):
            if isinstance(msg, ModelResponse):
                return msg
        return None
    
    
    
    # --- Status Methods ---
    
    def mark_paused(self) -> None:
        """Mark the run as paused."""
        self.status = RunStatus.paused
        self.updated_at = int(current_time())
    
    def mark_cancelled(self) -> None:
        """Mark the run as cancelled."""
        self.status = RunStatus.cancelled
        self.updated_at = int(current_time())
    
    def mark_completed(self) -> None:
        """Mark the run as completed."""
        self.status = RunStatus.completed
        self.updated_at = int(current_time())
        if self.content is None and self._accumulated_text:
            self.content = self._accumulated_text
    
    def mark_error(self, error: Optional[str] = None) -> None:
        """Mark the run as having an error."""
        self.status = RunStatus.error
        if error:
            if self.metadata is None:
                self.metadata = {}
            self.metadata["error"] = error
        self.updated_at = int(current_time())
    
    # --- Context Sync Methods ---
    
    def sync_from_context(self, agent_run_context: "AgentRunContext") -> None:
        """Sync output attributes from the agent_run_context.
        
        This ensures that AgentRunOutput reflects the current state of
        AgentRunContext. Should be called after context updates.
        
        NOTE: context.messages now contains only THIS run's messages (same semantics
        as AgentRunOutput.messages), while context.chat_history contains the full
        conversation history for LLM execution.
        """
        
        # Sync messages - both now have the same semantics (only this run's messages)
        self.messages = agent_run_context.messages
        
        # Sync requirements
        self.requirements = agent_run_context.requirements
        
        # Sync events
        self.events = agent_run_context.events
        
        # Sync session_state
        self.session_state = agent_run_context.session_state
        
        # Sync accumulated text
        self._accumulated_text = agent_run_context.accumulated_text
        
        # Sync step_results
        self.step_results = agent_run_context.step_results
        
        # Sync execution_stats
        self.execution_stats = agent_run_context.execution_stats
        
        # Sync final output
        if agent_run_context.final_output is not None:
            self.content = agent_run_context.final_output
        
        # Derive status from step results
        self.status = self._derive_status_from_context(agent_run_context)
        
        # Derive pause reason from requirements
        if agent_run_context.requirements:
            active_reqs = agent_run_context.get_active_requirements()
            if active_reqs:
                self.pause_reason = active_reqs[-1].pause_type
        
        self.updated_at = int(current_time())
    
    def _derive_status_from_context(self, agent_run_context: "AgentRunContext") -> RunStatus:
        """Derive RunStatus from step results in context."""
        if not agent_run_context or not agent_run_context.step_results:
            return self.status  # Keep current status
        
        from upsonic.agent.pipeline.step import StepStatus
        
        last_step = agent_run_context.step_results[-1]
        return StepStatus.to_run_status(last_step.status)
    
    # --- Serialization ---
    
    def _serialize_item(self, item: Any) -> Any:
        """Serialize an item to dict if it has to_dict method."""
        if item is None:
            return None
        if hasattr(item, 'to_dict'):
            return item.to_dict()
        return item
    
    def _serialize_list(self, items: Optional[List[Any]]) -> Optional[List[Any]]:
        """Serialize a list of items, calling to_dict on each if available."""
        if items is None:
            return None
        return [self._serialize_item(item) for item in items]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (pure transformation, nested objects call their to_dict())."""
        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "parent_run_id": self.parent_run_id,
            "user_id": self.user_id,
            "status": self.status.value if hasattr(self.status, 'value') else self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "input": self.input.to_dict() if self.input else None,
            "content": self.content,
            "output_schema": str(self.output_schema) if self.output_schema else None,
            "thinking_content": self.thinking_content,
            "thinking_parts": self._serialize_list(self.thinking_parts),
            "model": self.model,
            "model_provider": self.model_provider,
            "model_provider_profile": self._serialize_item(self.model_provider_profile),
            "messages": self._serialize_list(self.messages),
            "usage": self._serialize_item(self.usage),
            "additional_input_message": self._serialize_list(self.additional_input_message),
            "tools": [t.to_dict() for t in self.tools] if self.tools else None,
            "images": self._serialize_list(self.images),
            "files": self._serialize_list(self.files),
            "requirements": [r.to_dict() for r in self.requirements] if self.requirements else None,
            "step_results": [sr.to_dict() for sr in self.step_results] if self.step_results else [],
            "execution_stats": self.execution_stats.to_dict() if self.execution_stats else None,
            "events": self._serialize_list(self.events),
            "metadata": self.metadata,
            "session_state": self.session_state,
            "_run_boundaries": self._run_boundaries,
            "pause_reason": self.pause_reason,
            "error_details": self.error_details,
        }
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRunOutput":
        """Reconstruct from dictionary (pure transformation, nested objects reconstructed)."""
        from upsonic.run.agent.input import AgentRunInput
        from upsonic.run.requirements import RunRequirement
        from upsonic.run.tools.tools import ToolExecution
        from upsonic.run.events.events import AgentEvent
        from upsonic.run.pipeline.stats import PipelineExecutionStats
        from upsonic.agent.pipeline.step import StepResult

        # Handle step_results (list of dicts)
        step_results: List[Any] = []
        step_results_data = data.get("step_results", [])
        for sr in step_results_data:
            if isinstance(sr, dict):
                step_results.append(StepResult.from_dict(sr))
            else:
                step_results.append(sr)

        # Handle execution_stats (dict)
        execution_stats = None
        execution_stats_data = data.get("execution_stats")
        if isinstance(execution_stats_data, dict):
            execution_stats = PipelineExecutionStats.from_dict(execution_stats_data)
        else:
            execution_stats = execution_stats_data
        
        # Handle input (dict or object)
        input_data = data.get("input")
        input_obj = None
        if isinstance(input_data, dict):
            input_obj = AgentRunInput.from_dict(input_data)
        else:
            input_obj = input_data
        
        # Handle tools (list of dicts or objects)
        tools = None
        tools_data = data.get("tools")
        if tools_data:
            tools = []
            for t in tools_data:
                if isinstance(t, dict):
                    tools.append(ToolExecution.from_dict(t))
                else:
                    tools.append(t)
        
        # Handle requirements (list of dicts or objects)
        requirements = None
        requirements_data = data.get("requirements")
        if requirements_data:
            requirements = []
            for r in requirements_data:
                if isinstance(r, dict):
                    requirements.append(RunRequirement.from_dict(r))
                else:
                    requirements.append(r)
        
        # Handle status (str or RunStatus)
        status = data.get("status", RunStatus.running)
        if isinstance(status, str):
            status = RunStatus(status)
        
        # Handle events (list of dicts or objects)
        events = None
        events_data = data.get("events")
        if events_data:
            events = []
            for e in events_data:
                if isinstance(e, dict):
                    events.append(AgentEvent.from_dict(e))
                else:
                    events.append(e)
        
        return cls(
            run_id=data.get("run_id"),
            agent_id=data.get("agent_id"),
            agent_name=data.get("agent_name"),
            session_id=data.get("session_id"),
            parent_run_id=data.get("parent_run_id"),
            user_id=data.get("user_id"),
            status=status,
            created_at=data.get("created_at", int(current_time())),
            updated_at=data.get("updated_at"),
            input=input_obj,
            content=data.get("content"),
            output_schema=data.get("output_schema"),
            thinking_content=data.get("thinking_content"),
            thinking_parts=data.get("thinking_parts"),
            model=data.get("model"),
            model_provider=data.get("model_provider"),
            model_provider_profile=data.get("model_provider_profile"),
            messages=data.get("messages"),  # Raw objects (cloudpickle handles)
            usage=data.get("usage"),
            additional_input_message=data.get("additional_input_message"),
            tools=tools,
            images=data.get("images"),
            files=data.get("files"),
            requirements=requirements,
            step_results=step_results,
            execution_stats=execution_stats,
            events=events,
            metadata=data.get("metadata"),
            session_state=data.get("session_state"),
            _run_boundaries=data.get("_run_boundaries", []),
            pause_reason=data.get("pause_reason"),
            error_details=data.get("error_details"),
        )
    
    def __str__(self) -> str:
        return str(self.content if self.content is not None else "")
    
    def __repr__(self) -> str:
        return f"AgentRunOutput(run_id={self.run_id!r}, status={self.status.value}, content_length={len(str(self.content or ''))})"
    
    def serialize(self) -> bytes:
        """Serialize to bytes for storage."""
        import cloudpickle
        return cloudpickle.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "AgentRunOutput":
        """Deserialize from bytes."""
        import cloudpickle
        dict_data = cloudpickle.loads(data)
        return cls.from_dict(dict_data)
