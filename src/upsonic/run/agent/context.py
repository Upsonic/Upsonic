from __future__ import annotations

import cloudpickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

if TYPE_CHECKING:
    from pydantic import BaseModel
    from upsonic.run.requirements import RunRequirement
    from upsonic.schemas.kb_filter import KBFilterExpr
    from upsonic.messages.messages import ModelMessage, ModelResponse
    from upsonic.run.events.events import AgentEvent
    from upsonic.agent.pipeline.step import StepResult
    from upsonic.run.pipeline.stats import PipelineExecutionStats
    from upsonic.tasks.tasks import Task



@dataclass
class AgentRunContext:
    """
    Complete runtime context for an agent run.
    
    This context flows through the pipeline and is modified by each step.
    Contains all execution state and data needed for the run.
    
    PipelineManager holds task, agent, model references and passes them to steps.
    This context is pure data - no business logic.
    
    Attributes:
        run_id: Unique identifier for this run
        session_id: Session identifier
        user_id: User identifier
        task: Embedded task (single source of truth for HITL)
        step_results: List of StepResult tracking each step's execution
        execution_stats: Pipeline execution statistics
        requirements: HITL requirements (contains ALL continuation data)
        agent_knowledge_base_filter: Vector search filters
        session_state: Session state persisted across runs
        output_schema: Output schema constraint
        is_streaming: Whether this is streaming execution
        accumulated_text: Text accumulated during streaming
        chat_history: Full conversation history (historical + current run messages) for LLM
        messages: Only NEW messages from THIS run (synced with AgentRunOutput.messages)
        response: Current ModelResponse (single, not list)
        final_output: Final processed output (text, BaseModel, etc.)
        events: All events emitted during execution
    """
    
    run_id: str
    session_id: str
    user_id: Optional[str] = None
    
    # Task - embedded for single source of truth
    task: Optional["Task"] = None
    
    # Step execution tracking
    step_results: List["StepResult"] = field(default_factory=list) # Step results for that run
    
    # Pipeline execution statistics
    execution_stats: Optional["PipelineExecutionStats"] = None
    
    # HITL requirements - contains ALL continuation data
    requirements: List["RunRequirement"] = field(default_factory=list)
    
    # Configuration
    agent_knowledge_base_filter: Optional["KBFilterExpr"] = None
    session_state: Optional[Dict[str, Any]] = None # Session state for that run
    output_schema: Optional[Union[str, Type["BaseModel"]]] = None
    
    # Execution state
    is_streaming: bool = False
    accumulated_text: str = "" # Text accumulated during streaming
    
    # Tool tracking (moved from Agent class - run-specific)
    tool_call_count: int = 0
    tool_limit_reached: bool = False
    
    # Chat history - Full conversation history (historical messages + current run) for LLM execution
    chat_history: List["ModelMessage"] = field(default_factory=list)
    
    # Messages - Only NEW messages from THIS run (same as AgentRunOutput.messages)
    messages: List["ModelMessage"] = field(default_factory=list)
    response: Optional["ModelResponse"] = None
    final_output: Optional[Union[str, bytes]] = None
    events: List["AgentEvent"] = field(default_factory=list) # Events for that run
    
    # Current step result (set by Step.execute, read by Step.run)
    current_step_result: Optional["StepResult"] = None
    
    def add_requirement(self, requirement: "RunRequirement") -> None:
        """Add a HITL requirement to this context."""
        self.requirements.append(requirement)
    
    def get_active_requirements(self) -> List["RunRequirement"]:
        """Get unresolved requirements."""
        return [req for req in self.requirements if not req.is_resolved()]
    
    def has_pending_requirements(self) -> bool:
        """Check if there are any unresolved requirements."""
        return len(self.get_active_requirements()) > 0
    
    def get_last_successful_step(self) -> Optional["StepResult"]:
        """Get the last successfully completed step."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in reversed(self.step_results):
            if result.status == StepStatus.COMPLETED:
                return result
        return None
    
    def get_error_step(self) -> Optional["StepResult"]:
        """Get the step that failed with ERROR status (for durable execution)."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in self.step_results:
            if result.status == StepStatus.ERROR:
                return result
        return None
    
    def get_cancelled_step(self) -> Optional["StepResult"]:
        """Get the step that was CANCELLED (for cancel run resumption)."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in self.step_results:
            if result.status == StepStatus.CANCELLED:
                return result
        return None
    
    def get_paused_step(self) -> Optional["StepResult"]:
        """Get the step that is PAUSED (for external tool execution)."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in self.step_results:
            if result.status == StepStatus.PAUSED:
                return result
        return None
    
    def _serialize_message(self, msg: Any) -> Any:
        """Serialize a message to dict if it has to_dict method."""
        if msg is None:
            return None
        if hasattr(msg, 'to_dict'):
            return msg.to_dict()
        return msg
    
    def _serialize_event(self, event: Any) -> Any:
        """Serialize an event to dict if it has to_dict method."""
        if event is None:
            return None
        if hasattr(event, 'to_dict'):
            return event.to_dict()
        return str(event)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (pure transformation, nested objects call their to_dict())."""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "task": self.task.to_dict() if self.task and hasattr(self.task, 'to_dict') else None,
            "step_results": [sr.to_dict() for sr in self.step_results] if self.step_results else [],
            "execution_stats": self.execution_stats.to_dict() if self.execution_stats else None,
            "session_state": self.session_state,
            "is_streaming": self.is_streaming,
            "accumulated_text": self.accumulated_text,
            "tool_call_count": self.tool_call_count,
            "tool_limit_reached": self.tool_limit_reached,
            "requirements": [r.to_dict() for r in self.requirements] if self.requirements else [],
            "agent_knowledge_base_filter": self.agent_knowledge_base_filter,
            "output_schema": str(self.output_schema) if self.output_schema else None,
            "chat_history": [self._serialize_message(m) for m in self.chat_history] if self.chat_history else [],
            "messages": [self._serialize_message(m) for m in self.messages] if self.messages else [],
            "response": self._serialize_message(self.response),
            "final_output": self.final_output,
            "events": [self._serialize_event(e) for e in self.events] if self.events else [],
        }
    
    @classmethod
    def _deserialize_messages(cls, messages_data: List[Any]) -> List[Any]:
        """Helper to deserialize a list of messages."""
        messages = []
        for m in messages_data:
            if isinstance(m, dict) and 'parts' in m:
                # It's a serialized ModelRequest/ModelResponse - keep as-is for cloudpickle
                messages.append(m)
            else:
                messages.append(m)
        return messages
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRunContext":
        """Reconstruct from dictionary (pure transformation, nested objects reconstructed)."""
        from upsonic.run.requirements import RunRequirement
        from upsonic.agent.pipeline.step import StepResult
        from upsonic.run.pipeline.stats import PipelineExecutionStats
        from upsonic.tasks.tasks import Task
        
        # Handle step_results (list of dicts) - use model_validate if Pydantic
        step_results: List[Any] = []
        step_results_data = data.get("step_results", [])
        for sr in step_results_data:
            if isinstance(sr, dict):
                if hasattr(StepResult, 'model_validate'):
                    step_results.append(StepResult.model_validate(sr))
                else:
                    step_results.append(StepResult.from_dict(sr))
            else:
                step_results.append(sr)
        
        # Handle execution_stats (dict) - use model_validate if Pydantic
        execution_stats = None
        execution_stats_data = data.get("execution_stats")
        if isinstance(execution_stats_data, dict):
            if hasattr(PipelineExecutionStats, 'model_validate'):
                execution_stats = PipelineExecutionStats.model_validate(execution_stats_data)
            else:
                execution_stats = PipelineExecutionStats.from_dict(execution_stats_data)
        else:
            execution_stats = execution_stats_data
        
        # Handle requirements (list of dicts) - use model_validate if Pydantic
        requirements: List[Any] = []
        requirements_data = data.get("requirements", [])
        for r in requirements_data:
            if isinstance(r, dict):
                if hasattr(RunRequirement, 'model_validate'):
                    requirements.append(RunRequirement.model_validate(r))
                else:
                    requirements.append(RunRequirement.from_dict(r))
            else:
                requirements.append(r)
        
        # Handle task (dict or Task object) - use from_dict because Task has private attributes
        task = None
        task_data = data.get("task")
        if isinstance(task_data, dict):
            # Task has private attributes (_response, _context_formatted) that 
            # Pydantic's model_validate can't handle, so always use from_dict
            if hasattr(Task, 'from_dict'):
                task = Task.from_dict(task_data)
            else:
                task = Task.model_validate(task_data)
        else:
            task = task_data
        
        # Handle chat_history (list of dicts or objects)
        chat_history = cls._deserialize_messages(data.get("chat_history", []))
        
        # Handle messages (list of dicts or objects)
        messages = cls._deserialize_messages(data.get("messages", []))
        
        # Handle events (list of dicts or objects)
        events = []
        events_data = data.get("events", [])
        for e in events_data:
            if isinstance(e, dict) and '__event_class__' in e:
                from upsonic.run.events.events import AgentEvent
                events.append(AgentEvent.from_dict(e))
            elif isinstance(e, dict):
                from upsonic.run.events.events import AgentEvent
                try:
                    events.append(AgentEvent.from_dict(e))
                except Exception:
                    events.append(e)
            else:
                events.append(e)
        
        return cls(
            run_id=data["run_id"],
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            task=task,
            step_results=step_results,
            execution_stats=execution_stats,
            requirements=requirements,
            agent_knowledge_base_filter=data.get("agent_knowledge_base_filter"),
            session_state=data.get("session_state"),
            output_schema=data.get("output_schema"),
            is_streaming=data.get("is_streaming", False),
            accumulated_text=data.get("accumulated_text", ""),
            tool_call_count=data.get("tool_call_count", 0),
            tool_limit_reached=data.get("tool_limit_reached", False),
            chat_history=chat_history,
            messages=messages,
            response=data.get("response"),
            final_output=data.get("final_output"),
            events=events,
        )
    
    def serialize(self) -> bytes:
        """Serialize to bytes for storage."""
        return cloudpickle.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "AgentRunContext":
        """Deserialize from bytes."""
        dict_data = cloudpickle.loads(data)
        return cls.from_dict(dict_data)
