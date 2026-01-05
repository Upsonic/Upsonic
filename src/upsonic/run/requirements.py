import cloudpickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

from upsonic.run.tools.tools import ToolExecution

if TYPE_CHECKING:
    from upsonic.agent.pipeline.step import StepResult
    from upsonic.run.pipeline.stats import PipelineExecutionStats
    from upsonic.tools.deferred import ExternalToolCall


@dataclass
class RunRequirement:
    """
    Self-contained requirement for paused run continuation (HITL flows).
    
    Contains ALL data needed to resume execution from any pause point:
    - External tool execution
    - Error recovery using Durable Execution
    - Cancel run resumption
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    tool_execution: Optional[Union[ToolExecution, "ExternalToolCall"]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None

    # User confirmation
    confirmation: Optional[bool] = None
    confirmation_note: Optional[str] = None

    # User input
    user_input_schema: Optional[List[Dict[str, Any]]] = None

    # External execution
    external_execution_result: Optional[str] = None
    
    # Continuation data - ALL data needed to resume from pause
    pause_type: Optional[Literal['external_tool', 'durable_execution', 'cancel']] = None
    step_result: Optional["StepResult"] = None
    execution_stats: Optional["PipelineExecutionStats"] = None
    continuation_messages: Optional[List[Any]] = None
    continuation_response: Optional[Any] = None
    agent_state: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        tool_execution: Optional[ToolExecution] = None,
        id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        pause_type: Optional[Literal['external_tool', 'durable_execution', 'cancel']] = None,
    ):
        self.id = id or str(uuid4())
        self.tool_execution = tool_execution
        self.user_input_schema = getattr(tool_execution, 'user_input_schema', None) if tool_execution else None
        self.created_at = created_at or datetime.now(timezone.utc)
        self.resolved_at = None
        self.confirmation = None
        self.confirmation_note = None
        self.external_execution_result = None
        self.pause_type = pause_type
        self.step_result = None
        self.execution_stats = None
        self.continuation_messages = None
        self.continuation_response = None
        self.agent_state = None

    @property
    def needs_confirmation(self) -> bool:
        if self.confirmation is not None:
            return False
        if not self.tool_execution:
            return False
        if self.tool_execution.confirmed is True:
            return True

        return self.tool_execution.requires_confirmation or False

    @property
    def needs_user_input(self) -> bool:
        if not self.tool_execution:
            return False
        if self.tool_execution.answered is True:
            return False
        if self.user_input_schema and not all(value is not None for value in self.user_input_schema.values()):
            return True

        return self.tool_execution.requires_user_input or False

    @property
    def needs_external_execution(self) -> bool:
        if not self.tool_execution:
            return False
        # If result is already set (either on tool_execution or external_execution_result), no longer needs execution
        if self.external_execution_result is not None:
            return False
        if self.tool_execution.result is not None:
            return False

        return self.tool_execution.external_execution_required or False

    @property
    def is_external_tool_execution(self) -> bool:
        """Check if this requirement is for external tool execution."""
        return self.pause_type == 'external_tool' and self.needs_external_execution

    def confirm(self):
        if not self.needs_confirmation:
            raise ValueError("This requirement does not require confirmation")
        self.confirmation = True
        if self.tool_execution:
            self.tool_execution.confirmed = True

    def reject(self):
        if not self.needs_confirmation:
            raise ValueError("This requirement does not require confirmation")
        self.confirmation = False
        if self.tool_execution:
            self.tool_execution.confirmed = False

    def set_external_execution_result(self, result: str):
        if not self.needs_external_execution:
            raise ValueError("This requirement does not require external execution")
        self.external_execution_result = result
        if self.tool_execution:
            self.tool_execution.result = result

    def update_tool(self):
        if not self.tool_execution:
            return
        if self.confirmation is True:
            self.tool_execution.confirmed = True
        elif self.confirmation is False:
            self.tool_execution.confirmed = False
        else:
            raise ValueError("This requirement does not require confirmation or user input")

    def _serialize_message(self, msg: Any) -> Any:
        """Serialize a message to dict if it has to_dict method."""
        if msg is None:
            return None
        if hasattr(msg, 'to_dict'):
            return msg.to_dict()
        return msg
    
    def is_resolved(self) -> bool:
        """Return True if the requirement has been resolved"""
        # If explicitly marked as resolved via resolved_at, it's resolved
        if self.resolved_at is not None:
            return True
        # For external tool requirements, check tool needs
        if self.pause_type == 'external_tool':
            return not self.needs_confirmation and not self.needs_user_input and not self.needs_external_execution
        # For other pause types (durable_execution, cancel), they're not resolved until explicitly marked
        return False
    
    def mark_resolved(self) -> None:
        """
        Mark the requirement as resolved.
        
        Sets resolved_at timestamp. For external tool requirements,
        ensure tool_execution.result is set before calling this.
        """
        self.resolved_at = datetime.now(timezone.utc)
    
    def set_continuation_data(
        self,
        messages: List[Any],
        response: Any,
        agent_state: Dict[str, Any],
        step_result: "StepResult",
        execution_stats: "PipelineExecutionStats"
    ) -> None:
        """
        Store continuation data for resuming execution.
        
        Args:
            messages: Full message history at pause point
            response: Model response that triggered pause (if any)
            agent_state: Agent internal state (tool_call_count, etc.)
            step_result: StepResult of the step where pause occurred
            execution_stats: Pipeline execution statistics at pause point
        """
        self.continuation_messages = messages
        self.continuation_response = response
        self.agent_state = agent_state
        self.step_result = step_result
        self.execution_stats = execution_stats
    
    def get_continuation_data(self) -> Tuple[List[Any], Any, Dict[str, Any]]:
        """
        Retrieve continuation data for resuming execution.
        
        Returns:
            Tuple of (messages, response, agent_state)
        """
        messages = self.continuation_messages or []
        response = self.continuation_response
        agent_state = self.agent_state or {}
        return messages, response, agent_state

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (pure transformation, nested objects call their to_dict())."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "confirmation": self.confirmation,
            "confirmation_note": self.confirmation_note,
            "external_execution_result": self.external_execution_result,
            "pause_type": self.pause_type,
            "agent_state": self.agent_state,
            "user_input_schema": self.user_input_schema,
            "step_result": self.step_result.to_dict() if self.step_result else None,
            "execution_stats": self.execution_stats.to_dict() if self.execution_stats else None,
            "tool_execution": self.tool_execution.to_dict() if self.tool_execution else None,
            "continuation_messages": [self._serialize_message(m) for m in self.continuation_messages] if self.continuation_messages else None,
            "continuation_response": self._serialize_message(self.continuation_response),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRequirement":
        """Reconstruct from dictionary."""
        if data is None:
            raise ValueError("RunRequirement.from_dict() requires a non-None dict")

        from upsonic.agent.pipeline.step import StepResult
        from upsonic.run.pipeline.stats import PipelineExecutionStats

        # Handle tool_execution (dict only)
        tool_data = data.get("tool_execution")
        tool_execution: Optional[ToolExecution] = None
        if isinstance(tool_data, dict):
            tool_execution = ToolExecution.from_dict(tool_data)

        # Handle created_at (ISO string from dict)
        created_at_raw = data.get("created_at")
        created_at: Optional[datetime] = None
        if isinstance(created_at_raw, str):
            created_at = datetime.fromisoformat(created_at_raw)
        elif isinstance(created_at_raw, datetime):
            created_at = created_at_raw
        
        # Handle resolved_at (ISO string from dict)
        resolved_at_raw = data.get("resolved_at")
        resolved_at: Optional[datetime] = None
        if isinstance(resolved_at_raw, str):
            resolved_at = datetime.fromisoformat(resolved_at_raw)
        elif isinstance(resolved_at_raw, datetime):
            resolved_at = resolved_at_raw

        # Handle step_result
        step_result = None
        step_result_data = data.get("step_result")
        if step_result_data:
            step_result = StepResult.from_dict(step_result_data)
        
        # Handle execution_stats
        execution_stats = None
        execution_stats_data = data.get("execution_stats")
        if execution_stats_data:
            execution_stats = PipelineExecutionStats.from_dict(execution_stats_data)

        # Build requirement
        requirement = cls(
            tool_execution=tool_execution,
            id=data.get("id"),
            created_at=created_at,
            pause_type=data.get("pause_type"),
        )
        
        # Set optional fields
        requirement.resolved_at = resolved_at
        requirement.confirmation = data.get("confirmation")
        requirement.confirmation_note = data.get("confirmation_note")
        requirement.external_execution_result = data.get("external_execution_result")
        requirement.agent_state = data.get("agent_state")
        requirement.user_input_schema = data.get("user_input_schema")
        requirement.step_result = step_result
        requirement.execution_stats = execution_stats
        
        # Handle continuation_messages (keep as-is since complex types preserved by cloudpickle)
        requirement.continuation_messages = data.get("continuation_messages")
        requirement.continuation_response = data.get("continuation_response")

        return requirement
    
    def serialize(self) -> bytes:
        """Serialize to bytes for storage."""
        return cloudpickle.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "RunRequirement":
        """Deserialize from bytes."""
        dict_data = cloudpickle.loads(data)
        return cls.from_dict(dict_data)