from __future__ import annotations
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """
    Represents a single, concrete step in an execution plan.

    This model provides a rigid structure for the LLM to follow, ensuring
    that every step in its plan includes the exact tool name and the
    necessary parameters.
    """
    tool_name: str = Field(
        ..., 
        description="The exact name of the tool to be called for this step."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="The dictionary of parameters to pass to the tool. Can be an empty dictionary if the tool takes no parameters."
    )


class Thought(BaseModel):
    """
    Represents a structured thinking process for the AI agent.

    This model serves as a cognitive blueprint, forcing the LLM to deconstruct
    its reasoning, formulate a concrete plan, and critique its own approach
    before taking action.
    """

    reasoning: str = Field(
        ...,
        description="The 'inner monologue' of the agent. A detailed explanation of its understanding of the user's request, the current context, and the rationale behind the chosen plan."
    )

    plan: List[PlanStep] = Field(
        ...,
        description="A machine-readable, step-by-step execution plan. Each item in the list must be a PlanStep object, containing a 'tool_name' and 'parameters'."
    )

    criticism: str = Field(
        ...,
        description="A mandatory self-critique of the formulated plan. The agent must identify potential flaws, ambiguities, or missing information before proceeding. If clarification is needed, this field should explain why."
    )

    action: Literal['execute_plan', 'request_clarification'] = Field(
        ...,
        description="The explicit next action to take. Use 'execute_plan' to begin running the tools in the plan, or 'request_clarification' if user input is required."
    )