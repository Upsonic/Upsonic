from __future__ import annotations
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """
    Represents a single, concrete tool call in a high-level plan.

    The LLM provides a sequence of these steps, and the orchestrator
    executes them.
    """
    tool_name: str = Field(
        ..., 
        description="The exact name of the tool to be called for this step."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="The dictionary of parameters to pass to the tool. Can be an empty dictionary if the tool takes no parameters."
    )


class AnalysisResult(BaseModel):
    """
    Represents the structured output of an automated 'analysis' step.

    After a tool is executed, the agent is prompted for this object to decide
    what to do next based on the outcome.
    """
    evaluation: str = Field(
        ...,
        description="The detailed reasoning and evaluation of the last tool's result in the context of the overall goal."
    )
    next_action: Literal['continue_plan', 'revise_plan', 'final_answer'] = Field(
        ...,
        description="The explicit directive for the orchestrator. 'continue_plan' proceeds to the next tool, 'revise_plan' triggers a new planning phase, and 'final_answer' moves to synthesis."
    )


class Thought(BaseModel):
    """
    Represents the initial structured thinking process for the AI agent.

    This model serves as the blueprint for the 'plan_and_execute' tool,
    containing the high-level sequence of tool calls to be attempted.
    """

    reasoning: str = Field(
        ...,
        description="The 'inner monologue' of the agent. A detailed explanation of its understanding of the user's request and its high-level strategy."
    )

    plan: List[PlanStep] = Field(
        ...,
        description="A machine-readable, step-by-step execution plan containing only the sequence of tool calls to attempt."
    )

    criticism: str = Field(
        ...,
        description="A mandatory self-critique of the formulated plan. The agent must identify potential flaws or ambiguities in the user's request."
    )

    action: Literal['execute_plan', 'request_clarification'] = Field(
        ...,
        description="The explicit next action to take. Always 'execute_plan' unless user clarification is required before any tools can be called."
    )