"""Orchestration and planning tools for complex agent tasks."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

from upsonic.tools.base import Tool
from upsonic.tools.config import tool

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task


class PlanStep(BaseModel):
    """Single tool call in a high-level plan."""
    tool_name: str = Field(
        ..., 
        description="The exact name of the tool to be called for this step."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="The dictionary of parameters to pass to the tool."
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of what this step accomplishes."
    )


class AnalysisResult(BaseModel):
    """Structured output of an automated analysis step."""
    evaluation: str = Field(
        ...,
        description="Detailed reasoning and evaluation of the last tool's result."
    )
    next_action: Literal['continue_plan', 'revise_plan', 'final_answer'] = Field(
        ...,
        description="Agent directive for the orchestrator: continue, revise, or finalize."
    )
    reasoning: Optional[str] = Field(
        None,
        description="Additional reasoning for the chosen next action."
    )


class Thought(BaseModel):
    """Initial structured thinking process for the AI agent."""
    reasoning: str = Field(
        ...,
        description="Detailed explanation of understanding and strategy."
    )
    plan: List[PlanStep] = Field(
        ...,
        description="Step-by-step execution plan of tool calls."
    )
    criticism: str = Field(
        ...,
        description="Self-critique identifying potential flaws or ambiguities."
    )
    action: Literal['execute_plan', 'request_clarification'] = Field(
        'execute_plan',
        description="Next action: execute the plan or request clarification."
    )
    clarification_needed: Optional[str] = Field(
        None,
        description="Specific clarification needed if action is 'request_clarification'."
    )


class ExecutionResult(BaseModel):
    """Result of executing an orchestrated plan."""
    
    success: bool = Field(
        ...,
        description="Whether the plan execution was successful."
    )
    final_result: Any = Field(
        ...,
        description="The final synthesized result."
    )
    execution_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of all tool executions."
    )
    total_steps: int = Field(
        ...,
        description="Total number of steps executed."
    )
    revisions: int = Field(
        0,
        description="Number of plan revisions made."
    )


@tool(
    requires_confirmation=False,
    show_result=False,
    sequential=True,
    docstring_format='google'
)
def plan_and_execute(thought: Thought) -> str:
    """Master tool for complex tasks. Executes multi-step plans sequentially.
    
    Args:
        thought: Structured thought object with reasoning, plan, and criticism.
    
    Returns:
        Placeholder string - actual execution handled by orchestrator.
    """
    # This is a pseudo-tool - actual implementation is in the processor
    return "Plan received and will be executed by the orchestrator."


class Orchestrator(Tool):
    """Orchestrator for complex multi-step tool executions with optional reasoning."""
    
    def __init__(
        self,
        agent_instance: Any,
        task: Optional['Task'],
        wrapped_tools: Dict[str, Callable]
    ):
        """Initialize the orchestrator."""
        # Initialize Tool base class
        super().__init__(
            name="orchestrator",
            description="Orchestrates multi-step tool execution with reasoning",
            tool_id=f"Orchestrator_{id(agent_instance)}"  # Unique per agent instance
        )
        
        self.agent_instance = agent_instance
        self.task = task
        self.wrapped_tools = wrapped_tools
        self.is_reasoning_enabled = agent_instance.enable_reasoning_tool if agent_instance else False
        self.original_user_request = task.description if task else ""
        
        self.execution_history = f"Orchestrator's execution history for the user's request:\n"
        self.program_counter = 0
        self.pending_plan = []
        self.revision_count = 0
        
        self.all_tools = {
            name: func 
            for name, func in wrapped_tools.items()
            if name != 'plan_and_execute'
        }
    
    async def execute(self, thought: Thought) -> Any:
        """Main entry point for orchestrator execution."""
        from upsonic.utils.printing import console, spacing
        
        console.print("[bold magenta]Orchestrator Activated:[/bold magenta] Received initial plan.")
        spacing()
        
        if not self.agent_instance:
            return "Error: Orchestrator was not properly initialized with an agent instance."
        
        self.execution_history += f"Initial Thought & Plan: {thought.plan}\nReasoning: {thought.reasoning}\n Criticism: {thought.criticism}\n\n"
        self.pending_plan = thought.plan
        self.program_counter = 0
        
        while self.program_counter < len(self.pending_plan):
            step = self.pending_plan[self.program_counter]
            
            result = await self._execute_single_step(step)
            
            if self.is_reasoning_enabled:
                should_continue = await self._handle_reasoning_step(step.tool_name, result)
                if not should_continue:
                    break
            else:
                self.program_counter += 1
        
        return await self._synthesize_final_answer()
    
    async def _execute_single_step(self, step: PlanStep) -> Any:
        """Execute a single tool step from the plan."""
        from upsonic.utils.printing import console, print_orchestrator_tool_step
        
        tool_name = step.tool_name.split('.')[-1]
        params = step.parameters
        
        console.print(
            f"[bold blue]Executing Tool Step {self.program_counter + 1}/{len(self.pending_plan)}:[/bold blue] "
            f"Calling tool [cyan]{tool_name}[/cyan] with params {params}"
        )
        
        if tool_name not in self.all_tools:
            result = f"Error: Tool '{tool_name}' is not an available tool."
            console.print(f"[bold red]{result}[/bold red]")
        else:
            try:
                tool_to_call = self.all_tools[tool_name]
                result = await tool_to_call(**params)
            except Exception as e:
                error_message = f"An error occurred while executing tool '{tool_name}': {e}"
                console.print(f"[bold red]{error_message}[/bold red]")
                result = error_message
        
        print_orchestrator_tool_step(tool_name, params, result)
        self.execution_history += f"\nStep {self.program_counter + 1} (Tool: {tool_name}):\nResult: {result}\n"
        
        return result
    
    async def _inject_analysis(self) -> AnalysisResult:
        """Inject mandatory analysis step after tool execution."""
        from upsonic.tasks.tasks import Task
        from upsonic.agent.agent import Agent
        
        analysis_prompt = (
            f"Original user request(This is just for remembrance. You have to follow instructions below based on this. But this is not the main focus you will try to fulfill right now): '{self.original_user_request}'\n\n"
            "You are in the middle of a multi-step plan. An action has just been completed. You must now analyze the outcome before proceeding. "
            "Based on the execution history, evaluate the result of the last tool call and decide the "
            "most logical next action.\n\n"
            "CRITICAL: You are ONLY analyzing the results. DO NOT call any tools. DO NOT execute any actions. "
            "ONLY provide your evaluation based on the execution history below.\n\n"
            "<ExecutionHistory>\n"
            f"{self.execution_history}"
            "</ExecutionHistory>"
        )
        
        # Create analysis task with NO tools - analysis agent should only evaluate, not execute
        analysis_task = Task(
            description=analysis_prompt, 
            not_main_task=True, 
            response_format=AnalysisResult,
            tools=[]  # Explicitly no tools for analysis
        )
        
        # Create a fresh analysis agent with NO tools at all
        # We cannot use copy because it shares tool references
        analysis_agent = Agent(
            model=self.agent_instance.model,
            name=f"{self.agent_instance.name}_analysis",
            tools=[],  # NO tools for analysis
            enable_thinking_tool=False,
            enable_reasoning_tool=False
        )
        
        analysis_run_result = await analysis_agent.do_async(analysis_task)
        analysis_result: AnalysisResult = analysis_run_result.output if hasattr(analysis_run_result, 'output') else analysis_run_result
        self.execution_history += f"\n--- Injected Analysis ---\nEvaluation: {analysis_result.evaluation}\n"
        
        return analysis_result
    
    async def _handle_reasoning_step(self, tool_name: str, result: Any) -> bool:
        """Handle reasoning injection after tool execution."""
        from upsonic.utils.printing import console
        
        console.print(f"[bold yellow]Injecting Mandatory Analysis Step after Tool '{tool_name}'...[/bold yellow]")
        
        analysis_result = await self._inject_analysis()
        
        if analysis_result.next_action == 'continue_plan':
            console.print("[bold green]Analysis complete. Continuing with the original plan.[/bold green]")
            self.program_counter += 1
            return True
        
        elif analysis_result.next_action == 'final_answer':
            console.print("[bold green]Analysis concluded that the task is complete. Proceeding to final synthesis.[/bold green]")
            return False
        
        elif analysis_result.next_action == 'revise_plan':
            console.print("[bold red]Analysis concluded that the plan is flawed. Requesting a new plan.[/bold red]")
            await self._request_plan_revision()
            return True
        
        return True
    
    async def _request_plan_revision(self) -> None:
        """Request revised plan based on execution history."""
        from upsonic.tasks.tasks import Task
        from upsonic.utils.printing import console
        from upsonic.agent.agent import Agent
        
        revision_prompt = (
            f"Original user request(This is just for remembrance. You have to follow instructions below based on this. But this is not the main focus you will try to fulfill right now): '{self.original_user_request}'\n\n"
            "You are in the middle of a multi-step plan. Your own analysis has determined that the "
            "original plan is flawed or insufficient. Based on the *entire* execution history so far, "
            "formulate a new, complete `Thought` object with a better plan to achieve the user's "
            "original goal.\n\n"
            "CRITICAL: You are ONLY creating a new plan. DO NOT call any tools. DO NOT execute any actions. "
            "ONLY provide a revised Thought with a better plan based on the execution history.\n\n"
            "<ExecutionHistory>\n"
            f"{self.execution_history}"
            "</ExecutionHistory>"
        )
        
        # Create revision task with NO tools - revision agent should only plan, not execute
        revision_task = Task(
            description=revision_prompt, 
            not_main_task=True, 
            response_format=Thought,
            tools=[]  # Explicitly no tools for revision
        )
        
        # Create a fresh revision agent with NO tools at all
        revision_agent = Agent(
            model=self.agent_instance.model,
            name=f"{self.agent_instance.name}_revision",
            tools=[],  # NO tools for revision
            enable_thinking_tool=False,
            enable_reasoning_tool=False
        )
        
        revision_run_result = await revision_agent.do_async(revision_task)
        new_thought: Thought = revision_run_result.output if hasattr(revision_run_result, 'output') else revision_run_result
        
        console.print("[bold magenta]Orchestrator:[/bold magenta] Received revised plan. Restarting execution.")
        self.pending_plan = new_thought.plan
        self.program_counter = 0
        self.revision_count += 1
        self.execution_history += f"\n--- PLAN REVISED ---\nNew Reasoning: {new_thought.reasoning}\n"
    
    async def _synthesize_final_answer(self) -> Any:
        """Synthesize final answer based on execution history."""
        from upsonic.tasks.tasks import Task
        from upsonic.utils.printing import console, spacing
        from upsonic.agent.agent import Agent
        
        console.print("[bold magenta]Orchestrator:[/bold magenta] Plan complete. Preparing for final synthesis.")
        spacing()
        
        synthesis_prompt = (
            f"Original user request(This is just for remembrance. You have to follow instructions below based on this. But this is not the main focus you will try to fulfill right now): '{self.original_user_request}'\n\n"
            "You are in the final step of a multi-step task. "
            "You have already executed a plan and gathered all necessary information. "
            "Based *only* on the execution history provided below, synthesize a complete "
            "and final answer for the user's original request.\n\n"
            "CRITICAL: You are ONLY synthesizing the final answer. DO NOT call any tools. DO NOT execute any actions. "
            "ONLY provide a comprehensive summary based on the execution history.\n\n"
            "<ExecutionHistory>\n"
            f"{self.execution_history}"
            "</ExecutionHistory>"
        )
        
        # Create synthesis task with NO tools - synthesis agent should only summarize, not execute
        synthesis_task = Task(
            description=synthesis_prompt, 
            not_main_task=True,
            tools=[]  # Explicitly no tools for synthesis
        )
        
        # Create a fresh synthesis agent with NO tools at all
        synthesis_agent = Agent(
            model=self.agent_instance.model,
            name=f"{self.agent_instance.name}_synthesis",
            tools=[],  # NO tools for synthesis
            enable_thinking_tool=False,
            enable_reasoning_tool=False
        )
        
        final_response = await synthesis_agent.do_async(synthesis_task)
        return final_response.output if hasattr(final_response, 'output') else final_response
    
