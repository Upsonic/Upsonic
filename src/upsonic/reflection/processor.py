"""
Reflection processor for implementing self-evaluation logic.
"""

import asyncio
from typing import Any, Optional, TYPE_CHECKING

from .models import (
    ReflectionConfig, ReflectionState, EvaluationResult, 
    ReflectionAction, ReflectionPrompts
)

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task
    from upsonic.agent.agent import Agent


class ReflectionProcessor:
    """Processor for reflection-based self-evaluation and improvement."""
    
    def __init__(self, config: ReflectionConfig):
        self.config = config
    
    async def process_with_reflection(
        self, 
        agent: "Agent", 
        task: "Task", 
        initial_response: Any
    ) -> Any:
        """
        Process a task response through reflection and improvement cycles.
        
        Args:
            agent: The main agent that generated the response
            task: The task being processed
            initial_response: The initial response to evaluate and improve
            
        Returns:
            The final improved response
        """
        # Initialize reflection state
        state = ReflectionState()
        
        # Convert response to string for evaluation
        current_response = self._extract_response_text(initial_response)
        
        # Log reflection start
        from upsonic.utils.printing import reflection_started
        reflection_started(iteration=state.iteration + 1, max_iterations=self.config.max_iterations)
        
        # Create evaluator agent
        evaluator_agent = self._create_evaluator_agent(agent)
        
        while state.should_continue(self.config):
            # Evaluate current response
            evaluation = await self._evaluate_response(
                evaluator_agent, task, current_response, state
            )
            
            # Log evaluation results
            from upsonic.utils.printing import reflection_evaluation
            action_str = evaluation.action.value if hasattr(evaluation.action, 'value') else str(evaluation.action)
            reflection_evaluation(
                iteration=state.iteration + 1,
                overall_score=evaluation.overall_score,
                accuracy=evaluation.criteria.accuracy,
                completeness=evaluation.criteria.completeness,
                relevance=evaluation.criteria.relevance,
                clarity=evaluation.criteria.clarity,
                action=action_str,
                confidence=evaluation.confidence
            )
            
            # Add to state
            state.add_evaluation(current_response, evaluation)
            
            # Check if we should accept the response
            if evaluation.overall_score >= self.config.acceptance_threshold:
                state.final_response = current_response
                state.terminated_reason = "acceptance_threshold_met"
                break
            
            # Check if we should continue based on action
            if evaluation.action == ReflectionAction.ACCEPT:
                state.final_response = current_response
                state.terminated_reason = "evaluator_accepted"
                break
            elif evaluation.action == ReflectionAction.CLARIFY:
                state.final_response = current_response
                state.terminated_reason = "clarification_needed"
                break
            
            # Generate improved response
            if evaluation.action in [ReflectionAction.REVISE, ReflectionAction.RETRY]:
                # Log improvement start
                from upsonic.utils.printing import reflection_improvement_started
                reflection_improvement_started(
                    iteration=state.iteration + 1,
                    feedback=evaluation.feedback
                )
                
                improved_response = await self._generate_improved_response(
                    agent, task, current_response, evaluation, state
                )
                
                if improved_response:
                    current_response = self._extract_response_text(improved_response)
                else:
                    # If improvement failed, accept current response
                    state.final_response = current_response
                    state.terminated_reason = "improvement_failed"
                    break
        
        # Handle max iterations reached
        if not state.final_response:
            state.final_response = current_response
            state.terminated_reason = "max_iterations_reached"
        
        # Log reflection completion
        from upsonic.utils.printing import reflection_completed
        final_score = state.evaluations[-1].overall_score if state.evaluations else 0.0
        reflection_completed(
            final_score=final_score,
            total_iterations=len(state.evaluations),
            termination_reason=state.terminated_reason
        )
        
        # Convert back to original response format
        return self._convert_to_response_format(state.final_response, initial_response, task)
    
    def _create_evaluator_agent(self, main_agent: "Agent") -> "Agent":
        """Create an evaluator agent for reflection."""
        from upsonic.agent.agent import Agent
        
        evaluator_model = self.config.evaluator_model or main_agent.model
        
        return Agent(
            model=evaluator_model,
            name=f"{main_agent.name}_Evaluator" if main_agent.name else "Evaluator",
            role="Response Evaluator",
            goal="Evaluate and provide feedback on AI responses",
            instructions=(
                "You are an expert evaluator. Your role is to critically assess AI responses "
                "for accuracy, completeness, relevance, and clarity. Provide honest, constructive "
                "feedback that helps improve response quality."
            ),
            debug=main_agent.debug
        )
    
    async def _evaluate_response(
        self, 
        evaluator: "Agent", 
        task: "Task", 
        response: str,
        state: ReflectionState
    ) -> EvaluationResult:
        """Evaluate a response using the evaluator agent."""
        from upsonic.tasks.tasks import Task
        
        # Build context for evaluation
        context = self._build_evaluation_context(task, state)
        
        # Create evaluation prompt
        evaluation_prompt = ReflectionPrompts.EVALUATION_PROMPT.format(
            task_description=task.description,
            response=response,
            context=context
        )
        
        # Create evaluation task
        eval_task = Task(
            description=evaluation_prompt,
            response_format=EvaluationResult,
            not_main_task=True
        )
        
        try:
            # Get evaluation from evaluator agent
            evaluation = await evaluator.do_async(eval_task)
            return evaluation
        except Exception as e:
            # Fallback evaluation if agent fails
            return self._create_fallback_evaluation(response, str(e))
    
    async def _generate_improved_response(
        self, 
        agent: "Agent", 
        task: "Task", 
        previous_response: str,
        evaluation: EvaluationResult,
        state: ReflectionState
    ) -> Any:
        """Generate an improved response based on evaluation feedback."""
        from upsonic.tasks.tasks import Task
        
        # Build context for improvement
        context = self._build_improvement_context(task, state)
        
        # Create improvement prompt
        improvement_prompt = ReflectionPrompts.IMPROVEMENT_PROMPT.format(
            task_description=task.description,
            previous_response=previous_response,
            feedback=evaluation.feedback,
            improvements="\n".join(f"- {imp}" for imp in evaluation.suggested_improvements),
            context=context
        )
        
        # Create improvement task with same format as original
        improved_task = Task(
            description=improvement_prompt,
            response_format=task.response_format,
            tools=task.tools,
            attachments=task.attachments,
            not_main_task=True
        )
        
        try:
            # Generate improved response
            return await agent.do_async(improved_task)
        except Exception:
            # Return None if improvement fails
            return None
    
    def _extract_response_text(self, response: Any) -> str:
        """Extract text representation from response."""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'model_dump_json'):
            return response.model_dump_json()
        elif hasattr(response, '__str__'):
            return str(response)
        else:
            return repr(response)
    
    def _convert_to_response_format(self, response_text: str, original_response: Any, task: "Task") -> Any:
        """Convert improved response back to original format."""
        # If original was string, return string
        if isinstance(original_response, str):
            return response_text
        
        # If task expects structured format, try to parse
        if hasattr(task.response_format, 'model_validate_json'):
            try:
                return task.response_format.model_validate_json(response_text)
            except Exception:
                pass
        
        # Fallback to string
        return response_text
    
    def _build_evaluation_context(self, task: "Task", state: ReflectionState) -> str:
        """Build context for evaluation."""
        context_parts = []
        
        if task.context:
            context_parts.append("Task Context:")
            for i, ctx in enumerate(task.context):
                context_parts.append(f"{i+1}. {str(ctx)}")
        
        if state.iteration > 0:
            context_parts.append(f"\nThis is iteration {state.iteration + 1} of the reflection process.")
            context_parts.append("Previous evaluations have been conducted.")
        
        return "\n".join(context_parts) if context_parts else "No additional context."
    
    def _build_improvement_context(self, task: "Task", state: ReflectionState) -> str:
        """Build context for improvement."""
        context_parts = []
        
        if task.context:
            context_parts.append("Original Task Context:")
            for i, ctx in enumerate(task.context):
                context_parts.append(f"{i+1}. {str(ctx)}")
        
        context_parts.append(f"\nImprovement Iteration: {state.iteration}")
        
        if len(state.evaluations) > 1:
            context_parts.append("Previous improvement attempts have been made.")
            context_parts.append("Focus on addressing the core issues identified.")
        
        return "\n".join(context_parts)
    
    def _create_fallback_evaluation(self, response: str, error: str) -> EvaluationResult:
        """Create a fallback evaluation when the evaluator fails."""
        from .models import EvaluationCriteria
        
        return EvaluationResult(
            criteria=EvaluationCriteria(
                accuracy=0.5,
                completeness=0.5, 
                relevance=0.5,
                clarity=0.5
            ),
            overall_score=0.5,
            feedback=f"Evaluation failed due to error: {error}. Accepting response with medium confidence.",
            suggested_improvements=["Review response manually for accuracy and completeness"],
            action=ReflectionAction.ACCEPT,
            confidence=0.3
        )
