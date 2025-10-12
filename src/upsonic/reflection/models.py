"""
Reflection models for self-evaluation and improvement.

This module provides classes for implementing reflection logic where an evaluator LLM
evaluates the main LLM's response and provides feedback for improvement.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ReflectionAction(str, Enum):
    """Action to take based on evaluation."""
    ACCEPT = "accept"          # Accept the current response
    REVISE = "revise"          # Revise the response based on feedback
    RETRY = "retry"            # Retry with different approach
    CLARIFY = "clarify"        # Need clarification from user


class EvaluationCriteria(BaseModel):
    """Criteria for evaluating responses."""
    accuracy: float = Field(description="Accuracy of the information (0-1)")
    completeness: float = Field(description="Completeness of the response (0-1)")
    relevance: float = Field(description="Relevance to the task (0-1)")
    clarity: float = Field(description="Clarity and readability (0-1)")
    
    def overall_score(self) -> float:
        """Calculate overall evaluation score."""
        return (self.accuracy + self.completeness + self.relevance + self.clarity) / 4.0


class EvaluationResult(BaseModel):
    """Result of response evaluation."""
    criteria: EvaluationCriteria
    overall_score: float = Field(description="Overall evaluation score (0-1)")
    feedback: str = Field(description="Detailed feedback for improvement")
    suggested_improvements: List[str] = Field(description="Specific improvement suggestions")
    action: ReflectionAction = Field(description="Recommended action")
    confidence: float = Field(description="Confidence in evaluation (0-1)")
    
    def __post_init__(self):
        """Calculate overall score after initialization."""
        self.overall_score = self.criteria.overall_score()


class ReflectionConfig(BaseModel):
    """Configuration for reflection process."""
    max_iterations: int = Field(default=3, description="Maximum reflection iterations")
    acceptance_threshold: float = Field(default=0.8, description="Minimum score to accept response")
    evaluator_model: Optional[str] = Field(default=None, description="Model for evaluation")
    enable_self_critique: bool = Field(default=True, description="Enable self-critique")
    enable_improvement_suggestions: bool = Field(default=True, description="Enable improvement suggestions")
    

class ReflectionState(BaseModel):
    """State tracking for reflection process."""
    iteration: int = Field(default=0, description="Current iteration number")
    evaluations: List[EvaluationResult] = Field(default_factory=list, description="History of evaluations")
    responses: List[str] = Field(default_factory=list, description="History of responses")
    final_response: Optional[str] = Field(default=None, description="Final accepted response")
    terminated_reason: Optional[str] = Field(default=None, description="Reason for termination")
    
    def add_evaluation(self, response: str, evaluation: EvaluationResult):
        """Add an evaluation result to the state."""
        self.responses.append(response)
        self.evaluations.append(evaluation)
        self.iteration += 1
    
    def get_latest_evaluation(self) -> Optional[EvaluationResult]:
        """Get the most recent evaluation."""
        return self.evaluations[-1] if self.evaluations else None
    
    def should_continue(self, config: ReflectionConfig) -> bool:
        """Check if reflection should continue."""
        if self.iteration >= config.max_iterations:
            return False
        
        latest_eval = self.get_latest_evaluation()
        if latest_eval and latest_eval.overall_score >= config.acceptance_threshold:
            return False
        
        return True


class ReflectionPrompts:
    """Prompt templates for reflection process."""
    
    EVALUATION_PROMPT = """
    You are an expert evaluator. Evaluate the following response based on the given task and criteria.
    
    TASK: {task_description}
    
    RESPONSE TO EVALUATE: {response}
    
    Please evaluate this response on the following criteria (scale 0-1):
    1. Accuracy: How accurate is the information provided?
    2. Completeness: How complete is the response in addressing the task?
    3. Relevance: How relevant is the response to the specific task?
    4. Clarity: How clear and well-structured is the response?
    
    Provide specific feedback and actionable suggestions for improvement.
    Consider the context and requirements carefully.
    
    Additional Context:
    {context}
    """
    
    IMPROVEMENT_PROMPT = """
    Based on the evaluation feedback, improve your previous response.
    
    ORIGINAL TASK: {task_description}
    
    PREVIOUS RESPONSE: {previous_response}
    
    EVALUATION FEEDBACK: {feedback}
    
    SPECIFIC IMPROVEMENTS NEEDED:
    {improvements}
    
    Please provide an improved response that addresses the feedback while maintaining accuracy and relevance.
    Focus specifically on the areas identified for improvement.
    
    Additional Context:
    {context}
    """
    
    SELF_CRITIQUE_PROMPT = """
    Review and critique your own response to identify potential issues or areas for improvement.
    
    TASK: {task_description}
    
    YOUR RESPONSE: {response}
    
    Please analyze your response critically and identify:
    1. Any potential inaccuracies or errors
    2. Missing information or incomplete coverage
    3. Areas where clarity could be improved
    4. Relevance issues or off-topic content
    
    Be honest and thorough in your self-assessment.
    
    Context: {context}
    """
