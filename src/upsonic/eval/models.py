from typing import List, Dict
from pydantic import BaseModel, Field

class EvaluationScore(BaseModel):
    """
    Represents the structured judgment from the LLM-as-a-judge for a single
    evaluation of an agent's response. This model ensures that the evaluation
    is not just a number but a comprehensive report.
    """
    score: float = Field(
        ...,
        description="A numerical score, on a scale of 1-10, representing the quality and accuracy of the generated response when compared to the expected output and guidelines.",
        ge=1,
        le=10
    )
    
    reasoning: str = Field(
        ...,
        description="The detailed, step-by-step inner monologue of the judge, explaining exactly why the given score was assigned. This should reference the query, expected output, and guidelines."
    )

    is_met: bool = Field(
        ...,
        description="A definitive boolean flag indicating if the generated output successfully meets the core requirements and spirit of the expected output."
    )

    critique: str = Field(
        ...,
        description="Constructive, actionable feedback on how the agent's response could have been improved. If the response was perfect, this can state that no improvements are needed."
    )

class PerformanceRunResult(BaseModel):
    """
    Captures the raw performance metrics from a single execution run of an
    agent, graph, or team. This is the atomic unit of a performance evaluation.
    """
    latency_seconds: float = Field(
        ...,
        description="The total wall-clock time taken for the execution, measured in high-precision seconds."
    )
    
    memory_increase_bytes: int = Field(
        ...,
        description="The net increase in memory allocated by Python objects specifically during this run, measured in bytes. This isolates the memory cost of the operation."
    )

    memory_peak_bytes: int = Field(
        ...,
        description="The peak memory usage recorded at any point during this specific run, relative to the start of the run, measured in bytes."
    )


class ToolCallCheck(BaseModel):
    """
    Represents the verification result for a single expected tool call.
    """

    tool_name: str = Field(
        ...,
        description="The name of the tool that was being checked for."
    )
    
    was_called: bool = Field(
        ...,
        description="A boolean flag that is True if the tool was found in the execution history, otherwise False."
    )

    times_called: int = Field(
        ...,
        description="The total number of times this specific tool was called during the run."
    )


class AccuracyEvaluationResult(BaseModel):
    """
    The final, aggregated result of an accuracy evaluation. This object is
    returned to the user and contains all inputs, outputs, and the
    comprehensive judgments from the evaluation process.
    """
    evaluation_scores: List[EvaluationScore] = Field(
        ...,
        description="A list containing the detailed EvaluationScore object from each iteration of the test."
    )

    average_score: float = Field(
        ...,
        description="The calculated average score from all evaluation iterations."
    )

    user_query: str = Field(
        ...,
        description="The original input query that was provided to the agent under test."
    )

    expected_output: str = Field(
        ...,
        description="The 'gold-standard' or ground-truth answer that was used as a benchmark for the evaluation."
    )

    generated_output: str = Field(
        ...,
        description="The final output that was actually produced by the agent, graph, or team under test."
    )

    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class PerformanceEvaluationResult(BaseModel):
    """
    The final, aggregated report of a performance evaluation. It provides
    meaningful statistics that reveal the stability and characteristics
    of an agent's performance.
    """
    all_runs: List[PerformanceRunResult] = Field(
        ...,
        description="A list containing the raw PerformanceRunResult object from every measured iteration."
    )
    
    num_iterations: int = Field(
        ...,
        description="The number of measurement runs that were performed."
    )
    
    warmup_runs: int = Field(
        ...,
        description="The number of warmup runs that were performed before measurements began."
    )

    latency_stats: Dict[str, float] = Field(
        ...,
        description="A dictionary of key statistical measures for latency (in seconds), including 'average', 'median', 'min', 'max', and 'std_dev'."
    )
    
    memory_increase_stats: Dict[str, float] = Field(
        ...,
        description="A dictionary of statistical measures for the net memory increase (in bytes), including 'average', 'median', 'min', 'max', and 'std_dev'."
    )
    
    memory_peak_stats: Dict[str, float] = Field(
        ...,
        description="A dictionary of statistical measures for the peak memory usage (in bytes), including 'average', 'median', 'min', 'max', and 'std_dev'."
    )

    class Config:
        """Pydantic model configuration."""
        from_attributes = True



class ReliabilityEvaluationResult(BaseModel):
    """
    The final, comprehensive report of a reliability evaluation. It contains
    the overall pass/fail status, diagnostic tool call lists, and detailed
    checks for integrating into automated test suites.
    """
    passed: bool = Field(
        ...,
        description="The overall pass/fail status of the entire reliability check."
    )
    
    summary: str = Field(
        ...,
        description="A human-readable summary explaining the final outcome of the evaluation."
    )

    expected_tool_calls: List[str] = Field(
        ...,
        description="The original list of tool names that the user expected to be called."
    )

    actual_tool_calls: List[str] = Field(
        ...,
        description="The complete, ordered list of tool names that were actually called during the execution."
    )
    
    checks: List[ToolCallCheck] = Field(
        ...,
        description="A detailed list of the check results for each individual expected tool."
    )

    missing_tool_calls: List[str] = Field(
        ...,
        description="A convenience list containing the names of expected tools that were not found in the actual tool calls."
    )
    
    unexpected_tool_calls: List[str] = Field(
        ...,
        description="A list of tools that were called but were not in the expected list. This is only populated if the `exact_match` setting was used."
    )

    class Config:
        """Pydantic model configuration."""
        from_attributes = True

    def assert_passed(self) -> None:
        """
        Raises an AssertionError if the evaluation did not pass.

        This method allows for seamless integration into testing frameworks
        like pytest. If the `passed` attribute is False, an informative
        error is raised.
        """
        if not self.passed:
            raise AssertionError(f"Reliability evaluation failed: {self.summary}")