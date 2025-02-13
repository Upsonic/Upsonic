from copy import deepcopy
from typing import Any, Optional, Union, Type
from .client.tasks.tasks import Task

from .client.agent_configuration.agent_configuration import AgentConfiguration

from pydantic import Field

from .client.tasks.task_response import ObjectResponse


class ValidationResult(ObjectResponse):
    result: bool
    feedback: str = Field(description="Feedback on the result if it is incorrect. Only fill this field if the result is incorrect")


class ReliabilityProcessor:
    @staticmethod
    def process_result(
        result: Any, 
        reliability_layer: Optional[Any] = None, 
        task: Optional[Task] = None,
        llm_model: Optional[str] = None
    ) -> Any:
        """
        Process the result based on reliability layer settings.
        If reliability_layer is None or fields are 0, return original result.
        
        Args:
            result: The result to process
            reliability_layer: Configuration for reliability checks (can be instance or class)
            task: The original task that generated this result
            llm_model: The LLM model used to generate the result
        """
        if reliability_layer is None:
            return result

        # Get prevent_hallucination value
        prevent_hallucination = getattr(reliability_layer, 'prevent_hallucination', 0)
        if isinstance(prevent_hallucination, property):
            prevent_hallucination = prevent_hallucination.fget(reliability_layer)

        processed_result = result

        # Check prevent_hallucination
        if prevent_hallucination > 0:

            if prevent_hallucination == 10:
                validator_agent = AgentConfiguration("Information Validator Agent", model=llm_model, sub_task=False)
                copy_task = deepcopy(task)
                copy_task._response = result
                validator_task = Task("Evaluate the old question and its answer for correctness, consistency, and completeness. Also thing for the requested output. Do not allow unverified information and guesses.", context=[copy_task, copy_task.response_format], response_format=ValidationResult, tools=task.tools)
                validator_agent.do(validator_task)

                if validator_task.response.result == True:
                    return result
                else:
                    editor_agent = AgentConfiguration("Information Editor Agent", model=llm_model, sub_task=False)
                    editor_task = Task("Edit the answer to the question to correct the inaccuracies by the feedback provided", context=[copy_task, copy_task.response_format, validator_task], response_format=task.response_format, tools=task.tools)
                    editor_agent.do(editor_task)

                    return editor_task.response



        return processed_result 