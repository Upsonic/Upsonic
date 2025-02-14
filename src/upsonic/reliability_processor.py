from copy import deepcopy
from typing import Any, Optional, Union, Type
from .client.tasks.tasks import Task

from .client.agent_configuration.agent_configuration import AgentConfiguration

from pydantic import Field

from .client.tasks.task_response import ObjectResponse


class ValidationResult(ObjectResponse):
    any_suspicion: bool
    feedback: str

validator_task_prompt = """
Evaluate the previous question and its answer for accuracy, consistency, and completeness. Also consider the requested output format. Be extremely vigilant for any non-referenced values or unsupported claims. Pay particular attention to:

Unverified information: Identify and flag any claims, statistics, or statements that lack credible sources or proper citations.

Unsupported assertions: Watch for any information presented as fact without adequate evidence or references.

Vague attributions: Be wary of phrases like "Studies show..." or "Experts say..." without specific, verifiable sources.

Misuse of data: Check for any misinterpretation or misapplication of statistical information.

Temporal inconsistencies: Ensure all historical claims or dates are properly verified and sourced.

Speculative content: Identify any conjectures or hypothetical scenarios presented without clear indication of their speculative nature.

Examples of issues to flag:

Any statistic or numerical claim without a specific, credible source
Historical events or dates mentioned without verifiable references
Scientific claims or breakthroughs without links to peer-reviewed research
Expert opinions quoted without naming the expert and their credentials
Trends or patterns described without supporting data
Do not allow any unverified information, guesses, or assumptions to pass unchallenged. Rigorously check all data and its sources. If there is any information without a valuable and meaningful reference, highlight it for removal or verification. Ensure all information is factual, current, and properly supported by credible, cited sources.

"""


editor_task_prompt = """
As the editor agent, your task is to review and refine the output provided. Pay close attention to the issues highlighted in the previous evaluation. Your primary responsibilities are:

Remove or correct any misunderstandings, unverified information, or hallucinations identified in the output.

Ensure all claims are supported by credible, up-to-date sources.

Eliminate any statements that present personal opinions as facts.

Verify all statistical data, historical dates, and scientific claims.

Check for and remove any mentions of non-existent technologies, people, or places.

Maintain the requested output format while improving accuracy and reliability.

Remember, you should not return any data that doesn't meet these high standards of verification and accuracy. If you cannot verify a piece of information or find a reliable source for it, remove it from the output.

Your goal is to produce a refined version of the output that is factually accurate, consistently reliable, and complete within the bounds of verified information. Quality and accuracy are paramount - it's better to have less information that is fully verified than more information that includes questionable data.

"""

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
                validator_task = Task(validator_task_prompt, context=[copy_task, copy_task.response_format], response_format=ValidationResult, tools=task.tools)
                validator_agent.do(validator_task)

                if validator_task.response.any_suspicion == False:
                    return result
                else:
                    editor_agent = AgentConfiguration("Information Editor Agent", model=llm_model, sub_task=False)
                    editor_task = Task(editor_task_prompt, context=[copy_task, copy_task.response_format, validator_task, str(validator_task.response.model_dump(mode="json"))], response_format=task.response_format, tools=task.tools)
                    editor_agent.do(editor_task)

                    return editor_task.response



        return processed_result 