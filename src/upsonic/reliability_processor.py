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
Evaluate the previous question and its answer for accuracy, consistency, and completeness, with special attention to URL validation, standard identifier codes, and data type expectations. Also consider the requested output format. Be extremely vigilant for any non-referenced values, unsupported claims, URL-related issues, and invalid identifiers. Pay particular attention to:

URL Validation and AI-Generated URL Assessment:
- Identify any URLs that appear to be AI-generated without proper context or verification
- Flag cases where URLs are presented without evidence of their actual existence
- Check if the generated URLs align with the expected domain structure and patterns
- Look for any assumptions about URL existence without proper verification
- Watch for URLs that might be hypothetical or generated without actual endpoints
- Verify if the context provides any proof or validation of the URL's existence
- Flag URLs that are presented as factual but may be speculative or non-existent
- Pay special attention to URLs that claim to point to specific resources without verification
- Consider whether the URL generation follows logical patterns or is purely speculative

Standard Identifier Validation:
- Verify that all standard identifiers follow their respective format rules and industry standards
- Check for proper length and character composition in identifier codes
- Validate check digits and control characters where applicable
- Ensure identifiers match their described categories and specifications
- Flag any identifier codes that appear malformed or don't match standard patterns
- Verify that custom IDs follow their documented format requirements
- Check for consistency in identifier formatting across related items

Data Type and Format Validation:
- Check if returned data types match user expectations (e.g., human-readable strings vs numeric IDs)
- Flag cases where numeric or ID values are returned instead of expected descriptive strings
- Verify that lists contain appropriate content types (e.g., keywords instead of category IDs)
- Ensure returned data is in a user-friendly format when that is the expectation
- Watch for system identifiers being exposed instead of their human-readable equivalents
- Validate that enumerated items use meaningful labels rather than internal codes
- Check for proper translation of internal representations to user-facing values
- Ensure categorical data is represented with descriptive terms rather than numeric codes
- Flag any response where machine-readable formats overshadow human readability

Unverified information: Identify and flag any claims, statistics, or statements that lack credible sources or proper citations.

Unsupported assertions: Watch for any information presented as fact without adequate evidence, references, or valid URLs.

Vague attributions: Be wary of phrases like "Studies show..." or "Experts say..." without specific, verifiable sources or linked documentation.

Misuse of data: Check for any misinterpretation or misapplication of statistical information and ensure data sources are properly linked.

Temporal inconsistencies: Ensure all historical claims or dates are properly verified and sourced with appropriate references.

Speculative content: Identify any conjectures or hypothetical scenarios presented without clear indication of their speculative nature.

Examples of issues to flag:

- AI-generated URLs without verification of their existence
- URLs presented without proper context or validation
- Generated URLs that make assumptions about resource locations
- Numeric IDs returned where descriptive strings were expected
- Lists containing internal codes instead of human-readable values
- Category identifiers without corresponding descriptive names
- System-level identifiers exposed in user-facing responses
- Machine-readable formats when human-readable was requested
- Raw database IDs instead of meaningful descriptions
- AI-generated URLs without verification of their existence
- URLs presented without proper context or validation
- Generated URLs that make assumptions about resource locations
- URLs that appear to be hypothetical or speculative
- Links claiming to point to specific resources without verification
- Any URL patterns that cannot be confirmed as valid
- Any URL that appears incomplete, malformed, or suspicious
- Links pointing to non-existent or inaccessible resources
- URLs that don't match the context or appear randomly inserted
- Invalid or improperly formatted standard identifiers
- Inconsistent or non-standard identifier formats
- Missing validation elements in standard codes
- Any statistic or numerical claim without a specific, credible source or valid URL
- Historical events or dates mentioned without verifiable references
- Scientific claims or breakthroughs without links to peer-reviewed research
- Expert opinions quoted without naming the expert and their credentials
- Trends or patterns described without supporting data and proper citations

Do not allow any unverified information, guesses, assumptions, invalid URLs, or improperly formatted identifier codes to pass unchallenged. Be especially careful with AI-generated URLs that lack verification of their actual existence and with data types that don't match user expectations. Watch for cases where internal identifiers or codes are returned instead of human-readable values. Rigorously check all data, sources, URLs, and standard codes. If there is any information without a valuable and meaningful reference, or if URLs or identifier codes appear to be generated without verification, or if the response format doesn't match user expectations, highlight it for removal or verification. Ensure all information is factual, current, properly supported by credible, cited sources with verified URLs, and presented in appropriate, user-friendly formats.

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