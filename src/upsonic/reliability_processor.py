from copy import deepcopy
from typing import Any, Optional, Union, Type, List
from pydantic import BaseModel, Field
from enum import Enum
import re
from urllib.parse import urlparse
import requests
from .client.tasks.tasks import Task
from .client.agent_configuration.agent_configuration import AgentConfiguration
from .client.tasks.task_response import ObjectResponse

# Define the validation prompts
url_validation_prompt = """
Focus on basic URL source validation:

Source Verification:
- Check if the URL is from a verified source
- Verify if the source is official
- Check if the source is appropriate for the content. But dont make assumption just check the context and try to find exact things. If not flag it.

IMPORTANT: If the URL source cannot be verified, flag it as suspicious.
"""

number_validation_prompt = """
Focus on basic numerical validation:

Number Verification:
- Check if numbers are Valid
- Verify if units are appropriate
- Check if numbers make logical sense in context
- Dont check lenghts or other calculations needed verifications
- If the numbers are come from original context thats okay no need to verify these numbers.

IMPORTANT: If the numbers cannot be verified, flag them as suspicious.
"""

code_validation_prompt = """
Focus on basic code validation:

Code Verification:
- Check if security codes, API keys, Defination Codes, or tokens follow proper formats
- Verify if code snippets are from trusted sources
- Check for potential security vulnerabilities or malicious code
- Validate code syntax and structure
- Check for sensitive information exposure
- Verify if code references are appropriate and safe

IMPORTANT: If the code cannot be verified or appears suspicious, flag it as suspicious.
"""

information_validation_prompt = """
Focus on basic information validation:

Information Verification:
- Check if the information is factually correct
- Verify if the information is consistent
- Check if the information is relevant to the context
- Dont check lenghts or other calculations needed verifications

IMPORTANT: If the information cannot be verified, flag it as suspicious.
"""

editor_task_prompt = """
Clean and validate the output by handling suspicious content:

Processing Rules:
1. For ANY suspicious content identified in validation:
- Replace the suspicious value with None
- Do not suggest alternatives
- Do not provide explanations
- Do not modify other parts of the content

2. For non-suspicious content:
- Keep the original value unchanged
- Do not enhance or modify
- Do not add additional information

Processing Steps:
- Set suspicious fields to None
- Keep other fields as is
- Remove any suspicious content entirely
- Maintain original structure

Validation Issues Found:
{validation_feedback}

IMPORTANT:
- Set ALL suspicious values to None
- Keep verified values unchanged
- No explanations or suggestions
- No partial validations
- Maintain response format
"""

class SourceReliability(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class ValidationPoint(ObjectResponse):
    is_suspicious: bool
    feedback: str
    suspicious_points: list[str] = Field(description = "Suspicious informations raw name")
    source_reliability: SourceReliability = SourceReliability.UNKNOWN
    verification_method: str = ""
    confidence_score: float = 0.0

class ValidationResult(ObjectResponse):
    url_validation: ValidationPoint
    number_validation: ValidationPoint
    information_validation: ValidationPoint
    code_validation: ValidationPoint
    any_suspicion: bool
    suspicious_points: list[str]
    overall_feedback: str
    overall_confidence: float = 0.0

    def calculate_suspicion(self) -> str:
        self.any_suspicion = any([
            self.url_validation.is_suspicious,
            self.number_validation.is_suspicious,
            self.information_validation.is_suspicious,
            self.code_validation.is_suspicious
        ])

        self.suspicious_points = []
        validation_details = []

        # Collect URL validation details
        if self.url_validation.is_suspicious:
            self.suspicious_points.extend(self.url_validation.suspicious_points)
            validation_details.append(f"URL Issues: {self.url_validation.feedback}")
            validation_details.extend([f"- {point}" for point in self.url_validation.suspicious_points])

        # Collect number validation details
        if self.number_validation.is_suspicious:
            self.suspicious_points.extend(self.number_validation.suspicious_points)
            validation_details.append(f"Number Issues: {self.number_validation.feedback}")
            validation_details.extend([f"- {point}" for point in self.number_validation.suspicious_points])
            
        # Collect information validation details
        if self.information_validation.is_suspicious:
            self.suspicious_points.extend(self.information_validation.suspicious_points)
            validation_details.append(f"Information Issues: {self.information_validation.feedback}")
            validation_details.extend([f"- {point}" for point in self.information_validation.suspicious_points])

        # Collect code validation details
        if self.code_validation.is_suspicious:
            self.suspicious_points.extend(self.code_validation.suspicious_points)
            validation_details.append(f"Code Issues: {self.code_validation.feedback}")
            validation_details.extend([f"- {point}" for point in self.code_validation.suspicious_points])

        # Calculate overall confidence
        self.overall_confidence = sum([
            self.url_validation.confidence_score,
            self.number_validation.confidence_score,
            self.information_validation.confidence_score,
            self.code_validation.confidence_score
        ]) / 4.0

        # Generate overall feedback
        if validation_details:
            self.overall_feedback = "\n".join(validation_details)
        else:
            self.overall_feedback = "No suspicious content detected."

        # Return complete validation summary for editor
        validation_summary = [
            "Validation Summary:",
            f"Overall Confidence: {self.overall_confidence:.2f}",
            f"Suspicious Content Detected: {'Yes' if self.any_suspicion else 'No'}",
            "\nDetailed Feedback:",
            self.overall_feedback
        ]
        
        return "\n".join(validation_summary)

class ReliabilityProcessor:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def process_result(
        result: Any,
        reliability_layer: Optional[Any] = None,
        task: Optional[Task] = None,
        llm_model: Optional[str] = None
    ) -> Any:
        if reliability_layer is None:
            return result
    
        old_task_output = result
        try:
            old_task_output = result.model_dump()
        except:
            pass

        prevent_hallucination = getattr(reliability_layer, 'prevent_hallucination', 0)
        if isinstance(prevent_hallucination, property):
            prevent_hallucination = prevent_hallucination.fget(reliability_layer)

        processed_result = result

        if prevent_hallucination > 0:
            if prevent_hallucination == 10:
                copy_task = deepcopy(task)
                copy_task._response = result

                validation_result = ValidationResult(
                    url_validation=ValidationPoint(
                        is_suspicious=False, 
                        feedback="",
                        suspicious_points=[],
                        source_reliability=SourceReliability.UNKNOWN,
                        verification_method="",
                        confidence_score=0.0
                    ),
                    number_validation=ValidationPoint(
                        is_suspicious=False, 
                        feedback="",
                        suspicious_points=[],
                        source_reliability=SourceReliability.UNKNOWN,
                        verification_method="",
                        confidence_score=0.0
                    ),
                    information_validation=ValidationPoint(
                        is_suspicious=False, 
                        feedback="",
                        suspicious_points=[],
                        source_reliability=SourceReliability.UNKNOWN,
                        verification_method="",
                        confidence_score=0.0
                    ),
                    code_validation=ValidationPoint(
                        is_suspicious=False, 
                        feedback="",
                        suspicious_points=[],
                        source_reliability=SourceReliability.UNKNOWN,
                        verification_method="",
                        confidence_score=0.0
                    ),
                    any_suspicion=False,
                    suspicious_points=[],
                    overall_feedback=""
                )

                # Run validations
                for validation_type, prompt in [
                    ("url_validation", url_validation_prompt),
                    ("number_validation", number_validation_prompt),
                    ("information_validation", information_validation_prompt),
                    ("code_validation", code_validation_prompt),
                ]:
                    validator_agent = AgentConfiguration(
                        f"{validation_type.replace('_', ' ').title()} Agent",
                        model=llm_model,
                        sub_task=False
                    )
                    the_context = [copy_task, copy_task.response_format]
                    the_context += copy_task.context

                    prompt += f"Current AI Response: {old_task_output}"
                    validator_task = Task(
                        prompt,
                        context=the_context,
                        response_format=ValidationPoint,
                        tools=task.tools
                    )
                    validator_agent.do(validator_task)
                    setattr(validation_result, validation_type, validator_task.response)

                validation_result.calculate_suspicion()

                if validation_result.any_suspicion:
                    editor_agent = AgentConfiguration(
                        "Information Editor Agent",
                        model=llm_model,
                        sub_task=False
                    )
                    formatted_prompt = editor_task_prompt.format(
                        validation_feedback=validation_result.overall_feedback
                    )
                    formatted_prompt += f"OLD AI Response: {old_task_output}"

                    the_context = [copy_task, copy_task.response_format, validation_result]
                    the_context += copy_task.context
                    editor_task = Task(
                        formatted_prompt,
                        context=the_context,
                        response_format=task.response_format,
                        tools=task.tools
                    )
                    editor_agent.do(editor_task)
                    return editor_task.response

                return result

        return processed_result