from copy import deepcopy
from typing import Any, Optional, Union, Type, List
from pydantic import BaseModel
from enum import Enum
import re
from urllib.parse import urlparse
import requests
from .client.tasks.tasks import Task
from .client.agent_configuration.agent_configuration import AgentConfiguration
from .client.tasks.task_response import ObjectResponse

# Define the validation prompts
url_validation_prompt = """
Focus on URL and source validation with strict contextual verification:

Contextual Verification:
- MUST verify URL's relationship to the claimed content
- MUST validate source credibility
- MUST verify URL matches the context
- MUST check if source is appropriate for the information
- MUST verify URL ownership matches claimed organization

Source Authority:
- MUST verify source has authority to provide information
- MUST check if URL represents official source
- MUST validate source's relationship to topic
- MUST verify source's credibility
- MUST check if source is appropriate

Red Flags:
- URLs without clear connection to context
- Sources claiming authority without verification
- URLs representing unrelated organizations
- Marketing URLs without established connection
- Product URLs without verified relationship
- URLs making claims beyond their scope
- Sources without clear authority

IMPORTANT: Any URL that cannot be verified against these criteria MUST be flagged as suspicious.
"""

number_validation_prompt = """
Focus on numerical value validation with strict contextual verification:

Contextual Verification:
- MUST verify if numbers are within reasonable ranges for their context
- MUST validate numerical relationships and proportions
- MUST check if units are appropriate and consistent
- MUST verify if numbers align with known benchmarks
- MUST check for statistical anomalies

Value Authority:
- MUST verify if numbers come from reliable sources
- MUST check if numerical claims are supported by context
- MUST validate mathematical consistency
- MUST verify numerical precision matches the context
- MUST check if numbers follow expected patterns

Red Flags:
- Numbers outside typical ranges without explanation
- Statistical impossibilities
- Inconsistent units or conversions
- Numbers that contradict known facts
- Suspiciously round numbers in precise contexts
- Mathematical impossibilities
- Numbers without proper context or units

IMPORTANT: Any numerical value that cannot be verified against these criteria MUST be flagged as suspicious.
"""

editor_task_prompt = """
Validate and clean the output following these rules:

Validation Rules:
1. For ANY field that cannot be verified:
- Set the field value to None
- Do not provide alternatives
- Do not provide explanations
- Do not attempt modifications

2. For fields that can be verified:
- Keep the original value exactly as is
- Do not modify or enhance
- Do not add information

Field Processing:
- URLs: If source/context not verified = None


Specific issues from validation:
{validation_feedback}

IMPORTANT:
- No partial validations
- No alternative suggestions
- No explanations
- Only None or original value
- Keep response format as specified
"""

class SourceReliability(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class ValidationPoint(ObjectResponse):
    is_suspicious: bool
    feedback: str
    source_reliability: SourceReliability = SourceReliability.UNKNOWN
    verification_method: str = ""
    confidence_score: float = 0.0

class ValidationResult(ObjectResponse):
    url_validation: ValidationPoint

    number_validation: ValidationPoint
    any_suspicion: bool
    overall_feedback: str
    overall_confidence: float = 0.0

    def calculate_suspicion(self):
        self.any_suspicion = any([
            self.url_validation.is_suspicious,

            self.number_validation.is_suspicious
        ])

        suspicious_points = []
        if self.url_validation.is_suspicious:
            suspicious_points.append(f"URL Issues: {self.url_validation.feedback}")

        if self.number_validation.is_suspicious:
            suspicious_points.append(f"Number Issues: {self.number_validation.feedback}")

        self.overall_feedback = " | ".join(suspicious_points) if suspicious_points else "No suspicious content detected."
        
        self.overall_confidence = sum([
            self.url_validation.confidence_score,

            self.number_validation.confidence_score
        ]) / 4.0

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

        prevent_hallucination = getattr(reliability_layer, 'prevent_hallucination', 0)
        if isinstance(prevent_hallucination, property):
            prevent_hallucination = prevent_hallucination.fget(reliability_layer)

        processed_result = result

        if prevent_hallucination > 0:
            if prevent_hallucination == 10:
                copy_task = deepcopy(task)
                copy_task._response = result

                validation_result = ValidationResult(
                    url_validation=ValidationPoint(is_suspicious=False, feedback=""),

                    number_validation=ValidationPoint(is_suspicious=False, feedback=""),
                    any_suspicion=False,
                    overall_feedback=""
                )

                # Run validations
                for validation_type, prompt in [
                    ("url_validation", url_validation_prompt),
                    ("number_validation", number_validation_prompt),
                ]:
                    validator_agent = AgentConfiguration(
                        f"{validation_type.replace('_', ' ').title()} Agent",
                        model=llm_model,
                        sub_task=False
                    )
                    validator_task = Task(
                        prompt,
                        context=[copy_task, copy_task.response_format],
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
                    editor_task = Task(
                        formatted_prompt,
                        context=[copy_task, copy_task.response_format, validation_result],
                        response_format=task.response_format,
                        tools=task.tools
                    )
                    editor_agent.do(editor_task)
                    return editor_task.response

                return result

        return processed_result