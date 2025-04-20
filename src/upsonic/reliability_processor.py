from copy import deepcopy
from typing import Any, Optional, Union, Type, List
from pydantic import BaseModel, Field
from enum import Enum
import re
from urllib.parse import urlparse
import requests
import asyncio
from .client.tasks.tasks import Task
from .client.agent_configuration.agent_configuration import AgentConfiguration
from .client.tasks.task_response import ObjectResponse

# Define the validation prompts
url_validation_prompt = """
Focus on basic URL source validation:

Source Verification:
- Check if the source is come from the content. But dont make assumption just check the context and try to find exact things. If not flag it.
- If you can see the things in the context everything okay (Trusted Source).

IMPORTANT: If the URL source cannot be verified, flag it as suspicious.
"""

number_validation_prompt = """
Focus on basic numerical validation:

Number Verification:
- Check if the source is come from the content. But dont make assumption just check the context and try to find exact things. If not flag it.
- If you can see the things in the context everything okay (Trusted Source).

IMPORTANT: If the numbers cannot be verified, flag them as suspicious.
"""

code_validation_prompt = """
Focus on basic code validation:

Code Verification:
- Check if the source is come from the content. But dont make assumption just check the context and try to find exact things. If not flag it.
- If you can see the things in the context everything okay (Trusted Source).

IMPORTANT: If the code cannot be verified or appears suspicious, flag it as suspicious.
"""

information_validation_prompt = """
Focus on basic information validation:

Information Verification:
- Check if the source is come from the content. But dont make assumption just check the context and try to find exact things. If not flag it.
- If you can see the things in the context everything okay (Trusted Source).

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
    async def process_result(
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

                # Create a list to store validation tasks
                validation_tasks = []
                validation_types = []
                validator_agents = {}

                # Process context strings once for all validations
                context_strings = []
                context_strings.append(f"Given Task: {copy_task.description}")

                # Process context items if they exist
                if copy_task.context:
                    context_items = copy_task.context if isinstance(copy_task.context, list) else [copy_task.context]
                    if copy_task.response_format:
                        context_items.append(copy_task.response_format)
                    for item in context_items:
                        type_string = type(item).__name__
                        the_class_string = None
                        try:
                            the_class_string = item.__bases__[0].__name__
                        except:
                            pass

                        if the_class_string == ObjectResponse.__name__ or the_class_string == BaseModel.__name__:
                            context_strings.append(f"\n\nUser requested output: ```Requested Output {item.model_fields}```")
                        elif isinstance(item, str):
                            context_strings.append(f"\n\nContext That Came From User (Trusted Source): ```User given context {item}```")
                        else:
                            pass

                # Add the current AI response to context
                context_strings.append(f"\nCurrent AI Response (Untrusted Source, last AI responose that we are checking now): {old_task_output}")

                # Prepare validation tasks
                for validation_type, prompt in [
                    ("url_validation", url_validation_prompt),
                    ("number_validation", number_validation_prompt),
                    ("information_validation", information_validation_prompt),
                    ("code_validation", code_validation_prompt),
                ]:
                    # Create a specific agent for each validation type
                    agent_name = f"{validation_type.replace('_', ' ').title()} Agent"
                    validator_agents[validation_type] = AgentConfiguration(
                        agent_name,
                        model=llm_model,
                        sub_task=False
                    )
                    
                    # For URL validation, skip if no URLs are present
                    if validation_type == "url_validation":
                        if not contains_urls([prompt] + context_strings):
                            # Set a default "no URLs found" validation point
                            setattr(validation_result, validation_type, ValidationPoint(
                                is_suspicious=False,
                                feedback="No URLs found in content to validate",
                                suspicious_points=[],
                                source_reliability=SourceReliability.UNKNOWN,
                                verification_method="regex_url_detection",
                                confidence_score=1.0
                            ))
                            continue

                    # Create validation task
                    validator_task = Task(
                        prompt,
                        images=task.images,
                        response_format=ValidationPoint,
                        tools=task.tools,
                        context=context_strings,  # Pass the processed context strings
                        price_id_=task.price_id,
                        not_main_task=True
                    )
                    
                    # Add task to the list
                    validation_tasks.append(validator_task)
                    validation_types.append(validation_type)

                # Execute all validation tasks in parallel if there are any
                if validation_tasks:
                    # Run each validation task with its specific agent
                    validation_coroutines = []
                    for i, validation_type in enumerate(validation_types):
                        validation_coroutines.append(
                            validator_agents[validation_type].do_async(validation_tasks[i])
                        )
                    
                    # Wait for all validation tasks to complete
                    await asyncio.gather(*validation_coroutines)
                    
                    # Process results
                    for i, validation_type in enumerate(validation_types):
                        setattr(validation_result, validation_type, validation_tasks[i].response)

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
                        images=task.images,
                        context=the_context,
                        response_format=task.response_format,
                        tools=task.tools,
                        price_id_=task.price_id,
                        not_main_task=True
                    )
                    await editor_agent.do_async(editor_task)
                    return editor_task.response

                return result

        return processed_result

def find_urls_in_text(text: str) -> List[str]:
    """Find all URLs in the given text using regex pattern matching."""
    # This pattern matches URLs starting with http://, https://, ftp://, or www.
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def contains_urls(texts: List[str]) -> bool:
    """Check if any of the provided texts contain URLs."""
    for text in texts:
        if not isinstance(text, str):
            continue
        if find_urls_in_text(text):
            return True
    return False