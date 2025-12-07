"""
Test 24: Guardrail testing task class
Success criteria: We check attributes, what we log and results
"""
import pytest
from io import StringIO
from contextlib import redirect_stdout

from upsonic import Agent, Task
from pydantic import BaseModel, ValidationError

pytestmark = pytest.mark.timeout(180)


class EmailResponse(BaseModel):
    """Structured email response."""
    subject: str
    body: str
    is_professional: bool


def email_guardrail(response: EmailResponse) -> bool:
    """Guardrail to ensure email is professional."""
    if not response.is_professional:
        return False
    if len(response.subject) == 0:
        return False
    if "urgent" in response.body.lower() and "!!!" in response.body:
        return False  # Too many exclamation marks for urgent emails
    return True


def length_guardrail(response: str) -> bool:
    """Guardrail to ensure response is not too short."""
    if len(response) < 10:
        return False
    return True


def keyword_guardrail(response: str) -> bool:
    """Guardrail that checks for specific keywords."""
    required_keywords = ["python", "programming"]
    response_lower = response.lower()
    for keyword in required_keywords:
        if keyword not in response_lower:
            return False
    return True


@pytest.mark.asyncio
async def test_task_guardrail_basic():
    """Test basic guardrail functionality with Task."""
    
    # Create task with guardrail
    task = Task(
        description="Explain Python in exactly 5 words or less",
        guardrail=length_guardrail,
        guardrail_retries=2
    )
    
    # Verify task attributes
    assert task.guardrail is not None, "Guardrail should be set"
    assert task.guardrail == length_guardrail, "Guardrail function should match"
    assert task.guardrail_retries == 2, "Guardrail retries should be 2"
    
    # Execute task
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    assert isinstance(result, str), "Result should be a string"
    assert len(result) >= 10, "Result should pass guardrail (length >= 10)"
    
    # Verify logging
    assert len(output) > 0, "Should have logging output"


@pytest.mark.asyncio
async def test_task_guardrail_with_retries(capsys):
    """Test guardrail with multiple retries."""
    
    task = Task(
        description="Write about Python programming language. Must mention both 'Python' and 'programming'.",
        guardrail=keyword_guardrail,
        guardrail_retries=3
    )
    
    # Verify attributes
    assert task.guardrail == keyword_guardrail, "Guardrail should be keyword_guardrail"
    assert task.guardrail_retries == 3, "Should have 3 retries"
    
    agent = Agent(model="openai/gpt-4o", debug=True)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result passes guardrail
    assert result is not None, "Result should not be None"
    assert isinstance(result, str), "Result should be a string"
    
    # Test that the guardrail actually validates the result
    assert keyword_guardrail(result), "Final result should pass guardrail validation"
    
    # Check that result contains required keywords
    result_lower = result.lower()
    assert "python" in result_lower, "Result should contain 'python'"
    assert "programming" in result_lower, "Result should contain 'programming'"
    
    # Verify logging output shows the process
    assert len(output) > 0, "Should have logging output"


@pytest.mark.asyncio
async def test_task_guardrail_with_structured_output():
    """Test guardrail with structured output (Pydantic model)."""
    
    task = Task(
        description="Write a professional email about a project update",
        response_format=EmailResponse,
        guardrail=email_guardrail,
        guardrail_retries=2
    )
    
    # Verify attributes
    assert task.guardrail == email_guardrail, "Guardrail should be email_guardrail"
    assert task.guardrail_retries == 2, "Should have 2 retries"
    assert task.response_format == EmailResponse, "Response format should be EmailResponse"
    
    agent = Agent(model="openai/gpt-4o", debug=True)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    assert isinstance(result, EmailResponse), "Result should be EmailResponse instance"
    
    # Verify result structure
    assert hasattr(result, 'subject'), "Result should have subject"
    assert hasattr(result, 'body'), "Result should have body"
    assert hasattr(result, 'is_professional'), "Result should have is_professional"
    
    assert isinstance(result.subject, str), "Subject should be a string"
    assert isinstance(result.body, str), "Body should be a string"
    assert isinstance(result.is_professional, bool), "is_professional should be bool"
    
    # Verify guardrail validation
    assert email_guardrail(result), "Result should pass email guardrail"
    assert result.is_professional is True, "Email should be professional"
    assert len(result.subject) > 0, "Subject should not be empty"
    
    # Verify logging
    assert len(output) > 0, "Should have logging output"


def test_task_guardrail_defaults():
    """Test Task guardrail default values."""
    
    task = Task(description="Test task without guardrail")
    
    # Verify defaults
    assert task.guardrail is None, "Default guardrail should be None"
    assert task.guardrail_retries is None, "Default guardrail_retries should be None"


def test_task_guardrail_assignment():
    """Test Task guardrail can be assigned and modified."""
    
    def initial_guardrail(response: str) -> bool:
        return len(response) > 5
    
    def new_guardrail(response: str) -> bool:
        return len(response) > 20
    
    task = Task(
        description="Test",
        guardrail=initial_guardrail,
        guardrail_retries=1
    )
    
    # Verify initial values
    assert task.guardrail == initial_guardrail
    assert task.guardrail_retries == 1
    
    # Modify values
    task.guardrail = new_guardrail
    task.guardrail_retries = 5
    
    # Verify modified values
    assert task.guardrail == new_guardrail
    assert task.guardrail_retries == 5


@pytest.mark.asyncio
async def test_task_guardrail_none_retries():
    """Test Task with guardrail but no retries specified."""
    
    task = Task(
        description="Write a sentence about AI that is at least 10 characters long",
        guardrail=length_guardrail,
        guardrail_retries=None  # Explicitly set to None
    )
    
    # Verify attributes
    assert task.guardrail == length_guardrail, "Guardrail should be set"
    assert task.guardrail_retries is None, "Retries should be None"
    
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    result = await agent.do_async(task)
    
    # Should still execute, just with default retry behavior (likely no retries)
    assert result is not None, "Result should not be None"


@pytest.mark.asyncio
async def test_task_guardrail_logging_validation(capsys):
    """Test that guardrail validation is properly logged."""
    
    validation_calls = []
    
    def tracking_guardrail(response: str) -> bool:
        """Guardrail that tracks when it's called."""
        validation_calls.append(response)
        # Make it pass on first try for predictability
        return len(response) > 5
    
    task = Task(
        description="Say hello",
        guardrail=tracking_guardrail,
        guardrail_retries=2
    )
    
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify guardrail was called at least once
    assert len(validation_calls) >= 1, "Guardrail should have been called"
    
    # Verify the final result passed validation
    assert tracking_guardrail(result), "Final result should pass guardrail"
    
    # Verify logging
    assert len(output) > 0, "Should have logging output"


@pytest.mark.asyncio  
async def test_task_guardrail_callable_verification():
    """Test that guardrail must be callable."""
    
    task = Task(
        description="Test",
        guardrail=length_guardrail
    )
    
    # Verify it's callable
    assert callable(task.guardrail), "Guardrail should be callable"
    
    # Test the guardrail directly
    assert task.guardrail("This is a long enough string") is True
    assert task.guardrail("Short") is False

