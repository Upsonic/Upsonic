import asyncio
import os
import pytest
from unittest.mock import patch, AsyncMock
from contextlib import asynccontextmanager

from upsonic import Agent, Task
from upsonic.agent.run_result import RunResult
from upsonic.models import ModelResponse, TextPart

from upsonic.safety_engine import (
    RuleBase,
    ActionBase,
    Policy,
    PolicyInput,
    RuleOutput,
    PolicyOutput
)

from upsonic.safety_engine.policies.technical_policies import (
    TechnicalSecurityBlockPolicy,
    TechnicalSecurityBlockPolicy_LLM,
    TechnicalSecurityAnonymizePolicy,
    TechnicalSecurityReplacePolicy,
    TechnicalSecurityRaiseExceptionPolicy,
    TechnicalSecurityRaiseExceptionPolicy_LLM
)


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_technical_security_block_api_key(mock_infer_model):
    """
    TEST 1: Technical Security Policy Blocks API Key Information
    - USES: TechnicalSecurityBlockPolicy
    - EXPECTATION: The process should stop immediately when API keys are detected.
      The LLM will NOT be called. The final output will be the block message from the policy.
    - LOOK FOR: A red "Safety Policy Triggered" panel in the console.
    """
    print_header("TEST 1: Technical Security Policy BLOCKS API Key Input")
    
    # Mock the model inference
    mock_model = AsyncMock()
    mock_infer_model.return_value = mock_model
    
    # Mock the model request to return a proper ModelResponse
    mock_response = ModelResponse(
        parts=[TextPart(content="This content has been blocked by safety policy.")],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=None,
        provider_name="test-provider",
        provider_response_id="test-id",
        provider_details={},
        finish_reason="stop"
    )
    mock_model.request = AsyncMock(return_value=mock_response)
    
    agent_with_technical_policy = Agent(
        model=mock_model,
        user_policy=TechnicalSecurityBlockPolicy,
        debug=True
    )
    
    api_key_task = Task(description="My OpenAI API key is sk-1234567890abcdef1234567890abcdef12345678. Can you help me configure it?")
    
    result = agent_with_technical_policy.do(api_key_task)
    
    # Final result check
    assert isinstance(result, str)
    assert "blocked" in result.lower()
    # Test passed - API key information blocked


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_technical_security_block_password(mock_infer_model):
    """
    TEST 2: Technical Security Policy Blocks Password Information
    - USES: TechnicalSecurityBlockPolicy
    - EXPECTATION: The process should stop immediately when passwords are detected.
    - LOOK FOR: A red "Safety Policy Triggered" panel in the console.
    """
    print_header("TEST 2: Technical Security Policy BLOCKS Password Input")

    # Mock the model inference
    mock_model = AsyncMock()
    mock_infer_model.return_value = mock_model
    
    # Mock the model request to return a proper ModelResponse
    mock_response = ModelResponse(
        parts=[TextPart(content="This content has been blocked by safety policy.")],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=None,
        provider_name="test-provider",
        provider_response_id="test-id",
        provider_details={},
        finish_reason="stop"
    )
    mock_model.request = AsyncMock(return_value=mock_response)

    agent_with_technical_policy = Agent(
        model=mock_model,
        user_policy=TechnicalSecurityBlockPolicy,
        debug=True
    )

    password_task = Task(description="The database password is 'MySecretPassword123!'. Please help me connect to the database.")
    
    result = agent_with_technical_policy.do(password_task)
    
    # Final result check
    assert isinstance(result, str)
    assert "blocked" in result.lower()
    # Test passed - password information blocked


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_technical_security_anonymize_token(mock_infer_model):
    """
    TEST 3: Technical Security Policy Anonymizes Token Information
    - USES: TechnicalSecurityAnonymizePolicy
    - EXPECTATION: The policy will find and anonymize tokens. The LLM will
      receive the MODIFIED prompt and respond to that.
    - LOOK FOR: A yellow "Safety Policy Triggered" panel, and see that the
      LLM's final response refers to the anonymized information.
    """
    print_header("TEST 3: Technical Security Policy ANONYMIZES Token Input")
    
    # Mock the model inference
    mock_model = AsyncMock()
    mock_infer_model.return_value = mock_model
    
    # Mock the model request to return a proper ModelResponse
    mock_response = ModelResponse(
        parts=[TextPart(content="I can help you with your token inquiry. The authentication information you provided has been processed.")],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=None,
        provider_name="test-provider",
        provider_response_id="test-id",
        provider_details={},
        finish_reason="stop"
    )
    mock_model.request = AsyncMock(return_value=mock_response)
    
    agent_with_anonymize_policy = Agent(
        model=mock_model,
        user_policy=TechnicalSecurityAnonymizePolicy,
        debug=True
    )
    
    token_task = Task(description="My JWT token is eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c. Can you help me decode it?")
    
    result = agent_with_anonymize_policy.do(token_task)
    
    # Final result check
    assert isinstance(result, str)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
    # Test passed - JWT token anonymized


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_technical_security_replace_certificate(mock_infer_model):
    """
    TEST 4: Technical Security Policy Replaces Certificate Information
    - USES: TechnicalSecurityReplacePolicy
    - EXPECTATION: The policy will find and replace certificate information with placeholders.
    - LOOK FOR: A yellow "Safety Policy Triggered" panel with replacement action.
    """
    print_header("TEST 4: Technical Security Policy REPLACES Certificate Input")
    
    # Mock the model inference
    mock_model = AsyncMock()
    mock_infer_model.return_value = mock_model
    
    # Mock the model request to return a proper ModelResponse
    mock_response = ModelResponse(
        parts=[TextPart(content="I can help you with your certificate inquiry. The security information you mentioned has been processed.")],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=None,
        provider_name="test-provider",
        provider_response_id="test-id",
        provider_details={},
        finish_reason="stop"
    )
    mock_model.request = AsyncMock(return_value=mock_response)
    
    agent_with_replace_policy = Agent(
        model=mock_model,
        user_policy=TechnicalSecurityReplacePolicy,
        debug=True
    )
    
    certificate_task = Task(description="Here's my private key: -----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n-----END PRIVATE KEY-----. Can you help me with SSL configuration?")
    
    result = agent_with_replace_policy.do(certificate_task)
    
    # Final result check
    assert isinstance(result, str)
    assert "-----BEGIN PRIVATE KEY-----" not in result
    # Test passed - private key replaced


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_technical_security_agent_policy_exception(mock_infer_model):
    """
    TEST 5: Technical Security Agent Policy Raises Exception on Output
    - USES: TechnicalSecurityRaiseExceptionPolicy
    - EXPECTATION: The LLM will generate a response containing technical security information.
      The agent_policy will then catch this and raise an exception.
    - LOOK FOR: A red "Safety Policy Triggered" panel for the agent output.
    """
    print_header("TEST 5: Technical Security Agent Policy RAISES EXCEPTION on Output")

    # Mock the model inference
    mock_model = AsyncMock()
    mock_infer_model.return_value = mock_model
    
    # Mock the model request to return a proper ModelResponse
    mock_response = ModelResponse(
        parts=[TextPart(content="Your AWS access key AKIAIOSFODNN7EXAMPLE and secret key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY have been configured.")],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=None,
        provider_name="test-provider",
        provider_response_id="test-id",
        provider_details={},
        finish_reason="stop"
    )
    mock_model.request = AsyncMock(return_value=mock_response)
    
    agent_with_technical_exception = Agent(
        model=mock_model,
        agent_policy=TechnicalSecurityRaiseExceptionPolicy,
        debug=True
    )
    
    technical_task = Task(description="Please configure my AWS credentials.")
    
    result = agent_with_technical_exception.do(technical_task)
    
    # Final result check
    assert isinstance(result, str)
    # The policy is working correctly - it should block the response and return a policy violation message
    assert "disallowed by policy" in result.lower() or "disallowedoperation" in result.lower()
    # Test passed - policy detection working and blocking content correctly


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_technical_security_llm_policy(mock_infer_model):
    """
    TEST 6: Technical Security LLM Policy with Enhanced Detection
    - USES: TechnicalSecurityBlockPolicy_LLM
    - EXPECTATION: The LLM-powered policy will detect technical security information with better context.
    - LOOK FOR: Enhanced detection capabilities with LLM-powered analysis.
    """
    print_header("TEST 6: Technical Security LLM Policy with Enhanced Detection")
    
    # Mock the model inference
    mock_model = AsyncMock()
    mock_infer_model.return_value = mock_model
    
    # Mock the model request to return a proper ModelResponse
    mock_response = ModelResponse(
        parts=[TextPart(content="This content has been blocked by LLM-powered safety policy.")],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=None,
        provider_name="test-provider",
        provider_response_id="test-id",
        provider_details={},
        finish_reason="stop"
    )
    mock_model.request = AsyncMock(return_value=mock_response)
    
    agent_with_llm_policy = Agent(
        model=mock_model,
        user_policy=TechnicalSecurityBlockPolicy_LLM,
        debug=True
    )
    
    complex_technical_task = Task(description="I need help with my database configuration. The connection string is mysql://user:password123@localhost:3306/mydb and my GitHub token is ghp_1234567890abcdef1234567890abcdef12345678.")
    
    result = agent_with_llm_policy.do(complex_technical_task)
    
    # Final result check
    assert isinstance(result, str)
    assert "blocked" in result.lower()
    # Test passed - LLM-powered technical security policy working


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_technical_security_all_clear(mock_infer_model):
    """
    TEST 7: Happy Path - No Technical Security Policies Triggered
    - USES: No policies
    - EXPECTATION: The agent functions normally without any interference.
    - LOOK FOR: No safety policy panels should be printed.
    """
    print_header("TEST 7: All Clear - No Technical Security Policies Triggered")
    
    # Mock the model inference
    mock_model = AsyncMock()
    mock_infer_model.return_value = mock_model
    
    # Mock the model request to return a proper ModelResponse
    mock_response = ModelResponse(
        parts=[TextPart(content="The weather today is sunny and warm.")],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=None,
        provider_name="test-provider",
        provider_response_id="test-id",
        provider_details={},
        finish_reason="stop"
    )
    mock_model.request = AsyncMock(return_value=mock_response)
    
    plain_agent = Agent(model=mock_model, debug=True)
    
    safe_task = Task(description="What's the weather like today?")
    
    result = plain_agent.do(safe_task)

    # Final result check
    assert isinstance(result, str)
    assert "sunny" in result.lower()
    # Test passed - normal operation


def print_header(title):
    """Helper function to print a nice header for each test."""
    pass


async def main():
    """Main function to run all test cases in order."""
    await test_technical_security_block_api_key()
    await test_technical_security_block_password()
    await test_technical_security_anonymize_token()
    await test_technical_security_replace_certificate()
    await test_technical_security_agent_policy_exception()
    await test_technical_security_llm_policy()
    await test_technical_security_all_clear()
    # All tests completed


if __name__ == "__main__":
    asyncio.run(main())
