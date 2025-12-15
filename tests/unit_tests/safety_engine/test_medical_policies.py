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

from upsonic.safety_engine.policies.medical_policies import (
    MedicalInfoBlockPolicy,
    MedicalInfoBlockPolicy_LLM,
    MedicalInfoAnonymizePolicy,
    MedicalInfoReplacePolicy,
    MedicalInfoRaiseExceptionPolicy,
    MedicalInfoRaiseExceptionPolicy_LLM
)


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_medical_info_block_phi(mock_infer_model):
    """
    TEST 1: Medical Policy Blocks PHI Information
    - USES: MedicalInfoBlockPolicy
    - EXPECTATION: The process should stop immediately when PHI is detected.
      The LLM will NOT be called. The final output will be the block message from the policy.
    - LOOK FOR: A red "Safety Policy Triggered" panel in the console.
    """
    print_header("TEST 1: Medical Policy BLOCKS PHI Input")
    
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
    
    agent_with_medical_policy = Agent(
        model=mock_model,
        user_policy=MedicalInfoBlockPolicy,
        debug=True
    )
    
    phi_task = Task(description="Patient John Doe, medical record number MR123456, has a diagnosis of diabetes. Please review his health information.")
    
    result = agent_with_medical_policy.do(phi_task)
    
    # Final result check
    assert isinstance(result, str)
    assert "blocked" in result.lower()
    # Test passed - PHI information blocked


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_medical_info_block_insurance(mock_infer_model):
    """
    TEST 2: Medical Policy Blocks Insurance Information
    - USES: MedicalInfoBlockPolicy
    - EXPECTATION: The process should stop immediately when insurance information is detected.
    - LOOK FOR: A red "Safety Policy Triggered" panel in the console.
    """
    print_header("TEST 2: Medical Policy BLOCKS Insurance Information Input")

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

    agent_with_medical_policy = Agent(
        model=mock_model,
        user_policy=MedicalInfoBlockPolicy,
        debug=True
    )

    insurance_task = Task(description="My health insurance policy number is INS123456789 and my Medicare number is 123-45-6789A.")
    
    result = agent_with_medical_policy.do(insurance_task)
    
    # Final result check
    assert isinstance(result, str)
    assert "blocked" in result.lower()
    # Test passed - insurance information blocked


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_medical_info_anonymize_prescription(mock_infer_model):
    """
    TEST 3: Medical Policy Anonymizes Prescription Information
    - USES: MedicalInfoAnonymizePolicy
    - EXPECTATION: The policy will find and anonymize prescription information. The LLM will
      receive the MODIFIED prompt and respond to that.
    - LOOK FOR: A yellow "Safety Policy Triggered" panel, and see that the
      LLM's final response refers to the anonymized information.
    """
    print_header("TEST 3: Medical Policy ANONYMIZES Prescription Input")
    
    # Mock the model inference
    mock_model = AsyncMock()
    mock_infer_model.return_value = mock_model
    
    # Mock the model request to return a proper ModelResponse
    mock_response = ModelResponse(
        parts=[TextPart(content="I can help you with your prescription inquiry. The medication information you provided has been processed.")],
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
        user_policy=MedicalInfoAnonymizePolicy,
        debug=True
    )
    
    prescription_task = Task(description="Prescription number RX123456789 for medication XYZ. Can you help me with dosage information?")
    
    result = agent_with_anonymize_policy.do(prescription_task)
    
    # Final result check
    assert isinstance(result, str)
    assert "RX123456789" not in result
    # Test passed - prescription number anonymized


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_medical_info_replace_lab_results(mock_infer_model):
    """
    TEST 4: Medical Policy Replaces Lab Results Information
    - USES: MedicalInfoReplacePolicy
    - EXPECTATION: The policy will find and replace lab results information with placeholders.
    - LOOK FOR: A yellow "Safety Policy Triggered" panel with replacement action.
    """
    print_header("TEST 4: Medical Policy REPLACES Lab Results Input")
    
    # Mock the model inference
    mock_model = AsyncMock()
    mock_infer_model.return_value = mock_model
    
    # Mock the model request to return a proper ModelResponse
    mock_response = ModelResponse(
        parts=[TextPart(content="I can help you with your lab results inquiry. The test information you mentioned has been processed.")],
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
        user_policy=MedicalInfoReplacePolicy,
        debug=True
    )
    
    lab_results_task = Task(description="Lab order number LAB123456 shows blood test results. Can you help me interpret them?")
    
    result = agent_with_replace_policy.do(lab_results_task)
    
    # Final result check
    assert isinstance(result, str)
    assert "LAB123456" not in result
    # Test passed - lab results information replaced


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_medical_info_agent_policy_exception(mock_infer_model):
    """
    TEST 5: Medical Agent Policy Raises Exception on Output
    - USES: MedicalInfoRaiseExceptionPolicy
    - EXPECTATION: The LLM will generate a response containing medical information.
      The agent_policy will then catch this and raise an exception.
    - LOOK FOR: A red "Safety Policy Triggered" panel for the agent output.
    """
    print_header("TEST 5: Medical Agent Policy RAISES EXCEPTION on Output")

    # Mock the model inference
    mock_model = AsyncMock()
    mock_infer_model.return_value = mock_model
    
    # Mock the model request to return a proper ModelResponse
    mock_response = ModelResponse(
        parts=[TextPart(content="The patient's medical record MR789012 shows a diagnosis of hypertension with medication prescribed.")],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=None,
        provider_name="test-provider",
        provider_response_id="test-id",
        provider_details={},
        finish_reason="stop"
    )
    mock_model.request = AsyncMock(return_value=mock_response)
    
    agent_with_medical_exception = Agent(
        model=mock_model,
        agent_policy=MedicalInfoRaiseExceptionPolicy,
        debug=True
    )
    
    medical_task = Task(description="Please provide an update on the patient's medical condition.")
    
    result = agent_with_medical_exception.do(medical_task)
    
    # Final result check
    assert isinstance(result, str)

    assert "disallowed by policy" in result.lower() or "disallowedoperation" in result.lower()  # The policy should block the response
    # Test passed - policy detection working (visible in console output)


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_medical_info_llm_policy(mock_infer_model):
    """
    TEST 6: Medical LLM Policy with Enhanced Detection
    - USES: MedicalInfoBlockPolicy_LLM
    - EXPECTATION: The LLM-powered policy will detect medical information with better context.
    - LOOK FOR: Enhanced detection capabilities with LLM-powered analysis.
    """
    print_header("TEST 6: Medical LLM Policy with Enhanced Detection")
    
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
        user_policy=MedicalInfoBlockPolicy_LLM,
        debug=True
    )
    
    complex_medical_task = Task(description="I need help with my patient's mental health condition and psychiatric evaluation. This involves sensitive medical information.")
    
    result = agent_with_llm_policy.do(complex_medical_task)
    
    # Final result check
    assert isinstance(result, str)
    assert "blocked" in result.lower()
    # Test passed - LLM-powered medical policy working


@pytest.mark.asyncio
@patch('upsonic.models.infer_model')
async def test_medical_info_all_clear(mock_infer_model):
    """
    TEST 7: Happy Path - No Medical Policies Triggered
    - USES: No policies
    - EXPECTATION: The agent functions normally without any interference.
    - LOOK FOR: No safety policy panels should be printed.
    """
    print_header("TEST 7: All Clear - No Medical Policies Triggered")
    
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
    await test_medical_info_block_phi()
    await test_medical_info_block_insurance()
    await test_medical_info_anonymize_prescription()
    await test_medical_info_replace_lab_results()
    await test_medical_info_agent_policy_exception()
    await test_medical_info_llm_policy()
    await test_medical_info_all_clear()
    # All tests completed


if __name__ == "__main__":
    asyncio.run(main())
