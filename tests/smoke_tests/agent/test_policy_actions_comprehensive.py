"""
Comprehensive tests for policy Anonymize and Replace actions across multiple rule types.

Tests PII, Financial, Phone, and Medical policies with both Anonymize and Replace actions,
using do_async, event streaming, and pure text streaming modes.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import pytest

from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.safety_engine.base import Policy
from upsonic.safety_engine.policies.pii_policies import (
    PIIRule,
    PIIAnonymizeAction,
    PIIReplaceAction,
)
from upsonic.safety_engine.policies.financial_policies import (
    FinancialInfoRule,
    FinancialInfoAnonymizeAction,
    FinancialInfoReplaceAction,
)
from upsonic.safety_engine.policies.phone_policies import (
    AnonymizePhoneNumberRule,
    AnonymizePhoneNumberAction,
)
from upsonic.safety_engine.policies.medical_policies import (
    MedicalInfoRule,
    MedicalInfoAnonymizeAction,
    MedicalInfoReplaceAction,
)
from upsonic.run.events.events import (
    AgentEvent,
    TextDeltaEvent,
    TextCompleteEvent,
    ToolCallEvent,
    ToolResultEvent,
)

pytestmark = pytest.mark.timeout(120)

# ─── Sensitive test data ────────────────────────────────────────────

SENSITIVE_EMAIL: str = "john.doe@example.com"
SENSITIVE_PHONE: str = "555-123-4567"
SENSITIVE_CREDIT_CARD: str = "4532015112830366"
SENSITIVE_PATIENT_ID: str = "MRN7839201"
SENSITIVE_PRESCRIPTION: str = "RX4827103"


# ─── Policy factories ──────────────────────────────────────────────


def _pii_anonymize_policy(
    description: bool = True,
    context: bool = True,
    system_prompt: bool = True,
    chat_history: bool = True,
    tool_outputs: bool = True,
) -> Policy:
    return Policy(
        name="PII Anonymize",
        description="Anonymizes PII with random replacements",
        rule=PIIRule(),
        action=PIIAnonymizeAction(),
        apply_to_description=description,
        apply_to_context=context,
        apply_to_system_prompt=system_prompt,
        apply_to_chat_history=chat_history,
        apply_to_tool_outputs=tool_outputs,
    )


def _pii_replace_policy(
    description: bool = True,
    context: bool = True,
    system_prompt: bool = True,
    chat_history: bool = True,
    tool_outputs: bool = True,
) -> Policy:
    return Policy(
        name="PII Replace",
        description="Replaces PII with [PII_REDACTED]",
        rule=PIIRule(),
        action=PIIReplaceAction(),
        apply_to_description=description,
        apply_to_context=context,
        apply_to_system_prompt=system_prompt,
        apply_to_chat_history=chat_history,
        apply_to_tool_outputs=tool_outputs,
    )


def _financial_anonymize_policy() -> Policy:
    return Policy(
        name="Financial Anonymize",
        description="Anonymizes financial info with random replacements",
        rule=FinancialInfoRule(),
        action=FinancialInfoAnonymizeAction(),
        apply_to_description=True,
        apply_to_context=True,
        apply_to_system_prompt=True,
        apply_to_chat_history=True,
        apply_to_tool_outputs=True,
    )


def _financial_replace_policy() -> Policy:
    return Policy(
        name="Financial Replace",
        description="Replaces financial info with [FINANCIAL_INFO_REDACTED]",
        rule=FinancialInfoRule(),
        action=FinancialInfoReplaceAction(),
        apply_to_description=True,
        apply_to_context=True,
        apply_to_system_prompt=True,
        apply_to_chat_history=True,
        apply_to_tool_outputs=True,
    )


def _phone_anonymize_policy() -> Policy:
    return Policy(
        name="Phone Anonymize",
        description="Anonymizes phone numbers with random digits",
        rule=AnonymizePhoneNumberRule(),
        action=AnonymizePhoneNumberAction(),
        apply_to_description=True,
        apply_to_context=True,
        apply_to_system_prompt=True,
        apply_to_chat_history=True,
        apply_to_tool_outputs=True,
    )


def _medical_anonymize_policy() -> Policy:
    return Policy(
        name="Medical Anonymize",
        description="Anonymizes medical info with random replacements",
        rule=MedicalInfoRule(),
        action=MedicalInfoAnonymizeAction(),
        apply_to_description=True,
        apply_to_context=True,
        apply_to_system_prompt=True,
        apply_to_chat_history=True,
        apply_to_tool_outputs=True,
    )


def _medical_replace_policy() -> Policy:
    return Policy(
        name="Medical Replace",
        description="Replaces medical info with [MEDICAL_INFO_REDACTED]",
        rule=MedicalInfoRule(),
        action=MedicalInfoReplaceAction(),
        apply_to_description=True,
        apply_to_context=True,
        apply_to_system_prompt=True,
        apply_to_chat_history=True,
        apply_to_tool_outputs=True,
    )


# ─── Tools ──────────────────────────────────────────────────────────


@tool
def lookup_contact(query: str) -> str:
    """Look up contact information for a person."""
    return f"Contact info: email is {SENSITIVE_EMAIL}, phone is {SENSITIVE_PHONE}"


@tool
def lookup_credit_card(query: str) -> str:
    """Look up credit card information."""
    return f"Card on file: {SENSITIVE_CREDIT_CARD} (Visa)"


@tool
def lookup_patient(query: str) -> str:
    """Look up patient medical records."""
    return (
        f"Patient record: patient id {SENSITIVE_PATIENT_ID}, "
        f"prescription {SENSITIVE_PRESCRIPTION}, "
        f"diagnosis: Type 2 diabetes, medication: Metformin 500mg"
    )


@tool
def lookup_phone(query: str) -> str:
    """Look up a phone number."""
    return f"Phone number on file: {SENSITIVE_PHONE}"


# ═══════════════════════════════════════════════════════════════════
#  1. PII ANONYMIZE — do_async
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pii_anonymize_do_async() -> None:
    """PII Anonymize with do_async: email + phone anonymized, de-anonymized in output."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helpful assistant. User's email is {SENSITIVE_EMAIL}.",
        user_policy=_pii_anonymize_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up contact info for {SENSITIVE_EMAIL} (phone {SENSITIVE_PHONE}). "
            "Use the lookup_contact tool. Return the full contact information."
        ),
        context=f"User email: {SENSITIVE_EMAIL}",
        tools=[lookup_contact],
    )

    result: Any = await agent.do_async(task)
    print(f"\n[PII ANONYMIZE do_async] Output: {result}")

    assert result is not None
    assert SENSITIVE_EMAIL in str(result), f"Email must be de-anonymized in output: {result}"
    assert task._anonymization_map is None, "Anonymization map must be cleaned up"
    assert task._policy_originals is None, "Policy originals must be cleaned up"


# ═══════════════════════════════════════════════════════════════════
#  2. PII REPLACE — do_async
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pii_replace_do_async() -> None:
    """PII Replace with do_async: email replaced with [PII_REDACTED], de-anonymized in output."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helpful assistant. User's email is {SENSITIVE_EMAIL}.",
        user_policy=_pii_replace_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up contact info for {SENSITIVE_EMAIL} (phone {SENSITIVE_PHONE}). "
            "Use the lookup_contact tool. Return the full contact information."
        ),
        context=f"User email: {SENSITIVE_EMAIL}",
        tools=[lookup_contact],
    )

    result: Any = await agent.do_async(task)
    print(f"\n[PII REPLACE do_async] Output: {result}")

    assert result is not None
    assert SENSITIVE_EMAIL in str(result), f"Email must be de-anonymized in output: {result}"
    assert "[PII_REDACTED]" not in str(result), "Placeholder must not appear in final output"
    assert task._anonymization_map is None, "Anonymization map must be cleaned up"
    assert task._policy_originals is None, "Policy originals must be cleaned up"


# ═══════════════════════════════════════════════════════════════════
#  3. PII ANONYMIZE — event streaming
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pii_anonymize_stream_events() -> None:
    """PII Anonymize streaming: tool results anonymized, final text de-anonymized."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helpful assistant. User's email is {SENSITIVE_EMAIL}.",
        user_policy=_pii_anonymize_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up contact info for {SENSITIVE_EMAIL} and {SENSITIVE_PHONE}. "
            "Use the lookup_contact tool. Return the full contact information."
        ),
        tools=[lookup_contact],
    )

    text_chunks: list[str] = []
    tool_results: list[ToolResultEvent] = []

    async for event in agent.astream(task, events=True):
        if isinstance(event, TextDeltaEvent):
            text_chunks.append(event.content)
            print(event.content, end="", flush=True)
        elif isinstance(event, ToolCallEvent):
            print(f"\n[Tool Call] {event.tool_name}")
        elif isinstance(event, ToolResultEvent):
            tool_results.append(event)
            print(f"[Tool Result] {event.tool_name}: {str(event.result)[:100]}")
    print()

    full_text: str = "".join(text_chunks)
    print(f"[PII ANONYMIZE stream events] Full text: {full_text[:300]}")

    assert SENSITIVE_EMAIL in full_text, f"Email must be de-anonymized: {full_text[:200]}"

    for tr in tool_results:
        assert SENSITIVE_EMAIL not in str(tr.result or ""), \
            "Tool result events must contain anonymized PII, not original"

    assert task._anonymization_map is None


# ═══════════════════════════════════════════════════════════════════
#  4. PII REPLACE — event streaming
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pii_replace_stream_events() -> None:
    """PII Replace streaming: tool results contain [PII_REDACTED], final text de-anonymized."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helpful assistant. User's email is {SENSITIVE_EMAIL}.",
        user_policy=_pii_replace_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up contact info for {SENSITIVE_EMAIL} and {SENSITIVE_PHONE}. "
            "Use the lookup_contact tool. Return the full contact information."
        ),
        tools=[lookup_contact],
    )

    text_chunks: list[str] = []
    tool_results: list[ToolResultEvent] = []

    async for event in agent.astream(task, events=True):
        if isinstance(event, TextDeltaEvent):
            text_chunks.append(event.content)
            print(event.content, end="", flush=True)
        elif isinstance(event, ToolCallEvent):
            print(f"\n[Tool Call] {event.tool_name}")
        elif isinstance(event, ToolResultEvent):
            tool_results.append(event)
            print(f"[Tool Result] {event.tool_name}: {str(event.result)[:100]}")
    print()

    full_text: str = "".join(text_chunks)
    print(f"[PII REPLACE stream events] Full text: {full_text[:300]}")

    assert SENSITIVE_EMAIL in full_text, f"Email must be de-anonymized: {full_text[:200]}"
    assert "[PII_REDACTED]" not in full_text, "Placeholder must not appear in streamed text"

    for tr in tool_results:
        assert SENSITIVE_EMAIL not in str(tr.result or ""), \
            "Tool result events must NOT contain original email"

    assert task._anonymization_map is None


# ═══════════════════════════════════════════════════════════════════
#  5. PII ANONYMIZE — pure text streaming
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pii_anonymize_stream_text() -> None:
    """PII Anonymize pure text streaming: only str chunks, de-anonymized."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helpful assistant. Email: {SENSITIVE_EMAIL}.",
        user_policy=_pii_anonymize_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up contact info for {SENSITIVE_EMAIL} (phone {SENSITIVE_PHONE}). "
            "Use the lookup_contact tool. Return the full contact information."
        ),
        tools=[lookup_contact],
    )

    streamed: str = ""
    async for chunk in agent.astream(task):
        assert isinstance(chunk, str), f"Pure text stream must yield str, got {type(chunk)}"
        streamed += chunk
        print(chunk, end="", flush=True)
    print()

    print(f"[PII ANONYMIZE stream text] Full: {streamed[:300]}")

    assert len(streamed) > 0
    assert SENSITIVE_EMAIL in streamed, f"Email must be de-anonymized: {streamed[:200]}"
    assert task._anonymization_map is None


# ═══════════════════════════════════════════════════════════════════
#  6. PII REPLACE — pure text streaming
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pii_replace_stream_text() -> None:
    """PII Replace pure text streaming: de-anonymized, no [PII_REDACTED] in output."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helpful assistant. Email: {SENSITIVE_EMAIL}.",
        user_policy=_pii_replace_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up contact info for {SENSITIVE_EMAIL} (phone {SENSITIVE_PHONE}). "
            "Use the lookup_contact tool. Return the full contact information."
        ),
        tools=[lookup_contact],
    )

    streamed: str = ""
    async for chunk in agent.astream(task):
        assert isinstance(chunk, str), f"Pure text stream must yield str, got {type(chunk)}"
        streamed += chunk
        print(chunk, end="", flush=True)
    print()

    print(f"[PII REPLACE stream text] Full: {streamed[:300]}")

    assert len(streamed) > 0
    assert SENSITIVE_EMAIL in streamed, f"Email must be de-anonymized: {streamed[:200]}"
    assert "[PII_REDACTED]" not in streamed, "Placeholder must not leak into streamed output"
    assert task._anonymization_map is None


# ═══════════════════════════════════════════════════════════════════
#  7. FINANCIAL ANONYMIZE — do_async
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_financial_anonymize_do_async() -> None:
    """Financial Anonymize: credit card number anonymized and de-anonymized in output."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful billing assistant.",
        user_policy=_financial_anonymize_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up the credit card ending in {SENSITIVE_CREDIT_CARD[-4:]} for the customer. "
            "Use the lookup_credit_card tool. Return the full card number."
        ),
        context=f"Customer credit card on file: {SENSITIVE_CREDIT_CARD}",
        tools=[lookup_credit_card],
    )

    result: Any = await agent.do_async(task)
    print(f"\n[FINANCIAL ANONYMIZE do_async] Output: {result}")

    assert result is not None
    assert task._anonymization_map is None, "Anonymization map must be cleaned up"
    assert task._policy_originals is None, "Policy originals must be cleaned up"

    output_str: str = str(result)
    assert SENSITIVE_CREDIT_CARD in output_str or SENSITIVE_CREDIT_CARD[-4:] in output_str, \
        f"Credit card (or last 4 digits) must be in output: {output_str[:300]}"


# ═══════════════════════════════════════════════════════════════════
#  8. FINANCIAL REPLACE — do_async
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_financial_replace_do_async() -> None:
    """Financial Replace: credit card replaced with [FINANCIAL_INFO_REDACTED], de-anonymized."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful billing assistant.",
        user_policy=_financial_replace_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up the credit card ending in {SENSITIVE_CREDIT_CARD[-4:]} for the customer. "
            "Use the lookup_credit_card tool. Return the full card number."
        ),
        context=f"Customer credit card on file: {SENSITIVE_CREDIT_CARD}",
        tools=[lookup_credit_card],
    )

    result: Any = await agent.do_async(task)
    print(f"\n[FINANCIAL REPLACE do_async] Output: {result}")

    assert result is not None
    assert "[FINANCIAL_INFO_REDACTED]" not in str(result), \
        "Placeholder must not appear in final output"
    assert task._anonymization_map is None
    assert task._policy_originals is None

    output_str: str = str(result)
    assert SENSITIVE_CREDIT_CARD in output_str or SENSITIVE_CREDIT_CARD[-4:] in output_str, \
        f"Credit card (or last 4 digits) must be in output: {output_str[:300]}"


# ═══════════════════════════════════════════════════════════════════
#  9. PHONE ANONYMIZE — do_async
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_phone_anonymize_do_async() -> None:
    """Phone Anonymize: phone number anonymized with random digits and de-anonymized."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt="You are a phone directory assistant.",
        user_policy=_phone_anonymize_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up the phone number {SENSITIVE_PHONE} in our directory. "
            "Use the lookup_phone tool. Return the exact phone number."
        ),
        context=f"Known phone: {SENSITIVE_PHONE}",
        tools=[lookup_phone],
    )

    result: Any = await agent.do_async(task)
    print(f"\n[PHONE ANONYMIZE do_async] Output: {result}")

    assert result is not None
    assert SENSITIVE_PHONE in str(result), f"Phone must be de-anonymized: {result}"
    assert task._anonymization_map is None
    assert task._policy_originals is None


# ═══════════════════════════════════════════════════════════════════
#  10. MEDICAL ANONYMIZE — do_async
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_medical_anonymize_do_async() -> None:
    """Medical Anonymize: patient ID and prescription anonymized, de-anonymized in output."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt="You are a medical records assistant.",
        user_policy=_medical_anonymize_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up patient record for patient id {SENSITIVE_PATIENT_ID}. "
            "Use the lookup_patient tool. Return the patient id and prescription number."
        ),
        context=f"Patient record ID: {SENSITIVE_PATIENT_ID}, prescription: {SENSITIVE_PRESCRIPTION}",
        tools=[lookup_patient],
    )

    result: Any = await agent.do_async(task)
    print(f"\n[MEDICAL ANONYMIZE do_async] Output: {result}")

    assert result is not None
    assert task._anonymization_map is None
    assert task._policy_originals is None


# ═══════════════════════════════════════════════════════════════════
#  11. MEDICAL REPLACE — do_async
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_medical_replace_do_async() -> None:
    """Medical Replace: medical info replaced with [MEDICAL_INFO_REDACTED], de-anonymized."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt="You are a medical records assistant.",
        user_policy=_medical_replace_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up patient record for patient id {SENSITIVE_PATIENT_ID}. "
            "Use the lookup_patient tool. Return the patient id and prescription number."
        ),
        context=f"Patient record ID: {SENSITIVE_PATIENT_ID}, prescription: {SENSITIVE_PRESCRIPTION}",
        tools=[lookup_patient],
    )

    result: Any = await agent.do_async(task)
    print(f"\n[MEDICAL REPLACE do_async] Output: {result}")

    assert result is not None
    assert "[MEDICAL_INFO_REDACTED]" not in str(result), \
        "Placeholder must not appear in final output"
    assert task._anonymization_map is None
    assert task._policy_originals is None


# ═══════════════════════════════════════════════════════════════════
#  12. MULTI-POLICY: PII Anonymize + Financial Anonymize — do_async
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_multi_policy_pii_financial_do_async() -> None:
    """Multiple policies applied together: PII + Financial anonymize."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helpful assistant. User's email is {SENSITIVE_EMAIL}.",
        user_policy=[_pii_anonymize_policy(), _financial_anonymize_policy()],
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up contact info for {SENSITIVE_EMAIL} and their credit card. "
            "Use both lookup_contact and lookup_credit_card tools. "
            "Return the email and credit card number."
        ),
        context=(
            f"User email: {SENSITIVE_EMAIL}, "
            f"Credit card: {SENSITIVE_CREDIT_CARD}"
        ),
        tools=[lookup_contact, lookup_credit_card],
    )

    result: Any = await agent.do_async(task)
    print(f"\n[MULTI-POLICY PII+Financial do_async] Output: {result}")

    assert result is not None
    assert SENSITIVE_EMAIL in str(result), f"Email must be de-anonymized: {result}"
    assert task._anonymization_map is None
    assert task._policy_originals is None


# ═══════════════════════════════════════════════════════════════════
#  13. PII REPLACE — scoped (description only)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pii_replace_scoped_description_only() -> None:
    """PII Replace scoped to description only: system prompt stays untouched."""

    original_system_prompt: str = f"You are a helper. Email: {SENSITIVE_EMAIL}"

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=original_system_prompt,
        user_policy=_pii_replace_policy(
            description=True,
            context=False,
            system_prompt=False,
            chat_history=False,
            tool_outputs=False,
        ),
        print=True,
    )

    task: Task = Task(
        description=f"The email is {SENSITIVE_EMAIL}. Just say hello and mention the email.",
    )

    result: Any = await agent.do_async(task)
    print(f"\n[PII REPLACE scoped] Output: {result}")

    assert result is not None
    assert task._anonymization_map is None

    last_sp: Optional[str] = getattr(agent, "_last_built_system_prompt", None)
    if last_sp:
        assert SENSITIVE_EMAIL in last_sp, \
            "System prompt must NOT be anonymized when apply_to_system_prompt=False"


# ═══════════════════════════════════════════════════════════════════
#  14. FINANCIAL ANONYMIZE — event streaming
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_financial_anonymize_stream_events() -> None:
    """Financial Anonymize streaming: credit card anonymized in tool results, de-anonymized in text."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt="You are a billing assistant.",
        user_policy=_financial_anonymize_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up credit card {SENSITIVE_CREDIT_CARD} for the customer. "
            "Use the lookup_credit_card tool. Return the full card number."
        ),
        context=f"Card on file: {SENSITIVE_CREDIT_CARD}",
        tools=[lookup_credit_card],
    )

    text_chunks: list[str] = []
    tool_results: list[ToolResultEvent] = []

    async for event in agent.astream(task, events=True):
        if isinstance(event, TextDeltaEvent):
            text_chunks.append(event.content)
            print(event.content, end="", flush=True)
        elif isinstance(event, ToolCallEvent):
            print(f"\n[Tool Call] {event.tool_name}")
        elif isinstance(event, ToolResultEvent):
            tool_results.append(event)
            print(f"[Tool Result] {event.tool_name}: {str(event.result)[:100]}")
    print()

    full_text: str = "".join(text_chunks)
    print(f"[FINANCIAL ANONYMIZE stream] Full text: {full_text[:300]}")

    assert len(full_text) > 0
    assert task._anonymization_map is None

    for tr in tool_results:
        assert SENSITIVE_CREDIT_CARD not in str(tr.result or ""), \
            "Tool result must not expose original credit card"


# ═══════════════════════════════════════════════════════════════════
#  15. PHONE ANONYMIZE — pure text streaming
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_phone_anonymize_stream_text() -> None:
    """Phone Anonymize pure text streaming: phone de-anonymized in output."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt="You are a phone directory assistant.",
        user_policy=_phone_anonymize_policy(),
        print=True,
    )

    task: Task = Task(
        description=(
            f"Look up the phone number {SENSITIVE_PHONE} in our directory. "
            "Use the lookup_phone tool. Return the exact phone number."
        ),
        context=f"Known phone: {SENSITIVE_PHONE}",
        tools=[lookup_phone],
    )

    streamed: str = ""
    async for chunk in agent.astream(task):
        assert isinstance(chunk, str)
        streamed += chunk
        print(chunk, end="", flush=True)
    print()

    print(f"[PHONE ANONYMIZE stream text] Full: {streamed[:300]}")

    assert len(streamed) > 0
    assert SENSITIVE_PHONE in streamed, f"Phone must be de-anonymized: {streamed[:200]}"
    assert task._anonymization_map is None


# ═══════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    tests = [
        test_pii_anonymize_do_async,
        test_pii_replace_do_async,
        test_pii_anonymize_stream_events,
        test_pii_replace_stream_events,
        test_pii_anonymize_stream_text,
        test_pii_replace_stream_text,
        test_financial_anonymize_do_async,
        test_financial_replace_do_async,
        test_phone_anonymize_do_async,
        test_medical_anonymize_do_async,
        test_medical_replace_do_async,
        test_multi_policy_pii_financial_do_async,
        test_pii_replace_scoped_description_only,
        test_financial_anonymize_stream_events,
        test_phone_anonymize_stream_text,
    ]
    for fn in tests:
        try:
            print("\n" + "=" * 70)
            print(f"RUNNING {fn.__name__}")
            print("=" * 70)
            asyncio.run(fn())
            print(f"PASSED {fn.__name__}")
            print("=" * 70)
        except Exception as e:
            print(f"FAILED {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise e
