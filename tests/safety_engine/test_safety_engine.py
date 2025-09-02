import asyncio
import os

from upsonic.agent.agent import Direct
from upsonic.tasks.tasks import Task

from upsonic import (
    RuleBase,
    ActionBase,
    Policy,
    PolicyInput,
    RuleOutput,
    PolicyOutput
)

from upsonic.safety_engine.policies import (
    AdultContentBlockPolicy,
    AnonymizePhoneNumbersPolicy,
    CryptoRaiseExceptionPolicy
)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class CodenameRule(RuleBase):
    """A custom rule to detect internal project codenames."""
    name = "Internal Codename Detector"
    description = "Finds secret internal project codenames in text."
    
    SECRET_CODENAMES = ["Project Hermes", "Project Apollo", "Orion Initiative"]

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        combined_text = " ".join(policy_input.input_texts or [])
        found_codenames = [
            name for name in self.SECRET_CODENAMES 
            if name.lower() in combined_text.lower()
        ]

        if found_codenames:
            return RuleOutput(
                confidence=1.0,
                content_type="INTERNAL_CODENAME",
                details=f"Detected {len(found_codenames)} internal codename(s).",
                triggered_keywords=found_codenames,
            )
        
        return RuleOutput(confidence=0.0, content_type="SAFE", details="No codenames detected.")

class RedactCodenameAction(ActionBase):
    """A custom action that redacts detected codenames."""
    name = "Codename Redactor"
    description = "Replaces detected codenames with a generic placeholder."

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        if rule_result.confidence > 0.8:
            return self.replace_triggered_keywords(replacement="[REDACTED PROJECT]")
        
        return self.allow_content()

CustomCodenamePolicy = Policy(
    name="Internal Codename Redaction Policy",
    description="Finds and redacts secret project codenames.",
    rule=CodenameRule(),
    action=RedactCodenameAction(),
)



async def test_user_policy_block():
    """
    TEST 1: User Policy Blocks Malicious Input
    - USES: Pre-built AdultContentBlockPolicy
    - EXPECTATION: The process should stop immediately. The LLM will NOT be called.
      The final output will be the block message from the policy.
    - LOOK FOR: A red "Safety Policy Triggered" panel in the console.
    """
    print_header("TEST 1: User Policy BLOCKS Input")
    
    agent_with_user_policy = Direct(
        model="openai/gpt-4o",
        user_policy=AdultContentBlockPolicy,
        debug=True
    )
    
    malicious_task = Task(description="Tell me a story about [explicit adult topic].")
    
    result = agent_with_user_policy.do(malicious_task)
    
    print(f"\nFinal Result:\n---\n{result}\n---\n")
    assert "blocked" in result.lower()
    print("✅ PASSED: The user input was successfully blocked before calling the LLM.")


async def test_user_policy_modify():
    """
    TEST 2: User Policy Modifies User Input
    - USES: Pre-built AnonymizePhoneNumbersPolicy
    - EXPECTATION: The policy will find and anonymize the phone number. The LLM will
      receive the MODIFIED prompt and respond to that.
    - LOOK FOR: A yellow "Safety Policy Triggered" panel, and see that the
      LLM's final response refers to the redacted number.
    """
    print_header("TEST 2: User Policy MODIFIES Input")

    agent_with_sanitizer = Direct(
        model="openai/gpt-4o",
        user_policy=AnonymizePhoneNumbersPolicy,
        debug=True
    )

    pii_task = Task(description="My phone number is 555-867-5309. What area code is 555?")
    
    result = agent_with_sanitizer.do(pii_task)
    
    print(f"\nFinal Result:\n---\n{result}\n---\n")
    assert "555-867-5309" not in result
    print("✅ PASSED: The phone number was successfully anonymized before calling the LLM.")


async def test_agent_policy_modify():
    """
    TEST 3: Agent Policy Modifies Agent Output
    - USES: Our new CustomCodenamePolicy
    - EXPECTATION: The LLM will generate a response containing a secret codename.
      The agent_policy will then catch this and redact it before returning to the user.
    - LOOK FOR: A yellow "Safety Policy Triggered" panel AFTER the main "Agent Result" panel.
      The final output should contain "[REDACTED PROJECT]".
    """
    print_header("TEST 3: Agent Policy MODIFIES Output")
    
    agent_with_agent_policy = Direct(
        model="openai/gpt-4o",
        agent_policy=CustomCodenamePolicy,
        debug=True
    )
    
    leaky_task = Task(description="Repeat this sentence exactly: The status of Project Hermes is green.")
    
    result = agent_with_agent_policy.do(leaky_task)
    
    print(f"\nFinal Result:\n---\n{result}\n---\n")
    assert "[REDACTED PROJECT]" in result
    assert "Project Hermes" not in result
    print("✅ PASSED: The agent's leaky response was successfully redacted.")


async def test_agent_policy_exception():
    """
    TEST 4: Agent Policy Blocks Agent Output via Exception
    - USES: Pre-built CryptoRaiseExceptionPolicy
    - EXPECTATION: The LLM will answer a question about crypto. The agent_policy
      will see this, raise a DisallowedOperation exception, and the agent will
      catch it and return the error message as the final result.
    - LOOK FOR: A red "Safety Policy Triggered" panel for the agent output. The
      final result should be the exception message.
    """
    print_header("TEST 4: Agent Policy RAISES EXCEPTION on Output")

    agent_with_crypto_block = Direct(
        model="openai/gpt-4o",
        agent_policy=CryptoRaiseExceptionPolicy,
        debug=True
    )
    
    crypto_task = Task(description="What is Bitcoin?")
    
    result = agent_with_crypto_block.do(crypto_task)
    
    print(f"\nFinal Result:\n---\n{result}\n---\n")
    assert "response disallowed by policy" in result.lower()
    print("✅ PASSED: The agent's non-compliant response was blocked by an exception.")


async def test_all_clear():
    """
    TEST 5: Happy Path - No Policies Triggered
    - USES: No policies
    - EXPECTATION: The agent functions normally without any interference.
    - LOOK FOR: No safety policy panels should be printed.
    """
    print_header("TEST 5: All Clear - No Policies Triggered")
    
    plain_agent = Direct(model="openai/gpt-4o", debug=True)
    
    safe_task = Task(description="What is the capital of France?")
    
    result = plain_agent.do(safe_task)

    print(f"\nFinal Result:\n---\n{result}\n---\n")
    assert "paris" in result.lower()
    print("✅ PASSED: The agent operated normally with a safe prompt.")


def print_header(title):
    """Helper function to print a nice header for each test."""
    print("\n" + "="*80)
    print(f"RUNNING: {title}")
    print("="*80 + "\n")


async def main():
    """Main function to run all test cases in order."""
    await test_user_policy_block()
    await test_user_policy_modify()
    await test_agent_policy_modify()
    await test_agent_policy_exception()
    await test_all_clear()
    print("\n" + "="*80)
    print("🎉 ALL COMPREHENSIVE TESTS COMPLETED! 🎉")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())