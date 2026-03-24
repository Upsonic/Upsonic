"""
Skill-specific Safety Policies

Policies designed for validating skill content (instructions, references, scripts)
before they are loaded and executed by agents. These follow the same RuleBase + ActionBase
→ Policy pattern as all other safety_engine policies and can be mixed freely with them.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


class SkillPromptInjectionRule(RuleBase):
    """
    Keyword/pattern-based rule to detect prompt injection attempts
    inside skill instructions or references.

    Catches phrases like "ignore previous instructions", "you are now",
    "system prompt override", and similar adversarial patterns that could
    hijack the agent.
    """

    name = "Skill Prompt Injection Detection Rule"
    description = "Detects prompt injection patterns in skill content"
    language = "en"

    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)

        self.injection_patterns: List[str] = [
            # Direct instruction override
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"ignore\s+(all\s+)?prior\s+instructions",
            r"disregard\s+(all\s+)?previous\s+instructions",
            r"disregard\s+(all\s+)?prior\s+instructions",
            r"forget\s+(all\s+)?previous\s+instructions",
            r"forget\s+(all\s+)?prior\s+instructions",
            r"override\s+(all\s+)?previous\s+instructions",
            r"do\s+not\s+follow\s+(any\s+)?previous\s+instructions",

            # Role hijacking
            r"you\s+are\s+now\s+(?:a\s+)?(?:new|different)",
            r"your\s+new\s+(?:role|purpose|objective|goal)\s+is",
            r"act\s+as\s+(?:a\s+)?(?:different|new)\s+(?:agent|assistant|system)",
            r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?:different|new)",
            r"switch\s+(?:to|into)\s+(?:a\s+)?(?:different|new)\s+(?:mode|role|persona)",

            # System prompt manipulation
            r"system\s+prompt\s+(?:override|change|update|replace|inject)",
            r"reveal\s+(?:your\s+)?system\s+prompt",
            r"show\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions)",
            r"print\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)",
            r"output\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)",

            # Boundary escape
            r"end\s+of\s+(?:system\s+)?(?:prompt|instructions)",
            r"begin\s+(?:new\s+)?(?:system\s+)?(?:prompt|instructions)",
            r"\[/?system\]",
            r"</?system>",

            # Tool/action manipulation
            r"execute\s+(?:the\s+)?following\s+(?:command|code|script)\s+(?:silently|quietly|without)",
            r"run\s+(?:this|the\s+following)\s+(?:silently|quietly|without\s+(?:telling|showing|logging))",
            r"do\s+not\s+(?:log|report|mention|tell|show)",
        ]

        if options and "custom_patterns" in options:
            self.injection_patterns.extend(options["custom_patterns"])

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        combined_text = " ".join(policy_input.input_texts or [])
        if not combined_text.strip():
            return RuleOutput(
                confidence=0.0,
                content_type="NO_INJECTION",
                details="No content to check",
            )

        combined_lower = combined_text.lower()
        triggered: List[str] = []

        for pattern in self.injection_patterns:
            matches = re.findall(pattern, combined_lower)
            if matches:
                triggered.append(pattern)

        if not triggered:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_INJECTION",
                details="No prompt injection patterns detected",
            )

        confidence = min(1.0, len(triggered) * 0.5)
        return RuleOutput(
            confidence=confidence,
            content_type="PROMPT_INJECTION",
            details=f"Detected {len(triggered)} prompt injection pattern(s) in skill content",
            triggered_keywords=triggered,
        )


class SkillPromptInjectionRule_LLM(RuleBase):
    """LLM-powered rule to detect prompt injection in skill content."""

    name = "Skill Prompt Injection Detection Rule (LLM)"
    description = "Uses LLM to detect prompt injection patterns in skill content"
    language = "en"

    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        if not self.text_finder_llm:
            fallback = SkillPromptInjectionRule()
            return fallback.process(policy_input)

        try:
            triggered_keywords = self._llm_find_keywords_with_input(
                "PROMPT_INJECTION", policy_input
            )
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_INJECTION",
                    details="No prompt injection detected by LLM",
                )
            confidence = min(1.0, len(triggered_keywords) * 0.7)
            return RuleOutput(
                confidence=confidence,
                content_type="PROMPT_INJECTION",
                details=f"LLM detected {len(triggered_keywords)} prompt injection indicator(s)",
                triggered_keywords=triggered_keywords,
            )
        except Exception:
            fallback = SkillPromptInjectionRule()
            return fallback.process(policy_input)


class SkillSecretLeakRule(RuleBase):
    """
    Regex-based rule to detect secrets (API keys, tokens, passwords,
    connection strings) inside skill content.
    """

    name = "Skill Secret Leak Detection Rule"
    description = "Detects API keys, tokens, passwords and connection strings in skill content"
    language = "en"

    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)

        self.secret_patterns: Dict[str, str] = {
            # API keys with known prefixes
            "AWS_ACCESS_KEY": r"(?:AKIA|ASIA)[0-9A-Z]{16}",
            "AWS_SECRET_KEY": r"(?:aws_secret_access_key|aws_secret)\s*[=:]\s*[A-Za-z0-9/+=]{40}",
            "GITHUB_TOKEN": r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}",
            "GITLAB_TOKEN": r"glpat-[A-Za-z0-9\-_]{20,}",
            "SLACK_TOKEN": r"xox[boaprs]-[A-Za-z0-9\-]{10,}",
            "OPENAI_API_KEY": r"sk-[A-Za-z0-9]{20,}",
            "ANTHROPIC_API_KEY": r"sk-ant-[A-Za-z0-9\-]{20,}",
            "STRIPE_KEY": r"(?:sk|pk)_(?:test|live)_[A-Za-z0-9]{20,}",
            "SENDGRID_KEY": r"SG\.[A-Za-z0-9\-_]{22,}\.[A-Za-z0-9\-_]{43,}",
            "TWILIO_KEY": r"SK[a-f0-9]{32}",
            "GOOGLE_API_KEY": r"AIza[A-Za-z0-9\-_]{35}",
            "AZURE_KEY": r"[A-Za-z0-9+/]{86}==",

            # Generic secret patterns
            "GENERIC_PASSWORD": r"(?:password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
            "GENERIC_SECRET": r"(?:secret|token|api_key|apikey|api-key)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
            "GENERIC_BEARER": r"Bearer\s+[A-Za-z0-9\-_.~+/]{20,}",
            "PRIVATE_KEY_HEADER": r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
            "CONNECTION_STRING": r"(?:mongodb|postgres|mysql|redis|amqp)://[^\s'\"]{10,}",
        }

        if options and "custom_patterns" in options:
            self.secret_patterns.update(options["custom_patterns"])

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        combined_text = " ".join(policy_input.input_texts or [])
        if not combined_text.strip():
            return RuleOutput(
                confidence=0.0,
                content_type="NO_SECRET",
                details="No content to check",
            )

        triggered: List[str] = []

        for secret_type, pattern in self.secret_patterns.items():
            matches = re.findall(pattern, combined_text)
            for match in matches:
                triggered.append(f"{secret_type}:{match[:12]}***")

        if not triggered:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_SECRET",
                details="No secrets detected",
            )

        confidence = min(1.0, len(triggered) * 0.6)
        return RuleOutput(
            confidence=confidence,
            content_type="SECRET_LEAK",
            details=f"Detected {len(triggered)} potential secret(s) in skill content",
            triggered_keywords=triggered,
        )


class SkillSecretLeakRule_LLM(RuleBase):
    """LLM-powered rule to detect secrets in skill content."""

    name = "Skill Secret Leak Detection Rule (LLM)"
    description = "Uses LLM to detect secrets and credentials in skill content"
    language = "en"

    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        if not self.text_finder_llm:
            fallback = SkillSecretLeakRule()
            return fallback.process(policy_input)

        try:
            triggered_keywords = self._llm_find_keywords_with_input(
                "SECRET_CREDENTIAL_LEAK", policy_input
            )
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_SECRET",
                    details="No secrets detected by LLM",
                )
            confidence = min(1.0, len(triggered_keywords) * 0.7)
            return RuleOutput(
                confidence=confidence,
                content_type="SECRET_LEAK",
                details=f"LLM detected {len(triggered_keywords)} secret(s)",
                triggered_keywords=triggered_keywords,
            )
        except Exception:
            fallback = SkillSecretLeakRule()
            return fallback.process(policy_input)


class SkillCodeInjectionRule(RuleBase):
    """
    Pattern-based rule to detect dangerous code patterns inside skill
    instructions and scripts (e.g. ``eval()``, ``exec()``, ``os.system()``).
    """

    name = "Skill Code Injection Detection Rule"
    description = "Detects dangerous code execution patterns in skill content"
    language = "en"

    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)

        self.dangerous_patterns: List[str] = [
            # Python dangerous functions
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"\bcompile\s*\(",
            r"\b__import__\s*\(",
            r"\bgetattr\s*\(\s*__builtins__",
            r"\bos\.system\s*\(",
            r"\bos\.popen\s*\(",
            r"\bsubprocess\.(?:call|run|Popen|check_output|check_call)\s*\(",
            r"\bimportlib\.import_module\s*\(",

            # Shell injection
            r"\bos\.exec[lv]p?e?\s*\(",
            r"\bcommands\.getoutput\s*\(",
            r";\s*(?:rm|dd|mkfs|chmod|chown|curl|wget)\s+",

            # Pickle / deserialization
            r"\bpickle\.loads?\s*\(",
            r"\byaml\.(?:load|unsafe_load)\s*\(",
            r"\bjsonpickle\.decode\s*\(",
            r"\bshelve\.open\s*\(",

            # Network exfiltration
            r"\brequests\.(?:get|post|put|delete)\s*\(",
            r"\burllib\.request\.urlopen\s*\(",
            r"\bhttp\.client\.HTTP",
            r"\bsocket\.socket\s*\(",
        ]

        if options and "custom_patterns" in options:
            self.dangerous_patterns.extend(options["custom_patterns"])

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        combined_text = " ".join(policy_input.input_texts or [])
        if not combined_text.strip():
            return RuleOutput(
                confidence=0.0,
                content_type="NO_CODE_INJECTION",
                details="No content to check",
            )

        triggered: List[str] = []

        for pattern in self.dangerous_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                triggered.append(pattern)

        if not triggered:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_CODE_INJECTION",
                details="No dangerous code patterns detected",
            )

        confidence = min(1.0, len(triggered) * 0.4)
        return RuleOutput(
            confidence=confidence,
            content_type="CODE_INJECTION",
            details=f"Detected {len(triggered)} dangerous code pattern(s) in skill content",
            triggered_keywords=triggered,
        )


class SkillCodeInjectionRule_LLM(RuleBase):
    """LLM-powered rule to detect dangerous code patterns in skill content."""

    name = "Skill Code Injection Detection Rule (LLM)"
    description = "Uses LLM to detect dangerous code patterns in skill content"
    language = "en"

    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        if not self.text_finder_llm:
            fallback = SkillCodeInjectionRule()
            return fallback.process(policy_input)

        try:
            triggered_keywords = self._llm_find_keywords_with_input(
                "CODE_INJECTION_DANGEROUS_EXECUTION", policy_input
            )
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_CODE_INJECTION",
                    details="No dangerous code detected by LLM",
                )
            confidence = min(1.0, len(triggered_keywords) * 0.6)
            return RuleOutput(
                confidence=confidence,
                content_type="CODE_INJECTION",
                details=f"LLM detected {len(triggered_keywords)} dangerous code pattern(s)",
                triggered_keywords=triggered_keywords,
            )
        except Exception:
            fallback = SkillCodeInjectionRule()
            return fallback.process(policy_input)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class SkillBlockAction(ActionBase):
    """Blocks skill content with a descriptive message."""

    name = "Skill Block Action"
    description = "Blocks skill content that violates safety policies"
    language = "en"

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        if rule_result.confidence < 0.3:
            return self.allow_content()

        messages = {
            "PROMPT_INJECTION": (
                "Skill content blocked: prompt injection patterns detected. "
                f"Details: {rule_result.details}. "
                "Skill instructions must not attempt to override agent behavior."
            ),
            "SECRET_LEAK": (
                "Skill content blocked: secrets or credentials detected. "
                f"Details: {rule_result.details}. "
                "Remove all API keys, tokens, and passwords from skill content."
            ),
            "CODE_INJECTION": (
                "Skill content blocked: dangerous code patterns detected. "
                f"Details: {rule_result.details}. "
                "Skill content must not contain eval(), exec(), or similar patterns."
            ),
        }

        block_message = messages.get(
            rule_result.content_type,
            f"Skill content blocked by safety policy. Details: {rule_result.details}",
        )
        return self.raise_block_error(block_message)


class SkillBlockAction_LLM(ActionBase):
    """LLM-powered action to block skill content with contextual messages."""

    name = "Skill Block Action (LLM)"
    description = "Uses LLM to generate appropriate block messages for skill content"
    language = "en"

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        if rule_result.confidence < 0.3:
            return self.allow_content()

        reason = f"Skill safety violation ({rule_result.content_type}): {rule_result.details}"
        return self.llm_raise_block_error(reason)


class SkillRaiseExceptionAction(ActionBase):
    """Raises DisallowedOperation exception for unsafe skill content."""

    name = "Skill Raise Exception Action"
    description = "Raises DisallowedOperation exception for unsafe skill content"
    language = "en"

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        if rule_result.confidence < 0.3:
            return self.allow_content()

        exception_message = (
            f"DisallowedOperation: Skill safety policy violation "
            f"({rule_result.content_type}). {rule_result.details}"
        )
        return self.raise_exception(exception_message)


class SkillRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exceptions for unsafe skill content."""

    name = "Skill Raise Exception Action (LLM)"
    description = "Raises DisallowedOperation with LLM-generated message for unsafe skill content"
    language = "en"

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        if rule_result.confidence < 0.3:
            return self.allow_content()

        reason = f"Skill safety violation ({rule_result.content_type}): {rule_result.details}"
        return self.llm_raise_exception(reason)


# ---------------------------------------------------------------------------
# Pre-built Policies — Prompt Injection
# ---------------------------------------------------------------------------

## Prompt Injection Block
SkillPromptInjectionBlockPolicy = Policy(
    name="Skill Prompt Injection Block Policy",
    description="Blocks skill content containing prompt injection patterns",
    rule=SkillPromptInjectionRule(),
    action=SkillBlockAction(),
)

## Prompt Injection Block (LLM detection)
SkillPromptInjectionBlockPolicy_LLM = Policy(
    name="Skill Prompt Injection Block Policy (LLM)",
    description="Uses LLM to detect and block prompt injection in skill content",
    rule=SkillPromptInjectionRule_LLM(),
    action=SkillBlockAction(),
)

## Prompt Injection Block (LLM detection + LLM messaging)
SkillPromptInjectionBlockPolicy_LLM_Finder = Policy(
    name="Skill Prompt Injection Block Policy (LLM Finder)",
    description="Uses LLM for both detection and blocking messages for prompt injection",
    rule=SkillPromptInjectionRule_LLM(),
    action=SkillBlockAction_LLM(),
)

## Prompt Injection Raise Exception
SkillPromptInjectionRaiseExceptionPolicy = Policy(
    name="Skill Prompt Injection Raise Exception Policy",
    description="Raises exception when prompt injection is detected in skill content",
    rule=SkillPromptInjectionRule(),
    action=SkillRaiseExceptionAction(),
)

## Prompt Injection Raise Exception (LLM)
SkillPromptInjectionRaiseExceptionPolicy_LLM = Policy(
    name="Skill Prompt Injection Raise Exception Policy (LLM)",
    description="Raises LLM-generated exception for prompt injection in skill content",
    rule=SkillPromptInjectionRule_LLM(),
    action=SkillRaiseExceptionAction_LLM(),
)


# ---------------------------------------------------------------------------
# Pre-built Policies — Secret Leak
# ---------------------------------------------------------------------------

## Secret Leak Block
SkillSecretLeakBlockPolicy = Policy(
    name="Skill Secret Leak Block Policy",
    description="Blocks skill content containing API keys, tokens, or passwords",
    rule=SkillSecretLeakRule(),
    action=SkillBlockAction(),
)

## Secret Leak Block (LLM)
SkillSecretLeakBlockPolicy_LLM = Policy(
    name="Skill Secret Leak Block Policy (LLM)",
    description="Uses LLM to detect and block secrets in skill content",
    rule=SkillSecretLeakRule_LLM(),
    action=SkillBlockAction(),
)

## Secret Leak Block (LLM Finder)
SkillSecretLeakBlockPolicy_LLM_Finder = Policy(
    name="Skill Secret Leak Block Policy (LLM Finder)",
    description="Uses LLM for both detection and blocking messages for secret leaks",
    rule=SkillSecretLeakRule_LLM(),
    action=SkillBlockAction_LLM(),
)

## Secret Leak Raise Exception
SkillSecretLeakRaiseExceptionPolicy = Policy(
    name="Skill Secret Leak Raise Exception Policy",
    description="Raises exception when secrets are detected in skill content",
    rule=SkillSecretLeakRule(),
    action=SkillRaiseExceptionAction(),
)

## Secret Leak Raise Exception (LLM)
SkillSecretLeakRaiseExceptionPolicy_LLM = Policy(
    name="Skill Secret Leak Raise Exception Policy (LLM)",
    description="Raises LLM-generated exception for secret leaks in skill content",
    rule=SkillSecretLeakRule_LLM(),
    action=SkillRaiseExceptionAction_LLM(),
)


# ---------------------------------------------------------------------------
# Pre-built Policies — Code Injection
# ---------------------------------------------------------------------------

## Code Injection Block
SkillCodeInjectionBlockPolicy = Policy(
    name="Skill Code Injection Block Policy",
    description="Blocks skill content containing dangerous code patterns",
    rule=SkillCodeInjectionRule(),
    action=SkillBlockAction(),
)

## Code Injection Block (LLM)
SkillCodeInjectionBlockPolicy_LLM = Policy(
    name="Skill Code Injection Block Policy (LLM)",
    description="Uses LLM to detect and block dangerous code in skill content",
    rule=SkillCodeInjectionRule_LLM(),
    action=SkillBlockAction(),
)

## Code Injection Block (LLM Finder)
SkillCodeInjectionBlockPolicy_LLM_Finder = Policy(
    name="Skill Code Injection Block Policy (LLM Finder)",
    description="Uses LLM for both detection and blocking messages for code injection",
    rule=SkillCodeInjectionRule_LLM(),
    action=SkillBlockAction_LLM(),
)

## Code Injection Raise Exception
SkillCodeInjectionRaiseExceptionPolicy = Policy(
    name="Skill Code Injection Raise Exception Policy",
    description="Raises exception when dangerous code is detected in skill content",
    rule=SkillCodeInjectionRule(),
    action=SkillRaiseExceptionAction(),
)

## Code Injection Raise Exception (LLM)
SkillCodeInjectionRaiseExceptionPolicy_LLM = Policy(
    name="Skill Code Injection Raise Exception Policy (LLM)",
    description="Raises LLM-generated exception for dangerous code in skill content",
    rule=SkillCodeInjectionRule_LLM(),
    action=SkillRaiseExceptionAction_LLM(),
)
