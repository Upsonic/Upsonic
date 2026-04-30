---
name: safety-engine
description: Use when working with Upsonic's content-filtering and policy-enforcement layer that scans agent inputs/outputs/tool calls and applies allow/block/replace/anonymize/raise actions. Use when a user asks to add a safety policy to an Agent, build a custom Rule/Action/Policy, anonymize PII or de-anonymize LLM responses, block crypto/adult/hate/phishing/fraud content, redact API keys or credentials, validate tool registration or tool calls, configure feedback retry loops, or wire up PolicyManager / ToolPolicyManager. Trigger when the user mentions safety_engine, RuleBase, ActionBase, Policy, PolicyInput, RuleOutput, PolicyOutput, PolicyManager, ToolPolicyManager, DisallowedOperation, UpsonicLLMProvider, Anonymizer, StreamDeanonymizer, transformation_map, user_policy, agent_policy, tool_policy, PII, financial, medical, legal, technical_security, cybersecurity, fraud_detection, phishing, insider_threat, profanity, Detoxify, skill_policies, prompt injection, secret leak, code injection, HarmfulToolRule, MaliciousToolCallRule, CryptoBlockPolicy, PIIAnonymizePolicy, AdultContentBlockPolicy, AnonymizePhoneNumbersPolicy, or content moderation.
---

# `src/upsonic/safety_engine/` — Content Filtering and Policy Enforcement

## 1. What this folder is

The `safety_engine` package is Upsonic's **content-filtering and policy-enforcement layer**. It sits between the user / agent / tool boundary and the LLM, scanning text for sensitive, harmful, or otherwise disallowed content and applying a configurable response (allow, block, replace, anonymize, raise an exception).

Conceptually it is a small **rule engine** built around three abstractions:

| Concept     | Class       | Purpose |
|-------------|-------------|---------|
| Rule        | `RuleBase`  | Inspect a `PolicyInput` and return a `RuleOutput` (confidence, content type, triggered keywords). |
| Action      | `ActionBase`| Decide what to do with the content given a `RuleOutput`: allow / block / replace / anonymize / raise. |
| Policy      | `Policy`    | Pair one rule with one action plus optional LLM helpers and per-source scoping flags. |

Detection runs in two flavors for nearly every policy family:

- **Pattern / keyword detection** — fast, deterministic regex + word-list scanning. No LLM cost.
- **LLM detection** — `*_LLM_Finder` rules call `UpsonicLLMProvider.find_keywords(...)` for context-aware extraction, with automatic fallback to the pattern rule when no LLM is wired.

The engine also owns:

- A reversible **anonymization layer** that replaces detected values with random characters of the same shape (digits → digits, letters → letters, special chars preserved) and stores a `transformation_map` so the LLM's response can be de-anonymized.
- A `DisallowedOperation` exception used to abort the agent run from inside an action.
- A custom `UpsonicLLMProvider` that wraps `upsonic.agent.agent.Agent` for keyword extraction, language detection, translation, block-message generation, tool-safety analysis, and policy feedback.

The pre-built policies cover **adult content, crypto, sensitive social/hate speech, phone numbers, PII, financial information, medical / PHI, legal / confidential, technical security secrets, cybersecurity threats, fraud / scams, phishing, insider threats, profanity (Detoxify), tool safety, and skill safety (prompt injection / secret leak / code injection)**.

The engine is consumed by two managers in `src/upsonic/agent/`:

- `PolicyManager` — runs user-input or agent-output policies, supports a feedback-retry loop, and aggregates transformation maps across policies.
- `ToolPolicyManager` — runs tool-registration and tool-call policies that use the same `Policy` shape but pass tool metadata through `PolicyInput.extra_data`.

## 2. Folder layout (tree)

```text
src/upsonic/safety_engine/
├── __init__.py                          # Lazy re-exports of base/models/exceptions/policies
├── anonymization.py                     # Anonymizer, AnonymizationResult, StreamDeanonymizer, helpers
├── exceptions.py                        # DisallowedOperation
├── models.py                            # PolicyInput, RuleOutput, PolicyOutput (+ aliases)
├── base/
│   ├── __init__.py                      # Lazy re-exports
│   ├── policy.py                        # Policy class (rule + action + LLM wiring)
│   ├── rule_base.py                     # RuleBase ABC + LLM keyword helpers
│   └── action_base.py                   # ActionBase ABC + allow/block/replace/anonymize/raise primitives
├── llm/
│   ├── __init__.py                      # Lazy re-export of UpsonicLLMProvider
│   └── upsonic_llm.py                   # UpsonicLLMProvider + Pydantic response schemas
└── policies/
    ├── __init__.py                      # `from .* import *` aggregator
    ├── adult_content_policies.py
    ├── crypto_policies.py
    ├── cybersecurity_policies.py
    ├── financial_policies.py
    ├── fraud_detection_policies.py
    ├── insider_threat_policies.py
    ├── legal_policies.py
    ├── medical_policies.py
    ├── phishing_policies.py
    ├── phone_policies.py
    ├── pii_policies.py
    ├── profanity_policies.py            # Detoxify-backed (optional dep)
    ├── sensitive_social_policies.py
    ├── skill_policies.py                # Prompt-injection / secret-leak / code-injection
    ├── technical_policies.py            # API keys, passwords, tokens, certs
    └── tool_safety_policies.py          # Harmful tool / malicious tool call (LLM-driven)
```

## 3. Top-level files

### 3.1 `__init__.py` — lazy public surface

The package uses a **lazy `__getattr__` pattern** to avoid eagerly importing heavy dependencies (Detoxify, the agent stack, etc.). Five lookup tables are built on demand:

```python
def _get_base_classes():     # RuleBase, ActionBase, Policy
def _get_model_classes():    # PolicyInput, RuleOutput, PolicyOutput, ...
def _get_exception_classes():# DisallowedOperation
def _get_anonymization_classes(): # anonymize_content, deanonymize_content, StreamDeanonymizer, ...
def _get_policy_classes():   # All pre-built Policy instances (~150)

def __getattr__(name):
    for table in (_get_base_classes(), _get_model_classes(), ...):
        if name in table:
            return table[name]
    raise AttributeError(...)
```

`__all__` enumerates every exportable symbol so `from upsonic.safety_engine import *` works without triggering imports for things you do not use. `__version__ = "0.1.0"`.

### 3.2 `models.py` — data contracts

```python
class PolicyInput(BaseModel):
    input_texts: Optional[List[str]] = None
    input_images: Optional[List[str]] = None
    input_videos: Optional[List[str]] = None
    input_audio: Optional[List[str]] = None
    input_files: Optional[List[str]] = None
    extra_data: Optional[Dict[str, Any]] = None
    existing_transformation_map: Optional[Dict[int, Dict[str, str]]] = None

class RuleOutput(BaseModel):
    confidence: float
    content_type: str
    details: str
    triggered_keywords: Optional[List[str]] = None

class PolicyOutput(BaseModel):
    output_texts: Optional[List[str]] = None
    output_images: Optional[List[str]] = None
    output_videos: Optional[List[str]] = None
    output_audio: Optional[List[str]] = None
    output_files: Optional[List[str]] = None
    action_output: Optional[Dict[str, Any]] = None
    transformation_map: Optional[Dict[int, Dict[str, str]]] = None

# Backward-compat aliases
RuleInput = PolicyInput
ActionResult = PolicyOutput
ActionOutput = PolicyOutput
```

A few practical notes:

- `input_texts` is the primary channel; nearly every rule does `combined_text = " ".join(policy_input.input_texts or [])`.
- `extra_data` is the side channel for **non-text** inputs. Tool-safety rules read `tool_name`, `tool_description`, `tool_parameters_schema`, and `tool_call_args` from it.
- `existing_transformation_map` lets a downstream policy *extend* a map produced by an earlier policy so multiple anonymization passes stay consistent.
- `triggered_keywords` carries either raw values or **typed strings** like `"EMAIL:john@example.com"` or `"CREDIT_CARD:4111111111111111"`. The `:` prefix is a soft type marker the action layer parses to drive replacement / anonymization (see `replace_triggered_keywords`).
- The `action_output` dict on `PolicyOutput` always contains `action_taken` (`"ALLOW" | "BLOCK" | "REPLACE" | "ANONYMIZE"`), `success: bool`, and a human-readable `message`.

### 3.3 `exceptions.py`

```python
class DisallowedOperation(Exception):
    """Exception raised when an operation is not allowed by policy"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
```

A single, intentional exception type. It is raised by `ActionBase.raise_exception` and `ActionBase.llm_raise_exception` and is caught explicitly by `PolicyManager` / `ToolPolicyManager` so a single misbehaving policy aborts the run cleanly.

### 3.4 `anonymization.py` — reversible random-shape anonymization

The anonymizer's job is to turn `"My phone is 555-123-4567"` into something like `"My phone is 430-779-1195"`, send the masked text to the LLM, and de-anonymize the LLM's response back to the original number.

Key surface:

```python
@dataclass
class AnonymizationResult:
    anonymized_content: str
    transformation_map: Dict[int, Dict[str, str]]  # idx -> {"original", "anonymous"}
    anonymized_count: int

class Anonymizer:
    def anonymize(self, content: str, triggered_keywords: List[str]) -> AnonymizationResult: ...
    def anonymize_multiple(self, contents: List[str], triggered_keywords: List[str]): ...

def anonymize_content(content, triggered_keywords, existing_map=None) -> AnonymizationResult
def anonymize_contents(contents, triggered_keywords, existing_map=None)
def deanonymize_content(content, transformation_map) -> str
def deanonymize_contents(contents, transformation_map) -> List[str]
def deanonymize_mapping_content(content, transformation_map)  # recurses dict/list/str

class StreamDeanonymizer:  # buffer-based, used by streaming pipeline
    def process_token(self, token: str) -> str: ...
    def flush(self) -> str: ...
```

Key behaviors:

- **Format-preserving substitution.** `_generate_random_replacement` walks each character: digits → `random.randint(0,9)`, uppercase letters → `random.choice(ascii_uppercase)`, lowercase → `ascii_lowercase`, everything else passes through.
- **Idempotent within a run.** `_anonymized_cache` and `_transformation_map` ensure the same input value always produces the same anonymous value. This is critical because the LLM's response must contain the same fake string to be reversible.
- **Type-prefix stripping.** `_get_value_from_keyword` understands the `"EMAIL:..."` / `"CREDIT_CARD:..."` form and drops the prefix. Items prefixed `"PII_KEYWORD:"` are skipped — those are *detection markers*, not real PII values.
- **Long-first de-anonymization.** `deanonymize_content` sorts the map by descending anonymous-string length so longer matches are replaced before shorter ones (otherwise a substring of a longer fake value could be replaced first and corrupt the result).
- **Streaming support.** `StreamDeanonymizer` keeps a buffer and only emits text once it can prove no anonymous value is being split across token boundaries — important for tool outputs that stream back through the agent.

### 3.5 `base/policy.py` — `Policy`

A `Policy` is the configurable unit of safety. It owns one rule, one action, an optional language hint (`"en"`, `"tr"`, ..., or `"auto"`), and three optional LLM provider slots:

| Slot                   | Used by                                  |
|------------------------|------------------------------------------|
| `language_identify_llm`| `ActionBase._detect_content_language`    |
| `base_llm`             | `ActionBase._translate`, `llm_raise_block_error`, `llm_raise_exception` |
| `text_finder_llm`      | `RuleBase._llm_find_keywords_with_input` |

Each slot can be passed directly as an `UpsonicLLMProvider` instance, or as a `*_model="gpt-4o"` string — in which case the constructor will lazily build a provider with that model. If `text_finder_llm` is provided, `Policy.__init__` propagates it onto the rule (`self.rule.text_finder_llm = text_finder_llm`).

Per-source scoping flags govern *which* parts of the prompt a policy is applied to:

```python
apply_to_description: Optional[bool]
apply_to_context: Optional[bool]
apply_to_system_prompt: Optional[bool]
apply_to_chat_history: Optional[bool]
apply_to_tool_outputs: Optional[bool]
```

`None` means "inherit from Task / Agent" (resolved by `PolicyManager.resolve_policy_scope`). A value of `True` / `False` overrides everything below it.

Execution surface:

```python
def check(self, policy_input) -> RuleOutput
async def check_async(self, policy_input) -> RuleOutput

def execute(self, policy_input) -> tuple[RuleOutput, ActionOutput, PolicyOutput]
async def execute_async(self, policy_input) -> tuple[RuleOutput, ActionOutput, PolicyOutput]
```

The async paths use `process_async` / `execute_action_async` if the rule / action defines them, otherwise they fall back to `asyncio.to_thread` so blocking detectors (regex, Detoxify) do not stall the event loop.

### 3.6 `base/rule_base.py` — `RuleBase`

```python
class RuleBase(ABC):
    name: str = "Base Rule"
    description: str = "Base rule description"
    language: str = "en"

    def __init__(self, options=None, text_finder_llm=None):
        self.options = options or {}
        self.text_finder_llm = text_finder_llm

    @abstractmethod
    def process(self, policy_input: PolicyInput) -> RuleOutput: ...

    async def process_async(self, policy_input): ...   # to_thread by default
    def _llm_find_keywords_with_input(self, content_type: str, policy_input): ...
    async def _llm_find_keywords_with_input_async(self, content_type, policy_input): ...
```

Subclasses are free to use any detection technique they want. The convention seen across the policies is:

1. Concatenate `policy_input.input_texts` into a single string.
2. Walk a series of regex / keyword groups, collecting `triggered_items`.
3. Bucket the matches by severity (critical / high / medium / low).
4. Compute a weighted confidence and clamp with `min(1.0, ...)`.
5. Return a `RuleOutput(confidence, content_type, details, triggered_keywords)`.

The LLM-finder pairs (`*_LLM_Finder`) instead delegate to `_llm_find_keywords_with_input("PII", policy_input)` which spins up an `UpsonicLLMProvider` agent, asks it to "extract only explicit instances", and falls back to the pattern rule if the LLM returns nothing or raises.

### 3.7 `base/action_base.py` — `ActionBase`

The action base provides a small, opinionated toolbox; subclasses just decide which primitive to call given a `RuleOutput`.

State carried during execution:

```python
self.rule_result: Optional[RuleOutput]
self.original_content: Optional[List[str]]
self.transformation_map: Dict[int, Dict[str, str]]
self.transformation_index: int
self.detected_language: str  # "en" by default
```

`execute_action` (and its `_async` twin) is the entry point invoked by `Policy.execute`. It snapshots the original content, seeds the transformation map from `existing_transformation_map`, resolves language (explicit, `"auto"` → LLM detection, or `"en"` fallback), and calls the abstract `action(rule_result)` hook.

Primitives every concrete action can call:

| Primitive | Resulting `action_taken` | Notes |
|-----------|--------------------------|-------|
| `allow_content()` | `"ALLOW"` | Returns original content unchanged. |
| `raise_block_error(message)` | `"BLOCK"` | Translates `message` to `detected_language` then returns it as `output_texts[0]`. |
| `replace_triggered_keywords(replacement)` | `"REPLACE"` | Strips type prefixes, regex-replaces case-insensitively, records each substitution in `transformation_map`. |
| `anonymize_triggered_keywords()` | `"ANONYMIZE"` | Uses `_generate_unique_replacement` for format-preserving, idempotent random replacement. Skips `"PII_KEYWORD:"` markers. |
| `llm_raise_block_error(reason)` | `"BLOCK"` | Calls `UpsonicLLMProvider.generate_block_message(reason, language=detected_language)` for a contextual message. |
| `raise_exception(message)` | — (raises) | Raises `DisallowedOperation(message)` directly. |
| `llm_raise_exception(reason)` | — (raises) | LLM-generates the message, then raises. |

`_generate_unique_replacement` has one subtle bit: when the keyword and a leading-space variant both occur (e.g. `"555-1234"` and `" 555-1234"`), it derives a consistent anonymous twin so the LLM cannot produce an un-matchable bare variant.

Translation is opportunistic: `_translate(text, target_language)` only invokes the LLM when `self.__class__.language != target_language`, so English-by-default actions running in English mode pay zero LLM cost.

### 3.8 `llm/upsonic_llm.py` — `UpsonicLLMProvider`

A thin wrapper around `upsonic.agent.agent.Agent` that exposes a small "policy primitives" API. Each method:

1. Constructs an internal `Task` with a Pydantic `response_format` (so the LLM is forced into a structured shape).
2. Calls `self.agent.do(task)` (or `do_async`) to run it.
3. Falls back to a sensible default on exceptions instead of letting them propagate.
4. Accumulates `RunUsage` from each sub-agent run via `_accumulate_usage_from_output` so `PolicyManager.drain_accumulated_usage()` can roll cost up to the parent.

Pydantic response schemas defined in this file:

```python
class KeywordDetectionResponse(BaseModel):
    detected_keywords: List[str]
    confidence: float
    reasoning: str

class BlockMessageResponse(BaseModel):
    block_message: str
    severity: str
    reasoning: str

class AnonymizationResponse(BaseModel):
    anonymized_content: str
    anonymized_parts: List[str]
    reasoning: str

class LanguageDetectionResponse(BaseModel):
    language_code: str   # ISO 639-1
    language_name: str
    confidence: float

class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    confidence: float

class ToolSafetyAnalysisResponse(BaseModel):
    is_harmful: bool = False
    is_malicious: bool = False
    confidence: float
    reasons: List[str]
    threat_categories: List[str] = []
    suspicious_args: List[str] = []
    recommendation: str

class PolicyFeedbackResponse(BaseModel):
    feedback_message: str
    suggested_approach: str
    violation_type: str
    severity: str
```

Public methods (each has a `_async` twin):

| Method | Used by |
|--------|---------|
| `find_keywords(content_type, text, language="en")` | `RuleBase._llm_find_keywords_with_input` |
| `generate_block_message(reason, language="en")` | `ActionBase.llm_raise_block_error`, `llm_raise_exception` |
| `anonymize_content(text, keywords, language="en")` | Available for advanced custom actions; the built-in pipeline uses the deterministic anonymizer instead |
| `detect_language(text)` | `ActionBase._detect_content_language` |
| `translate_text(text, target_language)` | `ActionBase._translate` (built-in language map covers ~80 codes) |
| `analyze_tool_safety(tool_info, analysis_type)` | `HarmfulToolRule_LLM`, `MaliciousToolCallRule_LLM` (`"HARMFUL_FUNCTIONALITY"` vs `"MALICIOUS_CALL"`) |
| `generate_policy_feedback(...)` | `PolicyManager._generate_feedback_if_enabled` |
| `drain_accumulated_usage()` | Cost rollup |

The translation method has a robustness trick: if the model returns the source text verbatim, it appends a warning to the prompt and tries once more, then falls back to a hard-coded TR-EN dictionary for two well-known crypto-block messages.

## 4. Subfolders

### 4.1 `base/`

Three files (`rule_base.py`, `action_base.py`, `policy.py`) plus a lazy-loading `__init__.py`. Already covered in §3.5–§3.7.

### 4.2 `llm/`

Two files (`__init__.py`, `upsonic_llm.py`). Already covered in §3.8. Only the lazy export of `UpsonicLLMProvider` is exposed.

### 4.3 `policies/` — pre-built policies

The `policies/__init__.py` is a star-import aggregator over every individual `*_policies.py` module so `from upsonic.safety_engine.policies import PIIBlockPolicy` works directly.

Every policy module follows the same skeleton:

1. **One pattern rule.** Defines lists of regex patterns / keyword sets, weights matches by severity, returns a `RuleOutput` with `triggered_keywords` formatted as `"<TYPE>:<value>"`.
2. **One LLM finder rule** (sometimes more). Inherits from `RuleBase`, calls `_llm_find_keywords_with_input("CONTENT_TYPE_LABEL", policy_input)`, falls back to the pattern rule on missing-LLM or exception.
3. **A small fan of action classes.** Most modules ship a *Block*, *Anonymize*, *Replace*, and *RaiseException* variant, plus `_LLM` versions that route the message through `UpsonicLLMProvider.generate_block_message`.
4. **Pre-built `Policy` instances** — each combination of detection method × action variant gets its own named module-level constant, ready to drop into `Agent(user_policy=..., agent_policy=...)`.

The shared confidence threshold for actions is `0.3` — anything under that returns `allow_content()`. Crypto and phone policies use `0.8` instead because their detection is exact (regex / keyword equality on currency tickers and digit sequences), so a hit is essentially binary.

#### 4.3.1 Adult content (`adult_content_policies.py`)

Detects explicit sexual material, suggestive language, age-restricted terms, and adult-platform names. The rule splits matches into three buckets:

| Bucket | Weight | Examples |
|--------|--------|----------|
| Explicit keywords | 0.9 | "porn", "explicit", "intercourse", platform names |
| Adult patterns | 0.85 | regex like `\b(?:looking for|seeking)\s+(?:sex|hookup|...)` |
| Suggestive | 0.4 | "flirt", "kiss", "romance" |
| Age verification | 0.7 | "18+", "must be 18", "age of consent" |

Content type tag is one of `"EXPLICIT_ADULT_CONTENT" | "AGE_RESTRICTED_CONTENT" | "SUGGESTIVE_CONTENT" | "SAFE_CONTENT"`.

| Pre-built Policy | Detection | Action |
|------------------|-----------|--------|
| `AdultContentBlockPolicy` | Pattern | Static block message |
| `AdultContentBlockPolicy_LLM` | Pattern | LLM-generated block message |
| `AdultContentBlockPolicy_LLM_Finder` | LLM | Static block message |
| `AdultContentRaiseExceptionPolicy` | Pattern | `DisallowedOperation` |
| `AdultContentRaiseExceptionPolicy_LLM` | Pattern | LLM-generated `DisallowedOperation` |

#### 4.3.2 Crypto (`crypto_policies.py`)

Word-boundary keyword detection (`bitcoin`, `btc`, `ethereum`, ..., `nft`, `defi`, `proof of work`, ...) with **explicit false-positive patterns** (`"currency exchange"`, `"foreign exchange"`, `"student exchange"`, ...). Confidence is `1.0` on any hit, `0.0` otherwise. The action threshold is `>= 0.8`.

| Pre-built Policy | Detection | Action |
|------------------|-----------|--------|
| `CryptoBlockPolicy` | Pattern | Static block |
| `CryptoBlockPolicy_LLM_Block` | Pattern | LLM block message |
| `CryptoBlockPolicy_LLM_Finder` | LLM finder | Static block |
| `CryptoReplace` | Pattern | Replace each crypto term with `"NO_CRYPTO_CONTENT"` |
| `CryptoRaiseExceptionPolicy` | Pattern | `DisallowedOperation` |
| `CryptoRaiseExceptionPolicy_LLM_Raise` | Pattern | LLM-generated `DisallowedOperation` |

#### 4.3.3 Sensitive social / hate speech (`sensitive_social_policies.py`)

Three buckets:
- Hate-speech keywords (slurs are stored with asterisks like `"n*gger"` and the regex expands `*` to `[a-z*]` to catch coded variants).
- Hate-speech regex patterns ("kill all X", "go back to Y", "Hitler was right", ...).
- Discriminatory context keywords ("racial profiling", "white privilege", "antisemitism", ...).

Weights: hate keywords 0.9, hate patterns 0.95, discriminatory 0.3. Six pre-built policies parallel to the adult-content table.

#### 4.3.4 Phone numbers (`phone_policies.py`)

Pure regex `r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'` plus an LLM finder. The action is always anonymization (digits randomized, format preserved). Two pre-built policies: `AnonymizePhoneNumbersPolicy` and `AnonymizePhoneNumbersPolicy_LLM_Finder`.

#### 4.3.5 PII (`pii_policies.py`)

Highest-fan-out rule. Pattern groups for emails, phones (multi-format), SSN (excluding invalid prefixes 000/666/9xx), credit cards (Visa / MC / Amex / Discover prefixes), addresses (street suffixes + ZIP / UK postal), DOB (MM/DD/YYYY, DD/MM/YYYY, YYYY/MM/DD), driver's licenses, passports, IPv4 / IPv6, MAC, plus a long PII keyword list with email-context false-positive filtering.

Triggered items use **typed prefixes** (`"EMAIL:..."`, `"PHONE:..."`, `"SSN:..."`, `"PII_KEYWORD:..."`), so the action layer can decide which to anonymize, replace, or skip. `"PII_KEYWORD:"` items are detection markers and are skipped during anonymization.

| Pre-built Policy | Detection | Action |
|------------------|-----------|--------|
| `PIIBlockPolicy` | Pattern | Block |
| `PIIBlockPolicy_LLM` | Pattern | LLM block |
| `PIIBlockPolicy_LLM_Finder` | LLM | Block |
| `PIIAnonymizePolicy` | Pattern | Anonymize (random shape) |
| `PIIReplacePolicy` | Pattern | Replace with `"[PII_REDACTED]"` |
| `PIIRaiseExceptionPolicy` | Pattern | `DisallowedOperation` |
| `PIIRaiseExceptionPolicy_LLM` | Pattern | LLM `DisallowedOperation` |

#### 4.3.6 Financial (`financial_policies.py`)

Patterns for credit cards (Luhn-shaped), CVV/CVC, bank accounts, IBAN, SWIFT/BIC, routing numbers, SSN, EIN/TIN, financial statement amounts (balance, APR, ...), investment account numbers, and crypto wallets (BTC `1...` / `3...` / `bc1...`, ETH `0x...`).

Severity buckets: critical (CC / SSN / bank / routing) 1.0, high (CVV / TaxID / Crypto) 0.8, medium (statements / investment) 0.6, low (keyword) 0.3.

Seven pre-built policies parallel to PII (Block, Block_LLM, Block_LLM_Finder, Anonymize, Replace=`"[FINANCIAL_INFO_REDACTED]"`, RaiseException, RaiseException_LLM).

#### 4.3.7 Medical (`medical_policies.py`)

HIPAA-oriented. Patterns for medical record numbers, insurance / Medicare / group numbers, prescription numbers, NDC codes, DEA numbers, lab order / specimen numbers, medical device serials, ICD-10 / ICD-9 / CPT codes. Plus a large keyword list (conditions, professionals, mental-health terms, reproductive-health terms, HIPAA terminology) and **sensitive medical regex** for HIV/AIDS, mental-health, substance-abuse, sexual-health context.

Severity: critical (records / insurance / sensitive) 1.0, high (prescription / lab / device) 0.8, medium (codes) 0.6, low (keywords) 0.3.

Seven policies, same shape as Financial.

#### 4.3.8 Legal (`legal_policies.py`)

Patterns for contract / case / docket numbers, confidentiality markers, attorney-client privilege, patents, trademarks, copyrights, trade secrets, business-sensitive (revenue, customer list, business plan, ...), legal proceedings, IP filings, sensitive-legal regex (whistleblower, internal investigation, ...). Replace placeholder is `"[LEGAL_INFO_REDACTED]"`. Seven policies, same shape as Financial.

#### 4.3.9 Technical security (`technical_policies.py`)

Detects credentials and configuration secrets:

- API keys: `sk-...` (OpenAI), `AKIA...` (AWS), `AIza...` (Google), `ghp_/gho_/ghu_/ghs_/ghr_` (GitHub), generic `apikey=...`.
- Passwords: `password=...`, DB connection URIs with embedded credentials, base64-shaped secrets.
- Tokens: JWT (`eyJ...eyJ...`), OAuth bearer, session, CSRF.
- Certificates: PEM blocks for private keys, certs, CSRs, SSH keys.
- Database URIs (`mysql://user:pass@...`, `jdbc:postgresql://...`).
- Cloud (`aws_access_key_id=...`, `azure_storage_account_key=...`, ...).
- Config / env files.

Replace placeholder is `"[TECHNICAL_SECURITY_REDACTED]"`. Seven policies, same shape.

#### 4.3.10 Cybersecurity (`cybersecurity_policies.py`)

Threat-oriented detector. Buckets: malware names, malicious code patterns (shellcode, SQLi, XSS), attack vectors (DDoS, MITM, ...), suspicious file extensions, network threats, social engineering ("urgent action required", "click here", ...), crypto threats (mining botnet, ransom payment), system-compromise indicators, vulnerabilities (CVE-YYYY-NNNN, buffer overflow, SQLi, deserialization, SSRF, ...).

Weights: critical (malware / compromise / vulnerability) 1.0, high (network / social-eng / crypto) 0.8, medium (suspicious file) 0.6. Replace placeholder is `"[SECURITY_THREAT_REDACTED]"`. Seven policies.

#### 4.3.11 Fraud detection (`fraud_detection_policies.py`)

Buckets: fraud keywords, financial scam phrases ("guaranteed risk-free returns", "limited time offer"), identity-theft phrases, romance / dating scam patterns, investment / crypto scam patterns, tech-support scams, charity scams, urgency tactics, suspicious payment-method demands (gift cards / wire / Bitcoin only). Weights: critical (fraud keyword / identity / investment) 1.0, high (financial / romance / tech-support) 0.8, medium (charity / urgency / payment) 0.6. Replace placeholder `"[FRAUD_REDACTED]"`. Seven policies.

#### 4.3.12 Phishing (`phishing_policies.py`)

Phishing-specific patterns — urgent action, suspicious links / shorteners, credential harvesting forms, brand impersonation (Microsoft / Apple / Google / IRS / banks), prize / reward bait, fake account-security alerts, payment-failed scams, social-media account-suspended scams, fake software updates / antivirus expiry. Critical (phishing keyword / credential harvesting / impersonation) 1.0, high (urgent / link / account-security) 0.8, medium (prize / payment / social / technical) 0.6. Replace placeholder `"[PHISHING_REDACTED]"`. Seven policies.

#### 4.3.13 Insider threat (`insider_threat_policies.py`)

Behavioral patterns. Buckets: insider-threat keywords, data-exfiltration ("download all the company database", "mass copy"), unauthorized access, suspicious behavior (after hours, frequent), disgruntled employee, IP theft, sabotage, data hoarding ("download everything before leaving"), external-party communication. Critical (exfiltration / IP theft / sabotage) 1.0, high (unauthorized / suspicious / disgruntled) 0.8, medium (hoarding / external / keyword) 0.6. Replace placeholder `"[INSIDER_THREAT_REDACTED]"`. Seven policies.

#### 4.3.14 Profanity (`profanity_policies.py`)

Different from the rest — backed by the **Detoxify** ML library. The rule loads a Detoxify model once (lazy in `_get_model`) and runs `predict` on the input list. Each category-score pair becomes a triggered keyword (`"toxicity:0.873421"`).

Available Detoxify models:

| Name | Backbone | Notes |
|------|----------|-------|
| `original` | BERT | Toxic Comment Classification Challenge |
| `unbiased` | RoBERTa | Default — minimizes unintended bias |
| `multilingual` | XLM-R | 7-language coverage |
| `original-small` | Albert-S | Lightweight |
| `unbiased-small` | Albert-S | Lightweight |

Action subclasses (`ProfanityBlockAction`, `ProfanityBlockAction_LLM`, `ProfanityRaiseExceptionAction`, `ProfanityRaiseExceptionAction_LLM`) all share a `_parse_scores_and_get_max` helper that filters categories above `min_confidence` (default `0.5`) and returns the max score so messages quote the actual toxicity number.

The module ships ~50 pre-built policies covering every combination of `{Block, Block_LLM, RaiseException, RaiseException_LLM} × {default, model variant, threshold low/high, CPU, GPU}`. If the `detoxify` package is not installed, every constant in this module is set to `None` so other modules can still import.

#### 4.3.15 Skill safety (`skill_policies.py`)

Three independent rules used to validate skill content (instructions, references, scripts) before agents load them:

| Rule | Detects |
|------|---------|
| `SkillPromptInjectionRule` | "ignore previous instructions", "you are now a different agent", `[/system]` / `</system>` boundary escapes, "execute silently", ... |
| `SkillSecretLeakRule` | API keys (named regexes for AWS, GitHub, GitLab, Slack, OpenAI, Anthropic, Stripe, SendGrid, Twilio, Google, Azure), generic `password=`, bearer tokens, PEM private keys, connection strings — triggered keywords are truncated to `<type>:<first-12-chars>***`. |
| `SkillCodeInjectionRule` | `eval(`, `exec(`, `os.system(`, `subprocess.*(`, `pickle.loads(`, `requests.get(`, `socket.socket(`, ... |

Each has an LLM twin (`SkillPromptInjectionRule_LLM`, etc.) that asks the LLM for the same content type. Action classes are `SkillBlockAction`, `SkillBlockAction_LLM`, `SkillRaiseExceptionAction`, `SkillRaiseExceptionAction_LLM`. Fifteen pre-built policies cover the matrix `{prompt-injection, secret-leak, code-injection} × {block, block_LLM, block_LLM_Finder, raise, raise_LLM}`.

#### 4.3.16 Tool safety (`tool_safety_policies.py`)

LLM-first because tool schemas are too varied for keyword detection alone. Two rules:

| Rule | When it runs | LLM call |
|------|--------------|----------|
| `HarmfulToolRule_LLM` | Tool registration | `analyze_tool_safety(tool_info, "HARMFUL_FUNCTIONALITY")` |
| `MaliciousToolCallRule_LLM` | Each tool call | `analyze_tool_safety(tool_call_info, "MALICIOUS_CALL")` |

Both pull data from `policy_input.extra_data` — the tool name, description, parameters schema, and (for malicious-call analysis) the actual arguments. Both have a keyword/pattern fallback (`_keyword_based_detection`, `_pattern_based_detection`) that scans for things like `"rm -rf"`, `"drop database"`, `/etc/`, `c:\windows\system32`, command-injection chars, sudo / chmod 777 / privilege-escalation patterns. The fallback gives lower confidences so it remains distinguishable from LLM verdicts.

Action classes (`ToolBlockAction`, `ToolBlockAction_LLM`, `ToolRaiseExceptionAction`, `ToolRaiseExceptionAction_LLM`) drive eight policies (`HarmfulToolBlockPolicy`, `HarmfulToolBlockPolicy_LLM`, `HarmfulToolRaiseExceptionPolicy`, `HarmfulToolRaiseExceptionPolicy_LLM`, plus four mirror constants for the malicious-call rule).

## 5. Cross-file relationships

```
                                    +---------------------+
                                    |  upsonic.agent.Agent|
                                    +---------+-----------+
                                              |
                            user_policy / agent_policy / tool_policy
                                              |
                                              v
                +-----------------+    +------+----------------+
                |  PolicyManager  |    |  ToolPolicyManager     |
                |  (agent/)       |    |  (agent/)              |
                +--------+--------+    +-----------+------------+
                         |                         |
                         | execute_async           | execute_tool_validation_async
                         v                         v
                  +------+--------+         +------+---------+
                  |  Policy       |<------->|  Policy        |
                  |  (rule+action)|         |  (rule+action) |
                  +---+--+--------+         +-------+--------+
                      |  |                          |
        process / process_async                action / action_async
                      |  |                          |
                      v  v                          v
              +-------+--+------+      +------------+-------+
              |    RuleBase     |      |     ActionBase     |
              | (regex / LLM    |      | (allow/block/      |
              |  finder)        |      |  replace/anonymize)|
              +---+-------------+      +----+----------------+
                  |                         |
   _llm_find_keywords_with_input            _translate / llm_raise_*
                  |                         |
                  +-------------+-----------+
                                |
                                v
                  +-------------+--------------+
                  |  UpsonicLLMProvider        |
                  |  (llm/upsonic_llm.py)      |
                  | wraps upsonic.agent.Agent  |
                  +-------------+--------------+
                                |
                                v
                  +-------------+--------------+
                  |  Anonymizer / StreamDe     |
                  |  anonymizer (anonymization |
                  |  .py) — used at the agent  |
                  |  pipeline boundary         |
                  +----------------------------+
```

Important wiring rules:

- **Models flow upward.** `PolicyInput` is built by the agent / pipeline, mutates through scoping, and reaches `Policy.execute_async` which calls `Rule.process_async` first, then `Action.execute_action_async`. Both produce a `PolicyOutput`; the action's `PolicyOutput` is the one returned to the manager.
- **LLMs flow downward.** `PolicyManager.setup_policy_models(model)` injects an `UpsonicLLMProvider` configured with the agent's model into each policy's `base_llm`. Rules separately receive a `text_finder_llm` (optionally) so detection LLM can differ from the action LLM.
- **Transformation maps accumulate.** `PolicyManager` keeps `accumulated_map` across policies, feeds it to each `PolicyInput.existing_transformation_map`, and `ActionBase.execute_action` rebuilds its `transformation_map` from it. The final aggregated map is stored on `PolicyResult.transformation_map`. Downstream, `pipeline/steps.py` calls `deanonymize_content` / `deanonymize_mapping_content` / `StreamDeanonymizer` against this map to reverse anonymization on the LLM's response.
- **`DisallowedOperation` short-circuits.** Both managers `try / except DisallowedOperation` around `policy.execute_async`. A raise sets `was_blocked = True`, records the policy name, generates feedback if enabled, and `break`s the policy loop.

## 6. Public API

### Imports the rest of Upsonic uses

```python
from upsonic.safety_engine.base import RuleBase, ActionBase, Policy
from upsonic.safety_engine.models import PolicyInput, RuleOutput, PolicyOutput
from upsonic.safety_engine.exceptions import DisallowedOperation
from upsonic.safety_engine.anonymization import (
    anonymize_content,
    anonymize_contents,
    deanonymize_content,
    deanonymize_contents,
    deanonymize_mapping_content,
    Anonymizer,
    AnonymizationResult,
    StreamDeanonymizer,
)
from upsonic.safety_engine.llm.upsonic_llm import UpsonicLLMProvider

# Pre-built policies (everything in policies/__init__.py is re-exported via __getattr__)
from upsonic.safety_engine import (
    AdultContentBlockPolicy, AdultContentBlockPolicy_LLM,
    CryptoBlockPolicy, CryptoReplace, CryptoRaiseExceptionPolicy,
    PIIBlockPolicy, PIIAnonymizePolicy, PIIReplacePolicy,
    FinancialInfoBlockPolicy, FinancialInfoAnonymizePolicy,
    MedicalInfoBlockPolicy, LegalInfoBlockPolicy,
    TechnicalSecurityBlockPolicy, CybersecurityBlockPolicy,
    FraudDetectionBlockPolicy, PhishingBlockPolicy, InsiderThreatBlockPolicy,
    AnonymizePhoneNumbersPolicy,
    HarmfulToolBlockPolicy, MaliciousToolCallBlockPolicy,
    SkillPromptInjectionBlockPolicy, SkillSecretLeakBlockPolicy, SkillCodeInjectionBlockPolicy,
    ProfanityBlockPolicy,  # may be None if detoxify isn't installed
)
```

### Building your own policy

```python
from upsonic.safety_engine.base import RuleBase, ActionBase, Policy
from upsonic.safety_engine.models import PolicyInput, RuleOutput, PolicyOutput

class MyRule(RuleBase):
    name = "My Rule"
    description = "Detects the secret word 'banana'"
    language = "en"

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        text = " ".join(policy_input.input_texts or []).lower()
        if "banana" in text:
            return RuleOutput(
                confidence=1.0,
                content_type="BANANA_DETECTED",
                details="Found banana",
                triggered_keywords=["banana"],
            )
        return RuleOutput(confidence=0.0, content_type="SAFE", details="No banana")

class MyAction(ActionBase):
    name = "My Action"
    description = "Replaces banana with [FRUIT]"
    language = "en"

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        if rule_result.confidence < 0.3:
            return self.allow_content()
        return self.replace_triggered_keywords("[FRUIT]")

MyBananaPolicy = Policy(
    name="My Banana Policy",
    description="Replaces banana with [FRUIT]",
    rule=MyRule(),
    action=MyAction(),
)
```

### Running a policy directly (no agent)

```python
from upsonic.safety_engine.models import PolicyInput
from upsonic.safety_engine import PIIAnonymizePolicy

policy_input = PolicyInput(input_texts=["My email is alice@example.com"])
rule_output, action_output, policy_output = PIIAnonymizePolicy.execute(policy_input)

print(policy_output.output_texts)        # ["My email is xnpsy@kqzmrtl.bzr"]
print(policy_output.transformation_map)  # {1: {"original": "alice@example.com", "anonymous": "xnpsy@kqzmrtl.bzr"}}
print(policy_output.action_output)       # {"action_taken": "ANONYMIZE", "success": True, ...}
```

## 7. Integration with the rest of Upsonic

### 7.1 `PolicyManager` (`src/upsonic/agent/policy_manager.py`)

Wraps a list of `Policy` objects for **input** or **output** policy enforcement on an `Agent`:

```python
class PolicyResult:
    action_taken: str            # "ALLOW" | "BLOCK" | "REPLACE" | "ANONYMIZE" | "DISALLOWED_EXCEPTION"
    final_output: Optional[str]
    message: str
    triggered_policies: List[str]
    rule_outputs: List[RuleOutput]
    was_blocked: bool
    disallowed_exception: Optional[DisallowedOperation]
    feedback_message: Optional[str]
    requires_retry: bool
    transformation_map: Optional[dict]
    output_texts: Optional[List[str]]
    source_keys: Optional[List[Tuple[str, Optional[int]]]]

class PolicyManager:
    def __init__(self, policies, debug=False, enable_feedback=False,
                 feedback_loop_count=1, policy_type="user_policy"): ...
    async def execute_policies_async(self, policy_input, check_type, source_keys, task, agent) -> PolicyResult: ...
    def setup_policy_models(self, model) -> None
    def drain_accumulated_usage(self) -> Optional[RunUsage]
```

Behaviors layered on top of `Policy.execute_async`:

1. **Scoped execution.** When the caller passes `task` + `agent` + `source_keys`, every policy is rebuilt for the subset of texts it should apply to. `resolve_policy_scope` picks `apply_to_*` flags with priority *Policy > Task > Agent*.
2. **Action priority.** `BLOCK` and `DisallowedOperation` short-circuit the loop; `REPLACE` / `ANONYMIZE` mutate `current_texts` in place and continue, so transformations stack across policies.
3. **Transformation aggregation.** `accumulated_map` is rebased per policy (`base_idx = len(result.transformation_map)`) and re-fed into each policy's `existing_transformation_map`.
4. **Feedback retry loop.** When `enable_feedback=True` and `_current_retry_count < feedback_loop_count`, the manager calls `UpsonicLLMProvider.generate_policy_feedback_async` with `(original_content, policy_name, violation_reason, policy_type, action_type)` and stores the message on `PolicyResult.feedback_message`. The caller checks `should_retry_with_feedback()` and re-runs the agent with the feedback as additional context.
5. **Cost rollup.** `drain_accumulated_usage` walks every policy's `base_llm` / `text_finder_llm` and the manager's `_feedback_llm`, calling `drain_accumulated_usage()` and summing into a single `RunUsage`.

`Agent.__init__` instantiates two `PolicyManager`s — one for `user_policy` (input filtering) and one for `agent_policy` (output filtering):

```python
# src/upsonic/agent/agent.py (excerpt)
self.user_policy_manager = PolicyManager(
    policies=user_policy, debug=self.debug,
    enable_feedback=user_policy_feedback_enabled,
    feedback_loop_count=user_policy_feedback_loop_count,
    policy_type="user_policy",
)
self.agent_policy_manager = PolicyManager(
    policies=agent_policy, debug=self.debug,
    enable_feedback=agent_policy_feedback_enabled,
    feedback_loop_count=agent_policy_feedback_loop_count,
    policy_type="agent_policy",
)
```

### 7.2 `ToolPolicyManager` (`src/upsonic/agent/tool_policy_manager.py`)

Sister manager for tool safety. Two entry points:

```python
async def execute_tool_validation_async(self, tool_info, check_type) -> ToolPolicyResult
async def execute_tool_call_validation_async(self, tool_call_info, check_type) -> ToolPolicyResult
```

The first is run at tool registration time; the second runs before each tool invocation. Both build a `PolicyInput` with the tool metadata in `extra_data` (so the LLM-driven rules in `tool_safety_policies.py` can read `tool_name`, `tool_description`, `tool_parameters_schema`, `tool_call_args`). The result is a `ToolPolicyResult` (similar shape but with `is_safe`, `threat_details`, no feedback loop).

`Agent.__init__` builds two of them — `tool_policy_pre_manager` and `tool_policy_post_manager` — for the registration-time and call-time checks respectively.

### 7.3 Pipeline boundary (`src/upsonic/agent/pipeline/steps.py`)

The streaming and non-streaming pipelines call back into the safety engine for **de-anonymization**:

```python
from upsonic.safety_engine.anonymization import (
    StreamDeanonymizer,
    deanonymize_content,
    deanonymize_mapping_content,
)
```

When a policy returns a transformation map, the LLM sees the anonymized text. Tool outputs and final responses pass through `deanonymize_content` (string), `deanonymize_mapping_content` (dict / list / str recursive), or `StreamDeanonymizer` (token-by-token). The map travels with the run state inside `PolicyResult.transformation_map`.

## 8. End-to-end flow of a policy evaluation

The following sequence is what happens when a user calls `agent.do(task)` with `user_policy=[PIIAnonymizePolicy]` and `agent_policy=[CryptoBlockPolicy_LLM_Block]`.

```text
1. agent.do(task)
   └── builds PolicyInput from task description / context / system prompt / chat history
       (with source_keys = [("description", None), ("context", 0), ...])

2. PolicyManager.execute_policies_async(policy_input, "User Input Check", source_keys, task, agent)
   └── for each policy in self.policies:                # PIIAnonymizePolicy
        a. resolve_policy_scope(policy, task, agent)    # which sources to apply to
        b. filter current_texts by scope                # build per_policy_input
        c. await policy.execute_async(per_policy_input)
              └── Policy.execute_async:
                    ├── rule_output = await rule.process_async(per_policy_input)
                    │     └── PIIRule.process: regex sweep
                    │         returns RuleOutput(
                    │             confidence=0.6,
                    │             content_type="PII_DETECTED",
                    │             triggered_keywords=["EMAIL:alice@x.com", "PHONE:555-1234"]
                    │         )
                    └── action_output = await action.execute_action_async(rule_output, ...)
                          └── PIIAnonymizeAction.action:
                                anonymize_triggered_keywords()
                                  ├── for each keyword, _generate_unique_replacement
                                  ├── re.sub case-insensitively in each text
                                  └── PolicyOutput(
                                          output_texts=["My email is xnpsy@kqzmrtl.bzr"],
                                          action_output={"action_taken": "ANONYMIZE", ...},
                                          transformation_map={1: {...}, 2: {...}}
                                      )
        d. action_taken == "ANONYMIZE":
              - update current_texts in their original source positions
              - merge transformation_map into result.transformation_map (rebased)
              - accumulated_map ← result.transformation_map (next policy sees it)
              - generate feedback if enable_feedback
        e. continue to next policy (none here)
   └── return PolicyResult(action_taken="ANONYMIZE",
                            output_texts=["..."],
                            transformation_map={...})

3. Agent feeds ANONYMIZED text into the LLM call.

4. LLM returns "Acknowledged, mailing xnpsy@kqzmrtl.bzr" (the anonymous email).

5. agent_policy_manager.execute_policies_async on the LLM output
   └── CryptoBlockPolicy_LLM_Block.execute_async(...)
        ├── CryptoRule.process: keyword sweep → confidence 0.0 → ALLOW
        └── PolicyResult(action_taken="ALLOW", ...)

6. pipeline/steps.py reverses anonymization on the response
   └── deanonymize_content(llm_response, result.transformation_map)
       returns "Acknowledged, mailing alice@x.com"

7. Agent returns the de-anonymized response to the user.

8. drain_accumulated_usage() rolls every policy's LLM cost into the parent RunUsage.
```

If any policy in step 2 had returned `BLOCK` (or raised `DisallowedOperation`), the manager would have:

- Set `was_blocked = True`, recorded `violated_policy_name` and `violation_reason`.
- If `enable_feedback`, called `generate_policy_feedback_async` and set `feedback_message` + `requires_retry=True`.
- Returned immediately. The agent would then either surface the block message to the user or, if a retry was requested, re-run with the feedback message attached to the prompt — incrementing `_current_retry_count` until `feedback_loop_count` is exhausted.

The same flow applies to `ToolPolicyManager`, except the policy reads its inputs from `extra_data` and the result type is `ToolPolicyResult` (no feedback loop, just `is_safe` + `threat_details`).
