"""
Fraud Detection Policies
Handles detection and protection against fraud, scams, and financial crimes.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class FraudDetectionRule(RuleBase):
    """Rule to detect fraud, scams, and financial crimes"""
    
    name = "Fraud Detection Rule"
    description = "Detects fraud, scams, financial crimes, and deceptive practices"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Common fraud and scam keywords
        self.fraud_keywords = [
            "fraud", "scam", "con", "swindle", "cheat", "deceive", "defraud",
            "embezzlement", "money laundering", "identity theft", "credit card fraud",
            "wire fraud", "bank fraud", "insurance fraud", "tax fraud", "securities fraud",
            "ponzi scheme", "pyramid scheme", "advance fee fraud", "419 scam",
            "romance scam", "investment fraud", "charity fraud", "lottery scam",
            "inheritance scam", "tech support scam", "phishing", "vishing", "smishing"
        ]
        
        # Financial scam patterns
        self.financial_scam_patterns = [
            r'\b(?:guaranteed|risk-free|no risk|guaranteed returns|high returns)\s+(?:investment|profit|income|earnings)\b',
            r'\b(?:quick|fast|easy|instant)\s+(?:money|cash|profit|wealth|rich)\b',
            r'\b(?:limited time|act now|urgent|expires soon|don\'t miss out)\s+(?:offer|deal|opportunity)\b',
            r'\b(?:secret|exclusive|insider|confidential)\s+(?:investment|trading|strategy|tip)\b',
            r'\b(?:work from home|make money online|earn from home)\s+(?:guaranteed|easy|quick)\b',
            r'\b(?:free money|free cash|free gift|no cost|no fee)\s+(?:just pay|only pay|small fee)\b',
            r'\b(?:congratulations|you won|you\'re selected|you\'ve been chosen)\s+(?:prize|lottery|sweepstakes)\b',
            r'\b(?:send money|wire transfer|western union|moneygram|bitcoin)\s+(?:urgent|immediate|now)\b',
        ]
        
        # Identity theft patterns
        self.identity_theft_patterns = [
            r'\b(?:verify|confirm|update|validate)\s+(?:your|account|personal|banking|credit card)\s+(?:information|details|data)\b',
            r'\b(?:suspicious activity|unauthorized access|security breach|account compromised)\s+(?:detected|found|reported)\b',
            r'\b(?:click here|click link|verify now|update now)\s+(?:to|in order to|so you can)\s+(?:verify|secure|protect)\b',
            r'\b(?:social security|ssn|credit card|bank account|routing number)\s+(?:verification|confirmation|update)\b',
            r'\b(?:tax refund|irs|internal revenue service)\s+(?:verification|confirmation|update|claim)\b',
        ]
        
        # Romance and dating scam patterns
        self.romance_scam_patterns = [
            r'\b(?:military|soldier|deployed|overseas)\s+(?:needs|requesting|asking for)\s+(?:money|financial help|emergency funds)\b',
            r'\b(?:stranded|stuck|trapped)\s+(?:in|at)\s+(?:foreign country|airport|hospital)\s+(?:needs|requesting)\s+(?:money|help)\b',
            r'\b(?:medical emergency|hospital bill|surgery|treatment)\s+(?:urgent|immediate|needs money)\b',
            r'\b(?:love|relationship|marriage)\s+(?:but first|however|unfortunately)\s+(?:need|require|must have)\s+(?:money|financial help)\b',
            r'\b(?:gift cards|itunes|google play|amazon)\s+(?:for|to help with|to pay for)\s+(?:emergency|urgent|immediate)\b',
        ]
        
        # Investment and crypto scam patterns
        self.investment_scam_patterns = [
            r'\b(?:cryptocurrency|bitcoin|ethereum|trading)\s+(?:bot|robot|algorithm|software)\s+(?:guaranteed|automatic|passive)\s+(?:profit|income|returns)\b',
            r'\b(?:forex|fx|binary options|cfd)\s+(?:trading|investment)\s+(?:guaranteed|risk-free|high returns)\b',
            r'\b(?:pump and dump|penny stock|hot tip|insider trading)\s+(?:opportunity|chance|deal)\b',
            r'\b(?:initial coin offering|ico|token sale)\s+(?:exclusive|limited|pre-sale)\s+(?:opportunity|chance)\b',
            r'\b(?:multi-level marketing|mlm|network marketing)\s+(?:opportunity|business|income)\s+(?:guaranteed|easy|quick)\b',
        ]
        
        # Tech support and software scam patterns
        self.tech_support_scam_patterns = [
            r'\b(?:microsoft|apple|google|amazon)\s+(?:tech support|customer service|security team)\s+(?:calling|contacting|reaching out)\b',
            r'\b(?:computer|device|account)\s+(?:infected|compromised|hacked|virus detected)\s+(?:immediate action|urgent|call now)\b',
            r'\b(?:remote access|teamviewer|anydesk|logmein)\s+(?:to fix|to repair|to clean)\s+(?:your|the)\s+(?:computer|device)\b',
            r'\b(?:subscription|renewal|payment)\s+(?:failed|expired|due)\s+(?:click here|call now|update now)\b',
            r'\b(?:software|antivirus|security)\s+(?:license|subscription)\s+(?:expired|expiring|renewal required)\b',
        ]
        
        # Charity and donation scam patterns
        self.charity_scam_patterns = [
            r'\b(?:natural disaster|hurricane|earthquake|flood|fire)\s+(?:victims|survivors|relief)\s+(?:urgent|immediate|help needed)\b',
            r'\b(?:sick|ill|hospitalized|medical treatment)\s+(?:child|family member|friend)\s+(?:needs|requires|urgent help)\b',
            r'\b(?:charity|non-profit|foundation)\s+(?:urgent|immediate|emergency)\s+(?:donation|fundraising|help)\b',
            r'\b(?:go fund me|gofundme|crowdfunding)\s+(?:urgent|emergency|help needed|medical bills)\b',
            r'\b(?:veteran|military|police|firefighter)\s+(?:family|fund|charity)\s+(?:urgent|emergency|help needed)\b',
        ]
        
        # Urgency and pressure tactics
        self.urgency_patterns = [
            r'\b(?:act now|don\'t wait|limited time|expires soon|last chance)\b',
            r'\b(?:urgent|immediate|asap|right now|today only)\b',
            r'\b(?:exclusive|secret|confidential|insider|private)\s+(?:information|deal|opportunity)\b',
            r'\b(?:once in a lifetime|never again|unique|special|rare)\s+(?:opportunity|chance|deal)\b',
            r'\b(?:everyone is doing it|don\'t miss out|join thousands|be part of)\b',
        ]
        
        # Payment and money transfer patterns
        self.payment_patterns = [
            r'\b(?:wire transfer|western union|moneygram|bitcoin|cryptocurrency)\s+(?:only|required|necessary)\b',
            r'\b(?:gift cards|itunes|google play|amazon|steam)\s+(?:payment|payment method|required)\b',
            r'\b(?:prepaid|reloadable|debit)\s+(?:card|visa|mastercard)\s+(?:required|needed|only)\b',
            r'\b(?:cash|money order|certified check)\s+(?:only|required|no other payment)\b',
            r'\b(?:send money|transfer funds|make payment)\s+(?:immediately|urgently|right now)\b',
        ]
        
        # Allow custom patterns from options
        if options:
            if "custom_fraud_keywords" in options:
                self.fraud_keywords.extend(options["custom_fraud_keywords"])
            if "custom_scam_patterns" in options:
                self.financial_scam_patterns.extend(options["custom_scam_patterns"])
            if "custom_urgency_patterns" in options:
                self.urgency_patterns.extend(options["custom_urgency_patterns"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for fraud and scam detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or [])
        
        # Find matching patterns
        triggered_items = []
        
        # Check fraud keywords
        combined_text_lower = combined_text.lower()
        for keyword in self.fraud_keywords:
            if keyword.lower() in combined_text_lower:
                triggered_items.append(f"FRAUD_KEYWORD:{keyword}")
        
        # Check financial scam patterns
        for pattern in self.financial_scam_patterns:
            scam_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"FINANCIAL_SCAM:{scam}" for scam in scam_matches])
        
        # Check identity theft patterns
        for pattern in self.identity_theft_patterns:
            identity_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"IDENTITY_THEFT:{identity}" for identity in identity_matches])
        
        # Check romance scam patterns
        for pattern in self.romance_scam_patterns:
            romance_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"ROMANCE_SCAM:{romance}" for romance in romance_matches])
        
        # Check investment scam patterns
        for pattern in self.investment_scam_patterns:
            investment_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"INVESTMENT_SCAM:{investment}" for investment in investment_matches])
        
        # Check tech support scam patterns
        for pattern in self.tech_support_scam_patterns:
            tech_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"TECH_SUPPORT_SCAM:{tech}" for tech in tech_matches])
        
        # Check charity scam patterns
        for pattern in self.charity_scam_patterns:
            charity_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CHARITY_SCAM:{charity}" for charity in charity_matches])
        
        # Check urgency patterns
        for pattern in self.urgency_patterns:
            urgency_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"URGENCY_TACTIC:{urgency}" for urgency in urgency_matches])
        
        # Check payment patterns
        for pattern in self.payment_patterns:
            payment_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SUSPICIOUS_PAYMENT:{payment}" for payment in payment_matches])
        
        # Calculate confidence based on number and type of matches
        if not triggered_items:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_FRAUD_DETECTED",
                details="No fraud or scam indicators detected"
            )
        
        # Weight different types of fraud indicators
        critical_count = len([item for item in triggered_items if any(x in item for x in ["FRAUD_KEYWORD:", "IDENTITY_THEFT:", "INVESTMENT_SCAM:"])])
        high_risk_count = len([item for item in triggered_items if any(x in item for x in ["FINANCIAL_SCAM:", "ROMANCE_SCAM:", "TECH_SUPPORT_SCAM:"])])
        medium_risk_count = len([item for item in triggered_items if any(x in item for x in ["CHARITY_SCAM:", "URGENCY_TACTIC:", "SUSPICIOUS_PAYMENT:"])])
        
        # Calculate confidence with weighted scoring
        confidence = min(1.0, (critical_count * 1.0 + high_risk_count * 0.8 + medium_risk_count * 0.6))
        
        return RuleOutput(
            confidence=confidence,
            content_type="FRAUD_DETECTED",
            details=f"Detected {len(triggered_items)} fraud indicators: {critical_count} critical, {high_risk_count} high-risk, {medium_risk_count} medium-risk",
            triggered_keywords=triggered_items
        )


class FraudDetectionRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect fraud and scams with better context understanding"""
    
    name = "Fraud Detection LLM Rule"
    description = "Uses LLM to detect fraud and scams with context awareness and better accuracy"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for fraud and scam detection"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            fallback_rule = FraudDetectionRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find fraud and scam indicators
            triggered_keywords = self._llm_find_keywords_with_input("FRAUD_SCAM", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_FRAUD_DETECTED",
                    details="No fraud or scam indicators detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="FRAUD_DETECTED",
                details=f"LLM detected {len(triggered_keywords)} fraud and scam indicators",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to pattern-based detection on error
            fallback_rule = FraudDetectionRule()
            return fallback_rule.process(policy_input)


class FraudDetectionBlockAction(ActionBase):
    """Action to block content containing fraud and scam indicators"""
    
    name = "Fraud Detection Block Action"
    description = "Blocks content containing fraud, scams, and deceptive practices"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for fraud and scam content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        block_message = (
            "This content has been blocked as it contains indicators of fraud, scams, "
            "or deceptive practices that could pose financial risks. Please be cautious "
            "of any requests for money, personal information, or urgent actions."
        )
        
        return self.raise_block_error(block_message)


class FraudDetectionBlockAction_LLM(ActionBase):
    """LLM-powered action to block fraud and scams with contextual messaging"""
    
    name = "Fraud Detection Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for fraud and scam content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for fraud and scam content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains fraud or scam indicators: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class FraudDetectionAnonymizeAction(ActionBase):
    """Action to anonymize fraud and scam content"""
    
    name = "Fraud Detection Anonymize Action"
    description = "Anonymizes fraud and scam indicators while preserving content structure"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute anonymization action for fraud and scam content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.anonymize_triggered_keywords()


class FraudDetectionReplaceAction(ActionBase):
    """Action to replace fraud and scam indicators with safe placeholders"""
    
    name = "Fraud Detection Replace Action"
    description = "Replaces fraud and scam indicators with safe placeholders"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute replacement action for fraud and scam content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.replace_triggered_keywords("[FRAUD_REDACTED]")


class FraudDetectionRaiseExceptionAction(ActionBase):
    """Action to raise exception for fraud and scam content"""
    
    name = "Fraud Detection Raise Exception Action"
    description = "Raises DisallowedOperation exception for fraud and scam content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for fraud and scam content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains fraud, scam, or deceptive practices "
            "that violate platform policies and could pose financial risks to users."
        )
        return self.raise_exception(exception_message)


class FraudDetectionRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for fraud and scam content"""
    
    name = "Fraud Detection Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for fraud and scam content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for fraud and scam content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains fraud or scam indicators: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Fraud Detection Block Policy
# Standard policy that blocks content containing fraud and scam indicators using pattern detection
FraudDetectionBlockPolicy = Policy(
    name="Fraud Detection Block Policy",
    description="Blocks content containing fraud, scams, and deceptive practices",
    rule=FraudDetectionRule(),
    action=FraudDetectionBlockAction()
)

## Fraud Detection Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
FraudDetectionBlockPolicy_LLM = Policy(
    name="Fraud Detection Block Policy LLM",
    description="Uses LLM to detect and block fraud and scam content with contextual understanding",
    rule=FraudDetectionRule(),
    action=FraudDetectionBlockAction_LLM()
)

## Fraud Detection Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
FraudDetectionBlockPolicy_LLM_Finder = Policy(
    name="Fraud Detection Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of fraud and scam content",
    rule=FraudDetectionRule_LLM_Finder(),
    action=FraudDetectionBlockAction()
)

## Fraud Detection Anonymize Policy
# Policy that anonymizes fraud and scam indicators while preserving content structure
FraudDetectionAnonymizePolicy = Policy(
    name="Fraud Detection Anonymize Policy",
    description="Anonymizes fraud and scam indicators while preserving content",
    rule=FraudDetectionRule(),
    action=FraudDetectionAnonymizeAction()
)

## Fraud Detection Replace Policy
# Policy that replaces fraud and scam indicators with safe placeholders
FraudDetectionReplacePolicy = Policy(
    name="Fraud Detection Replace Policy",
    description="Replaces fraud and scam indicators with safe placeholders",
    rule=FraudDetectionRule(),
    action=FraudDetectionReplaceAction()
)

## Fraud Detection Raise Exception Policy
# Policy that raises exceptions for fraud and scam content
FraudDetectionRaiseExceptionPolicy = Policy(
    name="Fraud Detection Raise Exception Policy",
    description="Raises DisallowedOperation exception when fraud or scam indicators are detected",
    rule=FraudDetectionRule(),
    action=FraudDetectionRaiseExceptionAction()
)

## Fraud Detection Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
FraudDetectionRaiseExceptionPolicy_LLM = Policy(
    name="Fraud Detection Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for fraud and scam content",
    rule=FraudDetectionRule(),
    action=FraudDetectionRaiseExceptionAction_LLM()
)
