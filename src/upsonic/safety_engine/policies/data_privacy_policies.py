"""
Data Privacy Policies
Handles detection and protection of data privacy violations, GDPR compliance, and data protection.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class DataPrivacyRule(RuleBase):
    """Rule to detect data privacy violations and GDPR compliance issues"""
    
    name = "Data Privacy Rule"
    description = "Detects data privacy violations, GDPR compliance issues, and data protection concerns"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # GDPR and data protection keywords
        self.gdpr_keywords = [
            "gdpr", "general data protection regulation", "data protection act",
            "personal data", "sensitive personal data", "special category data",
            "data subject", "data controller", "data processor", "data protection officer",
            "consent", "lawful basis", "legitimate interest", "vital interest",
            "data minimization", "purpose limitation", "storage limitation",
            "accuracy", "integrity", "confidentiality", "accountability",
            "privacy by design", "privacy by default", "data protection impact assessment",
            "breach notification", "right to be forgotten", "right to erasure",
            "right to rectification", "right to portability", "right to access",
            "right to object", "automated decision making", "profiling"
        ]
        
        # Data collection and processing patterns
        self.data_collection_patterns = [
            r'\b(?:collect|gather|harvest|extract|obtain)\s+(?:personal|sensitive|private)\s+(?:data|information|details)\b',
            r'\b(?:track|monitor|surveil|observe)\s+(?:user|customer|visitor|individual)\s+(?:behavior|activity|movement)\b',
            r'\b(?:profile|analyze|process)\s+(?:personal|individual|user)\s+(?:data|information|behavior)\b',
            r'\b(?:store|retain|keep|hold)\s+(?:personal|sensitive|private)\s+(?:data|information|records)\b',
            r'\b(?:share|transfer|disclose|distribute)\s+(?:personal|sensitive|private)\s+(?:data|information)\b',
            r'\b(?:sell|trade|monetize)\s+(?:personal|user|customer)\s+(?:data|information|profiles)\b',
        ]
        
        # Consent and authorization patterns
        self.consent_patterns = [
            r'\b(?:without|without proper|without explicit|without informed)\s+(?:consent|permission|authorization)\b',
            r'\b(?:opt-out|opt out|unsubscribe)\s+(?:difficult|complicated|hidden|buried)\b',
            r'\b(?:pre-ticked|pre-selected|default)\s+(?:consent|agreement|permission)\b',
            r'\b(?:forced|mandatory|required)\s+(?:consent|agreement|permission)\s+(?:for|to)\b',
            r'\b(?:bundle|combined|tied)\s+(?:consent|agreement|permission)\b',
            r'\b(?:withdraw|revoke|cancel)\s+(?:consent|permission)\s+(?:difficult|impossible|not allowed)\b',
        ]
        
        # Data breach and security patterns
        self.data_breach_patterns = [
            r'\b(?:data breach|security breach|privacy breach|information leak)\b',
            r'\b(?:unauthorized access|illegal access|unauthorized disclosure)\b',
            r'\b(?:data loss|information loss|data theft|information theft)\b',
            r'\b(?:hack|hacked|compromised|infiltrated)\s+(?:database|system|server|network)\b',
            r'\b(?:exposed|leaked|disclosed)\s+(?:personal|sensitive|private)\s+(?:data|information)\b',
            r'\b(?:notify|notification)\s+(?:authorities|dpa|ico|supervisory authority)\s+(?:within|after)\s+(?:72 hours|24 hours)\b',
        ]
        
        # Cross-border data transfer patterns
        self.data_transfer_patterns = [
            r'\b(?:transfer|transmit|send)\s+(?:personal|sensitive|private)\s+(?:data|information)\s+(?:to|across|outside)\b',
            r'\b(?:third country|non-adequate country|unsafe country)\s+(?:data|information)\s+(?:transfer|transmission)\b',
            r'\b(?:adequacy decision|standard contractual clauses|binding corporate rules)\b',
            r'\b(?:data localization|data residency|keep data local|data sovereignty)\b',
            r'\b(?:cloud storage|cloud service|saas|paas|iaas)\s+(?:outside|in)\s+(?:eu|europe|adequate country)\b',
        ]
        
        # Automated decision making patterns
        self.automated_decision_patterns = [
            r'\b(?:automated decision|algorithmic decision|ai decision|machine learning decision)\b',
            r'\b(?:profiling|behavioral profiling|predictive analytics|scoring)\b',
            r'\b(?:algorithm|ai|artificial intelligence|machine learning)\s+(?:bias|discrimination|unfair)\b',
            r'\b(?:human intervention|human review|human oversight)\s+(?:not available|not provided|denied)\b',
            r'\b(?:explainable ai|algorithmic transparency|decision explanation)\s+(?:not provided|unavailable)\b',
        ]
        
        # Data subject rights patterns
        self.data_rights_patterns = [
            r'\b(?:right to access|subject access request|data portability)\s+(?:denied|refused|ignored)\b',
            r'\b(?:right to erasure|right to be forgotten)\s+(?:denied|refused|ignored)\b',
            r'\b(?:right to rectification|data correction)\s+(?:denied|refused|ignored)\b',
            r'\b(?:right to object|opt-out)\s+(?:denied|refused|ignored)\b',
            r'\b(?:data subject|individual)\s+(?:rights|requests)\s+(?:violated|ignored|not respected)\b',
        ]
        
        # Privacy policy and transparency patterns
        self.transparency_patterns = [
            r'\b(?:privacy policy|data policy|terms of service)\s+(?:unclear|confusing|misleading)\b',
            r'\b(?:data processing|data collection)\s+(?:not disclosed|hidden|secret)\b',
            r'\b(?:purpose|legal basis)\s+(?:not specified|unclear|misleading)\b',
            r'\b(?:retention period|data retention)\s+(?:not specified|unclear|excessive)\b',
            r'\b(?:contact information|dpo contact)\s+(?:not provided|unavailable|hidden)\b',
        ]
        
        # Children's data protection patterns
        self.children_data_patterns = [
            r'\b(?:children|minors|under 16|under 18)\s+(?:personal|sensitive)\s+(?:data|information)\b',
            r'\b(?:parental consent|guardian consent)\s+(?:not obtained|not verified|fake)\b',
            r'\b(?:age verification|age check)\s+(?:not performed|bypassed|fake)\b',
            r'\b(?:child|children)\s+(?:profiling|targeted advertising|behavioral tracking)\b',
        ]
        
        # Allow custom patterns from options
        if options:
            if "custom_gdpr_keywords" in options:
                self.gdpr_keywords.extend(options["custom_gdpr_keywords"])
            if "custom_consent_patterns" in options:
                self.consent_patterns.extend(options["custom_consent_patterns"])
            if "custom_breach_patterns" in options:
                self.data_breach_patterns.extend(options["custom_breach_patterns"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for data privacy violation detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or [])
        
        # Find matching patterns
        triggered_items = []
        
        # Check GDPR keywords
        combined_text_lower = combined_text.lower()
        for keyword in self.gdpr_keywords:
            if keyword.lower() in combined_text_lower:
                triggered_items.append(f"GDPR_KEYWORD:{keyword}")
        
        # Check data collection patterns
        for pattern in self.data_collection_patterns:
            collection_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"DATA_COLLECTION:{collection}" for collection in collection_matches])
        
        # Check consent patterns
        for pattern in self.consent_patterns:
            consent_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CONSENT_VIOLATION:{consent}" for consent in consent_matches])
        
        # Check data breach patterns
        for pattern in self.data_breach_patterns:
            breach_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"DATA_BREACH:{breach}" for breach in breach_matches])
        
        # Check data transfer patterns
        for pattern in self.data_transfer_patterns:
            transfer_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"DATA_TRANSFER:{transfer}" for transfer in transfer_matches])
        
        # Check automated decision patterns
        for pattern in self.automated_decision_patterns:
            decision_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"AUTOMATED_DECISION:{decision}" for decision in decision_matches])
        
        # Check data rights patterns
        for pattern in self.data_rights_patterns:
            rights_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"DATA_RIGHTS:{rights}" for rights in rights_matches])
        
        # Check transparency patterns
        for pattern in self.transparency_patterns:
            transparency_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"TRANSPARENCY:{transparency}" for transparency in transparency_matches])
        
        # Check children's data patterns
        for pattern in self.children_data_patterns:
            children_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CHILDREN_DATA:{children}" for children in children_matches])
        
        # Calculate confidence based on number and type of matches
        if not triggered_items:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_DATA_PRIVACY_VIOLATION",
                details="No data privacy violations detected"
            )
        
        # Weight different types of violations
        critical_count = len([item for item in triggered_items if any(x in item for x in ["DATA_BREACH:", "CONSENT_VIOLATION:", "CHILDREN_DATA:"])])
        high_risk_count = len([item for item in triggered_items if any(x in item for x in ["DATA_COLLECTION:", "DATA_TRANSFER:", "AUTOMATED_DECISION:"])])
        medium_risk_count = len([item for item in triggered_items if any(x in item for x in ["DATA_RIGHTS:", "TRANSPARENCY:", "GDPR_KEYWORD:"])])
        
        # Calculate confidence with weighted scoring
        confidence = min(1.0, (critical_count * 1.0 + high_risk_count * 0.8 + medium_risk_count * 0.6))
        
        return RuleOutput(
            confidence=confidence,
            content_type="DATA_PRIVACY_VIOLATION_DETECTED",
            details=f"Detected {len(triggered_items)} data privacy violations: {critical_count} critical, {high_risk_count} high-risk, {medium_risk_count} medium-risk",
            triggered_keywords=triggered_items
        )


class DataPrivacyRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect data privacy violations with better context understanding"""
    
    name = "Data Privacy LLM Detection Rule"
    description = "Uses LLM to detect data privacy violations with context awareness and better accuracy"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for data privacy violation detection"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            fallback_rule = DataPrivacyRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find data privacy violations
            triggered_keywords = self._llm_find_keywords_with_input("DATA_PRIVACY_VIOLATION", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_DATA_PRIVACY_VIOLATION",
                    details="No data privacy violations detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="DATA_PRIVACY_VIOLATION_DETECTED",
                details=f"LLM detected {len(triggered_keywords)} data privacy violations",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to pattern-based detection on error
            fallback_rule = DataPrivacyRule()
            return fallback_rule.process(policy_input)


class DataPrivacyBlockAction(ActionBase):
    """Action to block content containing data privacy violations"""
    
    name = "Data Privacy Block Action"
    description = "Blocks content containing data privacy violations and GDPR compliance issues"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for data privacy violation content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        block_message = (
            "This content has been blocked as it contains data privacy violations or "
            "GDPR compliance issues that could pose legal and regulatory risks. "
            "Please ensure your content complies with data protection regulations."
        )
        
        return self.raise_block_error(block_message)


class DataPrivacyBlockAction_LLM(ActionBase):
    """LLM-powered action to block data privacy violations with contextual messaging"""
    
    name = "Data Privacy Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for data privacy violations"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for data privacy violation content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains data privacy violations: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class DataPrivacyAnonymizeAction(ActionBase):
    """Action to anonymize data privacy violation content"""
    
    name = "Data Privacy Anonymize Action"
    description = "Anonymizes data privacy violations while preserving content structure"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute anonymization action for data privacy violation content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.anonymize_triggered_keywords()


class DataPrivacyReplaceAction(ActionBase):
    """Action to replace data privacy violations with safe placeholders"""
    
    name = "Data Privacy Replace Action"
    description = "Replaces data privacy violations with safe placeholders"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute replacement action for data privacy violation content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.replace_triggered_keywords("[DATA_PRIVACY_REDACTED]")


class DataPrivacyRaiseExceptionAction(ActionBase):
    """Action to raise exception for data privacy violation content"""
    
    name = "Data Privacy Raise Exception Action"
    description = "Raises DisallowedOperation exception for data privacy violation content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for data privacy violation content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains data privacy violations or "
            "GDPR compliance issues that violate data protection policies."
        )
        return self.raise_exception(exception_message)


class DataPrivacyRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for data privacy violation content"""
    
    name = "Data Privacy Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for data privacy violations"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for data privacy violation content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains data privacy violations: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Data Privacy Block Policy
# Standard policy that blocks content containing data privacy violations using pattern detection
DataPrivacyBlockPolicy = Policy(
    name="Data Privacy Block Policy",
    description="Blocks content containing data privacy violations and GDPR compliance issues",
    rule=DataPrivacyRule(),
    action=DataPrivacyBlockAction()
)

## Data Privacy Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
DataPrivacyBlockPolicy_LLM = Policy(
    name="Data Privacy Block Policy LLM",
    description="Uses LLM to detect and block data privacy violations with contextual understanding",
    rule=DataPrivacyRule(),
    action=DataPrivacyBlockAction_LLM()
)

## Data Privacy Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
DataPrivacyBlockPolicy_LLM_Finder = Policy(
    name="Data Privacy Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of data privacy violations",
    rule=DataPrivacyRule_LLM_Finder(),
    action=DataPrivacyBlockAction()
)

## Data Privacy Anonymize Policy
# Policy that anonymizes data privacy violations while preserving content structure
DataPrivacyAnonymizePolicy = Policy(
    name="Data Privacy Anonymize Policy",
    description="Anonymizes data privacy violations while preserving content",
    rule=DataPrivacyRule(),
    action=DataPrivacyAnonymizeAction()
)

## Data Privacy Replace Policy
# Policy that replaces data privacy violations with safe placeholders
DataPrivacyReplacePolicy = Policy(
    name="Data Privacy Replace Policy",
    description="Replaces data privacy violations with safe placeholders",
    rule=DataPrivacyRule(),
    action=DataPrivacyReplaceAction()
)

## Data Privacy Raise Exception Policy
# Policy that raises exceptions for data privacy violation content
DataPrivacyRaiseExceptionPolicy = Policy(
    name="Data Privacy Raise Exception Policy",
    description="Raises DisallowedOperation exception when data privacy violations are detected",
    rule=DataPrivacyRule(),
    action=DataPrivacyRaiseExceptionAction()
)

## Data Privacy Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
DataPrivacyRaiseExceptionPolicy_LLM = Policy(
    name="Data Privacy Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for data privacy violations",
    rule=DataPrivacyRule(),
    action=DataPrivacyRaiseExceptionAction_LLM()
)
