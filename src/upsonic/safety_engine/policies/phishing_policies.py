"""
Phishing and Social Engineering Policies
Handles detection and protection against phishing attacks, social engineering, and deceptive communications.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class PhishingRule(RuleBase):
    """Rule to detect phishing attacks and social engineering attempts"""
    
    name = "Phishing Detection Rule"
    description = "Detects phishing attacks, social engineering, and deceptive communications"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Phishing and social engineering keywords
        self.phishing_keywords = [
            "phishing", "spear phishing", "whaling", "vishing", "smishing", "pharming",
            "social engineering", "pretexting", "baiting", "quid pro quo", "tailgating",
            "impersonation", "spoofing", "credential harvesting", "account takeover",
            "business email compromise", "ceo fraud", "vendor fraud", "invoice fraud"
        ]
        
        # Urgent action patterns
        self.urgent_action_patterns = [
            r'\b(?:urgent|immediate|asap|emergency|critical)\s+(?:action|response|verification|update)\s+(?:required|needed|necessary)\b',
            r'\b(?:act now|respond immediately|verify now|update now|click here now)\b',
            r'\b(?:your account|your profile|your information)\s+(?:will be|is about to be|has been)\s+(?:suspended|locked|deleted|closed)\b',
            r'\b(?:within|in)\s+(?:24 hours|48 hours|72 hours|1 hour|30 minutes)\s+(?:or|otherwise|or else)\b',
            r'\b(?:last chance|final notice|final warning|expires soon|time sensitive)\b',
        ]
        
        # Suspicious link and domain patterns
        self.suspicious_link_patterns = [
            r'\b(?:click here|click link|click button|follow link|visit link)\s+(?:to|in order to|so you can)\b',
            r'\b(?:http|https|www)\s*[^\s]*(?:bit\.ly|tinyurl|short\.link|goo\.gl|t\.co)\b',
            r'\b(?:suspicious|fake|phishing|malicious)\s+(?:website|link|url|domain)\b',
            r'\b(?:domain|website|site)\s+(?:looks|appears|seems)\s+(?:suspicious|fake|phishing)\b',
            r'\b(?:typo|misspelled|similar)\s+(?:domain|website|url|link)\b',
        ]
        
        # Credential and information harvesting patterns
        self.credential_harvesting_patterns = [
            r'\b(?:enter|provide|submit|confirm|verify)\s+(?:your|account|personal|banking|credit card)\s+(?:password|pin|ssn|social security)\b',
            r'\b(?:username|email|password|login|credentials)\s+(?:verification|confirmation|update|reset)\s+(?:required|needed)\b',
            r'\b(?:two-factor|2fa|multi-factor|mfa)\s+(?:authentication|verification|code)\s+(?:required|needed|expired)\b',
            r'\b(?:security question|backup code|recovery code)\s+(?:verification|confirmation|update)\s+(?:required|needed)\b',
            r'\b(?:bank account|routing number|credit card|cvv|cvc)\s+(?:verification|confirmation|update)\s+(?:required|needed)\b',
        ]
        
        # Impersonation patterns
        self.impersonation_patterns = [
            r'\b(?:microsoft|apple|google|amazon|paypal|ebay|netflix|spotify)\s+(?:support|customer service|security team|billing)\b',
            r'\b(?:irs|internal revenue service|tax|government|federal)\s+(?:verification|confirmation|update|notice)\b',
            r'\b(?:bank|credit union|financial institution)\s+(?:security|fraud|verification|update)\s+(?:department|team|center)\b',
            r'\b(?:ceo|manager|director|supervisor|boss)\s+(?:urgent|immediate|confidential|private)\s+(?:request|message|email)\b',
            r'\b(?:law enforcement|police|fbi|cia|secret service)\s+(?:investigation|verification|confirmation)\b',
        ]
        
        # Prize and reward patterns
        self.prize_reward_patterns = [
            r'\b(?:congratulations|you won|you\'re selected|you\'ve been chosen)\s+(?:prize|lottery|sweepstakes|contest|award)\b',
            r'\b(?:claim|collect|receive)\s+(?:your|the)\s+(?:prize|reward|gift|money|cash)\s+(?:now|immediately|urgently)\b',
            r'\b(?:free|no cost|no charge)\s+(?:gift|prize|reward|money|cash|iphone|ipad|laptop)\b',
            r'\b(?:click here|visit|go to)\s+(?:to claim|to collect|to receive)\s+(?:your|the)\s+(?:prize|reward|gift)\b',
            r'\b(?:limited time|act fast|don\'t miss out)\s+(?:offer|deal|opportunity|chance)\b',
        ]
        
        # Account security patterns
        self.account_security_patterns = [
            r'\b(?:suspicious activity|unauthorized access|security breach|account compromised)\s+(?:detected|found|reported|alert)\b',
            r'\b(?:login|sign-in|access)\s+(?:attempt|activity)\s+(?:from|at)\s+(?:unusual|suspicious|unknown)\s+(?:location|device|ip)\b',
            r'\b(?:password|account|profile)\s+(?:expired|expiring|outdated|needs update)\s+(?:immediately|urgently|now)\b',
            r'\b(?:security|privacy|safety)\s+(?:settings|preferences|options)\s+(?:update|change|modify)\s+(?:required|needed)\b',
            r'\b(?:verify|confirm|authenticate)\s+(?:your|account|profile)\s+(?:identity|information|details)\s+(?:now|immediately)\b',
        ]
        
        # Payment and financial patterns
        self.payment_financial_patterns = [
            r'\b(?:payment|invoice|bill|charge)\s+(?:failed|declined|rejected|overdue|past due)\s+(?:update|fix|resolve)\s+(?:now|immediately)\b',
            r'\b(?:credit card|debit card|bank account|payment method)\s+(?:expired|invalid|needs update|verification)\s+(?:required|needed)\b',
            r'\b(?:refund|rebate|cashback|reward)\s+(?:available|pending|ready)\s+(?:claim|collect|receive)\s+(?:now|immediately)\b',
            r'\b(?:wire transfer|money transfer|payment)\s+(?:urgent|immediate|emergency)\s+(?:request|needed|required)\b',
            r'\b(?:gift card|prepaid card|voucher)\s+(?:balance|funds|credit)\s+(?:expired|expiring|needs verification)\b',
        ]
        
        # Social media and communication patterns
        self.social_media_patterns = [
            r'\b(?:facebook|twitter|instagram|linkedin|tiktok)\s+(?:account|profile|page)\s+(?:suspended|locked|compromised|hacked)\b',
            r'\b(?:friend|follower|connection)\s+(?:request|invitation|message)\s+(?:urgent|immediate|suspicious)\b',
            r'\b(?:post|photo|video|message)\s+(?:reported|flagged|removed|blocked)\s+(?:appeal|review|verification)\s+(?:required|needed)\b',
            r'\b(?:privacy|security|safety)\s+(?:concern|issue|problem)\s+(?:with|regarding)\s+(?:your|account|profile)\b',
            r'\b(?:verify|confirm|authenticate)\s+(?:your|account|profile)\s+(?:to|in order to)\s+(?:continue|proceed|access)\b',
        ]
        
        # Technical and software patterns
        self.technical_patterns = [
            r'\b(?:software|app|program|system)\s+(?:update|upgrade|patch|fix)\s+(?:required|needed|available)\s+(?:install|download)\s+(?:now|immediately)\b',
            r'\b(?:virus|malware|spyware|trojan)\s+(?:detected|found|infected)\s+(?:your|the)\s+(?:computer|device|system)\b',
            r'\b(?:antivirus|security|firewall)\s+(?:expired|expiring|outdated|needs renewal)\s+(?:update|renew|upgrade)\s+(?:now|immediately)\b',
            r'\b(?:backup|recovery|restore)\s+(?:failed|corrupted|unavailable)\s+(?:fix|resolve|restore)\s+(?:now|immediately)\b',
            r'\b(?:license|subscription|membership)\s+(?:expired|expiring|suspended)\s+(?:renew|reactivate|update)\s+(?:now|immediately)\b',
        ]
        
        # Allow custom patterns from options
        if options:
            if "custom_phishing_keywords" in options:
                self.phishing_keywords.extend(options["custom_phishing_keywords"])
            if "custom_urgent_patterns" in options:
                self.urgent_action_patterns.extend(options["custom_urgent_patterns"])
            if "custom_impersonation_patterns" in options:
                self.impersonation_patterns.extend(options["custom_impersonation_patterns"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for phishing and social engineering detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or [])
        
        # Find matching patterns
        triggered_items = []
        
        # Check phishing keywords
        combined_text_lower = combined_text.lower()
        for keyword in self.phishing_keywords:
            if keyword.lower() in combined_text_lower:
                triggered_items.append(f"PHISHING_KEYWORD:{keyword}")
        
        # Check urgent action patterns
        for pattern in self.urgent_action_patterns:
            urgent_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"URGENT_ACTION:{urgent}" for urgent in urgent_matches])
        
        # Check suspicious link patterns
        for pattern in self.suspicious_link_patterns:
            link_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SUSPICIOUS_LINK:{link}" for link in link_matches])
        
        # Check credential harvesting patterns
        for pattern in self.credential_harvesting_patterns:
            credential_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CREDENTIAL_HARVESTING:{credential}" for credential in credential_matches])
        
        # Check impersonation patterns
        for pattern in self.impersonation_patterns:
            impersonation_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"IMPERSONATION:{impersonation}" for impersonation in impersonation_matches])
        
        # Check prize/reward patterns
        for pattern in self.prize_reward_patterns:
            prize_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"PRIZE_REWARD:{prize}" for prize in prize_matches])
        
        # Check account security patterns
        for pattern in self.account_security_patterns:
            security_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"ACCOUNT_SECURITY:{security}" for security in security_matches])
        
        # Check payment/financial patterns
        for pattern in self.payment_financial_patterns:
            payment_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"PAYMENT_FINANCIAL:{payment}" for payment in payment_matches])
        
        # Check social media patterns
        for pattern in self.social_media_patterns:
            social_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SOCIAL_MEDIA:{social}" for social in social_matches])
        
        # Check technical patterns
        for pattern in self.technical_patterns:
            technical_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"TECHNICAL:{technical}" for technical in technical_matches])
        
        # Calculate confidence based on number and type of matches
        if not triggered_items:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_PHISHING_DETECTED",
                details="No phishing or social engineering indicators detected"
            )
        
        # Weight different types of phishing indicators
        critical_count = len([item for item in triggered_items if any(x in item for x in ["PHISHING_KEYWORD:", "CREDENTIAL_HARVESTING:", "IMPERSONATION:"])])
        high_risk_count = len([item for item in triggered_items if any(x in item for x in ["URGENT_ACTION:", "SUSPICIOUS_LINK:", "ACCOUNT_SECURITY:"])])
        medium_risk_count = len([item for item in triggered_items if any(x in item for x in ["PRIZE_REWARD:", "PAYMENT_FINANCIAL:", "SOCIAL_MEDIA:", "TECHNICAL:"])])
        
        # Calculate confidence with weighted scoring
        confidence = min(1.0, (critical_count * 1.0 + high_risk_count * 0.8 + medium_risk_count * 0.6))
        
        return RuleOutput(
            confidence=confidence,
            content_type="PHISHING_DETECTED",
            details=f"Detected {len(triggered_items)} phishing indicators: {critical_count} critical, {high_risk_count} high-risk, {medium_risk_count} medium-risk",
            triggered_keywords=triggered_items
        )


class PhishingRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect phishing and social engineering with better context understanding"""
    
    name = "Phishing Detection LLM Rule"
    description = "Uses LLM to detect phishing and social engineering with context awareness and better accuracy"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for phishing and social engineering detection"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            fallback_rule = PhishingRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find phishing and social engineering indicators
            triggered_keywords = self._llm_find_keywords_with_input("PHISHING_SOCIAL_ENGINEERING", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_PHISHING_DETECTED",
                    details="No phishing or social engineering indicators detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="PHISHING_DETECTED",
                details=f"LLM detected {len(triggered_keywords)} phishing and social engineering indicators",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to pattern-based detection on error
            fallback_rule = PhishingRule()
            return fallback_rule.process(policy_input)


class PhishingBlockAction(ActionBase):
    """Action to block content containing phishing and social engineering indicators"""
    
    name = "Phishing Block Action"
    description = "Blocks content containing phishing attacks and social engineering attempts"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for phishing and social engineering content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        block_message = (
            "This content has been blocked as it contains indicators of phishing attacks "
            "or social engineering attempts that could compromise your security. "
            "Please be cautious of any requests for personal information or urgent actions."
        )
        
        return self.raise_block_error(block_message)


class PhishingBlockAction_LLM(ActionBase):
    """LLM-powered action to block phishing and social engineering with contextual messaging"""
    
    name = "Phishing Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for phishing and social engineering content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for phishing and social engineering content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains phishing or social engineering indicators: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class PhishingAnonymizeAction(ActionBase):
    """Action to anonymize phishing and social engineering content"""
    
    name = "Phishing Anonymize Action"
    description = "Anonymizes phishing and social engineering indicators while preserving content structure"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute anonymization action for phishing and social engineering content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.anonymize_triggered_keywords()


class PhishingReplaceAction(ActionBase):
    """Action to replace phishing and social engineering indicators with safe placeholders"""
    
    name = "Phishing Replace Action"
    description = "Replaces phishing and social engineering indicators with safe placeholders"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute replacement action for phishing and social engineering content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.replace_triggered_keywords("[PHISHING_REDACTED]")


class PhishingRaiseExceptionAction(ActionBase):
    """Action to raise exception for phishing and social engineering content"""
    
    name = "Phishing Raise Exception Action"
    description = "Raises DisallowedOperation exception for phishing and social engineering content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for phishing and social engineering content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains phishing attacks or social engineering "
            "attempts that violate security policies and could compromise user safety."
        )
        return self.raise_exception(exception_message)


class PhishingRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for phishing and social engineering content"""
    
    name = "Phishing Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for phishing and social engineering content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for phishing and social engineering content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains phishing or social engineering indicators: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Phishing Block Policy
# Standard policy that blocks content containing phishing and social engineering indicators using pattern detection
PhishingBlockPolicy = Policy(
    name="Phishing Block Policy",
    description="Blocks content containing phishing attacks and social engineering attempts",
    rule=PhishingRule(),
    action=PhishingBlockAction()
)

## Phishing Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
PhishingBlockPolicy_LLM = Policy(
    name="Phishing Block Policy LLM",
    description="Uses LLM to detect and block phishing and social engineering content with contextual understanding",
    rule=PhishingRule(),
    action=PhishingBlockAction_LLM()
)

## Phishing Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
PhishingBlockPolicy_LLM_Finder = Policy(
    name="Phishing Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of phishing and social engineering content",
    rule=PhishingRule_LLM_Finder(),
    action=PhishingBlockAction()
)

## Phishing Anonymize Policy
# Policy that anonymizes phishing and social engineering indicators while preserving content structure
PhishingAnonymizePolicy = Policy(
    name="Phishing Anonymize Policy",
    description="Anonymizes phishing and social engineering indicators while preserving content",
    rule=PhishingRule(),
    action=PhishingAnonymizeAction()
)

## Phishing Replace Policy
# Policy that replaces phishing and social engineering indicators with safe placeholders
PhishingReplacePolicy = Policy(
    name="Phishing Replace Policy",
    description="Replaces phishing and social engineering indicators with safe placeholders",
    rule=PhishingRule(),
    action=PhishingReplaceAction()
)

## Phishing Raise Exception Policy
# Policy that raises exceptions for phishing and social engineering content
PhishingRaiseExceptionPolicy = Policy(
    name="Phishing Raise Exception Policy",
    description="Raises DisallowedOperation exception when phishing or social engineering indicators are detected",
    rule=PhishingRule(),
    action=PhishingRaiseExceptionAction()
)

## Phishing Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
PhishingRaiseExceptionPolicy_LLM = Policy(
    name="Phishing Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for phishing and social engineering content",
    rule=PhishingRule(),
    action=PhishingRaiseExceptionAction_LLM()
)
