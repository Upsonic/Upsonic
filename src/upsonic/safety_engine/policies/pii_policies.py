"""
Personal Identifiable Information (PII) Protection Policies
Handles detection and protection of personal data like names, addresses, emails, etc.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class PIIRule(RuleBase):
    """Rule to detect Personal Identifiable Information (PII)"""
    
    name = "PII Detection Rule"
    description = "Detects personal identifiable information including names, addresses, emails, and other sensitive data"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Email patterns
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Phone number patterns (international and various formats)
        self.phone_patterns = [
            r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',  # US format
            r'\+?[0-9]{1,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}',  # International
            r'\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',  # US without country code
            r'\b[0-9]{3}[-.\s]?[0-9]{4}\b',
        ]
        
        # Social Security Number patterns (US)
        self.ssn_pattern = r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b'
        
        # Credit Card patterns (various formats)
        self.credit_card_patterns = [
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',  # Major cards
            r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',  # Generic 16-digit format
        ]
        
        # Address patterns
        self.address_patterns = [
            r'\b\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Place|Pl|Court|Ct)\b',
            r'\b\d{5}(?:-\d{4})?\b',  # US ZIP codes
            r'\b[A-Za-z]{1,2}\d{1,2}[A-Za-z]?\s?\d[A-Za-z]{2}\b',  # UK postal codes
        ]
        
        # Date of birth patterns
        self.dob_patterns = [
            r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b',  # MM/DD/YYYY
            r'\b(?:0[1-9]|[12][0-9]|3[01])[-/](?:0[1-9]|1[0-2])[-/](?:19|20)\d{2}\b',  # DD/MM/YYYY
            r'\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])\b',  # YYYY/MM/DD
        ]
        
        # Driver's License patterns (US)
        self.dl_patterns = [
            r'\b[A-Z]{1,2}\d{6,8}\b',  # Generic format
            r'\b\d{8,9}\b',  # Numeric format
        ]
        
        # Passport patterns
        self.passport_patterns = [
            r'\b[A-Z]{1,2}\d{6,9}\b',  # Generic format
            r'\b\d{9}\b',  # US passport format
        ]
        
        # IP Address patterns
        self.ip_patterns = [
            r'\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',  # IPv4
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',  # IPv6 (simplified)
        ]
        
        # MAC Address patterns
        self.mac_pattern = r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b'
        
        # Common PII keywords - more specific to avoid false positives
        self.pii_keywords = [
            "full name", "first name", "last name", "middle name", "maiden name",
            "date of birth", "birth date", "dob", "age", "birthday",
            "social security", "ssn", "social security number",
            "driver's license", "drivers license", "dl number", "license number",
            "passport number", "passport", "passport id",
            "credit card", "debit card", "card number", "cvv", "cvc",
            "bank account", "account number", "routing number", "iban", "swift",
            "address", "home address", "mailing address", "billing address",
            "phone number", "mobile number", "cell phone", "telephone",
            "email address", "contact email", "personal email", "my email",
            "ip address", "mac address", "device id", "user id",
            "employee id", "student id", "customer id", "member id",
            "tax id", "ein", "tin", "vat number",
            "mother's maiden name", "father's name", "emergency contact",
            "medical record", "patient id", "health insurance", "policy number"
        ]
        
        # Context-aware patterns to avoid false positives
        self.false_positive_patterns = [
            r'\bemail\s+system\b',  # "email system"
            r'\bemail\s+server\b',  # "email server"
            r'\bemail\s+service\b', # "email service"
            r'\bemail\s+client\b',  # "email client"
            r'\bemail\s+protocol\b', # "email protocol"
            r'\bemail\s+security\b', # "email security"
            r'\bemail\s+marketing\b', # "email marketing"
            r'\bemail\s+campaign\b', # "email campaign"
            r'\bemail\s+template\b', # "email template"
            r'\bemail\s+delivery\b', # "email delivery"
            r'\bemail\s+infrastructure\b', # "email infrastructure"
            r'\bemail\s+platform\b', # "email platform"
            r'\bemail\s+software\b', # "email software"
            r'\bemail\s+application\b', # "email application"
            r'\bemail\s+management\b', # "email management"
            r'\bemail\s+administration\b', # "email administration"
            r'\bemail\s+configuration\b', # "email configuration"
            r'\bemail\s+settings\b', # "email settings"
            r'\bemail\s+policy\b', # "email policy"
            r'\bemail\s+compliance\b', # "email compliance"
        ]
        
        # Allow custom patterns from options
        if options:
            if "custom_patterns" in options:
                self.phone_patterns.extend(options["custom_patterns"].get("phone", []))
                self.credit_card_patterns.extend(options["custom_patterns"].get("credit_card", []))
                self.address_patterns.extend(options["custom_patterns"].get("address", []))
            if "custom_keywords" in options:
                self.pii_keywords.extend(options["custom_keywords"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for PII detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or [])
        
        # Find matching patterns
        triggered_items = []
        
        # Check email addresses
        email_matches = re.findall(self.email_pattern, combined_text)
        triggered_items.extend([f"EMAIL:{email}" for email in email_matches])
        
        # Check phone numbers
        for pattern in self.phone_patterns:
            phone_matches = re.findall(pattern, combined_text)
            triggered_items.extend([f"PHONE:{phone}" for phone in phone_matches])
        
        # Check SSN
        ssn_matches = re.findall(self.ssn_pattern, combined_text)
        triggered_items.extend([f"SSN:{ssn}" for ssn in ssn_matches])
        
        # Check credit cards
        for pattern in self.credit_card_patterns:
            cc_matches = re.findall(pattern, combined_text)
            triggered_items.extend([f"CREDIT_CARD:{cc}" for cc in cc_matches])
        
        # Check addresses
        for pattern in self.address_patterns:
            addr_matches = re.findall(pattern, combined_text)
            triggered_items.extend([f"ADDRESS:{addr}" for addr in addr_matches])
        
        # Check dates of birth
        for pattern in self.dob_patterns:
            dob_matches = re.findall(pattern, combined_text)
            triggered_items.extend([f"DOB:{dob}" for dob in dob_matches])
        
        # Check driver's licenses
        for pattern in self.dl_patterns:
            dl_matches = re.findall(pattern, combined_text)
            triggered_items.extend([f"DRIVERS_LICENSE:{dl}" for dl in dl_matches])
        
        # Check passports
        for pattern in self.passport_patterns:
            passport_matches = re.findall(pattern, combined_text)
            triggered_items.extend([f"PASSPORT:{passport}" for passport in passport_matches])
        
        # Check IP addresses
        for pattern in self.ip_patterns:
            ip_matches = re.findall(pattern, combined_text)
            triggered_items.extend([f"IP_ADDRESS:{ip}" for ip in ip_matches])
        
        # Check MAC addresses
        mac_matches = re.findall(self.mac_pattern, combined_text)
        triggered_items.extend([f"MAC_ADDRESS:{mac}" for mac in mac_matches])
        
        # Check PII keywords with false positive filtering
        combined_text_lower = combined_text.lower()
        for keyword in self.pii_keywords:
            if keyword.lower() in combined_text_lower:
                # Check for false positive patterns
                is_false_positive = False
                for pattern in self.false_positive_patterns:
                    if re.search(pattern, combined_text_lower):
                        is_false_positive = True
                        break
                
                if not is_false_positive:
                    triggered_items.append(f"PII_KEYWORD:{keyword}")
        
        # Calculate confidence based on number and type of matches
        if not triggered_items:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_PII",
                details="No PII detected"
            )
        
        # Weight different types of PII
        high_risk_count = len([item for item in triggered_items if any(x in item for x in ["SSN:", "CREDIT_CARD:", "PASSPORT:"])])
        medium_risk_count = len([item for item in triggered_items if any(x in item for x in ["EMAIL:", "PHONE:", "ADDRESS:", "DOB:"])])
        low_risk_count = len([item for item in triggered_items if "PII_KEYWORD:" in item])
        
        # Calculate confidence with weighted scoring
        confidence = min(1.0, (high_risk_count * 0.9 + medium_risk_count * 0.6 + low_risk_count * 0.3))
        
        return RuleOutput(
            confidence=confidence,
            content_type="PII_DETECTED",
            details=f"Detected {len(triggered_items)} PII items: {high_risk_count} high-risk, {medium_risk_count} medium-risk, {low_risk_count} keyword matches",
            triggered_keywords=triggered_items
        )


class PIIRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect PII with better context understanding"""
    
    name = "PII LLM Detection Rule"
    description = "Uses LLM to detect PII with context awareness and better accuracy"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for PII detection"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            fallback_rule = PIIRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find PII
            triggered_keywords = self._llm_find_keywords_with_input("PERSONAL_IDENTIFIABLE_INFORMATION", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_PII",
                    details="No PII detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="PII_DETECTED",
                details=f"LLM detected {len(triggered_keywords)} PII items",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to pattern-based detection on error
            fallback_rule = PIIRule()
            return fallback_rule.process(policy_input)


class PIIBlockAction(ActionBase):
    """Action to block content containing PII"""
    
    name = "PII Block Action"
    description = "Blocks content containing personal identifiable information"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for PII content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        block_message = (
            "This content has been blocked as it contains personal identifiable information (PII) "
            "such as names, addresses, phone numbers, emails, or other sensitive personal data. "
            "Please remove or anonymize any personal information before resubmitting."
        )
        
        return self.raise_block_error(block_message)


class PIIBlockAction_LLM(ActionBase):
    """LLM-powered action to block PII content with contextual messaging"""
    
    name = "PII Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for PII content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for PII content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains personal identifiable information: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class PIIAnonymizeAction(ActionBase):
    """Action to anonymize PII content"""
    
    name = "PII Anonymize Action"
    description = "Anonymizes personal identifiable information while preserving content structure"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute anonymization action for PII content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self._anonymize_pii_with_specific_format()
    
    def _anonymize_pii_with_specific_format(self) -> PolicyOutput:
        """Anonymize PII with specific format: email -> xxxx@xxxxxxx.xxx, phone -> XXX-XXXX"""
        original_content = self.original_content or []
        triggered_keywords = self.rule_result.triggered_keywords if self.rule_result else []
        
        transformed_content = []
        for text in original_content:
            transformed_text = text
            
            for keyword in triggered_keywords:
                if keyword.startswith("EMAIL:"):
                    # Extract the actual email from "EMAIL:john@example.com"
                    email = keyword.split(":", 1)[1]
                    transformed_text = transformed_text.replace(email, "xxxx@xxxxxxx.xxx")
                    
                elif keyword.startswith("PHONE:"):
                    # Extract the actual phone from "PHONE:555-0123"
                    phone = keyword.split(":", 1)[1]
                    transformed_text = transformed_text.replace(phone, "XXX-XXXX")
                    
                elif keyword.startswith("SSN:"):
                    # Extract the actual SSN from "SSN:123-45-6789"
                    ssn = keyword.split(":", 1)[1]
                    transformed_text = transformed_text.replace(ssn, "XXX-XX-XXXX")
                    
                elif keyword.startswith("CREDIT_CARD:"):
                    # Extract the actual credit card from "CREDIT_CARD:1234-5678-9012-3456"
                    cc = keyword.split(":", 1)[1]
                    transformed_text = transformed_text.replace(cc, "XXXX-XXXX-XXXX-XXXX")
                    
                elif keyword.startswith("ADDRESS:"):
                    # Extract the actual address from "ADDRESS:123 Main St"
                    address = keyword.split(":", 1)[1]
                    transformed_text = transformed_text.replace(address, "[ADDRESS_REDACTED]")
                    
                elif keyword.startswith("DOB:"):
                    # Extract the actual DOB from "DOB:01/01/1990"
                    dob = keyword.split(":", 1)[1]
                    transformed_text = transformed_text.replace(dob, "XX/XX/XXXX")
                    
                elif keyword.startswith("DRIVERS_LICENSE:"):
                    # Extract the actual DL from "DRIVERS_LICENSE:ABC123456"
                    dl = keyword.split(":", 1)[1]
                    transformed_text = transformed_text.replace(dl, "XXX-XXXXXX")
                    
                elif keyword.startswith("PASSPORT:"):
                    # Extract the actual passport from "PASSPORT:AB1234567"
                    passport = keyword.split(":", 1)[1]
                    transformed_text = transformed_text.replace(passport, "XX-XXXXXXX")
                    
                elif keyword.startswith("IP_ADDRESS:"):
                    # Extract the actual IP from "IP_ADDRESS:192.168.1.1"
                    ip = keyword.split(":", 1)[1]
                    transformed_text = transformed_text.replace(ip, "XXX.XXX.XXX.XXX")
                    
                elif keyword.startswith("MAC_ADDRESS:"):
                    # Extract the actual MAC from "MAC_ADDRESS:AA:BB:CC:DD:EE:FF"
                    mac = keyword.split(":", 1)[1]
                    transformed_text = transformed_text.replace(mac, "XX:XX:XX:XX:XX:XX")
                    
                elif keyword.startswith("PII_KEYWORD:"):
                    # For PII keywords, we don't need to replace anything as they're just indicators
                    pass
            
            transformed_content.append(transformed_text)
        
        translated_message = self._translate("PII anonymized with specific format", self.detected_language)
        
        return PolicyOutput(
            output_texts=transformed_content,
            action_output={
                "action_taken": "ANONYMIZE",
                "success": True,
                "message": translated_message
            },
            transformation_map=self.transformation_map.copy()
        )


class PIIReplaceAction(ActionBase):
    """Action to replace PII with placeholders"""
    
    name = "PII Replace Action"
    description = "Replaces personal identifiable information with safe placeholders"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute replacement action for PII content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.replace_triggered_keywords("[PII_REDACTED]")


class PIIRaiseExceptionAction(ActionBase):
    """Action to raise exception for PII content"""
    
    name = "PII Raise Exception Action"
    description = "Raises DisallowedOperation exception for PII content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for PII content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains personal identifiable information (PII) "
            "that violates privacy and data protection policies."
        )
        return self.raise_exception(exception_message)


class PIIRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for PII content"""
    
    name = "PII Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for PII content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for PII content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains personal identifiable information: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## PII Block Policy
# Standard policy that blocks content containing PII using pattern detection
PIIBlockPolicy = Policy(
    name="PII Block Policy",
    description="Blocks content containing personal identifiable information",
    rule=PIIRule(),
    action=PIIBlockAction()
)

## PII Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
PIIBlockPolicy_LLM = Policy(
    name="PII Block Policy LLM",
    description="Uses LLM to detect and block PII content with contextual understanding",
    rule=PIIRule(),
    action=PIIBlockAction_LLM()
)

## PII Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
PIIBlockPolicy_LLM_Finder = Policy(
    name="PII Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of PII content",
    rule=PIIRule_LLM_Finder(),
    action=PIIBlockAction()
)

## PII Anonymize Policy
# Policy that anonymizes PII while preserving content structure
PIIAnonymizePolicy = Policy(
    name="PII Anonymize Policy",
    description="Anonymizes personal identifiable information while preserving content",
    rule=PIIRule(),
    action=PIIAnonymizeAction()
)

## PII Replace Policy
# Policy that replaces PII with safe placeholders
PIIReplacePolicy = Policy(
    name="PII Replace Policy",
    description="Replaces personal identifiable information with safe placeholders",
    rule=PIIRule(),
    action=PIIReplaceAction()
)

## PII Raise Exception Policy
# Policy that raises exceptions for PII content
PIIRaiseExceptionPolicy = Policy(
    name="PII Raise Exception Policy",
    description="Raises DisallowedOperation exception when PII is detected",
    rule=PIIRule(),
    action=PIIRaiseExceptionAction()
)

## PII Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
PIIRaiseExceptionPolicy_LLM = Policy(
    name="PII Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for PII content",
    rule=PIIRule(),
    action=PIIRaiseExceptionAction_LLM()
)
