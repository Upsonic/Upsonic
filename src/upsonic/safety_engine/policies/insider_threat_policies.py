"""
Insider Threat Detection Policies
Handles detection and protection against insider threats, data exfiltration, and internal security violations.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class InsiderThreatRule(RuleBase):
    """Rule to detect insider threats and internal security violations"""
    
    name = "Insider Threat Detection Rule"
    description = "Detects insider threats, data exfiltration, and internal security violations"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Insider threat keywords
        self.insider_threat_keywords = [
            "insider threat", "internal threat", "malicious insider", "rogue employee",
            "data exfiltration", "data theft", "intellectual property theft", "trade secret theft",
            "sabotage", "espionage", "corporate espionage", "industrial espionage",
            "unauthorized access", "privilege abuse", "account compromise", "credential theft",
            "lateral movement", "persistence", "data hoarding", "unauthorized copying"
        ]
        
        # Data exfiltration patterns
        self.data_exfiltration_patterns = [
            r'\b(?:download|copy|transfer|upload|export|backup)\s+(?:all|entire|complete|bulk)\s+(?:database|data|files|documents|records)\b',
            r'\b(?:mass|bulk|large|huge)\s+(?:download|copy|transfer|upload|export)\s+(?:of|for)\s+(?:data|files|documents|information)\b',
            r'\b(?:unauthorized|illegal|illegitimate)\s+(?:access|download|copy|transfer|export)\s+(?:of|to)\s+(?:sensitive|confidential|proprietary)\b',
            r'\b(?:exfiltrate|steal|take|remove)\s+(?:data|information|files|documents|intellectual property)\s+(?:from|out of)\s+(?:company|organization|system)\b',
            r'\b(?:personal|external|private)\s+(?:email|cloud|storage|device)\s+(?:for|to)\s+(?:storing|keeping|saving)\s+(?:company|work|sensitive)\s+(?:data|files|information)\b',
        ]
        
        # Unauthorized access patterns
        self.unauthorized_access_patterns = [
            r'\b(?:access|login|connect)\s+(?:to|into)\s+(?:systems|databases|servers|networks)\s+(?:without|outside)\s+(?:authorization|permission|approval)\b',
            r'\b(?:escalate|elevate|increase)\s+(?:privileges|permissions|access rights|user rights)\s+(?:without|outside)\s+(?:authorization|approval|justification)\b',
            r'\b(?:bypass|circumvent|avoid)\s+(?:security|authentication|authorization|controls|policies)\b',
            r'\b(?:use|utilize|exploit)\s+(?:shared|stolen|borrowed|found)\s+(?:credentials|passwords|tokens|keys)\b',
            r'\b(?:access|view|download)\s+(?:sensitive|confidential|proprietary|restricted)\s+(?:data|information|files)\s+(?:outside|beyond)\s+(?:job|role|responsibility)\b',
        ]
        
        # Suspicious behavior patterns
        self.suspicious_behavior_patterns = [
            r'\b(?:unusual|abnormal|suspicious|odd|strange)\s+(?:activity|behavior|pattern|access|download)\b',
            r'\b(?:after hours|outside|beyond)\s+(?:business hours|work hours|normal hours|office hours)\s+(?:access|activity|download|transfer)\b',
            r'\b(?:frequent|repeated|multiple|excessive)\s+(?:access|download|copy|transfer)\s+(?:of|to)\s+(?:sensitive|confidential|proprietary)\b',
            r'\b(?:large|huge|massive|excessive)\s+(?:amount|volume|quantity)\s+(?:of|of data|of files|of information)\s+(?:accessed|downloaded|copied|transferred)\b',
            r'\b(?:sudden|rapid|quick|immediate)\s+(?:increase|spike|jump|rise)\s+(?:in|of)\s+(?:data|file|information)\s+(?:access|download|transfer)\b',
        ]
        
        # Disgruntled employee patterns
        self.disgruntled_employee_patterns = [
            r'\b(?:angry|upset|frustrated|disappointed|disgruntled)\s+(?:employee|worker|staff|personnel)\b',
            r'\b(?:threaten|threat|warn|warning)\s+(?:to|about|regarding)\s+(?:quit|leave|resign|harm|damage|destroy)\b',
            r'\b(?:retaliation|revenge|payback|get back|get even)\s+(?:against|for|on)\s+(?:company|employer|boss|manager|colleague)\b',
            r'\b(?:unfair|unjust|wrongful)\s+(?:treatment|termination|dismissal|firing|layoff)\b',
            r'\b(?:discrimination|harassment|bullying|hostile)\s+(?:workplace|environment|atmosphere|culture)\b',
        ]
        
        # Intellectual property theft patterns
        self.ip_theft_patterns = [
            r'\b(?:steal|take|copy|download|transfer)\s+(?:intellectual property|trade secrets|proprietary information|confidential data)\b',
            r'\b(?:competitor|rival|competing)\s+(?:company|firm|organization)\s+(?:for|to|in order to)\s+(?:gain|obtain|acquire)\s+(?:advantage|information|secrets)\b',
            r'\b(?:sell|trade|share|provide|give)\s+(?:intellectual property|trade secrets|proprietary information)\s+(?:to|with)\s+(?:competitor|rival|external party)\b',
            r'\b(?:start|launch|create|establish)\s+(?:competing|rival|similar)\s+(?:business|company|venture)\s+(?:using|with|based on)\s+(?:stolen|proprietary|confidential)\b',
            r'\b(?:patent|copyright|trademark|intellectual property)\s+(?:infringement|violation|theft|misappropriation)\b',
        ]
        
        # System sabotage patterns
        self.sabotage_patterns = [
            r'\b(?:destroy|damage|corrupt|delete|remove)\s+(?:data|files|systems|databases|networks)\b',
            r'\b(?:disable|shut down|turn off|stop|halt)\s+(?:systems|servers|networks|services|processes)\b',
            r'\b(?:introduce|inject|install|deploy)\s+(?:malware|virus|trojan|backdoor|logic bomb)\b',
            r'\b(?:modify|alter|change|tamper)\s+(?:systems|databases|configurations|settings)\s+(?:to|in order to)\s+(?:cause|create|result in)\s+(?:damage|harm|failure)\b',
            r'\b(?:overload|flood|overwhelm)\s+(?:systems|servers|networks)\s+(?:to|in order to)\s+(?:cause|create|result in)\s+(?:crash|failure|outage)\b',
        ]
        
        # Data hoarding patterns
        self.data_hoarding_patterns = [
            r'\b(?:collect|gather|accumulate|hoard)\s+(?:large|huge|excessive|massive)\s+(?:amount|volume|quantity)\s+(?:of|of data|of files|of information)\b',
            r'\b(?:store|keep|save|archive)\s+(?:data|files|information)\s+(?:on|in|to)\s+(?:personal|private|external|unauthorized)\s+(?:devices|storage|accounts|systems)\b',
            r'\b(?:backup|copy|duplicate)\s+(?:entire|complete|all)\s+(?:databases|systems|files|data)\s+(?:to|on|in)\s+(?:personal|private|external)\b',
            r'\b(?:download|save|keep)\s+(?:everything|all|entire|complete)\s+(?:before|prior to|in preparation for)\s+(?:leaving|quitting|resigning|termination)\b',
        ]
        
        # Communication with external parties patterns
        self.external_communication_patterns = [
            r'\b(?:contact|communicate|meet|talk)\s+(?:with|to)\s+(?:competitor|rival|external party|unauthorized person)\b',
            r'\b(?:share|provide|give|send)\s+(?:information|data|files|documents)\s+(?:with|to)\s+(?:competitor|rival|external party|unauthorized person)\b',
            r'\b(?:discuss|talk about|reveal|disclose)\s+(?:company|proprietary|confidential|sensitive)\s+(?:information|secrets|data)\s+(?:with|to)\s+(?:external|unauthorized)\b',
            r'\b(?:job interview|employment|position|opportunity)\s+(?:with|at)\s+(?:competitor|rival|competing company)\b',
        ]
        
        # Allow custom patterns from options
        if options:
            if "custom_insider_keywords" in options:
                self.insider_threat_keywords.extend(options["custom_insider_keywords"])
            if "custom_exfiltration_patterns" in options:
                self.data_exfiltration_patterns.extend(options["custom_exfiltration_patterns"])
            if "custom_sabotage_patterns" in options:
                self.sabotage_patterns.extend(options["custom_sabotage_patterns"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for insider threat detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or [])
        
        # Find matching patterns
        triggered_items = []
        
        # Check insider threat keywords
        combined_text_lower = combined_text.lower()
        for keyword in self.insider_threat_keywords:
            if keyword.lower() in combined_text_lower:
                triggered_items.append(f"INSIDER_THREAT_KEYWORD:{keyword}")
        
        # Check data exfiltration patterns
        for pattern in self.data_exfiltration_patterns:
            exfiltration_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"DATA_EXFILTRATION:{exfiltration}" for exfiltration in exfiltration_matches])
        
        # Check unauthorized access patterns
        for pattern in self.unauthorized_access_patterns:
            access_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"UNAUTHORIZED_ACCESS:{access}" for access in access_matches])
        
        # Check suspicious behavior patterns
        for pattern in self.suspicious_behavior_patterns:
            behavior_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SUSPICIOUS_BEHAVIOR:{behavior}" for behavior in behavior_matches])
        
        # Check disgruntled employee patterns
        for pattern in self.disgruntled_employee_patterns:
            disgruntled_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"DISGRUNTLED_EMPLOYEE:{disgruntled}" for disgruntled in disgruntled_matches])
        
        # Check IP theft patterns
        for pattern in self.ip_theft_patterns:
            ip_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"IP_THEFT:{ip}" for ip in ip_matches])
        
        # Check sabotage patterns
        for pattern in self.sabotage_patterns:
            sabotage_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SABOTAGE:{sabotage}" for sabotage in sabotage_matches])
        
        # Check data hoarding patterns
        for pattern in self.data_hoarding_patterns:
            hoarding_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"DATA_HOARDING:{hoarding}" for hoarding in hoarding_matches])
        
        # Check external communication patterns
        for pattern in self.external_communication_patterns:
            communication_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"EXTERNAL_COMMUNICATION:{communication}" for communication in communication_matches])
        
        # Calculate confidence based on number and type of matches
        if not triggered_items:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_INSIDER_THREAT_DETECTED",
                details="No insider threat indicators detected"
            )
        
        # Weight different types of insider threat indicators
        critical_count = len([item for item in triggered_items if any(x in item for x in ["DATA_EXFILTRATION:", "IP_THEFT:", "SABOTAGE:"])])
        high_risk_count = len([item for item in triggered_items if any(x in item for x in ["UNAUTHORIZED_ACCESS:", "SUSPICIOUS_BEHAVIOR:", "DISGRUNTLED_EMPLOYEE:"])])
        medium_risk_count = len([item for item in triggered_items if any(x in item for x in ["DATA_HOARDING:", "EXTERNAL_COMMUNICATION:", "INSIDER_THREAT_KEYWORD:"])])
        
        # Calculate confidence with weighted scoring
        confidence = min(1.0, (critical_count * 1.0 + high_risk_count * 0.8 + medium_risk_count * 0.6))
        
        return RuleOutput(
            confidence=confidence,
            content_type="INSIDER_THREAT_DETECTED",
            details=f"Detected {len(triggered_items)} insider threat indicators: {critical_count} critical, {high_risk_count} high-risk, {medium_risk_count} medium-risk",
            triggered_keywords=triggered_items
        )


class InsiderThreatRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect insider threats with better context understanding"""
    
    name = "Insider Threat LLM Detection Rule"
    description = "Uses LLM to detect insider threats with context awareness and better accuracy"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for insider threat detection"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            fallback_rule = InsiderThreatRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find insider threat indicators
            triggered_keywords = self._llm_find_keywords_with_input("INSIDER_THREAT", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_INSIDER_THREAT_DETECTED",
                    details="No insider threat indicators detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="INSIDER_THREAT_DETECTED",
                details=f"LLM detected {len(triggered_keywords)} insider threat indicators",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to pattern-based detection on error
            fallback_rule = InsiderThreatRule()
            return fallback_rule.process(policy_input)


class InsiderThreatBlockAction(ActionBase):
    """Action to block content containing insider threat indicators"""
    
    name = "Insider Threat Block Action"
    description = "Blocks content containing insider threat indicators and internal security violations"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for insider threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        block_message = (
            "This content has been blocked as it contains indicators of insider threats "
            "or internal security violations that could pose risks to organizational security. "
            "Please ensure your communications comply with security policies."
        )
        
        return self.raise_block_error(block_message)


class InsiderThreatBlockAction_LLM(ActionBase):
    """LLM-powered action to block insider threats with contextual messaging"""
    
    name = "Insider Threat Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for insider threat content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for insider threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains insider threat indicators: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class InsiderThreatAnonymizeAction(ActionBase):
    """Action to anonymize insider threat content"""
    
    name = "Insider Threat Anonymize Action"
    description = "Anonymizes insider threat indicators while preserving content structure"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute anonymization action for insider threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.anonymize_triggered_keywords()


class InsiderThreatReplaceAction(ActionBase):
    """Action to replace insider threat indicators with safe placeholders"""
    
    name = "Insider Threat Replace Action"
    description = "Replaces insider threat indicators with safe placeholders"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute replacement action for insider threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.replace_triggered_keywords("[INSIDER_THREAT_REDACTED]")


class InsiderThreatRaiseExceptionAction(ActionBase):
    """Action to raise exception for insider threat content"""
    
    name = "Insider Threat Raise Exception Action"
    description = "Raises DisallowedOperation exception for insider threat content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for insider threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains insider threat indicators or "
            "internal security violations that violate organizational security policies."
        )
        return self.raise_exception(exception_message)


class InsiderThreatRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for insider threat content"""
    
    name = "Insider Threat Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for insider threat content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for insider threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains insider threat indicators: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Insider Threat Block Policy
# Standard policy that blocks content containing insider threat indicators using pattern detection
InsiderThreatBlockPolicy = Policy(
    name="Insider Threat Block Policy",
    description="Blocks content containing insider threat indicators and internal security violations",
    rule=InsiderThreatRule(),
    action=InsiderThreatBlockAction()
)

## Insider Threat Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
InsiderThreatBlockPolicy_LLM = Policy(
    name="Insider Threat Block Policy LLM",
    description="Uses LLM to detect and block insider threat content with contextual understanding",
    rule=InsiderThreatRule(),
    action=InsiderThreatBlockAction_LLM()
)

## Insider Threat Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
InsiderThreatBlockPolicy_LLM_Finder = Policy(
    name="Insider Threat Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of insider threat content",
    rule=InsiderThreatRule_LLM_Finder(),
    action=InsiderThreatBlockAction()
)

## Insider Threat Anonymize Policy
# Policy that anonymizes insider threat indicators while preserving content structure
InsiderThreatAnonymizePolicy = Policy(
    name="Insider Threat Anonymize Policy",
    description="Anonymizes insider threat indicators while preserving content",
    rule=InsiderThreatRule(),
    action=InsiderThreatAnonymizeAction()
)

## Insider Threat Replace Policy
# Policy that replaces insider threat indicators with safe placeholders
InsiderThreatReplacePolicy = Policy(
    name="Insider Threat Replace Policy",
    description="Replaces insider threat indicators with safe placeholders",
    rule=InsiderThreatRule(),
    action=InsiderThreatReplaceAction()
)

## Insider Threat Raise Exception Policy
# Policy that raises exceptions for insider threat content
InsiderThreatRaiseExceptionPolicy = Policy(
    name="Insider Threat Raise Exception Policy",
    description="Raises DisallowedOperation exception when insider threat indicators are detected",
    rule=InsiderThreatRule(),
    action=InsiderThreatRaiseExceptionAction()
)

## Insider Threat Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
InsiderThreatRaiseExceptionPolicy_LLM = Policy(
    name="Insider Threat Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for insider threat content",
    rule=InsiderThreatRule(),
    action=InsiderThreatRaiseExceptionAction_LLM()
)
