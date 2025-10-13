"""
Cybersecurity Policies
Handles detection and protection against malware, malicious code, and cyber threats.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class CybersecurityRule(RuleBase):
    """Rule to detect cybersecurity threats including malware, malicious code, and cyber attacks"""
    
    name = "Cybersecurity Threat Rule"
    description = "Detects malware, malicious code, cyber attacks, and security vulnerabilities"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Malware and virus patterns
        self.malware_patterns = [
            # Common malware signatures
            r'\b(?:trojan|virus|worm|rootkit|backdoor|keylogger|spyware|adware|ransomware)\b',
            r'\b(?:malware|malicious|infected|compromised|exploit|payload)\b',
            r'\b(?:botnet|zombie|command and control|c2|cnc)\b',
            r'\b(?:phishing|spear phishing|whaling|vishing|smishing)\b',
            
            # Malicious code patterns
            r'\b(?:shellcode|exploit code|buffer overflow|sql injection|xss|csrf)\b',
            r'\b(?:privilege escalation|lateral movement|persistence|exfiltration)\b',
            r'\b(?:zero day|0day|vulnerability|cve-\d{4}-\d{4,7})\b',
            
            # Attack vectors
            r'\b(?:ddos|dos|brute force|dictionary attack|rainbow table)\b',
            r'\b(?:man in the middle|mitm|arp spoofing|dns poisoning)\b',
            r'\b(?:social engineering|pretexting|baiting|quid pro quo)\b',
        ]
        
        # Suspicious file extensions and patterns
        self.suspicious_file_patterns = [
            r'\b(?:\.exe|\.bat|\.cmd|\.scr|\.pif|\.com|\.vbs|\.js|\.jar|\.ps1)\b',
            r'\b(?:\.dll|\.sys|\.drv|\.ocx|\.cpl|\.msi|\.msp|\.mst)\b',
            r'\b(?:\.zip|\.rar|\.7z|\.tar|\.gz)\s+(?:password|protected|encrypted)\b',
            r'\b(?:suspicious|malicious|infected|quarantined)\s+(?:file|attachment|download)\b',
        ]
        
        # Network security threats
        self.network_threat_patterns = [
            r'\b(?:port scan|nmap|masscan|zmap|network reconnaissance)\b',
            r'\b(?:firewall bypass|ids evasion|ips evasion|waf bypass)\b',
            r'\b(?:vpn|proxy|tor|anonymizer|steganography)\s+(?:abuse|misuse|illegal)\b',
            r'\b(?:packet sniffing|network monitoring|traffic analysis)\b',
            r'\b(?:wireless|wifi|bluetooth|rfid)\s+(?:hack|crack|exploit|attack)\b',
        ]
        
        # Social engineering and phishing patterns
        self.social_engineering_patterns = [
            r'\b(?:urgent|immediate|act now|limited time|expires soon)\s+(?:action|response|verification)\b',
            r'\b(?:click here|download now|verify account|update information)\b',
            r'\b(?:suspicious activity|account locked|security breach|unauthorized access)\b',
            r'\b(?:congratulations|you won|free gift|claim prize|lottery winner)\b',
            r'\b(?:bank|paypal|amazon|microsoft|apple)\s+(?:suspended|locked|compromised|verification)\b',
        ]
        
        # Cryptocurrency and dark web threats
        self.crypto_threat_patterns = [
            r'\b(?:cryptocurrency|bitcoin|ethereum|monero)\s+(?:mining|miner|botnet|malware)\b',
            r'\b(?:dark web|deep web|onion|tor)\s+(?:market|service|transaction)\b',
            r'\b(?:ransom|extortion|blackmail)\s+(?:payment|demand|threat)\b',
            r'\b(?:crypto|bitcoin)\s+(?:wallet|address|private key)\s+(?:steal|theft|compromise)\b',
        ]
        
        # System compromise indicators
        self.compromise_patterns = [
            r'\b(?:system compromised|breach detected|unauthorized access|intrusion)\b',
            r'\b(?:data exfiltration|information theft|credential harvesting)\b',
            r'\b(?:lateral movement|privilege escalation|persistence mechanism)\b',
            r'\b(?:command and control|c2 communication|beacon|callback)\b',
            r'\b(?:anti-virus|security software)\s+(?:disabled|bypassed|removed)\b',
        ]
        
        # Vulnerability and exploit patterns
        self.vulnerability_patterns = [
            r'\b(?:cve-\d{4}-\d{4,7}|vulnerability|exploit|0day|zero-day)\b',
            r'\b(?:buffer overflow|stack overflow|heap overflow|format string)\b',
            r'\b(?:sql injection|xss|cross-site scripting|csrf|clickjacking)\b',
            r'\b(?:path traversal|directory traversal|file inclusion|ldap injection)\b',
            r'\b(?:deserialization|xml external entity|server-side request forgery)\b',
        ]
        
        # Allow custom patterns from options
        if options:
            if "custom_malware_patterns" in options:
                self.malware_patterns.extend(options["custom_malware_patterns"])
            if "custom_threat_patterns" in options:
                self.network_threat_patterns.extend(options["custom_threat_patterns"])
            if "custom_vulnerability_patterns" in options:
                self.vulnerability_patterns.extend(options["custom_vulnerability_patterns"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for cybersecurity threat detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or [])
        
        # Find matching patterns
        triggered_items = []
        
        # Check malware patterns
        for pattern in self.malware_patterns:
            malware_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"MALWARE:{malware}" for malware in malware_matches])
        
        # Check suspicious file patterns
        for pattern in self.suspicious_file_patterns:
            file_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SUSPICIOUS_FILE:{file}" for file in file_matches])
        
        # Check network threat patterns
        for pattern in self.network_threat_patterns:
            network_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"NETWORK_THREAT:{network}" for network in network_matches])
        
        # Check social engineering patterns
        for pattern in self.social_engineering_patterns:
            social_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SOCIAL_ENGINEERING:{social}" for social in social_matches])
        
        # Check crypto threat patterns
        for pattern in self.crypto_threat_patterns:
            crypto_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CRYPTO_THREAT:{crypto}" for crypto in crypto_matches])
        
        # Check compromise patterns
        for pattern in self.compromise_patterns:
            compromise_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"COMPROMISE:{compromise}" for compromise in compromise_matches])
        
        # Check vulnerability patterns
        for pattern in self.vulnerability_patterns:
            vuln_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"VULNERABILITY:{vuln}" for vuln in vuln_matches])
        
        # Calculate confidence based on number and type of matches
        if not triggered_items:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_CYBERSECURITY_THREAT",
                details="No cybersecurity threats detected"
            )
        
        # Weight different types of threats
        critical_count = len([item for item in triggered_items if any(x in item for x in ["MALWARE:", "COMPROMISE:", "VULNERABILITY:"])])
        high_risk_count = len([item for item in triggered_items if any(x in item for x in ["NETWORK_THREAT:", "SOCIAL_ENGINEERING:", "CRYPTO_THREAT:"])])
        medium_risk_count = len([item for item in triggered_items if "SUSPICIOUS_FILE:" in item])
        
        # Calculate confidence with weighted scoring
        confidence = min(1.0, (critical_count * 1.0 + high_risk_count * 0.8 + medium_risk_count * 0.6))
        
        return RuleOutput(
            confidence=confidence,
            content_type="CYBERSECURITY_THREAT_DETECTED",
            details=f"Detected {len(triggered_items)} cybersecurity threats: {critical_count} critical, {high_risk_count} high-risk, {medium_risk_count} medium-risk",
            triggered_keywords=triggered_items
        )


class CybersecurityRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect cybersecurity threats with better context understanding"""
    
    name = "Cybersecurity Threat LLM Detection Rule"
    description = "Uses LLM to detect cybersecurity threats with context awareness and better accuracy"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for cybersecurity threat detection"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            fallback_rule = CybersecurityRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find cybersecurity threats
            triggered_keywords = self._llm_find_keywords_with_input("CYBERSECURITY_THREAT", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_CYBERSECURITY_THREAT",
                    details="No cybersecurity threats detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="CYBERSECURITY_THREAT_DETECTED",
                details=f"LLM detected {len(triggered_keywords)} cybersecurity threats",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to pattern-based detection on error
            fallback_rule = CybersecurityRule()
            return fallback_rule.process(policy_input)


class CybersecurityBlockAction(ActionBase):
    """Action to block content containing cybersecurity threats"""
    
    name = "Cybersecurity Block Action"
    description = "Blocks content containing cybersecurity threats and malicious code"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for cybersecurity threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        block_message = (
            "This content has been blocked as it contains cybersecurity threats, "
            "malicious code, or security vulnerabilities that could pose a risk to "
            "system security. Please ensure your content is free from any security threats."
        )
        
        return self.raise_block_error(block_message)


class CybersecurityBlockAction_LLM(ActionBase):
    """LLM-powered action to block cybersecurity threats with contextual messaging"""
    
    name = "Cybersecurity Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for cybersecurity threats"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for cybersecurity threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains cybersecurity threats: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class CybersecurityAnonymizeAction(ActionBase):
    """Action to anonymize cybersecurity threat content"""
    
    name = "Cybersecurity Anonymize Action"
    description = "Anonymizes cybersecurity threats while preserving content structure"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute anonymization action for cybersecurity threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.anonymize_triggered_keywords()


class CybersecurityReplaceAction(ActionBase):
    """Action to replace cybersecurity threats with safe placeholders"""
    
    name = "Cybersecurity Replace Action"
    description = "Replaces cybersecurity threats with safe placeholders"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute replacement action for cybersecurity threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.replace_triggered_keywords("[SECURITY_THREAT_REDACTED]")


class CybersecurityRaiseExceptionAction(ActionBase):
    """Action to raise exception for cybersecurity threat content"""
    
    name = "Cybersecurity Raise Exception Action"
    description = "Raises DisallowedOperation exception for cybersecurity threat content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for cybersecurity threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains cybersecurity threats, malicious code, "
            "or security vulnerabilities that violate security policies."
        )
        return self.raise_exception(exception_message)


class CybersecurityRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for cybersecurity threat content"""
    
    name = "Cybersecurity Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for cybersecurity threats"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for cybersecurity threat content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains cybersecurity threats: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Cybersecurity Block Policy
# Standard policy that blocks content containing cybersecurity threats using pattern detection
CybersecurityBlockPolicy = Policy(
    name="Cybersecurity Block Policy",
    description="Blocks content containing cybersecurity threats and malicious code",
    rule=CybersecurityRule(),
    action=CybersecurityBlockAction()
)

## Cybersecurity Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
CybersecurityBlockPolicy_LLM = Policy(
    name="Cybersecurity Block Policy LLM",
    description="Uses LLM to detect and block cybersecurity threats with contextual understanding",
    rule=CybersecurityRule(),
    action=CybersecurityBlockAction_LLM()
)

## Cybersecurity Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
CybersecurityBlockPolicy_LLM_Finder = Policy(
    name="Cybersecurity Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of cybersecurity threats",
    rule=CybersecurityRule_LLM_Finder(),
    action=CybersecurityBlockAction()
)

## Cybersecurity Anonymize Policy
# Policy that anonymizes cybersecurity threats while preserving content structure
CybersecurityAnonymizePolicy = Policy(
    name="Cybersecurity Anonymize Policy",
    description="Anonymizes cybersecurity threats while preserving content",
    rule=CybersecurityRule(),
    action=CybersecurityAnonymizeAction()
)

## Cybersecurity Replace Policy
# Policy that replaces cybersecurity threats with safe placeholders
CybersecurityReplacePolicy = Policy(
    name="Cybersecurity Replace Policy",
    description="Replaces cybersecurity threats with safe placeholders",
    rule=CybersecurityRule(),
    action=CybersecurityReplaceAction()
)

## Cybersecurity Raise Exception Policy
# Policy that raises exceptions for cybersecurity threat content
CybersecurityRaiseExceptionPolicy = Policy(
    name="Cybersecurity Raise Exception Policy",
    description="Raises DisallowedOperation exception when cybersecurity threats are detected",
    rule=CybersecurityRule(),
    action=CybersecurityRaiseExceptionAction()
)

## Cybersecurity Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
CybersecurityRaiseExceptionPolicy_LLM = Policy(
    name="Cybersecurity Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for cybersecurity threats",
    rule=CybersecurityRule(),
    action=CybersecurityRaiseExceptionAction_LLM()
)
