"""
Technical Security Policies
Handles detection and protection of API keys, passwords, tokens, and other technical credentials.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class TechnicalSecurityRule(RuleBase):
    """Rule to detect technical security information including API keys, passwords, tokens, etc."""
    
    name = "Technical Security Rule"
    description = "Detects API keys, passwords, tokens, certificates, and other technical credentials"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # API Key patterns
        self.api_key_patterns = [
            # OpenAI API keys
            r'\bsk-[a-zA-Z0-9]{20,}\b',
            # AWS Access Keys
            r'\bAKIA[0-9A-Z]{16}\b',
            # AWS Secret Keys
            r'\b[A-Za-z0-9/+=]{40}\b',
            # Google API keys
            r'\bAIza[0-9A-Za-z\\-_]{35}\b',
            # GitHub tokens
            r'\bghp_[a-zA-Z0-9]{36}\b',
            r'\bgho_[a-zA-Z0-9]{36}\b',
            r'\bghu_[a-zA-Z0-9]{36}\b',
            r'\bghs_[a-zA-Z0-9]{36}\b',
            r'\bghr_[a-zA-Z0-9]{36}\b',
            # Generic API key patterns
            r'\b(?:api[_-]?key|apikey|access[_-]?key|secret[_-]?key)\s*[:=]\s*[a-zA-Z0-9_\-]{20,}\b',
            r'\b[a-zA-Z0-9_\-]{20,}\s*(?:api[_-]?key|apikey|access[_-]?key|secret[_-]?key)\b',
        ]
        
        # Password patterns
        self.password_patterns = [
            # Common password field patterns
            r'\b(?:password|passwd|pwd|pass)\s*[:=]\s*[^\s]{6,}\b',
            r'\b(?:password|passwd|pwd|pass)\s*[:=]\s*["\'][^"\']{6,}["\']\b',
            # Database connection strings with passwords
            r'\b(?:mysql|postgresql|postgres|mongodb|redis|sqlite)\s*://[^:]+:[^@]+@',
            # Environment variable patterns
            r'\b(?:password|passwd|pwd|pass|secret|token|key)\s*=\s*[^\s]{6,}\b',
            # Base64 encoded passwords (common in configs)
            r'\b(?:password|passwd|pwd|pass)\s*[:=]\s*[A-Za-z0-9+/]{20,}={0,2}\b',
        ]
        
        # Token patterns
        self.token_patterns = [
            # JWT tokens
            r'\beyJ[a-zA-Z0-9_\-]+\.eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\b',
            # OAuth tokens
            r'\b(?:oauth|bearer|access[_-]?token|refresh[_-]?token)\s*[:=]\s*[a-zA-Z0-9_\-]{20,}\b',
            # Session tokens
            r'\b(?:session[_-]?id|session[_-]?token|jsessionid)\s*[:=]\s*[a-zA-Z0-9_\-]{20,}\b',
            # CSRF tokens
            r'\b(?:csrf[_-]?token|xsrf[_-]?token)\s*[:=]\s*[a-zA-Z0-9_\-]{20,}\b',
            # Generic token patterns
            r'\b[a-zA-Z0-9_\-]{32,}\s*(?:token|auth|session)\b',
        ]
        
        # Certificate patterns
        self.certificate_patterns = [
            # Private keys (PEM format)
            r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
            # Certificates (PEM format)
            r'-----BEGIN\s+CERTIFICATE-----[\s\S]*?-----END\s+CERTIFICATE-----',
            # Certificate requests
            r'-----BEGIN\s+(?:NEW\s+)?CERTIFICATE\s+REQUEST-----[\s\S]*?-----END\s+(?:NEW\s+)?CERTIFICATE\s+REQUEST-----',
            # SSH private keys
            r'-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH)\s+PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA|DSA|EC|OPENSSH)\s+PRIVATE\s+KEY-----',
        ]
        
        # Database connection patterns
        self.database_patterns = [
            # Connection strings with credentials
            r'\b(?:mysql|postgresql|postgres|mongodb|redis|sqlite|oracle|sqlserver)\s*://[^:]+:[^@]+@[^\s]+\b',
            # JDBC connection strings
            r'\bjdbc:(?:mysql|postgresql|oracle|sqlserver)://[^:]+:[^@]+@[^\s]+\b',
            # Database URLs with embedded credentials
            r'\b(?:database[_-]?url|db[_-]?url|connection[_-]?string)\s*[:=]\s*[^\s]*://[^:]+:[^@]+@[^\s]+\b',
        ]
        
        # Cloud service patterns
        self.cloud_patterns = [
            # AWS patterns
            r'\b(?:aws[_-]?access[_-]?key[_-]?id|aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*[a-zA-Z0-9_\-]{20,}\b',
            # Azure patterns
            r'\b(?:azure[_-]?storage[_-]?account[_-]?key|azure[_-]?account[_-]?key)\s*[:=]\s*[a-zA-Z0-9_\-]{20,}\b',
            # Google Cloud patterns
            r'\b(?:google[_-]?application[_-]?credentials|gcp[_-]?service[_-]?account[_-]?key)\s*[:=]\s*[^\s]+\b',
            # Generic cloud patterns
            r'\b(?:cloud[_-]?key|service[_-]?key|account[_-]?key)\s*[:=]\s*[a-zA-Z0-9_\-]{20,}\b',
        ]
        
        # Configuration file patterns
        self.config_patterns = [
            # Environment files
            r'\b(?:\.env|environment|config)\s*[:=]\s*[^\s]+\b',
            # Configuration with secrets
            r'\b(?:secret|password|token|key|credential)\s*[:=]\s*[^\s]{6,}\b',
            # YAML/JSON config patterns
            r'\b(?:secret|password|token|key|credential)\s*:\s*[^\s]{6,}\b',
        ]
        
        # Technical security keywords
        self.technical_keywords = [
            # API and authentication
            "api key", "apikey", "api_key", "access key", "access_key", "secret key", "secret_key",
            "authentication token", "auth token", "bearer token", "oauth token", "jwt token",
            "session token", "session id", "jsessionid", "csrf token", "xsrf token",
            "refresh token", "access token", "client secret", "client_secret",
            
            # Passwords and credentials
            "password", "passwd", "pwd", "pass", "login", "credential", "credentials",
            "username", "user name", "userid", "user id", "email", "e-mail",
            "private key", "private_key", "public key", "public_key", "ssh key", "ssh_key",
            
            # Database and connection
            "database password", "db password", "connection string", "connection_string",
            "database url", "db url", "database_url", "db_url", "connection url",
            "mysql", "postgresql", "postgres", "mongodb", "redis", "sqlite", "oracle",
            "jdbc", "connection pool", "connection_pool", "data source", "data_source",
            
            # Cloud services
            "aws access key", "aws_access_key", "aws secret key", "aws_secret_key",
            "azure storage key", "azure_storage_key", "azure account key", "azure_account_key",
            "google api key", "google_api_key", "gcp service account", "gcp_service_account",
            "cloud credentials", "cloud_credentials", "service account", "service_account",
            
            # Certificates and keys
            "ssl certificate", "ssl_certificate", "tls certificate", "tls_certificate",
            "x509 certificate", "x509_certificate", "pem file", "pem_file", "p12 file", "p12_file",
            "keystore", "key store", "truststore", "trust store", "certificate authority", "ca",
            
            # Configuration and environment
            "environment variable", "env var", "env_var", "config file", "config_file",
            ".env file", ".env_file", "configuration", "settings", "properties",
            "environment", "env", "config", "settings file", "settings_file",
            
            # Security tokens
            "security token", "security_token", "identity token", "identity_token",
            "federation token", "federation_token", "assume role", "assume_role",
            "temporary credentials", "temporary_credentials", "sts token", "sts_token",
            
            # Webhooks and integrations
            "webhook secret", "webhook_secret", "webhook key", "webhook_key",
            "integration key", "integration_key", "app secret", "app_secret",
            "client id", "client_id", "application id", "application_id",
            
            # Development and deployment
            "deployment key", "deployment_key", "build secret", "build_secret",
            "ci cd token", "ci_cd_token", "pipeline secret", "pipeline_secret",
            "docker registry", "docker_registry", "container registry", "container_registry"
        ]
        
        # Sensitive technical patterns
        self.sensitive_technical_patterns = [
            # Hardcoded credentials
            r'\b(?:password|passwd|pwd|pass|secret|token|key)\s*[:=]\s*["\'][^"\']{6,}["\']\b',
            # Base64 encoded secrets
            r'\b(?:secret|password|token|key)\s*[:=]\s*[A-Za-z0-9+/]{20,}={0,2}\b',
            # Hex encoded secrets
            r'\b(?:secret|password|token|key)\s*[:=]\s*[0-9a-fA-F]{32,}\b',
        ]
        
        # Allow custom patterns from options
        if options:
            if "custom_patterns" in options:
                self.api_key_patterns.extend(options["custom_patterns"].get("api_key", []))
                self.password_patterns.extend(options["custom_patterns"].get("password", []))
                self.token_patterns.extend(options["custom_patterns"].get("token", []))
            if "custom_keywords" in options:
                self.technical_keywords.extend(options["custom_keywords"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for technical security information detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or [])
        
        # Find matching patterns
        triggered_items = []
        
        # Check API keys
        for pattern in self.api_key_patterns:
            api_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"API_KEY:{api}" for api in api_matches])
        
        # Check passwords
        for pattern in self.password_patterns:
            pwd_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"PASSWORD:{pwd}" for pwd in pwd_matches])
        
        # Check tokens
        for pattern in self.token_patterns:
            token_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"TOKEN:{token}" for token in token_matches])
        
        # Check certificates
        for pattern in self.certificate_patterns:
            cert_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CERTIFICATE:{cert[:50]}..." for cert in cert_matches])
        
        # Check database connections
        for pattern in self.database_patterns:
            db_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"DATABASE:{db}" for db in db_matches])
        
        # Check cloud services
        for pattern in self.cloud_patterns:
            cloud_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CLOUD:{cloud}" for cloud in cloud_matches])
        
        # Check configuration files
        for pattern in self.config_patterns:
            config_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CONFIG:{config}" for config in config_matches])
        
        # Check sensitive technical patterns
        for pattern in self.sensitive_technical_patterns:
            sensitive_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SENSITIVE_TECHNICAL:{sensitive}" for sensitive in sensitive_matches])
        
        # Check technical keywords
        combined_text_lower = combined_text.lower()
        for keyword in self.technical_keywords:
            if keyword.lower() in combined_text_lower:
                triggered_items.append(f"TECHNICAL_KEYWORD:{keyword}")
        
        # Calculate confidence based on number and type of matches
        if not triggered_items:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_TECHNICAL_SECURITY",
                details="No technical security information detected"
            )
        
        # Weight different types of technical security information
        critical_count = len([item for item in triggered_items if any(x in item for x in ["API_KEY:", "PASSWORD:", "CERTIFICATE:", "SENSITIVE_TECHNICAL:"])])
        high_risk_count = len([item for item in triggered_items if any(x in item for x in ["TOKEN:", "DATABASE:", "CLOUD:"])])
        medium_risk_count = len([item for item in triggered_items if "CONFIG:" in item])
        low_risk_count = len([item for item in triggered_items if "TECHNICAL_KEYWORD:" in item])
        
        # Calculate confidence with weighted scoring
        confidence = min(1.0, (critical_count * 1.0 + high_risk_count * 0.8 + medium_risk_count * 0.6 + low_risk_count * 0.3))
        
        return RuleOutput(
            confidence=confidence,
            content_type="TECHNICAL_SECURITY_DETECTED",
            details=f"Detected {len(triggered_items)} technical security items: {critical_count} critical, {high_risk_count} high-risk, {medium_risk_count} medium-risk, {low_risk_count} keyword matches",
            triggered_keywords=triggered_items
        )


class TechnicalSecurityRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect technical security information with better context understanding"""
    
    name = "Technical Security LLM Detection Rule"
    description = "Uses LLM to detect technical security information with context awareness and better accuracy"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for technical security information detection"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            fallback_rule = TechnicalSecurityRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find technical security information
            triggered_keywords = self._llm_find_keywords_with_input("TECHNICAL_SECURITY_INFORMATION", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_TECHNICAL_SECURITY",
                    details="No technical security information detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="TECHNICAL_SECURITY_DETECTED",
                details=f"LLM detected {len(triggered_keywords)} technical security information items",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to pattern-based detection on error
            fallback_rule = TechnicalSecurityRule()
            return fallback_rule.process(policy_input)


class TechnicalSecurityBlockAction(ActionBase):
    """Action to block content containing technical security information"""
    
    name = "Technical Security Block Action"
    description = "Blocks content containing technical security information like API keys and passwords"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for technical security information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        block_message = (
            "This content has been blocked as it contains technical security information "
            "such as API keys, passwords, tokens, certificates, or other sensitive credentials. "
            "Please remove any technical security information before resubmitting."
        )
        
        return self.raise_block_error(block_message)


class TechnicalSecurityBlockAction_LLM(ActionBase):
    """LLM-powered action to block technical security information content with contextual messaging"""
    
    name = "Technical Security Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for technical security information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for technical security information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains technical security information: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class TechnicalSecurityAnonymizeAction(ActionBase):
    """Action to anonymize technical security information content"""
    
    name = "Technical Security Anonymize Action"
    description = "Anonymizes technical security information while preserving content structure"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute anonymization action for technical security information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.anonymize_triggered_keywords()


class TechnicalSecurityReplaceAction(ActionBase):
    """Action to replace technical security information with placeholders"""
    
    name = "Technical Security Replace Action"
    description = "Replaces technical security information with safe placeholders"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute replacement action for technical security information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.replace_triggered_keywords("[TECHNICAL_SECURITY_REDACTED]")


class TechnicalSecurityRaiseExceptionAction(ActionBase):
    """Action to raise exception for technical security information content"""
    
    name = "Technical Security Raise Exception Action"
    description = "Raises DisallowedOperation exception for technical security information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for technical security information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains technical security information "
            "such as API keys, passwords, or tokens that violates security policies."
        )
        return self.raise_exception(exception_message)


class TechnicalSecurityRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for technical security information content"""
    
    name = "Technical Security Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for technical security information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for technical security information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains technical security information: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Technical Security Block Policy
# Standard policy that blocks content containing technical security information using pattern detection
TechnicalSecurityBlockPolicy = Policy(
    name="Technical Security Block Policy",
    description="Blocks content containing technical security information like API keys and passwords",
    rule=TechnicalSecurityRule(),
    action=TechnicalSecurityBlockAction()
)

## Technical Security Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
TechnicalSecurityBlockPolicy_LLM = Policy(
    name="Technical Security Block Policy LLM",
    description="Uses LLM to detect and block technical security information content with contextual understanding",
    rule=TechnicalSecurityRule(),
    action=TechnicalSecurityBlockAction_LLM()
)

## Technical Security Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
TechnicalSecurityBlockPolicy_LLM_Finder = Policy(
    name="Technical Security Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of technical security information content",
    rule=TechnicalSecurityRule_LLM_Finder(),
    action=TechnicalSecurityBlockAction()
)

## Technical Security Anonymize Policy
# Policy that anonymizes technical security information while preserving content structure
TechnicalSecurityAnonymizePolicy = Policy(
    name="Technical Security Anonymize Policy",
    description="Anonymizes technical security information while preserving content",
    rule=TechnicalSecurityRule(),
    action=TechnicalSecurityAnonymizeAction()
)

## Technical Security Replace Policy
# Policy that replaces technical security information with safe placeholders
TechnicalSecurityReplacePolicy = Policy(
    name="Technical Security Replace Policy",
    description="Replaces technical security information with safe placeholders",
    rule=TechnicalSecurityRule(),
    action=TechnicalSecurityReplaceAction()
)

## Technical Security Raise Exception Policy
# Policy that raises exceptions for technical security information content
TechnicalSecurityRaiseExceptionPolicy = Policy(
    name="Technical Security Raise Exception Policy",
    description="Raises DisallowedOperation exception when technical security information is detected",
    rule=TechnicalSecurityRule(),
    action=TechnicalSecurityRaiseExceptionAction()
)

## Technical Security Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
TechnicalSecurityRaiseExceptionPolicy_LLM = Policy(
    name="Technical Security Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for technical security information content",
    rule=TechnicalSecurityRule(),
    action=TechnicalSecurityRaiseExceptionAction_LLM()
)
