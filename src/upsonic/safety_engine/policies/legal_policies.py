"""
Legal Document and Confidential Information Protection Policies
Handles detection and protection of legal documents, confidential information, and sensitive business data.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class LegalInfoRule(RuleBase):
    """Rule to detect legal documents, confidential information, and sensitive business data"""
    
    name = "Legal Information Rule"
    description = "Detects legal documents, confidential information, trade secrets, and sensitive business data"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Legal document patterns
        self.legal_document_patterns = [
            # Contract numbers and case numbers
            r'\b(?:contract|agreement|case|lawsuit|litigation)\s*(?:number|num|#|id|identifier)?\s*:?\s*[A-Z0-9]{4,12}\b',
            # Legal case citations
            r'\b\d{4}\s+[A-Z]{2,4}\s+\d+\b',  # Year Court Number format
            r'\b[A-Z]{2,4}\s+\d{4}\s+\d+\b',  # Court Year Number format
            # Court docket numbers
            r'\b(?:docket|case)\s*(?:number|num|#)?\s*:?\s*[A-Z0-9]{6,12}\b',
            # Legal document identifiers
            r'\b(?:document|doc|exhibit|attachment)\s*(?:number|num|#|id|identifier)?\s*:?\s*[A-Z0-9]{3,10}\b',
        ]
        
        # Confidential information patterns
        self.confidential_patterns = [
            # Confidential classification markers
            r'\b(?:confidential|proprietary|restricted|classified|sensitive|private)\s*(?:information|data|document|material|content)\b',
            # Security clearance levels
            r'\b(?:top secret|secret|confidential|restricted|unclassified|public)\s*(?:clearance|level|classification)\b',
            # Internal use only markers
            r'\b(?:internal use only|for internal use|company confidential|business confidential)\b',
            # Attorney-client privilege
            r'\b(?:attorney client privilege|attorney-client privilege|legal privilege|privileged communication)\b',
        ]
        
        # Trade secret patterns
        self.trade_secret_patterns = [
            # Patent numbers
            r'\b(?:patent|pat)\s*(?:number|num|#)?\s*:?\s*(?:US|EP|WO)?\d{4,12}\b',
            # Trademark numbers
            r'\b(?:trademark|tm|service mark|sm)\s*(?:number|num|#)?\s*:?\s*[A-Z0-9]{6,12}\b',
            # Copyright information
            r'\b(?:copyright|copyrighted|©)\s*(?:year|date)?\s*:?\s*\d{4}\b',
            # Trade secret identifiers
            r'\b(?:trade secret|proprietary formula|secret recipe|confidential process)\b',
        ]
        
        # Business sensitive patterns
        self.business_sensitive_patterns = [
            # Financial performance data
            r'\b(?:revenue|sales|profit|loss|earnings|income|expense)\s*(?:data|information|report|statement)\b',
            # Customer data
            r'\b(?:customer list|customer database|client list|client database|customer information)\b',
            # Employee data
            r'\b(?:employee|staff|personnel)\s*(?:information|data|records|files|database)\b',
            # Strategic planning
            r'\b(?:business plan|strategic plan|marketing plan|expansion plan|acquisition plan)\b',
            # Pricing information
            r'\b(?:pricing|cost|rate|fee|charge)\s*(?:information|data|strategy|model|structure)\b',
        ]
        
        # Legal proceeding patterns
        self.legal_proceeding_patterns = [
            # Court proceedings
            r'\b(?:court|trial|hearing|deposition|arbitration|mediation)\s*(?:proceeding|case|matter|action)\b',
            # Legal actions
            r'\b(?:lawsuit|litigation|legal action|legal proceeding|civil action|criminal action)\b',
            # Regulatory matters
            r'\b(?:regulatory|compliance|investigation|audit|inquiry|proceeding)\s*(?:matter|case|action)\b',
            # Settlement information
            r'\b(?:settlement|settlement agreement|confidential settlement|non-disclosure agreement|nda)\b',
        ]
        
        # Intellectual property patterns
        self.ip_patterns = [
            # Patent applications
            r'\b(?:patent application|patent filing|patent pending|patent prosecution)\b',
            # Trademark applications
            r'\b(?:trademark application|trademark filing|trademark registration|service mark application)\b',
            # Copyright registrations
            r'\b(?:copyright registration|copyright filing|copyright application)\b',
            # Trade dress and design
            r'\b(?:trade dress|design patent|industrial design|design registration)\b',
        ]
        
        # Legal keywords and phrases
        self.legal_keywords = [
            # Legal document types
            "contract", "agreement", "lease", "license", "permit", "certificate", "warrant",
            "subpoena", "summons", "affidavit", "deposition", "testimony", "evidence",
            "legal opinion", "legal advice", "legal counsel", "attorney", "lawyer", "counsel",
            "legal document", "legal filing", "legal proceeding", "legal action", "legal matter",
            
            # Confidentiality and privacy
            "confidential", "proprietary", "restricted", "classified", "sensitive", "private",
            "confidential information", "proprietary information", "trade secret", "business secret",
            "internal use only", "company confidential", "business confidential", "client confidential",
            "attorney-client privilege", "legal privilege", "privileged communication",
            "non-disclosure agreement", "nda", "confidentiality agreement", "secrecy agreement",
            
            # Legal proceedings
            "lawsuit", "litigation", "legal action", "legal proceeding", "court case", "trial",
            "hearing", "deposition", "arbitration", "mediation", "settlement", "judgment",
            "verdict", "appeal", "motion", "petition", "complaint", "answer", "counterclaim",
            "discovery", "interrogatory", "request for production", "request for admission",
            
            # Regulatory and compliance
            "regulatory", "compliance", "investigation", "audit", "inquiry", "enforcement",
            "regulatory matter", "compliance issue", "regulatory investigation", "regulatory action",
            "cease and desist", "injunction", "restraining order", "temporary restraining order",
            "regulatory filing", "compliance report", "regulatory submission",
            
            # Intellectual property
            "patent", "trademark", "copyright", "trade secret", "intellectual property", "ip",
            "patent application", "patent filing", "patent prosecution", "patent pending",
            "trademark application", "trademark filing", "trademark registration",
            "copyright registration", "copyright filing", "copyright application",
            "trade dress", "design patent", "industrial design", "design registration",
            
            # Business sensitive information
            "business plan", "strategic plan", "marketing plan", "expansion plan", "acquisition plan",
            "financial information", "revenue data", "sales data", "profit data", "earnings data",
            "customer list", "customer database", "client list", "client database",
            "employee information", "personnel records", "staff information", "employee database",
            "pricing information", "cost data", "rate structure", "fee schedule",
            "competitive intelligence", "market research", "business intelligence",
            
            # Legal entities and relationships
            "corporation", "llc", "partnership", "joint venture", "subsidiary", "affiliate",
            "board of directors", "executive committee", "management team", "leadership team",
            "merger", "acquisition", "divestiture", "spin-off", "reorganization", "restructuring",
            
            # Risk and liability
            "liability", "risk", "exposure", "indemnification", "hold harmless", "warranty",
            "guarantee", "surety", "bond", "insurance", "coverage", "claim", "damages",
            "breach of contract", "breach of warranty", "negligence", "malpractice",
            
            # Government and regulatory
            "government", "federal", "state", "local", "municipal", "regulatory agency",
            "sec", "fda", "epa", "osha", "ftc", "doj", "treasury", "irs", "customs",
            "export control", "import control", "sanctions", "embargo", "restricted party"
        ]
        
        # Sensitive legal patterns
        self.sensitive_legal_patterns = [
            # Whistleblower information
            r'\b(?:whistleblower|whistle blower|retaliation|protected disclosure)\b',
            # Internal investigations
            r'\b(?:internal investigation|corporate investigation|misconduct|violation)\b',
            # Regulatory violations
            r'\b(?:regulatory violation|compliance violation|policy violation|code of conduct)\b',
            # Criminal matters
            r'\b(?:criminal|felony|misdemeanor|indictment|prosecution|conviction)\b',
        ]
        
        # Allow custom patterns from options
        if options:
            if "custom_patterns" in options:
                self.legal_document_patterns.extend(options["custom_patterns"].get("legal_document", []))
                self.confidential_patterns.extend(options["custom_patterns"].get("confidential", []))
                self.trade_secret_patterns.extend(options["custom_patterns"].get("trade_secret", []))
            if "custom_keywords" in options:
                self.legal_keywords.extend(options["custom_keywords"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for legal information detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or [])
        
        # Find matching patterns
        triggered_items = []
        
        # Check legal documents
        for pattern in self.legal_document_patterns:
            legal_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"LEGAL_DOCUMENT:{legal}" for legal in legal_matches])
        
        # Check confidential information
        for pattern in self.confidential_patterns:
            conf_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CONFIDENTIAL:{conf}" for conf in conf_matches])
        
        # Check trade secrets
        for pattern in self.trade_secret_patterns:
            trade_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"TRADE_SECRET:{trade}" for trade in trade_matches])
        
        # Check business sensitive information
        for pattern in self.business_sensitive_patterns:
            business_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"BUSINESS_SENSITIVE:{business}" for business in business_matches])
        
        # Check legal proceedings
        for pattern in self.legal_proceeding_patterns:
            proceeding_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"LEGAL_PROCEEDING:{proceeding}" for proceeding in proceeding_matches])
        
        # Check intellectual property
        for pattern in self.ip_patterns:
            ip_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"INTELLECTUAL_PROPERTY:{ip}" for ip in ip_matches])
        
        # Check sensitive legal patterns
        for pattern in self.sensitive_legal_patterns:
            sensitive_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SENSITIVE_LEGAL:{sensitive}" for sensitive in sensitive_matches])
        
        # Check legal keywords
        combined_text_lower = combined_text.lower()
        for keyword in self.legal_keywords:
            if keyword.lower() in combined_text_lower:
                triggered_items.append(f"LEGAL_KEYWORD:{keyword}")
        
        # Calculate confidence based on number and type of matches
        if not triggered_items:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_LEGAL_INFO",
                details="No legal information detected"
            )
        
        # Weight different types of legal information
        critical_count = len([item for item in triggered_items if any(x in item for x in ["CONFIDENTIAL:", "TRADE_SECRET:", "SENSITIVE_LEGAL:"])])
        high_risk_count = len([item for item in triggered_items if any(x in item for x in ["LEGAL_DOCUMENT:", "BUSINESS_SENSITIVE:", "LEGAL_PROCEEDING:"])])
        medium_risk_count = len([item for item in triggered_items if "INTELLECTUAL_PROPERTY:" in item])
        low_risk_count = len([item for item in triggered_items if "LEGAL_KEYWORD:" in item])
        
        # Calculate confidence with weighted scoring
        confidence = min(1.0, (critical_count * 1.0 + high_risk_count * 0.8 + medium_risk_count * 0.6 + low_risk_count * 0.3))
        
        return RuleOutput(
            confidence=confidence,
            content_type="LEGAL_INFO_DETECTED",
            details=f"Detected {len(triggered_items)} legal items: {critical_count} critical, {high_risk_count} high-risk, {medium_risk_count} medium-risk, {low_risk_count} keyword matches",
            triggered_keywords=triggered_items
        )


class LegalInfoRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect legal information with better context understanding"""
    
    name = "Legal Info LLM Detection Rule"
    description = "Uses LLM to detect legal information and confidential data with context awareness and better accuracy"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for legal information detection"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            fallback_rule = LegalInfoRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find legal information
            triggered_keywords = self._llm_find_keywords_with_input("LEGAL_INFORMATION", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_LEGAL_INFO",
                    details="No legal information detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="LEGAL_INFO_DETECTED",
                details=f"LLM detected {len(triggered_keywords)} legal information items",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to pattern-based detection on error
            fallback_rule = LegalInfoRule()
            return fallback_rule.process(policy_input)


class LegalInfoBlockAction(ActionBase):
    """Action to block content containing legal information"""
    
    name = "Legal Info Block Action"
    description = "Blocks content containing legal documents and confidential information"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for legal information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        block_message = (
            "This content has been blocked as it contains legal documents, confidential information, "
            "or sensitive business data that may be subject to attorney-client privilege, "
            "trade secret protection, or other confidentiality obligations. Please remove any "
            "legal or confidential information before resubmitting."
        )
        
        return self.raise_block_error(block_message)


class LegalInfoBlockAction_LLM(ActionBase):
    """LLM-powered action to block legal information content with contextual messaging"""
    
    name = "Legal Info Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for legal information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for legal information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains legal documents or confidential information: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class LegalInfoAnonymizeAction(ActionBase):
    """Action to anonymize legal information content"""
    
    name = "Legal Info Anonymize Action"
    description = "Anonymizes legal information while preserving content structure"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute anonymization action for legal information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.anonymize_triggered_keywords()


class LegalInfoReplaceAction(ActionBase):
    """Action to replace legal information with placeholders"""
    
    name = "Legal Info Replace Action"
    description = "Replaces legal information with safe placeholders"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute replacement action for legal information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.replace_triggered_keywords("[LEGAL_INFO_REDACTED]")


class LegalInfoRaiseExceptionAction(ActionBase):
    """Action to raise exception for legal information content"""
    
    name = "Legal Info Raise Exception Action"
    description = "Raises DisallowedOperation exception for legal information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for legal information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains legal documents or confidential information "
            "that violates attorney-client privilege, trade secret protection, or confidentiality obligations."
        )
        return self.raise_exception(exception_message)


class LegalInfoRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for legal information content"""
    
    name = "Legal Info Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for legal information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for legal information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains legal documents or confidential information: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Legal Info Block Policy
# Standard policy that blocks content containing legal information using pattern detection
LegalInfoBlockPolicy = Policy(
    name="Legal Info Block Policy",
    description="Blocks content containing legal documents and confidential information",
    rule=LegalInfoRule(),
    action=LegalInfoBlockAction()
)

## Legal Info Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
LegalInfoBlockPolicy_LLM = Policy(
    name="Legal Info Block Policy LLM",
    description="Uses LLM to detect and block legal information content with contextual understanding",
    rule=LegalInfoRule(),
    action=LegalInfoBlockAction_LLM()
)

## Legal Info Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
LegalInfoBlockPolicy_LLM_Finder = Policy(
    name="Legal Info Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of legal information content",
    rule=LegalInfoRule_LLM_Finder(),
    action=LegalInfoBlockAction()
)

## Legal Info Anonymize Policy
# Policy that anonymizes legal information while preserving content structure
LegalInfoAnonymizePolicy = Policy(
    name="Legal Info Anonymize Policy",
    description="Anonymizes legal information while preserving content",
    rule=LegalInfoRule(),
    action=LegalInfoAnonymizeAction()
)

## Legal Info Replace Policy
# Policy that replaces legal information with safe placeholders
LegalInfoReplacePolicy = Policy(
    name="Legal Info Replace Policy",
    description="Replaces legal information with safe placeholders",
    rule=LegalInfoRule(),
    action=LegalInfoReplaceAction()
)

## Legal Info Raise Exception Policy
# Policy that raises exceptions for legal information content
LegalInfoRaiseExceptionPolicy = Policy(
    name="Legal Info Raise Exception Policy",
    description="Raises DisallowedOperation exception when legal information is detected",
    rule=LegalInfoRule(),
    action=LegalInfoRaiseExceptionAction()
)

## Legal Info Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
LegalInfoRaiseExceptionPolicy_LLM = Policy(
    name="Legal Info Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for legal information content",
    rule=LegalInfoRule(),
    action=LegalInfoRaiseExceptionAction_LLM()
)
