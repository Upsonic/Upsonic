"""
Financial Information Protection Policies
Handles detection and protection of financial data like credit cards, bank accounts, SSN, etc.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class FinancialInfoRule(RuleBase):
    """Rule to detect financial information including credit cards, bank accounts, and financial data"""
    
    name = "Financial Information Rule"
    description = "Detects credit cards, bank accounts, SSN, financial statements, and other sensitive financial data"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Credit Card patterns (Luhn algorithm compatible)
        self.credit_card_patterns = [
            # Visa: 4XXXXXXXXXXXXXXX
            r'\b4[0-9]{12}(?:[0-9]{3})?\b',
            # MasterCard: 5[1-5]XXXXXXXXXXXXXX or 2[2-7]XXXXXXXXXXXXXX
            r'\b5[1-5][0-9]{14}\b',
            r'\b2[2-7][0-9]{14}\b',
            # American Express: 3[47]XXXXXXXXXXXXX
            r'\b3[47][0-9]{13}\b',
            # Discover: 6(?:011|5[0-9]{2})XXXXXXXXXXXX
            r'\b6(?:011|5[0-9]{2})[0-9]{12}\b',
            # Generic 16-digit format with separators
            r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
            # Generic 15-digit format (Amex)
            r'\b\d{4}[-.\s]?\d{6}[-.\s]?\d{5}\b',
        ]
        
        # CVV/CVC patterns
        self.cvv_patterns = [
            r'\b(?:cvv|cvc|security code|card verification)\s*:?\s*\d{3,4}\b',
            r'\b\d{3,4}\s*(?:cvv|cvc|security code)\b',
        ]
        
        # Bank Account patterns
        self.bank_account_patterns = [
            # US Bank Account (8-17 digits)
            r'\b(?:account|acct|acc)\s*(?:number|num|#)?\s*:?\s*\d{8,17}\b',
            r'\b\d{8,17}\s*(?:account|acct|acc)\s*(?:number|num|#)?\b',
            # IBAN (International Bank Account Number)
            r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b',
            # SWIFT/BIC codes
            r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b',
        ]
        
        # Routing Number patterns (US)
        self.routing_patterns = [
            r'\b(?:routing|aba|transit)\s*(?:number|num|#)?\s*:?\s*\d{9}\b',
            r'\b\d{9}\s*(?:routing|aba|transit)\s*(?:number|num|#)?\b',
        ]
        
        # Social Security Number patterns (US)
        self.ssn_patterns = [
            r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b',
            r'\b(?:ssn|social security)\s*(?:number|num|#)?\s*:?\s*(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b',
        ]
        
        # Tax ID patterns
        self.tax_id_patterns = [
            # EIN (Employer Identification Number)
            r'\b(?:ein|employer id|tax id)\s*(?:number|num|#)?\s*:?\s*\d{2}[-.\s]?\d{7}\b',
            # TIN (Taxpayer Identification Number)
            r'\b(?:tin|taxpayer id)\s*(?:number|num|#)?\s*:?\s*\d{2}[-.\s]?\d{7}\b',
        ]
        
        # Financial statement patterns
        self.financial_statement_patterns = [
            r'\b(?:balance|account balance|available balance|current balance)\s*:?\s*\$?[\d,]+\.?\d*\b',
            r'\b(?:credit limit|spending limit|limit)\s*:?\s*\$?[\d,]+\.?\d*\b',
            r'\b(?:minimum payment|min payment|payment due)\s*:?\s*\$?[\d,]+\.?\d*\b',
            r'\b(?:interest rate|apr|annual percentage rate)\s*:?\s*\d+\.?\d*%?\b',
        ]
        
        # Investment account patterns
        self.investment_patterns = [
            r'\b(?:portfolio|investment|trading)\s*(?:account|acct)\s*(?:number|num|#)?\s*:?\s*\d{6,12}\b',
            r'\b(?:brokerage|trading|investment)\s*(?:account|acct)\s*(?:number|num|#)?\s*:?\s*[A-Z0-9]{6,12}\b',
        ]
        
        # Financial keywords
        self.financial_keywords = [
            "credit card", "debit card", "bank card", "card number", "card details",
            "bank account", "checking account", "savings account", "account number",
            "routing number", "aba number", "transit number", "swift code", "bic code",
            "iban", "international bank account", "wire transfer", "ach transfer",
            "social security number", "ssn", "social security", "tax id", "ein", "tin",
            "credit score", "fico score", "credit report", "credit history",
            "loan number", "mortgage number", "account balance", "available balance",
            "credit limit", "spending limit", "minimum payment", "payment due",
            "interest rate", "apr", "annual percentage rate", "finance charge",
            "investment account", "trading account", "brokerage account", "portfolio",
            "stock symbol", "ticker", "securities", "mutual fund", "etf",
            "crypto wallet", "bitcoin address", "ethereum address", "wallet address",
            "paypal", "venmo", "cash app", "zelle", "apple pay", "google pay",
            "financial statement", "bank statement", "credit report", "tax return",
            "w-2", "1099", "tax form", "irs", "internal revenue service"
        ]
        
        # Cryptocurrency patterns
        self.crypto_patterns = [
            # Bitcoin addresses
            r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
            r'\bbc1[a-z0-9]{39,59}\b',
            # Ethereum addresses
            r'\b0x[a-fA-F0-9]{40}\b',
            # Generic crypto wallet patterns
            r'\b(?:bitcoin|btc|ethereum|eth|wallet)\s*(?:address|addr)?\s*:?\s*[a-zA-Z0-9]{20,60}\b',
        ]
        
        # Allow custom patterns from options
        if options:
            if "custom_patterns" in options:
                self.credit_card_patterns.extend(options["custom_patterns"].get("credit_card", []))
                self.bank_account_patterns.extend(options["custom_patterns"].get("bank_account", []))
                self.crypto_patterns.extend(options["custom_patterns"].get("crypto", []))
            if "custom_keywords" in options:
                self.financial_keywords.extend(options["custom_keywords"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for financial information detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or [])
        
        # Find matching patterns
        triggered_items = []
        
        # Check credit cards
        for pattern in self.credit_card_patterns:
            cc_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CREDIT_CARD:{cc}" for cc in cc_matches])
        
        # Check CVV/CVC
        for pattern in self.cvv_patterns:
            cvv_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CVV:{cvv}" for cvv in cvv_matches])
        
        # Check bank accounts
        for pattern in self.bank_account_patterns:
            bank_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"BANK_ACCOUNT:{bank}" for bank in bank_matches])
        
        # Check routing numbers
        for pattern in self.routing_patterns:
            routing_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"ROUTING_NUMBER:{routing}" for routing in routing_matches])
        
        # Check SSN
        for pattern in self.ssn_patterns:
            ssn_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SSN:{ssn}" for ssn in ssn_matches])
        
        # Check tax IDs
        for pattern in self.tax_id_patterns:
            tax_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"TAX_ID:{tax}" for tax in tax_matches])
        
        # Check financial statements
        for pattern in self.financial_statement_patterns:
            stmt_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"FINANCIAL_STATEMENT:{stmt}" for stmt in stmt_matches])
        
        # Check investment accounts
        for pattern in self.investment_patterns:
            inv_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"INVESTMENT_ACCOUNT:{inv}" for inv in inv_matches])
        
        # Check cryptocurrency
        for pattern in self.crypto_patterns:
            crypto_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"CRYPTO:{crypto}" for crypto in crypto_matches])
        
        # Check financial keywords
        combined_text_lower = combined_text.lower()
        for keyword in self.financial_keywords:
            if keyword.lower() in combined_text_lower:
                triggered_items.append(f"FINANCIAL_KEYWORD:{keyword}")
        
        # Calculate confidence based on number and type of matches
        if not triggered_items:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_FINANCIAL_INFO",
                details="No financial information detected"
            )
        
        # Weight different types of financial information
        critical_count = len([item for item in triggered_items if any(x in item for x in ["CREDIT_CARD:", "SSN:", "BANK_ACCOUNT:", "ROUTING_NUMBER:"])])
        high_risk_count = len([item for item in triggered_items if any(x in item for x in ["CVV:", "TAX_ID:", "CRYPTO:"])])
        medium_risk_count = len([item for item in triggered_items if any(x in item for x in ["FINANCIAL_STATEMENT:", "INVESTMENT_ACCOUNT:"])])
        low_risk_count = len([item for item in triggered_items if "FINANCIAL_KEYWORD:" in item])
        
        # Calculate confidence with weighted scoring
        confidence = min(1.0, (critical_count * 1.0 + high_risk_count * 0.8 + medium_risk_count * 0.6 + low_risk_count * 0.3))
        
        return RuleOutput(
            confidence=confidence,
            content_type="FINANCIAL_INFO_DETECTED",
            details=f"Detected {len(triggered_items)} financial items: {critical_count} critical, {high_risk_count} high-risk, {medium_risk_count} medium-risk, {low_risk_count} keyword matches",
            triggered_keywords=triggered_items
        )


class FinancialInfoRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect financial information with better context understanding"""
    
    name = "Financial Info LLM Detection Rule"
    description = "Uses LLM to detect financial information with context awareness and better accuracy"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for financial information detection"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            fallback_rule = FinancialInfoRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find financial information
            triggered_keywords = self._llm_find_keywords_with_input("FINANCIAL_INFORMATION", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_FINANCIAL_INFO",
                    details="No financial information detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="FINANCIAL_INFO_DETECTED",
                details=f"LLM detected {len(triggered_keywords)} financial information items",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to pattern-based detection on error
            fallback_rule = FinancialInfoRule()
            return fallback_rule.process(policy_input)


class FinancialInfoBlockAction(ActionBase):
    """Action to block content containing financial information"""
    
    name = "Financial Info Block Action"
    description = "Blocks content containing sensitive financial information"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for financial information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        block_message = (
            "This content has been blocked as it contains sensitive financial information "
            "such as credit card numbers, bank account details, social security numbers, "
            "or other financial data. Please remove any financial information before resubmitting."
        )
        
        return self.raise_block_error(block_message)


class FinancialInfoBlockAction_LLM(ActionBase):
    """LLM-powered action to block financial information content with contextual messaging"""
    
    name = "Financial Info Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for financial information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for financial information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains sensitive financial information: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class FinancialInfoAnonymizeAction(ActionBase):
    """Action to anonymize financial information content"""
    
    name = "Financial Info Anonymize Action"
    description = "Anonymizes financial information while preserving content structure"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute anonymization action for financial information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.anonymize_triggered_keywords()


class FinancialInfoReplaceAction(ActionBase):
    """Action to replace financial information with placeholders"""
    
    name = "Financial Info Replace Action"
    description = "Replaces financial information with safe placeholders"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute replacement action for financial information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.replace_triggered_keywords("[FINANCIAL_INFO_REDACTED]")


class FinancialInfoRaiseExceptionAction(ActionBase):
    """Action to raise exception for financial information content"""
    
    name = "Financial Info Raise Exception Action"
    description = "Raises DisallowedOperation exception for financial information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for financial information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains sensitive financial information "
            "that violates financial data protection and privacy policies."
        )
        return self.raise_exception(exception_message)


class FinancialInfoRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for financial information content"""
    
    name = "Financial Info Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for financial information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for financial information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains sensitive financial information: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Financial Info Block Policy
# Standard policy that blocks content containing financial information using pattern detection
FinancialInfoBlockPolicy = Policy(
    name="Financial Info Block Policy",
    description="Blocks content containing sensitive financial information",
    rule=FinancialInfoRule(),
    action=FinancialInfoBlockAction()
)

## Financial Info Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
FinancialInfoBlockPolicy_LLM = Policy(
    name="Financial Info Block Policy LLM",
    description="Uses LLM to detect and block financial information content with contextual understanding",
    rule=FinancialInfoRule(),
    action=FinancialInfoBlockAction_LLM()
)

## Financial Info Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
FinancialInfoBlockPolicy_LLM_Finder = Policy(
    name="Financial Info Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of financial information content",
    rule=FinancialInfoRule_LLM_Finder(),
    action=FinancialInfoBlockAction()
)

## Financial Info Anonymize Policy
# Policy that anonymizes financial information while preserving content structure
FinancialInfoAnonymizePolicy = Policy(
    name="Financial Info Anonymize Policy",
    description="Anonymizes financial information while preserving content",
    rule=FinancialInfoRule(),
    action=FinancialInfoAnonymizeAction()
)

## Financial Info Replace Policy
# Policy that replaces financial information with safe placeholders
FinancialInfoReplacePolicy = Policy(
    name="Financial Info Replace Policy",
    description="Replaces financial information with safe placeholders",
    rule=FinancialInfoRule(),
    action=FinancialInfoReplaceAction()
)

## Financial Info Raise Exception Policy
# Policy that raises exceptions for financial information content
FinancialInfoRaiseExceptionPolicy = Policy(
    name="Financial Info Raise Exception Policy",
    description="Raises DisallowedOperation exception when financial information is detected",
    rule=FinancialInfoRule(),
    action=FinancialInfoRaiseExceptionAction()
)

## Financial Info Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
FinancialInfoRaiseExceptionPolicy_LLM = Policy(
    name="Financial Info Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for financial information content",
    rule=FinancialInfoRule(),
    action=FinancialInfoRaiseExceptionAction_LLM()
)
