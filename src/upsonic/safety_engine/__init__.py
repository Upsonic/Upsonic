"""
Upsonic AI Safety Engine - Content filtering and policy enforcement
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import RuleBase, ActionBase, Policy
    from .models import RuleInput, RuleOutput, ActionResult, PolicyInput, PolicyOutput
    from .exceptions import DisallowedOperation
    from .policies import *

def _get_base_classes():
    """Lazy import of base classes."""
    from .base import RuleBase, ActionBase, Policy
    
    return {
        'RuleBase': RuleBase,
        'ActionBase': ActionBase,
        'Policy': Policy,
    }

def _get_model_classes():
    """Lazy import of model classes."""
    from .models import RuleInput, RuleOutput, ActionResult, PolicyInput, PolicyOutput
    
    return {
        'RuleInput': RuleInput,
        'RuleOutput': RuleOutput,
        'ActionResult': ActionResult,
        'PolicyInput': PolicyInput,
        'PolicyOutput': PolicyOutput,
    }

def _get_exception_classes():
    """Lazy import of exception classes."""
    from .exceptions import DisallowedOperation
    
    return {
        'DisallowedOperation': DisallowedOperation,
    }

def _get_policy_classes():
    """Lazy import of policy classes."""
    from .policies import (
        # Original policies
        AdultContentBlockPolicy,
        AnonymizePhoneNumbersPolicy,
        CryptoBlockPolicy,
        CryptoBlockPolicy_LLM_Block,
        CryptoBlockPolicy_LLM_Finder,
        CryptoReplace,
        CryptoRaiseExceptionPolicy,
        CryptoRaiseExceptionPolicy_LLM_Raise,
        SensitiveSocialBlockPolicy,
        SensitiveSocialRaiseExceptionPolicy,
        AdultContentBlockPolicy_LLM,
        AdultContentBlockPolicy_LLM_Finder,
        AdultContentRaiseExceptionPolicy,
        AdultContentRaiseExceptionPolicy_LLM,
        SensitiveSocialBlockPolicy_LLM,
        SensitiveSocialBlockPolicy_LLM_Finder,
        SensitiveSocialRaiseExceptionPolicy_LLM,
        AnonymizePhoneNumbersPolicy_LLM_Finder,
        
        # PII Policies
        PIIBlockPolicy,
        PIIBlockPolicy_LLM,
        PIIBlockPolicy_LLM_Finder,
        PIIAnonymizePolicy,
        PIIReplacePolicy,
        PIIRaiseExceptionPolicy,
        PIIRaiseExceptionPolicy_LLM,
        
        # Financial Policies
        FinancialInfoBlockPolicy,
        FinancialInfoBlockPolicy_LLM,
        FinancialInfoBlockPolicy_LLM_Finder,
        FinancialInfoAnonymizePolicy,
        FinancialInfoReplacePolicy,
        FinancialInfoRaiseExceptionPolicy,
        FinancialInfoRaiseExceptionPolicy_LLM,
        
        # Medical Policies
        MedicalInfoBlockPolicy,
        MedicalInfoBlockPolicy_LLM,
        MedicalInfoBlockPolicy_LLM_Finder,
        MedicalInfoAnonymizePolicy,
        MedicalInfoReplacePolicy,
        MedicalInfoRaiseExceptionPolicy,
        MedicalInfoRaiseExceptionPolicy_LLM,
        
        # Legal Policies
        LegalInfoBlockPolicy,
        LegalInfoBlockPolicy_LLM,
        LegalInfoBlockPolicy_LLM_Finder,
        LegalInfoAnonymizePolicy,
        LegalInfoReplacePolicy,
        LegalInfoRaiseExceptionPolicy,
        LegalInfoRaiseExceptionPolicy_LLM,
        
        # Technical Security Policies
        TechnicalSecurityBlockPolicy,
        TechnicalSecurityBlockPolicy_LLM,
        TechnicalSecurityBlockPolicy_LLM_Finder,
        TechnicalSecurityAnonymizePolicy,
        TechnicalSecurityReplacePolicy,
        TechnicalSecurityRaiseExceptionPolicy,
        TechnicalSecurityRaiseExceptionPolicy_LLM,
        
        # Cybersecurity Policies
        CybersecurityBlockPolicy,
        CybersecurityBlockPolicy_LLM,
        CybersecurityBlockPolicy_LLM_Finder,
        CybersecurityAnonymizePolicy,
        CybersecurityReplacePolicy,
        CybersecurityRaiseExceptionPolicy,
        CybersecurityRaiseExceptionPolicy_LLM,
        
        # Data Privacy Policies
        DataPrivacyBlockPolicy,
        DataPrivacyBlockPolicy_LLM,
        DataPrivacyBlockPolicy_LLM_Finder,
        DataPrivacyAnonymizePolicy,
        DataPrivacyReplacePolicy,
        DataPrivacyRaiseExceptionPolicy,
        DataPrivacyRaiseExceptionPolicy_LLM,
        
        # Fraud Detection Policies
        FraudDetectionBlockPolicy,
        FraudDetectionBlockPolicy_LLM,
        FraudDetectionBlockPolicy_LLM_Finder,
        FraudDetectionAnonymizePolicy,
        FraudDetectionReplacePolicy,
        FraudDetectionRaiseExceptionPolicy,
        FraudDetectionRaiseExceptionPolicy_LLM,
        
        # Phishing Policies
        PhishingBlockPolicy,
        PhishingBlockPolicy_LLM,
        PhishingBlockPolicy_LLM_Finder,
        PhishingAnonymizePolicy,
        PhishingReplacePolicy,
        PhishingRaiseExceptionPolicy,
        PhishingRaiseExceptionPolicy_LLM,
        
        # Insider Threat Policies
        InsiderThreatBlockPolicy,
        InsiderThreatBlockPolicy_LLM,
        InsiderThreatBlockPolicy_LLM_Finder,
        InsiderThreatAnonymizePolicy,
        InsiderThreatReplacePolicy,
        InsiderThreatRaiseExceptionPolicy,
        InsiderThreatRaiseExceptionPolicy_LLM,
    )
    
    return {
        # Original policies
        'AdultContentBlockPolicy': AdultContentBlockPolicy,
        'AnonymizePhoneNumbersPolicy': AnonymizePhoneNumbersPolicy,
        'CryptoBlockPolicy': CryptoBlockPolicy,
        'CryptoRaiseExceptionPolicy': CryptoRaiseExceptionPolicy,
        'SensitiveSocialBlockPolicy': SensitiveSocialBlockPolicy,
        'SensitiveSocialRaiseExceptionPolicy': SensitiveSocialRaiseExceptionPolicy,
        'AdultContentBlockPolicy_LLM': AdultContentBlockPolicy_LLM,
        'AdultContentBlockPolicy_LLM_Finder': AdultContentBlockPolicy_LLM_Finder,
        'AdultContentRaiseExceptionPolicy': AdultContentRaiseExceptionPolicy,
        'AdultContentRaiseExceptionPolicy_LLM': AdultContentRaiseExceptionPolicy_LLM,
        'SensitiveSocialBlockPolicy_LLM': SensitiveSocialBlockPolicy_LLM,
        'SensitiveSocialBlockPolicy_LLM_Finder': SensitiveSocialBlockPolicy_LLM_Finder,
        'SensitiveSocialRaiseExceptionPolicy_LLM': SensitiveSocialRaiseExceptionPolicy_LLM,
        'AnonymizePhoneNumbersPolicy_LLM_Finder': AnonymizePhoneNumbersPolicy_LLM_Finder,
        
        # PII Policies
        'PIIBlockPolicy': PIIBlockPolicy,
        'PIIBlockPolicy_LLM': PIIBlockPolicy_LLM,
        'PIIBlockPolicy_LLM_Finder': PIIBlockPolicy_LLM_Finder,
        'PIIAnonymizePolicy': PIIAnonymizePolicy,
        'PIIReplacePolicy': PIIReplacePolicy,
        'PIIRaiseExceptionPolicy': PIIRaiseExceptionPolicy,
        'PIIRaiseExceptionPolicy_LLM': PIIRaiseExceptionPolicy_LLM,
        
        # Financial Policies
        'FinancialInfoBlockPolicy': FinancialInfoBlockPolicy,
        'FinancialInfoBlockPolicy_LLM': FinancialInfoBlockPolicy_LLM,
        'FinancialInfoBlockPolicy_LLM_Finder': FinancialInfoBlockPolicy_LLM_Finder,
        'FinancialInfoAnonymizePolicy': FinancialInfoAnonymizePolicy,
        'FinancialInfoReplacePolicy': FinancialInfoReplacePolicy,
        'FinancialInfoRaiseExceptionPolicy': FinancialInfoRaiseExceptionPolicy,
        'FinancialInfoRaiseExceptionPolicy_LLM': FinancialInfoRaiseExceptionPolicy_LLM,
        
        # Medical Policies
        'MedicalInfoBlockPolicy': MedicalInfoBlockPolicy,
        'MedicalInfoBlockPolicy_LLM': MedicalInfoBlockPolicy_LLM,
        'MedicalInfoBlockPolicy_LLM_Finder': MedicalInfoBlockPolicy_LLM_Finder,
        'MedicalInfoAnonymizePolicy': MedicalInfoAnonymizePolicy,
        'MedicalInfoReplacePolicy': MedicalInfoReplacePolicy,
        'MedicalInfoRaiseExceptionPolicy': MedicalInfoRaiseExceptionPolicy,
        'MedicalInfoRaiseExceptionPolicy_LLM': MedicalInfoRaiseExceptionPolicy_LLM,
        
        # Legal Policies
        'LegalInfoBlockPolicy': LegalInfoBlockPolicy,
        'LegalInfoBlockPolicy_LLM': LegalInfoBlockPolicy_LLM,
        'LegalInfoBlockPolicy_LLM_Finder': LegalInfoBlockPolicy_LLM_Finder,
        'LegalInfoAnonymizePolicy': LegalInfoAnonymizePolicy,
        'LegalInfoReplacePolicy': LegalInfoReplacePolicy,
        'LegalInfoRaiseExceptionPolicy': LegalInfoRaiseExceptionPolicy,
        'LegalInfoRaiseExceptionPolicy_LLM': LegalInfoRaiseExceptionPolicy_LLM,
        
        # Technical Security Policies
        'TechnicalSecurityBlockPolicy': TechnicalSecurityBlockPolicy,
        'TechnicalSecurityBlockPolicy_LLM': TechnicalSecurityBlockPolicy_LLM,
        'TechnicalSecurityBlockPolicy_LLM_Finder': TechnicalSecurityBlockPolicy_LLM_Finder,
        'TechnicalSecurityAnonymizePolicy': TechnicalSecurityAnonymizePolicy,
        'TechnicalSecurityReplacePolicy': TechnicalSecurityReplacePolicy,
        'TechnicalSecurityRaiseExceptionPolicy': TechnicalSecurityRaiseExceptionPolicy,
        'TechnicalSecurityRaiseExceptionPolicy_LLM': TechnicalSecurityRaiseExceptionPolicy_LLM,
        
        # Cybersecurity Policies
        'CybersecurityBlockPolicy': CybersecurityBlockPolicy,
        'CybersecurityBlockPolicy_LLM': CybersecurityBlockPolicy_LLM,
        'CybersecurityBlockPolicy_LLM_Finder': CybersecurityBlockPolicy_LLM_Finder,
        'CybersecurityAnonymizePolicy': CybersecurityAnonymizePolicy,
        'CybersecurityReplacePolicy': CybersecurityReplacePolicy,
        'CybersecurityRaiseExceptionPolicy': CybersecurityRaiseExceptionPolicy,
        'CybersecurityRaiseExceptionPolicy_LLM': CybersecurityRaiseExceptionPolicy_LLM,
        
        # Data Privacy Policies
        'DataPrivacyBlockPolicy': DataPrivacyBlockPolicy,
        'DataPrivacyBlockPolicy_LLM': DataPrivacyBlockPolicy_LLM,
        'DataPrivacyBlockPolicy_LLM_Finder': DataPrivacyBlockPolicy_LLM_Finder,
        'DataPrivacyAnonymizePolicy': DataPrivacyAnonymizePolicy,
        'DataPrivacyReplacePolicy': DataPrivacyReplacePolicy,
        'DataPrivacyRaiseExceptionPolicy': DataPrivacyRaiseExceptionPolicy,
        'DataPrivacyRaiseExceptionPolicy_LLM': DataPrivacyRaiseExceptionPolicy_LLM,
        
        # Fraud Detection Policies
        'FraudDetectionBlockPolicy': FraudDetectionBlockPolicy,
        'FraudDetectionBlockPolicy_LLM': FraudDetectionBlockPolicy_LLM,
        'FraudDetectionBlockPolicy_LLM_Finder': FraudDetectionBlockPolicy_LLM_Finder,
        'FraudDetectionAnonymizePolicy': FraudDetectionAnonymizePolicy,
        'FraudDetectionReplacePolicy': FraudDetectionReplacePolicy,
        'FraudDetectionRaiseExceptionPolicy': FraudDetectionRaiseExceptionPolicy,
        'FraudDetectionRaiseExceptionPolicy_LLM': FraudDetectionRaiseExceptionPolicy_LLM,
        
        # Phishing Policies
        'PhishingBlockPolicy': PhishingBlockPolicy,
        'PhishingBlockPolicy_LLM': PhishingBlockPolicy_LLM,
        'PhishingBlockPolicy_LLM_Finder': PhishingBlockPolicy_LLM_Finder,
        'PhishingAnonymizePolicy': PhishingAnonymizePolicy,
        'PhishingReplacePolicy': PhishingReplacePolicy,
        'PhishingRaiseExceptionPolicy': PhishingRaiseExceptionPolicy,
        'PhishingRaiseExceptionPolicy_LLM': PhishingRaiseExceptionPolicy_LLM,
        
        # Insider Threat Policies
        'InsiderThreatBlockPolicy': InsiderThreatBlockPolicy,
        'InsiderThreatBlockPolicy_LLM': InsiderThreatBlockPolicy_LLM,
        'InsiderThreatBlockPolicy_LLM_Finder': InsiderThreatBlockPolicy_LLM_Finder,
        'InsiderThreatAnonymizePolicy': InsiderThreatAnonymizePolicy,
        'InsiderThreatReplacePolicy': InsiderThreatReplacePolicy,
        'InsiderThreatRaiseExceptionPolicy': InsiderThreatRaiseExceptionPolicy,
        'InsiderThreatRaiseExceptionPolicy_LLM': InsiderThreatRaiseExceptionPolicy_LLM,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    base_classes = _get_base_classes()
    if name in base_classes:
        return base_classes[name]
    
    model_classes = _get_model_classes()
    if name in model_classes:
        return model_classes[name]
    
    exception_classes = _get_exception_classes()
    if name in exception_classes:
        return exception_classes[name]
    
    policy_classes = _get_policy_classes()
    if name in policy_classes:
        return policy_classes[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__version__ = "0.1.0"
__all__ = [
    "RuleBase", 
    "ActionBase", 
    "Policy", 
    "RuleInput", 
    "RuleOutput", 
    "ActionResult",
    "PolicyInput",
    "PolicyOutput",
    "DisallowedOperation",
    
    # Original policies
    "AdultContentBlockPolicy",
    "AnonymizePhoneNumbersPolicy",
    "CryptoBlockPolicy",
    "CryptoBlockPolicy_LLM_Block",
    "CryptoBlockPolicy_LLM_Finder",
    "CryptoReplace",
    "CryptoRaiseExceptionPolicy",
    "CryptoRaiseExceptionPolicy_LLM_Raise",
    "SensitiveSocialBlockPolicy",
    "SensitiveSocialRaiseExceptionPolicy",
    "AdultContentBlockPolicy_LLM",
    "AdultContentBlockPolicy_LLM_Finder",
    "AdultContentRaiseExceptionPolicy",
    "AdultContentRaiseExceptionPolicy_LLM",
    "SensitiveSocialBlockPolicy_LLM",
    "SensitiveSocialBlockPolicy_LLM_Finder",
    "SensitiveSocialRaiseExceptionPolicy_LLM",
    "AnonymizePhoneNumbersPolicy_LLM_Finder",
    
    # PII Policies
    "PIIBlockPolicy",
    "PIIBlockPolicy_LLM",
    "PIIBlockPolicy_LLM_Finder",
    "PIIAnonymizePolicy",
    "PIIReplacePolicy",
    "PIIRaiseExceptionPolicy",
    "PIIRaiseExceptionPolicy_LLM",
    
    # Financial Policies
    "FinancialInfoBlockPolicy",
    "FinancialInfoBlockPolicy_LLM",
    "FinancialInfoBlockPolicy_LLM_Finder",
    "FinancialInfoAnonymizePolicy",
    "FinancialInfoReplacePolicy",
    "FinancialInfoRaiseExceptionPolicy",
    "FinancialInfoRaiseExceptionPolicy_LLM",
    
    # Medical Policies
    "MedicalInfoBlockPolicy",
    "MedicalInfoBlockPolicy_LLM",
    "MedicalInfoBlockPolicy_LLM_Finder",
    "MedicalInfoAnonymizePolicy",
    "MedicalInfoReplacePolicy",
    "MedicalInfoRaiseExceptionPolicy",
    "MedicalInfoRaiseExceptionPolicy_LLM",
    
    # Legal Policies
    "LegalInfoBlockPolicy",
    "LegalInfoBlockPolicy_LLM",
    "LegalInfoBlockPolicy_LLM_Finder",
    "LegalInfoAnonymizePolicy",
    "LegalInfoReplacePolicy",
    "LegalInfoRaiseExceptionPolicy",
    "LegalInfoRaiseExceptionPolicy_LLM",
    
    # Technical Security Policies
    "TechnicalSecurityBlockPolicy",
    "TechnicalSecurityBlockPolicy_LLM",
    "TechnicalSecurityBlockPolicy_LLM_Finder",
    "TechnicalSecurityAnonymizePolicy",
    "TechnicalSecurityReplacePolicy",
    "TechnicalSecurityRaiseExceptionPolicy",
    "TechnicalSecurityRaiseExceptionPolicy_LLM",
    
    # Cybersecurity Policies
    "CybersecurityBlockPolicy",
    "CybersecurityBlockPolicy_LLM",
    "CybersecurityBlockPolicy_LLM_Finder",
    "CybersecurityAnonymizePolicy",
    "CybersecurityReplacePolicy",
    "CybersecurityRaiseExceptionPolicy",
    "CybersecurityRaiseExceptionPolicy_LLM",
    
    # Data Privacy Policies
    "DataPrivacyBlockPolicy",
    "DataPrivacyBlockPolicy_LLM",
    "DataPrivacyBlockPolicy_LLM_Finder",
    "DataPrivacyAnonymizePolicy",
    "DataPrivacyReplacePolicy",
    "DataPrivacyRaiseExceptionPolicy",
    "DataPrivacyRaiseExceptionPolicy_LLM",
    
    # Fraud Detection Policies
    "FraudDetectionBlockPolicy",
    "FraudDetectionBlockPolicy_LLM",
    "FraudDetectionBlockPolicy_LLM_Finder",
    "FraudDetectionAnonymizePolicy",
    "FraudDetectionReplacePolicy",
    "FraudDetectionRaiseExceptionPolicy",
    "FraudDetectionRaiseExceptionPolicy_LLM",
    
    # Phishing Policies
    "PhishingBlockPolicy",
    "PhishingBlockPolicy_LLM",
    "PhishingBlockPolicy_LLM_Finder",
    "PhishingAnonymizePolicy",
    "PhishingReplacePolicy",
    "PhishingRaiseExceptionPolicy",
    "PhishingRaiseExceptionPolicy_LLM",
    
    # Insider Threat Policies
    "InsiderThreatBlockPolicy",
    "InsiderThreatBlockPolicy_LLM",
    "InsiderThreatBlockPolicy_LLM_Finder",
    "InsiderThreatAnonymizePolicy",
    "InsiderThreatReplacePolicy",
    "InsiderThreatRaiseExceptionPolicy",
    "InsiderThreatRaiseExceptionPolicy_LLM",
]