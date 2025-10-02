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
        AdultContentBlockPolicy,
        AnonymizePhoneNumbersPolicy,
        CryptoBlockPolicy,
        CryptoRaiseExceptionPolicy,
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
    )
    
    return {
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
    "AdultContentBlockPolicy",
    "AnonymizePhoneNumbersPolicy",
    "CryptoBlockPolicy",
    "CryptoRaiseExceptionPolicy",
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
]