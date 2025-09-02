"""
Base class for rules
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio

from ..llm.upsonic_llm import UpsonicLLMProvider
from ..models import PolicyInput, RuleOutput



class RuleBase(ABC):
    """Base class for all rules"""
    
    name: str = "Base Rule"
    description: str = "Base rule description"
    language: str = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        self.options = options or {}
        self.text_finder_llm = text_finder_llm
    
    @abstractmethod
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process the input and return a rule result"""
        pass
    


    
    def _llm_find_keywords_with_input(self, content_type: str, policy_input: PolicyInput) -> List[str]:
        """Internal method to find keywords using LLM with rule input"""
        combined_text = " ".join(policy_input.input_texts or [])
        llm = UpsonicLLMProvider(agent_name="Text Finder Agent", model=self.text_finder_llm)
        
        # Use rule's default language or auto-detect
        try:
            detected_language = llm.detect_language(combined_text)
        except Exception as e:
            detected_language = self.language
            
        return llm.find_keywords(content_type, combined_text, language=detected_language)

    async def process_async(self, policy_input: PolicyInput) -> RuleOutput:
        """Async wrapper for process using a thread by default.

        Subclasses can override for true-async processing.
        """
        return await asyncio.to_thread(self.process, policy_input)

    async def _llm_find_keywords_with_input_async(self, content_type: str, policy_input: PolicyInput) -> List[str]:
        """Async variant that uses async LLM helpers when available."""
        combined_text = " ".join(policy_input.input_texts or [])
        llm = UpsonicLLMProvider(agent_name="Text Finder Agent", model=self.text_finder_llm)
        try:
            detected_language = await llm.detect_language_async(combined_text)
        except Exception as e:
            detected_language = self.language
        return await llm.find_keywords_async(content_type, combined_text, language=detected_language)