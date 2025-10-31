"""
Base class for actions
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import asyncio
import random
import string
import re
from ..models import RuleOutput, PolicyOutput
from ..exceptions import DisallowedOperation
from ..llm.upsonic_llm import UpsonicLLMProvider


class ActionBase(ABC):
    """Base class for all actions"""
    
    name: str = "Base Action"
    description: str = "Base action description"
    language: str = "en"  # Default language for this action
    
    def __init__(self):
        self.rule_result: Optional[RuleOutput] = None
        self.original_content: Optional[List[str]] = None
        self.transformation_map: Dict[int, Dict[str, str]] = {}
        self.transformation_index: int = 0
        self.detected_language: str = "en"  # Default language
    
    def execute_action(self, rule_result: RuleOutput, original_content: List[str], 
                      language: Optional[str] = None,
                      language_identify_llm=None,
                      base_llm=None,
                      text_finder_llm=None) -> PolicyOutput:
        """Wrapper method that saves rule_result and original_content, then calls the actual action"""
        self.rule_result = rule_result
        self.original_content = original_content.copy()  # Copy here once!
        self.transformation_map = {}  # Reset transformation map
        self.transformation_index = 0  # Reset index
        
        # Store LLM specifications
        self.language_identify_llm = language_identify_llm
        self.base_llm = base_llm
        self.text_finder_llm = text_finder_llm
        
        # Language detection/setting
        if language and language != "auto":
            self.detected_language = language
        elif language == "auto" or not language:
            # Auto-detect language from content
            self.detected_language = self._detect_content_language(original_content)
        else:
            self.detected_language = "en"  # Default fallback
            
        return self.action(rule_result)

    async def execute_action_async(self, rule_result: RuleOutput, original_content: List[str], 
                      language: Optional[str] = None,
                      language_identify_llm=None,
                      base_llm=None,
                      text_finder_llm=None) -> PolicyOutput:
        """Async wrapper to execute action without blocking the event loop by default."""
        self.rule_result = rule_result
        self.original_content = original_content.copy()
        self.transformation_map = {}
        self.transformation_index = 0
        self.language_identify_llm = language_identify_llm
        self.base_llm = base_llm
        self.text_finder_llm = text_finder_llm
        if language and language != "auto":
            self.detected_language = language
        elif language == "auto" or not language:
            self.detected_language = await self._detect_content_language_async(original_content)
        else:
            self.detected_language = "en"
        return await self.action_async(rule_result)
    
    def _detect_content_language(self, content: List[str]) -> str:
        """Detect language from content using LLM"""
        if not content:
            return "en"
        
        # Combine first few texts for language detection
        combined_text = " ".join(content[:3])  # Use first 3 texts for detection
        if len(combined_text.strip()) == 0:
            return "en"
        
        # Use specified LLM for language detection if provided, otherwise use default
        if isinstance(self.language_identify_llm, UpsonicLLMProvider):
            llm = self.language_identify_llm
        else:
            llm = UpsonicLLMProvider(agent_name="Language Detection Agent", model=self.language_identify_llm)
        try:
            detected_lang = llm.detect_language(combined_text)
            return detected_lang
        except Exception as e:
            return "en"  # Fallback to English
    
    async def _detect_content_language_async(self, content: List[str]) -> str:
        if not content:
            return "en"
        combined_text = " ".join(content[:3])
        if len(combined_text.strip()) == 0:
            return "en"
        # Use existing UpsonicLLMProvider if language_identify_llm is already one, otherwise create new
        if isinstance(self.language_identify_llm, UpsonicLLMProvider):
            llm = self.language_identify_llm
        else:
            llm = UpsonicLLMProvider(agent_name="Language Detection Agent", model=self.language_identify_llm)
        try:
            return await llm.detect_language_async(combined_text)
        except Exception as e:
            return "en"
    

        

    def _translate(self, text: str, target_language: str) -> str:
        """Translate text using specified LLM"""
        if self.__class__.language != target_language:
            # Use existing UpsonicLLMProvider if base_llm is already one, otherwise create new
            if isinstance(self.base_llm, UpsonicLLMProvider):
                llm = self.base_llm
            else:
                llm = UpsonicLLMProvider(agent_name="Translation Agent", model=self.base_llm)
            return llm.translate_text(text, target_language)
        else:
            return text

    async def _translate_async(self, text: str, target_language: str) -> str:
        if self.__class__.language != target_language:
            # Use existing UpsonicLLMProvider if base_llm is already one, otherwise create new
            if isinstance(self.base_llm, UpsonicLLMProvider):
                llm = self.base_llm
            else:
                llm = UpsonicLLMProvider(agent_name="Translation Agent", model=self.base_llm)
            return await llm.translate_text_async(text, target_language)
        return text

    @abstractmethod
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute the action based on rule result"""
        pass

    async def action_async(self, rule_result: RuleOutput) -> PolicyOutput:
        """Async wrapper that defaults to thread execution of sync action."""
        return await asyncio.to_thread(self.action, rule_result)

    def _generate_unique_replacement(self, original: str) -> str:
        """Generate unique replacement maintaining character types"""
        # Check if already transformed
        for entry in self.transformation_map.values():
            if entry["original"] == original:
                return entry["anonymous"]
        
        # Generate new replacement
        replacement = ""
        for char in original:
            if char.isdigit():
                replacement += str(random.randint(0, 9))
            elif char.isalpha():
                if char.isupper():
                    replacement += random.choice(string.ascii_uppercase)
                else:
                    replacement += random.choice(string.ascii_lowercase)
            else:
                replacement += char  # Keep special characters as is
        
        # Store with index
        self.transformation_index += 1
        self.transformation_map[self.transformation_index] = {
            "original": original,
            "anonymous": replacement
        }
        return replacement
    

    def allow_content(self) -> PolicyOutput:
        """Allow content to pass through"""
        original_content = self.original_content or []
        

        return PolicyOutput(
            output_texts=self.original_content or [],
            action_output={
                "action_taken": "ALLOW",
                "success": True,
                "message": self._translate("Content allowed", self.detected_language)
            }
        )

    async def allow_content_async(self) -> PolicyOutput:
        return PolicyOutput(
            output_texts=self.original_content or [],
            action_output={
                "action_taken": "ALLOW",
                "success": True,
                "message": await self._translate_async("Content allowed", self.detected_language)
            }
        )
    
    def raise_block_error(self, message: str) -> PolicyOutput:
        """Block content with a message"""
        # Apply translation if needed
        translated_message = self._translate(message, self.detected_language)
        
        return PolicyOutput(
            output_texts=[translated_message],
            action_output={
                "action_taken": "BLOCK",
                "success": True,
                "message": translated_message
            }
        )

    async def raise_block_error_async(self, message: str) -> PolicyOutput:
        translated_message = await self._translate_async(message, self.detected_language)
        return PolicyOutput(
            output_texts=[translated_message],
            action_output={
                "action_taken": "BLOCK",
                "success": True,
                "message": translated_message
            }
        )
    
    def replace_triggered_keywords(self, replacement: str) -> PolicyOutput:
        """Replace triggered keywords with a replacement string"""
        original_content = self.original_content or []
        triggered_keywords = self.rule_result.triggered_keywords if self.rule_result else []
        
        transformed_content = []
        for text in original_content:
            transformed_text = text
            for keyword in triggered_keywords:
                # Support typed keywords like "CREDIT_CARD:xxxx" by stripping the type prefix
                target = keyword.split(":", 1)[1] if ":" in keyword else keyword
                # Case-insensitive replacement using regex
                pattern = re.compile(re.escape(target), re.IGNORECASE)
                # Store mapping for fixed replacement
                self.transformation_index += 1
                self.transformation_map[self.transformation_index] = {
                    "original": target,
                    "anonymous": replacement
                }
                transformed_text = pattern.sub(replacement, transformed_text)
            transformed_content.append(transformed_text)
        
        # Apply translation if needed

        translated_message = self._translate(f"Keywords replaced with: {replacement}", self.detected_language)
        
        return PolicyOutput(
            output_texts=transformed_content,
            action_output={
                "action_taken": "REPLACE",
                "success": True,
                "message": translated_message
            },
            transformation_map=self.transformation_map.copy()
        )

    async def replace_triggered_keywords_async(self, replacement: str) -> PolicyOutput:
        original_content = self.original_content or []
        triggered_keywords = self.rule_result.triggered_keywords if self.rule_result else []
        transformed_content = []
        for text in original_content:
            transformed_text = text
            for keyword in triggered_keywords:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                self.transformation_index += 1
                self.transformation_map[self.transformation_index] = {
                    "original": keyword,
                    "anonymous": replacement
                }
                transformed_text = pattern.sub(replacement, transformed_text)
            transformed_content.append(transformed_text)
        translated_message = await self._translate_async(f"Keywords replaced with: {replacement}", self.detected_language)
        return PolicyOutput(
            output_texts=transformed_content,
            action_output={
                "action_taken": "REPLACE",
                "success": True,
                "message": translated_message
            },
            transformation_map=self.transformation_map.copy()
        )
    
    def anonymize_triggered_keywords(self) -> PolicyOutput:
        """Anonymize triggered keywords with unique replacements"""
        original_content = self.original_content or []
        triggered_keywords = self.rule_result.triggered_keywords if self.rule_result else []
        
        transformed_content = []
        for text in original_content:
            transformed_text = text
            for keyword in triggered_keywords:
                # Support typed keywords like "CREDIT_CARD:xxxx" by stripping the type prefix
                target = keyword.split(":", 1)[1] if ":" in keyword else keyword
                # Generate unique replacement maintaining character types
                replacement = self._generate_unique_replacement(target)
                # Case-insensitive replacement using regex for robustness
                pattern = re.compile(re.escape(target), re.IGNORECASE)
                transformed_text = pattern.sub(replacement, transformed_text)
            transformed_content.append(transformed_text)
        
        # Apply translation if needed

        translated_message = self._translate("Keywords anonymized with unique replacements", self.detected_language)
        
        return PolicyOutput(
            output_texts=transformed_content,
            action_output={
                "action_taken": "ANONYMIZE",
                "success": True,
                "message": translated_message
            },
            transformation_map=self.transformation_map.copy()
        )

    async def anonymize_triggered_keywords_async(self) -> PolicyOutput:
        original_content = self.original_content or []
        triggered_keywords = self.rule_result.triggered_keywords if self.rule_result else []
        transformed_content = []
        for text in original_content:
            transformed_text = text
            for keyword in triggered_keywords:
                replacement = self._generate_unique_replacement(keyword)
                transformed_text = transformed_text.replace(keyword, replacement)
            transformed_content.append(transformed_text)
        translated_message = await self._translate_async("Keywords anonymized with unique replacements", self.detected_language)
        return PolicyOutput(
            output_texts=transformed_content,
            action_output={
                "action_taken": "ANONYMIZE",
                "success": True,
                "message": translated_message
            },
            transformation_map=self.transformation_map.copy()
        )
    

    def llm_raise_block_error(self, reason: str) -> PolicyOutput:
        """Use LLM to generate block error message"""
        # Use existing UpsonicLLMProvider if base_llm is already one, otherwise create new
        if isinstance(self.base_llm, UpsonicLLMProvider):
            llm = self.base_llm
        else:
            llm = UpsonicLLMProvider(agent_name="Block Error Message Agent", model=self.base_llm)
        llm_message = llm.generate_block_message(reason, language=self.detected_language)
        return PolicyOutput(
            output_texts=[llm_message],
            action_output={
                "action_taken": "BLOCK",
                "success": True,
                "message": llm_message
            }
        )

    async def llm_raise_block_error_async(self, reason: str) -> PolicyOutput:
        # Use existing UpsonicLLMProvider if base_llm is already one, otherwise create new
        if isinstance(self.base_llm, UpsonicLLMProvider):
            llm = self.base_llm
        else:
            llm = UpsonicLLMProvider(agent_name="Block Error Message Agent", model=self.base_llm)
        llm_message = await llm.generate_block_message_async(reason, language=self.detected_language)
        return PolicyOutput(
            output_texts=[llm_message],
            action_output={
                "action_taken": "BLOCK",
                "success": True,
                "message": llm_message
            }
        )
    
    def raise_exception(self, message: str) -> PolicyOutput:
        """Raise DisallowedOperation exception with given message"""
        raise DisallowedOperation(message)
    
    def llm_raise_exception(self, reason: str) -> PolicyOutput:
        """Use LLM to generate exception message and raise DisallowedOperation"""
        # Use existing UpsonicLLMProvider if base_llm is already one, otherwise create new
        if isinstance(self.base_llm, UpsonicLLMProvider):
            llm = self.base_llm
        else:
            llm = UpsonicLLMProvider(agent_name="Exception Message Agent", model=self.base_llm)
        llm_message = llm.generate_block_message(reason)
        raise DisallowedOperation(llm_message)

    async def llm_raise_exception_async(self, reason: str) -> PolicyOutput:
        # Use existing UpsonicLLMProvider if base_llm is already one, otherwise create new
        if isinstance(self.base_llm, UpsonicLLMProvider):
            llm = self.base_llm
        else:
            llm = UpsonicLLMProvider(agent_name="Exception Message Agent", model=self.base_llm)
        llm_message = await llm.generate_block_message_async(reason)
        raise DisallowedOperation(llm_message)