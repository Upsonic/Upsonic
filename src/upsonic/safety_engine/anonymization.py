"""
Anonymization Utilities for Safety Engine

This module provides standardized functions for:
- Anonymizing sensitive data (PII, etc.) with random values before sending to LLM
- De-anonymizing LLM responses before returning to user

The anonymization uses random character replacement while preserving format:
- Digits are replaced with random digits
- Letters are replaced with random letters  
- Special characters are preserved

Usage:
    from upsonic.safety_engine.anonymization import (
        anonymize_content,
        deanonymize_content,
        AnonymizationResult
    )
    
    # Anonymize sensitive data
    result = anonymize_content(
        content="My email is john@example.com",
        triggered_keywords=["EMAIL:john@example.com"]
    )
    # result.anonymized_content = "My email is xkqp@fhsmwtr.lzn"
    
    # Send to LLM...
    llm_response = "Your email is xkqp@fhsmwtr.lzn"
    
    # De-anonymize the response
    original_response = deanonymize_content(
        content=llm_response,
        transformation_map=result.transformation_map
    )
    # Result: "Your email is john@example.com"
"""

import random
import string
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class AnonymizationResult:
    """Result of anonymization operation"""
    anonymized_content: str
    transformation_map: Dict[int, Dict[str, str]] = field(default_factory=dict)
    anonymized_count: int = 0


class Anonymizer:
    """
    Handles reversible anonymization of sensitive content with random values.
    
    Replaces sensitive data with random characters while preserving format:
    - Digits are replaced with random digits
    - Letters are replaced with random letters
    - Special characters are preserved
    """
    
    def __init__(self):
        self._counter = 0
        self._transformation_map: Dict[int, Dict[str, str]] = {}
        self._anonymized_cache: Dict[str, str] = {}
    
    def reset(self):
        """Reset the anonymizer state"""
        self._counter = 0
        self._transformation_map = {}
        self._anonymized_cache = {}
    
    @property
    def transformation_map(self) -> Dict[int, Dict[str, str]]:
        """Get the current transformation map"""
        return self._transformation_map.copy()
    
    def _get_value_from_keyword(self, keyword: str) -> Optional[str]:
        """
        Extract value from a triggered keyword.
        
        Args:
            keyword: Triggered keyword like "EMAIL:john@example.com" or plain value
            
        Returns:
            The value to anonymize, or None if should be skipped
        """
        if keyword.startswith("PII_KEYWORD:"):
            return None  # Skip keyword indicators
        
        if ":" in keyword:
            return keyword.split(":", 1)[1]
        return keyword
    
    def _generate_random_replacement(self, original: str) -> str:
        """
        Generate a random replacement preserving character types.
        
        Args:
            original: Original string to anonymize
            
        Returns:
            Random replacement with same format
        """
        if original in self._anonymized_cache:
            return self._anonymized_cache[original]
        
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
                replacement += char
        
        self._counter += 1
        self._transformation_map[self._counter] = {
            "original": original,
            "anonymous": replacement
        }
        self._anonymized_cache[original] = replacement
        
        return replacement
    
    def anonymize(self, content: str, triggered_keywords: List[str]) -> AnonymizationResult:
        """
        Anonymize content by replacing triggered keywords with random values.
        
        Args:
            content: Original content to anonymize
            triggered_keywords: List of keywords to anonymize
            
        Returns:
            AnonymizationResult with anonymized content and transformation map
        """
        anonymized_content = content
        anonymized_count = 0
        
        for keyword in triggered_keywords:
            value = self._get_value_from_keyword(keyword)
            if not value:
                continue
            
            replacement = self._generate_random_replacement(value)
            
            pattern = re.compile(re.escape(value), re.IGNORECASE)
            new_content = pattern.sub(replacement, anonymized_content)
            
            if new_content != anonymized_content:
                anonymized_count += 1
                anonymized_content = new_content
        
        return AnonymizationResult(
            anonymized_content=anonymized_content,
            transformation_map=self.transformation_map,
            anonymized_count=anonymized_count
        )
    
    def anonymize_multiple(
        self,
        contents: List[str],
        triggered_keywords: List[str]
    ) -> tuple[List[str], Dict[int, Dict[str, str]]]:
        """
        Anonymize multiple content strings with consistent mappings.
        """
        anonymized_contents = []
        for content in contents:
            result = self.anonymize(content, triggered_keywords)
            anonymized_contents.append(result.anonymized_content)
        return anonymized_contents, self.transformation_map


def anonymize_content(
    content: str,
    triggered_keywords: List[str],
    existing_map: Optional[Dict[int, Dict[str, str]]] = None
) -> AnonymizationResult:
    """
    Anonymize sensitive content with random values.
    
    Args:
        content: The content to anonymize
        triggered_keywords: List of detected sensitive keywords 
        existing_map: Optional existing transformation map to extend
        
    Returns:
        AnonymizationResult with anonymized content and transformation_map
    """
    anonymizer = Anonymizer()
    
    if existing_map:
        anonymizer._transformation_map = existing_map.copy()
        anonymizer._counter = max(existing_map.keys()) if existing_map else 0
        for entry in existing_map.values():
            anonymizer._anonymized_cache[entry["original"]] = entry["anonymous"]
    
    return anonymizer.anonymize(content, triggered_keywords)


def anonymize_contents(
    contents: List[str],
    triggered_keywords: List[str],
    existing_map: Optional[Dict[int, Dict[str, str]]] = None
) -> tuple[List[str], Dict[int, Dict[str, str]]]:
    """
    Anonymize multiple content strings with consistent mappings.
    """
    anonymizer = Anonymizer()
    
    if existing_map:
        anonymizer._transformation_map = existing_map.copy()
        anonymizer._counter = max(existing_map.keys()) if existing_map else 0
        for entry in existing_map.values():
            anonymizer._anonymized_cache[entry["original"]] = entry["anonymous"]
    
    return anonymizer.anonymize_multiple(contents, triggered_keywords)


def deanonymize_content(
    content: str,
    transformation_map: Dict[int, Dict[str, str]]
) -> str:
    """
    De-anonymize content by replacing anonymous values with originals.
    
    This reverses the anonymization process, converting anonymous
    placeholders back to their original values.
    
    Args:
        content: Content with anonymous placeholders
        transformation_map: Map from anonymize_content() containing original values
        
    Returns:
        Content with original values restored
        
    Example:
        >>> transformation_map = {
        ...     1: {"original": "john@example.com", "anonymous": "anon001@anonymized.local"}
        ... }
        >>> deanonymize_content(
        ...     "Your email is anon001@anonymized.local",
        ...     transformation_map
        ... )
        "Your email is john@example.com"
    """
    if not transformation_map:
        return content
    
    result = content
    
    # Sort by anonymous value length (longest first) to avoid partial replacements
    sorted_entries = sorted(
        transformation_map.values(),
        key=lambda x: len(x.get("anonymous", "")),
        reverse=True
    )
    
    for entry in sorted_entries:
        anonymous = entry.get("anonymous", "")
        original = entry.get("original", "")
        
        if anonymous and original:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(anonymous), re.IGNORECASE)
            result = pattern.sub(original, result)
    
    return result


def deanonymize_contents(
    contents: List[str],
    transformation_map: Dict[int, Dict[str, str]]
) -> List[str]:
    """
    De-anonymize multiple content strings.
    
    Args:
        contents: List of content strings with anonymous placeholders
        transformation_map: Map containing original values
        
    Returns:
        List of contents with original values restored
    """
    return [deanonymize_content(content, transformation_map) for content in contents]


# Convenience function for ActionBase integration
def create_anonymization_result(
    original_content: List[str],
    triggered_keywords: List[str],
    use_random_format: bool = False
) -> tuple[List[str], Dict[int, Dict[str, str]]]:
    """
    Create anonymization result for ActionBase integration.
    
    This is designed to be called from ActionBase subclasses to
    provide a standardized anonymization interface.
    
    Args:
        original_content: List of original content strings
        triggered_keywords: List of triggered keywords from rule
        use_random_format: Use random character replacement
        
    Returns:
        Tuple of (anonymized_contents, transformation_map)
    """
    return anonymize_contents(original_content, triggered_keywords, use_random_format)
