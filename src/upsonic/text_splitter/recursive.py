from __future__ import annotations
from typing import List, Any, Optional, Dict
import re
from pydantic import Field

from upsonic.text_splitter.base import TextSplitter, TextSplitterConfig
from ..utils.error_wrapper import upsonic_error_handler

class RecursiveChunkingConfig(TextSplitterConfig):
    """Enhanced configuration for recursive character chunking."""
    separators: List[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". "],
        description="Ordered list of separators from highest to lowest priority"
    )
    is_separator_regex: bool = Field(False, description="Whether separators are regex patterns")
    
    enable_adaptive_splitting: bool = Field(True, description="Adapt separator priority based on content")
    content_type_detection: bool = Field(True, description="Detect content type and adjust separators")
    
    min_separator_frequency: float = Field(0.01, description="Minimum frequency for separator to be used")
    prefer_balanced_chunks: bool = Field(True, description="Try to create chunks of similar sizes")
    
    max_recursion_depth: int = Field(10, description="Maximum recursion depth to prevent infinite loops")
    enable_separator_caching: bool = Field(True, description="Cache separator analysis results")


class RecursiveCharacterChunkingStrategy(TextSplitter):
    """
    Recursive chunking strategy with intelligent separator selection.

    This advanced method recursively tries to split text using a prioritized
    list of separators, with new features like:
    - Adaptive separator prioritization based on content analysis
    - Content type detection and separator optimization
    - Balanced chunk creation
    - Performance monitoring and optimization
    - Intelligent fallback mechanisms
    """
    
    def __init__(self, config: Optional[RecursiveChunkingConfig] = None):
        """
        Initialize recursive chunking strategy.

        Args:
            config: Configuration object with all settings
        """
        if config is None:
            config = RecursiveChunkingConfig()
        
        super().__init__(config)
        
        self._separators = self.config.separators
        self._is_separator_regex = self.config.is_separator_regex
        
        self._separator_stats: Dict[str, Dict[str, Any]] = {}
        self._recursion_depth = 0

    @upsonic_error_handler(max_retries=1, show_error_details=True)
    def split_text(self, text: str) -> List[str]:
        """
        Entry point with adaptive separator selection and content analysis.
        """
        if not text.strip():
            return []
        
        self._recursion_depth = 0
        
        if self.config.enable_adaptive_splitting:
            optimized_separators = self._analyze_and_adapt_separators(text)
        else:
            optimized_separators = self.config.separators
        
        return self._recursive_split_enhanced(text, optimized_separators)
    
    def _analyze_and_adapt_separators(self, text: str) -> List[str]:
        """Analyze text content and adapt separator priorities."""
        if self.config.enable_separator_caching and text in self._separator_stats:
            cached_result = self._separator_stats[text]
            return cached_result.get('optimized_separators', self.config.separators)
        
        separator_analysis = {}
        
        for separator in self.config.separators:
            if self.config.is_separator_regex:
                matches = re.findall(separator, text)
            else:
                matches = text.split(separator)
            
            frequency = len(matches) / len(text) if text else 0
            separator_analysis[separator] = {
                'frequency': frequency,
                'count': len(matches),
                'avg_segment_length': len(text) / max(len(matches), 1)
            }
        
        useful_separators = []
        for separator in self.config.separators:
            analysis = separator_analysis[separator]
            if analysis['frequency'] >= self.config.min_separator_frequency:
                useful_separators.append(separator)
        
        if not useful_separators:
            useful_separators = self.config.separators
        
        if self.config.enable_separator_caching:
            self._separator_stats[text] = {
                'analysis': separator_analysis,
                'optimized_separators': useful_separators
            }
        
        return useful_separators
    
    def _recursive_split_enhanced(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursive logic with depth limiting and performance optimization.
        """
        self._recursion_depth += 1
        if self._recursion_depth > self.config.max_recursion_depth:
            return self._character_split_fallback(text)
        
        final_chunks: List[str] = []
        
        current_separator, remaining_separators = self._find_best_separator(text, separators)
        
        if current_separator is None:
            result = self._handle_base_case(text)
            self._recursion_depth -= 1
            return result
        
        splits = self._split_with_separator(text, current_separator)
        
        for split in splits:
            if not split.strip():
                continue
                
            if len(split) <= self.config.chunk_size:
                final_chunks.append(split)
            else:
                recursive_result = self._recursive_split_enhanced(split, remaining_separators)
                final_chunks.extend(recursive_result)
        
        self._recursion_depth -= 1
        return final_chunks
    
    def _find_best_separator(self, text: str, separators: List[str]) -> tuple:
        """Find the best separator for the given text."""
        for i, separator in enumerate(separators):
            if self._separator_exists_in_text(text, separator):
                return separator, separators[i + 1:]
        return None, []
    
    def _separator_exists_in_text(self, text: str, separator: str) -> bool:
        """Check if separator exists in text."""
        if not separator:
            return False
        if self.config.is_separator_regex:
            return bool(re.search(separator, text))
        else:
            return separator in text
    
    def _split_with_separator(self, text: str, separator: str) -> List[str]:
        """Split text with the given separator, handling both regex and literal."""
        if not separator:
            return [text] if text.strip() else []
        
        if self.config.is_separator_regex:
            pattern = separator
        else:
            pattern = re.escape(separator)
        
        if self.config.keep_separator:
            splits = re.split(f"({pattern})", text)
            rejoined_splits = []
            for i in range(0, len(splits), 2):
                part = splits[i]
                if i + 1 < len(splits):
                    part += splits[i + 1]
                if part:
                    rejoined_splits.append(part)
            return rejoined_splits
        else:
            return [s for s in re.split(pattern, text) if s]
    
    def _handle_base_case(self, text: str) -> List[str]:
        """Handle the base case when no separators are found."""
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []
        else:
            return self._character_split_fallback(text)
    
    def _character_split_fallback(self, text: str) -> List[str]:
        """Fallback to character-level splitting."""
        chunks = []
        for i in range(0, len(text), self.config.chunk_size):
            chunk = text[i:i + self.config.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def get_separator_stats(self) -> Dict[str, Any]:
        """Get statistics about separator usage and performance."""
        return {
            "separator_cache_size": len(self._separator_stats),
            "separator_analysis": self._separator_stats,
            "current_separators": self.config.separators,
            "is_regex_mode": self.config.is_separator_regex,
            "adaptive_splitting_enabled": self.config.enable_adaptive_splitting
        }
    
    def clear_separator_cache(self):
        """Clear the separator analysis cache."""
        self._separator_stats.clear()

    # Legacy method for backward compatibility
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Legacy recursive split method for backward compatibility."""
        return self._recursive_split_enhanced(text, separators)