from __future__ import annotations
from typing import Any, List, Optional
from pydantic import Field

from upsonic.text_splitter.base import TextSplitter, TextSplitterConfig
from ..utils.error_wrapper import upsonic_error_handler


class CharacterChunkingConfig(TextSplitterConfig):
    """Enhanced configuration for character-based chunking."""
    separator: str = Field("\n\n", description="Character separator to split text by")
    strip_whitespace: bool = Field(True, description="Strip whitespace from splits")
    skip_empty_splits: bool = Field(True, description="Skip empty splits after separation")
    
    multiple_separators: Optional[List[str]] = Field(None, description="Try multiple separators in order")
    fallback_to_length: bool = Field(True, description="Fallback to length-based splitting if no separators found")
    
    enable_separator_optimization: bool = Field(True, description="Optimize separator choice based on content")


class CharacterChunkingStrategy(TextSplitter):
    """
    Character-based chunking strategy with framework-level features.

    This is the most straightforward and often the fastest chunking method. It is
    highly effective for documents that have a clear, consistent structure, such as
    text separated by double newlines.
    
    Features:
    - Multiple separator support with automatic optimization
    - Advanced whitespace handling and cleanup
    - Fallback mechanisms for edge cases
    - Performance monitoring and statistics
    - Content-aware separator selection
    
    """

    def __init__(self, config: Optional[CharacterChunkingConfig] = None):
        """
        Initialize character chunking strategy.

        Args:
            config: Configuration object with all settings
        """
        if config is None:
            config = CharacterChunkingConfig()
        
        super().__init__(config)
        
        self._separator = self.config.separator
        
        self._separator_stats = {}

    @upsonic_error_handler(max_retries=1, show_error_details=True)
    def split_text(self, text: str) -> List[str]:
        """
        Text splitting with multiple separator support and optimization.

        Args:
            text: The full text content of a Document.

        Returns:
            A list of smaller text strings (splits).
        """
        if not text.strip():
            return []
        
        separators_to_try = self._get_separators_to_try(text)
        
        for separator in separators_to_try:
            splits = self._split_with_separator(text, separator)
            
            if self._is_good_split(splits, text):
                self._track_separator_usage(separator, len(splits), len(text))
                return splits
        
        if self.config.fallback_to_length:
            return self._fallback_length_split(text)
        
        return [text] if text.strip() else []
    
    def _get_separators_to_try(self, text: str) -> List[str]:
        """Determine which separators to try based on configuration and content analysis."""
        separators = []
        
        separators.append(self.config.separator)
        
        if self.config.multiple_separators:
            for sep in self.config.multiple_separators:
                if sep not in separators:
                    separators.append(sep)
        
        if self.config.enable_separator_optimization:
            separators = self._optimize_separator_order(separators, text)
        
        return separators
    
    def _split_with_separator(self, text: str, separator: str) -> List[str]:
        """Split text with a specific separator and apply post-processing."""
        splits = text.split(separator)
        
        processed_splits = []
        for split in splits:
            if self.config.strip_whitespace:
                split = split.strip()
            
            if self.config.skip_empty_splits and not split:
                continue
            
            processed_splits.append(split)
        
        return processed_splits
    
    def _is_good_split(self, splits: List[str], original_text: str) -> bool:
        """Evaluate if a split result is good enough to use."""
        if not splits:
            return False
        
        if len(splits) == 1 and splits[0].strip() == original_text.strip():
            return False
        
        if len(splits) < 2:
            return False
        
        avg_split_size = sum(len(s) for s in splits) / len(splits)
        if avg_split_size < 10:
            return False
        
        return True
    
    def _optimize_separator_order(self, separators: List[str], text: str) -> List[str]:
        """Reorder separators based on their effectiveness for this content."""
        separator_scores = []
        
        for separator in separators:
            score = self._score_separator(separator, text)
            separator_scores.append((score, separator))
        
        separator_scores.sort(reverse=True)
        
        return [sep for _, sep in separator_scores]
    
    def _score_separator(self, separator: str, text: str) -> float:
        """Score a separator based on how well it would split the text."""
        if separator not in text:
            return 0.0
        
        count = text.count(separator)
        
        segments = text.split(separator)
        if not segments:
            return 0.0
        
        avg_length = len(text) / len(segments)
        
        count_score = min(count / 10, 1.0)
        length_score = 1.0 - abs(avg_length - 500) / 1000
        length_score = max(0.0, length_score)
        
        history_score = 0.5
        if separator in self._separator_stats:
            stats = self._separator_stats[separator]
            history_score = min(stats.get('success_rate', 0.5), 1.0)
        
        return (count_score * 0.4 + length_score * 0.4 + history_score * 0.2)
    
    def _track_separator_usage(self, separator: str, num_splits: int, text_length: int):
        """Track separator usage for future optimization."""
        if separator not in self._separator_stats:
            self._separator_stats[separator] = {
                'usage_count': 0,
                'total_splits': 0,
                'total_text_length': 0,
                'success_rate': 0.5
            }
        
        stats = self._separator_stats[separator]
        stats['usage_count'] += 1
        stats['total_splits'] += num_splits
        stats['total_text_length'] += text_length
        
        avg_split_size = text_length / max(num_splits, 1)
        success = 1.0 if 50 <= avg_split_size <= 2000 else 0.5
        
        current_rate = stats['success_rate']
        stats['success_rate'] = current_rate * 0.8 + success * 0.2
    
    def _fallback_length_split(self, text: str) -> List[str]:
        """Fallback to simple length-based splitting."""
        splits = []
        chunk_size = self.config.chunk_size
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if self.config.strip_whitespace:
                chunk = chunk.strip()
            if chunk or not self.config.skip_empty_splits:
                splits.append(chunk)
        
        return splits
    
    def get_separator_stats(self) -> dict:
        """Get statistics about separator usage and performance."""
        return {
            "separator_stats": self._separator_stats.copy(),
            "primary_separator": self.config.separator,
            "multiple_separators": self.config.multiple_separators,
            "optimization_enabled": self.config.enable_separator_optimization,
            "fallback_enabled": self.config.fallback_to_length
        }
    
    def reset_separator_stats(self):
        """Reset separator statistics."""
        self._separator_stats.clear()