from __future__ import annotations
from typing import List, Dict, Any, Optional
import re
import time
from pydantic import Field

from .base import ChunkingStrategy, ChunkingConfig, ChunkingMode
from .recursive import RecursiveCharacterChunkingStrategy, RecursiveChunkingConfig
from ..schemas.data_models import Document, Chunk
from ..utils.error_wrapper import upsonic_error_handler


class ContextualChunkingConfig(ChunkingConfig):
    """Configuration for contextual overlap chunking."""
    
    context_window_size: int = Field(300, description="Size of context window to preserve")
    semantic_overlap_ratio: float = Field(0.3, description="Ratio of semantic overlap between chunks")
    
    sentence_boundaries: bool = Field(True, description="Respect sentence boundaries")
    paragraph_boundaries: bool = Field(True, description="Respect paragraph boundaries")
    section_boundaries: bool = Field(True, description="Respect section boundaries (headers)")
    
    keyword_preservation: bool = Field(True, description="Preserve important keywords across chunks")
    topic_coherence: bool = Field(True, description="Maintain topic coherence in overlaps")
    
    min_meaningful_chunk_size: int = Field(100, description="Minimum size for a meaningful chunk")
    max_context_extension: int = Field(500, description="Maximum extension for context preservation")
    
    adaptive_overlap: bool = Field(True, description="Adapt overlap size based on content complexity")
    context_ranking: bool = Field(True, description="Rank and select best context for overlaps")


class ContextualOverlapChunkingStrategy(ChunkingStrategy):
    """
    Contextual overlap chunking strategy.
    
    This strategy creates chunks with intelligent overlaps that preserve semantic context,
    ensuring each chunk is meaningful when retrieved independently. It analyzes content
    structure and creates overlaps that include the most relevant contextual information.
    
    Features:
    - Semantic-aware overlap creation
    - Context preservation across chunk boundaries
    - Adaptive overlap sizing based on content complexity
    - Keyword and topic coherence maintenance
    - Multiple boundary type respect (sentence, paragraph, section)
    
    """
    
    def __init__(self, config: Optional[ContextualChunkingConfig] = None):
        """Initialize contextual overlap chunking strategy."""
        if config is None:
            config = ContextualChunkingConfig()
        
        super().__init__(config)
        
        section_separators = [
            "\n\n## ",
            "\n\n### ",
            "\n##",
            "\n###",
        ]
        
        standard_separators = ["\n\n", "\n", ". "]

        base_config = RecursiveChunkingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=0,
            separators=section_separators + standard_separators,
            preserve_sentences=self.config.sentence_boundaries,
            preserve_paragraphs=self.config.paragraph_boundaries
        )
        self._base_splitter = RecursiveCharacterChunkingStrategy(config=base_config)
        
        self._context_cache: Dict[str, Any] = {}
        
        self._keyword_patterns = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b',
            r'\b[A-Z]{2,}\b',
            r'\b\w+_\w+\b',
            r'\b\w+-\w+\b',
        ]
    
    @upsonic_error_handler(max_retries=1, show_error_details=True)
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Create chunks with contextual overlaps.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks with contextual overlaps
        """
        start_time = time.time()
        
        if not document.content.strip():
            return []
        
        base_chunks = self._create_base_chunks(document)
        
        if len(base_chunks) <= 1:
            return base_chunks
        
        content_analysis = self._analyze_content_structure(document.content)
        
        contextual_chunks = self._create_contextual_overlaps(
            base_chunks, document, content_analysis
        )
        
        final_chunks = self._optimize_chunks(contextual_chunks, document)
        
        processing_time = (time.time() - start_time) * 1000
        self._update_metrics(final_chunks, processing_time, document)
        
        return final_chunks
    
    def _create_base_chunks(self, document: Document) -> List[Chunk]:
        """Create initial chunks without contextual overlap."""
        return self._base_splitter.chunk(document)
    
    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure for better context preservation."""
        analysis = {
            'sentences': self._find_sentence_boundaries(content),
            'paragraphs': self._find_paragraph_boundaries(content),
            'sections': self._find_section_boundaries(content),
            'keywords': self._extract_keywords(content),
            'topics': self._identify_topic_shifts(content),
            'complexity_score': self._calculate_complexity(content)
        }
        
        return analysis
    
    def _find_sentence_boundaries(self, content: str) -> List[int]:
        """Find sentence boundary positions in content."""
        sentence_pattern = r'[.!?]+\s+'
        boundaries = []
        
        for match in re.finditer(sentence_pattern, content):
            boundaries.append(match.end())
        
        return boundaries
    
    def _find_paragraph_boundaries(self, content: str) -> List[int]:
        """Find paragraph boundary positions."""
        paragraph_pattern = r'\n\s*\n'
        boundaries = []
        
        for match in re.finditer(paragraph_pattern, content):
            boundaries.append(match.end())
        
        return boundaries
    
    def _find_section_boundaries(self, content: str) -> List[int]:
        """Find section boundaries (headers, etc.)."""
        section_patterns = [
            r'\n#{1,6}\s+',
            r'\n\d+\.\s+',
            r'\n[A-Z][^.]*:\s*\n',
        ]
        
        boundaries = []
        for pattern in section_patterns:
            for match in re.finditer(pattern, content):
                boundaries.append(match.start())
        
        return sorted(set(boundaries))
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from content."""
        keywords = set()
        
        for pattern in self._keyword_patterns:
            matches = re.findall(pattern, content)
            keywords.update(matches)
        
        word_freq = {}
        words = re.findall(r'\b\w+\b', content.lower())
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        keywords.update([word for word, freq in frequent_words if freq > 2])
        
        return list(keywords)
    
    def _identify_topic_shifts(self, content: str) -> List[int]:
        """Identify potential topic shift positions."""
        paragraphs = content.split('\n\n')
        topic_shifts = []
        current_pos = 0
        
        for i, paragraph in enumerate(paragraphs[:-1]):
            current_pos += len(paragraph) + 2
            
            next_paragraph = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
            
            if (paragraph.endswith(('.', '!', '?')) and 
                next_paragraph and 
                (next_paragraph[0].isupper() or 
                 any(word in next_paragraph.lower()[:50] for word in ['however', 'meanwhile', 'furthermore', 'in contrast']))):
                topic_shifts.append(current_pos)
        
        return topic_shifts
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate content complexity score."""
        sentences = re.split(r'[.!?]+', content)
        if not sentences:
            return 0.0
        
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        unique_words = len(set(re.findall(r'\b\w+\b', content.lower())))
        total_words = len(re.findall(r'\b\w+\b', content))
        
        vocabulary_richness = unique_words / max(total_words, 1)
        
        complexity = (
            min(avg_sentence_length / 20, 1.0) * 0.5 +
            vocabulary_richness * 0.5
        )
        
        return complexity
    
    def _create_contextual_overlaps(
        self, 
        base_chunks: List[Chunk], 
        document: Document, 
        content_analysis: Dict[str, Any]
    ) -> List[Chunk]:
        """Create chunks with contextual overlaps."""
        if len(base_chunks) <= 1:
            return base_chunks
        
        contextual_chunks = []
        
        for i, chunk in enumerate(base_chunks):
            context_size = self._calculate_adaptive_context_size(
                chunk, content_analysis['complexity_score']
            )
            
            if i > 0:
                prev_chunk = base_chunks[i - 1]
                overlap_text = self._create_optimal_overlap(
                    prev_chunk, chunk, content_analysis, context_size
                )
                
                if overlap_text:
                    enhanced_content = overlap_text + " " + chunk.text_content
                else:
                    enhanced_content = chunk.text_content
            else:
                enhanced_content = chunk.text_content
                overlap_text = ""
            original_start_pos = chunk.metadata.get('start_position', 0)
            new_start_pos = original_start_pos - (len(overlap_text) + 1) if overlap_text else original_start_pos
            enhanced_chunk = self._create_chunk(
                text_content=enhanced_content,
                document=document,
                chunk_index=i,
                total_chunks=len(base_chunks),
                start_pos=new_start_pos,
                end_pos=new_start_pos + len(enhanced_content)
            )
            
            enhanced_chunk.metadata.update({
                'has_contextual_overlap': i > 0,
                'context_size': context_size,
                'complexity_score': self._calculate_complexity(enhanced_content)
            })
            
            contextual_chunks.append(enhanced_chunk)
        
        return contextual_chunks
    
    def _calculate_adaptive_context_size(self, chunk: Chunk, complexity_score: float) -> int:
        """Calculate adaptive context window size based on content complexity."""
        base_size = self.config.context_window_size
        
        if not self.config.adaptive_overlap:
            return base_size
        
        chunk_complexity_score = self._calculate_complexity(chunk.text_content)
        complexity_multiplier = 0.5 + chunk_complexity_score
        adaptive_size = int(base_size * complexity_multiplier)
        
        min_size = max(50, base_size // 3)
        max_size = min(self.config.max_context_extension, base_size * 2)
        
        return max(min_size, min(adaptive_size, max_size))
    
    def _create_optimal_overlap(
        self,
        prev_chunk: Chunk,
        current_chunk: Chunk,
        content_analysis: Dict[str, Any],
        context_size: int
    ) -> str:
        """Create optimal overlap between two chunks."""
        prev_text = prev_chunk.text_content
        
        overlap_size = int(len(prev_text) * self.config.semantic_overlap_ratio)
        overlap_size = min(overlap_size, context_size)
        
        if overlap_size <= 0:
            return ""
        
        current_text = current_chunk.text_content
        if current_text.strip().startswith("##"):
            return ""
        
        if self.config.sentence_boundaries:
            sentence_pattern = r'[.!?]+\s*'
            sentences = re.split(sentence_pattern, prev_text)
            separators = re.findall(sentence_pattern, prev_text)
            
            overlap_text = ""
            for i in range(len(sentences) - 1, -1, -1):
                sentence = sentences[i].strip()
                if not sentence:
                    continue
                    
                separator = separators[i] if i < len(separators) else ""
                test_overlap = sentence + separator + overlap_text
                
                if len(test_overlap) <= overlap_size * 1.2:
                    overlap_text = test_overlap
                else:
                    break
            
            if overlap_text.strip():
                return overlap_text.strip()
        
        simple_overlap = prev_text[-overlap_size:]
        words = simple_overlap.split()
        if len(words) > 1:
            return " ".join(words[1:])
        else:
            return simple_overlap.strip()
    
    def _find_overlap_candidates(
        self, 
        text: str, 
        target_size: int, 
        content_analysis: Dict[str, Any]
    ) -> List[str]:
        """Find potential overlap candidates respecting boundaries."""
        candidates = []
        
        simple_overlap = text[-target_size:].strip()
        if simple_overlap:
            candidates.append(simple_overlap)
        
        if self.config.sentence_boundaries:
            sentences = re.split(r'[.!?]+\s*', text)
            sentence_overlap = ""
            for sentence in reversed(sentences):
                test_overlap = sentence + ". " + sentence_overlap
                if len(test_overlap) <= target_size * 1.2:
                    sentence_overlap = test_overlap
                else:
                    break
            
            if sentence_overlap.strip():
                candidates.append(sentence_overlap.strip())
        
        if self.config.keyword_preservation:
            keywords = content_analysis.get('keywords', [])
            keyword_overlap = self._create_keyword_preserving_overlap(
                text, target_size, keywords
            )
            if keyword_overlap:
                candidates.append(keyword_overlap)
        
        return candidates
    
    def _create_keyword_preserving_overlap(
        self, 
        text: str, 
        target_size: int, 
        keywords: List[str]
    ) -> str:
        """Create overlap that preserves important keywords."""
        text_end = text[-target_size * 2:]
        
        keyword_positions = []
        for keyword in keywords:
            for match in re.finditer(re.escape(keyword), text_end, re.IGNORECASE):
                keyword_positions.append((match.start(), match.end(), keyword))
        
        if not keyword_positions:
            return ""
        
        keyword_positions.sort()
        
        for start_pos, end_pos, keyword in keyword_positions:
            overlap = text_end[start_pos:start_pos + target_size]
            if len(overlap) >= target_size * 0.7:
                return overlap.strip()
        
        return ""
    
    def _select_best_overlap(self, candidates: List[str], keywords: List[str]) -> str:
        """Select the best overlap from candidates."""
        if not candidates:
            return ""
        
        if not self.config.context_ranking:
            return candidates[0]
        
        scored_candidates = []
        
        for candidate in candidates:
            score = self._score_overlap_candidate(candidate, keywords)
            scored_candidates.append((score, candidate))
        
        scored_candidates.sort(reverse=True)
        return scored_candidates[0][1]
    
    def _score_overlap_candidate(self, candidate: str, keywords: List[str]) -> float:
        """Score an overlap candidate based on various criteria."""
        score = 0.0
        
        keyword_count = sum(
            1 for keyword in keywords 
            if keyword.lower() in candidate.lower()
        )
        keyword_score = keyword_count / max(len(keywords), 1)
        score += keyword_score * 0.4
        
        if candidate.strip().endswith(('.', '!', '?')):
            score += 0.3
        
        ideal_length = self.config.context_window_size
        length_ratio = len(candidate) / ideal_length
        length_score = 1.0 - abs(1.0 - length_ratio)
        score += length_score * 0.3
        
        return score
    
    def _optimize_chunks(self, chunks: List[Chunk], document: Document) -> List[Chunk]:
        """Final optimization and validation of chunks."""
        optimized_chunks = []
        
        if len(chunks) <= 3 and len(document.content) <= self.config.chunk_size * 3:
            merged_content = " ".join(chunk.text_content for chunk in chunks)
            merged_chunk = self._create_chunk(
                text_content=merged_content,
                document=document,
                chunk_index=0,
                total_chunks=1,
                start_pos=0,
                end_pos=len(merged_content)
            )
            merged_chunk.metadata['merged_all_chunks'] = True
            return [merged_chunk]
        
        for chunk in chunks:
            should_merge = (
                len(chunk.text_content) < self.config.min_meaningful_chunk_size or
                (len(chunks) <= 3 and len(chunk.text_content) < self.config.chunk_size * 0.5)
            )
            
            if should_merge:
                if optimized_chunks:
                    prev_chunk = optimized_chunks[-1]
                    merged_content = prev_chunk.text_content + " " + chunk.text_content
                    
                    if len(merged_content) <= self.config.chunk_size * 1.5:
                        prev_chunk.text_content = merged_content
                        prev_chunk.metadata['merged_small_chunk'] = True
                        continue
            
            if len(chunk.text_content) > self.config.chunk_size * 2.0:
                split_chunks = self._split_oversized_chunk(chunk, document)
                optimized_chunks.extend(split_chunks)
            else:
                optimized_chunks.append(chunk)
        
        for i, chunk in enumerate(optimized_chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(optimized_chunks)
        
        return optimized_chunks
    
    def _split_oversized_chunk(self, chunk: Chunk, document: Document) -> List[Chunk]:
        """Split an oversized chunk while preserving context."""
        content = chunk.text_content
        target_size = self.config.chunk_size
        
        sentences = re.split(r'([.!?]+\s*)', content)
        
        sub_chunks = []
        current_content = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            separator = sentences[i + 1] if i + 1 < len(sentences) else ""
            
            test_content = current_content + sentence + separator
            
            if len(test_content) > target_size and current_content:
                sub_chunk = self._create_chunk(
                    text_content=current_content.strip(),
                    document=document,
                    chunk_index=len(sub_chunks),
                    total_chunks=1,
                    start_pos=chunk.metadata.get('start_position', 0),
                    end_pos=chunk.metadata.get('start_position', 0) + len(current_content)
                )
                sub_chunk.metadata['split_from_oversized'] = True
                sub_chunks.append(sub_chunk)
                
                current_content = sentence + separator
            else:
                current_content = test_content
        
        if current_content.strip():
            sub_chunk = self._create_chunk(
                text_content=current_content.strip(),
                document=document,
                chunk_index=len(sub_chunks),
                total_chunks=1,
                start_pos=chunk.metadata.get('start_position', 0),
                end_pos=chunk.metadata.get('end_position', len(current_content))
            )
            sub_chunk.metadata['split_from_oversized'] = True
            sub_chunks.append(sub_chunk)
        
        return sub_chunks if sub_chunks else [chunk]
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about contextual overlap performance."""
        return {
            "context_window_size": self.config.context_window_size,
            "semantic_overlap_ratio": self.config.semantic_overlap_ratio,
            "adaptive_overlap_enabled": self.config.adaptive_overlap,
            "keyword_preservation_enabled": self.config.keyword_preservation,
            "cache_size": len(self._context_cache),
            "boundary_preservation": {
                "sentences": self.config.sentence_boundaries,
                "paragraphs": self.config.paragraph_boundaries,
                "sections": self.config.section_boundaries
            }
        }
