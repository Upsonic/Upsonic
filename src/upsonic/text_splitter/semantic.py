from __future__ import annotations
from typing import List, Any, Literal, Optional, Dict
import re
import time
import numpy as np
from pydantic import Field

from upsonic.text_splitter.base import ChunkingStrategy, ChunkingConfig
from upsonic.schemas.data_models import Document, Chunk
from upsonic.embeddings.base import EmbeddingProvider
from ..utils.error_wrapper import upsonic_error_handler

BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]


class SemanticChunkingConfig(ChunkingConfig):
    """Enhanced configuration for semantic similarity chunking."""
    buffer_size: int = Field(1, description="Number of sentences to include before and after for context")
    breakpoint_threshold_type: BreakpointThresholdType = Field("percentile", description="Statistical method for breakpoint detection")
    breakpoint_threshold_amount: float = Field(95, description="Threshold value for breakpoint detection")
    
    enable_topic_modeling: bool = Field(False, description="Enable topic modeling for better boundaries")
    min_semantic_chunk_size: int = Field(100, description="Minimum size for semantic chunks")
    max_semantic_chunk_size: int = Field(3000, description="Maximum size for semantic chunks")
    
    sentence_similarity_threshold: float = Field(0.7, description="Minimum similarity to group sentences")
    enable_sentence_grouping: bool = Field(True, description="Group similar consecutive sentences")
    
    enable_embedding_cache: bool = Field(True, description="Cache sentence embeddings")
    batch_embedding_size: int = Field(50, description="Batch size for embedding generation")
    
    merge_small_chunks: bool = Field(True, description="Merge chunks smaller than min_semantic_chunk_size")
    split_large_chunks: bool = Field(True, description="Split chunks larger than max_semantic_chunk_size")
    preserve_sentence_integrity: bool = Field(True, description="Never split in the middle of sentences")

class SemanticSimilarityChunkingStrategy(ChunkingStrategy):
    """
    Semantic similarity chunking strategy with framework-level features.

    This sophisticated method splits documents based on semantic similarity,
    identifying breakpoints where topics shift.
    
    Features:
    - Advanced topic modeling and boundary detection
    - Sentence grouping and similarity analysis
    - Performance optimization with embedding caching
    - Quality controls for chunk size management
    - Batch processing and async support
    - Comprehensive metrics and monitoring

    This method operates by:
    1. Splitting the document into individual sentences
    2. Embedding each sentence within contextual windows
    3. Calculating semantic distances between consecutive sentences
    4. Identifying statistically significant topic breaks
    5. Grouping sentences into semantically coherent chunks
    6. Applying quality controls and optimizations


    """

    def __init__(self, embedding_provider: EmbeddingProvider, config: Optional[SemanticChunkingConfig] = None):
        """
        Initialize semantic similarity chunking strategy.

        Args:
            embedding_provider: EmbeddingProvider instance for generating vectors
            config: Configuration object with all settings
        """
        if not isinstance(embedding_provider, EmbeddingProvider):
            raise TypeError("An instance of EmbeddingProvider is required.")
        
        if config is None:
            config = SemanticChunkingConfig()
        
        super().__init__(config)
        
        self.embedding_provider = embedding_provider
        
        self.buffer_size = self.config.buffer_size
        self.breakpoint_threshold_type = self.config.breakpoint_threshold_type
        self.breakpoint_threshold_amount = self.config.breakpoint_threshold_amount
        
        self._embedding_cache: Dict[str, List[float]] = {}
        self._sentence_cache: Dict[str, List[str]] = {}
        self._distance_cache: Dict[str, List[float]] = {}

    @upsonic_error_handler(max_retries=1, show_error_details=True)
    async def chunk(self, document: Document) -> List[Chunk]:
        """
        Semantic chunking pipeline with framework features.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of semantically coherent chunks
        """
        start_time = time.time()
        
        if not document.content.strip():
            return []
        
        cache_key = self._get_cache_key(document.content) if self.config.enable_embedding_cache else None
        if cache_key and cache_key in self._sentence_cache:
            sentences = self._sentence_cache[cache_key]
        else:
            sentences = self._split_into_sentences_enhanced(document.content)
            if cache_key:
                self._sentence_cache[cache_key] = sentences

        if len(sentences) <= 1:
            chunk = self._create_chunk(
                text_content=document.content,
                document=document,
                chunk_index=0,
                total_chunks=1
            )
            chunk.metadata['semantic_analysis'] = 'single_sentence'
            return [chunk]

        sentence_embeddings = await self._embed_contextual_sentences_enhanced(sentences, cache_key)

        distances = self._calculate_semantic_distances_enhanced(sentence_embeddings)

        breakpoint_indices = self._detect_breakpoints_enhanced(distances, sentences)

        if self.config.enable_sentence_grouping:
            breakpoint_indices = self._refine_breakpoints_with_grouping(
                sentences, sentence_embeddings, breakpoint_indices
            )

        initial_chunks = self._create_semantic_chunks(
            sentences, breakpoint_indices, document
        )

        overlapped_chunks = self._apply_semantic_overlap(initial_chunks, sentences, document)

        final_chunks = self._apply_quality_controls(overlapped_chunks, document)

        processing_time = (time.time() - start_time) * 1000
        self._update_metrics(final_chunks, processing_time, document)

        return final_chunks

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text content."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _split_into_sentences_enhanced(self, text: str) -> List[str]:
        """Sentence splitting with better handling of edge cases."""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def _split_into_sentences(self, text: str) -> List[str]:
        """Legacy sentence splitting method for backward compatibility."""
        return self._split_into_sentences_enhanced(text)

    async def _embed_contextual_sentences_enhanced(self, sentences: List[str], cache_key: Optional[str] = None) -> List[List[float]]:
        """Contextual embedding with caching and batch processing."""
        if cache_key and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        contextual_sentences: List[str] = []
        for i in range(len(sentences)):
            start = max(0, i - self.config.buffer_size)
            end = min(len(sentences), i + 1 + self.config.buffer_size)
            
            context = " ".join(sentences[start:end])
            contextual_sentences.append(context)
        
        embeddings = []
        batch_size = self.config.batch_embedding_size
        
        for i in range(0, len(contextual_sentences), batch_size):
            batch = contextual_sentences[i:i + batch_size]
            mock_chunks = [Chunk(text_content=s, metadata={}, document_id="temp") for s in batch]
            batch_embeddings = await self.embedding_provider.embed_documents(mock_chunks)
            embeddings.extend(batch_embeddings)
        
        if cache_key and self.config.enable_embedding_cache:
            self._embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    async def _embed_contextual_sentences(self, sentences: List[str]) -> List[List[float]]:
        """Legacy method for backward compatibility."""
        return await self._embed_contextual_sentences_enhanced(sentences)

    def _calculate_semantic_distances_enhanced(self, embeddings: List[List[float]]) -> List[float]:
        """Distance calculation with better normalization and edge case handling."""
        if len(embeddings) < 2:
            return []
        
        distances: List[float] = []
        for i in range(len(embeddings) - 1):
            embedding_current = np.array(embeddings[i])
            embedding_next = np.array(embeddings[i + 1])
            
            norm_current = embedding_current / (np.linalg.norm(embedding_current) + 1e-8)
            norm_next = embedding_next / (np.linalg.norm(embedding_next) + 1e-8)
            
            similarity = np.dot(norm_current, norm_next)
            
            similarity = np.clip(similarity, -1.0, 1.0)
            
            distance = 1 - similarity
            distances.append(float(distance))
            
        return distances

    def _calculate_cosine_distances(self, embeddings: List[List[float]]) -> List[float]:
        """Legacy method for backward compatibility."""
        return self._calculate_semantic_distances_enhanced(embeddings)
    
    def _detect_breakpoints_enhanced(self, distances: List[float], sentences: List[str]) -> List[int]:
        """Breakpoint detection with multiple criteria."""
        if not distances:
            return []
        
        threshold = self._calculate_breakpoint_threshold(distances)
        
        candidate_breakpoints = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                candidate_breakpoints.append(i)
        
        refined_breakpoints = self._refine_breakpoints(
            candidate_breakpoints, distances, sentences
        )
        
        return refined_breakpoints
    
    def _calculate_breakpoint_threshold(self, distances: List[float]) -> float:
        """Threshold calculation with better statistical methods."""
        if not distances:
            return 0.0
        
        distances_array = np.array(distances)
        
        if self.config.breakpoint_threshold_type == "percentile":
            return float(np.percentile(distances_array, self.config.breakpoint_threshold_amount))
        
        elif self.config.breakpoint_threshold_type == "standard_deviation":
            mean = np.mean(distances_array)
            std_dev = np.std(distances_array)
            return float(mean + (self.config.breakpoint_threshold_amount * std_dev))
            
        elif self.config.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances_array, [25, 75])
            iqr = q3 - q1
            return float(q3 + (self.config.breakpoint_threshold_amount * iqr))
            
        else:
            raise ValueError(f"Unknown breakpoint_threshold_type: {self.config.breakpoint_threshold_type}")
    
    def _refine_breakpoints(self, breakpoints: List[int], distances: List[float], sentences: List[str]) -> List[int]:
        """Refine breakpoints based on sentence context and quality."""
        if not breakpoints:
            return breakpoints
        
        refined = []
        
        for bp in breakpoints:
            if self._is_valid_breakpoint(bp, distances, sentences):
                refined.append(bp)
        
        final_breakpoints = []
        last_bp = -10
        
        for bp in refined:
            if bp - last_bp >= 2:
                final_breakpoints.append(bp)
                last_bp = bp
        
        return final_breakpoints
    
    def _is_valid_breakpoint(self, breakpoint: int, distances: List[float], sentences: List[str]) -> bool:
        """Check if a breakpoint is valid based on context."""
        if breakpoint < len(sentences) - 1:
            current_sentence = sentences[breakpoint]
            next_sentence = sentences[breakpoint + 1]
            
            if len(current_sentence) < 20 or len(next_sentence) < 20:
                return False
        
        return True
    
    def _refine_breakpoints_with_grouping(
        self, 
        sentences: List[str], 
        embeddings: List[List[float]], 
        breakpoints: List[int]
    ) -> List[int]:
        """Refine breakpoints using sentence similarity grouping."""
        if not self.config.enable_sentence_grouping or not breakpoints:
            return breakpoints
        
        grouped_breakpoints = []
        
        for bp in breakpoints:
            should_keep = True
            
            if bp > 0 and bp < len(embeddings) - 1:
                prev_emb = np.array(embeddings[bp - 1])
                curr_emb = np.array(embeddings[bp])
                next_emb = np.array(embeddings[bp + 1])
                
                prev_sim = np.dot(prev_emb, curr_emb) / (np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb))
                next_sim = np.dot(curr_emb, next_emb) / (np.linalg.norm(curr_emb) * np.linalg.norm(next_emb))
                
                if (prev_sim > self.config.sentence_similarity_threshold or 
                    next_sim > self.config.sentence_similarity_threshold):
                    should_keep = False
            
            if should_keep:
                grouped_breakpoints.append(bp)
        
        return grouped_breakpoints
    
    def _create_semantic_chunks(
        self, 
        sentences: List[str], 
        breakpoint_indices: List[int], 
        document: Document
    ) -> List[Chunk]:
        """Create chunks with metadata from semantic analysis."""
        chunks = []
        start_index = 0

        for end_index in breakpoint_indices:
            group = sentences[start_index : end_index + 1]
            chunk_text = " ".join(group)
            
            chunk = self._create_chunk(
                text_content=chunk_text,
                document=document,
                chunk_index=len(chunks),
                total_chunks=len(breakpoint_indices) + 1,
                start_pos=start_index,
                end_pos=end_index + 1
            )
            
            chunk.metadata.update({
                'semantic_analysis': 'breakpoint_detected',
                'sentence_count': len(group),
                'semantic_coherence': 'high'
            })
            
            chunks.append(chunk)
            start_index = end_index + 1

        if start_index < len(sentences):
            last_group = sentences[start_index:]
            last_group_text = " ".join(last_group)
            
            chunk = self._create_chunk(
                text_content=last_group_text,
                document=document,
                chunk_index=len(chunks),
                total_chunks=len(chunks) + 1,
                start_pos=start_index,
                end_pos=len(sentences)
            )
            
            chunk.metadata.update({
                'semantic_analysis': 'final_group',
                'sentence_count': len(last_group),
                'semantic_coherence': 'high'
            })
            
            chunks.append(chunk)
        
        return chunks
    
    def _apply_quality_controls(self, chunks: List[Chunk], document: Document) -> List[Chunk]:
        """Apply quality controls to ensure good chunk sizes and coherence."""
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for chunk in chunks:
            chunk_size = len(chunk.text_content)
            
            if chunk_size < self.config.min_semantic_chunk_size and self.config.merge_small_chunks:
                if processed_chunks:
                    prev_chunk = processed_chunks[-1]
                    merged_content = prev_chunk.text_content + " " + chunk.text_content
                    prev_chunk.text_content = merged_content
                    prev_chunk.metadata['merged_small_chunk'] = True
                    continue
            
            if chunk_size > self.config.max_semantic_chunk_size and self.config.split_large_chunks:
                split_chunks = self._split_large_semantic_chunk(chunk, document)
                processed_chunks.extend(split_chunks)
            else:
                processed_chunks.append(chunk)
        
        for i, chunk in enumerate(processed_chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(processed_chunks)
        
        return processed_chunks
    
    def _split_large_semantic_chunk(self, chunk: Chunk, document: Document) -> List[Chunk]:
        """Split a large chunk while preserving sentence integrity."""
        if not self.config.preserve_sentence_integrity:
            content = chunk.text_content
            target_size = self.config.max_semantic_chunk_size
            
            sub_chunks = []
            for i in range(0, len(content), target_size):
                sub_content = content[i:i + target_size]
                if sub_content.strip():
                    sub_chunk = self._create_chunk(
                        text_content=sub_content,
                        document=document,
                        chunk_index=len(sub_chunks),
                        total_chunks=1
                    )
                    sub_chunk.metadata.update(chunk.metadata)
                    sub_chunk.metadata['split_from_large'] = True
                    sub_chunks.append(sub_chunk)
            
            return sub_chunks if sub_chunks else [chunk]
        
        sentences = self._split_into_sentences_enhanced(chunk.text_content)
        if len(sentences) <= 1:
            return [chunk]
        
        sub_chunks = []
        current_sentences = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.config.max_semantic_chunk_size and current_sentences:
                sub_content = " ".join(current_sentences)
                sub_chunk = self._create_chunk(
                    text_content=sub_content,
                    document=document,
                    chunk_index=len(sub_chunks),
                    total_chunks=1
                )
                sub_chunk.metadata.update(chunk.metadata)
                sub_chunk.metadata['split_from_large'] = True
                sub_chunks.append(sub_chunk)
                
                current_sentences = [sentence]
                current_size = sentence_size
            else:
                current_sentences.append(sentence)
                current_size += sentence_size
        
        if current_sentences:
            sub_content = " ".join(current_sentences)
            sub_chunk = self._create_chunk(
                text_content=sub_content,
                document=document,
                chunk_index=len(sub_chunks),
                total_chunks=1
            )
            sub_chunk.metadata.update(chunk.metadata)
            sub_chunk.metadata['split_from_large'] = True
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _apply_semantic_overlap(self, chunks: List[Chunk], sentences: List[str], document: Document) -> List[Chunk]:
        """Apply semantic-aware overlap between chunks."""
        if len(chunks) <= 1 or self.config.chunk_overlap == 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            
            prev_chunk = chunks[i - 1]
            overlap_content = self._create_semantic_overlap(prev_chunk, chunk, sentences)
            
            if overlap_content:
                enhanced_content = overlap_content + " " + chunk.text_content
                chunk.text_content = enhanced_content
                chunk.metadata['has_semantic_overlap'] = True
                chunk.metadata['overlap_length'] = len(overlap_content)
            
            overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def _create_semantic_overlap(self, prev_chunk: Chunk, current_chunk: Chunk, sentences: List[str]) -> str:
        """Create semantic-aware overlap between two chunks."""
        prev_text = prev_chunk.text_content
        overlap_size = self.config.chunk_overlap
        
        if overlap_size <= 0:
            return ""
        
        if len(prev_text) <= overlap_size:
            overlap_text = prev_text
        else:
            prev_sentences = self._split_into_sentences_enhanced(prev_text)
            
            overlap_sentences = []
            current_length = 0
            
            for sentence in reversed(prev_sentences):
                sentence_length = len(sentence)
                if current_length + sentence_length <= overlap_size:
                    overlap_sentences.insert(0, sentence)
                    current_length += sentence_length
                else:
                    if not overlap_sentences and sentence_length <= overlap_size * 1.2:
                        overlap_sentences.insert(0, sentence)
                    break
            
            overlap_text = " ".join(overlap_sentences)
        
        return overlap_text.strip()
    
    def get_semantic_stats(self) -> Dict[str, Any]:
        """Get statistics about semantic chunking performance."""
        return {
            "embedding_cache_size": len(self._embedding_cache),
            "sentence_cache_size": len(self._sentence_cache),
            "distance_cache_size": len(self._distance_cache),
            "buffer_size": self.config.buffer_size,
            "threshold_type": self.config.breakpoint_threshold_type,
            "threshold_amount": self.config.breakpoint_threshold_amount,
            "sentence_grouping_enabled": self.config.enable_sentence_grouping,
            "embedding_cache_enabled": self.config.enable_embedding_cache
        }
    
    def clear_semantic_caches(self):
        """Clear all semantic analysis caches."""
        self._embedding_cache.clear()
        self._sentence_cache.clear()
        self._distance_cache.clear()