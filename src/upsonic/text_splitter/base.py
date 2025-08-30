from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
import re
import time
from enum import Enum
from pydantic import BaseModel, Field
import asyncio

from upsonic.schemas.data_models import Document, Chunk
from ..utils.error_wrapper import upsonic_error_handler

class ChunkingMode(str, Enum):
    """Different modes for chunking operations."""
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"


class ChunkingMetrics(BaseModel):
    """Metrics for chunking performance and quality."""
    total_chunks: int = 0
    total_characters: int = 0
    avg_chunk_size: float = 0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    processing_time_ms: float = 0
    strategy_name: str = ""
    document_id: str = ""


class ChunkingConfig(BaseModel):
    """Base configuration for all chunking strategies."""
    chunk_size: int = Field(1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(200, description="Overlap between chunks in characters")
    mode: ChunkingMode = Field(ChunkingMode.STANDARD, description="Chunking mode")
    
    enable_async: bool = Field(False, description="Enable async processing when available")
    batch_size: int = Field(10, description="Batch size for processing multiple documents")
    show_progress: bool = Field(False, description="Show progress for large operations")
    
    min_chunk_size: int = Field(50, description="Minimum chunk size")
    max_chunk_size: Optional[int] = Field(None, description="Maximum chunk size")
    preserve_sentences: bool = Field(True, description="Try to preserve sentence boundaries")
    preserve_paragraphs: bool = Field(True, description="Try to preserve paragraph boundaries")
    
    add_chunk_index: bool = Field(True, description="Add chunk index to metadata")
    add_chunk_count: bool = Field(True, description="Add total chunk count to metadata")
    add_position_info: bool = Field(True, description="Add start/end position info")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata to add")
    
    on_empty_chunk: str = Field("skip", description="What to do with empty chunks: skip, keep, error")
    on_oversized_chunk: str = Field("split", description="What to do with oversized chunks: split, keep, error")
    
    enable_caching: bool = Field(False, description="Enable result caching")
    cache_key_fields: List[str] = Field(default_factory=lambda: ["content"], description="Fields to use for cache key")


class ChunkingStrategy(ABC):
    """
    Abstract contract for all text splitting and chunking algorithms.
    
    This base class provides a comprehensive framework for chunking operations
    with advanced features like performance monitoring, async support, caching, and
    configurable quality settings.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize the chunking strategy with configuration."""
        self.config = config or ChunkingConfig()
        self._metrics = ChunkingMetrics(strategy_name=self.__class__.__name__)
        self._cache: Dict[str, List[Chunk]] = {}
        
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Processes a single Document and splits it into a list of Chunks.
        
        Args:
            document: The Document to be chunked
            
        Returns:
            A list of Chunk objects
        """
        raise NotImplementedError
    
    async def chunk_async(self, document: Document) -> List[Chunk]:
        """
        Async version of chunk method. Default implementation runs sync version in executor.
        Override this method for truly async implementations.
        """
        if not self.config.enable_async:
            return self.chunk(document)
        
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chunk, document)
    
    def chunk_batch(self, documents: List[Document]) -> List[List[Chunk]]:
        """
        Process multiple documents in batches.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunk lists, one for each document
        """
        start_time = time.time()
        results = []
        
        for i, document in enumerate(documents):
            if self.config.show_progress and len(documents) > 1:
                print(f"Processing document {i+1}/{len(documents)}")
                
            chunks = self.chunk(document)
            results.append(chunks)
        
        processing_time = (time.time() - start_time) * 1000
        print(f"Batch processing completed in {processing_time:.2f}ms")
        
        return results
    
    async def chunk_batch_async(self, documents: List[Document]) -> List[List[Chunk]]:
        """Async version of batch chunking."""
        if not self.config.enable_async:
            return self.chunk_batch(documents)
        
        semaphore = asyncio.Semaphore(self.config.batch_size)
        
        async def chunk_with_semaphore(document: Document) -> List[Chunk]:
            async with semaphore:
                return await self.chunk_async(document)
        
        tasks = [chunk_with_semaphore(doc) for doc in documents]
        return await asyncio.gather(*tasks)
    
    def _create_chunk(
        self, 
        text_content: str, 
        document: Document, 
        chunk_index: int = 0, 
        total_chunks: int = 1,
        start_pos: int = 0,
        end_pos: Optional[int] = None
    ) -> Chunk:
        """
        Create a Chunk with metadata based on configuration.
        
        Args:
            text_content: The text content of the chunk
            document: The source document
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            start_pos: Starting position in original document
            end_pos: Ending position in original document
            
        Returns:
            Chunk object
        """
        metadata = document.metadata.copy()
        
        if self.config.add_chunk_index:
            metadata['chunk_index'] = chunk_index
            
        if self.config.add_chunk_count:
            metadata['total_chunks'] = total_chunks
            
        if self.config.add_position_info:
            metadata['start_position'] = start_pos
            metadata['end_position'] = end_pos or start_pos + len(text_content)
            
        metadata['chunking_strategy'] = self.__class__.__name__
        metadata['chunk_size_target'] = self.config.chunk_size
        metadata['chunk_overlap'] = self.config.chunk_overlap
        metadata['chunking_mode'] = self.config.mode.value
        
        metadata.update(self.config.custom_metadata)
        
        return Chunk(
            text_content=text_content,
            metadata=metadata,
            document_id=document.document_id
        )
    
    def _should_cache_result(self, document: Document) -> bool:
        """Determine if result should be cached."""
        return self.config.enable_caching
    
    def _get_cache_key(self, document: Document) -> str:
        """Generate cache key for document."""
        key_parts = []
        for field in self.config.cache_key_fields:
            if field == "content":
                import hashlib
                content_hash = hashlib.md5(document.content.encode()).hexdigest()
                key_parts.append(f"content:{content_hash}")
            elif hasattr(document, field):
                key_parts.append(f"{field}:{getattr(document, field)}")
            elif field in document.metadata:
                key_parts.append(f"{field}:{document.metadata[field]}")
        
        config_hash = hash(str(self.config.model_dump()))
        key_parts.append(f"config:{config_hash}")
        
        return "|".join(key_parts)
    
    def _handle_empty_chunk(self, chunk_text: str) -> bool:
        """Handle empty chunks according to configuration."""
        if not chunk_text.strip():
            if self.config.on_empty_chunk == "skip":
                return False
            elif self.config.on_empty_chunk == "error":
                raise ValueError("Empty chunk encountered")
        return True
    
    def _handle_oversized_chunk(self, chunk_text: str) -> List[str]:
        """Handle oversized chunks according to configuration."""
        if self.config.max_chunk_size and len(chunk_text) > self.config.max_chunk_size:
            if self.config.on_oversized_chunk == "split":
                chunks = []
                for i in range(0, len(chunk_text), self.config.max_chunk_size):
                    chunks.append(chunk_text[i:i + self.config.max_chunk_size])
                return chunks
            elif self.config.on_oversized_chunk == "error":
                raise ValueError(f"Chunk size {len(chunk_text)} exceeds maximum {self.config.max_chunk_size}")
        return [chunk_text]
    
    def _validate_chunk_size(self, chunk_text: str) -> bool:
        """Validate chunk size against configuration."""
        chunk_len = len(chunk_text)
        
        if chunk_len < self.config.min_chunk_size:
            return False
            
        if self.config.max_chunk_size and chunk_len > self.config.max_chunk_size:
            return False
            
        return True
    
    def _update_metrics(self, chunks: List[Chunk], processing_time_ms: float, document: Document):
        """Update performance metrics."""
        if not chunks:
            return
            
        chunk_sizes = [len(chunk.text_content) for chunk in chunks]
        
        self._metrics.total_chunks += len(chunks)
        self._metrics.total_characters += sum(chunk_sizes)
        self._metrics.avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
        self._metrics.min_chunk_size = min(chunk_sizes)
        self._metrics.max_chunk_size = max(chunk_sizes)
        self._metrics.processing_time_ms = processing_time_ms
        self._metrics.document_id = document.document_id
    
    def get_metrics(self) -> ChunkingMetrics:
        """Get current metrics for this chunking strategy."""
        return self._metrics.model_copy()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self._metrics = ChunkingMetrics(strategy_name=self.__class__.__name__)
    
    def clear_cache(self):
        """Clear the chunk cache."""
        self._cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        return {
            "enabled": self.config.enable_caching,
            "size": len(self._cache),
            "memory_estimate_mb": sum(
                len(str(chunks)) for chunks in self._cache.values()
            ) / (1024 * 1024)
        }

class TextSplitterConfig(ChunkingConfig):
    """Enhanced configuration for text-based splitting strategies."""
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "], description="List of separators to try")
    keep_separator: bool = Field(True, description="Whether to keep separators in chunks")
    is_separator_regex: bool = Field(False, description="Whether separators are regex patterns")
    
    sentence_splitter: str = Field(r'(?<=[.!?])\s+', description="Regex for sentence splitting")
    paragraph_splitter: str = Field(r'\n\s*\n', description="Regex for paragraph splitting")
    
    overlap_strategy: str = Field("token", description="Overlap strategy: token, sentence, percentage")
    smart_overlap: bool = Field(True, description="Use intelligent overlap based on content structure")


class TextSplitter(ChunkingStrategy, ABC):
    """
    Enhanced abstract base class for chunking strategies that operate by splitting text.

    This class provides comprehensive, reusable logic for taking a list of text splits,
    merging them into chunks of a desired size, and handling the overlap between
    those chunks with advanced features like sentence preservation, smart overlap,
    and performance optimization.
    """

    def __init__(self, config: Optional[TextSplitterConfig] = None):
        """
        Initializes the TextSplitter with enhanced configuration.

        Args:
            config: TextSplitterConfig object with all settings
        """
        if config is None:
            config = TextSplitterConfig()
        
        super().__init__(config)
        
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError(
                f"Chunk overlap ({self.config.chunk_overlap}) must be smaller than chunk size "
                f"({self.config.chunk_size})."
            )
        
        self._chunk_size = self.config.chunk_size
        self._chunk_overlap = self.config.chunk_overlap

    @upsonic_error_handler(max_retries=1, show_error_details=True)
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Enhanced public-facing method that executes the full chunking process.

        Args:
            document: The Document to be chunked.

        Returns:
            A list of enhanced Chunk objects with rich metadata.
        """
        start_time = time.time()
        
        if self._should_cache_result(document):
            cache_key = self._get_cache_key(document)
            if cache_key in self._cache:
                return self._cache[cache_key].copy()
        
        try:
            text_splits = self.split_text(document.content)
            
            if self.config.preserve_sentences or self.config.preserve_paragraphs:
                text_splits = self._enhance_splits_with_boundaries(text_splits, document.content)
            
            final_texts = self._merge_splits_enhanced(text_splits, document.content)

            final_chunks: List[Chunk] = []
            current_pos = 0
            
            for i, text in enumerate(final_texts):
                if not self._handle_empty_chunk(text):
                    continue
                
                text_parts = self._handle_oversized_chunk(text)
                
                for j, text_part in enumerate(text_parts):
                    if not text_part.strip():
                        continue
                        
                    start_pos = document.content.find(text_part, current_pos)
                    if start_pos == -1:
                        start_pos = current_pos
                    end_pos = start_pos + len(text_part)
                    
                    chunk = self._create_chunk(
                        text_content=text_part,
                        document=document,
                        chunk_index=len(final_chunks),
                        total_chunks=len(final_texts),
                        start_pos=start_pos,
                        end_pos=end_pos
                    )
                    
                    final_chunks.append(chunk)
                    current_pos = end_pos
            
            for chunk in final_chunks:
                chunk.metadata['total_chunks'] = len(final_chunks)
            
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(final_chunks, processing_time, document)
            
            if self._should_cache_result(document):
                self._cache[cache_key] = final_chunks.copy()
            
            return final_chunks
            
        except Exception as e:
            print(f"Warning: Advanced chunking failed, falling back to simple splitting: {e}")
            return self._simple_fallback_chunk(document)

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        An abstract method that concrete classes must implement. This is where
        the specific logic for splitting a long text into smaller pieces resides.
        """
        raise NotImplementedError
    
    def _enhance_splits_with_boundaries(self, splits: List[str], original_text: str) -> List[str]:
        """Enhance splits by respecting sentence and paragraph boundaries."""
        if not splits:
            return splits
        
        enhanced_splits = []
        
        for split in splits:
            if not split.strip():
                continue
                
            if self.config.preserve_sentences:
                sentences = re.split(self.config.sentence_splitter, split)
                enhanced_splits.extend([s.strip() for s in sentences if s.strip()])
            else:
                enhanced_splits.append(split)
        
        return enhanced_splits
    
    def _merge_splits_enhanced(self, splits: List[str], original_text: str) -> List[str]:
            """
            Merging with smart overlap and boundary preservation.
            """
            if not splits:
                return []

            final_chunks: List[str] = []
            current_chunk_parts: List[str] = []
            total_chars = 0
            separator = self._choose_separator(splits)

            for split in splits:
                stripped_split = split.strip()
                if not stripped_split:
                    continue

                split_len = len(stripped_split)
                
                if current_chunk_parts and total_chars + len(separator) + split_len > self.config.chunk_size:
                    chunk_text = self._finalize_chunk(current_chunk_parts, separator)
                    if chunk_text:
                        final_chunks.append(chunk_text)

                    if self.config.chunk_overlap > 0 and self.config.smart_overlap:
                        overlap_parts, overlap_chars = self._calculate_smart_overlap(
                            current_chunk_parts, separator, chunk_text
                        )
                        current_chunk_parts = overlap_parts + [stripped_split]
                        total_chars = overlap_chars + (len(separator) if overlap_parts else 0) + split_len
                    else:
                        current_chunk_parts = [stripped_split]
                        total_chars = split_len
                else:
                    if current_chunk_parts:
                        total_chars += len(separator)
                    current_chunk_parts.append(stripped_split)
                    total_chars += split_len

            if current_chunk_parts:
                chunk_text = self._finalize_chunk(current_chunk_parts, separator)
                if chunk_text:
                    final_chunks.append(chunk_text)

            return final_chunks
    
    def _choose_separator(self, splits: List[str]) -> str:
        """Choose appropriate separator based on content analysis."""
        total_text = " ".join(splits[:10])
        
        if "\n\n" in total_text:
            return "\n\n"
        elif "\n" in total_text:
            return "\n"
        else:
            return " "
    
    def _finalize_chunk(self, chunk_parts: List[str], separator: str) -> str:
        """Finalize a chunk with proper formatting."""
        if not chunk_parts:
            return ""
        
        chunk_text = separator.join(chunk_parts).strip()
        
        if self.config.preserve_sentences and not chunk_text.endswith(('.', '!', '?')):
            matches = list(re.finditer(self.config.sentence_splitter, chunk_text))
            if matches:
                last_boundary_end = matches[-1].end()
                if len(chunk_text) > last_boundary_end:
                    chunk_text = chunk_text[:last_boundary_end].strip()
        
        return chunk_text
    
    def _calculate_smart_overlap(self, current_parts: List[str], separator: str, last_chunk: str) -> tuple:
        """Calculate smart overlap based on configuration."""
        if not self.config.smart_overlap or not current_parts or self.config.chunk_overlap <= 0:
            return [], 0

        if not self.config.smart_overlap or not current_parts:
            return [], 0
        
        overlap_chars = 0
        overlap_parts = []
        
        if self.config.overlap_strategy == "sentence" and self.config.preserve_sentences:
            for part in reversed(current_parts):
                if overlap_chars + len(part) <= self.config.chunk_overlap:
                    overlap_parts.insert(0, part)
                    overlap_chars += len(part) + len(separator)
                    if re.search(r'[.!?]\s*$', part):
                        break
                else:
                    break
        else:
            for part in reversed(current_parts):
                if overlap_chars + len(part) + len(separator) <= self.config.chunk_overlap:
                    overlap_parts.insert(0, part)
                    overlap_chars += len(part) + len(separator)
                else:
                    break
        
        return overlap_parts, overlap_chars
    
    def _simple_fallback_chunk(self, document: Document) -> List[Chunk]:
        """Simple fallback chunking when features fail."""
        content = document.content
        chunks = []
        
        for i in range(0, len(content), self.config.chunk_size):
            chunk_text = content[i:i + self.config.chunk_size]
            if chunk_text.strip():
                chunk = self._create_chunk(
                    text_content=chunk_text,
                    document=document,
                    chunk_index=len(chunks),
                    total_chunks=len(content) // self.config.chunk_size + 1,
                    start_pos=i,
                    end_pos=i + len(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Legacy merge method for backward compatibility.
        """
        return self._merge_splits_enhanced(splits, "")