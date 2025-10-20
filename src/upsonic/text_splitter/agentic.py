from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from pydantic import Field

from upsonic.text_splitter.base import BaseChunker, BaseChunkingConfig
from upsonic.schemas.data_models import Document, Chunk
from upsonic.schemas.agentic import PropositionList, TopicAssignmentList, Topic, RefinedTopic
from upsonic.utils.error_wrapper import upsonic_error_handler
from upsonic.utils.logging_config import get_logger

if TYPE_CHECKING:
    from upsonic.agent.agent import Agent

logger = get_logger(__name__)


class AgenticChunkingConfig(BaseChunkingConfig):
    """Configuration for agentic chunking strategy."""
    
    max_agent_retries: int = Field(
        default=3,
        description="Maximum retries for agent calls"
    )
    
    min_proposition_length: int = Field(
        default=20,
        description="Minimum length for valid propositions"
    )
    max_propositions_per_chunk: int = Field(
        default=15,
        description="Maximum propositions in a single chunk"
    )
    min_propositions_per_chunk: int = Field(
        default=3,
        description="Minimum propositions to form a chunk"
    )
    
    enable_proposition_caching: bool = Field(
        default=True,
        description="Cache proposition extraction results"
    )
    enable_topic_caching: bool = Field(
        default=True,
        description="Cache topic assignment results"
    )
    enable_refinement_caching: bool = Field(
        default=True,
        description="Cache topic refinement results"
    )
    
    enable_proposition_validation: bool = Field(
        default=True,
        description="Validate proposition quality"
    )
    enable_topic_optimization: bool = Field(
        default=True,
        description="Optimize topic assignments"
    )
    enable_coherence_scoring: bool = Field(
        default=True,
        description="Score chunk coherence"
    )
    
    fallback_to_recursive: bool = Field(
        default=True,
        description="Fallback to recursive chunking on agent failure"
    )
    
    include_proposition_metadata: bool = Field(
        default=True,
        description="Include proposition-level metadata"
    )
    include_topic_scores: bool = Field(
        default=True,
        description="Include topic coherence scores"
    )
    include_agent_metadata: bool = Field(
        default=True,
        description="Include agent processing metadata"
    )


class AgenticChunker(BaseChunker[AgenticChunkingConfig]):
    """
    Agentic chunking strategy using AI agents for cognitive document processing.
    
    This strategy uses AI agents to:
    1. Extract atomic propositions from documents
    2. Group propositions into coherent topics
    3. Create semantically meaningful chunks
    4. Enrich chunks with AI-generated metadata
    
    Features:
    - Comprehensive caching system
    - Quality validation and scoring
    - Robust error handling with fallbacks
    - Rich metadata enrichment
    """
    
    def __init__(self, agent: "Agent", config: Optional[AgenticChunkingConfig] = None):
        """
        Initialize agentic chunker.
        
        Args:
            agent: Pre-configured Agent class for cognitive processing
            config: Configuration object with all settings
        """
        from upsonic.agent.agent import Agent
        if not isinstance(agent, Agent):
            raise TypeError("An instance of the `Agent` agent is required.")
        
        super().__init__(config)
        self.agent = agent
        
        self._proposition_cache: Dict[str, List[str]] = {}
        self._topic_cache: Dict[str, List[Topic]] = {}
        self._refinement_cache: Dict[str, RefinedTopic] = {}
        
        self._agent_call_count = 0
        self._cache_hits = 0
        self._fallback_count = 0

    def _chunk_document(self, document: Document) -> List[Chunk]:
        """
        Synchronous chunking - delegates to async implementation.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._achunk_document(document))
                return future.result()
        except RuntimeError:
            return asyncio.run(self._achunk_document(document))

    async def _achunk_document(self, document: Document) -> List[Chunk]:
        """
        Core agentic chunking pipeline.
        
        Args:
            document: Document to process with AI agents
            
        Returns:
            List of cognitively-optimized chunks with rich metadata
        """
        start_time = time.time()
        
        if not document.content.strip():
            return []
        
        try:
            propositions = await self._extract_propositions(document)
            if not propositions:
                return await self._fallback_chunking(document)

            if self.config.enable_proposition_validation:
                propositions = self._validate_propositions(propositions)
                if not propositions:
                    return await self._fallback_chunking(document)

            topics = await self._group_propositions_into_topics(propositions, document)
            if not topics:
                return await self._fallback_chunking(document)

            if self.config.enable_topic_optimization:
                topics = self._optimize_topics(topics)

            chunks = await self._create_chunks_from_topics(topics, document)

            if self.config.enable_coherence_scoring:
                chunks = self._score_chunk_coherence(chunks)

            processing_time = (time.time() - start_time) * 1000
            self._add_processing_metadata(chunks, processing_time)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Agentic chunking failed for document {document.document_id}: {e}")
            return await self._fallback_chunking(document)

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]

    @upsonic_error_handler(max_retries=1, show_error_details=True)
    async def _extract_propositions(self, document: Document) -> List[str]:
        cache_key = self._get_cache_key(document.content) if self.config.enable_proposition_caching else None
        
        if cache_key and cache_key in self._proposition_cache:
            self._cache_hits += 1
            return self._proposition_cache[cache_key]
        
        for attempt in range(self.config.max_agent_retries):
            try:
                from upsonic.tasks.tasks import Task
                
                prompt = f"""
                Extract atomic, self-contained factual statements from this document.
                
                Guidelines:
                - Each proposition should be a single piece of information
                - Propositions should be atomic and independent
                - Focus on factual content, not opinions
                - Each proposition should be at least {self.config.min_proposition_length} characters
                - Maximum {self.config.max_propositions_per_chunk * 3} propositions total
                
                Return as JSON conforming to PropositionList schema.

                <DOCUMENT_CONTENT>
                {document.content}
                </DOCUMENT_CONTENT>
                """
                
                task = Task(description=prompt, response_format=PropositionList)
                result = await self.agent.do_async(task)
                propositions = result.propositions if result else []
                
                if cache_key and self.config.enable_proposition_caching:
                    self._proposition_cache[cache_key] = propositions
                
                self._agent_call_count += 1
                return propositions
                
            except Exception as e:
                logger.warning(f"Proposition extraction attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_agent_retries - 1:
                    raise
        
        return []

    def _validate_propositions(self, propositions: List[str]) -> List[str]:
        validated = []
        
        for prop in propositions:
            prop = prop.strip()
            
            if (len(prop) >= self.config.min_proposition_length and 
                prop and not prop.isspace() and 
                prop not in validated):
                validated.append(prop)
        
        return validated

    @upsonic_error_handler(max_retries=1, show_error_details=True)
    async def _group_propositions_into_topics(self, propositions: List[str], document: Document) -> List[Topic]:
        cache_key = self._get_cache_key(str(propositions)) if self.config.enable_topic_caching else None
        
        if cache_key and cache_key in self._topic_cache:
            self._cache_hits += 1
            return self._topic_cache[cache_key]
        
        for attempt in range(self.config.max_agent_retries):
            try:
                from upsonic.tasks.tasks import Task
                
                formatted_propositions = "\n".join(f"- {p}" for p in propositions)
                
                prompt = f"""
                Group these propositions into coherent thematic topics.
                
                Guidelines:
                - Group by shared topics, themes, or subjects
                - Each topic should have {self.config.min_propositions_per_chunk}-{self.config.max_propositions_per_chunk} propositions
                - Create new topics for propositions that don't fit existing ones
                - Ensure topics are thematically coherent
                - Avoid single-proposition topics when possible
                
                Return as JSON conforming to TopicAssignmentList schema.

                <PROPOSITIONS>
                {formatted_propositions}
                </PROPOSITIONS>
                """
                
                task = Task(description=prompt, response_format=TopicAssignmentList)
                result = await self.agent.do_async(task)
                topics = result.topics if result else []
                
                topics = self._validate_topic_sizes(topics)
                
                if cache_key and self.config.enable_topic_caching:
                    self._topic_cache[cache_key] = topics
                
                self._agent_call_count += 1
                return topics
                
            except Exception as e:
                logger.warning(f"Topic assignment attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_agent_retries - 1:
                    raise
        
        return []

    def _validate_topic_sizes(self, topics: List[Topic]) -> List[Topic]:
        validated_topics = []
        
        for topic in topics:
            if (self.config.min_propositions_per_chunk <= len(topic.propositions) <= 
                self.config.max_propositions_per_chunk):
                validated_topics.append(topic)
            elif len(topic.propositions) > self.config.max_propositions_per_chunk:
                for i in range(0, len(topic.propositions), self.config.max_propositions_per_chunk):
                    chunk_props = topic.propositions[i:i + self.config.max_propositions_per_chunk]
                    if len(chunk_props) >= self.config.min_propositions_per_chunk:
                        split_topic = Topic(
                            topic_id=len(validated_topics) + 1,
                            propositions=chunk_props
                        )
                        validated_topics.append(split_topic)
        
        return validated_topics

    def _optimize_topics(self, topics: List[Topic]) -> List[Topic]:
        if not self.config.enable_topic_optimization:
            return topics
        
        optimized_topics = []
        small_propositions = []
        
        for topic in topics:
            if len(topic.propositions) >= self.config.min_propositions_per_chunk:
                optimized_topics.append(topic)
            else:
                small_propositions.extend(topic.propositions)
        
        if small_propositions:
            for i in range(0, len(small_propositions), self.config.max_propositions_per_chunk):
                chunk_props = small_propositions[i:i + self.config.max_propositions_per_chunk]
                if len(chunk_props) >= self.config.min_propositions_per_chunk:
                    merged_topic = Topic(
                        topic_id=len(optimized_topics) + 1,
                        propositions=chunk_props
                    )
                    optimized_topics.append(merged_topic)
        
        return optimized_topics

    def _find_chunk_indices_in_document(self, chunk_text: str, document_content: str) -> Tuple[int, int]:
        chunk_clean = chunk_text.strip()
        doc_clean = document_content.strip()
        
        if chunk_clean in doc_clean:
            start_index = doc_clean.find(chunk_clean)
            end_index = start_index + len(chunk_clean)
            return start_index, end_index
        
        chunk_words = chunk_clean.split()
        if len(chunk_words) < 2:
            return 0, min(len(chunk_clean), len(doc_clean))
        
        for word_count in range(min(5, len(chunk_words)), 1, -1):
            first_words = " ".join(chunk_words[:word_count])
            if first_words in doc_clean:
                start_index = doc_clean.find(first_words)
                end_index = min(start_index + len(chunk_clean), len(doc_clean))
                return start_index, end_index
        
        for word in chunk_words:
            if len(word) > 3 and word in doc_clean:
                start_index = doc_clean.find(word)
                end_index = min(start_index + len(chunk_clean), len(doc_clean))
                return start_index, end_index
        
        return 0, min(len(chunk_clean), len(doc_clean))

    async def _create_chunks_from_topics(self, topics: List[Topic], document: Document) -> List[Chunk]:
        chunks = []
        current_chunk_topics: List[Topic] = []
        current_length = 0
        min_chunk_size = self._get_effective_min_chunk_size()
        
        for i, topic in enumerate(topics):
            chunk_text = " ".join(topic.propositions)
            topic_length = self.config.length_function(chunk_text)
            
            if current_length + topic_length > self.config.chunk_size and current_chunk_topics:
                if current_length >= min_chunk_size:
                    final_chunk = await self._finalize_topic_chunk(current_chunk_topics, document)
                    chunks.append(final_chunk)
                    current_chunk_topics = []
                    current_length = 0
            current_chunk_topics.append(topic)
            current_length += topic_length
        
        if current_chunk_topics:
            final_chunk = await self._finalize_topic_chunk(current_chunk_topics, document)
            chunks.append(final_chunk)
        
        return chunks

    async def _finalize_topic_chunk(self, topics: List[Topic], document: Document) -> Chunk:
        all_propositions = []
        for topic in topics:
            all_propositions.extend(topic.propositions)
        
        chunk_text = " ".join(all_propositions)
        
        base_topic = topics[0]
        topic_ids = [str(topic.topic_id) for topic in topics]
        
        refined_metadata = await self._refine_topic_metadata(chunk_text, base_topic)
        
        start_index, end_index = self._find_chunk_indices_in_document(chunk_text, document.content)
        chunk = self._create_chunk(
            parent_document=document,
            text_content=chunk_text,
            start_index=start_index,
            end_index=end_index,
            extra_metadata={
                "agentic_title": refined_metadata.title,
                "agentic_summary": refined_metadata.summary,
                "topic_ids": "+".join(topic_ids) if len(topic_ids) > 1 else topic_ids[0],
                "proposition_count": len(all_propositions),
                "chunking_method": "agentic_cognitive",
                "agent_processed": True,
                "merged_topics": len(topics) > 1
            }
        )
        
        if self.config.include_proposition_metadata:
            chunk.metadata["propositions"] = all_propositions[:5]
            chunk.metadata["total_propositions"] = len(all_propositions)
        
        return chunk

    @upsonic_error_handler(max_retries=1, show_error_details=True)
    async def _refine_topic_metadata(self, chunk_text: str, topic: Topic) -> RefinedTopic:
        cache_key = self._get_cache_key(chunk_text) if self.config.enable_refinement_caching else None
        
        if cache_key and cache_key in self._refinement_cache:
            self._cache_hits += 1
            return self._refinement_cache[cache_key]
        
        for attempt in range(self.config.max_agent_retries):
            try:
                from upsonic.tasks.tasks import Task
                
                prompt = f"""
                Create a title and summary for this content.
                
                Guidelines:
                - Title: concise (3-8 words) and descriptive
                - Summary: comprehensive but concise (1-2 sentences)
                - Focus on main theme and key information
                - Use clear, professional language
                
                Return as JSON conforming to RefinedTopic schema.

                <CHUNK_TEXT>
                {chunk_text}
                </CHUNK_TEXT>
                """
                
                task = Task(description=prompt, response_format=RefinedTopic)
                result = await self.agent.do_async(task)
                refined = result if result else RefinedTopic(
                    title=f"Topic {topic.topic_id}", 
                    summary="Auto-generated summary."
                )
                
                if cache_key and self.config.enable_refinement_caching:
                    self._refinement_cache[cache_key] = refined
                
                self._agent_call_count += 1
                return refined
                
            except Exception as e:
                logger.warning(f"Metadata refinement attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_agent_retries - 1:
                    return RefinedTopic(
                        title=f"Topic {topic.topic_id}", 
                        summary="Auto-generated summary due to processing error."
                    )
        
        return RefinedTopic(title="Untitled Topic", summary="No summary available.")

    def _score_chunk_coherence(self, chunks: List[Chunk]) -> List[Chunk]:
        if not self.config.enable_coherence_scoring:
            return chunks
        
        for chunk in chunks:
            coherence_score = self._calculate_coherence_score(chunk)
            
            if self.config.include_topic_scores:
                chunk.metadata["coherence_score"] = coherence_score
                chunk.metadata["quality_assessment"] = self._assess_chunk_quality(coherence_score)
        
        return chunks

    def _calculate_coherence_score(self, chunk: Chunk) -> float:
        text_length = len(chunk.text_content)
        proposition_count = chunk.metadata.get("proposition_count", 1)
        
        length_score = min(text_length / 1000, 1.0)
        proposition_score = min(proposition_count / 10, 1.0)
        
        coherence_score = (length_score * 0.4 + proposition_score * 0.6)
        return round(coherence_score, 3)

    def _assess_chunk_quality(self, coherence_score: float) -> str:
        if coherence_score >= 0.8:
            return "excellent"
        elif coherence_score >= 0.6:
            return "good"
        elif coherence_score >= 0.4:
            return "fair"
        else:
            return "poor"

    async def _fallback_chunking(self, document: Document) -> List[Chunk]:
        if not self.config.fallback_to_recursive:
            return []
        
        self._fallback_count += 1
        logger.warning(f"Falling back to recursive chunking for document {document.document_id}")
        
        try:
            from .recursive import RecursiveChunker, RecursiveChunkingConfig
            fallback_config = RecursiveChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            fallback_chunker = RecursiveChunker(fallback_config)
            
            fallback_chunks = fallback_chunker._chunk_document(document)
            
            for chunk in fallback_chunks:
                chunk.metadata["agentic_fallback"] = True
                chunk.metadata["chunking_method"] = "recursive_fallback"
            
            return fallback_chunks
            
        except Exception as e:
            logger.error(f"Fallback chunking also failed: {e}")
            return []

    def _add_processing_metadata(self, chunks: List[Chunk], processing_time: float):
        if not self.config.include_agent_metadata:
            return
        
        for chunk in chunks:
            chunk.metadata.update({
                "agent_calls": self._agent_call_count,
                "cache_hits": self._cache_hits,
                "processing_time_ms": processing_time,
                "processing_stage": "completed"
            })

    def get_agentic_stats(self) -> Dict[str, any]:
        return {
            "agent_calls": self._agent_call_count,
            "cache_hits": self._cache_hits,
            "fallback_count": self._fallback_count,
            "proposition_cache_size": len(self._proposition_cache),
            "topic_cache_size": len(self._topic_cache),
            "refinement_cache_size": len(self._refinement_cache),
            "caching_enabled": {
                "propositions": self.config.enable_proposition_caching,
                "topics": self.config.enable_topic_caching,
                "refinements": self.config.enable_refinement_caching
            },
            "quality_features_enabled": {
                "proposition_validation": self.config.enable_proposition_validation,
                "topic_optimization": self.config.enable_topic_optimization,
                "coherence_scoring": self.config.enable_coherence_scoring
            }
        }

    def clear_agentic_caches(self):
        self._proposition_cache.clear()
        self._topic_cache.clear()
        self._refinement_cache.clear()
        self._cache_hits = 0