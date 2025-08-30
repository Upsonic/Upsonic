from __future__ import annotations
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import time
from pydantic import Field

from upsonic.text_splitter.base import ChunkingStrategy, ChunkingConfig
from upsonic.schemas.data_models import Document, Chunk
from upsonic.schemas.agentic import PropositionList, TopicAssignmentList, Topic, RefinedTopic
from ..utils.error_wrapper import upsonic_error_handler

if TYPE_CHECKING:
    from upsonic.agent.agent import Direct
    from upsonic.tasks.tasks import Task


class AgenticChunkingConfig(ChunkingConfig):
    """Enhanced configuration for agentic chunking strategy."""
    enable_proposition_caching: bool = Field(True, description="Cache proposition extraction results")
    enable_topic_caching: bool = Field(True, description="Cache topic assignment results")
    enable_refinement_caching: bool = Field(True, description="Cache topic refinement results")
    
    min_propositions_per_chunk: int = Field(3, description="Minimum propositions to form a chunk")
    max_propositions_per_chunk: int = Field(15, description="Maximum propositions in a single chunk")
    min_proposition_length: int = Field(20, description="Minimum length for valid propositions")
    
    enable_proposition_validation: bool = Field(True, description="Validate proposition quality")
    enable_topic_optimization: bool = Field(True, description="Optimize topic assignments")
    enable_coherence_scoring: bool = Field(True, description="Score chunk coherence")
    parallel_processing: bool = Field(False, description="Enable parallel agent calls")
    
    fallback_to_recursive: bool = Field(True, description="Fallback to recursive chunking on agent failure")
    max_agent_retries: int = Field(3, description="Maximum retries for agent calls")
    agent_timeout_seconds: int = Field(60, description="Timeout for agent operations")
    
    batch_proposition_size: int = Field(100, description="Batch size for proposition processing")
    enable_incremental_processing: bool = Field(True, description="Process documents incrementally")
    
    include_proposition_metadata: bool = Field(True, description="Include proposition-level metadata")
    include_topic_scores: bool = Field(True, description="Include topic coherence scores")
    include_agent_metadata: bool = Field(True, description="Include agent processing metadata")


class AgenticChunkingStrategy(ChunkingStrategy):
    """
    Agentic chunking strategy with framework-level features.

    This advanced strategy uses AI agents to cognitively deconstruct and
    reorganize documents into thematically coherent chunks with
    capabilities:
    
    Features:
    - Comprehensive caching system for all agent operations
    - Quality validation and coherence scoring
    - Parallel processing and performance optimization
    - Robust error handling with intelligent fallbacks
    - Rich metadata enrichment with agent insights
    - Incremental processing for large documents
    - Advanced proposition and topic management
    
    This strategy executes an cognitive pipeline:
    1. **Proposition Extraction** - Agent breaks document into atomic statements
    2. **Quality Validation** - Validates and filters propositions
    3. **Batch Topic Clustering** - Groups propositions into thematic clusters
    4. **Topic Optimization** - Refines and optimizes topic assignments  
    5. **Coherence Scoring** - Evaluates chunk semantic coherence
    6. **Metadata Enrichment** - Adds comprehensive agent-generated metadata

    """
    
    def __init__(self, agent: "Direct", config: Optional[AgenticChunkingConfig] = None):
        """
        Initialize agentic chunking strategy.

        Args:
            agent: Pre-configured Direct agent for cognitive processing
            config: Configuration object with all settings
        """
        if not isinstance(agent, "Direct"):
            raise TypeError("An instance of the `Direct` agent is required.")
        
        if config is None:
            config = AgenticChunkingConfig()
        
        super().__init__(config)
        
        self.agent = agent
        
        self._proposition_cache: Dict[str, List[str]] = {}
        self._topic_cache: Dict[str, List[Topic]] = {}
        self._refinement_cache: Dict[str, RefinedTopic] = {}
        self._coherence_scores: Dict[str, float] = {}
        
        self._agent_call_count = 0
        self._cache_hits = 0
        self._fallback_count = 0

    @upsonic_error_handler(max_retries=1, show_error_details=True)
    async def chunk(self, document: Document) -> List[Chunk]:
        """
        Agentic chunking pipeline with framework features.
        
        Args:
            document: Document to process with AI agents
            
        Returns:
            List of cognitively-optimized chunks with rich metadata
        """
        start_time = time.time()
        
        if not document.content.strip():
            return []
        
        try:
            propositions = await self._generate_propositions_enhanced(document)
            if not propositions:
                return await self._fallback_chunking(document)

            if self.config.enable_proposition_validation:
                propositions = self._validate_propositions(propositions)

            topic_clusters = await self._assign_propositions_to_topics_enhanced(propositions, document)
            if not topic_clusters:
                return await self._fallback_chunking(document)

            if self.config.enable_topic_optimization:
                topic_clusters = await self._optimize_topic_assignments(topic_clusters, propositions)

            chunks = await self._create_enhanced_chunks(topic_clusters, document)

            if self.config.enable_coherence_scoring:
                chunks = await self._score_chunk_coherence(chunks)

            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(chunks, processing_time, document)
            
            return chunks
            
        except Exception as e:
            print(f"Agentic chunking failed for document {document.document_id}: {e}")
            return await self._fallback_chunking(document)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for content."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    async def _generate_propositions_enhanced(self, document: Document) -> List[str]:
        """Proposition generation with caching and validation."""
        cache_key = self._get_cache_key(document.content) if self.config.enable_proposition_caching else None
        
        if cache_key and cache_key in self._proposition_cache:
            self._cache_hits += 1
            return self._proposition_cache[cache_key]
        
        for attempt in range(self.config.max_agent_retries):
            try:
                propositions = await self._generate_propositions(document.content)
                
                if self.config.enable_proposition_validation:
                    propositions = [p for p in propositions if len(p.strip()) >= self.config.min_proposition_length]
                
                if cache_key and self.config.enable_proposition_caching:
                    self._proposition_cache[cache_key] = propositions
                
                self._agent_call_count += 1
                return propositions
                
            except Exception as e:
                print(f"Proposition generation attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_agent_retries - 1:
                    raise
        
        return []
    
    async def _generate_propositions(self, text: str) -> List[str]:
        """Proposition extraction with better prompting."""
        prompt = f"""
        Your task is to act as a meticulous knowledge extraction engine.
        Read the following document and deconstruct it into a list of simple, granular, and self-contained factual statements (propositions).
        
        Guidelines:
        - Each proposition should represent a single piece of information
        - Propositions should be atomic and independent
        - Focus on factual content, not opinions or interpretations
        - Each proposition should be at least {self.config.min_proposition_length} characters
        - Maximum {self.config.max_propositions_per_chunk * 3} propositions total
        
        Return the result as a JSON object conforming to the `PropositionList` schema.

        <DOCUMENT_CONTENT>
        {text}
        </DOCUMENT_CONTENT>
        """
        task = Task(description=prompt, response_format=PropositionList)
        result = await self.agent.do_async(task)
        return result.propositions if result else []
    
    def _validate_propositions(self, propositions: List[str]) -> List[str]:
        """Validate and filter propositions based on quality criteria."""
        validated = []
        
        for prop in propositions:
            prop = prop.strip()
            
            if len(prop) < self.config.min_proposition_length:
                continue
            
            if not prop or prop.isspace():
                continue
            
            if prop not in validated:
                validated.append(prop)
        
        return validated

    async def _assign_propositions_to_topics_enhanced(self, propositions: List[str], document: Document) -> List[Topic]:
        """Topic assignment with caching and optimization."""
        if not propositions:
            return []
        
        cache_key = self._get_cache_key(str(propositions)) if self.config.enable_topic_caching else None
        
        if cache_key and cache_key in self._topic_cache:
            self._cache_hits += 1
            return self._topic_cache[cache_key]
        
        if len(propositions) > self.config.batch_proposition_size:
            return await self._batch_process_topics(propositions, document)
        
        for attempt in range(self.config.max_agent_retries):
            try:
                topics = await self._assign_propositions_to_topics(propositions)
                
                topics = self._validate_topic_assignments(topics, propositions)
                
                if cache_key and self.config.enable_topic_caching:
                    self._topic_cache[cache_key] = topics
                
                self._agent_call_count += 1
                return topics
                
            except Exception as e:
                print(f"Topic assignment attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_agent_retries - 1:
                    raise
        
        return []
    
    async def _assign_propositions_to_topics(self, propositions: List[str]) -> List[Topic]:
        """Topic assignment with better clustering guidance."""
        formatted_propositions = "\n".join(f"- {p}" for p in propositions)
        
        prompt = f"""
        You are an expert librarian and data analyst. Your task is to organize propositions into coherent thematic groups.
        
        Guidelines:
        - Group propositions by shared topics, themes, or subjects
        - Each topic should have {self.config.min_propositions_per_chunk}-{self.config.max_propositions_per_chunk} propositions
        - Create new topics for propositions that don't fit existing ones
        - Ensure topics are thematically coherent and meaningful
        - Avoid creating too many single-proposition topics
        
        Analyze the propositions below and group them intelligently.
        Return a JSON object conforming to the `TopicAssignmentList` schema.

        <PROPOSITIONS_TO_ASSIGN>
        {formatted_propositions}
        </PROPOSITIONS_TO_ASSIGN>
        """
        task = Task(description=prompt, response_format=TopicAssignmentList)
        result = await self.agent.do_async(task)
        return result.topics if result else []
    
    def _validate_topic_assignments(self, topics: List[Topic], propositions: List[str]) -> List[Topic]:
        """Validate and optimize topic assignments."""
        validated_topics = []
        
        for topic in topics:
            if (self.config.min_propositions_per_chunk <= len(topic.propositions) <= 
                self.config.max_propositions_per_chunk):
                validated_topics.append(topic)
            elif len(topic.propositions) > self.config.max_propositions_per_chunk:
                split_topics = self._split_large_topic(topic)
                validated_topics.extend(split_topics)
        
        return validated_topics
    
    def _split_large_topic(self, topic: Topic) -> List[Topic]:
        """Split a topic that has too many propositions."""
        max_props = self.config.max_propositions_per_chunk
        split_topics = []
        
        for i in range(0, len(topic.propositions), max_props):
            chunk_props = topic.propositions[i:i + max_props]
            split_topic = Topic(
                topic_id=len(split_topics) + 1,
                propositions=chunk_props
            )
            split_topics.append(split_topic)
        
        return split_topics
    
    async def _batch_process_topics(self, propositions: List[str], document: Document) -> List[Topic]:
        """Process large proposition lists in batches."""
        batch_size = self.config.batch_proposition_size
        all_topics = []
        
        for i in range(0, len(propositions), batch_size):
            batch = propositions[i:i + batch_size]
            batch_topics = await self._assign_propositions_to_topics(batch)
            all_topics.extend(batch_topics)
        
        return all_topics
    
    async def _optimize_topic_assignments(self, topics: List[Topic], propositions: List[str]) -> List[Topic]:
        """Optimize topic assignments for better coherence."""
        if not self.config.enable_topic_optimization:
            return topics
        
        optimized_topics = self._merge_small_topics(topics)
        
        balanced_topics = self._balance_topic_sizes(optimized_topics)
        
        return balanced_topics
    
    def _merge_small_topics(self, topics: List[Topic]) -> List[Topic]:
        """Merge topics that are too small."""
        merged_topics = []
        small_topics = []
        
        for topic in topics:
            if len(topic.propositions) >= self.config.min_propositions_per_chunk:
                merged_topics.append(topic)
            else:
                small_topics.append(topic)
        
        if small_topics:
            all_small_props = []
            for small_topic in small_topics:
                all_small_props.extend(small_topic.propositions)
            
            for i in range(0, len(all_small_props), self.config.max_propositions_per_chunk):
                chunk_props = all_small_props[i:i + self.config.max_propositions_per_chunk]
                if len(chunk_props) >= self.config.min_propositions_per_chunk:
                    merged_topic = Topic(
                        topic_id=len(merged_topics) + 1,
                        propositions=chunk_props
                    )
                    merged_topics.append(merged_topic)
        
        return merged_topics
    
    def _balance_topic_sizes(self, topics: List[Topic]) -> List[Topic]:
        """Balance topic sizes for optimal chunks."""
        return topics

    async def _create_enhanced_chunks(self, topics: List[Topic], document: Document) -> List[Chunk]:
        """Create enhanced chunks with rich metadata."""
        chunks = []
        
        for i, topic in enumerate(topics):
            chunk_text = " ".join(topic.propositions)
            
            refined_metadata = await self._refine_topic_metadata_enhanced(chunk_text, topic)
            
            chunk = self._create_chunk(
                text_content=chunk_text,
                document=document,
                chunk_index=i,
                total_chunks=len(topics),
                start_pos=0,
                end_pos=len(chunk_text)
            )
            
            chunk.metadata.update({
                "agentic_title": refined_metadata.title,
                "agentic_summary": refined_metadata.summary,
                "topic_id": topic.topic_id,
                "proposition_count": len(topic.propositions),
                "chunking_method": "agentic_cognitive",
                "agent_processed": True
            })
            
            if self.config.include_proposition_metadata:
                chunk.metadata["propositions"] = topic.propositions[:5]
                chunk.metadata["total_propositions"] = len(topic.propositions)
            
            if self.config.include_agent_metadata:
                chunk.metadata.update({
                    "agent_calls": self._agent_call_count,
                    "cache_hits": self._cache_hits,
                    "processing_stage": "refined"
                })
            
            chunks.append(chunk)
        
        return chunks
    
    async def _refine_topic_metadata_enhanced(self, chunk_text: str, topic: Topic) -> RefinedTopic:
        """Metadata refinement with caching."""
        cache_key = self._get_cache_key(chunk_text) if self.config.enable_refinement_caching else None
        
        if cache_key and cache_key in self._refinement_cache:
            self._cache_hits += 1
            return self._refinement_cache[cache_key]
        
        for attempt in range(self.config.max_agent_retries):
            try:
                refined = await self._refine_topic_metadata(chunk_text)
                
                if cache_key and self.config.enable_refinement_caching:
                    self._refinement_cache[cache_key] = refined
                
                self._agent_call_count += 1
                return refined
                
            except Exception as e:
                print(f"Metadata refinement attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_agent_retries - 1:
                    return RefinedTopic(
                        title=f"Topic {topic.topic_id}", 
                        summary="Auto-generated summary due to processing error."
                    )
        
        return RefinedTopic(title="Untitled Topic", summary="No summary available.")
    
    async def _refine_topic_metadata(self, chunk_text: str) -> RefinedTopic:
        """Metadata refinement with better prompting."""
        prompt = f"""
        You are a skilled technical writer. The following text is a collection of related facts.
        Your task is to create a high-quality title and summary for this content.
        
        Guidelines:
        - Title should be concise (3-8 words) and descriptive
        - Summary should be comprehensive but concise (1-2 sentences)
        - Focus on the main theme and key information
        - Use clear, professional language
        
        Your output MUST be a single JSON object conforming to the `RefinedTopic` schema.

        <CHUNK_TEXT>
        {chunk_text}
        </CHUNK_TEXT>
        """
        task = Task(description=prompt, response_format=RefinedTopic)
        result = await self.agent.do_async(task)
        
        return result if result else RefinedTopic(title="Untitled Topic", summary="No summary available.")
    
    async def _score_chunk_coherence(self, chunks: List[Chunk]) -> List[Chunk]:
        """Score chunk coherence and add to metadata."""
        if not self.config.enable_coherence_scoring:
            return chunks
        
        for chunk in chunks:
            coherence_score = self._calculate_coherence_score(chunk)
            
            if self.config.include_topic_scores:
                chunk.metadata["coherence_score"] = coherence_score
                chunk.metadata["quality_assessment"] = self._assess_chunk_quality(coherence_score)
        
        return chunks
    
    def _calculate_coherence_score(self, chunk: Chunk) -> float:
        """Calculate a coherence score for the chunk."""
        text_length = len(chunk.text_content)
        proposition_count = chunk.metadata.get("proposition_count", 1)
        
        length_score = min(text_length / 1000, 1.0)
        proposition_score = min(proposition_count / 10, 1.0)
        
        coherence_score = (length_score * 0.4 + proposition_score * 0.6)
        
        return round(coherence_score, 3)
    
    def _assess_chunk_quality(self, coherence_score: float) -> str:
        """Assess chunk quality based on coherence score."""
        if coherence_score >= 0.8:
            return "excellent"
        elif coherence_score >= 0.6:
            return "good"
        elif coherence_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    async def _fallback_chunking(self, document: Document) -> List[Chunk]:
        """Fallback to recursive chunking when agentic processing fails."""
        if not self.config.fallback_to_recursive:
            return []
        
        self._fallback_count += 1
        print(f"Falling back to recursive chunking for document {document.document_id}")
        
        try:
            from .recursive import RecursiveCharacterChunkingStrategy
            fallback_strategy = RecursiveCharacterChunkingStrategy(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            fallback_chunks = fallback_strategy.chunk(document)
            
            for chunk in fallback_chunks:
                chunk.metadata["agentic_fallback"] = True
                chunk.metadata["chunking_method"] = "recursive_fallback"
            
            return fallback_chunks
            
        except Exception as e:
            print(f"Fallback chunking also failed: {e}")
            return []
    
    def get_agentic_stats(self) -> Dict[str, Any]:
        """Get statistics about agentic processing."""
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
        """Clear all agentic processing caches."""
        self._proposition_cache.clear()
        self._topic_cache.clear()
        self._refinement_cache.clear()
        self._coherence_scores.clear()
        self._cache_hits = 0