from __future__ import annotations
from typing import List, Dict, Any
import time
import re

from pydantic import Field

from upsonic.text_splitter.base import ChunkingStrategy, ChunkingConfig
from upsonic.schemas.data_models import Document, Chunk
from upsonic.schemas.rule import RoutingRule


class RuleBasedChunkingConfig(ChunkingConfig):
    """Enhanced configuration for rule-based chunking strategy."""
    rules: List[RoutingRule] = Field(..., description="Prioritized list of RoutingRule objects")
    default_strategy: ChunkingStrategy = Field(..., description="Fallback strategy when no rules match")
    
    enable_rule_caching: bool = Field(True, description="Cache rule matching results")
    enable_strategy_caching: bool = Field(True, description="Cache strategy results per rule")
    rule_priority_optimization: bool = Field(True, description="Optimize rule order based on usage")
    
    parallel_rule_evaluation: bool = Field(False, description="Evaluate rules in parallel")
    enable_rule_statistics: bool = Field(True, description="Track rule usage statistics")
    strategy_timeout_seconds: int = Field(300, description="Timeout for individual strategy execution")
    
    enable_content_analysis: bool = Field(True, description="Analyze content for better rule matching")
    enable_fallback_chain: bool = Field(True, description="Try multiple strategies if primary fails")
    enable_rule_validation: bool = Field(True, description="Validate rules before execution")
    
    min_confidence_threshold: float = Field(0.7, description="Minimum confidence for rule matching")
    enable_strategy_validation: bool = Field(True, description="Validate strategy results")
    enable_cross_strategy_optimization: bool = Field(False, description="Optimize across multiple strategies")
    
    include_routing_metadata: bool = Field(True, description="Include rule routing information")
    include_strategy_metadata: bool = Field(True, description="Include strategy-specific metadata")
    include_performance_metadata: bool = Field(True, description="Include performance statistics")

    class Config:
        arbitrary_types_allowed = True


class RuleBasedChunkingStrategy(ChunkingStrategy):
    """
    Intelligent routing strategy with framework-level features.

    This meta-strategy acts as an intelligent router, delegating chunking to
    specialist strategies based on document characteristics. Enhanced with:
    
    Features:
    - Advanced rule caching and optimization
    - Content analysis for intelligent routing
    - Performance monitoring and statistics
    - Fallback chains and error handling
    - Cross-strategy optimization
    - Rich metadata and routing information
    - Parallel rule evaluation
    - Strategy validation and confidence scoring
    
    This enables sophisticated document processing pipelines that can:
    - Route different document types to optimal strategies
    - Learn from usage patterns to optimize routing
    - Handle complex multi-format document collections
    - Provide comprehensive routing analytics
    - Gracefully handle edge cases and errors
    
    """
    
    def __init__(self, config: RuleBasedChunkingConfig):
        """
        Initialize rule-based chunking strategy.

        Args:
            config: Configuration object with all settings including rules and default_strategy
        """
        if not isinstance(config, RuleBasedChunkingConfig):
            raise TypeError("The 'config' parameter must be an instance of RuleBasedChunkingConfig.")
        
        if not isinstance(config.rules, list) or not all(isinstance(r, RoutingRule) for r in config.rules):
            raise TypeError("The 'rules' parameter must be a list of RoutingRule instances.")
        
        if not isinstance(config.default_strategy, ChunkingStrategy):
            raise TypeError("The 'default_strategy' parameter must be an instance of a ChunkingStrategy.")
        
        super().__init__(config)
        
        self.rules = config.rules
        self.default_strategy = config.default_strategy
        
        self._rule_cache: Dict[str, int] = {}
        self._strategy_cache: Dict[str, List[Chunk]] = {}
        self._rule_stats: Dict[int, Dict[str, Any]] = {}
        self._strategy_performance: Dict[str, List[float]] = {}
        
        for i, rule in enumerate(self.rules):
            self._rule_stats[i] = {
                "usage_count": 0,
                "success_count": 0,
                "total_processing_time": 0,
                "avg_chunk_count": 0,
                "confidence_scores": []
            }
        
        self._routing_decisions = 0
        self._cache_hits = 0
        self._fallback_usage = 0

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Document processing with intelligent routing and optimization.

        Args:
            document: The Document to be chunked

        Returns:
            A list of Chunk objects created by the selected specialist strategy
        """
        start_time = time.time()
        self._routing_decisions += 1
        
        if not document.content.strip():
            return []
        
        try:
            cache_key = self._get_cache_key(document) if self.config.enable_strategy_caching else None
            if cache_key and cache_key in self._strategy_cache:
                self._cache_hits += 1
                cached_chunks = self._strategy_cache[cache_key]
                return self._enhance_cached_chunks(cached_chunks, document)
            
            selected_rule_index, selected_strategy, confidence = self._select_optimal_strategy(document)
            
            chunks = self._execute_chunking_with_strategy(
                selected_strategy, document, selected_rule_index, confidence
            )
            
            if self.config.enable_strategy_validation:
                chunks = self._validate_strategy_results(chunks, document, selected_rule_index)
            
            if cache_key and self.config.enable_strategy_caching:
                self._strategy_cache[cache_key] = [chunk.copy() for chunk in chunks]
            
            processing_time = (time.time() - start_time) * 1000
            self._update_rule_stats(selected_rule_index, chunks, processing_time, confidence)
            self._update_metrics(chunks, processing_time, document)
            
            return chunks
            
        except Exception as e:
            print(f"Rule-based chunking failed for document {document.document_id}: {e}")
            return self._fallback_chunking(document)

    def _get_cache_key(self, document: Document) -> str:
        """Generate cache key for document."""
        import hashlib
        content_hash = hashlib.md5(document.content.encode()).hexdigest()[:16]
        metadata_str = str(sorted(document.metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:8]
        return f"{content_hash}_{metadata_hash}"
    
    def _select_optimal_strategy(self, document: Document) -> tuple:
        """Strategy selection with content analysis and confidence scoring."""
        cache_key = self._get_cache_key(document) if self.config.enable_rule_caching else None
        if cache_key and cache_key in self._rule_cache:
            self._cache_hits += 1
            rule_index = self._rule_cache[cache_key]
            if rule_index < len(self.rules):
                return rule_index, self.rules[rule_index].strategy, 1.0
        
        content_features = self._analyze_content(document) if self.config.enable_content_analysis else {}
        
        rule_scores = []
        for i, rule in enumerate(self.rules):
            confidence = self._calculate_rule_confidence(rule, document, content_features)
            if confidence >= self.config.min_confidence_threshold:
                rule_scores.append((i, rule, confidence))
        
        if self.config.rule_priority_optimization:
            rule_scores = self._optimize_rule_priority(rule_scores)
        else:
            rule_scores.sort(key=lambda x: x[2], reverse=True)
        
        if rule_scores:
            rule_index, rule, confidence = rule_scores[0]
            
            if cache_key and self.config.enable_rule_caching:
                self._rule_cache[cache_key] = rule_index
            
            return rule_index, rule.strategy, confidence
        
        self._fallback_usage += 1
        return -1, self.default_strategy, 0.5
    
    def _calculate_rule_confidence(self, rule: RoutingRule, document: Document, content_features: Dict) -> float:
        """Calculate confidence score for rule matching."""
        confidence = 0.0
        checks_passed = 0
        total_checks = 0
        
        if rule.file_extension:
            total_checks += 1
            source = document.metadata.get("source", "")
            if isinstance(source, str) and source.endswith(rule.file_extension):
                confidence += 1.0
                checks_passed += 1
        
        if rule.metadata_filter:
            total_checks += len(rule.metadata_filter)
            for key, value in rule.metadata_filter.items():
                if document.metadata.get(key) == value:
                    confidence += 1.0
                    checks_passed += 1
        
        if self.config.enable_content_analysis and content_features:
            content_type = content_features.get("content_type", "unknown")
            if hasattr(rule, 'preferred_content_types'):
                total_checks += 1
                if content_type in getattr(rule, 'preferred_content_types', []):
                    confidence += 1.0
                    checks_passed += 1
        
        if total_checks > 0:
            final_confidence = confidence / total_checks
        else:
            final_confidence = 0.0
        
        rule_index = self.rules.index(rule) if rule in self.rules else -1
        if rule_index in self._rule_stats:
            stats = self._rule_stats[rule_index]
            if stats["usage_count"] > 0:
                success_rate = stats["success_count"] / stats["usage_count"]
                final_confidence = final_confidence * 0.8 + success_rate * 0.2
        
        return final_confidence
    
    def _analyze_content(self, document: Document) -> Dict[str, Any]:
        """Analyze document content for rule matching."""
        content = document.content.lower()
        features = {}
        
        if any(marker in content for marker in ["def ", "class ", "import ", "from "]):
            features["content_type"] = "code"
        elif any(marker in content for marker in ["# ", "## ", "### "]):
            features["content_type"] = "markdown"
        elif any(marker in content for marker in ["<html", "<div", "<p>", "<span"]):
            features["content_type"] = "html"
        elif "{" in content and "}" in content and (":" in content or "=" in content):
            features["content_type"] = "structured"
        else:
            features["content_type"] = "text"
        
        features["line_count"] = len(document.content.split('\n'))
        features["word_count"] = len(document.content.split())
        features["char_count"] = len(document.content)
        
        features["has_headers"] = bool(re.search(r'^#+\s+', document.content, re.MULTILINE))
        features["has_code_blocks"] = "```" in document.content
        features["has_lists"] = bool(re.search(r'^\s*[-*+]\s+', document.content, re.MULTILINE))
        
        return features
    
    def _optimize_rule_priority(self, rule_scores: List[tuple]) -> List[tuple]:
        """Optimize rule priority based on usage statistics."""
        def priority_score(rule_tuple):
            rule_index, rule, confidence = rule_tuple
            
            score = confidence
            
            if rule_index in self._rule_stats:
                stats = self._rule_stats[rule_index]
                if stats["usage_count"] > 0:
                    success_rate = stats["success_count"] / stats["usage_count"]
                    avg_time = stats["total_processing_time"] / stats["usage_count"]
                    
                    performance_boost = success_rate * 0.2 - min(avg_time / 1000, 0.1)
                    score += performance_boost
            
            return score
        
        return sorted(rule_scores, key=priority_score, reverse=True)
    
    def _execute_chunking_with_strategy(
        self, strategy: ChunkingStrategy, document: Document, rule_index: int, confidence: float
    ) -> List[Chunk]:
        """Execute chunking with selected strategy and handle errors."""
        try:
            chunks = strategy.chunk(document)
            
            if self.config.include_routing_metadata:
                for chunk in chunks:
                    chunk.metadata.update({
                        "routing_rule_index": rule_index,
                        "routing_confidence": confidence,
                        "routing_strategy": strategy.__class__.__name__,
                        "routed_by": "RuleBasedChunkingStrategy"
                    })
            
            if self.config.include_strategy_metadata:
                strategy_name = strategy.__class__.__name__
                for chunk in chunks:
                    chunk.metadata.update({
                        "primary_strategy": strategy_name,
                        "strategy_type": self._classify_strategy_type(strategy)
                    })
            
            return chunks
            
        except Exception as e:
            print(f"Strategy {strategy.__class__.__name__} failed: {e}")
            if self.config.enable_fallback_chain:
                return self._try_fallback_strategies(document, rule_index)
            else:
                raise
    
    def _try_fallback_strategies(self, document: Document, failed_rule_index: int) -> List[Chunk]:
        """Try fallback strategies when primary strategy fails."""
        for i, rule in enumerate(self.rules):
            if i != failed_rule_index:
                try:
                    if self._rule_matches(rule, document):
                        print(f"Trying fallback strategy: {rule.strategy.__class__.__name__}")
                        chunks = rule.strategy.chunk(document)
                        
                        for chunk in chunks:
                            chunk.metadata["fallback_processing"] = True
                            chunk.metadata["original_rule_failed"] = failed_rule_index
                        
                        return chunks
                except Exception as e:
                    print(f"Fallback strategy {rule.strategy.__class__.__name__} also failed: {e}")
                    continue
        
        print("Using default strategy as final fallback")
        return self.default_strategy.chunk(document)
    
    def _classify_strategy_type(self, strategy: ChunkingStrategy) -> str:
        """Classify strategy type for metadata."""
        strategy_name = strategy.__class__.__name__.lower()
        
        if "recursive" in strategy_name:
            return "recursive"
        elif "semantic" in strategy_name:
            return "semantic"
        elif "agentic" in strategy_name:
            return "agentic"
        elif "character" in strategy_name:
            return "character"
        elif "markdown" in strategy_name:
            return "markdown"
        elif "html" in strategy_name:
            return "html"
        elif "json" in strategy_name:
            return "json"
        elif "python" in strategy_name:
            return "code"
        else:
            return "custom"
    
    def _validate_strategy_results(self, chunks: List[Chunk], document: Document, rule_index: int) -> List[Chunk]:
        """Validate strategy results and handle quality issues."""
        if not chunks:
            print(f"Strategy produced no chunks for document {document.document_id}")
            return self._fallback_chunking(document)
        
        total_chunk_length = sum(len(chunk.text_content) for chunk in chunks)
        original_length = len(document.content)
        
        if total_chunk_length < original_length * 0.8:
            print(f"Strategy may have lost content: {total_chunk_length}/{original_length}")
            if self.config.enable_fallback_chain:
                return self._try_fallback_strategies(document, rule_index)
        
        return chunks
    
    def _enhance_cached_chunks(self, cached_chunks: List[Chunk], document: Document) -> List[Chunk]:
        """Cached chunks with current document information."""
        enhanced_chunks = []
        for chunk in cached_chunks:
            enhanced_chunk = chunk.copy()
            enhanced_chunk.document_id = document.document_id
            enhanced_chunk.metadata["cached_result"] = True
            enhanced_chunks.append(enhanced_chunk)
        return enhanced_chunks
    
    def _update_rule_stats(self, rule_index: int, chunks: List[Chunk], processing_time: float, confidence: float):
        """Update rule usage statistics."""
        if rule_index < 0:
            return
        
        if rule_index in self._rule_stats:
            stats = self._rule_stats[rule_index]
            stats["usage_count"] += 1
            
            if chunks:
                stats["success_count"] += 1
            
            stats["total_processing_time"] += processing_time
            
            if chunks:
                current_avg = stats["avg_chunk_count"]
                usage_count = stats["usage_count"]
                stats["avg_chunk_count"] = (current_avg * (usage_count - 1) + len(chunks)) / usage_count
            
            stats["confidence_scores"].append(confidence)
            
            if len(stats["confidence_scores"]) > 100:
                stats["confidence_scores"] = stats["confidence_scores"][-100:]
    
    def _fallback_chunking(self, document: Document) -> List[Chunk]:
        """Fallback chunking when all strategies fail."""
        self._fallback_usage += 1
        print(f"Using default strategy as final fallback for document {document.document_id}")
        
        try:
            chunks = self.default_strategy.chunk(document)
            for chunk in chunks:
                chunk.metadata["emergency_fallback"] = True
            return chunks
        except Exception as e:
            print(f"Even default strategy failed: {e}")
            return []
    
    def _rule_matches(self, rule: RoutingRule, document: Document) -> bool:
        """Legacy rule matching method for backward compatibility."""
        confidence = self._calculate_rule_confidence(rule, document, {})
        return confidence >= self.config.min_confidence_threshold
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        return {
            "routing_decisions": self._routing_decisions,
            "cache_hits": self._cache_hits,
            "fallback_usage": self._fallback_usage,
            "rule_statistics": self._rule_stats.copy(),
            "strategy_cache_size": len(self._strategy_cache),
            "rule_cache_size": len(self._rule_cache),
            "configuration": {
                "rule_caching_enabled": self.config.enable_rule_caching,
                "strategy_caching_enabled": self.config.enable_strategy_caching,
                "content_analysis_enabled": self.config.enable_content_analysis,
                "fallback_chain_enabled": self.config.enable_fallback_chain,
                "min_confidence_threshold": self.config.min_confidence_threshold
            }
        }
    
    def clear_routing_caches(self):
        """Clear all routing caches."""
        self._rule_cache.clear()
        self._strategy_cache.clear()
        self._cache_hits = 0
    
    def optimize_rule_order(self):
        """Optimize rule order based on usage statistics."""
        if not self.config.rule_priority_optimization:
            return
        
        rule_performance = []
        for i, rule in enumerate(self.rules):
            if i in self._rule_stats:
                stats = self._rule_stats[i]
                if stats["usage_count"] > 0:
                    success_rate = stats["success_count"] / stats["usage_count"]
                    avg_time = stats["total_processing_time"] / stats["usage_count"]
                    avg_confidence = sum(stats["confidence_scores"]) / len(stats["confidence_scores"]) if stats["confidence_scores"] else 0.5
                    
                    performance_score = success_rate * 0.5 + avg_confidence * 0.3 + max(0, 1 - avg_time / 1000) * 0.2
                else:
                    performance_score = 0.1
            else:
                performance_score = 0.1
            
            rule_performance.append((performance_score, i, rule))
        
        rule_performance.sort(reverse=True)
        
        self.rules = [rule for _, _, rule in rule_performance]
        
        new_rule_stats = {}
        for new_index, (_, old_index, _) in enumerate(rule_performance):
            if old_index in self._rule_stats:
                new_rule_stats[new_index] = self._rule_stats[old_index]
        
        self._rule_stats = new_rule_stats
        
        print(f"Optimized rule order based on performance statistics")