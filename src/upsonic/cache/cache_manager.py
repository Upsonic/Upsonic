import time
import hashlib
from typing import Any, Dict, List, Optional, Literal, Union
from upsonic.models import Model
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False

CacheMethod = Literal["vector_search", "llm_call"]
CacheEntry = Dict[str, Any]


class CacheManager:
    """
    Session-level cache manager for Task objects.
    
    This class manages cache storage and retrieval for tasks within a session,
    providing both vector search and exact match capabilities.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the cache manager.
        
        Args:
            session_id: Optional session identifier for cache isolation
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self._cache_data: Dict[str, CacheEntry] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _generate_cache_id(self, input_text: str) -> str:
        """Generate a unique cache ID for the input text."""
        if input_text is None:
            input_text = ""
        return hashlib.md5(input_text.encode()).hexdigest()
    
    def _is_cache_expired(self, cache_entry: CacheEntry, duration_minutes: int) -> bool:
        """Check if a cache entry has expired based on duration."""
        if not cache_entry or "timestamp" not in cache_entry:
            return True
        
        current_time = time.time()
        cache_time = cache_entry["timestamp"]
        duration_seconds = duration_minutes * 60
        
        return (current_time - cache_time) > duration_seconds
    
    def _cleanup_expired_cache(self, duration_minutes: int):
        """Remove expired cache entries."""
        if not self._cache_data:
            return
        
        expired_keys = []
        for key, entry in self._cache_data.items():
            if self._is_cache_expired(entry, duration_minutes):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache_data[key]
    
    def _calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not _NUMPY_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="numpy",
                install_command='pip install numpy',
                feature_name="vector similarity calculation"
            )

        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    async def _find_similar_cache_entry(
        self, 
        input_text: str, 
        input_vector: List[float],
        cache_threshold: float,
        duration_minutes: int
    ) -> Optional[CacheEntry]:
        """Find the most similar cache entry using vector search."""
        if not self._cache_data:
            return None
        
        self._cleanup_expired_cache(duration_minutes)
        
        best_match = None
        best_similarity = 0.0
        
        for cache_id, cache_entry in self._cache_data.items():
            if self._is_cache_expired(cache_entry, duration_minutes):
                continue
            
            if "input_vector" not in cache_entry:
                continue
            
            similarity = self._calculate_similarity(input_vector, cache_entry["input_vector"])
            
            if similarity >= cache_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = cache_entry.copy()
                best_match["similarity"] = similarity
        
        return best_match
    
    async def _find_exact_cache_entry(
        self, 
        input_text: str, 
        duration_minutes: int,
        llm_provider: Optional[Any] = None
    ) -> Optional[CacheEntry]:
        """Find exact cache entry using batch LLM-based semantic matching."""
        if not llm_provider:
            cache_id = self._generate_cache_id(input_text)
            if cache_id in self._cache_data:
                cache_entry = self._cache_data[cache_id]
                if not self._is_cache_expired(cache_entry, duration_minutes):
                    return cache_entry
            return None
        
        self._cleanup_expired_cache(duration_minutes)
        
        valid_entries = []
        for cache_id, cache_entry in self._cache_data.items():
            if self._is_cache_expired(cache_entry, duration_minutes):
                continue
            
            cached_input = cache_entry.get("input", "")
            if not cached_input:
                continue
            
            valid_entries.append((cache_id, cache_entry, cached_input))
        
        if not valid_entries:
            return None
        
        best_match = await self._llm_batch_compare_queries(input_text, valid_entries, llm_provider)
        
        return best_match
    
    async def _llm_batch_compare_queries(
        self, 
        input_query: str, 
        valid_entries: List[tuple], 
        llm_provider: Union[Model, str]
    ) -> Optional[CacheEntry]:
        """Use LLM to find the most relevant cached entry from a batch of entries."""
        try:
            from upsonic.tasks.tasks import Task
            from upsonic.agent.agent import Agent
            
            comparison_agent = Agent(
                model=llm_provider,
                debug=False
            )
            
            cached_queries_text = ""
            for i, (cache_id, cache_entry, cached_input) in enumerate(valid_entries, 1):
                cached_queries_text += f"{i}. \"{cached_input}\"\n"
            
            comparison_prompt = f"""You are a semantic matching expert. I have a new query and several cached queries. Your task is to find the cached query that is most semantically similar to the new query.

            NEW QUERY: "{input_query}"

            CACHED QUERIES:
            {cached_queries_text}

            Instructions:
            1. Compare the new query with each cached query
            2. Find the cached query that asks for the most similar information or is semantically equivalent
            3. If none of the cached queries are semantically similar enough, respond with "NONE"
            4. If you find a match, respond with only the number (1, 2, 3, etc.) of the most similar cached query

            Respond with only the number of the most similar cached query, or "NONE" if no good match exists."""

            comparison_task = Task(
                description=comparison_prompt,
                enable_cache=False,
                response_format=str
            )
            
            result = await comparison_agent.do_async(comparison_task)
            
            if result and isinstance(result, str):
                result = result.strip().upper()
                
                if result == "NONE":
                    return None
                
                try:
                    match_index = int(result) - 1
                    if 0 <= match_index < len(valid_entries):
                        cache_id, cache_entry, cached_input = valid_entries[match_index]
                        return cache_entry
                except ValueError:
                    pass
            
            return None
            
        except Exception as e:
            warning_log(f"Batch LLM comparison failed: {e}", context="CacheManager")
            return None
    
    async def get_cached_response(
        self, 
        input_text: str,
        cache_method: CacheMethod,
        cache_threshold: float,
        duration_minutes: int,
        embedding_provider: Optional[Any] = None,
        llm_provider: Optional[Union[Model, str]] = None
    ) -> Optional[Any]:
        """
        Get cached response for the given input text.
        
        Args:
            input_text: The input text to search for in cache
            cache_method: The cache method to use ("vector_search" or "llm_call")
            cache_threshold: Similarity threshold for vector search
            duration_minutes: Cache duration in minutes
            embedding_provider: Embedding provider for vector search
            llm_provider: LLM provider for semantic comparison
            
        Returns:
            Cached response if found, None otherwise
        """
        if cache_method == "vector_search":
            if not embedding_provider:
                return None
            
            try:
                input_vector = await embedding_provider.embed_query(input_text)
                cache_entry = await self._find_similar_cache_entry(
                    input_text, input_vector, cache_threshold, duration_minutes
                )
            except Exception:
                return None
        else:
            cache_entry = await self._find_exact_cache_entry(input_text, duration_minutes, llm_provider)
        
        if cache_entry:
            self._cache_hits += 1
            return cache_entry.get("output")
        
        self._cache_misses += 1
        return None
    
    async def store_cache_entry(
        self, 
        input_text: str, 
        output: Any,
        cache_method: CacheMethod,
        embedding_provider: Optional[Any] = None
    ):
        """
        Store a new cache entry.
        
        Args:
            input_text: The input text
            output: The corresponding output
            cache_method: The cache method used
            embedding_provider: Embedding provider for vector search
        """
        cache_id = self._generate_cache_id(input_text)
        
        cache_entry = {
            "input_id": cache_id,
            "input": input_text,
            "output": output,
            "timestamp": time.time(),
            "session_id": self.session_id
        }
        
        if cache_method == "vector_search" and embedding_provider:
            try:
                input_vector = await embedding_provider.embed_query(input_text)
                cache_entry["input_vector"] = input_vector
            except Exception:
                pass
        
        self._cache_data[cache_id] = cache_entry
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache_data)
        
        return {
            "session_id": self.session_id,
            "total_entries": total_entries,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0.0
        }
    
    def clear_cache(self):
        """Clear all cache entries."""
        self._cache_data.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_size(self) -> int:
        """Get the number of cache entries."""
        return len(self._cache_data)
    
    def get_session_id(self) -> str:
        """Get the session ID."""
        return self.session_id
