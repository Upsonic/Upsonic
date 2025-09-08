from __future__ import annotations
from typing import Any, List, Dict, Optional
import json
import time
from pydantic import Field, field_validator

from upsonic.text_splitter.base import ChunkingStrategy, ChunkingConfig
from upsonic.schemas.data_models import Document, Chunk


class JSONChunkingConfig(ChunkingConfig):
    """Enhanced configuration for JSON chunking strategy."""
    chunk_size: int = Field(4000, description="Target chunk size for JSON chunks in characters")
    preserve_structure: bool = Field(True, description="Preserve JSON structure where possible")
    include_path_metadata: bool = Field(True, description="Include JSON path in metadata")
    
    array_handling: str = Field("split", description="How to handle arrays: split, keep, or flatten")
    object_handling: str = Field("recursive", description="How to handle objects: recursive or flatten")
    null_handling: str = Field("keep", description="How to handle null values: keep, remove, or placeholder")
    
    enable_schema_detection: bool = Field(True, description="Detect and utilize JSON schema")
    validate_json_structure: bool = Field(True, description="Validate JSON structure before processing")
    enable_type_preservation: bool = Field(True, description="Preserve JSON data types in metadata")
    
    enable_structure_caching: bool = Field(True, description="Cache JSON structure analysis")
    lazy_parsing: bool = Field(False, description="Parse JSON incrementally for large files")
    batch_processing: bool = Field(True, description="Process large JSON in batches")
    
    min_object_size: int = Field(50, description="Minimum size for standalone JSON objects")
    max_nesting_depth: int = Field(10, description="Maximum allowed nesting depth")
    enable_content_deduplication: bool = Field(True, description="Remove duplicate JSON content")
    
    include_schema_info: bool = Field(True, description="Include schema information in metadata")
    include_statistics: bool = Field(True, description="Include JSON statistics in metadata")
    include_structure_info: bool = Field(True, description="Include structure information")

    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        """Validate that chunk_overlap is less than chunk_size."""
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError(
                f"Chunk overlap ({v}) must be smaller than chunk size "
                f"({info.data['chunk_size']})."
            )
        return v

class JSONChunkingStrategy(ChunkingStrategy):
    """
    JSON chunking strategy with framework-level features.

    This specialized strategy deconstructs large JSON objects into smaller,
    valid JSON chunks while preserving structural context.
    
    Features:
    - JSON structure analysis and schema detection
    - Array and object handling strategies
    - Type preservation and validation
    - Structure caching and performance optimization
    - Content deduplication and quality controls
    - Rich metadata with schema and statistics
    - Batch processing for large JSON files
    - Multiple serialization strategies
    
    This strategy provides:
    - Schema-aware chunking that respects JSON semantics
    - Path-based navigation and reconstruction
    - Type-safe chunk validation
    - Performance optimization for large JSON structures
    - Comprehensive metadata for retrieval
    
    """
    def __init__(self, config: Optional[JSONChunkingConfig] = None):
        """
        Initialize JSON chunking strategy.

        Args:
            config: Configuration object with all settings
        """
        if config is None:
            config = JSONChunkingConfig()
        
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError(
                f"Chunk overlap ({config.chunk_overlap}) must be smaller than chunk size "
                f"({config.chunk_size})."
            )
        
        super().__init__(config)
        
        self.max_chunk_size = self.config.chunk_size
        
        self._structure_cache: Dict[str, Dict] = {}
        self._schema_cache: Dict[str, Dict] = {}
        self._parsing_stats = {
            "total_objects": 0,
            "total_arrays": 0,
            "max_depth": 0,
            "total_keys": 0,
            "unique_keys": set()
        }

    def chunk(self, document: Document) -> List[Chunk]:
        """
        JSON parsing and chunking with framework features.

        Args:
            document: The Document object containing the raw JSON string content

        Returns:
            A list of Chunk objects with JSON metadata and structure preservation
        """
        start_time = time.time()
        
        if not document.content.strip():
            return []
        
        if self._should_cache_result(document):
            cache_key = self._get_cache_key(document)
            if cache_key in self._cache:
                return self._cache[cache_key].copy()
        
        try:
            json_data = self._parse_and_validate_json(document.content)
            if json_data is None:
                return self._fallback_text_chunking(document)
            
            structure_info = self._analyze_json_structure(json_data) if self.config.enable_schema_detection else {}
            
            json_chunks_as_dicts = self._enhanced_recursive_split(json_data, structure_info)
            
            final_chunks = self._create_enhanced_json_chunks(json_chunks_as_dicts, document, structure_info)
            
            if self.config.enable_content_deduplication:
                final_chunks = self._deduplicate_chunks(final_chunks)
            
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(final_chunks, processing_time, document)
            
            if self._should_cache_result(document):
                cache_key = self._get_cache_key(document)
                self._cache[cache_key] = final_chunks.copy()
            
            return final_chunks
            
        except Exception as e:
            print(f"JSON chunking failed for document {document.document_id}: {e}")
            return self._fallback_text_chunking(document)

    def _parse_and_validate_json(self, content: str) -> Any:
        """Parse and validate JSON content with error handling."""
        try:
            json_data = json.loads(content)
            
            if self.config.validate_json_structure:
                try:
                    self._validate_structure(json_data)
                except ValueError as e:
                    print(f"Warning: JSON validation failed: {e}. Proceeding with parsed data.")
            
            return json_data
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON: {e}. Returning None.")
            return None
        except Exception as e:
            print(f"Warning: JSON processing failed: {e}. Returning None.")
            return None
    
    def _validate_structure(self, data: Any, depth: int = 0):
        """Validate JSON structure against configuration limits."""
        if depth > self.config.max_nesting_depth:
            raise ValueError(f"JSON nesting depth {depth} exceeds maximum {self.config.max_nesting_depth}")
        
        if isinstance(data, dict):
            self._parsing_stats["total_objects"] += 1
            self._parsing_stats["total_keys"] += len(data)
            self._parsing_stats["unique_keys"].update(data.keys())
            for value in data.values():
                self._validate_structure(value, depth + 1)
        elif isinstance(data, list):
            self._parsing_stats["total_arrays"] += 1
            for item in data:
                self._validate_structure(item, depth + 1)
        
        self._parsing_stats["max_depth"] = max(self._parsing_stats["max_depth"], depth)
    
    def _analyze_json_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure for intelligent chunking."""
        structure_info = {
            "type": type(data).__name__,
            "size_estimate": len(json.dumps(data, separators=(",", ":"))),
            "complexity": self._calculate_complexity(data),
            "schema": self._detect_schema(data),
            "statistics": self._parsing_stats.copy()
        }
        
        return structure_info
    
    def _calculate_complexity(self, data: Any, depth: int = 0) -> float:
        """Calculate complexity score for JSON structure."""
        if depth > 5:
            return 1.0
        
        if isinstance(data, dict):
            if not data:
                return 0.1
            return 1 + sum(self._calculate_complexity(v, depth + 1) for v in data.values()) / len(data)
        elif isinstance(data, list):
            if not data:
                return 0.1
            return 0.5 + sum(self._calculate_complexity(item, depth + 1) for item in data[:5]) / min(len(data), 5)
        else:
            return 0.1
    
    def _detect_schema(self, data: Any) -> Dict[str, Any]:
        """Detect JSON schema for better chunking decisions."""
        if isinstance(data, dict):
            return {
                "type": "object",
                "properties": {k: self._detect_schema(v) for k, v in data.items()},
                "required": list(data.keys())
            }
        elif isinstance(data, list):
            if data:
                sample_schemas = [self._detect_schema(item) for item in data[:3]]
                return {
                    "type": "array",
                    "items": sample_schemas[0] if len(set(str(s) for s in sample_schemas)) == 1 else {"type": "mixed"}
                }
            return {"type": "array", "items": {"type": "unknown"}}
        else:
            return {"type": type(data).__name__.lower()}
    
    def _enhanced_recursive_split(self, data: Any, structure_info: Dict, path: str = "root") -> List[Dict[str, Any]]:
        """Recursive splitting with structure awareness."""
        serialized_data = json.dumps(data, separators=(",", ":"))
        
        if len(serialized_data) <= self.config.chunk_size:
            return self._create_base_chunk(data, path, structure_info)
        
        if isinstance(data, dict):
            return self._split_object_enhanced(data, path, structure_info)
        elif isinstance(data, list):
            return self._split_array_enhanced(data, path, structure_info)
        else:
            return self._create_base_chunk(data, path, structure_info)
    
    def _create_base_chunk(self, data: Any, path: str, structure_info: Dict) -> List[Dict[str, Any]]:
        """Create a base chunk with enhanced metadata."""
        if isinstance(data, dict):
            data_with_meta = data.copy()
            data_with_meta["json_path"] = path
            if self.config.include_structure_info:
                data_with_meta["_structure_info"] = {
                    "type": "object",
                    "keys": list(data.keys()),
                    "size": len(json.dumps(data))
                }
            return [data_with_meta]
        else:
            chunk = {"content": data, "json_path": path}
            if self.config.include_structure_info:
                chunk["_structure_info"] = {
                    "type": type(data).__name__,
                    "size": len(json.dumps(data))
                }
            return [chunk]
    
    def _split_object_enhanced(self, obj: Dict, path: str, structure_info: Dict) -> List[Dict[str, Any]]:
        """Enhanced object splitting with intelligent key grouping."""
        chunks = []
        current_chunk = {}
        
        sorted_keys = self._prioritize_object_keys(obj, structure_info)
        
        for key in sorted_keys:
            value = obj[key]
            new_path = f"{path}.{key}"
            
            test_chunk = {**current_chunk, key: value}
            formatted_size = len(json.dumps(test_chunk, indent=2, separators=(",", ": ")))
            if formatted_size > self.config.chunk_size * 0.1:
                if current_chunk:
                    current_chunk["json_path"] = path
                    chunks.append(current_chunk)
                current_chunk = {}
                
                if len(json.dumps(value, indent=2, separators=(",", ": "))) > self.config.chunk_size * 0.1:
                    value_chunks = self._enhanced_recursive_split(value, structure_info, new_path)
                    chunks.extend(value_chunks)
                else:
                    current_chunk[key] = value
            else:
                current_chunk[key] = value
        
        if current_chunk:
            current_chunk["json_path"] = path
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_array_enhanced(self, arr: List, path: str, structure_info: Dict) -> List[Dict[str, Any]]:
        """Enhanced array splitting based on configuration."""
        if self.config.array_handling == "keep":
            return self._create_base_chunk(arr, path, structure_info)
        
        elif self.config.array_handling == "flatten":
            chunks = []
            for i, item in enumerate(arr):
                new_path = f"{path}[{i}]"
                item_chunks = self._enhanced_recursive_split(item, structure_info, new_path)
                chunks.extend(item_chunks)
            return chunks
        
        else:
            chunks = []
            current_chunk = []
            
            for i, item in enumerate(arr):
                new_path = f"{path}[{i}]"
                test_chunk = current_chunk + [item]
                
                formatted_size = len(json.dumps(test_chunk, indent=2, separators=(",", ": ")))
                if formatted_size > self.config.chunk_size * 0.1:
                    if current_chunk:
                        chunks.append({"content": current_chunk, "json_path": f"{path}[...]"})
                    current_chunk = []
                    
                    if len(json.dumps(item, indent=2, separators=(",", ":"))) > self.config.chunk_size * 0.1:
                        item_chunks = self._enhanced_recursive_split(item, structure_info, new_path)
                        chunks.extend(item_chunks)
                    else:
                        current_chunk = [item]
                else:
                    current_chunk.append(item)
            
            if current_chunk:
                chunks.append({"content": current_chunk, "json_path": f"{path}[...]"})
            
            return chunks
    
    def _prioritize_object_keys(self, obj: Dict, structure_info: Dict) -> List[str]:
        """Prioritize object keys for optimal chunking."""
        keys = list(obj.keys())
        
        keys.sort(key=lambda k: (len(str(obj[k])), k))
        
        return keys
    
    def _create_enhanced_json_chunks(self, json_chunks: List[Dict], document: Document, structure_info: Dict) -> List[Chunk]:
        """Create enhanced chunks with rich JSON metadata."""
        final_chunks = []
        
        for i, chunk_dict in enumerate(json_chunks):
            json_path = chunk_dict.pop("json_path", "root")
            chunk_structure_info = chunk_dict.pop("_structure_info", {})
            
            chunk_content = json.dumps(chunk_dict, indent=2, separators=(",", ": "), ensure_ascii=False)
            
            if len(chunk_content) > self.config.chunk_size * 0.5:
                chunk_parts = self._handle_oversized_chunk(chunk_content)
                for j, part in enumerate(chunk_parts):
                    chunk = self._create_chunk(
                        text_content=part,
                        document=document,
                        chunk_index=len(final_chunks),
                        total_chunks=len(json_chunks),
                        start_pos=0,
                        end_pos=len(part)
                    )
                    
                    if self.config.include_path_metadata:
                        chunk.metadata["json_path"] = f"{json_path}_part{j+1}"
                        chunk.metadata["json_depth"] = json_path.count('.') + json_path.count('[')
                    
                    if self.config.include_schema_info and structure_info:
                        chunk.metadata["json_schema"] = structure_info.get("schema", {})
                        chunk.metadata["json_complexity"] = structure_info.get("complexity", 0)
                    
                    if self.config.include_statistics:
                        chunk.metadata["json_statistics"] = {
                            "chunk_size": len(part),
                            "original_size": structure_info.get("size_estimate", 0),
                            "compression_ratio": len(part) / max(structure_info.get("size_estimate", 1), 1)
                        }
                    
                    if self.config.enable_type_preservation:
                        chunk.metadata["json_types"] = self._extract_type_info(chunk_dict)
                    
                    chunk.metadata.update(chunk_structure_info)
                    
                    chunk.metadata["chunking_method"] = "json_structure_aware"
                    chunk.metadata["json_processed"] = True
                    
                    final_chunks.append(chunk)
            else:
                chunk = self._create_chunk(
                    text_content=chunk_content,
                    document=document,
                    chunk_index=i,
                    total_chunks=len(json_chunks),
                    start_pos=0,
                    end_pos=len(chunk_content)
                )
                
                if self.config.include_path_metadata:
                    chunk.metadata["json_path"] = json_path
                    chunk.metadata["json_depth"] = json_path.count('.') + json_path.count('[')
                
                if self.config.include_schema_info and structure_info:
                    chunk.metadata["json_schema"] = structure_info.get("schema", {})
                    chunk.metadata["json_complexity"] = structure_info.get("complexity", 0)
                
                if self.config.include_statistics:
                    chunk.metadata["json_statistics"] = {
                        "chunk_size": len(chunk_content),
                        "original_size": structure_info.get("size_estimate", 0),
                        "compression_ratio": len(chunk_content) / max(structure_info.get("size_estimate", 1), 1)
                    }
                
                if self.config.enable_type_preservation:
                    chunk.metadata["json_types"] = self._extract_type_info(chunk_dict)
                
                chunk.metadata.update(chunk_structure_info)
                
                chunk.metadata["chunking_method"] = "json_structure_aware"
                chunk.metadata["json_processed"] = True
                
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _extract_type_info(self, data: Any) -> Dict[str, Any]:
        """Extract type information from JSON data."""
        try:
            if isinstance(data, dict):
                return {k: type(v).__name__ for k, v in data.items()}
            elif isinstance(data, list):
                return {"array_types": [type(item).__name__ for item in data[:5]]}
            else:
                return {"type": type(data).__name__}
        except Exception:
            return {"type": "unknown"}
    
    def _deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove duplicate JSON chunks."""
        seen_content = set()
        deduplicated = []
        
        for chunk in chunks:
            content_hash = hash(chunk.text_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated.append(chunk)
            else:
                print(f"Removed duplicate JSON chunk at path {chunk.metadata.get('json_path', 'unknown')}")
        
        return deduplicated
    
    def _fallback_text_chunking(self, document: Document) -> List[Chunk]:
        """Fallback to text chunking when JSON processing fails."""
        print(f"Falling back to text chunking for document {document.document_id}")
        
        try:
            from .recursive import RecursiveCharacterChunkingStrategy, RecursiveChunkingConfig
            fallback_config = RecursiveChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            fallback_strategy = RecursiveCharacterChunkingStrategy(config=fallback_config)
            
            fallback_chunks = fallback_strategy.chunk(document)
            
            for chunk in fallback_chunks:
                chunk.metadata["json_fallback"] = True
                chunk.metadata["chunking_method"] = "text_fallback"
            
            return fallback_chunks
            
        except Exception as e:
            print(f"Fallback text chunking also failed: {e}")
            chunk = self._create_chunk(
                text_content=document.content,
                document=document,
                chunk_index=0,
                total_chunks=1,
                start_pos=0,
                end_pos=len(document.content)
            )
            chunk.metadata["json_fallback"] = True
            chunk.metadata["chunking_method"] = "simple_fallback"
            return [chunk]
    
    def get_json_stats(self) -> Dict[str, Any]:
        """Get JSON processing statistics."""
        return {
            "parsing_statistics": self._parsing_stats,
            "structure_cache_size": len(self._structure_cache),
            "schema_cache_size": len(self._schema_cache),
            "configuration": {
                "chunk_size": self.config.chunk_size,
                "preserve_structure": self.config.preserve_structure,
                "array_handling": self.config.array_handling,
                "object_handling": self.config.object_handling,
                "schema_detection_enabled": self.config.enable_schema_detection,
                "structure_caching_enabled": self.config.enable_structure_caching
            }
        }
    
    def clear_json_caches(self):
        """Clear JSON processing caches."""
        self._structure_cache.clear()
        self._schema_cache.clear()
    
    def _recursive_split(self, data: Any, path: str = "root") -> List[Dict[str, Any]]:
        """Legacy recursive split method for backward compatibility."""
        return self._enhanced_recursive_split(data, {}, path)