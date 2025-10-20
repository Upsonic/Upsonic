from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import Field

from upsonic.schemas.data_models import Chunk, Document
from upsonic.text_splitter.base import BaseChunkingConfig, BaseChunker
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)

class JSONChunkingConfig(BaseChunkingConfig):
    """
    A specialized configuration model for the path-aware JSON Chunker strategy.

    This configuration extends the base settings with parameters that provide
    granular control over how a JSON data graph is traversed, segmented, and
    serialized.
    """
    convert_lists_to_dicts: bool = Field(
        default=True,
        description=(
            "If True, recursively converts lists into dictionary-like objects "
            "(e.g., `['a', 'b']` becomes `{'0': 'a', '1': 'b'}`). This is essential "
            "for enabling the chunker to split long lists across multiple chunks."
        )
    )
    max_depth: Optional[int] = Field(
        default=50,
        description=(
            "An optional safeguard to limit the maximum recursion depth. Any object "
            "or array beyond this depth will be treated as an atomic value and will "
            "not be split further. Default is 50 to prevent stack overflow."
        ),
        ge=0,
    )
    json_encoder_options: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "A dictionary of options to be passed directly to the `json.dumps` "
            "function when serializing the final chunks into text strings. Allows "
            "control over formatting like `indent` or `ensure_ascii`."
        )
    )

# Rebuild the model to resolve forward references
JSONChunkingConfig.model_rebuild()


class JSONChunker(BaseChunker[JSONChunkingConfig]):
    """
    A structure-aware chunker for JSON documents that segments data recursively.

    This chunker operates on the parsed JSON data, not the raw string. It
    traverses the JSON graph and assembles chunks that are valid, self-contained
    JSON objects. Instead of character indices, it provides "path-aware"
    traceability by adding the JSON paths of the included data to each chunk's
    metadata.
    """

    def __init__(self, config: Optional[JSONChunkingConfig] = None):
        """Initializes the chunker with a specific or default configuration."""
        super().__init__(config or JSONChunkingConfig())

    def _chunk_document(self, document: Document) -> List[Chunk]:
        """
        The core implementation for splitting a single JSON document.

        Args:
            document: The document containing the raw JSON content string.

        Returns:
            A list of `Chunk` objects, where each chunk's content is a valid
            JSON string, and its metadata contains the paths of its data.
        """
        try:
            json_data = json.loads(document.content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for document {document.document_id}: {e}. Falling back to text chunking.")
            return self._fallback_to_text_chunking(document)

        min_chunk_size = self._get_effective_min_chunk_size()

        processed_data = json_data
        if self.config.convert_lists_to_dicts:
            processed_data = self._preprocess_lists(processed_data)

        chunk_builders: List[Dict[str, Any]] = [{"content": {}, "paths": []}]
        
        self._recursive_walk(
            data=processed_data,
            current_path=[],
            chunk_builders=chunk_builders,
            depth=0,
            min_chunk_size=min_chunk_size
        )
        final_chunks: List[Chunk] = []
        for builder in chunk_builders:
            if not builder["content"]:
                continue

            text_content = json.dumps(builder["content"], **self.config.json_encoder_options)
            
            final_chunks.append(
                self._create_chunk(
                    parent_document=document,
                    text_content=text_content,
                    start_index=None,
                    end_index=None,
                    extra_metadata={"chunk_json_paths": builder["paths"]},
                )
            )

        return final_chunks

    def _recursive_walk(
        self,
        data: Any,
        current_path: List[str],
        chunk_builders: List[Dict[str, Any]],
        depth: int,
        min_chunk_size: int,
    ):
        if self.config.max_depth is not None and depth >= self.config.max_depth:
            self._add_to_chunk(data, current_path, chunk_builders, min_chunk_size)
            return

        if isinstance(data, dict):
            for key, value in data.items():
                new_path = current_path + [str(key)]
                self._recursive_walk(value, new_path, chunk_builders, depth + 1, min_chunk_size)
        else:
            self._add_to_chunk(data, current_path, chunk_builders, min_chunk_size)

    def _add_to_chunk(
        self,
        value: Any,
        path: List[str],
        chunk_builders: List[Dict[str, Any]],
        min_chunk_size: int,
    ):
        item_to_add = {}
        self._set_nested_dict(item_to_add, path, value)
        item_size = self._json_size(item_to_add)

        current_chunk_builder = chunk_builders[-1]
        current_chunk_size = self._json_size(current_chunk_builder["content"])
        
        if current_chunk_size + item_size > self.config.chunk_size:
            if current_chunk_size >= min_chunk_size:
                chunk_builders.append({"content": {}, "paths": []})
                current_chunk_builder = chunk_builders[-1]

        self._set_nested_dict(current_chunk_builder["content"], path, value)
        current_chunk_builder["paths"].append(".".join(path))


    def _json_size(self, data: Dict) -> int:
        return len(json.dumps(data, **self.config.json_encoder_options))

    def _preprocess_lists(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._preprocess_lists(v) for k, v in data.items()}
        if isinstance(data, list):
            return {str(i): self._preprocess_lists(item) for i, item in enumerate(data)}
        return data

    def _set_nested_dict(self, d: Dict, path: List[str], value: Any):
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _fallback_to_text_chunking(self, document: Document) -> List[Chunk]:
        from upsonic.text_splitter.recursive import RecursiveChunker, RecursiveChunkingConfig
        
        fallback_config = RecursiveChunkingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        fallback_chunker = RecursiveChunker(fallback_config)
        
        chunks = fallback_chunker.chunk([document])
        
        for chunk in chunks:
            chunk.metadata["chunking_fallback"] = "json_to_text"
            chunk.metadata["original_strategy"] = "json"
        
        return chunks