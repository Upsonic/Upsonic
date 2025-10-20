from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field, ConfigDict
from upsonic.schemas.data_models import Chunk, Document
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)

class BaseChunkingConfig(BaseModel):
    """
    The base configuration for all chunking strategies.

    This Pydantic model defines the core parameters that are universally
    applicable to most chunking methods. Specific chunkers can extend this
    class to add their own unique configuration options.
    """
    chunk_size: int = Field(
        default=1024,
        description="The target size of each chunk.",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=200,
        description="The number of overlapping units (e.g., characters or tokens) between consecutive chunks to maintain context.",
        ge=0,
    )
    min_chunk_size: Optional[int] = Field(
        default=None,
        description=(
            "The minimum size for a chunk to be considered 'full'. This prevents "
            "the creation of numerous tiny, fragmented chunks. If None, it will be "
            "derived from `chunk_size` (e.g., 80% of it) in the chunker logic."
        ),
        ge=0,
    )
    length_function: Callable[[str], int] = Field(
        default=len,
        description="The function used to measure the length of a text segment. Defaults to character count (`len`). Can be replaced with a token counter.",
        exclude=True,
    )
    strip_whitespace: bool = Field(
        default=False,
        description="If True, strips leading and trailing whitespace from each chunk's content. Default is False to preserve content integrity.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


ConfigType = TypeVar("ConfigType", bound=BaseChunkingConfig)




class BaseChunker(ABC, Generic[ConfigType]):
    """
    Abstract Base Class for all chunking strategies.

    This class defines the unified interface for splitting a list of `Document`
    objects into a list of `Chunk` objects. It provides a robust, standardized

    foundation with both synchronous and asynchronous methods, optimized batch
    processing, and helper utilities to ensure consistency and correctness.

    To create a new chunking strategy, inherit from this class and implement
    the abstract `_chunk_document` method.
    """

    def __init__(self, config: Optional[ConfigType] = None):
        """
        Initializes the chunker with a given configuration.

        Args:
            config: A Pydantic model instance inheriting from `BaseChunkingConfig`.
                    If None, a default configuration will be instantiated.
        """

        self.config: ConfigType = config if config is not None else self._get_default_config()

        if self.config.chunk_overlap >= self.config.chunk_size:
            logger.warning(
                f"Chunk overlap ({self.config.chunk_overlap}) is greater than or equal to chunk size ({self.config.chunk_size}). This may result in infinite loops or unexpected behavior."
            )

    def _get_effective_min_chunk_size(self) -> int:
        """
        Get the effective minimum chunk size, deriving from chunk_size if not explicitly set.
        
        Returns:
            The minimum chunk size to use for chunking decisions.
        """
        if self.config.min_chunk_size is not None:
            return self.config.min_chunk_size
        
        return max(int(self.config.chunk_size * 0.8), 50)

    @classmethod
    def _get_default_config(cls) -> ConfigType:
        """
        A helper to get the default config. This is designed to be overridden
        by subclasses if their config has required fields.
        """

        config_class = cls.__orig_bases__[0].__args__[0]
        return config_class()



    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """
        Splits a list of documents into chunks (synchronous).

        This method iterates through the documents and applies the core
        chunking logic defined in `_chunk_document`.

        Args:
            documents: A list of `Document` objects to be chunked.

        Returns:
            A single list containing all `Chunk` objects from all documents.
        """
        all_chunks: List[Chunk] = []
        for doc in documents:
            try:
                if not doc.content or not doc.content.strip():
                    logger.info(f"Document {doc.document_id} has empty content, skipping chunking")
                    continue
                
                chunks = self._chunk_document(doc)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error chunking document {doc.document_id}: {e}")
                error_chunk = self._create_chunk(
                    parent_document=doc,
                    text_content=f"[CHUNKING ERROR: {str(e)}]",
                    start_index=0,
                    end_index=len(doc.content),
                    extra_metadata={"chunking_error": True, "error_message": str(e)}
                )
                all_chunks.append(error_chunk)
        
        return all_chunks

    async def achunk(self, documents: List[Document]) -> List[Chunk]:
        """
        Splits a list of documents into chunks (asynchronous).

        This method iterates through the documents and applies the async
        chunking logic defined in `_achunk_document`.

        Args:
            documents: A list of `Document` objects to be chunked.

        Returns:
            A single list containing all `Chunk` objects from all documents.
        """
        all_chunks: List[Chunk] = []
        for doc in documents:
            try:
                if not doc.content or not doc.content.strip():
                    logger.info(f"Document {doc.document_id} has empty content, skipping chunking")
                    continue
                
                chunks = await self._achunk_document(doc)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error chunking document {doc.document_id}: {e}")
                error_chunk = self._create_chunk(
                    parent_document=doc,
                    text_content=f"[CHUNKING ERROR: {str(e)}]",
                    start_index=0,
                    end_index=len(doc.content),
                    extra_metadata={"chunking_error": True, "error_message": str(e)}
                )
                all_chunks.append(error_chunk)
        
        return all_chunks

    def batch(self, documents: List[Document], **kwargs: Any) -> List[Chunk]:
        """
        Alias for the `chunk` method for API consistency.
        """
        return self.chunk(documents)

    async def abatch(self, documents: List[Document], **kwargs: Any) -> List[Chunk]:
        """
        Splits a list of documents into chunks concurrently (asynchronous).

        This is the most performant method for processing multiple documents.
        It leverages `asyncio.gather` to run the chunking operation for each
        document in parallel.

        Args:
            documents: A list of `Document` objects to be chunked.

        Returns:
            A single list containing all `Chunk` objects from all documents.
        """
        if not documents:
            return []

        tasks: List[Coroutine[Any, Any, List[Chunk]]] = [
            self._achunk_document(doc) for doc in documents
        ]
        list_of_chunk_lists = await asyncio.gather(*tasks)

        return [chunk for sublist in list_of_chunk_lists for chunk in sublist]


    @abstractmethod
    def _chunk_document(self, document: Document) -> List[Chunk]:
        """
        The core chunking logic for a single document (synchronous).


        Args:
            document: The `Document` to be chunked.

        Returns:
            A list of `Chunk` objects derived from the document.
        """
        raise NotImplementedError("Subclasses must implement the _chunk_document method.")

    async def _achunk_document(self, document: Document) -> List[Chunk]:
        """
        The core chunking logic for a single document (asynchronous).

        By default, this method wraps the synchronous `_chunk_document` method,
        making all chunkers async-compatible out of the box. Subclasses that
        perform I/O operations (e.g., AgenticChunker calling an API)
        should override this method for true non-blocking execution.

        Args:
            document: The `Document` to be chunked.

        Returns:
            A list of `Chunk` objects derived from the document.
        """

        return self._chunk_document(document)


    def _create_chunk(
        self,
        parent_document: Document,
        text_content: str,
        start_index: int,
        end_index: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        A robust factory for creating `Chunk` objects.

        This helper ensures that all required fields are correctly populated,
        including inheriting metadata and IDs from the parent document.

        Args:
            parent_document: The source `Document` object.
            text_content: The text content for this specific chunk.
            start_index: The starting character index of the chunk within the original document's content.
            end_index: The ending character index of the chunk.
            extra_metadata: An optional dictionary of chunk-specific metadata to add.

        Returns:
            A fully-formed `Chunk` object.
        """
        final_text = text_content.strip() if self.config.strip_whitespace else text_content
        
        chunk_metadata = copy.deepcopy(parent_document.metadata)
        
        if extra_metadata:
            chunk_metadata.update(extra_metadata)
            
        return Chunk(
            text_content=final_text,
            metadata=chunk_metadata,
            document_id=parent_document.document_id,
            start_index=start_index,
            end_index=end_index,
        )