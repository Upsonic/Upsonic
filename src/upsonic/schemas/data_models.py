from __future__ import annotations
import hashlib
import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class Document(BaseModel):
    """
    Represents a single, discrete source of information, such as a file or a web page,
    before it is processed into smaller chunks.

    This model serves as the standardized input for the chunking process. It ensures
    that every piece of knowledge entering the pipeline is tagged with its origin
    and can be uniquely identified.
    """
    content: str = Field(
        ...,
        description="The full, raw text content extracted from the source."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flexible dictionary to store metadata about the source, e.g., {'source': 'resume1.pdf', 'author': 'John Doe'}."
    )
    document_id: str = Field(
        ...,
        description="A unique, deterministic identifier for the source, typically an MD5 hash of its absolute path or URL."
    )
    doc_content_hash: str = Field(
        default="",
        description="MD5 hash of the document's full content, used for change detection and deduplication."
    )

class Chunk(BaseModel):
    """
    Represents a single, embeddable piece of a parent Document.

    This is the atomic unit that will be converted into a vector and stored in the
    vector database. It critically maintains a link to its parent Document and inherits
    its metadata, which can be augmented with chunk-specific details like page numbers.
    This rich, inherited metadata is the key to enabling advanced retrieval strategies,
    such as filtering by source or providing citations.
    """
    text_content: str = Field(
        ...,
        description="The actual text content of this specific chunk."
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Metadata inherited from the parent Document and potentially augmented with chunk-specific info, e.g., {'source': 'resume1.pdf', 'page_number': 3}."
    )
    document_id: str = Field(
        ...,
        description="The parent document's unique identifier, linking this chunk back to its source."
    )
    doc_content_hash: str = Field(
        default="",
        description="MD5 hash of the parent document's full content, inherited for change detection."
    )
    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="A unique identifier for this specific chunk."
    )
    chunk_content_hash: str = Field(
        default="",
        description="MD5 hash of this chunk's text_content, used for chunk-level deduplication and integrity."
    )

    start_index: Optional[int] = Field(
        default=None,
        description="The start index of the chunk in the original document."
    )

    end_index: Optional[int] = Field(
        default=None,
        description="The end index of the chunk in the original document."
    )



class RAGSearchResult(BaseModel):
    """
    Represents a single search result from a RAG query with both text content and metadata.
    
    This model provides a structured way to return search results that include
    both the retrieved text content and its associated metadata, enabling
    better context formatting and citation capabilities.
    """
    text: str = Field(
        ...,
        description="The text content retrieved from the search."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata associated with this search result, including source information, scores, etc."
    )
    score: Optional[float] = Field(
        default=None,
        description="The similarity score for this search result."
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="The unique identifier for this chunk."
    )