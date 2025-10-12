"""
Mock components for testing vector database integrations with Knowledge Base.
"""
import asyncio
import uuid
from typing import List, Dict, Any, Union, Optional
from unittest.mock import Mock, AsyncMock

from upsonic.schemas.data_models import Document, Chunk, RAGSearchResult
from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.embeddings.base import EmbeddingProvider, EmbeddingConfig, EmbeddingMode
from upsonic.text_splitter.base import BaseChunker, BaseChunkingConfig
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import LoaderConfig


class MockEmbeddingConfig(EmbeddingConfig):
    """Mock embedding configuration."""
    model_name: str = "mock-embedding-model"
    dimensions: int = 384


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    config: MockEmbeddingConfig
    
    def __init__(self, **kwargs):
        super().__init__(config=MockEmbeddingConfig(**kwargs))
    
    async def _embed_batch(self, texts: List[str], mode: EmbeddingMode = EmbeddingMode.DOCUMENT) -> List[List[float]]:
        """Generate mock embeddings."""
        # Generate deterministic embeddings based on text content
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding
            hash_val = hash(text) % 1000
            embedding = [float((hash_val + i) % 100) / 100.0 for i in range(self.config.dimensions)]
            embeddings.append(embedding)
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return mock model information."""
        return {
            "model_name": self.config.model_name,
            "dimensions": self.config.dimensions,
            "max_tokens": 512,
            "provider": "mock"
        }
    
    @property
    def supported_modes(self) -> List[EmbeddingMode]:
        """Return supported embedding modes."""
        return [EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY]
    
    @property
    def pricing_info(self) -> Dict[str, float]:
        """Return mock pricing information."""
        return {
            "per_token": 0.0001,
            "per_request": 0.001
        }


class MockChunkingConfig(BaseChunkingConfig):
    """Mock chunking configuration."""
    chunk_size: int = 100
    chunk_overlap: int = 20


class MockChunker(BaseChunker):
    """Mock chunker for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(config=MockChunkingConfig(**kwargs))
    
    def chunk(self, content, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Split content into mock chunks."""
        if metadata is None:
            metadata = {}
        
        # Handle both string content and list of documents
        if isinstance(content, list):
            # If content is a list of documents, process each document
            all_chunks = []
            for doc in content:
                doc_chunks = self._chunk_document(doc)
                all_chunks.extend(doc_chunks)
            return all_chunks
        
        # Handle string content
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        for i in range(0, len(content), chunk_size - overlap):
            end_idx = min(i + chunk_size, len(content))
            chunk_text = content[i:end_idx]
            
            if chunk_text.strip():  # Only create non-empty chunks
                chunk = Chunk(
                    text_content=chunk_text,
                    metadata=metadata.copy(),
                    document_id=metadata.get('document_id', str(uuid.uuid4())),
                    chunk_id=str(uuid.uuid4()),
                    start_index=i,
                    end_index=end_idx
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a document into smaller pieces."""
        return self.chunk(document.content, document.metadata)


class MockLoaderConfig(LoaderConfig):
    """Mock loader configuration."""
    pass


class MockLoader(BaseLoader):
    """Mock loader for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(config=MockLoaderConfig(**kwargs))
    
    def load(self, source: Union[str, List[str]]) -> List[Document]:
        """Load mock documents."""
        if isinstance(source, str):
            sources = [source]
        else:
            sources = source
        
        documents = []
        for src in sources:
            # Generate mock content based on source
            content = f"Mock content for {src}. " * 50  # Repeat to make it longer
            doc = Document(
                content=content,
                metadata={"source": src, "type": "mock"},
                document_id=str(uuid.uuid4())
            )
            documents.append(doc)
        
        return documents
    
    async def aload(self, source: Union[str, List[str]]) -> List[Document]:
        """Async load mock documents."""
        return self.load(source)
    
    def batch(self, sources: List[str]) -> List[Document]:
        """Batch load mock documents."""
        all_docs = []
        for source in sources:
            all_docs.extend(self.load(source))
        return all_docs
    
    async def abatch(self, sources: List[str]) -> List[Document]:
        """Async batch load mock documents."""
        return self.batch(sources)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Return supported file extensions."""
        return ['.txt', '.mock']


def create_mock_vector_search_result(
    id: str = None,
    score: float = 0.8,
    text: str = "Mock search result text",
    metadata: Dict[str, Any] = None
) -> VectorSearchResult:
    """Create a mock vector search result."""
    if id is None:
        id = str(uuid.uuid4())
    if metadata is None:
        metadata = {"source": "mock", "type": "test"}
    
    return VectorSearchResult(
        id=id,
        score=score,
        payload=metadata,
        vector=[0.1] * 384,  # Mock vector
        text=text
    )


def create_mock_rag_search_result(
    text: str = "Mock RAG result text",
    metadata: Dict[str, Any] = None,
    score: float = 0.8,
    chunk_id: str = None
) -> RAGSearchResult:
    """Create a mock RAG search result."""
    if metadata is None:
        metadata = {"source": "mock", "type": "test"}
    if chunk_id is None:
        chunk_id = str(uuid.uuid4())
    
    return RAGSearchResult(
        text=text,
        metadata=metadata,
        score=score,
        chunk_id=chunk_id
    )


def create_mock_document(
    content: str = "Mock document content",
    metadata: Dict[str, Any] = None,
    document_id: str = None
) -> Document:
    """Create a mock document."""
    if metadata is None:
        metadata = {"source": "mock", "type": "test"}
    if document_id is None:
        document_id = str(uuid.uuid4())
    
    return Document(
        content=content,
        metadata=metadata,
        document_id=document_id
    )


def create_mock_chunk(
    text_content: str = "Mock chunk content",
    metadata: Dict[str, Any] = None,
    document_id: str = None,
    chunk_id: str = None
) -> Chunk:
    """Create a mock chunk."""
    if metadata is None:
        metadata = {"source": "mock", "type": "test"}
    if document_id is None:
        document_id = str(uuid.uuid4())
    if chunk_id is None:
        chunk_id = str(uuid.uuid4())
    
    return Chunk(
        text_content=text_content,
        metadata=metadata,
        document_id=document_id,
        chunk_id=chunk_id
    )
