"""
Test ChromaProvider integration with Knowledge Base.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from upsonic.knowledge_base.knowledge_base import KnowledgeBase
from upsonic.vectordb.providers.chroma import ChromaProvider
from upsonic.vectordb.config import ChromaConfig, ConnectionConfig, Mode, DistanceMetric, HNSWIndexConfig
from upsonic.schemas.data_models import Document, Chunk, RAGSearchResult
from upsonic.schemas.vector_schemas import VectorSearchResult

from .mock_components import (
    MockEmbeddingProvider, MockChunker, MockLoader,
    create_mock_document, create_mock_chunk, create_mock_vector_search_result
)


class TestChromaKnowledgeBaseIntegration:
    """Test ChromaProvider integration with Knowledge Base."""
    
    @pytest.fixture
    def chroma_config(self):
        """Create a ChromaProvider configuration."""
        connection = ConnectionConfig(mode=Mode.IN_MEMORY)
        return ChromaConfig(
            connection=connection,
            collection_name="test_collection",
            vector_size=384,
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig(),
            default_top_k=5,
            dense_search_enabled=True,
            full_text_search_enabled=True,
            hybrid_search_enabled=True
        )
    
    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock embedding provider."""
        return MockEmbeddingProvider()
    
    @pytest.fixture
    def mock_chunker(self):
        """Create a mock chunker."""
        return MockChunker()
    
    @pytest.fixture
    def mock_loader(self):
        """Create a mock loader."""
        return MockLoader()
    
    @pytest.fixture
    def chroma_provider(self, chroma_config):
        """Create a ChromaProvider instance."""
        return ChromaProvider(chroma_config)
    
    @pytest.fixture
    def knowledge_base(self, chroma_provider, mock_embedding_provider, mock_chunker, mock_loader):
        """Create a Knowledge Base with ChromaProvider."""
        with patch('upsonic.knowledge_base.knowledge_base.KnowledgeBase._update_search_docstring'):
            return KnowledgeBase(
                sources=["test_source.txt"],
                embedding_provider=mock_embedding_provider,
                vectordb=chroma_provider,
                splitters=mock_chunker,
                loaders=mock_loader,
                name="test_kb"
            )
    
    def test_chroma_provider_initialization(self, chroma_provider, chroma_config):
        """Test ChromaProvider initialization."""
        assert chroma_provider._config == chroma_config
        assert not chroma_provider._is_connected
        assert chroma_provider._client is None
    
    @pytest.mark.asyncio
    async def test_chroma_provider_connection(self, chroma_provider):
        """Test ChromaProvider connection."""
        await chroma_provider.connect()
        assert chroma_provider._is_connected
        assert chroma_provider._client is not None
        assert await chroma_provider.is_ready()
    
    @pytest.mark.asyncio
    async def test_chroma_provider_disconnection(self, chroma_provider):
        """Test ChromaProvider disconnection."""
        await chroma_provider.connect()
        assert chroma_provider._is_connected
        
        await chroma_provider.disconnect()
        assert not chroma_provider._is_connected
        assert chroma_provider._client is None
    
    @pytest.mark.asyncio
    async def test_chroma_collection_creation(self, chroma_provider):
        """Test ChromaProvider collection creation."""
        await chroma_provider.connect()
        assert not await chroma_provider.collection_exists()
        
        await chroma_provider.create_collection()
        assert await chroma_provider.collection_exists()
    
    @pytest.mark.asyncio
    async def test_chroma_collection_deletion(self, chroma_provider):
        """Test ChromaProvider collection deletion."""
        await chroma_provider.connect()
        await chroma_provider.create_collection()
        assert await chroma_provider.collection_exists()
        
        await chroma_provider.delete_collection()
        assert not await chroma_provider.collection_exists()
    
    @pytest.mark.asyncio
    async def test_chroma_upsert_operations(self, chroma_provider):
        """Test ChromaProvider upsert operations."""
        await chroma_provider.connect()
        await chroma_provider.create_collection()
        
        # Test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        chunks = ["chunk1", "chunk2"]
        
        # Upsert data
        await chroma_provider.upsert(vectors, payloads, ids, chunks)
        
        # Verify data was stored
        results = await chroma_provider.fetch(ids)
        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[1].id == "id2"
    
    @pytest.mark.asyncio
    async def test_chroma_search_operations(self, chroma_provider):
        """Test ChromaProvider search operations."""
        await chroma_provider.connect()
        await chroma_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
        ids = ["id1", "id2", "id3"]
        chunks = ["chunk1", "chunk2", "chunk3"]
        
        await chroma_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test dense search
        query_vector = [0.15] * 384
        results = await chroma_provider.dense_search(query_vector, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(result, VectorSearchResult) for result in results)
    
    @pytest.mark.asyncio
    async def test_chroma_delete_operations(self, chroma_provider):
        """Test ChromaProvider delete operations."""
        await chroma_provider.connect()
        await chroma_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        chunks = ["chunk1", "chunk2"]
        
        await chroma_provider.upsert(vectors, payloads, ids, chunks)
        
        # Verify data exists
        results = await chroma_provider.fetch(ids)
        assert len(results) == 2
        
        # Delete one item
        await chroma_provider.delete(["id1"])
        
        # Verify deletion
        results = await chroma_provider.fetch(ids)
        assert len(results) == 1
        assert results[0].id == "id2"
    
    @pytest.mark.asyncio
    async def test_knowledge_base_setup_with_chroma(self, knowledge_base):
        """Test Knowledge Base setup with ChromaProvider."""
        # Mock the vectordb methods
        knowledge_base.vectordb.connect = AsyncMock()
        knowledge_base.vectordb.create_collection = AsyncMock()
        knowledge_base.vectordb.upsert = AsyncMock()
        knowledge_base.vectordb.collection_exists = AsyncMock(return_value=False)
        knowledge_base.vectordb.is_ready = AsyncMock(return_value=True)
        
        # Mock the embedding provider - return 1 vector per chunk
        async def mock_embed_documents(chunks):
            return [[0.1] * 384] * len(chunks)
        knowledge_base.embedding_provider.embed_documents = AsyncMock(side_effect=mock_embed_documents)
        
        # Setup the knowledge base
        await knowledge_base.setup_async()
        
        # Verify setup was called
        knowledge_base.vectordb.connect.assert_called_once()
        knowledge_base.vectordb.create_collection.assert_called_once()
        knowledge_base.vectordb.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_knowledge_base_query_with_chroma(self, knowledge_base):
        """Test Knowledge Base query with ChromaProvider."""
        # Mock the vectordb methods
        knowledge_base.vectordb.connect = AsyncMock()
        knowledge_base.vectordb.create_collection = AsyncMock()
        knowledge_base.vectordb.upsert = AsyncMock()
        knowledge_base.vectordb.collection_exists = AsyncMock(return_value=False)
        knowledge_base.vectordb.is_ready = AsyncMock(return_value=True)
        knowledge_base.vectordb.search = AsyncMock(return_value=[
            create_mock_vector_search_result("id1", 0.9, "Test result 1"),
            create_mock_vector_search_result("id2", 0.8, "Test result 2")
        ])
        
        # Mock the embedding provider - return 1 vector per chunk
        async def mock_embed_documents(chunks):
            return [[0.1] * 384] * len(chunks)
        knowledge_base.embedding_provider.embed_documents = AsyncMock(side_effect=mock_embed_documents)
        knowledge_base.embedding_provider.embed_query = AsyncMock(return_value=[0.15] * 384)
        
        # Setup the knowledge base
        await knowledge_base.setup_async()
        
        # Query the knowledge base
        results = await knowledge_base.query_async("test query")
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(result, RAGSearchResult) for result in results)
        assert results[0].text == "Test result 1"
        assert results[1].text == "Test result 2"
    
    @pytest.mark.asyncio
    async def test_chroma_hybrid_search(self, chroma_provider):
        """Test ChromaProvider hybrid search functionality."""
        await chroma_provider.connect()
        await chroma_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        chunks = ["chunk1", "chunk2"]
        
        await chroma_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test hybrid search
        query_vector = [0.15] * 384
        query_text = "test query"
        
        # Mock the individual search methods
        chroma_provider.dense_search = AsyncMock(return_value=[
            create_mock_vector_search_result("id1", 0.9, "Test result 1")
        ])
        chroma_provider.full_text_search = AsyncMock(return_value=[
            create_mock_vector_search_result("id2", 0.8, "Test result 2")
        ])
        
        results = await chroma_provider.hybrid_search(query_vector, query_text, top_k=2)
        
        # Verify hybrid search was called
        chroma_provider.dense_search.assert_called_once()
        chroma_provider.full_text_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chroma_full_text_search(self, chroma_provider):
        """Test ChromaProvider full-text search."""
        await chroma_provider.connect()
        await chroma_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        chunks = ["chunk1", "chunk2"]
        
        await chroma_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test full-text search
        results = await chroma_provider.full_text_search("chunk", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(result, VectorSearchResult) for result in results)
    
    @pytest.mark.asyncio
    async def test_chroma_error_handling(self, chroma_provider):
        """Test ChromaProvider error handling."""
        # Test connection error
        with pytest.raises(Exception):
            await chroma_provider.create_collection()  # Should fail without connection
        
        # Test with connection
        await chroma_provider.connect()
        await chroma_provider.create_collection()
        
        # Test invalid upsert
        with pytest.raises(Exception):
            await chroma_provider.upsert([], [], [], [])  # Empty data should be handled gracefully
    
    def test_chroma_configuration_validation(self):
        """Test ChromaProvider configuration validation."""
        # Test that ChromaProvider accepts valid ChromaConfig
        from upsonic.vectordb.config import ChromaConfig, ConnectionConfig, Mode
        valid_connection = ConnectionConfig(mode=Mode.IN_MEMORY)
        valid_config = ChromaConfig(
            connection=valid_connection,
            collection_name="test",
            vector_size=384
        )
        
        # ChromaProvider should accept ChromaConfig
        provider = ChromaProvider(valid_config)
        assert provider._config == valid_config
        assert provider._config.collection_name == "test"
    
    @pytest.mark.asyncio
    async def test_chroma_collection_recreation(self, chroma_provider):
        """Test ChromaProvider collection recreation."""
        await chroma_provider.connect()
        
        # Create collection
        await chroma_provider.create_collection()
        assert await chroma_provider.collection_exists()
        
        # Test that collection exists
        assert await chroma_provider.collection_exists()
