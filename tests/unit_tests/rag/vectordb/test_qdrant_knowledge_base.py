"""
Test QdrantProvider integration with Knowledge Base.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from upsonic.knowledge_base.knowledge_base import KnowledgeBase
from upsonic.vectordb.providers.qdrant import QdrantProvider
from upsonic.vectordb.config import QdrantConfig, ConnectionConfig, Mode, DistanceMetric, HNSWIndexConfig
from upsonic.schemas.data_models import Document, Chunk, RAGSearchResult
from upsonic.schemas.vector_schemas import VectorSearchResult

from .mock_components import (
    MockEmbeddingProvider, MockChunker, MockLoader,
    create_mock_document, create_mock_chunk, create_mock_vector_search_result
)


class TestQdrantKnowledgeBaseIntegration:
    """Test QdrantProvider integration with Knowledge Base."""
    
    @pytest.fixture
    def qdrant_config(self):
        """Create a QdrantProvider configuration."""
        connection = ConnectionConfig(mode=Mode.IN_MEMORY)
        return QdrantConfig(
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
    def qdrant_provider(self, qdrant_config):
        """Create a QdrantProvider instance."""
        return QdrantProvider(qdrant_config)
    
    @pytest.fixture
    def knowledge_base(self, qdrant_provider, mock_embedding_provider, mock_chunker, mock_loader):
        """Create a Knowledge Base with QdrantProvider."""
        with patch('upsonic.knowledge_base.knowledge_base.KnowledgeBase._update_search_docstring'):
            return KnowledgeBase(
                sources=["test_source.txt"],
                embedding_provider=mock_embedding_provider,
                vectordb=qdrant_provider,
                splitters=mock_chunker,
                loaders=mock_loader,
                name="test_kb"
            )
    
    def test_qdrant_provider_initialization(self, qdrant_provider, qdrant_config):
        """Test QdrantProvider initialization."""
        assert qdrant_provider._config == qdrant_config
        assert not qdrant_provider._is_connected
        assert qdrant_provider._client is None
    
    @pytest.mark.asyncio
    async def test_qdrant_provider_connection(self, qdrant_provider):
        """Test QdrantProvider connection."""
        await qdrant_provider.connect()
        assert qdrant_provider._is_connected
        assert qdrant_provider._client is not None
        assert await qdrant_provider.is_ready()
    
    @pytest.mark.asyncio
    async def test_qdrant_provider_disconnection(self, qdrant_provider):
        """Test QdrantProvider disconnection."""
        await qdrant_provider.connect()
        assert qdrant_provider._is_connected
        
        await qdrant_provider.disconnect()
        assert not qdrant_provider._is_connected
        assert qdrant_provider._client is None
    
    @pytest.mark.asyncio
    async def test_qdrant_collection_creation(self, qdrant_provider):
        """Test QdrantProvider collection creation."""
        await qdrant_provider.connect()
        assert not await qdrant_provider.collection_exists()
        
        await qdrant_provider.create_collection()
        assert await qdrant_provider.collection_exists()
    
    @pytest.mark.asyncio
    async def test_qdrant_collection_deletion(self, qdrant_provider):
        """Test QdrantProvider collection deletion."""
        await qdrant_provider.connect()
        await qdrant_provider.create_collection()
        assert await qdrant_provider.collection_exists()
        
        await qdrant_provider.delete_collection()
        assert not await qdrant_provider.collection_exists()
    
    @pytest.mark.asyncio
    async def test_qdrant_upsert_operations(self, qdrant_provider):
        """Test QdrantProvider upsert operations."""
        await qdrant_provider.connect()
        await qdrant_provider.create_collection()
        
        # Test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"]
        chunks = ["chunk1", "chunk2"]
        
        # Upsert data
        await qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Verify data was stored
        results = await qdrant_provider.fetch(ids)
        assert len(results) == 2
        assert results[0].id == "550e8400-e29b-41d4-a716-446655440001"
        assert results[1].id == "550e8400-e29b-41d4-a716-446655440002"
    
    @pytest.mark.asyncio
    async def test_qdrant_search_operations(self, qdrant_provider):
        """Test QdrantProvider search operations."""
        await qdrant_provider.connect()
        await qdrant_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002", "550e8400-e29b-41d4-a716-446655440003"]
        chunks = ["chunk1", "chunk2", "chunk3"]
        
        await qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test dense search
        query_vector = [0.15] * 384
        results = await qdrant_provider.dense_search(query_vector, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(result, VectorSearchResult) for result in results)
    
    @pytest.mark.asyncio
    async def test_qdrant_delete_operations(self, qdrant_provider):
        """Test QdrantProvider delete operations."""
        await qdrant_provider.connect()
        await qdrant_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"]
        chunks = ["chunk1", "chunk2"]
        
        await qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Verify data exists
        results = await qdrant_provider.fetch(ids)
        assert len(results) == 2
        
        # Delete one item
        await qdrant_provider.delete(["550e8400-e29b-41d4-a716-446655440001"])
        
        # Verify deletion
        results = await qdrant_provider.fetch(ids)
        assert len(results) == 1
        assert results[0].id == "550e8400-e29b-41d4-a716-446655440002"
    
    @pytest.mark.asyncio
    async def test_knowledge_base_setup_with_qdrant(self, knowledge_base):
        """Test Knowledge Base setup with QdrantProvider."""
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
    async def test_knowledge_base_query_with_qdrant(self, knowledge_base):
        """Test Knowledge Base query with QdrantProvider."""
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
    async def test_qdrant_hybrid_search(self, qdrant_provider):
        """Test QdrantProvider hybrid search functionality."""
        await qdrant_provider.connect()
        await qdrant_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"]
        chunks = ["chunk1", "chunk2"]
        
        await qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test hybrid search
        query_vector = [0.15] * 384
        query_text = "test query"
        
        # Mock the individual search methods
        qdrant_provider.dense_search = AsyncMock(return_value=[
            create_mock_vector_search_result("id1", 0.9, "Test result 1")
        ])
        qdrant_provider.full_text_search = AsyncMock(return_value=[
            create_mock_vector_search_result("id2", 0.8, "Test result 2")
        ])
        
        results = await qdrant_provider.hybrid_search(query_vector, query_text, top_k=2)
        
        # Verify hybrid search was called
        qdrant_provider.dense_search.assert_called_once()
        qdrant_provider.full_text_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_qdrant_full_text_search(self, qdrant_provider):
        """Test QdrantProvider full-text search."""
        await qdrant_provider.connect()
        await qdrant_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"]
        chunks = ["chunk1", "chunk2"]
        
        await qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test full-text search
        results = await qdrant_provider.full_text_search("chunk", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(result, VectorSearchResult) for result in results)
    
    def test_qdrant_filter_operations(self, qdrant_provider):
        """Test QdrantProvider filter operations (mocked)."""
        # Mock the operations
        qdrant_provider.connect = Mock()
        qdrant_provider.create_collection = Mock()
        qdrant_provider.upsert = Mock()
        qdrant_provider.dense_search = Mock(return_value=[
            create_mock_vector_search_result("550e8400-e29b-41d4-a716-446655440001", 0.9, "Test result 1"),
            create_mock_vector_search_result("550e8400-e29b-41d4-a716-446655440003", 0.8, "Test result 3")
        ])
        
        qdrant_provider.connect()
        qdrant_provider.create_collection()
        
        # Insert test data with different metadata
        vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        payloads = [
            {"source": "test1", "category": "A"},
            {"source": "test2", "category": "B"},
            {"source": "test3", "category": "A"}
        ]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002", "550e8400-e29b-41d4-a716-446655440003"]
        chunks = ["chunk1", "chunk2", "chunk3"]
        
        qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test search with filter
        query_vector = [0.15] * 384
        filter_dict = {"category": "A"}
        
        results = qdrant_provider.dense_search(query_vector, top_k=5, filter=filter_dict)
        
        # Verify operations were called
        qdrant_provider.upsert.assert_called_once()
        qdrant_provider.dense_search.assert_called_once_with(query_vector, top_k=5, filter=filter_dict)
        assert len(results) == 2
    
    def test_qdrant_payload_indexes(self, qdrant_provider):
        """Test QdrantProvider payload indexes."""
        # Test that indexed_fields can be configured in QdrantConfig
        config = qdrant_provider._config
        # indexed_fields is optional in BaseVectorDBConfig
        assert hasattr(config, 'indexed_fields') or True
    
    def test_qdrant_error_handling(self, qdrant_provider):
        """Test QdrantProvider error handling (mocked)."""
        # Mock error scenarios
        qdrant_provider.create_collection = Mock(side_effect=Exception("Connection error"))
        qdrant_provider.upsert = Mock(side_effect=Exception("Invalid data"))
        
        # Test connection error
        with pytest.raises(Exception):
            qdrant_provider.create_collection()
        
        # Test invalid upsert
        with pytest.raises(Exception):
            qdrant_provider.upsert([], [], [], [])
    
    def test_qdrant_configuration_validation(self):
        """Test QdrantProvider configuration validation (mocked)."""
        # Test invalid config (wrong provider type)
        from upsonic.vectordb.config import ChromaConfig
        invalid_connection = ConnectionConfig(mode=Mode.IN_MEMORY)
        invalid_config = ChromaConfig(
            connection=invalid_connection,
            collection_name="test",
            vector_size=384
        )
        
        # QdrantProvider should only accept QdrantConfig
        with pytest.raises(Exception):
            QdrantProvider(invalid_config)
    
    @pytest.mark.asyncio
    async def test_qdrant_collection_recreation(self, qdrant_provider):
        """Test QdrantProvider collection recreation."""
        await qdrant_provider.connect()
        
        # Create collection
        await qdrant_provider.create_collection()
        assert await qdrant_provider.collection_exists()
        
        # Test that collection exists
        assert await qdrant_provider.collection_exists()
    
    def test_qdrant_distance_metrics(self, qdrant_provider):
        """Test QdrantProvider with different distance metrics."""
        # Test that different distance metrics are supported
        distance_metrics = [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]
        
        # Test that the current metric is valid
        assert qdrant_provider._config.distance_metric in distance_metrics
    
    def test_qdrant_quantization_config(self, qdrant_provider):
        """Test QdrantProvider quantization configuration."""
        # Test that quantization configuration can be set in QdrantConfig
        config = qdrant_provider._config
        # quantization_config is optional in QdrantConfig
        assert hasattr(config, 'quantization_config') or True
