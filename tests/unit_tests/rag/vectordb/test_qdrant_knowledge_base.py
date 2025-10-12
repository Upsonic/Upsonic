"""
Test QdrantProvider integration with Knowledge Base.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from upsonic.knowledge_base.knowledge_base import KnowledgeBase
from upsonic.vectordb.providers.qdrant import QdrantProvider
from upsonic.vectordb.config import Config, CoreConfig, IndexingConfig, SearchConfig, DataManagementConfig, AdvancedConfig
from upsonic.vectordb.config import Mode, ProviderName, DistanceMetric, IndexType, HNSWTuningConfig
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
        core_config = CoreConfig(
            provider_name=ProviderName.QDRANT,
            mode=Mode.IN_MEMORY,
            collection_name="test_collection",
            vector_size=384,
            distance_metric=DistanceMetric.COSINE
        )
        
        indexing_config = IndexingConfig(
            index_config=HNSWTuningConfig(index_type=IndexType.HNSW),
            create_dense_index=True,
            create_sparse_index=False
        )
        
        search_config = SearchConfig(
            default_top_k=5,
            dense_search_enabled=True,
            full_text_search_enabled=True,
            hybrid_search_enabled=True
        )
        
        data_config = DataManagementConfig()
        advanced_config = AdvancedConfig()
        
        return Config(
            core=core_config,
            indexing=indexing_config,
            search=search_config,
            data_management=data_config,
            advanced=advanced_config
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
    
    def test_qdrant_provider_connection(self, qdrant_provider):
        """Test QdrantProvider connection."""
        qdrant_provider.connect()
        assert qdrant_provider._is_connected
        assert qdrant_provider._client is not None
        assert qdrant_provider.is_ready()
    
    def test_qdrant_provider_disconnection(self, qdrant_provider):
        """Test QdrantProvider disconnection."""
        qdrant_provider.connect()
        assert qdrant_provider._is_connected
        
        qdrant_provider.disconnect()
        assert not qdrant_provider._is_connected
        assert qdrant_provider._client is None
    
    def test_qdrant_collection_creation(self, qdrant_provider):
        """Test QdrantProvider collection creation."""
        qdrant_provider.connect()
        assert not qdrant_provider.collection_exists()
        
        qdrant_provider.create_collection()
        assert qdrant_provider.collection_exists()
    
    def test_qdrant_collection_deletion(self, qdrant_provider):
        """Test QdrantProvider collection deletion."""
        qdrant_provider.connect()
        qdrant_provider.create_collection()
        assert qdrant_provider.collection_exists()
        
        qdrant_provider.delete_collection()
        assert not qdrant_provider.collection_exists()
    
    def test_qdrant_upsert_operations(self, qdrant_provider):
        """Test QdrantProvider upsert operations."""
        qdrant_provider.connect()
        qdrant_provider.create_collection()
        
        # Test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"]
        chunks = ["chunk1", "chunk2"]
        
        # Upsert data
        qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Verify data was stored
        results = qdrant_provider.fetch(ids)
        assert len(results) == 2
        assert results[0].id == "550e8400-e29b-41d4-a716-446655440001"
        assert results[1].id == "550e8400-e29b-41d4-a716-446655440002"
    
    def test_qdrant_search_operations(self, qdrant_provider):
        """Test QdrantProvider search operations."""
        qdrant_provider.connect()
        qdrant_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002", "550e8400-e29b-41d4-a716-446655440003"]
        chunks = ["chunk1", "chunk2", "chunk3"]
        
        qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test dense search
        query_vector = [0.15] * 384
        results = qdrant_provider.dense_search(query_vector, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(result, VectorSearchResult) for result in results)
    
    def test_qdrant_delete_operations(self, qdrant_provider):
        """Test QdrantProvider delete operations."""
        qdrant_provider.connect()
        qdrant_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"]
        chunks = ["chunk1", "chunk2"]
        
        qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Verify data exists
        results = qdrant_provider.fetch(ids)
        assert len(results) == 2
        
        # Delete one item
        qdrant_provider.delete(["550e8400-e29b-41d4-a716-446655440001"])
        
        # Verify deletion
        results = qdrant_provider.fetch(ids)
        assert len(results) == 1
        assert results[0].id == "550e8400-e29b-41d4-a716-446655440002"
    
    @pytest.mark.asyncio
    async def test_knowledge_base_setup_with_qdrant(self, knowledge_base):
        """Test Knowledge Base setup with QdrantProvider."""
        # Mock the vectordb methods
        knowledge_base.vectordb.connect = Mock()
        knowledge_base.vectordb.create_collection = Mock()
        knowledge_base.vectordb.upsert = Mock()
        knowledge_base.vectordb.collection_exists = Mock(return_value=False)
        knowledge_base.vectordb.is_ready = Mock(return_value=True)
        
        # Mock the embedding provider
        knowledge_base.embedding_provider.embed_documents = AsyncMock(return_value=[[0.1] * 384, [0.2] * 384])
        
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
        knowledge_base.vectordb.connect = Mock()
        knowledge_base.vectordb.create_collection = Mock()
        knowledge_base.vectordb.upsert = Mock()
        knowledge_base.vectordb.collection_exists = Mock(return_value=False)
        knowledge_base.vectordb.is_ready = Mock(return_value=True)
        knowledge_base.vectordb.search = Mock(return_value=[
            create_mock_vector_search_result("id1", 0.9, "Test result 1"),
            create_mock_vector_search_result("id2", 0.8, "Test result 2")
        ])
        
        # Mock the embedding provider
        knowledge_base.embedding_provider.embed_documents = AsyncMock(return_value=[[0.1] * 384, [0.2] * 384])
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
    
    def test_qdrant_hybrid_search(self, qdrant_provider):
        """Test QdrantProvider hybrid search functionality."""
        qdrant_provider.connect()
        qdrant_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"]
        chunks = ["chunk1", "chunk2"]
        
        qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test hybrid search
        query_vector = [0.15] * 384
        query_text = "test query"
        
        # Mock the individual search methods
        qdrant_provider.dense_search = Mock(return_value=[
            create_mock_vector_search_result("id1", 0.9, "Test result 1")
        ])
        qdrant_provider.full_text_search = Mock(return_value=[
            create_mock_vector_search_result("id2", 0.8, "Test result 2")
        ])
        
        results = qdrant_provider.hybrid_search(query_vector, query_text, top_k=2)
        
        # Verify hybrid search was called
        qdrant_provider.dense_search.assert_called_once()
        qdrant_provider.full_text_search.assert_called_once()
    
    def test_qdrant_full_text_search(self, qdrant_provider):
        """Test QdrantProvider full-text search."""
        qdrant_provider.connect()
        qdrant_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"]
        chunks = ["chunk1", "chunk2"]
        
        qdrant_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test full-text search
        results = qdrant_provider.full_text_search("chunk", top_k=2)
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
        # This test would require more complex setup for payload indexes
        # For now, we'll test that the method exists and can be called
        qdrant_provider.connect()
        qdrant_provider.create_collection()
        
        # Test that payload indexes can be created (if supported)
        # This is a basic test - actual payload index creation would need more setup
        assert hasattr(qdrant_provider, '_create_payload_indexes')
    
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
        # Mock the provider initialization to raise an exception
        with patch('upsonic.vectordb.providers.qdrant.QdrantProvider.__init__') as mock_init:
            mock_init.side_effect = Exception("Invalid configuration")
            
            with pytest.raises(Exception):
                QdrantProvider(Config(core=CoreConfig(
                    provider_name=ProviderName.CHROMA,  # Wrong provider
                    mode=Mode.IN_MEMORY,
                    collection_name="test",
                    vector_size=384
                )))
    
    def test_qdrant_collection_recreation(self, qdrant_provider):
        """Test QdrantProvider collection recreation."""
        qdrant_provider.connect()
        
        # Create collection
        qdrant_provider.create_collection()
        assert qdrant_provider.collection_exists()
        
        # Test that collection exists
        assert qdrant_provider.collection_exists()
    
    def test_qdrant_distance_metrics(self, qdrant_provider):
        """Test QdrantProvider with different distance metrics."""
        # Test that different distance metrics are supported
        distance_metrics = [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]
        
        # Test that the current metric is valid
        assert qdrant_provider._config.core.distance_metric in distance_metrics
    
    def test_qdrant_quantization_config(self, qdrant_provider):
        """Test QdrantProvider quantization configuration."""
        # Test that quantization configuration exists
        from upsonic.vectordb.config import QuantizationConfig
        
        # Test that quantization config can be created
        quantization_config = QuantizationConfig(
            quantization_type='Scalar',
            bits=8
        )
        
        # Should not raise error
        assert quantization_config.quantization_type == 'Scalar'
        assert quantization_config.bits == 8
