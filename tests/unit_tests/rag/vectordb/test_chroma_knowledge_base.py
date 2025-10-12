"""
Test ChromaProvider integration with Knowledge Base.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from upsonic.knowledge_base.knowledge_base import KnowledgeBase
from upsonic.vectordb.providers.chroma import ChromaProvider
from upsonic.vectordb.config import Config, CoreConfig, IndexingConfig, SearchConfig, DataManagementConfig, AdvancedConfig
from upsonic.vectordb.config import Mode, ProviderName, DistanceMetric, IndexType, HNSWTuningConfig
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
        core_config = CoreConfig(
            provider_name=ProviderName.CHROMA,
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
    def chroma_provider(self, chroma_config):
        """Create a ChromaProvider instance."""
        return ChromaProvider(chroma_config)
    
    @pytest.fixture
    def knowledge_base(self, chroma_provider, mock_embedding_provider, mock_chunker, mock_loader):
        """Create a Knowledge Base with ChromaProvider."""
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
    
    def test_chroma_provider_connection(self, chroma_provider):
        """Test ChromaProvider connection."""
        chroma_provider.connect()
        assert chroma_provider._is_connected
        assert chroma_provider._client is not None
        assert chroma_provider.is_ready()
    
    def test_chroma_provider_disconnection(self, chroma_provider):
        """Test ChromaProvider disconnection."""
        chroma_provider.connect()
        assert chroma_provider._is_connected
        
        chroma_provider.disconnect()
        assert not chroma_provider._is_connected
        assert chroma_provider._client is None
    
    def test_chroma_collection_creation(self, chroma_provider):
        """Test ChromaProvider collection creation."""
        chroma_provider.connect()
        assert not chroma_provider.collection_exists()
        
        chroma_provider.create_collection()
        assert chroma_provider.collection_exists()
    
    def test_chroma_collection_deletion(self, chroma_provider):
        """Test ChromaProvider collection deletion."""
        chroma_provider.connect()
        chroma_provider.create_collection()
        assert chroma_provider.collection_exists()
        
        chroma_provider.delete_collection()
        assert not chroma_provider.collection_exists()
    
    def test_chroma_upsert_operations(self, chroma_provider):
        """Test ChromaProvider upsert operations."""
        chroma_provider.connect()
        chroma_provider.create_collection()
        
        # Test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        chunks = ["chunk1", "chunk2"]
        
        # Upsert data
        chroma_provider.upsert(vectors, payloads, ids, chunks)
        
        # Verify data was stored
        results = chroma_provider.fetch(ids)
        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[1].id == "id2"
    
    def test_chroma_search_operations(self, chroma_provider):
        """Test ChromaProvider search operations."""
        chroma_provider.connect()
        chroma_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
        ids = ["id1", "id2", "id3"]
        chunks = ["chunk1", "chunk2", "chunk3"]
        
        chroma_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test dense search
        query_vector = [0.15] * 384
        results = chroma_provider.dense_search(query_vector, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(result, VectorSearchResult) for result in results)
    
    def test_chroma_delete_operations(self, chroma_provider):
        """Test ChromaProvider delete operations."""
        chroma_provider.connect()
        chroma_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        chunks = ["chunk1", "chunk2"]
        
        chroma_provider.upsert(vectors, payloads, ids, chunks)
        
        # Verify data exists
        results = chroma_provider.fetch(ids)
        assert len(results) == 2
        
        # Delete one item
        chroma_provider.delete(["id1"])
        
        # Verify deletion
        results = chroma_provider.fetch(ids)
        assert len(results) == 1
        assert results[0].id == "id2"
    
    @pytest.mark.asyncio
    async def test_knowledge_base_setup_with_chroma(self, knowledge_base):
        """Test Knowledge Base setup with ChromaProvider."""
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
    async def test_knowledge_base_query_with_chroma(self, knowledge_base):
        """Test Knowledge Base query with ChromaProvider."""
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
    
    def test_chroma_hybrid_search(self, chroma_provider):
        """Test ChromaProvider hybrid search functionality."""
        chroma_provider.connect()
        chroma_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        chunks = ["chunk1", "chunk2"]
        
        chroma_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test hybrid search
        query_vector = [0.15] * 384
        query_text = "test query"
        
        # Mock the individual search methods
        chroma_provider.dense_search = Mock(return_value=[
            create_mock_vector_search_result("id1", 0.9, "Test result 1")
        ])
        chroma_provider.full_text_search = Mock(return_value=[
            create_mock_vector_search_result("id2", 0.8, "Test result 2")
        ])
        
        results = chroma_provider.hybrid_search(query_vector, query_text, top_k=2)
        
        # Verify hybrid search was called
        chroma_provider.dense_search.assert_called_once()
        chroma_provider.full_text_search.assert_called_once()
    
    def test_chroma_full_text_search(self, chroma_provider):
        """Test ChromaProvider full-text search."""
        chroma_provider.connect()
        chroma_provider.create_collection()
        
        # Insert test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        chunks = ["chunk1", "chunk2"]
        
        chroma_provider.upsert(vectors, payloads, ids, chunks)
        
        # Test full-text search
        results = chroma_provider.full_text_search("chunk", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(result, VectorSearchResult) for result in results)
    
    def test_chroma_error_handling(self, chroma_provider):
        """Test ChromaProvider error handling."""
        # Test connection error
        with pytest.raises(Exception):
            chroma_provider.create_collection()  # Should fail without connection
        
        # Test with connection
        chroma_provider.connect()
        chroma_provider.create_collection()
        
        # Test invalid upsert
        with pytest.raises(Exception):
            chroma_provider.upsert([], [], [], [])  # Empty data should be handled gracefully
    
    def test_chroma_configuration_validation(self):
        """Test ChromaProvider configuration validation."""
        # Test invalid provider
        invalid_config = CoreConfig(
            provider_name=ProviderName.QDRANT,  # Wrong provider
            mode=Mode.IN_MEMORY,
            collection_name="test",
            vector_size=384
        )
        
        with pytest.raises(Exception):
            ChromaProvider(Config(core=invalid_config))
    
    def test_chroma_collection_recreation(self, chroma_provider):
        """Test ChromaProvider collection recreation."""
        chroma_provider.connect()
        
        # Create collection
        chroma_provider.create_collection()
        assert chroma_provider.collection_exists()
        
        # Test that collection exists
        assert chroma_provider.collection_exists()
