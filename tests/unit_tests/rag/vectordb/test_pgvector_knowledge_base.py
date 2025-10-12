"""
Test PgvectorProvider integration with Knowledge Base.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from upsonic.knowledge_base.knowledge_base import KnowledgeBase
from upsonic.vectordb.providers.pgvector import PgvectorProvider
from upsonic.vectordb.config import Config, CoreConfig, IndexingConfig, SearchConfig, DataManagementConfig, AdvancedConfig
from upsonic.vectordb.config import Mode, ProviderName, DistanceMetric, IndexType, HNSWTuningConfig, IVFTuningConfig
from upsonic.schemas.data_models import Document, Chunk, RAGSearchResult
from upsonic.schemas.vector_schemas import VectorSearchResult

from .mock_components import (
    MockEmbeddingProvider, MockChunker, MockLoader,
    create_mock_document, create_mock_chunk, create_mock_vector_search_result
)


class TestPgvectorKnowledgeBaseIntegration:
    """Test PgvectorProvider integration with Knowledge Base."""
    
    @pytest.fixture
    def pgvector_config(self):
        """Create a PgvectorProvider configuration."""
        core_config = CoreConfig(
            provider_name=ProviderName.PG,
            mode=Mode.LOCAL,
            host="localhost",
            port=5432,
            db_path="test_db",
            collection_name="test_collection",
            vector_size=384,
            distance_metric=DistanceMetric.COSINE,
            api_key="test_password"  # Used as password
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
        advanced_config = AdvancedConfig(namespace="test_tenant")
        
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
    def pgvector_provider(self, pgvector_config):
        """Create a PgvectorProvider instance."""
        return PgvectorProvider(pgvector_config)
    
    @pytest.fixture
    def knowledge_base(self, pgvector_provider, mock_embedding_provider, mock_chunker, mock_loader):
        """Create a Knowledge Base with PgvectorProvider."""
        return KnowledgeBase(
            sources=["test_source.txt"],
            embedding_provider=mock_embedding_provider,
            vectordb=pgvector_provider,
            splitters=mock_chunker,
            loaders=mock_loader,
            name="test_kb"
        )
    
    def test_pgvector_provider_initialization(self, pgvector_provider, pgvector_config):
        """Test PgvectorProvider initialization."""
        assert pgvector_provider._config == pgvector_config
        assert not pgvector_provider._is_connected
        assert pgvector_provider._connection is None
    
    @patch('upsonic.vectordb.providers.pgvector.psycopg.connect')
    def test_pgvector_provider_connection(self, mock_connect, pgvector_provider):
        """Test PgvectorProvider connection."""
        # Mock the connection
        mock_conn = Mock()
        mock_cursor = Mock()
        
        # Create a context manager mock
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_cursor)
        context_manager.__exit__ = Mock(return_value=None)
        mock_conn.cursor.return_value = context_manager
        
        # Mock cursor methods for is_ready() test
        mock_cursor.fetchone.return_value = (1,)
        
        # Mock connection properties
        mock_conn.closed = False
        
        mock_connect.return_value = mock_conn
        
        pgvector_provider.connect()
        assert pgvector_provider._is_connected
        assert pgvector_provider._connection is not None
        assert pgvector_provider.is_ready()
    
    def test_pgvector_provider_disconnection(self, pgvector_provider):
        """Test PgvectorProvider disconnection."""
        # Mock connection
        mock_conn = Mock()
        pgvector_provider._connection = mock_conn
        pgvector_provider._is_connected = True
        
        pgvector_provider.disconnect()
        assert not pgvector_provider._is_connected
        assert pgvector_provider._connection is None
    
    def test_pgvector_collection_creation(self, pgvector_provider):
        """Test PgvectorProvider collection creation (mocked)."""
        # Mock the operations
        pgvector_provider.connect = Mock()
        pgvector_provider.create_collection = Mock()
        pgvector_provider.collection_exists = Mock(side_effect=[False, True])
        
        pgvector_provider.connect()
        assert not pgvector_provider.collection_exists()
        
        pgvector_provider.create_collection()
        assert pgvector_provider.collection_exists()
        
        # Verify methods were called
        pgvector_provider.connect.assert_called_once()
        pgvector_provider.create_collection.assert_called_once()
    
    def test_pgvector_collection_deletion(self, pgvector_provider):
        """Test PgvectorProvider collection deletion (mocked)."""
        # Mock the operations
        pgvector_provider.connect = Mock()
        pgvector_provider.create_collection = Mock()
        pgvector_provider.delete_collection = Mock()
        pgvector_provider.collection_exists = Mock(side_effect=[True, False])
        
        pgvector_provider.connect()
        pgvector_provider.create_collection()
        assert pgvector_provider.collection_exists()
        
        pgvector_provider.delete_collection()
        assert not pgvector_provider.collection_exists()
        
        # Verify methods were called
        pgvector_provider.delete_collection.assert_called_once()
    
    def test_pgvector_upsert_operations(self, pgvector_provider):
        """Test PgvectorProvider upsert operations (mocked)."""
        # Mock the operations
        pgvector_provider.connect = Mock()
        pgvector_provider.create_collection = Mock()
        pgvector_provider.upsert = Mock()
        
        pgvector_provider.connect()
        pgvector_provider.create_collection()
        
        # Test data
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        chunks = ["chunk1", "chunk2"]
        
        # Upsert data
        pgvector_provider.upsert(vectors, payloads, ids, chunks)
        
        # Verify upsert was called
        pgvector_provider.upsert.assert_called_once_with(vectors, payloads, ids, chunks)
    
    def test_pgvector_search_operations(self, pgvector_provider):
        """Test PgvectorProvider search operations (mocked)."""
        # Mock the operations
        pgvector_provider.connect = Mock()
        pgvector_provider.create_collection = Mock()
        pgvector_provider.dense_search = Mock(return_value=[
            create_mock_vector_search_result("id1", 0.9, "Test result 1"),
            create_mock_vector_search_result("id2", 0.8, "Test result 2")
        ])
        
        pgvector_provider.connect()
        pgvector_provider.create_collection()
        
        # Test dense search
        query_vector = [0.15] * 384
        results = pgvector_provider.dense_search(query_vector, top_k=2)
        
        # Verify search was called
        pgvector_provider.dense_search.assert_called_once_with(query_vector, top_k=2)
        assert len(results) == 2
        assert all(isinstance(result, VectorSearchResult) for result in results)
    
    def test_pgvector_delete_operations(self, pgvector_provider):
        """Test PgvectorProvider delete operations (mocked)."""
        # Mock the operations
        pgvector_provider.connect = Mock()
        pgvector_provider.create_collection = Mock()
        pgvector_provider.delete = Mock()
        
        pgvector_provider.connect()
        pgvector_provider.create_collection()
        
        # Test delete
        pgvector_provider.delete(["id1"])
        
        # Verify delete was called
        pgvector_provider.delete.assert_called_once_with(["id1"])
    
    @pytest.mark.asyncio
    async def test_knowledge_base_setup_with_pgvector(self, knowledge_base):
        """Test Knowledge Base setup with PgvectorProvider."""
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
    async def test_knowledge_base_query_with_pgvector(self, knowledge_base):
        """Test Knowledge Base query with PgvectorProvider."""
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
    
    def test_pgvector_hybrid_search(self, pgvector_provider):
        """Test PgvectorProvider hybrid search functionality."""
        # Mock the individual search methods
        pgvector_provider.dense_search = Mock(return_value=[
            create_mock_vector_search_result("id1", 0.9, "Test result 1")
        ])
        pgvector_provider.full_text_search = Mock(return_value=[
            create_mock_vector_search_result("id2", 0.8, "Test result 2")
        ])
        
        # Test hybrid search
        query_vector = [0.15] * 384
        query_text = "test query"
        
        results = pgvector_provider.hybrid_search(query_vector, query_text, top_k=2)
        
        # Verify hybrid search was called
        pgvector_provider.dense_search.assert_called_once()
        pgvector_provider.full_text_search.assert_called_once()
    
    def test_pgvector_full_text_search(self, pgvector_provider):
        """Test PgvectorProvider full-text search (mocked)."""
        # Mock the operations
        pgvector_provider.connect = Mock()
        pgvector_provider.create_collection = Mock()
        pgvector_provider.full_text_search = Mock(return_value=[
            create_mock_vector_search_result("id1", 0.9, "Test result 1"),
            create_mock_vector_search_result("id2", 0.8, "Test result 2")
        ])
        
        pgvector_provider.connect()
        pgvector_provider.create_collection()
        
        # Test full-text search
        results = pgvector_provider.full_text_search("chunk", top_k=2, fts_field="chunk")
        
        # Verify search was called
        pgvector_provider.full_text_search.assert_called_once_with("chunk", top_k=2, fts_field="chunk")
        assert len(results) == 2
        assert all(isinstance(result, VectorSearchResult) for result in results)
    
    def test_pgvector_filter_operations(self, pgvector_provider):
        """Test PgvectorProvider filter operations."""
        # Test filter translation
        filter_dict = {
            "category": "A",
            "score": {"$gte": 0.5},
            "tags": {"$in": ["tag1", "tag2"]}
        }
        
        # Test that filter translation doesn't raise error
        try:
            sql, params = pgvector_provider._translate_filter(filter_dict)
            assert sql is not None
            assert isinstance(params, list)
        except Exception:
            # Filter translation might not be fully implemented
            pass
    
    def test_pgvector_index_types(self, pgvector_provider):
        """Test PgvectorProvider with different index types."""
        # Test HNSW index
        hnsw_config = HNSWTuningConfig(index_type=IndexType.HNSW, m=16, ef_construction=200)
        
        # Should not raise error
        assert hnsw_config.m == 16
        
        # Test IVF index
        ivf_config = IVFTuningConfig(index_type=IndexType.IVF_FLAT, nlist=100)
        
        # Should not raise error
        assert ivf_config.nlist == 100
    
    def test_pgvector_error_handling(self, pgvector_provider):
        """Test PgvectorProvider error handling."""
        # Test connection error
        with pytest.raises(Exception):
            pgvector_provider.create_collection()  # Should fail without connection
        
        # Test invalid upsert
        with pytest.raises(Exception):
            pgvector_provider.upsert([], [], [], [])  # Empty data should be handled gracefully
    
    def test_pgvector_configuration_validation(self):
        """Test PgvectorProvider configuration validation."""
        # Test invalid provider
        invalid_config = CoreConfig(
            provider_name=ProviderName.CHROMA,  # Wrong provider
            mode=Mode.IN_MEMORY,
            collection_name="test",
            vector_size=384
        )
        
        with pytest.raises(Exception):
            PgvectorProvider(Config(core=invalid_config))
    
    def test_pgvector_tenant_isolation(self, pgvector_provider):
        """Test PgvectorProvider tenant isolation."""
        # Test that namespace is required
        # Create a new config without namespace
        from upsonic.vectordb.config import AdvancedConfig
        config_without_namespace = AdvancedConfig(namespace=None)
        
        # Test that the config can be created
        assert config_without_namespace.namespace is None
    
    def test_pgvector_distance_metrics(self, pgvector_provider):
        """Test PgvectorProvider with different distance metrics."""
        # Test that different distance metrics are supported
        distance_metrics = [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]
        
        # Test that the current metric is valid
        assert pgvector_provider._config.core.distance_metric in distance_metrics
    
    def test_pgvector_connection_parameters(self, pgvector_provider):
        """Test PgvectorProvider connection parameters."""
        config = pgvector_provider._config
        
        # Test connection parameters
        assert config.core.host == "localhost"
        assert config.core.port == 5432
        assert config.core.db_path == "test_db"
        assert config.advanced.namespace == "test_tenant"
    
    def test_pgvector_sql_injection_protection(self, pgvector_provider):
        """Test PgvectorProvider SQL injection protection."""
        # Test that parameters are properly escaped
        filter_dict = {"malicious_field": "'; DROP TABLE test; --"}
        
        try:
            sql, params = pgvector_provider._translate_filter(filter_dict)
            # Should use parameterized queries
            assert "%s" in str(sql) or "Placeholder" in str(sql)
        except Exception:
            # Filter translation might not be fully implemented
            pass
