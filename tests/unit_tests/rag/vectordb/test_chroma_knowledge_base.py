"""
Test ChromaProvider integration with Knowledge Base.
"""
import sys
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# Skip entire module on Python 3.14+ due to pydantic compatibility issues
if sys.version_info >= (3, 14):
    pytest.skip(
        "pydantic 2.12.5 is not compatible with Python 3.14 (typing._eval_type API change)",
        allow_module_level=True
    )

from upsonic.knowledge_base.knowledge_base import KnowledgeBase
from upsonic.vectordb.providers.chroma import ChromaProvider
from upsonic.vectordb.config import ChromaConfig, ConnectionConfig, Mode, DistanceMetric, HNSWIndexConfig
from upsonic.schemas.data_models import Document, Chunk, RAGSearchResult
from upsonic.schemas.vector_schemas import VectorSearchResult

from .mock_components import (
    MockEmbeddingProvider, MockChunker, MockLoader,
    create_mock_document, create_mock_chunk, create_mock_vector_search_result
)


async def safe_disconnect(provider: ChromaProvider, timeout: float = 5.0) -> None:
    """Safely disconnect ChromaProvider with timeout."""
    if not provider._is_connected:
        return
    
    try:
        await asyncio.wait_for(asyncio.to_thread(provider.disconnect), timeout=timeout)
    except asyncio.TimeoutError:
        # Force cleanup if timeout
        provider.client = None
        provider._is_connected = False
        provider._collection_instance = None
    except Exception:
        # Force cleanup on any error
        provider.client = None
        provider._is_connected = False
        provider._collection_instance = None


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
        assert chroma_provider.client is None
    
    @pytest.mark.asyncio
    async def test_chroma_provider_connection(self, chroma_provider):
        """Test ChromaProvider connection."""
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            assert chroma_provider._is_connected
            assert chroma_provider.client is not None
            assert await asyncio.wait_for(asyncio.to_thread(chroma_provider.is_ready), timeout=5.0)
        finally:
            await safe_disconnect(chroma_provider)
    
    @pytest.mark.asyncio
    async def test_chroma_provider_disconnection(self, chroma_provider):
        """Test ChromaProvider disconnection."""
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            assert chroma_provider._is_connected
            
            await safe_disconnect(chroma_provider)
            assert not chroma_provider._is_connected
            assert chroma_provider.client is None
        except Exception:
            await safe_disconnect(chroma_provider)
            raise
    
    @pytest.mark.asyncio
    async def test_chroma_collection_creation(self, chroma_provider):
        """Test ChromaProvider collection creation."""
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            assert not await asyncio.wait_for(asyncio.to_thread(chroma_provider.collection_exists), timeout=5.0)
            
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.create_collection), timeout=10.0)
            assert await asyncio.wait_for(asyncio.to_thread(chroma_provider.collection_exists), timeout=5.0)
        finally:
            await safe_disconnect(chroma_provider)
    
    @pytest.mark.asyncio
    async def test_chroma_collection_deletion(self, chroma_provider):
        """Test ChromaProvider collection deletion."""
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.create_collection), timeout=10.0)
            assert await asyncio.wait_for(asyncio.to_thread(chroma_provider.collection_exists), timeout=5.0)
            
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.delete_collection), timeout=10.0)
            assert not await asyncio.wait_for(asyncio.to_thread(chroma_provider.collection_exists), timeout=5.0)
        finally:
            await safe_disconnect(chroma_provider)
    
    @pytest.mark.asyncio
    async def test_chroma_upsert_operations(self, chroma_provider):
        """Test ChromaProvider upsert operations."""
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.create_collection), timeout=10.0)
            
            # Test data
            vectors = [[0.1] * 384, [0.2] * 384]
            payloads = [{"source": "test1"}, {"source": "test2"}]
            ids = ["id1", "id2"]
            chunks = ["chunk1", "chunk2"]
            
            # Upsert data
            await asyncio.wait_for(
                asyncio.to_thread(chroma_provider.upsert, vectors, payloads, ids, chunks),
                timeout=10.0,
            )
            
            # Verify data was stored
            results = await asyncio.wait_for(asyncio.to_thread(chroma_provider.fetch, ids), timeout=10.0)
            assert len(results) == 2
            assert results[0].id == "id1"
            assert results[1].id == "id2"
        finally:
            await safe_disconnect(chroma_provider)
    
    @pytest.mark.asyncio
    async def test_chroma_search_operations(self, chroma_provider):
        """Test ChromaProvider search operations."""
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.create_collection), timeout=10.0)
            
            # Insert test data
            vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
            payloads = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
            ids = ["id1", "id2", "id3"]
            chunks = ["chunk1", "chunk2", "chunk3"]
            
            await asyncio.wait_for(
                asyncio.to_thread(chroma_provider.upsert, vectors, payloads, ids, chunks),
                timeout=10.0,
            )
            
            # Test dense search
            query_vector = [0.15] * 384
            results = await asyncio.wait_for(
                asyncio.to_thread(chroma_provider.dense_search, query_vector, 2),
                timeout=10.0,
            )
            assert len(results) <= 2
            assert all(isinstance(result, VectorSearchResult) for result in results)
        finally:
            await safe_disconnect(chroma_provider)
    
    @pytest.mark.asyncio
    async def test_chroma_delete_operations(self, chroma_provider):
        """Test ChromaProvider delete operations."""
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.create_collection), timeout=10.0)
            
            # Insert test data
            vectors = [[0.1] * 384, [0.2] * 384]
            payloads = [{"source": "test1"}, {"source": "test2"}]
            ids = ["id1", "id2"]
            chunks = ["chunk1", "chunk2"]
            
            await asyncio.wait_for(
                asyncio.to_thread(chroma_provider.upsert, vectors, payloads, ids, chunks),
                timeout=10.0,
            )
            
            # Verify data exists
            results = await asyncio.wait_for(asyncio.to_thread(chroma_provider.fetch, ids), timeout=10.0)
            assert len(results) == 2
            
            # Delete one item
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.delete, ["id1"]), timeout=10.0)
            
            # Verify deletion
            results = await asyncio.wait_for(asyncio.to_thread(chroma_provider.fetch, ids), timeout=10.0)
            assert len(results) == 1
            assert results[0].id == "id2"
        finally:
            await safe_disconnect(chroma_provider)
    
    @pytest.mark.asyncio
    async def test_knowledge_base_setup_with_chroma(self, knowledge_base):
        """Test Knowledge Base setup with ChromaProvider."""
        try:
            # Mock the vectordb async API used by KnowledgeBase
            knowledge_base.vectordb.aconnect = AsyncMock()
            knowledge_base.vectordb.acreate_collection = AsyncMock()
            knowledge_base.vectordb.aupsert = AsyncMock()
            knowledge_base.vectordb.acollection_exists = AsyncMock(return_value=False)
            knowledge_base.vectordb.ais_ready = AsyncMock(return_value=True)
            knowledge_base.vectordb.adisconnect = AsyncMock()
            
            # Mock the embedding provider - return 1 vector per chunk
            async def mock_embed_documents(chunks):
                return [[0.1] * 384] * len(chunks)
            knowledge_base.embedding_provider.embed_documents = AsyncMock(side_effect=mock_embed_documents)
            
            # Setup the knowledge base
            await knowledge_base.setup_async()
            
            # Verify setup was called
            knowledge_base.vectordb.aconnect.assert_called_once()
            knowledge_base.vectordb.acreate_collection.assert_called_once()
            knowledge_base.vectordb.aupsert.assert_called_once()
        finally:
            if hasattr(knowledge_base, 'close'):
                await asyncio.wait_for(knowledge_base.close(), timeout=5.0)
            elif hasattr(knowledge_base.vectordb, 'disconnect') and knowledge_base.vectordb._is_connected:
                await safe_disconnect(knowledge_base.vectordb)
    
    @pytest.mark.asyncio
    async def test_knowledge_base_query_with_chroma(self, knowledge_base):
        """Test Knowledge Base query with ChromaProvider."""
        try:
            # Mock the vectordb async API used by KnowledgeBase
            knowledge_base.vectordb.aconnect = AsyncMock()
            knowledge_base.vectordb.acreate_collection = AsyncMock()
            knowledge_base.vectordb.aupsert = AsyncMock()
            knowledge_base.vectordb.acollection_exists = AsyncMock(return_value=False)
            knowledge_base.vectordb.ais_ready = AsyncMock(return_value=True)
            knowledge_base.vectordb.adisconnect = AsyncMock()
            knowledge_base.vectordb.asearch = AsyncMock(return_value=[
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
        finally:
            if hasattr(knowledge_base, 'close'):
                await asyncio.wait_for(knowledge_base.close(), timeout=5.0)
            elif hasattr(knowledge_base.vectordb, 'disconnect') and knowledge_base.vectordb._is_connected:
                await safe_disconnect(knowledge_base.vectordb)
    
    @pytest.mark.asyncio
    async def test_chroma_hybrid_search(self, chroma_provider):
        """Test ChromaProvider hybrid search functionality."""
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.create_collection), timeout=10.0)
            
            # Insert test data
            vectors = [[0.1] * 384, [0.2] * 384]
            payloads = [{"source": "test1"}, {"source": "test2"}]
            ids = ["id1", "id2"]
            chunks = ["chunk1", "chunk2"]
            
            await asyncio.wait_for(
                asyncio.to_thread(chroma_provider.upsert, vectors, payloads, ids, chunks),
                timeout=10.0,
            )
            
            # Test hybrid search
            query_vector = [0.15] * 384
            query_text = "test query"
            
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: chroma_provider.hybrid_search(query_vector, query_text, top_k=2)
                ),
                timeout=10.0,
            )
            
            assert len(results) <= 2
            assert all(isinstance(result, VectorSearchResult) for result in results)
        finally:
            await safe_disconnect(chroma_provider)
    
    @pytest.mark.asyncio
    async def test_chroma_full_text_search(self, chroma_provider):
        """Test ChromaProvider full-text search."""
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.create_collection), timeout=10.0)
            
            # Insert test data
            vectors = [[0.1] * 384, [0.2] * 384]
            payloads = [{"source": "test1"}, {"source": "test2"}]
            ids = ["id1", "id2"]
            chunks = ["chunk1", "chunk2"]
            
            await asyncio.wait_for(
                asyncio.to_thread(chroma_provider.upsert, vectors, payloads, ids, chunks),
                timeout=10.0,
            )
            
            # Test full-text search
            results = await asyncio.wait_for(
                asyncio.to_thread(chroma_provider.full_text_search, "chunk", 2),
                timeout=10.0,
            )
            assert len(results) <= 2
            assert all(isinstance(result, VectorSearchResult) for result in results)
        finally:
            await safe_disconnect(chroma_provider)
    
    @pytest.mark.asyncio
    async def test_chroma_error_handling(self, chroma_provider):
        """Test ChromaProvider error handling."""
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.create_collection), timeout=10.0)

            with pytest.raises(Exception):
                await asyncio.wait_for(
                    asyncio.to_thread(
                        chroma_provider.upsert,
                        [[0.1] * 384],
                        [],
                        ["only_one_id"],
                        ["only_one_chunk"],
                    ),
                    timeout=5.0,
                )
        finally:
            await safe_disconnect(chroma_provider)
    
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
        try:
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.connect), timeout=10.0)
            
            # Create collection
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.create_collection), timeout=10.0)
            assert await asyncio.wait_for(asyncio.to_thread(chroma_provider.collection_exists), timeout=5.0)
            
            # Test recreation - should not raise error
            await asyncio.wait_for(asyncio.to_thread(chroma_provider.create_collection), timeout=10.0)
            assert await asyncio.wait_for(asyncio.to_thread(chroma_provider.collection_exists), timeout=5.0)
        finally:
            await safe_disconnect(chroma_provider)
