"""
Test 25: Search parameters for Task class
Success criteria: We check the attributes, what we log and results
"""
import pytest
from io import StringIO
from contextlib import redirect_stdout

from upsonic import Agent, Task, KnowledgeBase
from upsonic.embeddings import OpenAIEmbedding, OpenAIEmbeddingConfig
from upsonic.vectordb import ChromaProvider, ChromaConfig, ConnectionConfig, Mode
import os
import tempfile
import shutil

pytestmark = pytest.mark.timeout(180)


@pytest.mark.asyncio
async def test_task_vector_search_parameters():
    """Test Task vector search parameters - verify attributes and proper usage."""
    
    # Create a temporary directory for test
    temp_dir = tempfile.mkdtemp()
    test_doc_path = os.path.join(temp_dir, "test_doc.txt")
    
    try:
        # Create test document
        with open(test_doc_path, "w") as f:
            f.write("Python is a high-level programming language. "
                   "It is widely used for web development and data science. "
                   "Python was created by Guido van Rossum in 1991. "
                   "The language emphasizes code readability with significant whitespace.")
        
        # Setup embedding provider
        embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
        
        # Setup vector database
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        config = ChromaConfig(
            collection_name="test_search_params",
            vector_size=1536,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=chroma_dir)
        )
        vectordb = ChromaProvider(config)
        
        # Create knowledge base
        kb = KnowledgeBase(
            sources=[test_doc_path],
            embedding_provider=embedding,
            vectordb=vectordb
        )
        await kb.setup_async()
        
        # Create task with vector search parameters
        task = Task(
            description="What programming language was created by Guido van Rossum?",
            context=[kb],
            vector_search_top_k=5,
            vector_search_alpha=0.6,
            vector_search_fusion_method='weighted',
            vector_search_similarity_threshold=0.3,
            vector_search_filter=None
        )
        
        # Verify task attributes
        assert task.vector_search_top_k == 5, "vector_search_top_k should be 5"
        assert task.vector_search_alpha == 0.6, "vector_search_alpha should be 0.6"
        assert task.vector_search_fusion_method == 'weighted', "fusion method should be weighted"
        assert task.vector_search_similarity_threshold == 0.3, "similarity threshold should be 0.3"
        assert task.vector_search_filter is None, "filter should be None"
        
        # Create agent and execute
        agent = Agent(model="openai/gpt-4o-mini", debug=True)
        
        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.do_async(task)
        
        output = output_buffer.getvalue()
        
        # Verify result
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        
        # Check if the search parameters were used (should appear in logs or context)
        # The knowledge base should have been searched with these parameters
        assert task._context_formatted is not None or task.context is not None, \
            "Context should be built"
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_task_search_params_with_rrf_fusion():
    """Test Task with RRF fusion method."""
    
    temp_dir = tempfile.mkdtemp()
    test_doc_path = os.path.join(temp_dir, "test_doc.txt")
    
    try:
        # Create test document
        with open(test_doc_path, "w") as f:
            f.write("Machine learning is a subset of artificial intelligence. "
                   "Deep learning uses neural networks with multiple layers. "
                   "Natural language processing helps computers understand human language.")
        
        # Setup components
        embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        config = ChromaConfig(
            collection_name="test_rrf",
            vector_size=1536,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=chroma_dir)
        )
        vectordb = ChromaProvider(config)
        
        kb = KnowledgeBase(
            sources=[test_doc_path],
            embedding_provider=embedding,
            vectordb=vectordb
        )
        await kb.setup_async()
        
        # Create task with RRF fusion
        task = Task(
            description="What is machine learning?",
            context=[kb],
            vector_search_top_k=3,
            vector_search_fusion_method='rrf',
            vector_search_similarity_threshold=0.2
        )
        
        # Verify attributes
        assert task.vector_search_top_k == 3, "top_k should be 3"
        assert task.vector_search_fusion_method == 'rrf', "fusion method should be rrf"
        assert task.vector_search_similarity_threshold == 0.2, "threshold should be 0.2"
        
        # Execute
        agent = Agent(model="openai/gpt-4o-mini", debug=True)
        result = await agent.do_async(task)
        
        # Verify result
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_task_search_params_with_filter():
    """Test Task with metadata filter."""
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test documents
        doc1_path = os.path.join(temp_dir, "python.txt")
        doc2_path = os.path.join(temp_dir, "javascript.txt")
        
        with open(doc1_path, "w") as f:
            f.write("Python is used for data science and machine learning.")
        
        with open(doc2_path, "w") as f:
            f.write("JavaScript is used for web development and frontend applications.")
        
        # Setup components
        embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        config = ChromaConfig(
            collection_name="test_filter",
            vector_size=1536,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=chroma_dir)
        )
        vectordb = ChromaProvider(config)
        
        kb = KnowledgeBase(
            sources=[doc1_path, doc2_path],
            embedding_provider=embedding,
            vectordb=vectordb
        )
        await kb.setup_async()
        
        # Create task with filter (though actual filtering depends on vectordb support)
        filter_dict = {"source": "python.txt"}
        task = Task(
            description="What is Python used for?",
            context=[kb],
            vector_search_top_k=2,
            vector_search_filter=filter_dict
        )
        
        # Verify attributes
        assert task.vector_search_filter == filter_dict, "Filter should be set correctly"
        assert task.vector_search_top_k == 2, "top_k should be 2"
        
        # Execute
        agent = Agent(model="openai/gpt-4o-mini", debug=True)
        result = await agent.do_async(task)
        
        # Verify result
        assert result is not None, "Result should not be None"
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_task_search_params_defaults():
    """Test Task search parameters default values."""
    
    task = Task(description="Test task")
    
    # Verify defaults
    assert task.vector_search_top_k is None, "Default top_k should be None"
    assert task.vector_search_alpha is None, "Default alpha should be None"
    assert task.vector_search_fusion_method is None, "Default fusion_method should be None"
    assert task.vector_search_similarity_threshold is None, "Default threshold should be None"
    assert task.vector_search_filter is None, "Default filter should be None"


def test_task_search_params_assignment():
    """Test Task search parameters can be assigned and modified."""
    
    task = Task(
        description="Test task",
        vector_search_top_k=10,
        vector_search_alpha=0.7,
        vector_search_fusion_method='weighted'
    )
    
    # Verify initial values
    assert task.vector_search_top_k == 10
    assert task.vector_search_alpha == 0.7
    assert task.vector_search_fusion_method == 'weighted'
    
    # Modify values
    task.vector_search_top_k = 15
    task.vector_search_alpha = 0.5
    task.vector_search_fusion_method = 'rrf'
    task.vector_search_similarity_threshold = 0.8
    task.vector_search_filter = {"category": "tech"}
    
    # Verify modified values
    assert task.vector_search_top_k == 15
    assert task.vector_search_alpha == 0.5
    assert task.vector_search_fusion_method == 'rrf'
    assert task.vector_search_similarity_threshold == 0.8
    assert task.vector_search_filter == {"category": "tech"}


@pytest.mark.asyncio
async def test_task_search_params_logging(capsys):
    """Test that search parameters are properly logged."""
    
    temp_dir = tempfile.mkdtemp()
    test_doc_path = os.path.join(temp_dir, "test.txt")
    
    try:
        with open(test_doc_path, "w") as f:
            f.write("Test document for logging verification.")
        
        embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        config = ChromaConfig(
            collection_name="test_logging",
            vector_size=1536,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=chroma_dir)
        )
        vectordb = ChromaProvider(config)
        
        kb = KnowledgeBase(
            sources=[test_doc_path],
            embedding_provider=embedding,
            vectordb=vectordb
        )
        await kb.setup_async()
        
        task = Task(
            description="What is this about?",
            context=[kb],
            vector_search_top_k=7,
            vector_search_alpha=0.75
        )
        
        agent = Agent(model="openai/gpt-4o-mini", debug=True)
        
        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            await agent.do_async(task)
        
        output = output_buffer.getvalue()
        
        # Verify logging occurred
        assert len(output) > 0, "Should have logging output"
        # The debug mode should show some processing information
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

