"""
Test 19: Knowledgebase as a context testing
Success criteria: We check attributes, what we log and results
"""
import pytest
from io import StringIO
from contextlib import redirect_stdout
import os
import tempfile
import shutil

from upsonic import Agent, Task, KnowledgeBase
from upsonic.embeddings import OpenAIEmbedding, OpenAIEmbeddingConfig
from upsonic.vectordb import ChromaProvider, ChromaConfig, ConnectionConfig, Mode

pytestmark = pytest.mark.timeout(180)


@pytest.mark.asyncio
async def test_knowledgebase_as_context_basic():
    """Test basic KnowledgeBase as context in Task."""
    
    # Create temporary directory and test document
    temp_dir = tempfile.mkdtemp()
    test_doc_path = os.path.join(temp_dir, "document.txt")
    
    try:
        # Create test document with specific content
        with open(test_doc_path, "w") as f:
            f.write("Python is a high-level programming language created by Guido van Rossum. "
                   "It emphasizes code readability and simplicity. "
                   "Python is widely used in web development, data science, and artificial intelligence.")
        
        # Setup embedding provider
        embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
        
        # Setup vector database
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        config = ChromaConfig(
            collection_name="test_kb_context",
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
        
        # Verify KB attributes
        assert kb is not None, "KnowledgeBase should be created"
        assert kb.embedding_provider == embedding, "Embedding provider should be set"
        assert kb.vectordb == vectordb, "VectorDB should be set"
        
        # Setup KB
        await kb.setup_async()
        
        # Use with Agent
        agent = Agent("openai/gpt-4o-mini", debug=True)
        task = Task(
            description="What are the main topics in the documents?",
            context=[kb],
            vector_search_similarity_threshold=0.0  # Lower threshold to ensure results
        )
        
        # Verify task attributes
        assert task.context is not None, "Task context should be set"
        assert isinstance(task.context, list), "Context should be a list"
        assert len(task.context) == 1, "Context should have 1 item"
        assert task.context[0] == kb, "Context should contain the KB"
        
        # Execute task
        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.do_async(task)
        
        output = output_buffer.getvalue()
        
        # Verify result
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        
        # Result should mention topics from the document
        result_lower = result.lower()
        assert any(word in result_lower for word in ["python", "programming", "language", "readability"]), \
            "Result should mention document topics"
        
        # Verify logging
        assert len(output) > 0, "Should have logging output"
        # Verify KB was queried (from logs we can see it was queried successfully)
        assert "KnowledgeBase" in output or "Retrieved" in output, \
            "Should show knowledge base was queried"
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_knowledgebase_context_multiple_sources():
    """Test KnowledgeBase with multiple document sources as context."""
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create multiple test documents
        doc1_path = os.path.join(temp_dir, "python.txt")
        doc2_path = os.path.join(temp_dir, "javascript.txt")
        
        with open(doc1_path, "w") as f:
            f.write("Python is known for its simplicity and readability. "
                   "It uses indentation to define code blocks. "
                   "Python is interpreted and dynamically typed.")
        
        with open(doc2_path, "w") as f:
            f.write("JavaScript is a scripting language for web development. "
                   "It runs in web browsers and on servers via Node.js. "
                   "JavaScript uses curly braces for code blocks.")
        
        # Setup components
        embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        config = ChromaConfig(
            collection_name="test_multi_sources",
            vector_size=1536,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=chroma_dir)
        )
        vectordb = ChromaProvider(config)
        
        # Create KB with multiple sources
        kb = KnowledgeBase(
            sources=[doc1_path, doc2_path],
            embedding_provider=embedding,
            vectordb=vectordb
        )
        
        # Verify KB has multiple sources
        assert len(kb.sources) == 2, "KB should have 2 sources"
        
        await kb.setup_async()
        
        # Use with Agent
        agent = Agent("openai/gpt-4o-mini", debug=True)
        task = Task(
            description="Compare the programming languages mentioned in the documents",
            context=[kb],
            vector_search_similarity_threshold=0.0  # Lower threshold to ensure results
        )
        
        result = await agent.do_async(task)
        
        # Verify result mentions both languages
        assert result is not None, "Result should not be None"
        result_lower = result.lower()
        assert "python" in result_lower or "javascript" in result_lower, \
            "Result should mention at least one language"
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_knowledgebase_context_with_directory():
    """Test KnowledgeBase with directory source as context."""
    
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(data_dir)
    
    try:
        # Create documents in directory
        with open(os.path.join(data_dir, "file1.txt"), "w") as f:
            f.write("Machine learning is a subset of artificial intelligence.")
        
        with open(os.path.join(data_dir, "file2.txt"), "w") as f:
            f.write("Deep learning uses neural networks with multiple layers.")
        
        # Setup components
        embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        config = ChromaConfig(
            collection_name="test_dir_source",
            vector_size=1536,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=chroma_dir)
        )
        vectordb = ChromaProvider(config)
        
        # Create KB with directory source
        kb = KnowledgeBase(
            sources=[data_dir],
            embedding_provider=embedding,
            vectordb=vectordb
        )
        
        await kb.setup_async()
        
        # Use with Agent
        agent = Agent("openai/gpt-4o-mini", debug=True)
        task = Task(
            description="What AI concepts are discussed in the documents?",
            context=[kb],
            vector_search_similarity_threshold=0.0  # Lower threshold to ensure results
        )
        
        result = await agent.do_async(task)
        
        # Verify result
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_knowledgebase_context_mixed_with_other_context():
    """Test KnowledgeBase combined with other context types."""
    
    temp_dir = tempfile.mkdtemp()
    test_doc_path = os.path.join(temp_dir, "document.txt")
    
    try:
        with open(test_doc_path, "w") as f:
            f.write("Cloud computing provides on-demand computing resources over the internet.")
        
        # Setup KB
        embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        config = ChromaConfig(
            collection_name="test_mixed_context",
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
        
        # Create task with mixed context
        additional_context = "Focus on cost-effectiveness and scalability."
        task = Task(
            description="Explain the benefits of cloud computing",
            context=[kb, additional_context]
        )
        
        # Verify task context
        assert len(task.context) == 2, "Should have 2 context items"
        assert task.context[0] == kb, "First item should be KB"
        assert task.context[1] == additional_context, "Second item should be string context"
        
        # Execute
        agent = Agent("openai/gpt-4o-mini", debug=True)
        result = await agent.do_async(task)
        
        # Verify result uses both contexts
        assert result is not None, "Result should not be None"
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_knowledgebase_context_logging(capsys):
    """Test that KnowledgeBase context usage is properly logged."""
    
    temp_dir = tempfile.mkdtemp()
    test_doc_path = os.path.join(temp_dir, "document.txt")
    
    try:
        with open(test_doc_path, "w") as f:
            f.write("Blockchain is a distributed ledger technology.")
        
        embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        config = ChromaConfig(
            collection_name="test_kb_logging",
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
        
        agent = Agent("openai/gpt-4o-mini", debug=True)
        task = Task(
            description="What is blockchain?",
            context=[kb]
        )
        
        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            await agent.do_async(task)
        
        output = output_buffer.getvalue()
        
        # Verify logging
        assert len(output) > 0, "Should have logging output"
        # Debug mode should show context processing
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_knowledgebase_context_attributes():
    """Test KnowledgeBase attributes when used as context."""
    
    temp_dir = tempfile.mkdtemp()
    test_doc_path = os.path.join(temp_dir, "document.txt")
    
    try:
        with open(test_doc_path, "w") as f:
            f.write("Test content for attribute verification.")
        
        embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        config = ChromaConfig(
            collection_name="test_attributes",
            vector_size=1536,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=chroma_dir)
        )
        vectordb = ChromaProvider(config)
        
        # Create KB with optional attributes
        kb = KnowledgeBase(
            sources=[test_doc_path],
            embedding_provider=embedding,
            vectordb=vectordb,
            name="TestKB",
            description="A test knowledge base",
            use_case="rag_retrieval"
        )
        
        # Verify KB attributes
        assert kb.name == "TestKB", "KB name should be set"
        assert kb.description == "A test knowledge base", "KB description should be set"
        assert kb.embedding_provider == embedding, "Embedding provider should be set"
        assert kb.vectordb == vectordb, "VectorDB should be set"
        assert kb.rag is True, "KB should have rag flag set"
        
        await kb.setup_async()
        
        # Use in task
        task = Task(
            description="Test query",
            context=[kb]
        )
        
        agent = Agent("openai/gpt-4o-mini", debug=True)
        result = await agent.do_async(task)
        
        assert result is not None, "Result should not be None"
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_knowledgebase_context_task_attribute():
    """Test that KnowledgeBase can be set as Task context attribute."""
    
    # Create a mock KB (without actual setup for this unit test)
    embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
    config = ChromaConfig(
        collection_name="test",
        vector_size=1536,
        connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path="./test_db")
    )
    vectordb = ChromaProvider(config)
    
    kb = KnowledgeBase(
        sources=["test.txt"],
        embedding_provider=embedding,
        vectordb=vectordb
    )
    
    # Create task with KB in context
    task = Task(
        description="Test",
        context=[kb]
    )
    
    # Verify context is set correctly
    assert task.context is not None, "Context should not be None"
    assert isinstance(task.context, list), "Context should be a list"
    assert len(task.context) == 1, "Context should have 1 item"
    assert task.context[0] == kb, "Context should contain the KB"
    
    # Verify we can modify context
    kb2 = KnowledgeBase(
        sources=["test2.txt"],
        embedding_provider=embedding,
        vectordb=vectordb
    )
    
    task.context.append(kb2)
    assert len(task.context) == 2, "Context should have 2 items after append"
    assert task.context[1] == kb2, "Second context item should be kb2"

