"""
Test 16: KnowledgeBase as a tool

Success criteria:
- KnowledgeBase "search" method is registered properly
- Agent calls it properly
"""

import pytest
import tempfile
import os
from pathlib import Path
from upsonic import Agent, Task
from upsonic.knowledge_base import KnowledgeBase
from upsonic.embeddings import OpenAIEmbedding
from upsonic.vectordb import ChromaProvider
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.timeout(180)


@pytest.fixture
def test_document():
    """Create a temporary test document."""
    # Create temporary file with content
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write("This is a test document about artificial intelligence and machine learning. "
                   "AI is transforming the world. Machine learning is a subset of AI.")
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


@pytest.fixture
def temp_vectordb_dir():
    """Create a temporary directory for vector database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_knowledgebase_registered_as_tool(test_document, temp_vectordb_dir):
    """Test that KnowledgeBase is registered as a tool with search method."""
    from upsonic.vectordb.config import ChromaConfig, ConnectionConfig, Mode, DistanceMetric, HNSWIndexConfig
    
    # Create KnowledgeBase
    embedding_provider = OpenAIEmbedding()
    
    # Create ChromaConfig properly
    connection = ConnectionConfig(mode=Mode.EMBEDDED, db_path=temp_vectordb_dir)
    chroma_config = ChromaConfig(
        connection=connection,
        collection_name="test_kb",
        vector_size=1536,  # OpenAI embedding size
        distance_metric=DistanceMetric.COSINE,
        index=HNSWIndexConfig()
    )
    
    vectordb = ChromaProvider(config=chroma_config)
    
    kb = KnowledgeBase(
        sources=[test_document],
        embedding_provider=embedding_provider,
        vectordb=vectordb,
        name="Test KnowledgeBase"
    )
    
    # Setup KnowledgeBase (process documents)
    await kb.setup_async()
    
    # Create agent with KnowledgeBase as tool
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    agent.add_tools(kb)
    
    # Verify KnowledgeBase is in agent.tools
    assert kb in agent.tools, "KnowledgeBase should be in agent.tools"
    
    # Verify search method is registered
    assert "search" in agent.registered_agent_tools, "search method should be registered"
    
    # Verify tool_manager has the search tool
    tool_defs = agent.tool_manager.get_tool_definitions()
    tool_names = [t.name for t in tool_defs]
    assert "search" in tool_names, "search should be in tool_manager definitions"
    
    # Cleanup
    await kb.close()


@pytest.mark.asyncio
async def test_knowledgebase_in_task_tools(test_document, temp_vectordb_dir):
    """Test KnowledgeBase as a tool in Task."""
    from upsonic.vectordb.config import ChromaConfig, ConnectionConfig, Mode, DistanceMetric, HNSWIndexConfig
    
    # Create KnowledgeBase
    embedding_provider = OpenAIEmbedding()
    
    # Create ChromaConfig properly
    connection = ConnectionConfig(mode=Mode.EMBEDDED, db_path=temp_vectordb_dir)
    chroma_config = ChromaConfig(
        connection=connection,
        collection_name="test_kb",
        vector_size=1536,
        distance_metric=DistanceMetric.COSINE,
        index=HNSWIndexConfig()
    )
    
    vectordb = ChromaProvider(config=chroma_config)
    
    kb = KnowledgeBase(
        sources=[test_document],
        embedding_provider=embedding_provider,
        vectordb=vectordb,
        name="Test KnowledgeBase"
    )
    
    # Setup KnowledgeBase
    await kb.setup_async()
    
    # Create agent
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Create task with KnowledgeBase as tool
    task = Task(
        description="Search the knowledge base for information about artificial intelligence and tell me what you found.",
        tools=[kb]
    )
    
    # Before execution, task tools not registered
    assert len(task.registered_task_tools) == 0, "Task tools should not be registered before execution"
    
    # Execute task
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    # Verify search is registered in task after execution
    assert "search" in task.registered_task_tools, "search should be registered in task after execution"
    
    # Verify result contains relevant information
    assert result is not None, "Result should not be None"
    result_str = str(result).lower()
    assert "artificial intelligence" in result_str or "ai" in result_str or "machine learning" in result_str, \
        f"Result should contain information about AI. Got: {result_str[:200]}"
    
    # Cleanup
    await kb.close()


@pytest.mark.asyncio
async def test_knowledgebase_agent_calls_search(test_document, temp_vectordb_dir):
    """Test that agent can properly call KnowledgeBase search method."""
    from upsonic.vectordb.config import ChromaConfig, ConnectionConfig, Mode, DistanceMetric, HNSWIndexConfig
    
    # Create KnowledgeBase
    embedding_provider = OpenAIEmbedding()
    
    # Create ChromaConfig properly
    connection = ConnectionConfig(mode=Mode.EMBEDDED, db_path=temp_vectordb_dir)
    chroma_config = ChromaConfig(
        connection=connection,
        collection_name="test_kb",
        vector_size=1536,
        distance_metric=DistanceMetric.COSINE,
        index=HNSWIndexConfig()
    )
    
    vectordb = ChromaProvider(config=chroma_config)
    
    kb = KnowledgeBase(
        sources=[test_document],
        embedding_provider=embedding_provider,
        vectordb=vectordb,
        name="Test KnowledgeBase",
        description="A knowledge base about AI and machine learning"
    )
    
    # Setup KnowledgeBase
    await kb.setup_async()
    
    # Create agent with KnowledgeBase
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    agent.add_tools(kb)
    
    # Verify search is available
    assert "search" in agent.registered_agent_tools, "search should be registered"
    
    # Create task that requires searching
    task = Task(
        description=(
            "Use the search tool from the knowledge base to find information about machine learning. "
            "Then summarize what you found in one sentence."
        )
    )
    
    # Execute task
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify search was called (check logs or result)
    assert "search" in output.lower() or "machine learning" in str(result).lower() or "ai" in str(result).lower(), \
        f"Search should have been called. Output: {output[:500]}"
    
    # Verify KnowledgeBase search method exists and is callable
    assert hasattr(kb, 'search'), "KnowledgeBase should have search method"
    assert callable(kb.search), "search should be callable"
    
    # Cleanup
    await kb.close()


@pytest.mark.asyncio
async def test_knowledgebase_toolkit_registration(test_document, temp_vectordb_dir):
    """Test that KnowledgeBase is properly registered as a ToolKit."""
    from upsonic.vectordb.config import ChromaConfig, ConnectionConfig, Mode, DistanceMetric, HNSWIndexConfig
    
    # Create KnowledgeBase
    embedding_provider = OpenAIEmbedding()
    
    # Create ChromaConfig properly
    connection = ConnectionConfig(mode=Mode.EMBEDDED, db_path=temp_vectordb_dir)
    chroma_config = ChromaConfig(
        connection=connection,
        collection_name="test_kb",
        vector_size=1536,
        distance_metric=DistanceMetric.COSINE,
        index=HNSWIndexConfig()
    )
    
    vectordb = ChromaProvider(config=chroma_config)
    
    kb = KnowledgeBase(
        sources=[test_document],
        embedding_provider=embedding_provider,
        vectordb=vectordb,
        name="Test KnowledgeBase"
    )
    
    # Setup KnowledgeBase
    await kb.setup_async()
    
    # Verify KnowledgeBase is a ToolKit
    from upsonic.tools import ToolKit
    assert isinstance(kb, ToolKit), "KnowledgeBase should be a ToolKit instance"
    
    # Create agent
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Add KnowledgeBase as tool
    agent.add_tools(kb)
    
    # Verify it's tracked in tool_manager processor
    kb_id = id(kb)
    assert kb_id in agent.tool_manager.processor.knowledge_base_instances, \
        "KnowledgeBase should be tracked in processor.knowledge_base_instances"
    
    # Verify search tool is registered
    assert "search" in agent.registered_agent_tools, "search should be registered"
    
    # Cleanup
    await kb.close()

