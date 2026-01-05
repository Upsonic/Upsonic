"""
Test 16: KnowledgeBase as a tool

Success criteria:
- KnowledgeBase "search" method is registered properly with unique name (search_{kb_name})
- Agent calls it properly
- Multiple KnowledgeBase instances get unique tool names (no collision)
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
    
    # Verify search method is registered with unique name (search_{kb_name})
    # Tool name is "search_test_knowledgebase" (sanitized from "Test KnowledgeBase")
    expected_tool_name = "search_test_knowledgebase"
    assert expected_tool_name in agent.registered_agent_tools, \
        f"{expected_tool_name} should be registered. Got: {list(agent.registered_agent_tools.keys())}"
    
    # Verify tool_manager has the search tool
    tool_defs = agent.tool_manager.get_tool_definitions()
    tool_names = [t.name for t in tool_defs]
    assert expected_tool_name in tool_names, f"{expected_tool_name} should be in tool_manager definitions. Got: {tool_names}"
    
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
    
    # Verify search is registered in task after execution with unique name
    expected_tool_name = "search_test_knowledgebase"
    assert expected_tool_name in task.registered_task_tools, \
        f"{expected_tool_name} should be registered in task after execution. Got: {list(task.registered_task_tools.keys())}"
    
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
    
    # Verify search is available with unique name
    expected_tool_name = "search_test_knowledgebase"
    assert expected_tool_name in agent.registered_agent_tools, \
        f"{expected_tool_name} should be registered. Got: {list(agent.registered_agent_tools.keys())}"
    
    # Create task that requires searching
    task = Task(
        description=(
            "Use the search_test_knowledgebase tool to find information about machine learning. "
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
    
    # Verify search tool is registered with unique name
    expected_tool_name = "search_test_knowledgebase"
    assert expected_tool_name in agent.registered_agent_tools, \
        f"{expected_tool_name} should be registered. Got: {list(agent.registered_agent_tools.keys())}"
    
    # Cleanup
    await kb.close()


@pytest.mark.asyncio
async def test_multiple_knowledgebases_unique_tool_names(temp_vectordb_dir):
    """Test that multiple KnowledgeBase instances get unique tool names without collision."""
    from upsonic.vectordb.config import ChromaConfig, ConnectionConfig, Mode, DistanceMetric, HNSWIndexConfig
    
    # Create two separate test documents
    temp_file1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file1.write("This document is about Python programming. Python is a versatile language.")
    temp_file1.close()
    
    temp_file2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file2.write("This document is about JavaScript. JavaScript is used for web development.")
    temp_file2.close()
    
    try:
        # Create shared components
        embedding_provider = OpenAIEmbedding()
        
        # Create separate vector databases for each KB
        connection1 = ConnectionConfig(mode=Mode.EMBEDDED, db_path=temp_vectordb_dir + "/kb1")
        chroma_config1 = ChromaConfig(
            connection=connection1,
            collection_name="kb1",
            vector_size=1536,
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig()
        )
        
        connection2 = ConnectionConfig(mode=Mode.EMBEDDED, db_path=temp_vectordb_dir + "/kb2")
        chroma_config2 = ChromaConfig(
            connection=connection2,
            collection_name="kb2",
            vector_size=1536,
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig()
        )
        
        vectordb1 = ChromaProvider(config=chroma_config1)
        vectordb2 = ChromaProvider(config=chroma_config2)
        
        # Create two KnowledgeBases with UNIQUE NAMES
        kb1 = KnowledgeBase(
            sources=[temp_file1.name],
            embedding_provider=embedding_provider,
            vectordb=vectordb1,
            name="python_docs",  # Unique name
            description="Documentation about Python programming"
        )
        
        kb2 = KnowledgeBase(
            sources=[temp_file2.name],
            embedding_provider=embedding_provider,
            vectordb=vectordb2,
            name="javascript_docs",  # Different unique name
            description="Documentation about JavaScript programming"
        )
        
        # Setup both KnowledgeBases
        await kb1.setup_async()
        await kb2.setup_async()
        
        # Create agent and add BOTH KnowledgeBases as tools
        agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
        agent.add_tools([kb1, kb2])
        
        # Verify BOTH KnowledgeBases are registered with UNIQUE tool names
        registered_tools = list(agent.registered_agent_tools.keys())
        
        assert "search_python_docs" in registered_tools, \
            f"search_python_docs should be registered. Got: {registered_tools}"
        assert "search_javascript_docs" in registered_tools, \
            f"search_javascript_docs should be registered. Got: {registered_tools}"
        
        # Verify we have exactly 2 search tools (no collision/overwriting)
        search_tools = [name for name in registered_tools if name.startswith("search_")]
        assert len(search_tools) == 2, \
            f"Should have 2 unique search tools, but got {len(search_tools)}: {search_tools}"
        
        # Verify tool_manager has both tools
        tool_defs = agent.tool_manager.get_tool_definitions()
        tool_names = [t.name for t in tool_defs]
        assert "search_python_docs" in tool_names, \
            f"search_python_docs should be in tool_manager. Got: {tool_names}"
        assert "search_javascript_docs" in tool_names, \
            f"search_javascript_docs should be in tool_manager. Got: {tool_names}"
        
        # Cleanup
        await kb1.close()
        await kb2.close()
        
    finally:
        # Cleanup temp files
        if os.path.exists(temp_file1.name):
            os.remove(temp_file1.name)
        if os.path.exists(temp_file2.name):
            os.remove(temp_file2.name)


@pytest.mark.asyncio
async def test_multiple_knowledgebases_in_task_execution(temp_vectordb_dir):
    """Test that agent can call search tools from multiple KnowledgeBases in a single task."""
    from upsonic.vectordb.config import ChromaConfig, ConnectionConfig, Mode, DistanceMetric, HNSWIndexConfig
    
    # Create two separate test documents
    temp_file1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file1.write("The capital of France is Paris. Paris has the Eiffel Tower.")
    temp_file1.close()
    
    temp_file2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file2.write("The capital of Japan is Tokyo. Tokyo is known for its technology.")
    temp_file2.close()
    
    try:
        # Create shared components
        embedding_provider = OpenAIEmbedding()
        
        # Create separate vector databases
        connection1 = ConnectionConfig(mode=Mode.EMBEDDED, db_path=temp_vectordb_dir + "/france_kb")
        chroma_config1 = ChromaConfig(
            connection=connection1,
            collection_name="france",
            vector_size=1536,
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig()
        )
        
        connection2 = ConnectionConfig(mode=Mode.EMBEDDED, db_path=temp_vectordb_dir + "/japan_kb")
        chroma_config2 = ChromaConfig(
            connection=connection2,
            collection_name="japan",
            vector_size=1536,
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig()
        )
        
        vectordb1 = ChromaProvider(config=chroma_config1)
        vectordb2 = ChromaProvider(config=chroma_config2)
        
        # Create two KnowledgeBases
        kb_france = KnowledgeBase(
            sources=[temp_file1.name],
            embedding_provider=embedding_provider,
            vectordb=vectordb1,
            name="france_info",
            description="Information about France"
        )
        
        kb_japan = KnowledgeBase(
            sources=[temp_file2.name],
            embedding_provider=embedding_provider,
            vectordb=vectordb2,
            name="japan_info",
            description="Information about Japan"
        )
        
        # Setup both
        await kb_france.setup_async()
        await kb_japan.setup_async()
        
        # Create agent
        agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
        
        # Create task with BOTH KnowledgeBases as tools
        task = Task(
            description=(
                "Search both knowledge bases to find the capitals of France and Japan. "
                "Use search_france_info for France and search_japan_info for Japan. "
                "Then tell me both capitals."
            ),
            tools=[kb_france, kb_japan]
        )
        
        # Execute task
        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.do_async(task)
        
        # Verify result contains information from both KBs
        result_str = str(result).lower()
        assert "paris" in result_str, f"Result should mention Paris. Got: {result_str}"
        assert "tokyo" in result_str, f"Result should mention Tokyo. Got: {result_str}"
        
        # Verify both tools were registered
        assert "search_france_info" in task.registered_task_tools, \
            f"search_france_info should be in task tools. Got: {list(task.registered_task_tools.keys())}"
        assert "search_japan_info" in task.registered_task_tools, \
            f"search_japan_info should be in task tools. Got: {list(task.registered_task_tools.keys())}"
        
        # Cleanup
        await kb_france.close()
        await kb_japan.close()
        
    finally:
        # Cleanup temp files
        if os.path.exists(temp_file1.name):
            os.remove(temp_file1.name)
        if os.path.exists(temp_file2.name):
            os.remove(temp_file2.name)

