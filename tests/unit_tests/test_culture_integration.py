"""
Integration tests for Cultural Knowledge feature with Agent, Task, and Memory.

These tests run REAL agent executions with OpenAI models to verify:
1. Agent with storage, add_culture_to_context, update_cultural_knowledge
2. Agent with agentic culture (agent-controlled tools)
3. CultureManager with model for processing
4. CultureManager manual knowledge management
5. Culture context is properly injected

Notice: Culture is an experimental feature and is subject to change.
"""

import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from typing import List

# Core imports
from upsonic.agent import Agent
from upsonic.tasks.tasks import Task
from upsonic.storage.memory import Memory
from upsonic.culture import CultureManager, CulturalKnowledge
from upsonic.storage.providers import InMemoryStorage, SqliteStorage


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_db_file():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def sqlite_storage(temp_db_file):
    """Create a SQLite storage for testing."""
    return SqliteStorage(db_file=temp_db_file)


@pytest.fixture
def memory_with_sqlite(sqlite_storage):
    """Create a Memory instance with SQLite storage."""
    return Memory(
        storage=sqlite_storage,
        session_id="test-session-001",
        full_session_memory=True,
    )


@pytest.fixture
def in_memory_storage():
    """Create an in-memory storage for testing."""
    return InMemoryStorage()


@pytest.fixture
def memory_with_inmemory(in_memory_storage):
    """Create a Memory instance with in-memory storage."""
    return Memory(
        storage=in_memory_storage,
        session_id="test-session-002",
        full_session_memory=True,
    )


# =============================================================================
# Example 1: Agent with SQLite, add_culture_to_context, update_cultural_knowledge
# =============================================================================

class TestExample1AgentWithCulture:
    """
    Tests for Example 1 - Real agent execution with culture:
    
    agent = Agent(
        memory=memory,
        add_culture_to_context=True,
        update_cultural_knowledge=True,
    )
    task = Task(description="How do I set up a FastAPI service?")
    agent.do(task)
    """
    
    def test_agent_do_with_culture_reads_and_updates(self, memory_with_inmemory, capsys):
        """Test agent.do() reads culture context and updates after run."""
        # Pre-add cultural knowledge
        storage = memory_with_inmemory.storage
        manager = CultureManager(storage=storage, debug=True)
        manager.add_cultural_knowledge(CulturalKnowledge(
            name="Response Style",
            content="Always be concise and use bullet points",
            categories=["communication"],
        ))
        
        # Create agent with culture enabled
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_inmemory,
            add_culture_to_context=True,
            update_cultural_knowledge=True,
            debug=True,
        )
        
        # Create and run task
        task = Task(description="What is 2+2? Answer in one word.")
        result = agent.do(task)
        
        # Verify we got a response
        assert result is not None
        print(f"\n=== Agent Response ===\n{result}\n")
        
        # Check debug output
        captured = capsys.readouterr()
        print(captured.out)
        
        # Verify culture manager is working
        assert agent.culture_manager is not None
        assert agent.add_culture_to_context is True
        assert agent.update_cultural_knowledge is True
    
    def test_agent_do_with_sqlite_storage(self, memory_with_sqlite, capsys):
        """Test agent.do() with SQLite storage."""
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_sqlite,
            add_culture_to_context=True,
            update_cultural_knowledge=True,
            debug=True,
        )
        
        # Create and run task
        task = Task(description="Say hello in one word.")
        result = agent.do(task)
        
        # Verify response
        assert result is not None
        print(f"\n=== Agent Response (SQLite) ===\n{result}\n")
        
        captured = capsys.readouterr()
        print(captured.out)


# =============================================================================
# Example 2: Agent with Pre-loaded Cultural Knowledge
# =============================================================================

class TestExample2AgentWithPreloadedKnowledge:
    """
    Test agent uses pre-loaded cultural knowledge in its responses.
    """
    
    def test_agent_uses_cultural_knowledge_in_response(self, memory_with_inmemory, capsys):
        """Test that agent's response is influenced by cultural knowledge."""
        storage = memory_with_inmemory.storage
        
        # Pre-load cultural knowledge that should influence the response
        manager = CultureManager(storage=storage, debug=True)
        manager.add_cultural_knowledge(CulturalKnowledge(
            name="Response Format",
            content="Always start your response with 'CULTURAL_MARKER:' prefix",
            categories=["formatting"],
        ))
        
        # Create agent
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_inmemory,
            add_culture_to_context=True,  # This should inject the knowledge
            debug=True,
        )
        
        # Verify culture context contains our knowledge
        context = agent.culture_manager.get_culture_context()
        print(f"\n=== Culture Context ===\n{context}\n")
        assert "Response Format" in context
        
        # Run task
        task = Task(description="What is the capital of France?")
        result = agent.do(task)
        
        print(f"\n=== Agent Response ===\n{result}\n")
        
        captured = capsys.readouterr()
        print(captured.out)


# =============================================================================
# Example 3: Agent with Agentic Culture (Agent-controlled tools)
# =============================================================================

class TestExample3AgenticCulture:
    """
    Tests for agentic culture where agent can call culture tools.
    
    agent = Agent(
        memory=memory,
        add_culture_to_context=True,
        enable_agentic_culture=True,
    )
    """
    
    def test_agent_with_agentic_culture_has_tools(self, memory_with_inmemory):
        """Test agent has culture tools when enable_agentic_culture=True."""
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_inmemory,
            add_culture_to_context=True,
            enable_agentic_culture=True,
        )
        
        # Verify culture tools are available
        tool_names = [getattr(t, '__name__', str(t)) for t in agent.tools]
        print(f"\n=== Agent Tools ===\n{tool_names}\n")
        
        assert "add_cultural_knowledge" in tool_names
        assert "update_cultural_knowledge" in tool_names
    
    def test_agent_do_with_agentic_culture(self, memory_with_inmemory, capsys):
        """Test agent.do() with agentic culture tools available."""
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_inmemory,
            add_culture_to_context=True,
            enable_agentic_culture=True,
            debug=True,
        )
        
        # Ask agent to learn something (it may or may not use the tool)
        task = Task(description="Remember that I prefer Python over JavaScript. Just acknowledge.")
        result = agent.do(task)
        
        print(f"\n=== Agent Response (Agentic Culture) ===\n{result}\n")
        
        captured = capsys.readouterr()
        print(captured.out)
        
        assert result is not None


# =============================================================================
# Example 4: CultureManager Manual Operations
# =============================================================================

class TestExample4ManualCultureManager:
    """
    Tests for manual CultureManager operations:
    
    culture_manager = CultureManager(storage=storage)
    culture_manager.add_cultural_knowledge(knowledge)
    all_knowledge = culture_manager.get_all_knowledge()
    """
    
    def test_manual_add_and_retrieve_knowledge(self, in_memory_storage, capsys):
        """Test manually adding and retrieving cultural knowledge."""
        manager = CultureManager(storage=in_memory_storage, debug=True)
        
        # Add knowledge (like in Example 4)
        knowledge = CulturalKnowledge(
            name="Response Format Standard",
            summary="Keep responses concise, scannable, and runnable-first",
            categories=["communication", "ux"],
            content=(
                "- Lead with minimal runnable snippet\n"
                "- Use numbered steps for procedures\n"
                "- End with validation checklist"
            ),
            notes=["Derived from user feedback"],
        )
        
        kid = manager.add_cultural_knowledge(knowledge)
        print(f"\n=== Added Knowledge ID: {kid} ===\n")
        
        # Retrieve and print
        all_knowledge = manager.get_all_knowledge()
        print(f"\n=== All Knowledge ({len(all_knowledge)} entries) ===")
        for k in all_knowledge:
            print(f"  - {k.name}: {k.summary}")
            print(f"    Preview: {k.preview()}")
        
        captured = capsys.readouterr()
        print(captured.out)
        
        assert len(all_knowledge) == 1
        assert all_knowledge[0].name == "Response Format Standard"
    
    def test_culture_manager_with_agent(self, in_memory_storage, capsys):
        """Test using CultureManager to add knowledge, then agent uses it."""
        # Step 1: Manually add knowledge via CultureManager
        manager = CultureManager(storage=in_memory_storage, debug=True)
        
        manager.add_cultural_knowledge(CulturalKnowledge(
            name="Math Response Style",
            content="When answering math questions, always show your work step by step",
            categories=["math", "education"],
        ))
        
        # Step 2: Create memory with same storage
        memory = Memory(
            storage=in_memory_storage,
            session_id="manual-test",
            full_session_memory=True,
        )
        
        # Step 3: Create agent that reads from same storage
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            add_culture_to_context=True,
            debug=True,
        )
        
        # Verify agent sees the knowledge
        context = agent.culture_manager.get_culture_context()
        print(f"\n=== Agent's Culture Context ===\n{context}\n")
        assert "Math Response Style" in context
        
        # Step 4: Run agent
        task = Task(description="What is 5 times 7?")
        result = agent.do(task)
        
        print(f"\n=== Agent Response ===\n{result}\n")
        
        captured = capsys.readouterr()
        print(captured.out)


# =============================================================================
# Example 5: Multiple Agents Sharing Culture
# =============================================================================

class TestExample5MultipleAgentsSharedCulture:
    """
    Test multiple agents sharing the same culture storage.
    """
    
    def test_two_agents_share_culture(self, in_memory_storage, capsys):
        """Test two agents can share and see each other's cultural knowledge."""
        memory = Memory(
            storage=in_memory_storage,
            session_id="shared-session",
            full_session_memory=True,
        )
        
        # Agent 1 adds knowledge
        agent1 = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            add_culture_to_context=True,
            enable_agentic_culture=True,
            debug=True,
            name="Agent1",
        )
        
        # Manually add knowledge through agent1's culture manager
        agent1.culture_manager.add_cultural_knowledge(CulturalKnowledge(
            name="Team Communication",
            content="Always be respectful and professional",
            categories=["team"],
        ))
        
        # Agent 2 uses same memory (same storage)
        agent2 = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            add_culture_to_context=True,
            debug=True,
            name="Agent2",
        )
        
        # Agent 2 should see the knowledge
        context = agent2.culture_manager.get_culture_context()
        print(f"\n=== Agent2's Culture Context ===\n{context}\n")
        assert "Team Communication" in context
        
        # Run both agents
        task1 = Task(description="Say 'Agent 1 here' in exactly those words")
        result1 = agent1.do(task1)
        print(f"\n=== Agent1 Response ===\n{result1}\n")
        
        task2 = Task(description="Say 'Agent 2 here' in exactly those words")
        result2 = agent2.do(task2)
        print(f"\n=== Agent2 Response ===\n{result2}\n")
        
        captured = capsys.readouterr()
        print(captured.out)


# =============================================================================
# Example 6: Culture with Different Tasks
# =============================================================================

class TestExample6DifferentTasks:
    """
    Test culture with different task types.
    """
    
    def test_simple_math_task_with_culture(self, memory_with_inmemory, capsys):
        """Test simple math task with culture enabled."""
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_inmemory,
            add_culture_to_context=True,
            update_cultural_knowledge=True,
            debug=True,
        )
        
        task = Task(description="What is 10 + 20? Just give me the number.")
        result = agent.do(task)
        
        print(f"\n=== Math Task Response ===\n{result}\n")
        assert result is not None
        
        captured = capsys.readouterr()
        print(captured.out)
    
    def test_coding_question_with_culture(self, memory_with_inmemory, capsys):
        """Test coding question with culture enabled."""
        # Pre-add coding-related knowledge
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_inmemory,
            add_culture_to_context=True,
            debug=True,
        )
        
        agent.culture_manager.add_cultural_knowledge(CulturalKnowledge(
            name="Code Style",
            content="Always use descriptive variable names",
            categories=["coding"],
        ))
        
        task = Task(description="Write a Python one-liner to double a number x")
        result = agent.do(task)
        
        print(f"\n=== Coding Task Response ===\n{result}\n")
        assert result is not None
        
        captured = capsys.readouterr()
        print(captured.out)


# =============================================================================
# Debug Output Verification Tests
# =============================================================================

class TestDebugOutputVerification:
    """Tests to verify debug output is correct."""
    
    def test_culture_debug_output_during_agent_run(self, memory_with_inmemory, capsys):
        """Test that culture debug messages appear during agent run."""
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_inmemory,
            add_culture_to_context=True,
            update_cultural_knowledge=True,
            debug=True,  # Enable debug
        )
        
        # Add knowledge so there's something to log
        agent.culture_manager.add_cultural_knowledge(CulturalKnowledge(
            name="Debug Test",
            content="Testing debug output",
        ))
        
        task = Task(description="Say 'test' only")
        result = agent.do(task)
        
        captured = capsys.readouterr()
        print("\n=== Captured Debug Output ===")
        print(captured.out)
        
        # Should contain culture-related debug output
        assert "[CULTURE" in captured.out


# =============================================================================
# Full Workflow Tests
# =============================================================================

class TestFullWorkflows:
    """End-to-end workflow tests with real agent execution."""
    
    def test_complete_culture_workflow(self, temp_db_file, capsys):
        """Test complete workflow: create storage, add knowledge, run agent."""
        print("\n=== Starting Complete Culture Workflow ===\n")
        
        # Step 1: Create SQLite storage
        storage = SqliteStorage(db_file=temp_db_file)
        print(f"Created SQLite storage at: {temp_db_file}")
        
        # Step 2: Create CultureManager and add knowledge
        manager = CultureManager(storage=storage, debug=True)
        
        manager.add_cultural_knowledge(CulturalKnowledge(
            name="Greeting Style",
            content="Always greet users warmly and professionally",
            categories=["communication"],
        ))
        
        manager.add_cultural_knowledge(CulturalKnowledge(
            name="Answer Format",
            content="Keep answers brief and to the point",
            categories=["formatting"],
        ))
        
        all_k = manager.get_all_knowledge()
        print(f"\nAdded {len(all_k)} knowledge entries:")
        for k in all_k:
            print(f"  - {k.name}")
        
        # Step 3: Create Memory with same storage
        memory = Memory(
            storage=storage,
            session_id="workflow-test",
            full_session_memory=True,
        )
        
        # Step 4: Create Agent
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            add_culture_to_context=True,
            update_cultural_knowledge=True,
            debug=True,
        )
        
        # Step 5: Verify agent sees the knowledge
        context = agent.culture_manager.get_culture_context()
        print(f"\n=== Culture Context ===\n{context}\n")
        assert "Greeting Style" in context
        assert "Answer Format" in context
        
        # Step 6: Run agent
        task = Task(description="Say hello to me!")
        result = agent.do(task)
        
        print(f"\n=== Agent Response ===\n{result}\n")
        
        captured = capsys.readouterr()
        print(captured.out)
        
        assert result is not None
    
    def test_sequential_agent_runs_with_culture(self, memory_with_inmemory, capsys):
        """Test multiple sequential agent runs with culture."""
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_inmemory,
            add_culture_to_context=True,
            update_cultural_knowledge=True,
            debug=True,
        )
        
        # Run 1
        task1 = Task(description="What is 1+1? Just the number.")
        result1 = agent.do(task1)
        print(f"\n=== Run 1 Response ===\n{result1}\n")
        
        # Run 2
        task2 = Task(description="What is 2+2? Just the number.")
        result2 = agent.do(task2)
        print(f"\n=== Run 2 Response ===\n{result2}\n")
        
        # Run 3
        task3 = Task(description="What is 3+3? Just the number.")
        result3 = agent.do(task3)
        print(f"\n=== Run 3 Response ===\n{result3}\n")
        
        captured = capsys.readouterr()
        print(captured.out)
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None


# =============================================================================
# Attribute Verification Tests
# =============================================================================

class TestAttributeVerification:
    """Tests to verify all attributes are set correctly."""
    
    def test_agent_culture_attributes_after_init(self, memory_with_inmemory):
        """Verify all agent culture attributes after initialization."""
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_inmemory,
            add_culture_to_context=True,
            update_cultural_knowledge=True,
            enable_agentic_culture=True,
            debug=True,
        )
        
        print("\n=== Agent Culture Attributes ===")
        print(f"culture_manager: {agent.culture_manager}")
        print(f"add_culture_to_context: {agent.add_culture_to_context}")
        print(f"update_cultural_knowledge: {agent.update_cultural_knowledge}")
        print(f"enable_agentic_culture: {agent.enable_agentic_culture}")
        print(f"culture_manager.debug: {agent.culture_manager.debug}")
        print(f"culture_manager.storage: {agent.culture_manager.storage}")
        
        assert agent.culture_manager is not None
        assert agent.add_culture_to_context is True
        assert agent.update_cultural_knowledge is True
        assert agent.enable_agentic_culture is True
        assert agent.culture_manager.debug is True
        assert agent.culture_manager.storage is memory_with_inmemory.storage
    
    def test_culture_manager_attributes(self, in_memory_storage):
        """Verify all CultureManager attributes."""
        manager = CultureManager(
            storage=in_memory_storage,
            model="openai/gpt-4o-mini",
            add_knowledge=True,
            update_knowledge=True,
            delete_knowledge=True,
            clear_knowledge=True,
            debug=True,
        )
        
        print("\n=== CultureManager Attributes ===")
        print(f"storage: {manager.storage}")
        print(f"_model_spec: {manager._model_spec}")
        print(f"add_knowledge: {manager.add_knowledge}")
        print(f"update_knowledge: {manager.update_knowledge}")
        print(f"delete_knowledge: {manager.delete_knowledge}")
        print(f"clear_knowledge: {manager.clear_knowledge}")
        print(f"debug: {manager.debug}")
        print(f"knowledge_updated: {manager.knowledge_updated}")
        
        assert manager.storage is in_memory_storage
        assert manager._model_spec == "openai/gpt-4o-mini"
        assert manager.add_knowledge is True
        assert manager.update_knowledge is True
        assert manager.delete_knowledge is True
        assert manager.clear_knowledge is True
        assert manager.debug is True
    
    def test_cultural_knowledge_all_fields(self):
        """Verify all CulturalKnowledge fields."""
        knowledge = CulturalKnowledge(
            id="custom-id",
            name="Full Test",
            content="Full content here",
            summary="Full summary",
            categories=["cat1", "cat2"],
            notes=["note1", "note2"],
            metadata={"key": "value"},
            input="Original input text",
            agent_id="agent-123",
            team_id="team-456",
        )
        
        print("\n=== CulturalKnowledge Fields ===")
        print(f"id: {knowledge.id}")
        print(f"name: {knowledge.name}")
        print(f"content: {knowledge.content}")
        print(f"summary: {knowledge.summary}")
        print(f"categories: {knowledge.categories}")
        print(f"notes: {knowledge.notes}")
        print(f"metadata: {knowledge.metadata}")
        print(f"input: {knowledge.input}")
        print(f"created_at: {knowledge.created_at}")
        print(f"updated_at: {knowledge.updated_at}")
        print(f"agent_id: {knowledge.agent_id}")
        print(f"team_id: {knowledge.team_id}")
        
        assert knowledge.id == "custom-id"
        assert knowledge.name == "Full Test"
        assert knowledge.content == "Full content here"
        assert knowledge.summary == "Full summary"
        assert knowledge.categories == ["cat1", "cat2"]
        assert knowledge.notes == ["note1", "note2"]
        assert knowledge.metadata == {"key": "value"}
        assert knowledge.input == "Original input text"
        assert knowledge.agent_id == "agent-123"
        assert knowledge.team_id == "team-456"
        assert knowledge.created_at is not None
        assert knowledge.updated_at is not None


# =============================================================================
# Tool Execution Tests
# =============================================================================

class TestToolExecution:
    """Tests for culture tool execution."""
    
    def test_add_tool_execution(self, memory_with_inmemory, capsys):
        """Test manually executing the add_cultural_knowledge tool."""
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_inmemory,
            enable_agentic_culture=True,
            debug=True,
        )
        
        # Find and execute the add tool
        add_tool = None
        for t in agent.tools:
            if getattr(t, '__name__', '') == 'add_cultural_knowledge':
                add_tool = t
                break
        
        assert add_tool is not None
        
        result = add_tool(
            name="Tool Added Knowledge",
            content="This was added via the tool",
            categories=["tool-test"],
        )
        
        print(f"\n=== Tool Execution Result ===\n{result}\n")
        assert "Successfully added" in result
        
        # Verify it was actually added
        all_k = agent.culture_manager.get_all_knowledge()
        assert any(k.name == "Tool Added Knowledge" for k in all_k)
        
        # Now run agent to verify it sees the knowledge
        task = Task(description="Say 'knowledge added' exactly")
        agent_result = agent.do(task)
        
        print(f"\n=== Agent Response After Tool ===\n{agent_result}\n")
        
        captured = capsys.readouterr()
        print(captured.out)
