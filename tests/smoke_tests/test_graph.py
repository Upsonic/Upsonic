import pytest
import asyncio
from io import StringIO
from contextlib import redirect_stdout

from upsonic import Agent, Task, Graph, DecisionFunc, DecisionLLM, Direct
from upsonic.graph.graph import TaskNode

pytestmark = pytest.mark.timeout(180)


@pytest.mark.asyncio
async def test_graph_initialization():
    """Test Graph initialization with various attributes."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    
    # Test with default_agent
    graph = Graph(default_agent=agent)
    assert graph.default_agent == agent, "default_agent should be set"
    assert graph.parallel_execution is False, "parallel_execution should default to False"
    assert graph.max_parallel_tasks == 4, "max_parallel_tasks should default to 4"
    assert graph.show_progress is True, "show_progress should default to True"
    assert len(graph.nodes) == 0, "nodes should be empty initially"
    assert len(graph.edges) == 0, "edges should be empty initially"
    assert graph.state is not None, "state should be initialized"
    
    # Test with custom settings
    graph2 = Graph(
        default_agent=agent,
        parallel_execution=True,
        max_parallel_tasks=8,
        show_progress=False
    )
    assert graph2.parallel_execution is True, "parallel_execution should be set"
    assert graph2.max_parallel_tasks == 8, "max_parallel_tasks should be set"
    assert graph2.show_progress is False, "show_progress should be set"


@pytest.mark.asyncio
async def test_graph_add_tasks_chain_operator():
    """Test adding tasks using the chain operator (>>)."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    graph = Graph(default_agent=agent)
    
    task1 = Task("What is 2 + 2?")
    task2 = Task("What is 3 + 3?")
    task3 = Task("What is 4 + 4?")
    
    # Add tasks using chain operator
    graph.add(task1 >> task2 >> task3)
    
    # Verify nodes are added
    assert len(graph.nodes) == 3, "Should have 3 nodes"
    assert all(hasattr(node, 'task') for node in graph.nodes), "All nodes should be TaskNodes"
    
    # Verify edges are created
    assert len(graph.edges) == 2, "Should have 2 edges (task1->task2, task2->task3)"
    
    # Verify task descriptions
    task_descriptions = [node.task.description for node in graph.nodes if hasattr(node, 'task')]
    assert "What is 2 + 2?" in task_descriptions, "task1 should be in nodes"
    assert "What is 3 + 3?" in task_descriptions, "task2 should be in nodes"
    assert "What is 4 + 4?" in task_descriptions, "task3 should be in nodes"


@pytest.mark.asyncio
async def test_graph_add_tasks_individually():
    """Test adding tasks individually."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    graph = Graph(default_agent=agent)
    
    task1 = Task("Analyze market trends")
    task2 = Task("Generate recommendations")
    
    # Add tasks individually
    graph.add(task1)
    graph.add(task2)
    
    # Verify nodes are added
    assert len(graph.nodes) == 2, "Should have 2 nodes"
    
    # Verify edges (should be separate chains, no connection)
    # When added individually, they become separate chains
    task_descriptions = [node.task.description for node in graph.nodes if hasattr(node, 'task')]
    assert "Analyze market trends" in task_descriptions, "task1 should be in nodes"
    assert "Generate recommendations" in task_descriptions, "task2 should be in nodes"


@pytest.mark.asyncio
async def test_graph_run_sequential_execution():
    """Test running a graph with sequential task execution."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    graph = Graph(default_agent=agent, show_progress=False)
    
    task1 = Task("What is the capital of France?")
    task2 = Task("What is the population of that city?")
    
    graph.add(task1 >> task2)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        state = graph.run(verbose=True, show_progress=False)
    
    output = output_buffer.getvalue()
    
    # Verify execution logs
    assert "Starting Graph Execution" in output, "Should see 'Starting Graph Execution' log"
    assert "Executing Task" in output, "Should see 'Executing Task' log"
    assert "Task Completed" in output, "Should see 'Task Completed' log"
    assert "Graph Execution Completed" in output, "Should see 'Graph Execution Completed' log"
    
    # Verify state
    assert state is not None, "State should be returned"
    assert len(state.task_outputs) == 2, "Should have outputs for 2 tasks"
    
    # Verify final output
    final_output = graph.get_output()
    assert final_output is not None, "Final output should not be None"
    assert isinstance(final_output, str), "Final output should be a string"
    
    # Verify task-specific output
    task1_output = graph.get_task_output("What is the capital of France?")
    assert task1_output is not None, "task1 output should be retrievable"
    assert "Paris" in str(task1_output) or "paris" in str(task1_output).lower(), \
        "task1 output should mention Paris"


@pytest.mark.asyncio
async def test_graph_run_async():
    """Test running a graph asynchronously."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    graph = Graph(default_agent=agent, show_progress=False)
    
    task1 = Task("What is 5 + 5?")
    task2 = Task("What is 10 + 10?")
    
    graph.add(task1 >> task2)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        state = await graph.run_async(verbose=True, show_progress=False)
    
    output = output_buffer.getvalue()
    
    # Verify execution logs
    assert "Starting Graph Execution" in output, "Should see 'Starting Graph Execution' log"
    assert "Graph Execution Completed" in output, "Should see 'Graph Execution Completed' log"
    
    # Verify state
    assert state is not None, "State should be returned"
    assert len(state.task_outputs) == 2, "Should have outputs for 2 tasks"
    
    # Verify final output
    final_output = graph.get_output()
    assert final_output is not None, "Final output should not be None"


@pytest.mark.asyncio
async def test_graph_with_direct_interface():
    """Test Graph with Direct interface instead of Agent.
    
    Note: Graph currently requires BaseAgent, so Direct may not work directly.
    This test verifies the expected behavior if Direct were supported.
    """
    # Direct doesn't inherit from BaseAgent, so Graph will reject it
    # This is a known limitation - Graph requires BaseAgent instances
    direct = Direct(model="openai/gpt-4o")
    
    # Graph will raise TypeError because Direct is not a BaseAgent
    # This test documents the current behavior
    with pytest.raises(TypeError, match="default_agent must be an instance of a class that inherits from BaseAgent"):
        graph = Graph(default_agent=direct, show_progress=False)
    
    # For now, Direct cannot be used as default_agent in Graph
    # Users should use Agent instead, or Direct needs to be made compatible


@pytest.mark.asyncio
async def test_graph_decision_func():
    """Test Graph with DecisionFunc decision node."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    graph = Graph(default_agent=agent, show_progress=False)
    
    # Create tasks
    country_task = Task("What's an interesting country in Central Asia?")
    geography_task = Task("What is the geography of this country?")
    culture_task = Task("What is the culture of this country?")
    mountain_task = Task("What is the most popular mountain in this country?")
    
    # Define decision function
    def has_mountains(output):
        return "mountain" in str(output).lower() or "mountainous" in str(output).lower()
    
    # Create decision node
    decision = DecisionFunc("Has mountains?", has_mountains)
    
    # Add tasks with conditional branching
    graph.add(country_task >> geography_task >> decision.if_true(mountain_task).if_false(culture_task))
    
    # Verify decision node is in graph before execution
    decision_nodes = [node for node in graph.nodes if isinstance(node, DecisionFunc)]
    assert len(decision_nodes) == 1, "Should have one DecisionFunc node"
    assert decision_nodes[0].description == "Has mountains?", "Decision description should match"
    
    # Verify branches are set
    assert decision_nodes[0].true_branch is not None, "true_branch should be set"
    assert decision_nodes[0].false_branch is not None, "false_branch should be set"
    
    # Verify decision function works
    test_output = "This country has many mountains and is very mountainous."
    assert decision_nodes[0].evaluate(test_output) is True, "Decision function should return True for mountain text"
    
    test_output2 = "This country is flat with no elevation."
    assert decision_nodes[0].evaluate(test_output2) is False, "Decision function should return False for non-mountain text"
    
    # Verify graph structure
    assert len(graph.nodes) >= 4, "Should have at least 4 nodes (country, geography, decision, and branch tasks)"
    assert len(graph.edges) > 0, "Should have edges connecting nodes"
    
    # Verify decision node is properly connected in graph
    geography_nodes = [node for node in graph.nodes if isinstance(node, TaskNode) and node.task.description == "What is the geography of this country?"]
    assert len(geography_nodes) == 1, "Should have geography task node"
    
    # Execute the graph
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        state = graph.run(verbose=True, show_progress=False)
    
    output = output_buffer.getvalue()
    
    # Verify execution logs
    assert "Starting Graph Execution" in output, "Should see 'Starting Graph Execution' log"
    assert "Executing graph with decision support" in output, "Should see decision support log"
    assert "Evaluating Decision" in output, "Should see 'Evaluating Decision' log"
    assert "Graph Execution Completed" in output, "Should see 'Graph Execution Completed' log"
    
    # Verify state
    assert state is not None, "State should be returned"
    assert len(state.task_outputs) >= 3, "Should have outputs for at least 3 tasks (country, geography, and one branch)"
    
    # Verify that one of the branches was executed (either mountain or culture task)
    mountain_output = graph.get_task_output("What is the most popular mountain in this country?")
    culture_output = graph.get_task_output("What is the culture of this country?")
    
    # One of them should have output (depending on decision result)
    assert (mountain_output is not None) or (culture_output is not None), \
        "At least one branch should have been executed"


@pytest.mark.asyncio
async def test_graph_decision_llm():
    """Test Graph with DecisionLLM decision node."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    graph = Graph(default_agent=agent, show_progress=False)
    
    # Create tasks
    country_task = Task("What's an interesting country which has the biggest mountains?")
    geography_task = Task("What is the geography of this country?")
    culture_task = Task("What is the culture of this country?")
    mountain_task = Task("What is the most popular mountain in this country?")
    
    # Create decision node using LLM
    decision = DecisionLLM("Has the biggest mountains?")
    
    # Add tasks with conditional branching
    graph.add(country_task >> geography_task >> decision.if_true(mountain_task).if_false(culture_task))
    
    # Verify decision node is in graph before execution
    decision_nodes = [node for node in graph.nodes if isinstance(node, DecisionLLM)]
    assert len(decision_nodes) == 1, "Should have one DecisionLLM node"
    assert decision_nodes[0].description == "Has the biggest mountains?", "Decision description should match"
    
    # Verify branches are set
    assert decision_nodes[0].true_branch is not None, "true_branch should be set"
    assert decision_nodes[0].false_branch is not None, "false_branch should be set"
    
    # Verify prompt generation
    test_data = "This country has the highest mountains in the world."
    prompt = decision_nodes[0]._generate_prompt(test_data)
    assert "Has the biggest mountains?" in prompt, "Prompt should contain decision description"
    assert test_data in prompt, "Prompt should contain input data"
    
    # Verify graph structure
    assert len(graph.nodes) >= 4, "Should have at least 4 nodes (country, geography, decision, and branch tasks)"
    assert len(graph.edges) > 0, "Should have edges connecting nodes"
    
    # Verify decision node is properly connected in graph
    geography_nodes = [node for node in graph.nodes if isinstance(node, TaskNode) and node.task.description == "What is the geography of this country?"]
    assert len(geography_nodes) == 1, "Should have geography task node"
    
    # Execute the graph
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        state = graph.run(verbose=True, show_progress=False)
    
    output = output_buffer.getvalue()
    
    # Verify execution logs
    assert "Starting Graph Execution" in output, "Should see 'Starting Graph Execution' log"
    assert "Executing graph with decision support" in output, "Should see decision support log"
    assert "Evaluating Decision" in output, "Should see 'Evaluating Decision' log"
    assert "LLM Decision Response" in output or "Decision Result" in output, "Should see LLM decision logs"
    assert "Graph Execution Completed" in output, "Should see 'Graph Execution Completed' log"
    
    # Verify state
    assert state is not None, "State should be returned"
    assert len(state.task_outputs) >= 3, "Should have outputs for at least 3 tasks"
    
    # Verify that one of the branches was executed (either mountain or culture task)
    mountain_output = graph.get_task_output("What is the most popular mountain in this country?")
    culture_output = graph.get_task_output("What is the culture of this country?")
    
    # One of them should have output (depending on decision result)
    assert (mountain_output is not None) or (culture_output is not None), \
        "At least one branch should have been executed"


@pytest.mark.asyncio
async def test_graph_state_management():
    """Test Graph state management and task output retrieval."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    graph = Graph(default_agent=agent, show_progress=False)
    
    task1 = Task("What is the capital of Italy?")
    task2 = Task("What is the population of that city?")
    
    graph.add(task1 >> task2)
    
    graph.run(verbose=False, show_progress=False)
    
    # Verify state.task_outputs
    assert len(graph.state.task_outputs) == 2, "Should have 2 task outputs in state"
    
    # Verify get_latest_output
    latest_output = graph.state.get_latest_output()
    assert latest_output is not None, "Latest output should not be None"
    
    # Verify get_task_output by description
    task1_output = graph.get_task_output("What is the capital of Italy?")
    assert task1_output is not None, "task1 output should be retrievable"
    assert "Rome" in str(task1_output) or "rome" in str(task1_output).lower(), \
        "task1 output should mention Rome"
    
    # Verify get_output (should return latest)
    final_output = graph.get_output()
    assert final_output is not None, "Final output should not be None"
    assert final_output == latest_output, "get_output should return latest output"


@pytest.mark.asyncio
async def test_graph_attributes_after_execution():
    """Test Graph attributes after execution."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    graph = Graph(
        default_agent=agent,
        parallel_execution=False,
        max_parallel_tasks=4,
        show_progress=False
    )
    
    task1 = Task("What is 1 + 1?")
    task2 = Task("What is 2 + 2?")
    
    graph.add(task1 >> task2)
    
    # Verify attributes before execution
    assert graph.default_agent == agent, "default_agent should be set"
    assert graph.parallel_execution is False, "parallel_execution should be False"
    assert graph.max_parallel_tasks == 4, "max_parallel_tasks should be 4"
    assert graph.show_progress is False, "show_progress should be False"
    assert len(graph.nodes) == 2, "Should have 2 nodes"
    assert len(graph.edges) == 1, "Should have 1 edge"
    
    # Run graph
    graph.run(verbose=False, show_progress=False)
    
    # Verify attributes after execution (should remain unchanged)
    assert graph.default_agent == agent, "default_agent should remain unchanged"
    assert graph.parallel_execution is False, "parallel_execution should remain unchanged"
    assert graph.max_parallel_tasks == 4, "max_parallel_tasks should remain unchanged"
    assert graph.show_progress is False, "show_progress should remain unchanged"
    assert len(graph.nodes) == 2, "nodes should remain unchanged"
    assert len(graph.edges) == 1, "edges should remain unchanged"
    assert len(graph.state.task_outputs) == 2, "state.task_outputs should have 2 outputs"


@pytest.mark.asyncio
async def test_graph_progress_bar():
    """Test Graph progress bar display."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    graph = Graph(default_agent=agent, show_progress=True)
    
    task1 = Task("What is Python?")
    task2 = Task("What is JavaScript?")
    task3 = Task("What is Java?")
    
    graph.add(task1 >> task2 >> task3)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        graph.run(verbose=False, show_progress=True)
    
    output = output_buffer.getvalue()
    
    # Progress bar may not be visible in captured output, but we verify show_progress attribute
    assert graph.show_progress is True, "show_progress should be True"
    
    # Verify execution completed
    assert len(graph.state.task_outputs) == 3, "Should have 3 task outputs"


@pytest.mark.asyncio
async def test_graph_complex_workflow():
    """Test Graph with a complex workflow (multiple chains and decisions)."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent")
    graph = Graph(default_agent=agent, show_progress=False)
    
    # Create multiple task chains
    task1 = Task("What is machine learning?")
    task2 = Task("What are its applications?")
    task3 = Task("Summarize the key points")
    
    graph.add(task1 >> task2 >> task3)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        state = graph.run(verbose=True, show_progress=False)
    
    output = output_buffer.getvalue()
    
    # Verify execution logs
    assert "Starting Graph Execution" in output, "Should see 'Starting Graph Execution' log"
    assert "Executing Task" in output, "Should see 'Executing Task' log"
    assert "Task Completed" in output, "Should see 'Task Completed' log"
    assert "Graph Execution Completed" in output, "Should see 'Graph Execution Completed' log"
    
    # Verify state
    assert state is not None, "State should be returned"
    assert len(state.task_outputs) == 3, "Should have outputs for 3 tasks"
    
    # Verify task order (later tasks should reference earlier ones)
    task2_output = graph.get_task_output("What are its applications?")
    task3_output = graph.get_task_output("Summarize the key points")
    
    assert task2_output is not None, "task2 output should exist"
    assert task3_output is not None, "task3 output should exist"
    
    # Verify final output
    final_output = graph.get_output()
    assert final_output is not None, "Final output should not be None"

