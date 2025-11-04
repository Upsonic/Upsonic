# AGENTS.md

This document provides comprehensive documentation for all agent types available in the Upsonic AI Agent Framework.

## Overview

Upsonic provides multiple agent types, each designed for different use cases and complexity levels. From simple direct LLM calls to complex multi-agent teams, the framework offers flexible solutions for production-ready AI applications.

## Agent Types

### 1. Direct - Simple LLM Interface

**Purpose**: Simplified, high-speed interface for direct LLM interactions without the complexity of memory, tools, or knowledge base integration.

**Use Cases**:
- Simple text generation tasks
- Direct data extraction from documents
- Quick structured outputs
- Maximum speed requirements

**Features**:
- No memory overhead
- No tool calls
- Direct model access
- Fast execution
- Structured output support

**Example**:
```python
from upsonic import Direct, Task
from pydantic import BaseModel

# Initialize Direct agent
direct = Direct(model="openai/gpt-4o")

# Define structured response
class TaxInfo(BaseModel):
    tax_number: str
    tax_rate: float

# Create task with context
task = Task(
    "Extract tax information from the document",
    context=["document.pdf"],
    response_format=TaxInfo
)

# Execute
result = direct.do(task)
print(result.tax_number)
```

**Methods**:
- `do(task, show_output=True)`: Execute task synchronously
- `do_async(task, show_output=True)`: Execute task asynchronously
- `print_do(task)`: Execute and print with visual output
- `print_do_async(task)`: Async version with visual output
- `with_model(model)`: Create new instance with different model
- `with_settings(settings)`: Create new instance with different settings
- `with_profile(profile)`: Create new instance with different profile
- `with_provider(provider)`: Create new instance with different provider

---

### 2. Agent - Full-Featured AI Agent

**Purpose**: Comprehensive, high-level AI Agent that integrates all framework components for production-ready applications.

**Use Cases**:
- Complex multi-step tasks requiring tool usage
- Applications requiring memory and conversation history
- Tasks needing safety policies and guardrails
- Production applications with reliability requirements
- Integration with knowledge bases and RAG systems

**Features**:
- Complete model abstraction (Model/Provider/Profile system)
- Advanced tool handling with ToolManager and Orchestrator
- Streaming and non-streaming execution modes
- Memory management and conversation history
- Context management and prompt engineering
- Caching capabilities
- Safety policies and guardrails (user_policy, agent_policy)
- Reliability layers
- Canvas integration for visual interactions
- External tool execution support
- Reflection capabilities
- Context compression strategies
- MCP (Model Context Protocol) integration

**Example**:
```python
from upsonic import Agent, Task
from upsonic.storage import Memory, InMemoryStorage

# Initialize with memory
memory = Memory(storage=InMemoryStorage())

agent = Agent(
    model="openai/gpt-4o",
    name="Research Assistant",
    memory=memory,
    enable_thinking_tool=True,
    tool_call_limit=10,
    system_prompt="You are a helpful research assistant.",
    reflection=True
)

task = Task("Research the latest developments in AI")
result = agent.do(task)

# Streaming version
for chunk in agent.stream(task):
    print(chunk)
```

**Key Configuration Options**:
- `model`: Model identifier (e.g., "openai/gpt-4o") or Model instance
- `name`: Agent name for identification
- `memory`: Memory instance for conversation history
- `system_prompt`: Custom system prompt
- `role`, `goal`, `instructions`: Agent personality configuration
- `education`, `work_experience`: Agent background information
- `tool_call_limit`: Maximum tool calls per execution
- `enable_thinking_tool`: Enable orchestrated thinking
- `enable_reasoning_tool`: Enable reasoning capabilities
- `user_policy`, `agent_policy`: Safety policies for inputs/outputs
- `reflection`: Enable reflection and self-evaluation
- `reliability_layer`: Reliability layer for robustness
- `compression_strategy`: Context compression method ("none", "simple", "llmlingua")
- `canvas`: Canvas instance for visual interactions
- `retry`: Number of retry attempts
- `mode`: Retry mode ("raise" or "return_false")

**Reasoning/Thinking Attributes**:
- `reasoning_effort`: "low", "medium", "high" (OpenAI models)
- `reasoning_summary`: "concise", "detailed" (OpenAI models)
- `thinking_enabled`: True/False (Anthropic/Google models)
- `thinking_budget`: Token budget for thinking
- `thinking_include_thoughts`: Include thoughts in output (Google models)
- `reasoning_format`: "hidden", "raw", "parsed" (Groq models)

**Methods**:
- `do(task)`: Execute task synchronously
- `do_async(task)`: Execute task asynchronously
- `stream(task)`: Stream results as they're generated
- `stream_async(task)`: Async streaming
- `print_do(task)`: Execute and print with visual output
- `print_do_async(task)`: Async version with visual output

---

### 3. DeepAgent - Advanced Multi-Step Agent

**Purpose**: Extended Agent with advanced capabilities for complex, multi-step tasks requiring planning, file operations, and subagent coordination.

**Use Cases**:
- Complex codebase analysis and refactoring
- Multi-step research and report generation
- Tasks requiring file manipulation
- Projects needing task planning and tracking
- Context isolation for parallel sub-tasks

**Features**:
- **Planning Tool**: `write_todos` for managing complex task plans
- **Virtual Filesystem**: `ls`, `read_file`, `write_file`, `edit_file` for file operations
- **Subagent System**: Spawn isolated subagents for context quarantine
- **Enhanced Prompts**: Specialized system prompts for deep reasoning
- **State Management**: Persistent todos and virtual filesystem across execution

**Example**:
```python
from upsonic import DeepAgent, Task, Agent

# Basic usage
agent = DeepAgent("openai/gpt-4o")
task = Task("Analyze the codebase and create a comprehensive refactoring plan")
result = agent.do(task)

# With custom subagents
researcher = Agent(
    "openai/gpt-4o",
    name="researcher",
    system_prompt="You are a research expert specializing in code analysis."
)

code_reviewer = Agent(
    "openai/gpt-4o",
    name="code-reviewer",
    system_prompt="You are a code review expert focusing on best practices."
)

agent = DeepAgent(
    "openai/gpt-4o",
    subagents=[researcher, code_reviewer]
)

# With initial files
agent = DeepAgent("openai/gpt-4o")
agent.add_file("/app/main.py", "def hello(): print('Hello')")
task = Task("Review and improve the code")
result = agent.do(task)
```

**Subagent System**:
DeepAgent can spawn isolated subagents for parallel or specialized tasks. Each subagent:
- Has isolated context windows
- Can be specialized with custom system prompts
- Returns results that are synthesized by the main agent
- Useful for breaking down complex objectives into manageable tasks

**Virtual Filesystem**:
- Files persist across agent execution
- Supports read, write, and edit operations
- Useful for code generation, document processing, and multi-step file operations

**Todo Management**:
- Track complex multi-step tasks
- Update progress in real-time
- Organize work systematically
- Demonstrates thoroughness to users

**Methods**:
- Inherits all methods from `Agent`
- Additional file operations via tools:
  - `ls`: List files in virtual filesystem
  - `read_file`: Read file contents
  - `write_file`: Write new file
  - `edit_file`: Edit existing file
  - `write_todos`: Create and manage task lists
  - `create_task_tool`: Spawn subagents

---

### 4. Team - Multi-Agent Coordination

**Purpose**: Coordinate multiple agents to work together on complex tasks, with support for different coordination modes.

**Use Cases**:
- Tasks requiring multiple specialized agents
- Parallel processing of independent tasks
- Complex workflows with agent coordination
- Leader-follower agent architectures
- Routing tasks to specialized agents

**Features**:
- Multiple coordination modes: `sequential`, `coordinate`, `route`
- Leader agent for coordination
- Context sharing between agents
- Task assignment and delegation
- Result combination and synthesis
- Memory sharing across team members
- Automatic tool integration (`ask_other_team_members`)

**Example**:
```python
from upsonic import Team, Agent, Task
from upsonic.storage import Memory, InMemoryStorage

# Create specialized agents
researcher = Agent("openai/gpt-4o", name="Researcher")
writer = Agent("openai/gpt-4o", name="Writer")
reviewer = Agent("openai/gpt-4o", name="Reviewer")

# Create team with sequential mode
team = Team(
    agents=[researcher, writer, reviewer],
    mode="sequential",
    memory=Memory(storage=InMemoryStorage())
)

# Execute tasks
tasks = [
    Task("Research the topic"),
    Task("Write a comprehensive article"),
    Task("Review and improve the article")
]

result = team.do(tasks)

# Coordinate mode (leader agent coordinates)
team = Team(
    agents=[researcher, writer, reviewer],
    mode="coordinate",
    ask_other_team_members=True
)

# Route mode (router assigns tasks)
team = Team(
    agents=[researcher, writer, reviewer],
    mode="route"
)
```

**Coordination Modes**:

1. **Sequential** (`mode="sequential"`):
   - Agents execute tasks one after another
   - Each agent receives output from previous agent
   - Simple pipeline execution

2. **Coordinate** (`mode="coordinate"`):
   - Leader agent coordinates team members
   - Can delegate tasks to specialized agents
   - Supports context sharing and collaboration

3. **Route** (`mode="route"`):
   - Router agent assigns tasks to appropriate team members
   - Intelligent task routing based on agent capabilities
   - Optimized for parallel execution

**Configuration Options**:
- `agents`: List of Agent instances
- `tasks`: Optional list of tasks to execute
- `model`: Model for leader/router agent
- `response_format`: Response format for final output
- `ask_other_team_members`: Automatically add agents as tools
- `mode`: Coordination mode ("sequential", "coordinate", "route")
- `memory`: Shared memory for team

**Methods**:
- `do(tasks)`: Execute tasks with team
- `do_async(tasks)`: Async execution
- `complete(tasks)`: Alias for `do()`
- `print_do(tasks)`: Execute and print with visual output
- `multi_agent(agent_configurations, tasks)`: Low-level multi-agent execution

---

### 5. BaseAgent - Abstract Base Class

**Purpose**: Abstract base class for all agent implementations, providing a common interface for integration with other framework components.

**Use Cases**:
- Creating custom agent implementations
- Integration with Graph and other framework components
- Breaking circular dependencies

**Note**: This is an abstract base class and should not be instantiated directly. Use `Agent`, `DeepAgent`, or `Direct` for actual implementations.

---

## Choosing the Right Agent

### When to Use Direct

- ✅ Simple, single-step tasks
- ✅ Maximum speed requirements
- ✅ No need for conversation history
- ✅ No tool calls required
- ✅ Direct document processing

### When to Use Agent

- ✅ Multi-step tasks requiring tools
- ✅ Need for conversation memory
- ✅ Safety policies required
- ✅ Integration with knowledge bases
- ✅ Production applications
- ✅ Need for reliability layers
- ✅ Streaming responses

### When to Use DeepAgent

- ✅ Complex multi-step workflows
- ✅ File operations and code generation
- ✅ Need for task planning and tracking
- ✅ Context isolation for sub-tasks
- ✅ Parallel subagent execution
- ✅ Codebase analysis and refactoring

### When to Use Team

- ✅ Tasks requiring multiple specialized agents
- ✅ Parallel processing of independent tasks
- ✅ Complex coordination workflows
- ✅ Leader-follower architectures
- ✅ Task routing to specialists

---

## Common Patterns

### Agent with Memory

```python
from upsonic import Agent, Task
from upsonic.storage import Memory, InMemoryStorage

memory = Memory(storage=InMemoryStorage())
agent = Agent("openai/gpt-4o", memory=memory)

# First conversation
task1 = Task("My name is John")
agent.do(task1)

# Second conversation (remembers context)
task2 = Task("What's my name?")
result = agent.do(task2)  # Returns "John"
```

### Agent with Safety Policies

```python
from upsonic import Agent, Task
from upsonic.safety_engine import Policy, Rule

# Create safety policy
policy = Policy(
    rules=[Rule(name="no_sensitive_data", pattern=r"\d{3}-\d{2}-\d{4}")],
    action="block"
)

agent = Agent(
    "openai/gpt-4o",
    user_policy=policy,  # Check user inputs
    agent_policy=policy   # Check agent outputs
)
```

### Agent with Tools

```python
from upsonic import Agent, Task, Tool

def calculate(expression: str) -> float:
    """Calculate a mathematical expression."""
    return eval(expression)

tool = Tool(name="calculate", function=calculate)
agent = Agent("openai/gpt-4o", tools=[tool])

task = Task("What is 15 * 23 + 45?")
result = agent.do(task)  # Agent uses calculate tool
```

### Agent with Structured Outputs

```python
from upsonic import Agent, Task
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    confidence: float
    recommendations: list[str]

agent = Agent("openai/gpt-4o")
task = Task(
    "Analyze this document",
    context=["document.pdf"],
    response_format=AnalysisResult
)

result = agent.do(task)
print(result.summary)
print(result.confidence)
```

---

## Best Practices

1. **Start Simple**: Begin with `Direct` for simple tasks, upgrade to `Agent` when you need additional features.

2. **Use Memory Wisely**: Implement memory for conversational applications, but skip it for stateless operations.

3. **Leverage Safety Policies**: Always use safety policies in production applications to protect against unwanted content.

4. **Plan Complex Tasks**: Use `DeepAgent` for complex multi-step tasks that require planning and file operations.

5. **Coordinate Teams**: Use `Team` when tasks require multiple specialized agents or parallel processing.

6. **Monitor Token Usage**: Use compression strategies and context management for long conversations.

7. **Handle Errors**: Implement retry logic and error handling for production reliability.

8. **Stream for UX**: Use streaming for better user experience in interactive applications.

---

## Migration Guide

### From Direct to Agent

If you need to add memory, tools, or safety features:

```python
# Before (Direct)
direct = Direct("openai/gpt-4o")
result = direct.do(task)

# After (Agent)
agent = Agent("openai/gpt-4o", memory=memory)
result = agent.do(task)
```

### From Agent to DeepAgent

If you need planning and file operations:

```python
# Before (Agent)
agent = Agent("openai/gpt-4o")
result = agent.do(task)

# After (DeepAgent)
agent = DeepAgent("openai/gpt-4o")
result = agent.do(task)  # Same interface, more capabilities
```

---

## Additional Resources

- [Upsonic Documentation](https://docs.upsonic.ai/)
- [CLAUDE.md](./CLAUDE.md) - Development guide
- [README.md](./README.md) - Framework overview
- Framework guides:
  1. [Create an Agent](https://docs.upsonic.ai/guides/1-create-a-task)
  2. [Create a Task](https://docs.upsonic.ai/guides/2-create-an-agent)
  3. [Add a Safety Engine](https://docs.upsonic.ai/guides/3-add-a-safety-engine)
  4. [Add a Tool](https://docs.upsonic.ai/guides/4-add-a-tool)
  5. [Add an MCP](https://docs.upsonic.ai/guides/5-add-an-mcp)
  6. [Integrate a Memory](https://docs.upsonic.ai/guides/6-integrate-a-memory)
  7. [Creating a Team of Agents](https://docs.upsonic.ai/guides/7-creating-a-team-of-agents)

---


## Code Quality Standards (Ruff Configuration)

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code quality enforcement. The following rules are configured in `pyproject.toml`:

### Rule Sets Enabled

```toml
select = [
  "E",  # pycodestyle errors (style)
  "W",  # pycodestyle warnings
  "F",  # pyflakes (undefined names, unused imports)
  "I",  # isort (import order and grouping)
  "B",  # flake8-bugbear (common logic errors)
  "UP", # pyupgrade (modern Python syntax suggestions)
]
```

**Explanation**:
- **E**: pycodestyle errors - Enforces PEP 8 style guidelines
- **W**: pycodestyle warnings - Catches style issues that don't break functionality
- **F**: pyflakes - Detects undefined names, unused imports, and other common errors
- **I**: isort - Ensures consistent import ordering and grouping
- **B**: flake8-bugbear - Identifies common bugs and design problems
- **UP**: pyupgrade - Suggests modern Python syntax improvements

### Rules Intentionally Ignored

```toml
ignore = [
  "E501",  # Line length (handled by formatter)
  "B008",  # Function default argument warning (too strict for ML/AI use)
]
```

**Explanation**:
- **E501**: Line length is handled by the code formatter, not the linter
- **B008**: Function default argument warnings are disabled because they're too strict for ML/AI use cases where mutable defaults are sometimes necessary

### Configuration

```toml
# Automatically fix issues where possible
fix = true

# Exclude generated or external files
exclude = [
  ".venv",
  "build",
  "dist",
  "__pycache__",
]
```

**Features**:
- **fix = true**: Automatically fixes issues where possible when running ruff
- **exclude**: Excludes virtual environments, build directories, and cache files from linting

### Running Ruff

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Check specific directory
ruff check src/upsonic/agent/
```

These rules ensure consistent code quality across the Upsonic framework while maintaining flexibility for ML/AI development patterns.


