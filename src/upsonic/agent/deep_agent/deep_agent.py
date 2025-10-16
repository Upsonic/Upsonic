import copy
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from upsonic.agent.agent import Agent
from upsonic.agent.deep_agent.state import DeepAgentState
from upsonic.agent.deep_agent.tools import (
    write_todos,
    ls,
    read_file,
    write_file,
    edit_file,
    create_task_tool,
    set_current_deep_agent
)
from upsonic.agent.deep_agent.prompts import (
    WRITE_TODOS_SYSTEM_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT,
    TASK_SYSTEM_PROMPT,
    BASE_AGENT_PROMPT
)

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task
    from upsonic.models import Model


class DeepAgent(Agent):
    """
    Deep Agent with advanced capabilities for complex, multi-step tasks.
    
    DeepAgent extends the base Agent with:
    - **Planning Tool**: write_todos for managing complex task plans
    - **Virtual Filesystem**: ls, read_file, write_file, edit_file for file operations
    - **Subagent System**: Spawn isolated subagents for context quarantine
    - **Enhanced Prompts**: Specialized system prompts for deep reasoning
    
    The DeepAgent maintains a virtual filesystem and todo list that persist
    across the execution, enabling sophisticated multi-step workflows.
    
    Usage:
        ```python
        from upsonic import DeepAgent, Task, Agent
        
        # Basic usage
        agent = DeepAgent("openai/gpt-4o")
        task = Task("Analyze the codebase and create a report")
        result = agent.do(task)
        
        # With custom subagents (Agent instances with names)
        researcher = Agent("openai/gpt-4o", name="researcher", 
                          system_prompt="You are a research expert...")
        code_reviewer = Agent("openai/gpt-4o", name="code-reviewer",
                             system_prompt="You are a code review expert...")
        
        agent = DeepAgent("openai/gpt-4o", subagents=[researcher, code_reviewer])
        
        # With initial files
        agent = DeepAgent("openai/gpt-4o")
        agent.add_file("/app/main.py", "code...")
        task = Task("Review the code")
        result = agent.do(task)
        ```
    
    Attributes:
        deep_agent_state: State containing todos and virtual filesystem
        subagents: List of Agent instances to use as subagents
    """
    
    def __init__(
        self,
        model: Union[str, "Model"] = "openai/gpt-4o",
        *,
        subagents: Optional[List[Agent]] = None,
        instructions: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a Deep Agent.
        
        Args:
            model: Model identifier or Model instance
            subagents: Optional list of Agent instances to use as subagents.
                      Each Agent must have a 'name' attribute set.
                      Example: [Agent(name="researcher", ...), Agent(name="reviewer", ...)]
            instructions: Additional instructions to append to system prompt
            **kwargs: Additional arguments passed to base Agent
        """
        # Store subagents (List of Agent instances)
        self.subagents: List[Agent] = subagents or []
        
        # Build enhanced system prompt
        enhanced_system_prompt = self._build_deep_agent_system_prompt(
            instructions or kwargs.get('system_prompt', '')
        )
        kwargs['system_prompt'] = enhanced_system_prompt
        
        # Initialize base agent
        super().__init__(model, **kwargs)
        
        # Initialize deep agent state
        self.deep_agent_state = DeepAgentState()
    
    def _build_deep_agent_system_prompt(self, user_instructions: str) -> str:
        """
        Build the comprehensive system prompt for Deep Agent.
        
        Combines user instructions with Deep Agent prompts for:
        - Base agent capabilities
        - Todo management (write_todos)
        - Filesystem operations
        - Subagent delegation
        
        Args:
            user_instructions: User-provided instructions
        
        Returns:
            Complete system prompt string
        """
        prompt_parts = []
        
        if user_instructions:
            prompt_parts.append(user_instructions)
        
        # Base agent prompt
        prompt_parts.append(BASE_AGENT_PROMPT)
        
        # Deep agent capability prompts
        prompt_parts.append(WRITE_TODOS_SYSTEM_PROMPT)
        prompt_parts.append(FILESYSTEM_SYSTEM_PROMPT)
        prompt_parts.append(TASK_SYSTEM_PROMPT)
        
        return "\n\n".join(prompt_parts)
    
    def _setup_tools(self, task: "Task") -> None:
        """
        Setup tools with Deep Agent capabilities.
        
        This extends the base _setup_tools to inject:
        - Planning tool (write_todos)
        - Filesystem tools (ls, read_file, write_file, edit_file)
        - Task delegation tool (task) for spawning subagents
        
        Note: Tools are set via Task, not on the agent itself.
        """
        # Set this agent as the current agent for tools to access
        set_current_deep_agent(self)
        
        deep_tools = [
            write_todos,
            ls,
            read_file,
            write_file,
            edit_file
        ]
        
        # Create task delegation tool with subagent info
        subagent_descriptions = self._get_subagent_descriptions()
        task_tool = create_task_tool(self, subagent_descriptions)
        deep_tools.append(task_tool)
        
        task_tools = task.tools if task.tools else []
        combined_tools = deep_tools + task_tools
        
        task.tools = combined_tools
        
        super()._setup_tools(task)
    
    def _get_subagent_descriptions(self) -> List[str]:
        """
        Get formatted descriptions of available subagents.
        
        Returns:
            List of description strings in format "- name: description"
        """
        descriptions = []
        for subagent in self.subagents:
            name = subagent.name if hasattr(subagent, 'name') and subagent.name else "unnamed"
            
            description = "Subagent for specialized tasks"
            if hasattr(subagent, 'system_prompt') and subagent.system_prompt:
                description = subagent.system_prompt
            
            descriptions.append(f"- {name}: {description}")
        return descriptions
    
    async def _execute_subagent(
        self,
        description: str,
        subagent_type: str
    ) -> str:
        """
        Execute a subagent to handle an isolated task.
        
        This uses an existing Agent instance with:
        - Isolated context and state
        - Its own model and configuration
        - No memory sharing for isolation
        
        Args:
            description: Task description for the subagent
            subagent_type: Type of subagent ('general-purpose' or agent name)
        
        Returns:
            The final output from the subagent
        """
        from upsonic.tasks.tasks import Task
        
        if subagent_type == 'general-purpose':
            subagent = Agent(
                model=self.model,
                system_prompt=BASE_AGENT_PROMPT,
                memory=None,
                debug=self.debug,
                show_tool_calls=self.show_tool_calls,
                tool_call_limit=self.tool_call_limit
            )
            
            task = Task(description)
        else:
            subagent = None
            for agent in self.subagents:
                if hasattr(agent, 'name') and agent.name == subagent_type:
                    subagent = agent
                    break
            
            if subagent is None:
                return f"Error: Subagent '{subagent_type}' not found"
            
            task = Task(description)
        
        try:
            result = await subagent.do_async(task)
            return str(result)
        except Exception as e:
            return f"Error executing subagent: {str(e)}"
    
    def add_subagent(self, agent: Agent) -> None:
        """
        Add a subagent to this Deep Agent.
        
        The agent must have a 'name' attribute set.
        
        Args:
            agent: Agent instance to use as subagent (must have 'name' attribute)
        """
        if not hasattr(agent, 'name') or not agent.name:
            raise ValueError("Subagent must have a 'name' attribute set")
        self.subagents.append(agent)
    
    def get_todos(self) -> List[Dict[str, Any]]:
        """
        Get the current todo list.
        
        Returns:
            List of todo dictionaries with 'content' and 'status' fields
        """
        return [
            {"content": todo.content, "status": todo.status}
            for todo in self.deep_agent_state.todos
        ]
    
    def get_files(self) -> Dict[str, str]:
        """
        Get the current virtual filesystem.
        
        Returns:
            Dictionary mapping file paths to content
        """
        return dict(self.deep_agent_state.files)
    
    def set_files(self, files: Dict[str, str]) -> None:
        """
        Set the virtual filesystem.
        
        Useful for providing initial files to the agent.
        
        Args:
            files: Dictionary mapping file paths to content
        """
        self.deep_agent_state.files = files
    
    def add_file(self, file_path: str, content: str) -> None:
        """
        Add a single file to the virtual filesystem.
        
        Args:
            file_path: Path for the file
            content: Content of the file
        """
        self.deep_agent_state.files[file_path] = content


# Alias for convenience
create_deep_agent = DeepAgent
