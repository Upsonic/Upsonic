"""
Deep Agent Tools

This module implements the core tools for Deep Agent functionality:
- Planning tool (write_todos) for task management
- Filesystem tools (ls, read_file, write_file, edit_file) for virtual file operations
- Task delegation tool for spawning subagents

These tools work with the DeepAgent state that is stored in the agent instance.
The state is accessed through the agent reference that will be injected during setup.
"""

from typing import List, Any
from upsonic.tools.config import tool
from upsonic.agent.deep_agent.state import Todo
from upsonic.agent.deep_agent.prompts import (
    WRITE_TODOS_TOOL_DESCRIPTION,
    LIST_FILES_TOOL_DESCRIPTION,
    READ_FILE_TOOL_DESCRIPTION,
    WRITE_FILE_TOOL_DESCRIPTION,
    EDIT_FILE_TOOL_DESCRIPTION,
    TASK_TOOL_DESCRIPTION
)


# Global reference to the current deep agent instance
# This will be set by DeepAgent during tool setup
_current_deep_agent = None


def set_current_deep_agent(agent):
    """Set the current deep agent instance for tools to access."""
    global _current_deep_agent
    _current_deep_agent = agent


def get_current_deep_agent():
    """Get the current deep agent instance."""
    return _current_deep_agent



@tool(
    sequential=False,
    show_result=False,
    docstring_format='google',
    timeout=60.0
)
def write_todos(todos: List[Todo]) -> str:
    """
    Create and manage a structured task list for the current work session.
    
    Use this tool to track progress, organize complex tasks, and demonstrate
    thoroughness. Only use for complex multi-step tasks (3+ steps).
    
    Args:
        todos: List of Todo objects with 'content' and 'status' fields.
               Status can be: 'pending', 'in_progress', or 'completed'.
    
    Returns:
        Confirmation message with the updated todo list.
    
    Examples:
        >>> write_todos([
        ...     Todo(content="Research topic", status="in_progress"),
        ...     Todo(content="Write report", status="pending")
        ... ])
        "Updated todo list with 2 items"
    """
    # The framework already converted dicts to Todo objects for us
    # Update the agent's state directly
    agent = get_current_deep_agent()
    if agent:
        agent.deep_agent_state.todos = todos
    
    # Format the todos for display
    todo_str = "\n".join([f"- [{t.status}] {t.content}" for t in todos])
    return f"Updated todo list with {len(todos)} items:\n{todo_str}"


# Override the docstring with the detailed prompt description
write_todos.__doc__ = WRITE_TODOS_TOOL_DESCRIPTION



@tool(
    sequential=False,
    show_result=False,
    docstring_format='google',
    timeout=60.0
)
def ls() -> List[str]:
    """
    List all files in the virtual filesystem.
    
    Returns a list of all file paths currently stored in the virtual filesystem.
    Very useful for exploring the file system before reading or editing files.
    
    Returns:
        List of file paths in the virtual filesystem.
    
    Examples:
        >>> ls()
        ['/app/main.py', '/app/config.json', '/docs/README.md']
    """
    agent = get_current_deep_agent()
    if agent:
        return list(agent.deep_agent_state.files.keys())
    return []


ls.__doc__ = LIST_FILES_TOOL_DESCRIPTION


@tool(
    sequential=False,
    show_result=False,
    docstring_format='google',
    timeout=60.0
)
def read_file(
    file_path: str,
    offset: int = 0,
    limit: int = 2000
) -> str:
    """
    Read a file from the virtual filesystem.
    
    Reads file content with optional line offset and limit for large files.
    Returns content in cat -n format with line numbers starting at 1.
    
    Args:
        file_path: Absolute path to the file to read.
        offset: Line number to start reading from (0-based).
        limit: Maximum number of lines to read.
    
    Returns:
        File content with line numbers, or error message if file not found.
    
    Examples:
        >>> read_file('/app/main.py')
        "     1\\tdef main():\\n     2\\t    print('Hello')\\n"
    """
    agent = get_current_deep_agent()
    if not agent:
        return f"Error: File '{file_path}' not found"
    
    filesystem = agent.deep_agent_state.files
    
    if file_path not in filesystem:
        return f"Error: File '{file_path}' not found"
    
    # Get file content
    content = filesystem[file_path]
    
    # Handle empty file
    if not content or content.strip() == "":
        return "System reminder: File exists but has empty contents"
    
    # Split content into lines
    lines = content.splitlines()
    
    # Apply line offset and limit
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))
    
    # Handle case where offset is beyond file length
    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
    
    # Format output with line numbers (cat -n format)
    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i]
        
        # Truncate lines longer than 2000 characters
        if len(line_content) > 2000:
            line_content = line_content[:2000]
        
        # Line numbers start at 1, so add 1 to the index
        line_number = i + 1
        result_lines.append(f"{line_number:6d}\t{line_content}")
    
    return "\n".join(result_lines)


read_file.__doc__ = READ_FILE_TOOL_DESCRIPTION


@tool(
    sequential=False,
    show_result=False,
    docstring_format='google',
    timeout=60.0
)
def write_file(
    file_path: str,
    content: str
) -> str:
    """
    Write content to a file in the virtual filesystem.
    
    Creates a new file or overwrites an existing file with the provided content.
    Prefer editing existing files when possible.
    
    Args:
        file_path: Absolute path where the file should be written.
        content: Content to write to the file.
    
    Returns:
        Confirmation message.
    
    Examples:
        >>> write_file('/app/config.json', '{"debug": true}')
        "Successfully wrote to file '/app/config.json'"
    """
    agent = get_current_deep_agent()
    if not agent:
        return f"Error: Unable to write to file '{file_path}'"
    
    agent.deep_agent_state.files[file_path] = content
    return f"Successfully wrote to file '{file_path}'"


write_file.__doc__ = WRITE_FILE_TOOL_DESCRIPTION


@tool(
    sequential=False,
    show_result=False,
    docstring_format='google',
    timeout=60.0
)
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False
) -> str:
    """
    Perform exact string replacement in a file.
    
    Replaces old_string with new_string in the specified file. The edit will
    fail if old_string is not unique unless replace_all is True.
    
    Args:
        file_path: Absolute path to the file to edit.
        old_string: Exact string to find and replace (must be unique unless replace_all=True).
        new_string: String to replace old_string with.
        replace_all: If True, replace all occurrences; if False, require uniqueness.
    
    Returns:
        Confirmation message or error if string not found/not unique.
    
    Examples:
        >>> edit_file('/app/main.py', 'old_var', 'new_var', replace_all=True)
        "Successfully replaced 3 instance(s) in '/app/main.py'"
    """
    agent = get_current_deep_agent()
    if not agent:
        return f"Error: File '{file_path}' not found"
    
    filesystem = agent.deep_agent_state.files
    
    # Check if file exists
    if file_path not in filesystem:
        return f"Error: File '{file_path}' not found"
    
    # Get current file content
    content = filesystem[file_path]
    
    # Check if old_string exists in the file
    if old_string not in content:
        return f"Error: String not found in file: '{old_string}'"
    
    # If not replace_all, check for uniqueness
    if not replace_all:
        occurrences = content.count(old_string)
        if occurrences > 1:
            return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        elif occurrences == 0:
            return f"Error: String not found in file: '{old_string}'"
    
    # Perform the replacement
    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacement_count = content.count(old_string)
        result_msg = f"Successfully replaced {replacement_count} instance(s) in '{file_path}'"
    else:
        new_content = content.replace(old_string, new_string, 1)
        result_msg = f"Successfully replaced string in '{file_path}'"
    
    # Update the filesystem
    filesystem[file_path] = new_content
    
    return result_msg


edit_file.__doc__ = EDIT_FILE_TOOL_DESCRIPTION



def create_task_tool(
    deep_agent_instance: Any,
    subagent_descriptions: List[str]
) -> Any:
    """
    Create a task delegation tool that spawns subagents.
    
    This factory function creates a tool that's bound to a specific DeepAgent
    instance and knows about available subagents.
    
    Args:
        deep_agent_instance: The parent DeepAgent instance
        subagent_descriptions: List of subagent description strings
    
    Returns:
        A configured task delegation tool
    """
    # Format the other agents string for the prompt
    other_agents_str = "\n".join(subagent_descriptions)
    
    @tool(
        sequential=False,
        show_result=False,
        docstring_format='google',
        timeout=60.0
    )
    async def task(
        description: str,
        subagent_type: str
    ) -> str:
        """
        Launch an ephemeral subagent to handle complex, multi-step independent tasks.
        
        Args:
            description: Detailed task description for the subagent to complete autonomously.
            subagent_type: Type of subagent to use (e.g., 'general-purpose', custom subagent names).
        
        Returns:
            The final result from the subagent's execution.
        """
        # Validate subagent type
        available_types = ['general-purpose'] + [
            sa.name for sa in deep_agent_instance.subagents
        ]
        
        if subagent_type not in available_types:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {available_types}"
        
        # Execute the subagent
        result = await deep_agent_instance._execute_subagent(
            description=description,
            subagent_type=subagent_type
        )
        
        return result
    
    # Set the dynamic docstring with available subagents
    task.__doc__ = TASK_TOOL_DESCRIPTION.format(other_agents=other_agents_str)
    
    return task


__all__ = [
    'write_todos',
    'ls',
    'read_file',
    'write_file',
    'edit_file',
    'create_task_tool',
    'set_current_deep_agent',
    'get_current_deep_agent'
]
