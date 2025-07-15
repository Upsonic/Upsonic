from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.mcp import MCPServerStdio, MCPServerSSE

from .agent_tool_register import agent_tool_register
from ..utils.error_wrapper import upsonic_error_handler
from .model import get_agent_model

@upsonic_error_handler(max_retries=2, show_error_details=True)
async def agent_create(llm_model, single_task):

    agent_model = get_agent_model(llm_model)

    mcp_servers = []
    tools_to_remove = []

    if len(single_task.tools) > 0:
        # For loop through the tools
        for tool in single_task.tools:

            if isinstance(tool, type):
                
                # Check if it's an MCP SSE server (has url property)
                if hasattr(tool, 'url'):
                    url = getattr(tool, 'url')
                    print(url)
                    
                    the_mcp_server = MCPServerSSE(url)
                    mcp_servers.append(the_mcp_server)
                    tools_to_remove.append(tool)
                
                # Check if it's a normal MCP server (has command property)
                elif hasattr(tool, 'command'):
                    # Some times the env is not dict at that situations we need to handle that
                    if hasattr(tool, 'env') and isinstance(tool.env, dict):
                        env = tool.env
                    else:
                        env = {}

                    command = getattr(tool, 'command', None)
                    args = getattr(tool, 'args', [])

                    the_mcp_server = MCPServerStdio(
                        command,
                        args=args,
                        env=env,
                    )

                    mcp_servers.append(the_mcp_server)
                    tools_to_remove.append(tool)

        # Remove MCP tools from the tools list
        for tool in tools_to_remove:
            single_task.tools.remove(tool)

    the_agent = PydanticAgent(agent_model, output_type=single_task.response_format, system_prompt="", end_strategy="exhaustive", retries=5, mcp_servers=mcp_servers)


    agent_tool_register(the_agent, single_task)

    return the_agent