import asyncio
import pathlib
import time
from typing import List, Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp import MCPToolkit
from langchain_core.tools import BaseTool




from typing import Any, Dict, List
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr





class MCPToolWrapper(BaseTool):
    """A wrapper for an individual tool managed by the SyncInvocationManager."""
    _manager: Any = PrivateAttr()
    _tool: Any = PrivateAttr()
    
    def __init__(self, tool: BaseTool, manager: "SyncInvocationManager"):
        super().__init__(name=tool.name, description=tool.description)
        self.name = tool.name
        self.description = tool.description
        self._manager = manager
        self._tool = tool

    def _run(self, **kwargs: Any) -> Any:
        """Run the tool synchronously using the SyncInvocationManager."""
        try:
            print(f"Running tool: {self.name} with args: {kwargs}")
            result = self._manager.invoke_tool_sync(self._tool, kwargs)
            if result is None:
                print(f"Tool {self.name} returned no result.")
            else:
                print(f"Tool {self.name} result: {result}")
            return result
        except Exception as e:
            print(f"Error while running tool {self.name}: {e}")
            return None

    async def _arun(self, **kwargs: Any) -> Any:
        """Asynchronous run (if needed), wraps the synchronous call."""
        return self._run(**kwargs)


class MCPToolManager:
    """Manages tools provided by the SyncInvocationManager and converts them into LangChain tools."""
    
    def __init__(self, manager: "SyncInvocationManager"):
        self.manager = manager
        self.tools: List[BaseTool] = []
    
    def load_tools(self) -> List[BaseTool]:
        """Load tools from SyncInvocationManager and wrap them in LangChain-compatible structure."""
        raw_tools = self.manager.get_tools_sync()
     
        self.tools = [MCPToolWrapper(tool, self.manager) for tool in raw_tools]
        return self.tools


class SyncInvocationManager:
    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None):
        self.loop = asyncio.new_event_loop()
        self.server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,

        )
        self.client_ctx = None
        self.client = None
        self.session_ctx = None
        self.session = None
        self.toolkit = None
        self._task = None  # Add this line


    async def _start_async(self):
        # Manually enter the stdio_client context
        self.client_ctx = stdio_client(self.server_params)
        self.client = await self.client_ctx.__aenter__()
        read, write = self.client

        # Manually enter the ClientSession context
        self.session_ctx = ClientSession(read, write)
        self.session = await self.session_ctx.__aenter__()

        self.toolkit = MCPToolkit(session=self.session)
        await self.toolkit.initialize()

    def get_tools_sync(self) -> List[BaseTool]:
        # Now that session is open, just return tools directly
        return self.toolkit.get_tools()

    def invoke_tool_sync(self, tool: BaseTool, input_data: Dict[str, Any]) -> Any:
        try:
            return self.loop.run_until_complete(tool.ainvoke(input_data))
        except Exception as e:
            print(f"Error invoking tool {tool.name}: {e}")
            return None

    def start(self):
        asyncio.set_event_loop(self.loop)
        self._task = self.loop.create_task(self._start_async())
        self.loop.run_until_complete(self._task)

    def stop(self):
        if self._task and not self._task.done():
            cleanup_task = self.loop.create_task(self._stop_async())
            self.loop.run_until_complete(cleanup_task)
        self.loop.close()

    async def _stop_async(self):
        # Exit contexts in the same task and loop they were entered
        if self.session_ctx:
            await self.session_ctx.__aexit__(None, None, None)
        if self.client_ctx:
            await self.client_ctx.__aexit__(None, None, None)




def file_system_tool():
    print("""
          
This is file_system_tool

          """)


    manager = SyncInvocationManager(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", str(pathlib.Path(__file__).parent.parent)])
    manager.start()
    tool_manager = MCPToolManager(manager)
    tools =  tool_manager.load_tools()
    print(tools)
    return tools


def memory_tool():

    print("""
          
This is memory_tool
          
            """)
    

    manager = SyncInvocationManager(command="npx", args=["-y", "@modelcontextprotocol/server-memory"])
    manager.start()
    tool_manager = MCPToolManager(manager)
    tools =  tool_manager.load_tools()
    print(tools)
    return tools


def playwright():

    print("""
          
This is playwright
          
            """)

    manager = SyncInvocationManager(command="npx", args=["-y", "@executeautomation/playwright-mcp-server"])
    manager.start()
    tool_manager = MCPToolManager(manager)
    tools =  tool_manager.load_tools()
    print(tools)
    return tools


def youtube_transcript():

    print("""
          
This is youtube_transcript
          
            """)

    manager = SyncInvocationManager(command="npx", args=["-y", "@kimtaeyoon83/mcp-server-youtube-transcript"])
    manager.start()
    tool_manager = MCPToolManager(manager)
    tools =  tool_manager.load_tools()
    print(tools)
    return tools

def fetch():

    print("""
          
This is fetch
          
            """)

    manager = SyncInvocationManager(command="uvx", args=["mcp-server-fetch"])
    manager.start()
    tool_manager = MCPToolManager(manager)
    tools =  tool_manager.load_tools()
    print(tools)
    return tools




def websearch():

    print("""
          
This is websearch
          
            """)


    manager = SyncInvocationManager(command="npx", args=["-y", "@mzxrai/mcp-webresearch"])
    manager.start()
    tool_manager = MCPToolManager(manager)
    tools =  tool_manager.load_tools()
    print(tools)
    return tools




the_tools_ = None
def mcp_tools():
    global the_tools_
    if the_tools_ is None:
        the_tools_ = file_system_tool()
    return the_tools_
