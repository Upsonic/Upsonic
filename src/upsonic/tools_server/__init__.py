from ..server_manager import ServerManager
from multiprocessing import freeze_support

_server_manager = ServerManager(
    app_path="upsonic.tools_server.server.api:app",
    host="localhost",
    port=8086,
    name="tools"
)

def run_tools_server(redirect_output: bool = False):
    """Start the tools server if it's not already running."""
    _server_manager.start(redirect_output=redirect_output)

def run_tools_server_internal(reload: bool = True):
    """Run the tools server directly (for development)"""
    import uvicorn
    uvicorn.run("upsonic.tools_server.server.api:app", host="localhost", port=8086, reload=reload)

def stop_tools_server():
    """Stop the tools server if it's running."""
    _server_manager.stop()

def is_tools_server_running() -> bool:
    """Check if the tools server is currently running."""
    return _server_manager.is_running()

if __name__ == '__main__':
    freeze_support()

__all__ = ["run_tools_server", "stop_tools_server", "is_tools_server_running", "app", "run_tools_server_internal"]
