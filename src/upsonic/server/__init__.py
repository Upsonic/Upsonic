from ..storage.configuration import Configuration
from .level_one.call import Call
from ..server_manager import ServerManager

from .api import app
from .level_one.server.server import *
from .level_two.server.server import *
from .storage.server.server import *
from .tools.server import *
from .markdown.server.server import *
from .others.server.server import *

import warnings
from multiprocessing import freeze_support
from ..tools_server import run_tools_server, stop_tools_server, is_tools_server_running

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

_server_manager = ServerManager(
    app_path="upsonic.server.api:app",
    host="localhost",
    port=7541,
    name="main"
)

def run_main_server(redirect_output: bool = False):
    """Start the main server if it's not already running."""
    _server_manager.start(redirect_output=redirect_output)

def run_main_server_internal(reload: bool = True):
    """Run the main server directly (for development)"""
    import uvicorn
    uvicorn.run("upsonic.server.api:app", host="0.0.0.0", port=7541, reload=reload)

def stop_main_server():
    """Stop the main server if it's running."""
    _server_manager.stop()

def is_main_server_running() -> bool:
    """Check if the main server is currently running."""
    return _server_manager.is_running()

def run_dev_server(redirect_output=True):
    """Run both main and tools servers for development"""
    run_main_server(redirect_output=redirect_output)
    run_tools_server(redirect_output=redirect_output)
    import time
    while not is_main_server_running() or not is_tools_server_running():
        time.sleep(0.1)

def stop_dev_server():
    """Stop both main and tools servers"""
    stop_main_server()
    stop_tools_server()

if __name__ == '__main__':
    freeze_support()

__all__ = ["Configuration", "Call", "app", "run_main_server", "stop_main_server", 
           "is_main_server_running", "run_main_server_internal", "run_dev_server", "stop_dev_server"]
