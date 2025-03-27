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
import threading
import time
import concurrent.futures
import traceback
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

def _start_server(server_func, server_name, redirect_output=True):
    """Start a server"""
    try:
        # Always start the server fresh
        server_func(redirect_output=redirect_output)
        return True
    except Exception as e:
        print(f"\nError starting {server_name} server:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        return False

def run_dev_server(redirect_output=True):
    """Run both main and tools servers for development with maximum parallelism"""
    
    try:
        # Use ThreadPoolExecutor to run both servers in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both server start tasks
            main_future = executor.submit(
                _start_server,
                run_main_server,
                "main",
                redirect_output
            )
            
            tools_future = executor.submit(
                _start_server,
                run_tools_server,
                "tools", 
                redirect_output
            )
            
            # Wait for both to complete with a timeout
            try:
                main_result = main_future.result(timeout=150)
                tools_result = tools_future.result(timeout=150)
                
                if not main_result or not tools_result:
                    # Clean up if either server failed
                    print("\nOne or both servers failed to start. Stopping all servers...")
                    stop_dev_server()
                    raise RuntimeError("Failed to start servers - check above logs for details")
                
                # Add a small delay to ensure servers are fully initialized
                time.sleep(0.5)


                return
                
            except concurrent.futures.TimeoutError:
                print("\nTimeout occurred while starting servers")
                stop_dev_server()
                raise RuntimeError("Timeout waiting for servers to start")
    except Exception as e:
        print("\nUnexpected error in run_dev_server:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        raise

def stop_dev_server():
    """Stop both main and tools servers"""
    stop_main_server()
    stop_tools_server()

if __name__ == '__main__':
    freeze_support()

__all__ = ["Configuration", "Call", "app", "run_main_server", "stop_main_server", 
           "is_main_server_running", "run_main_server_internal", "run_dev_server", "stop_dev_server"]
