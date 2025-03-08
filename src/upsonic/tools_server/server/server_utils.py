import logging
from typing import Dict, Any, List, Optional

# Global dictionary to store server instances
# Key: (name, command, tuple(args), frozenset(env.items()))
# Value: Server instance
_server_instances = {}

async def cleanup_all_servers():
    """
    Clean up all server instances.
    This should be called when the application is shutting down.
    """
    if not _server_instances:
        logging.info("No server instances to clean up")
        return
        
    logging.info(f"Cleaning up {len(_server_instances)} server instances")
    # We need to import Server here to avoid circular imports
    # This is safe because cleanup_all_servers is only called during shutdown
    for server in list(_server_instances.values()):
        try:
            await server.cleanup()
        except Exception as e:
            logging.error(f"Error cleaning up server {server.name}: {e}")
    
    # Clear the dictionary just to be sure
    _server_instances.clear()
    logging.info("All server instances cleaned up") 