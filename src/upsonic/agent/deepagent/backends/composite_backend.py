"""
Composite Backend - Route filesystem operations to different backends by path.

Allows using different storage backends for different parts of the filesystem.
For example, use MemoryBackend for /memories/ (persistent) and StateBackend
for /tmp/ (ephemeral).

Routing uses first-registered-wins strategy for matching patterns.
"""

from typing import Dict, List, Optional
from .protocol import BackendProtocol
from .state_backend import StateBackend


class CompositeBackend:
    """
    Composite filesystem backend with path-based routing.
    
    Routes filesystem operations to different backends based on path prefixes.
    Enables hybrid storage strategies where different parts of the filesystem
    use different backends.
    
    Routing Strategy:
    - First registered route that matches wins
    - Prefix matching (e.g., "/memories/" matches "/memories/file.txt")
    - Exact root matching (e.g., "/" only matches root)
    - Default backend used when no route matches
    
    Storage Layout Example:
        Routes:
            "/memories/" -> MemoryBackend (persistent)
            "/tmp/" -> StateBackend (ephemeral)
            Default -> StateBackend (ephemeral)
        
        Files:
            "/memories/notes.txt" -> Uses MemoryBackend
            "/tmp/temp.txt" -> Uses StateBackend
            "/workspace/file.txt" -> Uses default StateBackend
    
    Usage:
        ```python
        from upsonic.storage import SqliteStorage
        
        # Create backends
        memory_backend = MemoryBackend(SqliteStorage("storage.db"))
        state_backend = StateBackend()
        
        # Create composite with routes
        backend = CompositeBackend(
            default=state_backend,
            routes={
                "/memories/": memory_backend,
                "/important/": memory_backend,
                "/tmp/": state_backend
            }
        )
        
        # Files are automatically routed
        await backend.write("/memories/note.txt", "Persistent note")  # -> MemoryBackend
        await backend.write("/tmp/temp.txt", "Temporary file")  # -> StateBackend
        await backend.write("/work/file.txt", "Work file")  # -> default StateBackend
        ```
    """
    
    def __init__(
        self,
        default: BackendProtocol = StateBackend(),
        routes: Optional[Dict[str, BackendProtocol]] = None
    ):
        """
        Initialize the composite backend.
        
        Args:
            default: Default backend for paths that don't match any route
            routes: Dictionary mapping path prefixes to backends.
                   Keys should be absolute paths (start with '/').
                   Routes are checked in registration order (first match wins).
        """
        self.default = default
        
        # Store routes as ordered list for first-match-wins behavior
        self._routes: List[tuple[str, BackendProtocol]] = []
        
        if routes:
            for path_prefix, backend in routes.items():
                self.add_route(path_prefix, backend)
    
    def add_route(self, path_prefix: str, backend: BackendProtocol) -> None:
        """
        Add a routing rule.
        
        Args:
            path_prefix: Path prefix to match (e.g., "/memories/")
            backend: Backend to use for matching paths
            
        Raises:
            ValueError: If path_prefix is invalid
        """
        if not path_prefix:
            raise ValueError("Path prefix cannot be empty")
        
        if not isinstance(path_prefix, str):
            raise ValueError(f"Path prefix must be a string, got {type(path_prefix)}")
        
        if not path_prefix.startswith("/"):
            raise ValueError(f"Path prefix must start with '/': {path_prefix}")
        
        # Ensure trailing slash for directory prefixes (except root)
        if path_prefix != "/" and not path_prefix.endswith("/"):
            path_prefix = path_prefix + "/"
        
        # Add to routes (at the end, maintaining registration order)
        self._routes.append((path_prefix, backend))
    
    def _get_backend_for_path(self, path: str) -> BackendProtocol:
        """
        Get the appropriate backend for a path.
        
        Uses first-match-wins strategy: the first registered route
        that matches the path is used.
        
        Args:
            path: Absolute path to route
            
        Returns:
            Backend that should handle this path
        """
        # Check routes in registration order (first match wins)
        for path_prefix, backend in self._routes:
            if path == path_prefix.rstrip("/") or path.startswith(path_prefix):
                return backend
        
        # No match, use default
        return self.default
    
    async def read(self, path: str) -> str:
        """
        Read file content using the appropriate backend.
        
        Args:
            path: Absolute path to the file
            
        Returns:
            File content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid
            PermissionError: If path validation fails
            OSError: If path is a directory
        """
        backend = self._get_backend_for_path(path)
        return await backend.read(path)
    
    async def write(self, path: str, content: str) -> None:
        """
        Write content using the appropriate backend.
        
        Args:
            path: Absolute path to the file
            content: Content to write
            
        Raises:
            ValueError: If path or content is invalid
            PermissionError: If path validation fails
            OSError: If path is a directory
        """
        backend = self._get_backend_for_path(path)
        await backend.write(path, content)
    
    async def delete(self, path: str) -> None:
        """
        Delete a file using the appropriate backend.
        
        Args:
            path: Absolute path to the file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid
            PermissionError: If path validation fails
            OSError: If path is a directory
        """
        backend = self._get_backend_for_path(path)
        await backend.delete(path)
    
    async def exists(self, path: str) -> bool:
        """
        Check if a path exists using the appropriate backend.
        
        Args:
            path: Absolute path to check
            
        Returns:
            True if path exists, False otherwise
            
        Raises:
            ValueError: If path is invalid
            PermissionError: If path validation fails
        """
        backend = self._get_backend_for_path(path)
        return await backend.exists(path)
    
    async def list_dir(self, path: str = "/") -> List[str]:
        """
        List directory entries using the appropriate backend.
        
        Special handling for root directory: combines entries from all backends.
        For other directories, uses the appropriate backend.
        
        Args:
            path: Absolute directory path
            
        Returns:
            List of entry names (not full paths)
            
        Raises:
            ValueError: If path is invalid
            PermissionError: If path validation fails
        """
        # Special case for root: combine entries from all backends
        if path == "/":
            all_entries = set()
            
            # Get entries from default backend
            default_entries = await self.default.list_dir("/")
            all_entries.update(default_entries)
            
            # Get entries from all routed backends
            for path_prefix, backend in self._routes:
                if path_prefix != "/":
                    # Add the route directory name
                    route_name = path_prefix.strip("/").split("/")[0]
                    all_entries.add(route_name)
            
            return sorted(list(all_entries))
        
        # For other paths, use the appropriate backend
        backend = self._get_backend_for_path(path)
        return await backend.list_dir(path)
    
    async def glob(self, pattern: str) -> List[str]:
        """
        Find files matching a pattern across all backends.
        
        Searches in all backends and combines results.
        Duplicates are removed (though they shouldn't occur with proper routing).
        
        Args:
            pattern: Glob pattern (e.g., "/documents/**/*.txt")
            
        Returns:
            List of matching absolute paths
            
        Raises:
            ValueError: If pattern is invalid
            PermissionError: If path validation fails
        """
        all_matches = set()
        
        # Search in default backend
        default_matches = await self.default.glob(pattern)
        all_matches.update(default_matches)
        
        # Search in all routed backends
        for path_prefix, backend in self._routes:
            # Only search this backend if pattern could match its prefix
            if pattern.startswith(path_prefix) or "**" in pattern or "*" in pattern:
                backend_matches = await backend.glob(pattern)
                all_matches.update(backend_matches)
        
        return sorted(list(all_matches))
    
    def get_route_info(self) -> Dict[str, str]:
        """
        Get information about configured routes.
        
        Returns:
            Dictionary mapping path prefixes to backend class names
        """
        info = {}
        
        for path_prefix, backend in self._routes:
            backend_name = backend.__class__.__name__
            info[path_prefix] = backend_name
        
        info["default"] = self.default.__class__.__name__
        
        return info
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics from all backends.
        
        Returns:
            Dictionary mapping backend names to their stats
        """
        stats = {}
        
        # Get stats from default backend
        if hasattr(self.default, 'get_stats'):
            default_stats = self.default.get_stats()
            stats["default"] = default_stats
        
        # Get stats from routed backends
        for i, (path_prefix, backend) in enumerate(self._routes):
            if hasattr(backend, 'get_stats'):
                backend_stats = backend.get_stats()
                stats[f"route_{i}_{path_prefix}"] = backend_stats
        
        return stats

