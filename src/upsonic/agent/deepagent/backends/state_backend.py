import fnmatch
import os
from typing import Dict, List, Optional


class StateBackend:
    """
    Ephemeral filesystem backend using in-memory storage.
    
    Files are stored as a dictionary in the agent instance:
    - Fast operations (no I/O)
    - No persistence across sessions
    - Files persist across tasks within same agent instance
    - Suitable for temporary working files
    
    Storage Structure:
        {
            "/file.txt": "content",
            "/documents/report.txt": "content",
            "/documents/": None  # Directory marker
        }
    
    Usage:
        ```python
        backend = StateBackend()
        await backend.write("/test.txt", "Hello World")
        content = await backend.read("/test.txt")
        ```
    """
    
    def __init__(self):
        """Initialize the state backend with empty filesystem."""
        self._filesystem: Dict[str, Optional[str]] = {}
        self._initialized = False
    
    async def _init(self) -> None:
        """Lazy initialization - create root directory."""
        if self._initialized:
            return
        
        # Create root directory marker
        self._filesystem["/"] = None
        self._initialized = True
    
    def _validate_path(self, path: str) -> str:
        """
        Validate and normalize a filesystem path.
        
        Args:
            path: Path to validate
            
        Returns:
            Normalized path
            
        Raises:
            ValueError: If path is invalid
            PermissionError: If path contains security violations
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        if not isinstance(path, str):
            raise ValueError(f"Path must be a string, got {type(path)}")
        
        # Must be absolute
        if not path.startswith("/"):
            raise ValueError(f"Path must be absolute (start with '/'): {path}")
        
        # Prevent path traversal BEFORE normalization
        if ".." in path:
            raise PermissionError(f"Path traversal not allowed: {path}")
        
        # Prevent null bytes
        if "\x00" in path:
            raise PermissionError(f"Null bytes not allowed in path: {path}")
        
        # Normalize path
        normalized = os.path.normpath(path).replace("\\", "/")
        
        # Check again after normalization
        if ".." in normalized:
            raise PermissionError(f"Path traversal not allowed: {path}")
        
        # Check maximum path length (4096 is a common limit)
        if len(normalized) > 4096:
            raise ValueError(f"Path too long (max 4096 characters): {len(normalized)}")
        
        # Ensure it starts with /
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        
        # Remove trailing slash except for root
        if len(normalized) > 1 and normalized.endswith("/"):
            normalized = normalized.rstrip("/")
        
        return normalized
    
    def _ensure_parent_dirs(self, path: str) -> None:
        """
        Ensure all parent directories exist for a path.
        
        Args:
            path: File path to ensure parents for
        """
        # Get parent directory
        parts = path.split("/")
        
        # Create all parent directories
        current = ""
        for part in parts[:-1]:  # Exclude the file name
            if part:  # Skip empty parts
                current += "/" + part
            elif not current:  # Root
                current = "/"
            
            # Create directory marker if it doesn't exist
            if current and current not in self._filesystem:
                self._filesystem[current] = None
    
    def _is_directory(self, path: str) -> bool:
        """
        Check if a path is a directory.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is a directory, False otherwise
        """
        return path in self._filesystem and self._filesystem[path] is None
    
    async def read(self, path: str) -> str:
        """
        Read file content from memory.
        
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
        await self._init()
        
        path = self._validate_path(path)
        
        if path not in self._filesystem:
            raise FileNotFoundError(f"File not found: {path}")
        
        content = self._filesystem[path]
        
        if content is None:
            raise OSError(f"Cannot read directory as file: {path}")
        
        return content
    
    async def write(self, path: str, content: str) -> None:
        """
        Write content to a file in memory.
        
        Args:
            path: Absolute path to the file
            content: Content to write
            
        Raises:
            ValueError: If path or content is invalid
            PermissionError: If path validation fails
            OSError: If path is a directory
        """
        await self._init()
        
        path = self._validate_path(path)
        
        if not isinstance(content, str):
            raise ValueError(f"Content must be a string, got {type(content)}")
        
        # Check if path is an existing directory
        if self._is_directory(path):
            raise OSError(f"Cannot write to directory: {path}")
        
        # Ensure parent directories exist
        self._ensure_parent_dirs(path)
        
        # Write the file
        self._filesystem[path] = content
    
    async def delete(self, path: str) -> None:
        """
        Delete a file from memory.
        
        Args:
            path: Absolute path to the file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid
            PermissionError: If path validation fails
            OSError: If path is a directory
        """
        await self._init()
        
        path = self._validate_path(path)
        
        if path not in self._filesystem:
            raise FileNotFoundError(f"File not found: {path}")
        
        if self._is_directory(path):
            raise OSError(f"Cannot delete directory as file: {path}")
        
        del self._filesystem[path]
    
    async def exists(self, path: str) -> bool:
        """
        Check if a path exists in memory.
        
        Args:
            path: Absolute path to check
            
        Returns:
            True if path exists, False otherwise
            
        Raises:
            ValueError: If path is invalid
            PermissionError: If path validation fails
        """
        await self._init()
        
        path = self._validate_path(path)
        
        return path in self._filesystem
    
    async def list_dir(self, path: str = "/") -> List[str]:
        """
        List entries in a directory.
        
        Args:
            path: Absolute directory path
            
        Returns:
            List of entry names (not full paths)
            
        Raises:
            ValueError: If path is invalid
            PermissionError: If path validation fails
        """
        await self._init()
        
        path = self._validate_path(path)
        
        # Ensure path is treated as directory
        if path != "/" and not path.endswith("/"):
            path = path + "/"
        
        entries = set()
        
        for stored_path in self._filesystem.keys():
            # Skip the directory itself
            if stored_path == path or stored_path == path.rstrip("/"):
                continue
            
            # Check if path is under the directory
            if stored_path.startswith(path):
                # Get the relative part
                relative = stored_path[len(path):]
                
                # Get the first component
                first_component = relative.split("/")[0]
                
                if first_component:
                    entries.add(first_component)
        
        return sorted(list(entries))
    
    async def glob(self, pattern: str) -> List[str]:
        """
        Find files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "/documents/**/*.txt")
            
        Returns:
            List of matching absolute paths
            
        Raises:
            ValueError: If pattern is invalid
            PermissionError: If path validation fails
        """
        await self._init()
        
        # Validate pattern
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        
        if not isinstance(pattern, str):
            raise ValueError(f"Pattern must be a string, got {type(pattern)}")
        
        # Prevent path traversal in pattern
        if ".." in pattern:
            raise PermissionError(f"Path traversal not allowed in pattern: {pattern}")
        
        matches = []
        
        # Convert pattern to regex-style matching
        # Handle ** for recursive directory matching
        if "**" in pattern:
            # Split by **
            parts = pattern.split("**")
            
            for stored_path in self._filesystem.keys():
                # Skip directories
                if self._filesystem[stored_path] is None:
                    continue
                
                # Check if path matches pattern with ** expansion
                if self._match_pattern_with_recursive(stored_path, parts):
                    matches.append(stored_path)
        else:
            # Simple glob matching (no ** recursion)
            for stored_path in self._filesystem.keys():
                # Skip directories
                if self._filesystem[stored_path] is None:
                    continue
                
                # For non-recursive patterns, ensure we don't match across directory levels
                # unless the pattern explicitly includes directory separators
                if "*" in pattern and "**" not in pattern:
                    # Count directory levels in pattern
                    pattern_depth = pattern.count("/")
                    path_depth = stored_path.count("/")
                    
                    # If pattern is like "/dir/*.txt", it should only match files directly in /dir
                    # not in /dir/subdir/file.txt
                    if pattern_depth != path_depth:
                        continue
                
                if fnmatch.fnmatch(stored_path, pattern):
                    matches.append(stored_path)
        
        return sorted(matches)
    
    def _match_pattern_with_recursive(self, path: str, pattern_parts: List[str]) -> bool:
        """
        Match a path against a pattern with ** (recursive directory matching).
        
        Args:
            path: Path to match
            pattern_parts: Pattern split by **
            
        Returns:
            True if path matches pattern, False otherwise
        """
        if len(pattern_parts) == 1:
            # No ** in pattern
            return fnmatch.fnmatch(path, pattern_parts[0])
        
        # For pattern like "/documents/**/*.txt"
        # pattern_parts = ["/documents/", "/*.txt"]
        # Should match:
        #   - /documents/report.txt (** matches zero directories)
        #   - /documents/subfolder/data.txt (** matches one directory)
        #   - /documents/a/b/c/file.txt (** matches multiple directories)
        
        # First part: must match the start (prefix before **)
        if pattern_parts[0]:
            prefix = pattern_parts[0].rstrip("/")
            if prefix and not path.startswith(prefix):
                return False
            # Get remaining path after prefix
            if prefix:
                if path == prefix:
                    remaining = ""
                elif path.startswith(prefix + "/"):
                    remaining = path[len(prefix):]
                else:
                    return False
            else:
                remaining = path
        else:
            remaining = path
        
        # Last part: must match the suffix (after **)
        if pattern_parts[-1]:
            suffix_pattern = pattern_parts[-1].lstrip("/")
            if suffix_pattern:
                # The suffix can have wildcards, so use fnmatch
                # The remaining path must end with something matching the suffix pattern
                if not fnmatch.fnmatch(remaining.lstrip("/"), "**/" + suffix_pattern) and \
                   not fnmatch.fnmatch(remaining.lstrip("/"), suffix_pattern):
                    # Try matching with any number of directory levels
                    # Split remaining into parts and try to match suffix at different positions
                    parts = remaining.strip("/").split("/")
                    matched = False
                    for i in range(len(parts)):
                        suffix_path = "/".join(parts[i:])
                        if fnmatch.fnmatch(suffix_path, suffix_pattern.lstrip("/")):
                            matched = True
                            break
                    if not matched:
                        return False
        
        return True
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the filesystem.
        
        Returns:
            Dictionary with stats:
            - total_entries: Total number of files and directories
            - files: Number of files
            - directories: Number of directories
            - total_size: Total size of all files in bytes
        """
        files = 0
        directories = 0
        total_size = 0
        
        for path, content in self._filesystem.items():
            if content is None:
                directories += 1
            else:
                files += 1
                total_size += len(content.encode('utf-8'))
        
        return {
            "total_entries": len(self._filesystem),
            "files": files,
            "directories": directories,
            "total_size": total_size
        }

