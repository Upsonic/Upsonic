"""
Memory Backend - Persistent filesystem storage via Upsonic Storage.

Stores files in the Upsonic Storage system, enabling persistence
across sessions and agent instances. Files survive process restarts
and can be shared across multiple agents using the same storage.

This backend integrates with the existing Storage infrastructure.
"""

import fnmatch
import os
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from upsonic.storage.base import Storage


class FilesystemEntry(BaseModel):
    """
    Represents a file or directory in the persistent filesystem.
    
    Attributes:
        path: Absolute path to the entry
        content: File content (None for directories)
        is_directory: Whether this entry is a directory
        created_at: When the entry was created
        modified_at: When the entry was last modified
        size: Size of the content in bytes
    """
    path: str = Field(description="Absolute path to the entry")
    content: Optional[str] = Field(default=None, description="File content (None for directories)")
    is_directory: bool = Field(default=False, description="Whether this is a directory")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    modified_at: datetime = Field(default_factory=datetime.utcnow, description="Modification timestamp")
    size: int = Field(default=0, description="Content size in bytes")


class MemoryBackend:
    """
    Persistent filesystem backend using Upsonic Storage.
    
    Files are stored in a dedicated collection in the Storage system:
    - Persistent across sessions
    - Survives process restarts
    - Can be shared across agents
    - Suitable for long-term memory and important files
    
    Storage Structure:
        Collection: "deepagent_filesystem"
        Key: File path (e.g., "/documents/report.txt")
        Value: FilesystemEntry model
    
    Usage:
        ```python
        from upsonic.storage import SqliteStorage
        storage = SqliteStorage("agent_storage.db")
        backend = MemoryBackend(storage)
        
        await backend.write("/memory/notes.txt", "Important notes")
        content = await backend.read("/memory/notes.txt")
        ```
    """
    
    COLLECTION_NAME = "deepagent_filesystem"
    
    def __init__(self, storage: "Storage"):
        """
        Initialize the memory backend.
        
        Args:
            storage: Upsonic Storage instance for persistence
        """
        self.storage = storage
        self._initialized = False
        self._cache: Dict[str, FilesystemEntry] = {}  # In-memory cache
    
    async def _init(self) -> None:
        """Lazy initialization - ensure storage is connected."""
        if self._initialized:
            return
        
        # Ensure storage is connected
        if not await self.storage.is_connected_async():
            await self.storage.connect_async()
        
        # Create root directory if it doesn't exist
        root_path = "/"
        root_entry = await self._read_entry(root_path)
        
        if root_entry is None:
            await self._write_entry(
                FilesystemEntry(
                    path=root_path,
                    content=None,
                    is_directory=True
                )
            )
        
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
        
        # Normalize path
        normalized = os.path.normpath(path).replace("\\", "/")
        
        # Prevent path traversal
        if ".." in normalized:
            raise PermissionError(f"Path traversal not allowed: {path}")
        
        # Prevent null bytes
        if "\x00" in path:
            raise PermissionError(f"Null bytes not allowed in path: {path}")
        
        # Check maximum path length
        if len(normalized) > 4096:
            raise ValueError(f"Path too long (max 4096 characters): {len(normalized)}")
        
        # Ensure it starts with /
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        
        # Remove trailing slash except for root
        if len(normalized) > 1 and normalized.endswith("/"):
            normalized = normalized.rstrip("/")
        
        return normalized
    
    def _get_storage_key(self, path: str) -> str:
        """
        Get the storage key for a path.
        
        Args:
            path: Filesystem path
            
        Returns:
            Storage key
        """
        # Use path as key, but encode to handle special characters
        return path
    
    async def _read_entry(self, path: str) -> Optional[FilesystemEntry]:
        """
        Read a filesystem entry from storage.
        
        Args:
            path: Path to read
            
        Returns:
            FilesystemEntry if exists, None otherwise
        """
        # Check cache first
        if path in self._cache:
            return self._cache[path]
        
        # Read from storage
        key = self._get_storage_key(path)
        
        try:
            entry = await self.storage.read_async(key, FilesystemEntry)
            
            # Update cache
            if entry:
                self._cache[path] = entry
            
            return entry
        except Exception:
            # Entry doesn't exist or read failed
            return None
    
    async def _write_entry(self, entry: FilesystemEntry) -> None:
        """
        Write a filesystem entry to storage.
        
        Args:
            entry: Entry to write
        """
        # Update timestamp
        entry.modified_at = datetime.utcnow()
        
        # Update size
        if entry.content:
            entry.size = len(entry.content.encode('utf-8'))
        else:
            entry.size = 0
        
        # Write to storage using upsert
        await self.storage.upsert_async(entry)
        
        # Update cache
        self._cache[entry.path] = entry
    
    async def _delete_entry(self, path: str) -> None:
        """
        Delete a filesystem entry from storage.
        
        Args:
            path: Path to delete
        """
        key = self._get_storage_key(path)
        
        # Delete from storage
        await self.storage.delete_async(key, FilesystemEntry)
        
        # Remove from cache
        if path in self._cache:
            del self._cache[path]
    
    async def _ensure_parent_dirs(self, path: str) -> None:
        """
        Ensure all parent directories exist for a path.
        
        Args:
            path: File path to ensure parents for
        """
        parts = path.split("/")
        
        # Create all parent directories
        current = ""
        for part in parts[:-1]:  # Exclude the file name
            if part:
                current += "/" + part
            elif not current:
                current = "/"
            
            # Check if directory exists
            if current:
                existing = await self._read_entry(current)
                
                if existing is None:
                    # Create directory
                    await self._write_entry(
                        FilesystemEntry(
                            path=current,
                            content=None,
                            is_directory=True
                        )
                    )
    
    async def _list_all_entries(self) -> List[FilesystemEntry]:
        """
        List all entries in the filesystem from storage.
        
        Uses the Storage.list_all_async() method to query all FilesystemEntry objects.
        Results are cached for performance.
        
        Returns:
            List of all filesystem entries from storage
        """
        try:
            # Query all FilesystemEntry objects from storage
            entries = await self.storage.list_all_async(FilesystemEntry)
            
            # Update cache with queried entries
            for entry in entries:
                self._cache[entry.path] = entry
            
            return entries
        except Exception as e:
            # Fallback to cache if query fails
            # This maintains backward compatibility
            return list(self._cache.values())
    
    async def read(self, path: str) -> str:
        """
        Read file content from storage.
        
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
        
        entry = await self._read_entry(path)
        
        if entry is None:
            raise FileNotFoundError(f"File not found: {path}")
        
        if entry.is_directory:
            raise OSError(f"Cannot read directory as file: {path}")
        
        if entry.content is None:
            raise OSError(f"File has no content: {path}")
        
        return entry.content
    
    async def write(self, path: str, content: str) -> None:
        """
        Write content to a file in storage.
        
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
        existing = await self._read_entry(path)
        if existing and existing.is_directory:
            raise OSError(f"Cannot write to directory: {path}")
        
        # Ensure parent directories exist
        await self._ensure_parent_dirs(path)
        
        # Create or update the file
        if existing:
            # Update existing file
            existing.content = content
            await self._write_entry(existing)
        else:
            # Create new file
            await self._write_entry(
                FilesystemEntry(
                    path=path,
                    content=content,
                    is_directory=False
                )
            )
    
    async def delete(self, path: str) -> None:
        """
        Delete a file from storage.
        
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
        
        entry = await self._read_entry(path)
        
        if entry is None:
            raise FileNotFoundError(f"File not found: {path}")
        
        if entry.is_directory:
            raise OSError(f"Cannot delete directory as file: {path}")
        
        await self._delete_entry(path)
    
    async def exists(self, path: str) -> bool:
        """
        Check if a path exists in storage.
        
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
        
        entry = await self._read_entry(path)
        
        return entry is not None
    
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
            search_prefix = path + "/"
        else:
            search_prefix = path
        
        entries = set()
        
        # Get all entries from storage
        # Note: This is simplified - in production you'd want to query by prefix
        all_entries = await self._list_all_entries()
        
        for entry in all_entries:
            stored_path = entry.path
            
            # Skip the directory itself
            if stored_path == path or stored_path == path.rstrip("/"):
                continue
            
            # Check if path is under the directory
            if stored_path.startswith(search_prefix):
                # Get the relative part
                relative = stored_path[len(search_prefix):]
                
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
        
        # Get all entries
        all_entries = await self._list_all_entries()
        
        # Filter files (not directories)
        files = [e for e in all_entries if not e.is_directory]
        
        # Match pattern
        if "**" in pattern:
            # Handle recursive matching
            parts = pattern.split("**")
            
            for entry in files:
                if self._match_pattern_with_recursive(entry.path, parts):
                    matches.append(entry.path)
        else:
            # Simple glob matching
            for entry in files:
                if fnmatch.fnmatch(entry.path, pattern):
                    matches.append(entry.path)
        
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
        Get statistics about the filesystem from cache.
        
        Note: This returns cached statistics. For most accurate stats,
        call _list_all_entries() first to refresh the cache.
        
        Returns:
            Dictionary with stats:
            - total_entries: Total number of files and directories
            - files: Number of files
            - directories: Number of directories
            - total_size: Total size of all files in bytes
        """
        all_entries = list(self._cache.values())
        
        files = sum(1 for e in all_entries if not e.is_directory)
        directories = sum(1 for e in all_entries if e.is_directory)
        total_size = sum(e.size for e in all_entries)
        
        return {
            "total_entries": len(all_entries),
            "files": files,
            "directories": directories,
            "total_size": total_size
        }

