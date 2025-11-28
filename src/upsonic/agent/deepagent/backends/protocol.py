"""
Backend Protocol - Abstract interface for filesystem backends.

This protocol defines the contract that all filesystem backends must implement.
It provides the core operations needed for a virtual filesystem.
"""

from typing import Protocol, List, runtime_checkable


@runtime_checkable
class BackendProtocol(Protocol):
    """
    Abstract interface for filesystem storage backends.
    
    All backends must implement these async methods to provide
    filesystem operations for DeepAgent.
    
    Path Conventions:
    - All paths must be absolute (start with '/')
    - Path separators are always '/'
    - No path traversal allowed (../)
    - Case-sensitive paths
    
    Error Handling:
    - FileNotFoundError: When accessing non-existent files/directories
    - PermissionError: When path validation fails
    - ValueError: When invalid arguments are provided
    - OSError: When other filesystem operations fail
    """
    
    async def read(self, path: str) -> str:
        """
        Read the complete content of a file.
        
        Args:
            path: Absolute path to the file (must start with '/')
            
        Returns:
            The complete file content as a string
            
        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If path validation fails
            ValueError: If path is invalid
            OSError: If read operation fails
            
        Example:
            content = await backend.read("/documents/report.txt")
        """
        ...
    
    async def write(self, path: str, content: str) -> None:
        """
        Write content to a file, creating it if it doesn't exist.
        
        This operation will:
        - Create the file if it doesn't exist
        - Overwrite the file if it exists
        - Create parent directories if needed
        
        Args:
            path: Absolute path to the file (must start with '/')
            content: Content to write to the file
            
        Raises:
            PermissionError: If path validation fails
            ValueError: If path or content is invalid
            OSError: If write operation fails
            
        Example:
            await backend.write("/documents/report.txt", "Report content")
        """
        ...
    
    async def delete(self, path: str) -> None:
        """
        Delete a file.
        
        Args:
            path: Absolute path to the file (must start with '/')
            
        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If path validation fails
            ValueError: If path is invalid
            OSError: If delete operation fails
            
        Example:
            await backend.delete("/tmp/temp_file.txt")
        """
        ...
    
    async def exists(self, path: str) -> bool:
        """
        Check if a file or directory exists.
        
        Args:
            path: Absolute path to check (must start with '/')
            
        Returns:
            True if the path exists, False otherwise
            
        Raises:
            PermissionError: If path validation fails
            ValueError: If path is invalid
            
        Example:
            if await backend.exists("/documents/report.txt"):
                print("File exists")
        """
        ...
    
    async def list_dir(self, path: str = "/") -> List[str]:
        """
        List all entries in a directory.
        
        Returns files and subdirectories directly under the given path.
        Does not recurse into subdirectories.
        
        Args:
            path: Absolute directory path (must start with '/', defaults to '/')
            
        Returns:
            List of entry names (not full paths) in the directory.
            Returns empty list if directory is empty or doesn't exist.
            
        Raises:
            PermissionError: If path validation fails
            ValueError: If path is invalid
            OSError: If list operation fails
            
        Example:
            entries = await backend.list_dir("/documents")
            # Returns: ["report.txt", "notes.txt", "subfolder"]
        """
        ...
    
    async def glob(self, pattern: str) -> List[str]:
        """
        Find all files matching a glob pattern.
        
        Supports standard glob patterns:
        - * matches any characters in a filename
        - ** matches any characters including directory separators
        - ? matches a single character
        - [abc] matches any character in the set
        
        Args:
            pattern: Glob pattern (e.g., "/documents/**/*.txt")
            
        Returns:
            List of absolute paths matching the pattern.
            Returns empty list if no matches found.
            
        Raises:
            PermissionError: If path validation fails
            ValueError: If pattern is invalid
            OSError: If glob operation fails
            
        Example:
            files = await backend.glob("/documents/**/*.txt")
            # Returns: ["/documents/report.txt", "/documents/notes/file.txt"]
        """
        ...

