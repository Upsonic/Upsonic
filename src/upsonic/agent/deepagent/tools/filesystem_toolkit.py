"""
Filesystem ToolKit - Complete filesystem operations for DeepAgent.

Provides all filesystem tools with proper backend integration,
security validation, and comprehensive error handling.
"""

from typing import Optional, Literal, Set
from upsonic.tools import tool, ToolKit
from upsonic.agent.deepagent.backends import BackendProtocol
from upsonic.agent.deepagent.constants import (
    LIST_FILES_TOOL_DESCRIPTION,
    READ_FILE_TOOL_DESCRIPTION,
    WRITE_FILE_TOOL_DESCRIPTION,
    EDIT_FILE_TOOL_DESCRIPTION,
    GLOB_TOOL_DESCRIPTION,
    GREP_TOOL_DESCRIPTION,
)


class FilesystemToolKit(ToolKit):
    """
    Comprehensive filesystem toolkit for DeepAgent.
    
    Provides 6 essential filesystem operations:
    - ls: List directory contents
    - read_file: Read files with pagination support
    - write_file: Create/overwrite files
    - edit_file: Exact string replacement with read tracking
    - glob: Pattern-based file finding
    - grep: Search for text within files
    
    Features:
    - Backend abstraction for flexible storage
    - Read tracking for edit safety
    - Pagination for large results
    - Production-grade error handling
    - Security validation through backend
    
    Usage:
        ```python
        from upsonic.agent.deepagent.backends import StateBackend
        from upsonic.agent.deepagent.tools import FilesystemToolKit
        
        backend = StateBackend()
        toolkit = FilesystemToolKit(backend)
        
        agent = Agent(model="openai/gpt-4o", tools=[toolkit])
        ```
    """
    
    def __init__(self, backend: BackendProtocol):
        """
        Initialize the filesystem toolkit.
        
        Args:
            backend: Backend implementing BackendProtocol for storage
        """
        self.backend = backend
        # Track which files have been read for edit_file enforcement
        self._read_files: Set[str] = set()
    
    @tool
    async def ls(self, path: str = "/") -> str:
        """Placeholder docstring - will be replaced."""
        try:
            entries = await self.backend.list_dir(path)
            
            if not entries:
                return f"Directory '{path}' is empty"
            
            # Format output
            result = f"Contents of {path}:\n"
            result += "\n".join(f"  {entry}" for entry in entries)
            result += f"\n\nTotal: {len(entries)} entries"
            
            return result
            
        except FileNotFoundError:
            return f"Error: Directory not found: {path}"
        except PermissionError as e:
            return f"Error: Permission denied: {str(e)}"
        except ValueError as e:
            return f"Error: Invalid path: {str(e)}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    @tool
    async def read_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = 500
    ) -> str:
        """Placeholder docstring - will be replaced."""
        try:
            # Read the file content
            content = await self.backend.read(file_path)
            
            # Track that this file has been read (for edit_file enforcement)
            self._read_files.add(file_path)
            
            # Split into lines
            lines = content.split('\n')
            
            # Apply offset and limit
            start = offset if offset is not None else 0
            
            # Validate offset
            if start < 0:
                return f"Error: Offset must be non-negative, got {start}"
            
            if start >= len(lines):
                return f"Error: Offset {start} exceeds file length ({len(lines)} lines)"
            
            # Calculate end
            if limit is not None:
                if limit <= 0:
                    return f"Error: Limit must be positive, got {limit}"
                end = start + limit
            else:
                end = len(lines)
            
            # Get selected lines
            selected_lines = lines[start:end]
            
            # Format with line numbers (cat -n style)
            formatted_lines = []
            for i, line in enumerate(selected_lines, start=start + 1):           
                # Format: line_number (right-aligned, 6 chars) + |\t + content
                formatted_lines.append(f"{i:6d}|\t{line}")
            
            result = "\n".join(formatted_lines)
            
            # Add metadata
            total_lines = len(lines)
            lines_shown = len(selected_lines)
            
            if limit and start + limit < total_lines:
                result += f"\n\n[Showing lines {start+1}-{start+lines_shown} of {total_lines} total lines]"
                result += f"\n[Use offset={start+lines_shown} to continue reading]"
            else:
                result += f"\n\n[Showing lines {start+1}-{start+lines_shown} of {total_lines} total lines]"
            
            return result
            
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except PermissionError as e:
            return f"Error: Permission denied: {str(e)}"
        except ValueError as e:
            return f"Error: Invalid path: {str(e)}"
        except OSError as e:
            return f"Error: Cannot read file (might be a directory): {str(e)}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    @tool
    async def write_file(self, file_path: str, content: str) -> str:
        """Placeholder docstring - will be replaced."""
        try:
            # Write the file (backend handles parent directory creation)
            await self.backend.write(file_path, content)
            
            # Track that this file has been written (for edit_file enforcement)
            self._read_files.add(file_path)
            
            # Calculate file size
            size_bytes = len(content.encode('utf-8'))
            lines = content.count('\n') + 1
            
            return (
                f"✅ File written successfully: {file_path}\n"
                f"   Size: {size_bytes} bytes\n"
                f"   Lines: {lines}"
            )
            
        except PermissionError as e:
            return f"Error: Permission denied: {str(e)}"
        except ValueError as e:
            return f"Error: Invalid path or content: {str(e)}"
        except OSError as e:
            return f"Error: Cannot write to file: {str(e)}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    @tool
    async def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False
    ) -> str:
        """Placeholder docstring - will be replaced."""
        # Enforce read-before-edit rule
        if file_path not in self._read_files:
            return (
                f"❌ Error: You must use read_file on '{file_path}' before editing it.\n\n"
                f"This ensures you have the correct file content and line numbers.\n"
                f"Please use: read_file(\"{file_path}\") first, then retry your edit."
            )
        
        try:
            # Read current file content
            content = await self.backend.read(file_path)
            
            # Count occurrences
            occurrence_count = content.count(old_string)
            
            # Validate old_string presence
            if occurrence_count == 0:
                return (
                    f"❌ Error: old_string not found in {file_path}\n\n"
                    f"The exact string was not found in the file.\n"
                    f"Please verify:\n"
                    f"1. You're using the exact string from read_file output\n"
                    f"2. Include sufficient context to make it unique\n"
                    f"3. Match indentation exactly (tabs vs spaces)"
                )
            
            # Check uniqueness if not replace_all
            if not replace_all and occurrence_count > 1:
                return (
                    f"❌ Error: old_string appears {occurrence_count} times in {file_path}\n\n"
                    f"Options:\n"
                    f"1. Use replace_all=True to replace all {occurrence_count} occurrences\n"
                    f"2. Provide more context in old_string to make it unique\n"
                    f"   (Include surrounding lines or unique identifiers)"
                )
            
            # Perform replacement
            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                # Replace only first occurrence
                new_content = content.replace(old_string, new_string, 1)
            
            # Write back
            await self.backend.write(file_path, new_content)
            
            # Calculate changes
            old_lines = len(content.split('\n'))
            new_lines = len(new_content.split('\n'))
            line_diff = new_lines - old_lines
            
            result = f"✅ File edited successfully: {file_path}\n"
            result += f"   Replaced: {occurrence_count if replace_all else 1} occurrence(s)\n"
            result += f"   Lines: {old_lines} → {new_lines} ({line_diff:+d})"
            
            return result
            
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except PermissionError as e:
            return f"Error: Permission denied: {str(e)}"
        except ValueError as e:
            return f"Error: Invalid path: {str(e)}"
        except OSError as e:
            return f"Error: Cannot edit file: {str(e)}"
        except Exception as e:
            return f"Error editing file: {str(e)}"
    
    @tool
    async def glob(self, pattern: str) -> str:
        """Placeholder docstring - will be replaced."""
        try:
            matches = await self.backend.glob(pattern)
            
            if not matches:
                return f"No files match pattern: {pattern}"
            
            # Format output
            result = f"Files matching '{pattern}':\n"
            result += "\n".join(f"  {path}" for path in matches)
            result += f"\n\nTotal: {len(matches)} file(s)"
            
            return result
            
        except PermissionError as e:
            return f"Error: Permission denied: {str(e)}"
        except ValueError as e:
            return f"Error: Invalid pattern: {str(e)}"
        except Exception as e:
            return f"Error finding files: {str(e)}"
    
    @tool
    async def grep(
        self,
        pattern: str,
        path: str = "/",
        glob_pattern: Optional[str] = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
        max_results: int = 100
    ) -> str:
        """Placeholder docstring - will be replaced."""
        try:
            # Build search glob pattern
            if glob_pattern:
                # Combine path and glob pattern
                if path == "/":
                    search_pattern = f"/**/{glob_pattern}"
                else:
                    search_pattern = f"{path}/**/{glob_pattern}"
            else:
                # Search all files under path
                if path == "/":
                    search_pattern = "/**/*"
                else:
                    search_pattern = f"{path}/**/*"
            
            # Get files to search
            all_files = await self.backend.glob(search_pattern)
            
            if not all_files:
                return f"No files found matching criteria in {path}"
            
            # Perform search
            results = []
            files_searched = 0
            total_matches = 0
            
            for file_path in all_files:
                # Respect max_results limit
                if output_mode == "files_with_matches" and len(results) >= max_results:
                    break
                
                try:
                    # Read file content
                    content = await self.backend.read(file_path)
                    lines = content.split('\n')
                    
                    files_searched += 1
                    file_matches = []
                    
                    # Search for pattern (literal string matching)
                    for line_num, line in enumerate(lines, 1):
                        if pattern in line:
                            total_matches += 1
                            file_matches.append((line_num, line))
                            
                            # Respect max_results in content mode
                            if output_mode == "content" and total_matches >= max_results:
                                break
                    
                    # Format results based on output_mode
                    if file_matches:
                        if output_mode == "files_with_matches":
                            results.append(file_path)
                        
                        elif output_mode == "content":
                            for line_num, line in file_matches:
                                # Truncate long lines
                                if len(line) > 200:
                                    line = line[:200] + "..."
                                results.append(f"{file_path}:{line_num}:{line}")
                        
                        elif output_mode == "count":
                            results.append(f"{file_path}:{len(file_matches)}")
                    
                    # Break if we've hit max_results in content mode
                    if output_mode == "content" and total_matches >= max_results:
                        break
                        
                except Exception:
                    # Skip files that can't be read
                    continue
            
            # Format final output
            if not results:
                return (
                    f"No matches found for '{pattern}' in {files_searched} file(s)\n"
                    f"Search path: {path}\n"
                    f"Glob filter: {glob_pattern or 'none'}"
                )
            
            output = f"Search results for '{pattern}':\n"
            output += f"Searched: {files_searched} file(s)\n"
            
            if output_mode == "files_with_matches":
                output += f"Files with matches: {len(results)}\n\n"
                output += "\n".join(f"  {path}" for path in results)
            
            elif output_mode == "content":
                output += f"Total matches: {total_matches}\n\n"
                output += "\n".join(results)
            
            elif output_mode == "count":
                output += f"Files with matches: {len(results)}\n\n"
                output += "\n".join(results)
            
            # Add truncation notice if needed
            if output_mode == "files_with_matches" and len(results) >= max_results:
                output += f"\n\n⚠️  Results truncated at {max_results} files"
                output += f"\n   (Use max_results parameter to see more)"
            elif output_mode == "content" and total_matches >= max_results:
                output += f"\n\n⚠️  Results truncated at {max_results} matches"
                output += f"\n   (Use max_results parameter to see more)"
            
            return output
            
        except PermissionError as e:
            return f"Error: Permission denied: {str(e)}"
        except ValueError as e:
            return f"Error: Invalid parameters: {str(e)}"
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    def reset_read_tracking(self) -> None:
        """
        Reset the read file tracking.
        
        Useful for clearing the edit_file enforcement state.
        This might be called when starting a new task or conversation.
        """
        self._read_files.clear()
    
    def get_read_files(self) -> Set[str]:
        """
        Get the set of files that have been read.
        
        Returns:
            Set of file paths that have been read or written
        """
        return self._read_files.copy()


# Set proper docstrings from constants using __doc__
FilesystemToolKit.ls.__doc__ = LIST_FILES_TOOL_DESCRIPTION
FilesystemToolKit.read_file.__doc__ = READ_FILE_TOOL_DESCRIPTION
FilesystemToolKit.write_file.__doc__ = WRITE_FILE_TOOL_DESCRIPTION
FilesystemToolKit.edit_file.__doc__ = EDIT_FILE_TOOL_DESCRIPTION
FilesystemToolKit.glob.__doc__ = GLOB_TOOL_DESCRIPTION
FilesystemToolKit.grep.__doc__ = GREP_TOOL_DESCRIPTION
