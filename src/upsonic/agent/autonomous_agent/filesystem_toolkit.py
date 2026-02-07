"""
Autonomous Agent Filesystem Toolkit - Comprehensive filesystem operations.

Provides production-ready tools for file operations:
- read, write, edit, search, list, move, copy, delete
"""
from __future__ import annotations

import asyncio
import fnmatch
import re
import shutil
from pathlib import Path
from typing import List, Optional, Set

from upsonic.tools import ToolKit, tool


class AutonomousFilesystemToolKit(ToolKit):
    """
    Comprehensive filesystem toolkit for AutonomousAgent.
    
    Provides all essential filesystem operations:
    - read_file: Read file content with pagination support
    - write_file: Create/overwrite files with automatic directory creation
    - edit_file: Precise string replacement with read-tracking enforcement
    - list_files: List directory contents (recursive or non-recursive)
    - search_files: Find files by glob pattern
    - grep_files: Search text within files with regex support
    - move_file: Move or rename files/directories
    - copy_file: Copy files/directories
    - delete_file: Delete files/directories
    - file_info: Get detailed file/directory metadata
    - create_directory: Create directory with parents
    
    Features:
    - Workspace sandboxing for security
    - Read-before-edit enforcement for safe edits
    - Automatic parent directory creation
    - Comprehensive error handling
    - Line-numbered output for precise editing
    
    Usage:
        ```python
        from upsonic import AutonomousAgent
        
        agent = AutonomousAgent(workspace="/path/to/project")
        result = agent.do("Read the main.py file and add logging")
        ```
    """
    
    def __init__(self, workspace: Path) -> None:
        """
        Initialize filesystem toolkit with workspace sandboxing.
        
        Args:
            workspace: Workspace directory path. All operations are restricted to this directory.
        """
        super().__init__()
        self.workspace: Path = Path(workspace).resolve()
        self._read_files: Set[str] = set()
    
    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve path within workspace for security.
        
        Args:
            path: Path string (relative or absolute)
            
        Returns:
            Resolved absolute Path object
            
        Raises:
            ValueError: If path escapes workspace sandbox
        """
        if path.startswith("/"):
            resolved = Path(path).resolve()
        else:
            resolved = (self.workspace / path).resolve()
        
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"Path '{path}' is outside workspace '{self.workspace}'")
        
        return resolved
    
    def _get_relative_path(self, path: Path) -> str:
        """Get path relative to workspace for clean output."""
        try:
            return str(path.relative_to(self.workspace))
        except ValueError:
            return str(path)
    
    @tool
    def read_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str:
        """
        Read content from a file with optional pagination.
        
        Use this tool to read the contents of a file. The output includes line numbers
        for easy reference when editing. For large files, use offset and limit to
        read specific sections.
        
        Args:
            file_path: Path to file (relative to workspace or absolute)
            offset: Starting line number (0-indexed). If None, starts from beginning.
            limit: Maximum number of lines to read. If None, reads all remaining lines.
        
        Returns:
            File content with line numbers in format "LINE_NUM| CONTENT"
            
        Example:
            read_file("src/main.py")  # Read entire file
            read_file("src/main.py", offset=10, limit=20)  # Read lines 11-30
        """
        try:
            resolved = self._validate_path(file_path)
            
            if not resolved.exists():
                return f"Error: File not found: {file_path}"
            
            if not resolved.is_file():
                return f"Error: Not a file: {file_path}"
            
            content = resolved.read_text(encoding="utf-8")
            lines = content.split("\n")
            
            self._read_files.add(str(resolved))
            
            start = offset if offset is not None else 0
            if start < 0:
                start = 0
            if start >= len(lines):
                return f"Error: Offset {start} exceeds file length ({len(lines)} lines)"
            
            if limit is not None and limit > 0:
                end = min(start + limit, len(lines))
            else:
                end = len(lines)
            
            selected = lines[start:end]
            formatted: List[str] = []
            for i, line in enumerate(selected, start=start + 1):
                formatted.append(f"{i:6d}| {line}")
            
            result = "\n".join(formatted)
            
            if end < len(lines):
                result += f"\n\n[Showing lines {start + 1}-{end} of {len(lines)} total]"
                result += f"\n[Use offset={end} to continue reading]"
            else:
                result += f"\n\n[Showing lines {start + 1}-{end} of {len(lines)} total]"
            
            return result
            
        except UnicodeDecodeError:
            return f"Error: File appears to be binary and cannot be read as text: {file_path}"
        except PermissionError:
            return f"Error: Permission denied reading: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    @tool
    async def aread_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str:
        """
        Async version of read_file.
        
        Read content from a file with optional pagination.
        
        Args:
            file_path: Path to file (relative to workspace or absolute)
            offset: Starting line number (0-indexed)
            limit: Maximum number of lines to read
        
        Returns:
            File content with line numbers
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.read_file(file_path, offset, limit)
        )
    
    @tool
    def write_file(
        self,
        file_path: str,
        content: str,
        create_dirs: bool = True,
    ) -> str:
        """
        Write content to a file (creates or overwrites).
        
        Creates the file if it doesn't exist, overwrites if it does.
        Parent directories are created automatically if create_dirs is True.
        
        IMPORTANT for Python files:
        - Put ALL imports at the TOP of the file
        - Use 4-space indentation
        - Include proper type annotations
        
        For modifying existing files, consider using edit_file() instead
        after reading the file with read_file().
        
        Args:
            file_path: Path to file (relative to workspace or absolute)
            content: Complete content to write to the file
            create_dirs: If True, create parent directories if they don't exist
        
        Returns:
            Confirmation message with bytes written
            
        Example:
            write_file("config.json", '{"key": "value"}')
        """
        try:
            resolved = self._validate_path(file_path)
            
            if create_dirs:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            
            resolved.write_text(content, encoding="utf-8")
            self._read_files.add(str(resolved))
            
            size_bytes = len(content.encode("utf-8"))
            line_count = content.count("\n") + 1
            
            return (
                f"✅ Successfully wrote to {self._get_relative_path(resolved)}\n"
                f"   Size: {size_bytes} bytes\n"
                f"   Lines: {line_count}"
            )
            
        except PermissionError:
            return f"Error: Permission denied writing to: {file_path}"
        except OSError as e:
            return f"Error writing file: {str(e)}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    @tool
    async def awrite_file(
        self,
        file_path: str,
        content: str,
        create_dirs: bool = True,
    ) -> str:
        """
        Async version of write_file.
        
        Write content to a file (creates or overwrites).
        
        Args:
            file_path: Path to file
            content: Complete content to write
            create_dirs: Create parent directories if needed
        
        Returns:
            Confirmation message
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.write_file(file_path, content, create_dirs)
        )
    
    @tool
    def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """
        Edit a file by replacing text.
        
        CRITICAL: You MUST call read_file() FIRST before using this tool!
        
        This ensures you have the correct, up-to-date file content.
        The old_string must exactly match content in the file, including
        whitespace and indentation.
        
        Steps for safe editing:
        1. Call read_file(file_path) to see current content
        2. Copy the exact text you want to replace (including whitespace)
        3. Call edit_file with old_string and new_string
        
        Args:
            file_path: Path to file
            old_string: Exact text to find and replace (must match file content exactly)
            new_string: Replacement text
            replace_all: If True, replace all occurrences. If False, replace only first.
        
        Returns:
            Confirmation message with replacement count
            
        Example:
            edit_file("main.py", "def old_name()", "def new_name()")
        """
        resolved = self._validate_path(file_path)
        resolved_str = str(resolved)
        
        if resolved_str not in self._read_files:
            return (
                f"❌ Error: You must call read_file('{file_path}') before editing.\n\n"
                f"This ensures you have the correct file content and line numbers.\n"
                f"Please use: read_file(\"{file_path}\") first, then retry your edit."
            )
        
        try:
            if not resolved.exists():
                return f"Error: File not found: {file_path}"
            
            content = resolved.read_text(encoding="utf-8")
            
            if old_string not in content:
                old_preview = old_string[:100] + "..." if len(old_string) > 100 else old_string
                return (
                    f"❌ Error: old_string not found in {file_path}\n\n"
                    f"Looking for: {repr(old_preview)}\n\n"
                    f"Please verify:\n"
                    f"1. You're using the exact string from read_file output\n"
                    f"2. Include sufficient context to make it unique\n"
                    f"3. Match indentation exactly (tabs vs spaces)\n"
                    f"4. Consider re-reading the file as it may have changed"
                )
            
            occurrence_count = content.count(old_string)
            
            if not replace_all and occurrence_count > 1:
                return (
                    f"❌ Error: old_string appears {occurrence_count} times in {file_path}\n\n"
                    f"Options:\n"
                    f"1. Use replace_all=True to replace all {occurrence_count} occurrences\n"
                    f"2. Provide more context in old_string to make it unique\n"
                    f"   (Include surrounding lines or unique identifiers)"
                )
            
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replaced_count = occurrence_count
            else:
                new_content = content.replace(old_string, new_string, 1)
                replaced_count = 1
            
            resolved.write_text(new_content, encoding="utf-8")
            
            old_lines = len(content.split("\n"))
            new_lines = len(new_content.split("\n"))
            line_diff = new_lines - old_lines
            
            return (
                f"✅ Successfully edited {self._get_relative_path(resolved)}\n"
                f"   Replaced: {replaced_count} occurrence(s)\n"
                f"   Lines: {old_lines} → {new_lines} ({line_diff:+d})"
            )
            
        except PermissionError:
            return f"Error: Permission denied editing: {file_path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"
    
    @tool
    async def aedit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """
        Async version of edit_file.
        
        Edit a file by replacing text. Requires prior read_file() call.
        
        Args:
            file_path: Path to file
            old_string: Text to find
            new_string: Replacement text
            replace_all: Replace all occurrences
        
        Returns:
            Confirmation message
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.edit_file(file_path, old_string, new_string, replace_all)
        )
    
    @tool
    def list_files(
        self,
        directory: str = ".",
        recursive: bool = False,
        max_depth: Optional[int] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> str:
        """
        List files and directories.
        
        Args:
            directory: Directory path (relative to workspace)
            recursive: If True, list files recursively
            max_depth: Maximum depth for recursive listing (None = unlimited)
            exclude_patterns: Glob patterns to exclude (e.g., ["*.pyc", "__pycache__"])
        
        Returns:
            Formatted list of files and directories
            
        Example:
            list_files("src")  # List src directory
            list_files(".", recursive=True, max_depth=2)  # Recursive with depth limit
        """
        try:
            resolved = self._validate_path(directory)
            
            if not resolved.exists():
                return f"Error: Directory not found: {directory}"
            
            if not resolved.is_dir():
                return f"Error: Not a directory: {directory}"
            
            default_excludes = ["__pycache__", "node_modules", ".git", "venv", ".venv", "dist", "build", ".tox", "*.egg-info"]
            excludes = set(exclude_patterns or []) | set(default_excludes)
            
            def should_exclude(path: Path) -> bool:
                name = path.name
                for pattern in excludes:
                    if fnmatch.fnmatch(name, pattern):
                        return True
                return False
            
            entries: List[str] = []
            
            if recursive:
                def walk_dir(current: Path, depth: int = 0) -> None:
                    if max_depth is not None and depth > max_depth:
                        return
                    
                    try:
                        items = sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                    except PermissionError:
                        return
                    
                    for item in items:
                        if should_exclude(item):
                            continue
                        
                        try:
                            rel_path = item.relative_to(resolved)
                        except ValueError:
                            continue
                        
                        if item.is_dir():
                            entries.append(f"[DIR]  {rel_path}/")
                            walk_dir(item, depth + 1)
                        else:
                            entries.append(f"[FILE] {rel_path}")
                
                walk_dir(resolved)
            else:
                try:
                    items = sorted(resolved.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                except PermissionError:
                    return f"Error: Permission denied listing: {directory}"
                
                for entry in items:
                    if should_exclude(entry):
                        continue
                    if entry.is_dir():
                        entries.append(f"[DIR]  {entry.name}/")
                    else:
                        entries.append(f"[FILE] {entry.name}")
            
            if not entries:
                return f"Directory '{directory}' is empty (or all contents are excluded)"
            
            result = f"Contents of {directory}:\n"
            result += "\n".join(entries)
            result += f"\n\nTotal: {len(entries)} entries"
            
            return result
            
        except PermissionError:
            return f"Error: Permission denied: {directory}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    @tool
    async def alist_files(
        self,
        directory: str = ".",
        recursive: bool = False,
        max_depth: Optional[int] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> str:
        """Async version of list_files."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.list_files(directory, recursive, max_depth, exclude_patterns)
        )
    
    @tool
    def search_files(
        self,
        pattern: str,
        directory: str = ".",
        exclude_patterns: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> str:
        """
        Search for files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.md", "test_*.py")
            directory: Directory to search in
            exclude_patterns: Directories/files to exclude
            max_results: Maximum number of results to return
        
        Returns:
            List of matching file paths
            
        Example:
            search_files("*.py")  # All Python files
            search_files("**/*.test.js")  # All test files recursively
        """
        try:
            resolved = self._validate_path(directory)
            
            if not resolved.exists():
                return f"Error: Directory not found: {directory}"
            
            default_excludes = ["__pycache__", "node_modules", ".git", "venv", ".venv", "dist", "build"]
            excludes = set(exclude_patterns or []) | set(default_excludes)
            
            matches: List[str] = []
            for path in resolved.rglob(pattern):
                if len(matches) >= max_results:
                    break
                
                try:
                    rel_path = path.relative_to(resolved)
                    if any(excluded in rel_path.parts for excluded in excludes):
                        continue
                    matches.append(str(rel_path))
                except ValueError:
                    continue
            
            if not matches:
                return f"No files matching '{pattern}' found in {directory}"
            
            result = f"Files matching '{pattern}':\n"
            result += "\n".join(f"  {m}" for m in sorted(matches))
            
            if len(matches) >= max_results:
                result += f"\n\n⚠️  Results truncated at {max_results}"
            else:
                result += f"\n\nTotal: {len(matches)} file(s)"
            
            return result
            
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    @tool
    async def asearch_files(
        self,
        pattern: str,
        directory: str = ".",
        exclude_patterns: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> str:
        """Async version of search_files."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.search_files(pattern, directory, exclude_patterns, max_results)
        )
    
    @tool
    def grep_files(
        self,
        text: str,
        directory: str = ".",
        file_pattern: str = "*",
        case_sensitive: bool = False,
        is_regex: bool = False,
        max_results: int = 100,
        context_lines: int = 0,
    ) -> str:
        """
        Search for text within files.
        
        Args:
            text: Text or regex pattern to search for
            directory: Directory to search in
            file_pattern: Glob pattern for files to search (e.g., "*.py")
            case_sensitive: If True, search is case-sensitive
            is_regex: If True, treat text as regex pattern
            max_results: Maximum number of matching lines to return
            context_lines: Number of lines of context before/after each match
        
        Returns:
            Matching lines with file paths and line numbers
            
        Example:
            grep_files("TODO", file_pattern="*.py")  # Find TODOs in Python files
            grep_files("def.*async", is_regex=True)  # Find async functions
        """
        try:
            resolved = self._validate_path(directory)
            
            if not resolved.exists():
                return f"Error: Directory not found: {directory}"
            
            default_excludes = ["__pycache__", "node_modules", ".git", "venv", ".venv", "dist", "build"]
            
            flags = 0 if case_sensitive else re.IGNORECASE
            
            if is_regex:
                try:
                    pattern = re.compile(text, flags)
                except re.error as e:
                    return f"Error: Invalid regex pattern: {e}"
            else:
                pattern = re.compile(re.escape(text), flags)
            
            matches: List[str] = []
            files_searched = 0
            
            for file_path in resolved.rglob(file_pattern):
                if not file_path.is_file():
                    continue
                
                try:
                    rel_path = file_path.relative_to(resolved)
                    if any(excluded in rel_path.parts for excluded in default_excludes):
                        continue
                except ValueError:
                    continue
                
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    lines = content.split("\n")
                    files_searched += 1
                    
                    for line_num, line in enumerate(lines, start=1):
                        if pattern.search(line):
                            match_entry = f"{rel_path}:{line_num}: {line.strip()[:150]}"
                            
                            if context_lines > 0:
                                context_start = max(0, line_num - context_lines - 1)
                                context_end = min(len(lines), line_num + context_lines)
                                for ctx_num in range(context_start, context_end):
                                    if ctx_num + 1 != line_num:
                                        match_entry += f"\n  {rel_path}:{ctx_num + 1}: {lines[ctx_num].strip()[:100]}"
                            
                            matches.append(match_entry)
                            
                            if len(matches) >= max_results:
                                break
                    
                    if len(matches) >= max_results:
                        break
                        
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            if not matches:
                return f"No matches for '{text}' in {files_searched} files searched"
            
            result = f"Matches for '{text}':\n"
            result += f"Searched: {files_searched} file(s)\n\n"
            result += "\n".join(matches)
            
            if len(matches) >= max_results:
                result += f"\n\n⚠️  Results truncated at {max_results} matches"
            
            return result
            
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    @tool
    async def agrep_files(
        self,
        text: str,
        directory: str = ".",
        file_pattern: str = "*",
        case_sensitive: bool = False,
        is_regex: bool = False,
        max_results: int = 100,
        context_lines: int = 0,
    ) -> str:
        """Async version of grep_files."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.grep_files(
                text, directory, file_pattern, case_sensitive, is_regex, max_results, context_lines
            )
        )
    
    @tool
    def move_file(
        self,
        source: str,
        destination: str,
    ) -> str:
        """
        Move or rename a file or directory.
        
        Args:
            source: Source file/directory path
            destination: Destination path
        
        Returns:
            Confirmation message
            
        Example:
            move_file("old_name.py", "new_name.py")  # Rename
            move_file("file.py", "src/file.py")  # Move
        """
        try:
            src = self._validate_path(source)
            dst = self._validate_path(destination)
            
            if not src.exists():
                return f"Error: Source not found: {source}"
            
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(src), str(dst))
            
            if str(src) in self._read_files:
                self._read_files.discard(str(src))
                self._read_files.add(str(dst))
            
            return f"✅ Moved: {self._get_relative_path(src)} → {self._get_relative_path(dst)}"
            
        except PermissionError:
            return "Error: Permission denied"
        except Exception as e:
            return f"Error moving: {str(e)}"
    
    @tool
    async def amove_file(self, source: str, destination: str) -> str:
        """Async version of move_file."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.move_file(source, destination)
        )
    
    @tool
    def copy_file(
        self,
        source: str,
        destination: str,
    ) -> str:
        """
        Copy a file or directory.
        
        Args:
            source: Source file/directory path
            destination: Destination path
        
        Returns:
            Confirmation message
        """
        try:
            src = self._validate_path(source)
            dst = self._validate_path(destination)
            
            if not src.exists():
                return f"Error: Source not found: {source}"
            
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
                return f"✅ Copied directory: {self._get_relative_path(src)} → {self._get_relative_path(dst)}"
            else:
                shutil.copy2(str(src), str(dst))
                return f"✅ Copied: {self._get_relative_path(src)} → {self._get_relative_path(dst)}"
            
        except PermissionError:
            return "Error: Permission denied"
        except Exception as e:
            return f"Error copying: {str(e)}"
    
    @tool
    async def acopy_file(self, source: str, destination: str) -> str:
        """Async version of copy_file."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.copy_file(source, destination)
        )
    
    @tool
    def delete_file(
        self,
        path: str,
        recursive: bool = False,
    ) -> str:
        """
        Delete a file or directory.
        
        CAUTION: This operation is irreversible!
        
        Args:
            path: File or directory path to delete
            recursive: If True, delete directories with contents. Required for non-empty directories.
        
        Returns:
            Confirmation message
        """
        try:
            resolved = self._validate_path(path)
            
            if not resolved.exists():
                return f"Error: Path not found: {path}"
            
            if resolved.is_dir():
                if recursive:
                    shutil.rmtree(str(resolved))
                    return f"✅ Deleted directory and contents: {self._get_relative_path(resolved)}"
                else:
                    try:
                        resolved.rmdir()
                        return f"✅ Deleted empty directory: {self._get_relative_path(resolved)}"
                    except OSError:
                        return "Error: Directory not empty. Use recursive=True to delete with contents."
            else:
                resolved.unlink()
                self._read_files.discard(str(resolved))
                return f"✅ Deleted: {self._get_relative_path(resolved)}"
            
        except PermissionError:
            return "Error: Permission denied"
        except Exception as e:
            return f"Error deleting: {str(e)}"
    
    @tool
    async def adelete_file(self, path: str, recursive: bool = False) -> str:
        """Async version of delete_file."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.delete_file(path, recursive)
        )
    
    @tool
    def file_info(self, path: str) -> str:
        """
        Get detailed information about a file or directory.
        
        Args:
            path: File or directory path
        
        Returns:
            Detailed metadata including size, permissions, timestamps
        """
        try:
            resolved = self._validate_path(path)
            
            if not resolved.exists():
                return f"Error: Path not found: {path}"
            
            stat = resolved.stat()
            
            info: List[str] = [f"Path: {self._get_relative_path(resolved)}"]
            info.append(f"Type: {'Directory' if resolved.is_dir() else 'File'}")
            
            if resolved.is_file():
                size = stat.st_size
                if size < 1024:
                    size_str = f"{size} bytes"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                info.append(f"Size: {size_str}")
                info.append(f"Lines: {len(resolved.read_text(errors='ignore').split(chr(10)))}")
            
            import datetime
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
            ctime = datetime.datetime.fromtimestamp(stat.st_ctime)
            
            info.append(f"Modified: {mtime.isoformat()}")
            info.append(f"Created: {ctime.isoformat()}")
            info.append(f"Permissions: {oct(stat.st_mode)[-3:]}")
            
            if resolved.is_dir():
                try:
                    contents = list(resolved.iterdir())
                    dirs = sum(1 for c in contents if c.is_dir())
                    files = len(contents) - dirs
                    info.append(f"Contents: {dirs} directories, {files} files")
                except PermissionError:
                    info.append("Contents: Permission denied")
            
            return "\n".join(info)
            
        except Exception as e:
            return f"Error getting file info: {str(e)}"
    
    @tool
    async def afile_info(self, path: str) -> str:
        """Async version of file_info."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.file_info(path)
        )
    
    @tool
    def create_directory(self, path: str) -> str:
        """
        Create a directory (including parent directories).
        
        Args:
            path: Directory path to create
        
        Returns:
            Confirmation message
        """
        try:
            resolved = self._validate_path(path)
            
            if resolved.exists():
                if resolved.is_dir():
                    return f"Directory already exists: {self._get_relative_path(resolved)}"
                else:
                    return f"Error: Path exists as a file: {path}"
            
            resolved.mkdir(parents=True, exist_ok=True)
            return f"✅ Created directory: {self._get_relative_path(resolved)}"
            
        except PermissionError:
            return "Error: Permission denied"
        except Exception as e:
            return f"Error creating directory: {str(e)}"
    
    @tool
    async def acreate_directory(self, path: str) -> str:
        """Async version of create_directory."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.create_directory(path)
        )
    
    def reset_read_tracking(self) -> None:
        """Reset the read file tracking for edit_file enforcement."""
        self._read_files.clear()
    
    def get_read_files(self) -> Set[str]:
        """Get the set of files that have been read."""
        return self._read_files.copy()
