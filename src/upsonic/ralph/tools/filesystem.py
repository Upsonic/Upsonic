"""
Filesystem tools for RalphLoop subagents.

Provides simple file operations that work directly with the workspace filesystem.
These are used by subagents to read, write, search, and execute commands.
"""

from __future__ import annotations

import subprocess
import os
import fnmatch
import re
from pathlib import Path
from typing import List, Optional

from upsonic.tools import ToolKit, tool


class RalphFilesystemToolKit(ToolKit):
    """
    Filesystem toolkit for RalphLoop subagents.
    
    Provides essential filesystem operations:
    - read_file: Read file content
    - write_file: Create/overwrite files
    - edit_file: Edit file content with string replacement
    - list_files: List files in directory
    - search_files: Search for files by pattern
    - grep_files: Search for text within files
    - run_command: Execute shell commands
    """
    
    def __init__(self, workspace: Path):
        """
        Initialize filesystem toolkit.
        
        Args:
            workspace: Workspace directory path
        """
        super().__init__()
        self.workspace = Path(workspace).resolve()
    
    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve path within workspace.
        
        Args:
            path: Path string (relative or absolute)
            
        Returns:
            Resolved Path object
            
        Raises:
            ValueError: If path is outside workspace
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
    
    @tool
    def read_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str:
        """
        Read content from a file.
        
        Args:
            file_path: Path to file (relative to workspace or absolute)
            offset: Starting line number (0-indexed)
            limit: Maximum number of lines to read
        
        Returns:
            File content with line numbers
        """
        try:
            resolved = self._validate_path(file_path)
            
            if not resolved.exists():
                return f"Error: File not found: {file_path}"
            
            if not resolved.is_file():
                return f"Error: Not a file: {file_path}"
            
            content = resolved.read_text(encoding="utf-8")
            lines = content.split("\n")
            
            start = offset if offset is not None else 0
            if start < 0:
                start = 0
            
            if limit is not None and limit > 0:
                end = start + limit
            else:
                end = len(lines)
            
            selected = lines[start:end]
            formatted = []
            for i, line in enumerate(selected, start=start + 1):
                formatted.append(f"{i:6d}| {line}")
            
            result = "\n".join(formatted)
            
            if end < len(lines):
                result += f"\n\n[Showing lines {start+1}-{end} of {len(lines)} total]"
            
            return result
            
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    @tool
    def write_file(
        self,
        file_path: str,
        content: str,
        create_dirs: bool = True,
    ) -> str:
        """
        Write content to a file (creates or overwrites).
        
        IMPORTANT: When creating Python files:
        - Put ALL imports at the TOP of the file
        - Ensure proper class/function structure
        - Use 4-space indentation
        - Include proper type annotations
        
        If the file already exists and you want to modify it,
        consider using edit_file() instead after reading the file.
        
        Args:
            file_path: Path to file (relative to workspace or absolute)
            content: Content to write (must be complete, valid code)
            create_dirs: Create parent directories if they don't exist
        
        Returns:
            Confirmation message
        """
        try:
            resolved = self._validate_path(file_path)
            
            if create_dirs:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            
            resolved.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {file_path}"
            
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
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
        
        CRITICAL: You MUST read_file() FIRST before using this tool!
        
        BEFORE calling edit_file:
        1. Call read_file(file_path) to see the ENTIRE file
        2. Understand the file structure (imports, classes, functions)
        3. Choose old_string that uniquely identifies the edit location
        4. Make sure new_string maintains valid syntax
        
        FAILURE TO READ FIRST WILL CAUSE SYNTAX CORRUPTION!
        
        Args:
            file_path: Path to file
            old_string: Text to find and replace (must be unique in file)
            new_string: Replacement text (must maintain valid syntax)
            replace_all: If True, replace all occurrences; if False, replace first only
        
        Returns:
            Confirmation message with replacement count
        """
        try:
            resolved = self._validate_path(file_path)
            
            if not resolved.exists():
                return f"Error: File not found: {file_path}"
            
            content = resolved.read_text(encoding="utf-8")
            
            if old_string not in content:
                return f"Error: '{old_string[:50]}...' not found in file"
            
            if replace_all:
                count = content.count(old_string)
                new_content = content.replace(old_string, new_string)
            else:
                count = 1
                new_content = content.replace(old_string, new_string, 1)
            
            resolved.write_text(new_content, encoding="utf-8")
            return f"Replaced {count} occurrence(s) in {file_path}"
            
        except Exception as e:
            return f"Error editing file: {str(e)}"
    
    @tool
    def list_files(
        self,
        directory: str = ".",
        recursive: bool = False,
        exclude_dirs: Optional[List[str]] = None,
    ) -> str:
        """
        List files in a directory.
        
        Args:
            directory: Directory path (relative to workspace)
            recursive: If True, list files recursively
            exclude_dirs: Directories to exclude when recursive (defaults to node_modules, etc.)
        
        Returns:
            List of files and directories
        """
        try:
            resolved = self._validate_path(directory)
            
            if not resolved.exists():
                return f"Error: Directory not found: {directory}"
            
            if not resolved.is_dir():
                return f"Error: Not a directory: {directory}"
            
            # Default exclusions for recursive listing
            if exclude_dirs is None and recursive:
                exclude_dirs = ["node_modules", "__pycache__", ".git", "venv", ".venv"]
            elif exclude_dirs is None:
                exclude_dirs = []
            
            entries: List[str] = []
            
            if recursive:
                for root, dirs, files in os.walk(resolved):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    rel_root = Path(root).relative_to(resolved)
                    for d in dirs:
                        entries.append(f"[DIR]  {rel_root / d}/")
                    for f in files:
                        entries.append(f"[FILE] {rel_root / f}")
            else:
                for entry in sorted(resolved.iterdir()):
                    if entry.is_dir():
                        entries.append(f"[DIR]  {entry.name}/")
                    else:
                        entries.append(f"[FILE] {entry.name}")
            
            if not entries:
                return f"Directory '{directory}' is empty"
            
            result = f"Contents of {directory}:\n"
            result += "\n".join(entries)
            result += f"\n\nTotal: {len(entries)} entries"
            return result
            
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    @tool
    def search_files(
        self,
        pattern: str,
        directory: str = ".",
        exclude_dirs: Optional[List[str]] = None,
    ) -> str:
        """
        Search for files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.md")
            directory: Directory to search in
            exclude_dirs: Directories to exclude (defaults to node_modules, __pycache__, .git, venv)
        
        Returns:
            List of matching files
        """
        try:
            resolved = self._validate_path(directory)
            
            if not resolved.exists():
                return f"Error: Directory not found: {directory}"
            
            # Default exclusions for common large/irrelevant directories
            if exclude_dirs is None:
                exclude_dirs = ["node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build"]
            
            matches: List[str] = []
            for path in resolved.rglob(pattern):
                try:
                    rel_path = path.relative_to(resolved)
                    # Skip if any part of the path is in exclude_dirs
                    path_parts = rel_path.parts
                    if any(excluded in path_parts for excluded in exclude_dirs):
                        continue
                    matches.append(str(rel_path))
                except ValueError:
                    continue
            
            if not matches:
                return f"No files matching '{pattern}' found in {directory}"
            
            result = f"Files matching '{pattern}':\n"
            result += "\n".join(f"  {m}" for m in sorted(matches)[:100])
            
            if len(matches) > 100:
                result += f"\n\n... and {len(matches) - 100} more"
            
            return result
            
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    @tool
    def grep_files(
        self,
        text: str,
        directory: str = ".",
        file_pattern: str = "*",
        exclude_dirs: Optional[List[str]] = None,
    ) -> str:
        """
        Search for text within files.
        
        Args:
            text: Text or regex pattern to search for
            directory: Directory to search in
            file_pattern: Glob pattern for files to search (e.g., "*.py")
            exclude_dirs: Directories to exclude (defaults to node_modules, __pycache__, .git, venv)
        
        Returns:
            Matching lines with file and line numbers
        """
        try:
            resolved = self._validate_path(directory)
            
            if not resolved.exists():
                return f"Error: Directory not found: {directory}"
            
            # Default exclusions
            if exclude_dirs is None:
                exclude_dirs = ["node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build"]
            
            try:
                pattern = re.compile(text, re.IGNORECASE)
            except re.error:
                pattern = re.compile(re.escape(text), re.IGNORECASE)
            
            matches: List[str] = []
            files_searched = 0
            
            for file_path in resolved.rglob(file_pattern):
                if not file_path.is_file():
                    continue
                
                # Skip excluded directories
                try:
                    rel_path = file_path.relative_to(resolved)
                    if any(excluded in rel_path.parts for excluded in exclude_dirs):
                        continue
                except ValueError:
                    continue
                
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    files_searched += 1
                    
                    for line_num, line in enumerate(content.split("\n"), start=1):
                        if pattern.search(line):
                            matches.append(f"{rel_path}:{line_num}: {line.strip()[:100]}")
                            
                            if len(matches) >= 100:
                                break
                    
                    if len(matches) >= 100:
                        break
                        
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            if not matches:
                return f"No matches for '{text}' in {files_searched} files searched"
            
            result = f"Matches for '{text}':\n"
            result += "\n".join(matches)
            
            if len(matches) >= 100:
                result += "\n\n[Results truncated at 100 matches]"
            
            return result
            
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    @tool
    def run_command(
        self,
        command: str,
        timeout: int = 60,
    ) -> str:
        """
        Run a shell command in the workspace.
        
        Args:
            command: Shell command to execute
            timeout: Maximum execution time in seconds
        
        Returns:
            Command output (stdout and stderr)
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            output_parts: List[str] = []
            
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout}")
                
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            
            output_parts.append(f"\nExit code: {result.returncode}")
            
            full_output = "\n".join(output_parts)
            
            if len(full_output) > 5000:
                full_output = full_output[:5000] + "\n... (truncated)"
            
            return full_output
            
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error running command: {str(e)}"
