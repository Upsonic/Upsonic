import os
from typing import List, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task


class AttachmentFileNotFoundError(Exception):
    """
    Exception raised when an attachment file does not exist.
    
    This exception provides detailed information about which file is missing
    and helpful suggestions for resolving the issue.
    """
    
    def __init__(self, attachment_path: str, task_description: str = None):
        self.attachment_path = attachment_path
        self.task_description = task_description
        
        message = f"Attachment file not found: '{attachment_path}'"
        
        if task_description:
            message += f"\nTask: {task_description}"
        
        suggestions = []
        
        if not os.path.isabs(attachment_path):
            suggestions.append(f"• Check if the file path is correct relative to the current working directory: {os.getcwd()}")
            suggestions.append(f"• Try using an absolute path instead of a relative path")
        
        parent_dir = os.path.dirname(attachment_path)
        if parent_dir and os.path.exists(parent_dir) and not os.access(parent_dir, os.R_OK):
            suggestions.append(f"• Check if you have read permissions for the directory: {parent_dir}")
        
        if '.' in attachment_path:
            suggestions.append(f"• Verify the file extension is correct")
        
        if suggestions:
            message += f"\n\nSuggestions:\n" + "\n".join(suggestions)
        
        super().__init__(message)


def validate_attachments_exist(task: "Task") -> None:
    """
    Validates that all attachment files in a task actually exist on the filesystem.
    
    This is a critical validation for AI Agent frameworks to ensure that
    all referenced files are accessible before processing begins.
    
    Args:
        task: The Task object containing the list of attachments to validate.
        
    Raises:
        AttachmentFileNotFoundError: If any attachment file does not exist.
        
    Example:
        ```python
        from upsonic import Task
        from upsonic.utils.validators import validate_attachments_exist
        
        task = Task("Analyze this image", attachments=["image.jpg"])
        
        try:
            validate_attachments_exist(task)
            print("All attachments are valid!")
        except AttachmentFileNotFoundError as e:
            print(f"Validation failed: {e}")
        ```
    """
    if not task.attachments:
        return
    
    missing_files = []
    
    for attachment_path in task.attachments:
        if not attachment_path:
            continue
            
        path = Path(attachment_path)
        
        if not path.exists():
            missing_files.append(attachment_path)
        elif not path.is_file():
            missing_files.append(attachment_path)
    
    if missing_files:
        # Use the first missing file for the detailed error message
        raise AttachmentFileNotFoundError(
            attachment_path=missing_files[0],
            task_description=task.description
        )


def validate_attachments_readable(task: "Task") -> None:
    """
    Validates that all attachment files in a task are readable.
    
    This goes beyond just checking existence to ensure the files can actually
    be opened and read by the application.
    
    Args:
        task: The Task object containing the list of attachments to validate.
        
    Raises:
        AttachmentFileNotFoundError: If any attachment file cannot be read.
        
    Example:
        ```python
        from upsonic import Task
        from upsonic.utils.validators import validate_attachments_readable
        
        task = Task("Process this document", attachments=["document.pdf"])
        
        try:
            validate_attachments_readable(task)
            print("All attachments are readable!")
        except AttachmentFileNotFoundError as e:
            print(f"Read validation failed: {e}")
        ```
    """
    if not task.attachments:
        return
    
    unreadable_files = []
    
    for attachment_path in task.attachments:
        if not attachment_path:
            continue
            
        path = Path(attachment_path)
        
        if not path.exists():
            unreadable_files.append(attachment_path)
        elif not path.is_file():
            unreadable_files.append(attachment_path)
        elif not os.access(path, os.R_OK):
            unreadable_files.append(attachment_path)
        else:
            try:
                with open(path, 'rb') as f:
                    f.read(1024)
            except (IOError, OSError, PermissionError):
                unreadable_files.append(attachment_path)
    
    if unreadable_files:
        raise AttachmentFileNotFoundError(
            attachment_path=unreadable_files[0],
            task_description=task.description
        )


def get_attachment_info(task: "Task") -> List[dict]:
    """
    Get detailed information about all attachments in a task.
    
    This is useful for debugging and providing comprehensive information
    about the files that will be processed.
    
    Args:
        task: The Task object containing the list of attachments.
        
    Returns:
        List[dict]: A list of dictionaries containing file information.
                   Each dictionary contains:
                   - path: The file path
                   - exists: Whether the file exists
                   - is_file: Whether it's a file (not directory)
                   - readable: Whether the file is readable
                   - size: File size in bytes (if exists and readable)
                   - extension: File extension (if exists)
                   
    Example:
        ```python
        from upsonic import Task
        from upsonic.utils.validators import get_attachment_info
        
        task = Task("Process files", attachments=["file1.txt", "file2.jpg"])
        info = get_attachment_info(task)
        
        for file_info in info:
            from upsonic.utils.printing import info_log
            info_log(f"File: {file_info['path']}", "FileValidator")
            info_log(f"  Exists: {file_info['exists']}", "FileValidator")
            info_log(f"  Size: {file_info.get('size', 'N/A')} bytes", "FileValidator")
        ```
    """
    if not task.attachments:
        return []
    
    attachment_info = []
    
    for attachment_path in task.attachments:
        if not attachment_path:
            continue
            
        path = Path(attachment_path)
        info = {
            'path': attachment_path,
            'exists': path.exists(),
            'is_file': path.is_file() if path.exists() else False,
            'readable': False,
            'size': None,
            'extension': path.suffix.lower() if path.suffix else None
        }
        
        if info['exists'] and info['is_file']:
            try:
                if os.access(path, os.R_OK):
                    info['readable'] = True
                    info['size'] = path.stat().st_size
            except (OSError, PermissionError):
                pass
        
        attachment_info.append(info)
    
    return attachment_info