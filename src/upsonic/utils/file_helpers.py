import os

def get_clean_extension(attachment_path: str) -> str | None:
    """
    Extracts a clean, lowercase file extension from a given file path.

    This function is a simple utility with a single responsibility. It has
    no knowledge of modalities (video, image, etc.). It processes a
    string and returns the file extension without the leading dot.

    Examples:
        - "path/to/my_video.MP4"  -> "mp4"
        - "document.pdf"          -> "pdf"
        - "archive.tar.gz"        -> "gz"
        - "file_without_extension"-> None
        - "folder/."              -> None

    Args:
        attachment_path: The file path string to analyze.

    Returns:
        The lowercase file extension as a string, or None if no
        extension is found.
    """
    _root, extension = os.path.splitext(attachment_path)

    if extension:
        return extension.lstrip('.').lower()
    else:
        return None