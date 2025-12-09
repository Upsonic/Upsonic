import re
import base64
import os
import subprocess
import platform
from pathlib import Path
from typing import List, Optional, Union

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    _REQUESTS_AVAILABLE = False


def extract_image_urls(text: str) -> List[str]:
    """
    Extracts all image URLs from a string that contains Markdown image syntax.

    It specifically looks for the pattern `![alt text](URL)`.

    Args:
        text: The string content, typically a response from an LLM.

    Returns:
        A list of all found image URLs. Returns an empty list if none are found.
    """
    markdown_image_regex = r"!\[.*?\]\((https?://[^\s)]+)\)"
    urls = re.findall(markdown_image_regex, text)
    return urls


def urls_to_base64(image_urls: List[str]) -> List[str]:
    """
    Takes a list of image URLs, downloads each image, and converts it to a
    base64 encoded string.

    Args:
        image_urls: A list of URLs pointing to images.

    Returns:
        A list of base64 encoded strings. If a URL fails to download,
        it will be skipped.
    """
    if not _REQUESTS_AVAILABLE:
        from upsonic.utils.printing import import_error
        import_error(
            package_name="requests",
            install_command='pip install requests',
            feature_name="image URL downloading"
        )

    base64_images = []
    for url in image_urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            image_bytes = response.content
            
            b64_string = base64.b64encode(image_bytes).decode('utf-8')
            base64_images.append(b64_string)
        except requests.exceptions.RequestException as e:
            pass  # Failed to download image
            continue
            
    return base64_images


def save_base64_image(b64_string: str, file_name: str, ext: str) -> None:
    """
    Decodes a base64 string and saves it as an image file.

    Args:
        b64_string: The base64 encoded image data.
        file_name: The desired name of the file, without the extension.
        ext: The file extension (e.g., "png", "jpg").
    """
    if ext.startswith('.'):
        ext = ext[1:]

    full_filename = f"{file_name}.{ext}"
    
    try:
        image_data = base64.b64decode(b64_string)
        with open(full_filename, 'wb') as f:
            f.write(image_data)
        pass  # Image saved successfully
    except (base64.binascii.Error, TypeError) as e:
        pass  # Failed to decode base64 string


def create_images_folder(folder_path: str) -> str:
    """
    Creates a folder for storing images if it doesn't exist.
    
    Args:
        folder_path: Path to the folder to create.
        
    Returns:
        The absolute path to the created folder.
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder.absolute())


def save_image_to_folder(
    image_data: Union[str, bytes],
    folder_path: str,
    filename: str,
    is_base64: bool = True
) -> str:
    """
    Saves an image to a specified folder.
    
    Args:
        image_data: Image data as base64 string or bytes.
        folder_path: Path to the folder where the image will be saved.
        filename: Name of the file (with extension).
        is_base64: If True, treats image_data as base64 string, otherwise as bytes.
        
    Returns:
        The full path to the saved image file.
    """
    # Create folder if it doesn't exist
    create_images_folder(folder_path)
    
    # Construct full file path
    file_path = os.path.join(folder_path, filename)
    
    # Decode and save
    if is_base64:
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        else:
            raise ValueError("image_data must be a string when is_base64=True")
    else:
        if isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            raise ValueError("image_data must be bytes when is_base64=False")
    
    with open(file_path, 'wb') as f:
        f.write(image_bytes)
    
    return file_path


def extract_and_save_images_from_response(
    response_text: str,
    folder_path: str,
    base_filename: str = "image"
) -> List[str]:
    """
    Extracts image URLs from LLM response, downloads them, and saves to folder.
    
    Args:
        response_text: Text response from LLM containing image URLs.
        folder_path: Path to folder where images will be saved.
        base_filename: Base name for saved files (will be numbered).
        
    Returns:
        List of paths to saved image files.
    """
    # Extract URLs
    urls = extract_image_urls(response_text)
    
    if not urls:
        return []
    
    # Download as base64
    base64_images = urls_to_base64(urls)
    
    # Save each image
    saved_paths = []
    for i, b64_img in enumerate(base64_images, 1):
        filename = f"{base_filename}_{i}.png"
        file_path = save_image_to_folder(
            image_data=b64_img,
            folder_path=folder_path,
            filename=filename,
            is_base64=True
        )
        saved_paths.append(file_path)
    
    return saved_paths


def open_image_file(file_path: str) -> bool:
    """
    Opens an image file using the system's default image viewer.
    
    Args:
        file_path: Path to the image file to open.
        
    Returns:
        True if the image was successfully opened, False otherwise.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    try:
        system = platform.system()
        
        if system == "Darwin":  # macOS
            subprocess.run(["open", file_path], check=True)
        elif system == "Windows":
            os.startfile(file_path)
        elif system == "Linux":
            subprocess.run(["xdg-open", file_path], check=True)
        else:
            print(f"Unsupported platform: {system}")
            return False
            
        return True
    except Exception as e:
        print(f"Error opening image: {e}")
        return False


def open_images_from_folder(folder_path: str, limit: Optional[int] = None) -> List[str]:
    """
    Opens all images from a folder using the system's default image viewer.
    
    Args:
        folder_path: Path to the folder containing images.
        limit: Maximum number of images to open (None for all).
        
    Returns:
        List of paths that were successfully opened.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return []
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
    
    # Get all image files
    image_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(file_path)
    
    # Sort by modification time (newest first)
    image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Apply limit if specified
    if limit is not None:
        image_files = image_files[:limit]
    
    # Open each image
    opened_files = []
    for file_path in image_files:
        if open_image_file(file_path):
            opened_files.append(file_path)
    
    return opened_files


def list_images_in_folder(folder_path: str) -> List[str]:
    """
    Lists all image files in a folder.
    
    Args:
        folder_path: Path to the folder to scan.
        
    Returns:
        List of full paths to image files.
    """
    if not os.path.exists(folder_path):
        return []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
    
    image_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(file_path)
    
    # Sort by modification time (newest first)
    image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return image_files