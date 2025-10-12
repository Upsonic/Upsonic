import re
import base64
from typing import List

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