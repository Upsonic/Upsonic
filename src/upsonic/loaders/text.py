from __future__ import annotations
from typing import List, Optional, Dict, Any
import os
from datetime import datetime

from .base import DocumentLoader
from .config import TextLoaderConfig
from ..schemas.data_models import Document

try:
    import chardet
except ImportError:
    chardet = None


class TextLoader(DocumentLoader):
    """
    A loader for plain text (`.txt`) files.
    """

    def __init__(self, config: Optional[TextLoaderConfig] = None):
        """Initialize TextLoader with optional configuration."""
        super().__init__(config)
        self.config = config or TextLoaderConfig()

    def load(self, source: str) -> List[Document]:
        """
        Loads a plain text file into a single Document object.

        Args:
            source: The full file path to the .txt file.

        Returns:
            A list containing a single Document object if successful, or an
            empty list if the file cannot be read.
        """
        try:
            file_path = os.path.abspath(source)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Source path '{source}' is not a valid file.")

            stats = os.stat(file_path)
            metadata: Dict[str, Any] = {
                "source": source,
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": stats.st_size,
                "creation_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "last_modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            }

            encoding = self.config.encoding or "utf-8"
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                metadata["detected_encoding"] = encoding
            except UnicodeDecodeError:
                if self.config.error_handling in ["warn", "raise"]:
                    print(f"Warning: {encoding} decoding failed for '{source}'. Attempting to detect encoding.")
                
                if not chardet:
                    error_msg = f"Error: `chardet` library not installed. Cannot detect encoding for '{source}'. Please run 'pip install chardet'."
                    if self.config.error_handling == "ignore":
                        return []
                    elif self.config.error_handling == "warn":
                        print(error_msg)
                        return []
                    else:
                        raise ImportError(error_msg)
                
                with open(file_path, "rb") as f_raw:
                    raw_data = f_raw.read()
                    detection_result = chardet.detect(raw_data)
                    detected_encoding = detection_result.get("encoding")

                if detected_encoding:
                    metadata["detected_encoding"] = detected_encoding
                    if self.config.error_handling in ["warn", "raise"]:
                        print(f"Info: Detected encoding '{detected_encoding}' for '{source}'.")
                    content = raw_data.decode(detected_encoding)
                else:
                    raise IOError("Could not determine file encoding.")

            if self.config.skip_empty_content and not content.strip():
                return []

            if self.config.custom_metadata:
                metadata.update(self.config.custom_metadata)

            document = Document(content=content, metadata=metadata)
            return [document]

        except FileNotFoundError as e:
            error_msg = f"Error: [TextLoader] File not found at path: {e}"
            return self._handle_error(error_msg, e)
        except PermissionError as e:
            error_msg = f"Error: [TextLoader] Permission denied for file: {e}"
            return self._handle_error(error_msg, e)
        except Exception as e:
            error_msg = f"Error: [TextLoader] An unexpected error occurred while loading '{source}': {e}"
            return self._handle_error(error_msg, e)

    def _handle_error(self, message: str, exception: Exception) -> List[Document]:
        """Handle errors based on configuration."""
        if self.config.error_handling == "ignore":
            return []
        elif self.config.error_handling == "warn":
            print(message)
            return []
        else:
            raise exception

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of file extensions supported by this loader."""
        return ['.txt', '.text']