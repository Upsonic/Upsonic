import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union
import hashlib

from upsonic.schemas.data_models import Document
from upsonic.loaders.config import LoaderConfig
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)

class BaseLoader(ABC):
    """
    The abstract base class for all document loaders.

    This class defines a common interface for loading documents from various
    sources and provides shared helper methods for path resolution, metadata
    creation, and error handling to be used by all subclasses.
    """

    def __init__(self, config: LoaderConfig):
        """
        Initializes the loader with its configuration.

        Args:
            config: A LoaderConfig object or its subclass containing settings for the loader.
        """
        self.config = config
        self._processed_document_ids: set[str] = set()
        self._logger = get_logger(self.__class__.__module__)  # Instance logger for subclasses


    @abstractmethod
    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads all documents from the given source synchronously."""
        raise NotImplementedError

    @abstractmethod
    async def aload(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads all documents from the given source asynchronously."""
        raise NotImplementedError

    @abstractmethod
    def batch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Loads documents from a list of sources, potentially in parallel."""
        raise NotImplementedError

    @abstractmethod
    async def abatch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Loads documents from a list of sources asynchronously and in parallel."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Gets a list of file extensions supported by this loader.
        Extensions should be lowercase and include the leading dot (e.g., '.csv').
        """
        raise NotImplementedError



    @classmethod
    def can_load(cls, source: Union[str, Path]) -> bool:
        """
        Checks if this loader can handle the given source file path.

        By default, this checks if the file's extension is in the list of
        supported extensions. Subclasses can override this for more complex logic.
        """
        source_path = Path(source)
        if not source_path.is_file():
            return False
        return source_path.suffix.lower() in cls.get_supported_extensions()

    def _generate_document_id(self, path: Path) -> str:
        """Creates a deterministic MD5 hash of the file's absolute path to use as a document ID."""
        absolute_path_str = str(path.resolve())
        return hashlib.md5(absolute_path_str.encode("utf-8")).hexdigest()

    def _handle_loading_error(self, source: str, error: Exception) -> List[Document]:
        """Handles loading errors based on the configuration."""
        if not self.config:
            raise error

        if self.config.error_handling == "ignore":
            return []
        elif self.config.error_handling == "warn":
            logger.warning(f"Failed to load {source}: {error}")
            return []
        else:
            raise error

    def _create_metadata(self, source_path: Path) -> Dict[str, Any]:
        """
        Creates a base metadata dictionary for a given file.

        This method populates metadata based on file system information and
        the loader's configuration.
        """
        if not self.config.include_metadata:
            return {}

        try:
            metadata = {
                "source": str(source_path.resolve()),
                "file_name": source_path.name,
                "file_path": str(source_path),
                "file_size": source_path.stat().st_size,
                "creation_datetime_utc": source_path.stat().st_ctime,
                "last_modified_datetime_utc": source_path.stat().st_mtime,
                "file_extension": source_path.suffix.lower(),
                "content_type": self._detect_content_type(source_path),
                "loader_type": self.__class__.__name__.replace("Loader", "").lower(),
            }
        except FileNotFoundError:
            metadata = {
                "source": str(source_path),
                "file_name": source_path.name,
                "file_path": str(source_path),
            }

        if self.config.custom_metadata:
            metadata.update(self.config.custom_metadata)

        return metadata

    def _check_file_size(self, file_path: Path) -> bool:
        """
        Check if a file size is within the configured limits.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file size is acceptable, False if it should be skipped
        """
        if self.config.max_file_size is None:
            return True
            
        try:
            file_size = file_path.stat().st_size
            if file_size > self.config.max_file_size:
                logger.warning(
                    f"Skipping file {file_path} because its size ({file_size} bytes) "
                    f"exceeds the max_file_size of {self.config.max_file_size} bytes."
                )
                return False
            return True
        except OSError as e:
            logger.warning(f"Could not check file size for {file_path}: {e}")
            return False

    def _get_supported_files_from_directory(self, directory: Path) -> List[Path]:
        """Recursively finds all supported files within a directory."""
        supported_files = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and self.__class__.can_load(file_path):
                supported_files.append(file_path)
        return supported_files

    def _resolve_sources(
        self, source: Union[str, Path, List[Union[str, Path]]]
    ) -> List[Path]:
        """
        Resolves a flexible source input into a definitive list of file paths.
        Prevents duplicate paths from being added to the result.
        """
        if not isinstance(source, list):
            source_list = [source]
        else:
            source_list = source

        resolved_files: List[Path] = []
        added_paths: set[Path] = set()  # Track added paths to prevent duplicates
        
        for item in source_list:
            path_item = Path(item)
            if not path_item.exists():
                if self.config.error_handling == "raise":
                    raise FileNotFoundError(f"Source path does not exist: {path_item}")
                elif self.config.error_handling == "warn":
                    logger.warning(f"Source path does not exist, skipping: {path_item}")
                continue

            if path_item.is_file():
                if self.__class__.can_load(path_item) and path_item not in added_paths:
                    resolved_files.append(path_item)
                    added_paths.add(path_item)
            elif path_item.is_dir():
                supported_files = self._get_supported_files_from_directory(path_item)
                for file_path in supported_files:
                    if file_path not in added_paths:
                        resolved_files.append(file_path)
                        added_paths.add(file_path)

        return resolved_files

    def _detect_content_type(self, source_path: Path) -> str:
        """Detects content type based on file extension for RAG optimization."""
        extension = source_path.suffix.lower()
        
        content_type_map = {
            '.pdf': 'document',
            '.docx': 'document', 
            '.doc': 'document',
            '.odt': 'document',
            
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.html': 'web_content',
            '.htm': 'web_content',
            '.xml': 'structured_data',
            
            '.json': 'structured_data',
            '.jsonl': 'structured_data',
            '.csv': 'tabular_data',
            '.tsv': 'tabular_data',
            '.yaml': 'configuration',
            '.yml': 'configuration',
            
            '.py': 'code',
            '.js': 'code',
            '.ts': 'code',
            '.java': 'code',
            '.cpp': 'code',
            '.c': 'code',
            '.go': 'code',
            '.rs': 'code',
            '.php': 'code',
            '.rb': 'code',
            
            '.txt': 'plain_text',
            '.log': 'plain_text',
            '.rst': 'plain_text',
        }
        
        return content_type_map.get(extension, 'unknown')