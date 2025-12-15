import asyncio
import re
from pathlib import Path
from typing import List, Optional, Union

from upsonic.schemas.data_models import Document
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import TextLoaderConfig

try:
    import aiofiles
    _AIOFILES_AVAILABLE = True
except ImportError:
    aiofiles = None
    _AIOFILES_AVAILABLE = False


class TextLoader(BaseLoader):
    """
    A versatile and high-performance loader for various text-based files.

    This loader can structure content from a single file into multiple documents
    based on lines or paragraphs. It leverages asynchronous I/O to efficiently
    process large numbers of files and includes options for cleaning and
    filtering the extracted text.
    """

    def __init__(self, config: Optional[TextLoaderConfig] = None):
        """
        Initializes the TextLoader with its specific configuration.

        Args:
            config: A TextLoaderConfig object with settings for text processing.
        """
        if config is None:
            config = TextLoaderConfig()
        if not _AIOFILES_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="aiofiles",
                install_command='pip install "upsonic[loaders]"',
                feature_name="text loader"
            )
        super().__init__(config)
        self.config: TextLoaderConfig = config


    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Gets a list of file extensions supported by this loader."""
        return [
            ".txt", ".rst", ".log", ".py", ".js", ".ts", ".java",
            ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".php", ".rb",
            ".css", ".ini"
        ]

    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        Loads all text documents from the given source synchronously.

        This is a convenience wrapper around the async `aload` method.
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.aload(source))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.aload(source))

    async def aload(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        Loads all text documents from the given source asynchronously and concurrently.
        """
        file_paths = self._resolve_sources(source)
        if not file_paths:
            return []

        tasks = [self._process_single_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks)

        return [doc for doc_list in results for doc in doc_list]

    def batch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Loads documents from a list of sources, leveraging the core `load` method."""
        return self.load(sources)

    async def abatch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Loads documents from a list of sources asynchronously, leveraging `aload`."""
        return await self.aload(sources)

 

    async def _process_single_file(self, path: Path) -> List[Document]:
        """
        Processes a single text file, loading its entire content into one Document.
        """
        try:
            document_id = self._generate_document_id(path)
            if document_id in self._processed_document_ids:
                raise FileExistsError(
                    f"Source file '{path.resolve()}' has already been processed by this loader instance."
                )
            self._processed_document_ids.add(document_id)

            async with aiofiles.open(path, mode="r", encoding=self.config.encoding, errors="ignore") as f:
                content = await f.read()

            if self.config.strip_whitespace:
                content = content.strip()

            if self.config.skip_empty_content and not content:
                return []
            
            if len(content) < self.config.min_chunk_length:
                return []

            metadata = self._create_metadata(path)
            
            return [Document(document_id=document_id, content=content, metadata=metadata)]

        except Exception as e:
            return self._handle_loading_error(str(path), e)