from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator, Iterator
import asyncio
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ..schemas.data_models import Document
from .config import LoaderConfig


class LoadingResult:
    """Result container for loading operations."""
    
    def __init__(
        self, 
        documents: List[Document], 
        source: str, 
        success: bool = True, 
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.documents = documents
        self.source = source
        self.success = success
        self.error = error
        self.metadata = metadata or {}
        self.load_time = time.time()


class LoadingProgress:
    """Progress tracking for batch loading operations."""
    
    def __init__(self, total_sources: int):
        self.total_sources = total_sources
        self.processed_sources = 0
        self.successful_sources = 0
        self.failed_sources = 0
        self.start_time = time.time()
        self.current_source: Optional[str] = None
        
    def update(self, source: str, success: bool):
        """Update progress with completion of a source."""
        self.processed_sources += 1
        if success:
            self.successful_sources += 1
        else:
            self.failed_sources += 1
        self.current_source = source
        
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage."""
        return (self.processed_sources / self.total_sources) * 100 if self.total_sources > 0 else 0
        
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time


class DocumentLoader(ABC):
    """
    Abstract contract for all document ingestion components.

    This base class provides a comprehensive framework for document loading with:
    - Synchronous and asynchronous loading capabilities
    - Batch processing with progress tracking
    - Configurable error handling strategies
    - Extensible metadata enrichment
    - Performance monitoring and optimization
    """

    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        Initialize the loader with optional configuration.
        
        Args:
            config: Loader-specific configuration object
        """
        self.config = config
        self._stats = {
            'total_files_processed': 0,
            'total_documents_created': 0,
            'total_errors': 0,
            'avg_processing_time': 0.0,
            'last_processing_time': 0.0
        }

    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        Loads data from a given source and transforms it into a list of Document objects.

        Args:
            source: A string representing the data source, such as a file path or a URL.

        Returns:
            A list of Document objects, each representing a discrete piece of
            information extracted from the source. For a single file, this might be
            a list with just one Document.
        """
        raise NotImplementedError

    async def load_async(self, source: str) -> List[Document]:
        """
        Asynchronous version of load method.
        
        By default, runs the synchronous load method in a thread pool.
        Subclasses can override for true async implementations.
        
        Args:
            source: A string representing the data source
            
        Returns:
            A list of Document objects
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.load, source)

    def load_batch(
        self, 
        sources: List[str], 
        max_workers: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[LoadingResult]:
        """
        Load multiple sources in parallel.
        
        Args:
            sources: List of source paths
            max_workers: Maximum number of worker threads
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of LoadingResult objects
        """
        results = []
        progress = LoadingProgress(len(sources))
        
        if progress_callback:
            progress_callback(progress)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_source = {
                executor.submit(self._load_with_error_handling, source): source 
                for source in sources
            }
            
            for future in future_to_source:
                source = future_to_source[future]
                try:
                    documents = future.result()
                    result = LoadingResult(documents, source, success=True)
                    progress.update(source, True)
                except Exception as e:
                    result = LoadingResult([], source, success=False, error=str(e))
                    progress.update(source, False)
                
                results.append(result)
                
                if progress_callback:
                    progress_callback(progress)
        
        return results

    async def load_batch_async(
        self, 
        sources: List[str], 
        max_concurrency: int = 10,
        progress_callback: Optional[callable] = None
    ) -> List[LoadingResult]:
        """
        Load multiple sources asynchronously with concurrency control.
        
        Args:
            sources: List of source paths
            max_concurrency: Maximum number of concurrent operations
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of LoadingResult objects
        """
        results = []
        progress = LoadingProgress(len(sources))
        
        if progress_callback:
            progress_callback(progress)
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def load_with_semaphore(source: str) -> LoadingResult:
            async with semaphore:
                try:
                    documents = await self.load_async(source)
                    result = LoadingResult(documents, source, success=True)
                    progress.update(source, True)
                except Exception as e:
                    result = LoadingResult([], source, success=False, error=str(e))
                    progress.update(source, False)
                
                if progress_callback:
                    progress_callback(progress)
                
                return result
        
        tasks = [load_with_semaphore(source) for source in sources]
        results = await asyncio.gather(*tasks)
        
        return results

    def load_directory(
        self,
        directory_path: str,
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[LoadingResult]:
        """
        Load all supported files from a directory.
        
        Args:
            directory_path: Path to the directory to load files from
            recursive: Whether to search subdirectories recursively
            file_patterns: List of glob patterns to include (e.g., ["*.pdf", "*.txt"])
            exclude_patterns: List of glob patterns to exclude (e.g., ["*.tmp", "*.bak"])
            max_workers: Maximum number of worker threads
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of LoadingResult objects
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Get all files in directory
        files = self._get_files_from_directory(
            directory_path, 
            recursive, 
            file_patterns, 
            exclude_patterns
        )
        
        if not files:
            return []
        
        # Use existing batch loading functionality
        return self.load_batch(files, max_workers, progress_callback)

    async def load_directory_async(
        self,
        directory_path: str,
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_concurrency: int = 10,
        progress_callback: Optional[callable] = None
    ) -> List[LoadingResult]:
        """
        Asynchronously load all supported files from a directory.
        
        Args:
            directory_path: Path to the directory to load files from
            recursive: Whether to search subdirectories recursively
            file_patterns: List of glob patterns to include (e.g., ["*.pdf", "*.txt"])
            exclude_patterns: List of glob patterns to exclude (e.g., ["*.tmp", "*.bak"])
            max_concurrency: Maximum number of concurrent operations
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of LoadingResult objects
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Get all files in directory
        files = self._get_files_from_directory(
            directory_path, 
            recursive, 
            file_patterns, 
            exclude_patterns
        )
        
        if not files:
            return []
        
        # Use existing async batch loading functionality
        return await self.load_batch_async(files, max_concurrency, progress_callback)

    def stream_load(self, sources: List[str]) -> Iterator[LoadingResult]:
        """
        Stream loading results as they become available.
        
        Args:
            sources: List of source paths
            
        Yields:
            LoadingResult objects as they complete
        """
        for source in sources:
            try:
                documents = self.load(source)
                yield LoadingResult(documents, source, success=True)
            except Exception as e:
                yield LoadingResult([], source, success=False, error=str(e))

    async def stream_load_async(self, sources: List[str]) -> AsyncIterator[LoadingResult]:
        """
        Asynchronously stream loading results as they become available.
        
        Args:
            sources: List of source paths
            
        Yields:
            LoadingResult objects as they complete
        """
        tasks = [self._load_async_with_error_handling(source) for source in sources]
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            yield result

    def stream_load_directory(
        self,
        directory_path: str,
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Iterator[LoadingResult]:
        """
        Stream loading results from directory as they become available.
        
        Args:
            directory_path: Path to the directory to load files from
            recursive: Whether to search subdirectories recursively
            file_patterns: List of glob patterns to include
            exclude_patterns: List of glob patterns to exclude
            
        Yields:
            LoadingResult objects as they complete
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        files = self._get_files_from_directory(
            directory_path, 
            recursive, 
            file_patterns, 
            exclude_patterns
        )
        
        if not files:
            return
        
        # Use existing stream loading functionality
        yield from self.stream_load(files)

    async def stream_load_directory_async(
        self,
        directory_path: str,
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> AsyncIterator[LoadingResult]:
        """
        Asynchronously stream loading results from directory as they become available.
        
        Args:
            directory_path: Path to the directory to load files from
            recursive: Whether to search subdirectories recursively
            file_patterns: List of glob patterns to include
            exclude_patterns: List of glob patterns to exclude
            
        Yields:
            LoadingResult objects as they complete
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        files = self._get_files_from_directory(
            directory_path, 
            recursive, 
            file_patterns, 
            exclude_patterns
        )
        
        if not files:
            return
        
        # Use existing async stream loading functionality
        async for result in self.stream_load_async(files):
            yield result

    def _load_with_error_handling(self, source: str) -> List[Document]:
        """Load with error handling based on configuration."""
        start_time = time.time()
        
        try:
            # Validate source
            if not self._validate_source(source):
                raise ValueError(f"Invalid source: {source}")
            
            # Apply file size limits
            if self.config and self.config.max_file_size:
                if os.path.exists(source):
                    file_size = os.path.getsize(source)
                    if file_size > self.config.max_file_size:
                        raise ValueError(f"File size {file_size} exceeds limit {self.config.max_file_size}")
            
            documents = self.load(source)
            
            # Apply post-processing
            documents = self._post_process_documents(documents, source)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(documents), processing_time, success=True)
            
            return documents
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(0, processing_time, success=False)
            
            if self.config:
                if self.config.error_handling == "ignore":
                    return []
                elif self.config.error_handling == "warn":
                    print(f"Warning: Failed to load {source}: {e}")
                    return []
                else:  # raise
                    raise
            else:
                raise

    async def _load_async_with_error_handling(self, source: str) -> LoadingResult:
        """Async version of error handling wrapper."""
        try:
            documents = await self.load_async(source)
            return LoadingResult(documents, source, success=True)
        except Exception as e:
            return LoadingResult([], source, success=False, error=str(e))

    def _validate_source(self, source: str) -> bool:
        """Validate source path or URL."""
        if not source:
            return False
        
        # Check if it's a file path
        if os.path.exists(source):
            return os.path.isfile(source)
        
        # Check if it's a URL (basic validation)
        if source.startswith(('http://', 'https://', 'ftp://')):
            return True
        
        return False

    def _post_process_documents(self, documents: List[Document], source: str) -> List[Document]:
        """Apply post-processing to loaded documents."""
        if not documents:
            return documents
        
        processed_documents = []
        
        for doc in documents:
            # Skip empty content if configured
            if self.config and self.config.skip_empty_content:
                if not doc.content or not doc.content.strip():
                    continue
            
            # Add custom metadata
            if self.config and self.config.custom_metadata:
                doc.metadata.update(self.config.custom_metadata)
            
            # Add loader metadata
            doc.metadata['loader_type'] = self.__class__.__name__
            doc.metadata['source_path'] = source
            
            processed_documents.append(doc)
        
        return processed_documents

    def _update_stats(self, document_count: int, processing_time: float, success: bool):
        """Update internal performance statistics."""
        self._stats['total_files_processed'] += 1
        if success:
            self._stats['total_documents_created'] += document_count
        else:
            self._stats['total_errors'] += 1
        
        # Update average processing time
        total_time = self._stats['avg_processing_time'] * (self._stats['total_files_processed'] - 1)
        self._stats['avg_processing_time'] = (total_time + processing_time) / self._stats['total_files_processed']
        self._stats['last_processing_time'] = processing_time

    def get_stats(self) -> Dict[str, Any]:
        """Get loader performance statistics."""
        return self._stats.copy()

    def reset_stats(self):
        """Reset performance statistics."""
        self._stats = {
            'total_files_processed': 0,
            'total_documents_created': 0,
            'total_errors': 0,
            'avg_processing_time': 0.0,
            'last_processing_time': 0.0
        }

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of file extensions supported by this loader."""
        return []

    @classmethod
    def can_load(cls, source: str) -> bool:
        """Check if this loader can handle the given source."""
        if not source:
            return False
        
        extensions = cls.get_supported_extensions()
        if not extensions:
            return True
        
        source_ext = Path(source).suffix.lower()
        return source_ext in extensions

    def _get_files_from_directory(
        self,
        directory_path: str,
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of files from directory based on filters.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            file_patterns: List of glob patterns to include
            exclude_patterns: List of glob patterns to exclude
            
        Returns:
            List of file paths
        """
        import glob
        from fnmatch import fnmatch
        
        directory = Path(directory_path)
        files = []
        
        # Determine search pattern
        if recursive:
            search_pattern = "**/*"
        else:
            search_pattern = "*"
        
        # Get all files matching the search pattern
        all_files = list(directory.glob(search_pattern))
        
        for file_path in all_files:
            if not file_path.is_file():
                continue
            
            file_path_str = str(file_path)
            
            # Check if file matches include patterns
            if file_patterns:
                matches_pattern = False
                for pattern in file_patterns:
                    if fnmatch(file_path.name, pattern):
                        matches_pattern = True
                        break
                if not matches_pattern:
                    continue
            
            # Check if file matches exclude patterns
            if exclude_patterns:
                should_exclude = False
                for pattern in exclude_patterns:
                    if fnmatch(file_path.name, pattern):
                        should_exclude = True
                        break
                if should_exclude:
                    continue
            
            # Check if this loader can handle the file
            if self.can_load(file_path_str):
                files.append(file_path_str)
        
        return sorted(files)