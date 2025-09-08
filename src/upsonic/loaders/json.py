from __future__ import annotations
from typing import List, Any, Dict, Union, Iterator, Optional, Callable, AsyncIterator, Tuple
import os
import json
import asyncio
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass, field
from enum import Enum
import re
import gzip
import bz2
import lzma
from urllib.parse import urlparse
import aiohttp

from .base import DocumentLoader, LoadingResult, LoadingProgress
from .config import JSONLoaderConfig
from ..schemas.data_models import Document

try:
    import jq
    JQ_AVAILABLE = True
except ImportError:
    jq = None
    JQ_AVAILABLE = False

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    jsonschema = None
    JSONSCHEMA_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


class ValidationLevel(Enum):
    """JSON validation levels."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    SCHEMA = "schema"


@dataclass
class ProcessingStats:
    """Statistics for JSON processing operations."""
    total_objects_processed: int = 0
    total_documents_created: int = 0
    validation_errors: int = 0
    transformation_errors: int = 0
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    compression_ratio: float = 1.0
    start_time: float = field(default_factory=time.time)
    
    def update(self, objects_processed: int = 0, documents_created: int = 0, 
               validation_errors: int = 0, transformation_errors: int = 0):
        """Update statistics."""
        self.total_objects_processed += objects_processed
        self.total_documents_created += documents_created
        self.validation_errors += validation_errors
        self.transformation_errors += transformation_errors
        self.processing_time = time.time() - self.start_time


class JSONLoader(DocumentLoader):
    """
    JSON loader with comprehensive features for enterprise-grade JSON processing.
    
    Features:
    - Multi-format support (JSON, JSONL, compressed files)
    - Validation and schema checking
    - Streaming and batch processing
    - Custom transformations and filters
    - Memory-efficient processing
    - Async support with concurrency control
    - Comprehensive error handling and recovery
    - Performance monitoring and optimization
    - URL and remote file support
    - Data quality metrics and reporting
    """
    
    def __init__(
        self,
        config: Optional[JSONLoaderConfig] = None,
    ):
        """
        Initialize the JSON loader.
        
        Args:
            config: JSON loader configuration
        """
        if config is None:
            config = JSONLoaderConfig()
        
        
        super().__init__(config)
        
        self.validation_level = ValidationLevel(config.validation_level)
        self.schema_path = config.schema_path
        self.custom_validators = config.custom_validators or []
        self.transformers = config.transformers or []
        self.filters = config.filters or []
        self.max_memory_mb = config.max_memory_mb
        self.enable_streaming = config.enable_streaming
        self.enable_compression_detection = config.enable_compression_detection
        self.enable_url_support = config.enable_url_support
        self.chunk_size = config.chunk_size
        
        self.jq_schema = config.jq_schema
        self.is_jsonl = config.is_jsonl
        self.content_key = config.content_key
        self.flatten_metadata = config.flatten_metadata
        self.json_indent = config.json_indent
        
        self.schema = None
        self.stats = ProcessingStats()
        self.logger = logging.getLogger(__name__)
        
        if config.schema_path and JSONSCHEMA_AVAILABLE:
            self._load_schema(config.schema_path)
    
    def _load_schema(self, schema_path: str):
        """Load JSON schema from file."""
        try:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load schema from {schema_path}: {e}")
    
    def load(self, source: str) -> List[Document]:
        """
        Load JSON data with processing capabilities.
        
        Args:
            source: File path, URL, or raw JSON string
            
        Returns:
            List of processed Document objects
        """
        start_time = time.time()
        
        try:
            source_info = self._analyze_source(source)
            
            raw_data = self._load_raw_data(source_info)
            
            if not self._validate_data(raw_data, source_info):
                return []
            
            documents = self._process_data(raw_data, source_info)
            
            documents = self._apply_transformations(documents)
            documents = self._apply_filters(documents)
            
            self.stats.update(
                objects_processed=1,
                documents_created=len(documents)
            )
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading JSON from {source}: {e}")
            return []
    
    def load_object(self, obj: Union[Dict, List]) -> List[Document]:
        """
        Load JSON data from in-memory objects.
        
        Args:
            obj: Dictionary or list object to process
            
        Returns:
            List of processed Document objects
        """
        start_time = time.time()
        
        try:
            source_info = {
                'type': 'object',
                'compression': CompressionType.NONE,
                'size': len(str(obj)),
                'is_url': False,
                'is_file': False,
                'is_object': True,
                'is_string': False,
                'original_source': obj
            }
            
            if not self._validate_data(obj, source_info):
                return []
            
            documents = self._process_data(obj, source_info)
            
            documents = self._apply_transformations(documents)
            documents = self._apply_filters(documents)
            
            self.stats.update(
                objects_processed=1,
                documents_created=len(documents)
            )
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading JSON object: {e}")
            return []
    
    def _analyze_source(self, source: Union[str, Dict, List]) -> Dict[str, Any]:
        """Analyze source to determine type, compression, and other properties."""
        info = {
            'type': 'unknown',
            'compression': CompressionType.NONE,
            'size': 0,
            'is_url': False,
            'is_file': False,
            'is_object': False,
            'is_string': False,
            'original_source': source
        }
        
        if isinstance(source, (dict, list)):
            info.update({
                'type': 'object',
                'is_object': True,
                'size': len(str(source))
            })
        elif isinstance(source, str):
            info['is_string'] = True
            
            if self.enable_url_support and self._is_url(source):
                info.update({
                    'type': 'url',
                    'is_url': True
                })
            elif os.path.exists(source):
                info.update({
                    'type': 'file',
                    'is_file': True,
                    'size': os.path.getsize(source)
                })
                
                if self.enable_compression_detection:
                    info['compression'] = self._detect_compression(source)
            else:
                try:
                    json.loads(source)
                    info.update({
                        'type': 'json_string',
                        'size': len(source)
                    })
                except json.JSONDecodeError:
                    info['type'] = 'invalid_string'
        
        return info
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a valid URL."""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _detect_compression(self, file_path: str) -> CompressionType:
        """Detect compression type based on file extension and magic bytes."""
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.gz', '.gzip']:
            return CompressionType.GZIP
        elif ext in ['.bz2', '.bzip2']:
            return CompressionType.BZIP2
        elif ext in ['.xz', '.lzma']:
            return CompressionType.LZMA
        
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic.startswith(b'\x1f\x8b'):
                    return CompressionType.GZIP
                elif magic.startswith(b'BZ'):
                    return CompressionType.BZIP2
                elif magic.startswith(b'\xfd7zXZ'):
                    return CompressionType.LZMA
        except:
            pass
        
        return CompressionType.NONE
    
    def _load_raw_data(self, source_info: Dict[str, Any]) -> Any:
        """Load raw data based on source type and compression."""
        source = source_info.get('original_source')
        compression = source_info.get('compression')
        
        if source_info['is_object']:
            return source
        
        elif source_info['is_url']:
            return self._load_from_url(source)
        
        elif source_info['is_file']:
            return self._load_from_file(source, compression)
        
        elif source_info['type'] == 'json_string':
            return json.loads(source)
        
        else:
            raise ValueError(f"Unsupported source type: {source_info['type']}")
    
    def _load_from_url(self, url: str) -> Any:
        """Load JSON data from URL."""
        import requests
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError(f"Failed to load from URL {url}: {e}")
    
    def _load_from_file(self, file_path: str, compression: CompressionType) -> Any:
        """Load data from file with compression support."""
        open_func = open
        
        if compression == CompressionType.GZIP:
            open_func = gzip.open
        elif compression == CompressionType.BZIP2:
            open_func = bz2.open
        elif compression == CompressionType.LZMA:
            open_func = lzma.open
        
        mode = 'rt' if compression == CompressionType.NONE else 'rt'
        
        try:
            with open_func(file_path, mode, encoding='utf-8') as f:
                if self.is_jsonl:
                    data = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                                continue
                    return data
                else:
                    return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load file {file_path}: {e}")
    
    def _validate_data(self, data: Any, source_info: Dict[str, Any]) -> bool:
        """Validate JSON data according to configured validation level."""
        if self.validation_level == ValidationLevel.NONE:
            return True
        
        try:
            if self.validation_level == ValidationLevel.BASIC:
                if not isinstance(data, (dict, list)):
                    return False
            
            elif self.validation_level == ValidationLevel.STRICT:
                if not isinstance(data, (dict, list)):
                    return False
            
            elif self.validation_level == ValidationLevel.SCHEMA and self.schema:
                if not JSONSCHEMA_AVAILABLE:
                    self.logger.warning("jsonschema not available, skipping schema validation")
                    return True
                
                jsonschema.validate(instance=data, schema=self.schema)
            
            for validator in self.custom_validators:
                if not validator(data):
                    self.stats.validation_errors += 1
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            self.stats.validation_errors += 1
            return False
    
    def _process_data(self, data: Any, source_info: Dict[str, Any]) -> List[Document]:
        """Process JSON data into Document objects."""
        documents = []
        
        if self.enable_streaming and source_info.get('size', 0) > self.max_memory_mb * 1024 * 1024:
            for doc in self._stream_process(data):
                documents.append(doc)
        else:
            documents = self._process_in_memory(data)
        
        return documents
    
    def _stream_process(self, data: Any) -> Iterator[Document]:
        """Stream process large JSON data."""
        if isinstance(data, list):
            for i, item in enumerate(data):
                if i % self.chunk_size == 0:
                    pass
                
                try:
                    doc = self._create_document(item, {'index': i})
                    if doc:
                        yield doc
                except Exception as e:
                    self.logger.warning(f"Error processing item {i}: {e}")
                    self.stats.transformation_errors += 1
        else:
            doc = self._create_document(data, {})
            if doc:
                yield doc
    
    def _process_in_memory(self, data: Any) -> List[Document]:
        """Process JSON data in memory."""
        documents = []
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                try:
                    doc = self._create_document(item, {'index': i})
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    self.logger.warning(f"Error processing item {i}: {e}")
                    self.stats.transformation_errors += 1
        else:
            doc = self._create_document(data, {})
            if doc:
                documents.append(doc)
        
        return documents
    
    def _create_document(self, data: Any, metadata: Dict[str, Any]) -> Optional[Document]:
        """Create a Document from JSON data."""
        try:
            if self.jq_schema != '.' and JQ_AVAILABLE:
                content_data = list(jq.compile(self.jq_schema).iter(data))
                if not content_data:
                    return None
                content = content_data[0]
            elif self.content_key and isinstance(data, dict):
                content = data.get(self.content_key, data)
            else:
                content = data
            
            if isinstance(content, (dict, list)):
                content_str = json.dumps(content, indent=self.json_indent)
            else:
                content_str = str(content)
            
            doc_metadata = metadata.copy()
            if isinstance(data, dict):
                doc_metadata.update(data)
            
            if self.flatten_metadata:
                doc_metadata = self._flatten_dict(doc_metadata)
            
            doc_metadata.update({
                'loader_type': self.__class__.__name__,
                'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                'content_hash': hashlib.md5(content_str.encode()).hexdigest()
            })
            
            return Document(content=content_str, metadata=doc_metadata)
            
        except Exception as e:
            self.logger.error(f"Error creating document: {e}")
            return None
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Recursively flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _apply_transformations(self, documents: List[Document]) -> List[Document]:
        """Apply custom transformations to documents."""
        for transformer in self.transformers:
            try:
                documents = transformer(documents)
            except Exception as e:
                self.logger.error(f"Error in transformer {transformer.__name__}: {e}")
                self.stats.transformation_errors += 1
        
        return documents
    
    def _apply_filters(self, documents: List[Document]) -> List[Document]:
        """Apply custom filters to documents."""
        filtered_docs = []
        
        for doc in documents:
            include = True
            for filter_func in self.filters:
                try:
                    if not filter_func(doc):
                        include = False
                        break
                except Exception as e:
                    self.logger.error(f"Error in filter {filter_func.__name__}: {e}")
                    include = False
                    break
            
            if include:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    async def load_async(self, source: str) -> List[Document]:
        """Asynchronous version of load method."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.load, source)
    
    async def load_async_object(self, obj: Union[Dict, List]) -> List[Document]:
        """Asynchronous version of load_object method."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.load_object, obj)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            'total_objects_processed': self.stats.total_objects_processed,
            'total_documents_created': self.stats.total_documents_created,
            'validation_errors': self.stats.validation_errors,
            'transformation_errors': self.stats.transformation_errors,
            'processing_time_seconds': self.stats.processing_time,
            'memory_usage_mb': self.stats.memory_usage_mb,
            'compression_ratio': self.stats.compression_ratio,
            'success_rate': (
                (self.stats.total_documents_created / max(self.stats.total_objects_processed, 1)) * 100
            )
        }
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = ProcessingStats()
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return [
            '.json', '.jsonl', '.js', '.json.gz', '.json.bz2', '.json.xz',
            '.jsonl.gz', '.jsonl.bz2', '.jsonl.xz'
        ]
    
    @classmethod
    def can_load(cls, source: str) -> bool:
        """Check if this loader can handle the given source."""
        if not source:
            return False
        
        extensions = cls.get_supported_extensions()
        source_path = Path(source)
        
        suffixes = source_path.suffixes
        if not suffixes:
            return False
        
        if len(suffixes) == 1:
            return suffixes[0].lower() in extensions
        
        if len(suffixes) >= 2:
            double_ext = ''.join(suffixes[-2:]).lower()
            return double_ext in extensions
        
        return False
    
    def export_to_dataframe(self, documents: List[Document]) -> Optional['pd.DataFrame']:
        """Export documents to pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            self.logger.warning("pandas not available for DataFrame export")
            return None
        
        try:
            data = []
            for doc in documents:
                row = {
                    'content': doc.content,
                    'content_length': len(doc.content),
                    'content_hash': doc.metadata.get('content_hash', ''),
                    'processing_timestamp': doc.metadata.get('processing_timestamp', ''),
                    **{k: v for k, v in doc.metadata.items() 
                       if k not in ['content_hash', 'processing_timestamp']}
                }
                data.append(row)
            
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Error creating DataFrame: {e}")
            return None


def filter_by_content_length(min_length: int = 0, max_length: Optional[int] = None):
    """Create a filter function for content length."""
    def filter_func(doc: Document) -> bool:
        length = len(doc.content)
        if length < min_length:
            return False
        if max_length and length > max_length:
            return False
        return True
    return filter_func


def filter_by_metadata_key(key: str, value: Any):
    """Create a filter function for metadata key-value pairs."""
    def filter_func(doc: Document) -> bool:
        return doc.metadata.get(key) == value
    return filter_func


def transform_add_timestamp():
    """Create a transformation function to add timestamps."""
    def transform_func(documents: List[Document]) -> List[Document]:
        timestamp = datetime.now(timezone.utc).isoformat()
        for doc in documents:
            doc.metadata['transformed_at'] = timestamp
        return documents
    return transform_func


def transform_content_cleanup():
    """Create a transformation function to clean content."""
    def transform_func(documents: List[Document]) -> List[Document]:
        for doc in documents:
            doc.content = re.sub(r'\s+', ' ', doc.content).strip()
            doc.content = doc.content.replace('\x00', '')
        return documents
    return transform_func
