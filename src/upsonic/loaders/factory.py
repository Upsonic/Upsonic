from __future__ import annotations
from typing import Dict, Type, Optional, List, Union, Any, Tuple, TYPE_CHECKING
import os
import json
from pathlib import Path
from functools import lru_cache

from upsonic.utils.logging_config import get_logger
from .base import BaseLoader
from .config import LoaderConfig, LoaderConfigFactory

if TYPE_CHECKING:
    from upsonic.schemas.data_models import Document
    from .config import (
        TextLoaderConfig, CSVLoaderConfig, PdfLoaderConfig, DOCXLoaderConfig,
        JSONLoaderConfig, XMLLoaderConfig, YAMLLoaderConfig, MarkdownLoaderConfig,
        HTMLLoaderConfig
    )
    from .text import TextLoader
    from .csv import CSVLoader
    from .pdf import PdfLoader
    from .docx import DOCXLoader
    from .json import JSONLoader
    from .xml import XMLLoader
    from .yaml import YAMLLoader
    from .markdown import MarkdownLoader
    from .html import HTMLLoader

logger = get_logger(__name__)

class LoaderFactory:
    """
    A factory class for creating and managing document loaders.

    This factory provides intelligent loader creation, configuration management,
    and extension-based loader detection. It supports both synchronous and
    asynchronous operations with proper error handling and fallback mechanisms.
    """

    _REQUIRED_METHODS = ['load', 'aload', 'batch', 'abatch', 'get_supported_extensions']
    _URL_PREFIXES = ('http://', 'https://', 'ftp://')
    _CONTENT_PREVIEW_SIZE = 1024
    _LARGE_FILE_THRESHOLD = 100 * 1024 * 1024
    _PDF_LARGE_THRESHOLD = 50 * 1024 * 1024
    def __init__(self):
        self._loaders: Dict[str, Type[BaseLoader]] = {}
        self._extensions: Dict[str, str] = {}
        self._configs: Dict[str, Type[LoaderConfig]] = {}
        
        self._register_default_loaders()
        self._validate_registration()
    
    def _register_default_loaders(self):
        """Register all default loaders with their extensions."""
        default_loaders = self._get_default_loader_configs()
        
        for loader_class, extensions in default_loaders:
            self.register_loader(loader_class, extensions)
    
    @staticmethod
    def _get_default_loader_configs() -> List[Tuple[Type[BaseLoader], List[str]]]:
        """Get the default loader configurations with lazy imports.
        Each loader import is wrapped in try/except to handle cases where
        optional dependencies are not installed. This ensures auto-detection
        works even when some loaders cannot be imported.
        """
        loaders = []
        try:
             # CSVLoader - requires aiofiles
            from .csv import CSVLoader
            loaders.append((CSVLoader, ['.csv']))
        except ImportError as e:
            logger.debug(f"CSVLoader not available: {e}")
        
        # PdfLoader - requires pypdf
        try:
            from .pdf import PdfLoader
            loaders.append((PdfLoader, ['.pdf']))
        except ImportError as e:
            logger.debug(f"PdfLoader not available: {e}")
        
        # PyMuPDFLoader - requires pymupdf
        try:
            from .pymupdf import PyMuPDFLoader
            loaders.append((PyMuPDFLoader, ['.pdf']))
        except ImportError as e:
            logger.debug(f"PyMuPDFLoader not available: {e}")
        
        # PdfPlumberLoader - requires pdfplumber
        try:
            from .pdfplumber import PdfPlumberLoader
            loaders.append((PdfPlumberLoader, ['.pdf']))
        except ImportError as e:
            logger.debug(f"PdfPlumberLoader not available: {e}")
        
        # DOCXLoader - requires python-docx
        try:
            from .docx import DOCXLoader
            loaders.append((DOCXLoader, ['.docx']))
        except ImportError as e:
            logger.debug(f"DOCXLoader not available: {e}")
        
        # JSONLoader - requires jq
        try:
            from .json import JSONLoader
            loaders.append((JSONLoader, ['.json', '.jsonl']))
        except ImportError as e:
            logger.debug(f"JSONLoader not available: {e}")
        
        # XMLLoader - requires lxml
        try:
            from .xml import XMLLoader
            loaders.append((XMLLoader, ['.xml']))
        except (ImportError, AttributeError) as e:
            logger.debug(f"XMLLoader not available: {e}")
        
        # YAMLLoader - requires pyyaml and jq
        try:
            from .yaml import YAMLLoader
            loaders.append((YAMLLoader, ['.yaml', '.yml']))
        except ImportError as e:
            logger.debug(f"YAMLLoader not available: {e}")
        
        # MarkdownLoader - requires python-frontmatter
        try:
            from .markdown import MarkdownLoader
            loaders.append((MarkdownLoader, ['.md', '.markdown']))
        except ImportError as e:
            logger.debug(f"MarkdownLoader not available: {e}")
        
        # HTMLLoader - requires aiohttp, requests, beautifulsoup4
        try:
            from .html import HTMLLoader
            loaders.append((HTMLLoader, ['.html', '.htm', '.xhtml']))
        except ImportError as e:
            logger.debug(f"HTMLLoader not available: {e}")
        
        # TextLoader - requires aiofiles (minimal dependencies)
        try:
            from .text import TextLoader
            loaders.append((TextLoader, ['.txt', '.rst', '.log', '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.cs', '.go', '.rs', '.php', '.rb', '.css', '.ini']))
        except ImportError as e:
            logger.debug(f"TextLoader not available: {e}")
        
        # DoclingLoader - optional advanced loader
        try:
            from .docling import DoclingLoader
            # Docling supports a wide range of formats, register with high priority
            loaders.insert(0, (DoclingLoader, [
                '.pdf', '.docx', '.xlsx', '.pptx',
                '.html', '.htm',
                '.md', '.markdown',
                '.adoc', '.asciidoc',
                '.csv',
                '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'
            ]))
        except ImportError as e:
            logger.debug(f"DoclingLoader not available: {e}")
        
        return loaders
    
    def register_loader(self, loader_class: Type[BaseLoader], extensions: List[str]):
        """
        Register a loader class with its supported extensions.
        
        Args:
            loader_class: The loader class to register
            extensions: List of file extensions this loader supports
            
        Raises:
            ValueError: If loader_class is not a subclass of BaseLoader
            ValueError: If extensions list is empty or contains invalid extensions
        """
        if not issubclass(loader_class, BaseLoader):
            raise ValueError(f"Loader class {loader_class.__name__} must be a subclass of BaseLoader")
        
        if not extensions:
            raise ValueError("Extensions list cannot be empty")
        
        for ext in extensions:
            if not ext.startswith('.') or len(ext) < 2:
                raise ValueError(f"Invalid extension '{ext}'. Extensions must start with '.' and be at least 2 characters")
        
        loader_name = loader_class.__name__.lower().replace('loader', '')
        
        conflicting_extensions = []
        for ext in extensions:
            if ext.lower() in self._extensions and self._extensions[ext.lower()] != loader_name:
                conflicting_extensions.append(ext)
        
        if conflicting_extensions:
            logger.debug(f"Extension conflicts detected for {loader_name}: {conflicting_extensions}")
            for ext in conflicting_extensions:
                old_loader = self._extensions[ext.lower()]
                logger.debug(f"Replacing {old_loader} with {loader_name} for extension {ext}")
        
        self._loaders[loader_name] = loader_class
        
        for ext in extensions:
            self._extensions[ext.lower()] = loader_name
        
        config_mapping = self._get_config_mapping()
        
        if loader_name in config_mapping:
            self._configs[loader_name] = config_mapping[loader_name]
        
        logger.debug(f"Registered loader '{loader_name}' with extensions: {extensions}")
    
    @staticmethod
    def _get_config_mapping() -> Dict[str, Type[LoaderConfig]]:
        """Get the mapping of loader names to their config classes with lazy imports."""
        from .config import (
            TextLoaderConfig, CSVLoaderConfig, PdfLoaderConfig, PyMuPDFLoaderConfig, PdfPlumberLoaderConfig,
            DOCXLoaderConfig, JSONLoaderConfig, XMLLoaderConfig, YAMLLoaderConfig, MarkdownLoaderConfig,
            HTMLLoaderConfig
        )
        
        config_map = {
            'text': TextLoaderConfig,
            'csv': CSVLoaderConfig,
            'pdf': PdfLoaderConfig,
            'pymupdf': PyMuPDFLoaderConfig,
            'pdfplumber': PdfPlumberLoaderConfig,
            'docx': DOCXLoaderConfig,
            'json': JSONLoaderConfig,
            'xml': XMLLoaderConfig,
            'yaml': YAMLLoaderConfig,
            'markdown': MarkdownLoaderConfig,
            'html': HTMLLoaderConfig
        }
        
        # Try to add DoclingLoaderConfig if available
        try:
            from .config import DoclingLoaderConfig
            config_map['docling'] = DoclingLoaderConfig
        except ImportError:
            pass  # Docling not installed
        
        return config_map
    
    def _validate_registration(self):
        """Validate that all registered loaders are properly configured."""
        self._validate_loader_methods()
        self._validate_loader_configs()
        self._check_duplicate_extensions()
    
    def _validate_loader_methods(self):
        """Validate that all loaders have required methods."""
        for loader_name, loader_class in self._loaders.items():
            for method in self._REQUIRED_METHODS:
                if not hasattr(loader_class, method):
                    logger.warning(f"Loader {loader_name} missing required method: {method}")
    
    def _validate_loader_configs(self):
        """Validate that all loaders have config classes."""
        for loader_name in self._loaders:
            if loader_name not in self._configs:
                logger.warning(f"No config class found for loader: {loader_name}")
    
    def _check_duplicate_extensions(self):
        """Check for duplicate extensions across loaders."""
        extension_counts = {}
        for ext, loader_name in self._extensions.items():
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
        
        duplicates = {ext: count for ext, count in extension_counts.items() if count > 1}
        if duplicates:
            logger.warning(f"Duplicate extensions found: {duplicates}")
    
    def get_loader(self, source: str, loader_type: Optional[str] = None, **config_kwargs) -> BaseLoader:
        """
        Get a configured loader instance for the given source.
        
        Args:
            source: The source path or content to load
            loader_type: Optional specific loader type to use
            **config_kwargs: Configuration parameters for the loader
            
        Returns:
            A configured BaseLoader instance
            
        Raises:
            ValueError: If no suitable loader is found
            ValueError: If configuration is invalid
            FileNotFoundError: If source is a file path that doesn't exist
        """
        try:
            if not source:
                raise ValueError(f"Source cannot be None or empty")
            
            if not source.startswith(('http://', 'https://', 'ftp://')) and not source.startswith('{') and not source.startswith('<') and not source.startswith('#'):
                if not os.path.exists(source):
                    raise FileNotFoundError(f"Source file does not exist: {source}")
            
            if loader_type:
                loader_name = loader_type.lower()
                if loader_name not in self._loaders:
                    raise ValueError(f"No loader found for type '{loader_type}'. Available types: {list(self._loaders.keys())}")
            else:
                loader_name = self._detect_loader_type(source)
                if loader_name not in self._loaders:
                    raise ValueError(f"No loader found for source '{source}'. Detected type: '{loader_name}'. Available types: {list(self._loaders.keys())}")
            
            loader_class = self._loaders[loader_name]
            
            config = self._create_config(loader_name, **config_kwargs)
            
            return loader_class(config)
            
        except Exception as e:
            logger.error(f"Failed to create loader for '{source}': {e}")
            raise
    
    def create_intelligent_loaders(self, sources: List[Union[str, Path]], **global_config_kwargs) -> List[BaseLoader]:
        """
        Intelligently create appropriate loaders for a list of sources.
        
        This method analyzes each source and creates the most appropriate loader
        with optimized configuration based on the source type and content.
        String content sources are skipped (no loader needed for direct content).
        
        Args:
            sources: List of source paths (Path objects) or content strings
            **global_config_kwargs: Global configuration options to apply to all loaders
            
        Returns:
            List of configured BaseLoader instances (only for file sources)
        """
        loaders = []
        
        for source in sources:
            if isinstance(source, str):
                logger.debug(f"Skipping loader creation for string content: {source[:50]}...")
                continue
                
            try:
                source_str = str(source)
                
                loader_type = self._detect_loader_type(source_str)
                
                source_config = self._create_optimized_config(source_str, loader_type, **global_config_kwargs)
                
                loader = self.get_loader(source_str, loader_type, **source_config)
                loaders.append(loader)
                
                logger.info(f"Created {loader_type.capitalize()}Loader for {source}")
                
            except Exception as e:
                logger.warning(f"Failed to create loader for {source}: {e}")
                try:
                    fallback_config = self._create_optimized_config(source_str, 'text', **global_config_kwargs)
                    fallback_loader = self.get_loader(source_str, 'text', **fallback_config)
                    loaders.append(fallback_loader)
                    logger.info(f"Using text loader fallback for {source}")
                except Exception as fallback_error:
                    logger.error(f"Fallback loader also failed for {source}: {fallback_error}")
                    raise
        
        return loaders
    
    def _create_optimized_config(self, source: str, loader_type: str, **global_config_kwargs) -> Dict[str, Any]:
        """
        Create optimized configuration for a specific source and loader type.
        
        Args:
            source: Source path or content
            loader_type: Type of loader to configure
            **global_config_kwargs: Global configuration options
            
        Returns:
            Optimized configuration dictionary
        """
        config = global_config_kwargs.copy()
        
        if not (os.path.exists(source) and os.path.isfile(source)):
            return config
        
        file_size = os.path.getsize(source)
        
        if file_size > self._LARGE_FILE_THRESHOLD:
            config.setdefault('max_file_size', file_size)
        
        self._apply_loader_optimizations(config, loader_type, source, file_size)
        
        return config
    
    def _apply_loader_optimizations(self, config: Dict[str, Any], loader_type: str, source: str, file_size: int) -> None:
        """Apply loader-specific optimizations."""
        if loader_type == 'pdf':
            self._optimize_pdf_config(config, file_size)
        elif loader_type == 'pymupdf':
            self._optimize_pymupdf_config(config, file_size)
        elif loader_type == 'docling':
            self._optimize_docling_config(config, file_size, source)
        elif loader_type == 'html':
            self._optimize_html_config(config, source)
        elif loader_type == 'csv':
            self._optimize_csv_config(config, file_size)
        elif loader_type == 'json':
            self._optimize_json_config(config, file_size)
        elif loader_type == 'xml':
            self._optimize_xml_config(config, file_size)
        elif loader_type == 'yaml':
            self._optimize_yaml_config(config, file_size)
        elif loader_type == 'markdown':
            self._optimize_markdown_config(config, file_size)
        elif loader_type == 'docx':
            self._optimize_docx_config(config, file_size)
        elif loader_type == 'text':
            self._optimize_text_config(config, file_size)
    
    def _optimize_pdf_config(self, config: Dict[str, Any], file_size: int) -> None:
        """Optimize PDF loader configuration based on file size."""
        if file_size > self._PDF_LARGE_THRESHOLD:
            config.setdefault('extraction_mode', 'text_only')
        else:
            config.setdefault('extraction_mode', 'hybrid')
    
    def _optimize_pymupdf_config(self, config: Dict[str, Any], file_size: int) -> None:
        """Optimize PyMuPDF loader configuration based on file size."""
        if file_size > self._PDF_LARGE_THRESHOLD:
            config.setdefault('extraction_mode', 'text_only')
            config.setdefault('include_images', False)
            config.setdefault('extract_annotations', False)
            config.setdefault('image_dpi', 72)  # Lower DPI for large files
        else:
            config.setdefault('extraction_mode', 'hybrid')
            config.setdefault('include_images', True)
            config.setdefault('extract_annotations', True)
            config.setdefault('image_dpi', 150)  # Higher DPI for better quality
            config.setdefault('text_extraction_method', 'dict')  # Better structure for smaller files
    
    def _optimize_docling_config(self, config: Dict[str, Any], file_size: int, source: str) -> None:
        """Optimize Docling loader configuration based on file size and source type."""
        # Determine if source is URL
        is_url = source.startswith(('http://', 'https://'))
        
        if is_url:
            config.setdefault('support_urls', True)
            config.setdefault('url_timeout', 60)  # Longer timeout for URLs
        
        # Optimize based on file size
        if file_size > self._PDF_LARGE_THRESHOLD:
            # Large files: prioritize speed
            config.setdefault('extraction_mode', 'chunks')
            config.setdefault('chunker_type', 'hybrid')
            config.setdefault('max_pages', 100)  # Limit pages for very large files
            config.setdefault('parallel_processing', True)
            config.setdefault('batch_size', 5)
            config.setdefault('extract_document_metadata', False)
            
            # OCR: Disable for speed on large files
            config.setdefault('ocr_enabled', False)
            config.setdefault('enable_table_structure', True)
            config.setdefault('table_structure_cell_matching', False)  # Disable cell matching for speed
        else:
            # Smaller files: prioritize quality
            config.setdefault('extraction_mode', 'chunks')
            config.setdefault('chunker_type', 'hierarchical')  # Better structure for smaller files
            config.setdefault('parallel_processing', True)
            config.setdefault('batch_size', 10)
            config.setdefault('extract_document_metadata', True)
            config.setdefault('confidence_threshold', 0.7)  # Higher quality threshold
            
            # OCR: Full features for quality
            config.setdefault('ocr_enabled', True)
            config.setdefault('ocr_force_full_page', False)  # Hybrid mode (smart OCR)
            config.setdefault('ocr_backend', 'rapidocr')  # Fast and accurate
            config.setdefault('ocr_lang', ['english'])
            config.setdefault('ocr_backend_engine', 'onnxruntime')  # Best compatibility
            config.setdefault('ocr_text_score', 0.6)  # Good balance
            config.setdefault('enable_table_structure', True)
            config.setdefault('table_structure_cell_matching', True)  # Full quality
    
    def _optimize_html_config(self, config: Dict[str, Any], source: str) -> None:
        """Optimize HTML loader configuration for URLs."""
        if source.startswith(self._URL_PREFIXES):
            config.setdefault('extract_metadata', True)
            config.setdefault('extract_text', True)
            config.setdefault('user_agent', 'Upsonic HTML Loader 1.0')
    
    def _optimize_csv_config(self, config: Dict[str, Any], file_size: int) -> None:
        """Optimize CSV loader configuration based on file size and content."""
        if file_size > self._LARGE_FILE_THRESHOLD:
            config['include_metadata'] = False
            config['content_synthesis_mode'] = 'concatenated'
        else:
            config.setdefault('include_metadata', True)
            config.setdefault('content_synthesis_mode', 'json')
    
    def _optimize_json_config(self, config: Dict[str, Any], file_size: int) -> None:
        """Optimize JSON loader configuration based on file size and structure."""
        if file_size > self._LARGE_FILE_THRESHOLD:
            config['include_metadata'] = False
            config['mode'] = 'single'
        else:
            config.setdefault('include_metadata', True)
            config.setdefault('mode', 'multi')
    
    def _optimize_xml_config(self, config: Dict[str, Any], file_size: int) -> None:
        """Optimize XML loader configuration based on file size and structure."""
        if file_size > self._LARGE_FILE_THRESHOLD:
            config['include_metadata'] = False
            config['content_synthesis_mode'] = 'smart_text'
            config['strip_namespaces'] = False
        else:
            config.setdefault('include_metadata', True)
            config.setdefault('content_synthesis_mode', 'xml_snippet')
            config.setdefault('strip_namespaces', True)
    
    def _optimize_yaml_config(self, config: Dict[str, Any], file_size: int) -> None:
        """Optimize YAML loader configuration based on file size and structure."""
        if file_size > self._LARGE_FILE_THRESHOLD:
            config['include_metadata'] = False
            config['content_synthesis_mode'] = 'smart_text'
            config['flatten_metadata'] = False
        else:
            config.setdefault('include_metadata', True)
            config.setdefault('content_synthesis_mode', 'canonical_yaml')
            config.setdefault('flatten_metadata', True)
    
    def _optimize_markdown_config(self, config: Dict[str, Any], file_size: int) -> None:
        """Optimize Markdown loader configuration based on file size and content."""
        if file_size > self._LARGE_FILE_THRESHOLD:
            config['include_metadata'] = False
            config['code_block_language_metadata'] = False
            config['heading_metadata'] = False
        else:
            config.setdefault('include_metadata', True)
            config.setdefault('code_block_language_metadata', True)
            config.setdefault('heading_metadata', True)
            config.setdefault('parse_front_matter', True)
    
    def _optimize_docx_config(self, config: Dict[str, Any], file_size: int) -> None:
        """Optimize DOCX loader configuration based on file size and content."""
        if file_size > self._LARGE_FILE_THRESHOLD:
            config['include_metadata'] = False
            config['table_format'] = 'text'
            config['include_headers'] = False
            config['include_footers'] = False
        else:
            config.setdefault('include_metadata', True)
            config.setdefault('table_format', 'markdown')
            config.setdefault('include_headers', True)
            config.setdefault('include_footers', True)
    
    def _optimize_text_config(self, config: Dict[str, Any], file_size: int) -> None:
        """Optimize Text loader configuration based on file size and content."""
        if file_size > self._LARGE_FILE_THRESHOLD:
            config['include_metadata'] = False
            config['min_chunk_length'] = 100
        else:
            config.setdefault('include_metadata', True)
            config.setdefault('min_chunk_length', 1)
            config.setdefault('strip_whitespace', True)
    
    @lru_cache(maxsize=1000)
    def _detect_loader_type(self, source: str) -> str:
        """
        Detect the appropriate loader type for a given source.
        
        Args:
            source: The source path or content string
            
        Returns:
            The loader type name
        """
        if source.startswith(self._URL_PREFIXES):
            return 'html'
        
        file_path = Path(source)
        extension = file_path.suffix.lower()
        
        if extension in self._extensions:
            return self._extensions[extension]
        
        if len(file_path.suffixes) >= 2:
            double_ext = ''.join(file_path.suffixes[-2:]).lower()
            if double_ext in self._extensions:
                return self._extensions[double_ext]
        
        if os.path.exists(source) and os.path.isfile(source):
            return self._detect_file_content_type(source)
        
        return self._detect_content_type(source)
    
    def _detect_file_content_type(self, file_path: str) -> str:
        """Detect content type for existing files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content_preview = f.read(self._CONTENT_PREVIEW_SIZE)
                return self._detect_content_type(content_preview)
        except Exception:
            return 'text'
    
    def _detect_content_type(self, content: str) -> str:
        """Detect content type based on content structure."""
        content_stripped = content.strip()
        
        if self._is_json_content(content_stripped):
            return 'json'
        
        if self._is_html_content(content_stripped):
            return 'html'
        
        if self._is_xml_content(content_stripped):
            return 'xml'
        
        if self._is_yaml_content(content_stripped):
            return 'yaml'
        
        if self._is_markdown_content(content_stripped):
            return 'markdown'
        
        return 'text'
    
    @staticmethod
    def _is_json_content(content: str) -> bool:
        """Check if content is JSON."""
        if not ((content.startswith('{') or content.startswith('[')) and 
                (content.endswith('}') or content.endswith(']'))):
            return False
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    @staticmethod
    def _is_xml_content(content: str) -> bool:
        """Check if content is XML."""
        return content.startswith('<') and '>' in content
    
    @staticmethod
    def _is_yaml_content(content: str) -> bool:
        """Check if content is YAML."""
        return content.startswith('---') or content.startswith('- ')
    
    @staticmethod
    def _is_markdown_content(content: str) -> bool:
        """Check if content is Markdown."""
        markdown_prefixes = ['# ', '## ', '### ']
        return any(content.startswith(prefix) for prefix in markdown_prefixes)
    
    @staticmethod
    def _is_html_content(content: str) -> bool:
        """Check if content is HTML."""
        if not content.startswith('<'):
            return False
        html_tags = ['<html', '<head', '<body', '<div']
        return any(tag in content.lower() for tag in html_tags)
    
    def _create_config(self, loader_name: str, **config_kwargs) -> LoaderConfig:
        """
        Create a configuration instance for the specified loader.
        
        Args:
            loader_name: Name of the loader to create config for
            **config_kwargs: Configuration parameters
            
        Returns:
            A LoaderConfig instance
            
        Raises:
            ValueError: If configuration creation fails
        """
        try:
            if loader_name in self._configs:
                return self._create_specific_config(loader_name, config_kwargs)
            else:
                return self._create_fallback_config(loader_name, config_kwargs)
        except Exception as e:
            logger.warning(f"Failed to create config for '{loader_name}': {e}")
            return self._create_base_config(config_kwargs)
    
    def _create_specific_config(self, loader_name: str, config_kwargs: Dict[str, Any]) -> LoaderConfig:
        """Create config for a specific loader type."""
        config_class = self._configs[loader_name]
        self._validate_config_params(config_class, config_kwargs)
        return config_class(**config_kwargs)
    
    def _create_fallback_config(self, loader_name: str, config_kwargs: Dict[str, Any]) -> LoaderConfig:
        """Create config using LoaderConfigFactory as fallback."""
        if config_kwargs:
            return LoaderConfigFactory.create_config(loader_name, **config_kwargs)
        else:
            return LoaderConfigFactory.create_config(loader_name)
    
    def _create_base_config(self, config_kwargs: Dict[str, Any]) -> LoaderConfig:
        """Create base config with only valid parameters."""
        valid_kwargs = {k: v for k, v in config_kwargs.items() 
                      if k in LoaderConfig.model_fields}
        return LoaderConfig(**valid_kwargs) if valid_kwargs else LoaderConfig()
    
    def _validate_config_params(self, config_class: Type[LoaderConfig], config_kwargs: Dict[str, Any]) -> None:
        """Validate configuration parameters against the config class."""
        if not hasattr(config_class, 'model_fields'):
            return
        
        invalid_params = []
        for param_name in config_kwargs.keys():
            if param_name not in config_class.model_fields:
                invalid_params.append(param_name)
        
        if invalid_params:
            logger.warning(f"Invalid config parameters for {config_class.__name__}: {invalid_params}")
            for param in invalid_params:
                config_kwargs.pop(param, None)
    
    def get_loader_for_file(self, file_path: str, **config_kwargs) -> BaseLoader:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
        
        return self.get_loader(file_path, **config_kwargs)
    
    def get_loader_for_content(self, content: str, content_type: str, **config_kwargs) -> BaseLoader:
        return self.get_loader(content, content_type, **config_kwargs)
    
    def get_loaders_for_directory(
        self, 
        directory_path: str, 
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        **config_kwargs
    ) -> Dict[str, BaseLoader]:
        """
        Get loaders for all files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
            file_patterns: List of file patterns to include
            exclude_patterns: List of file patterns to exclude
            **config_kwargs: Configuration parameters for loaders
            
        Returns:
            Dictionary mapping file paths to their loaders
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        loaders = {}
        directory = Path(directory_path)
        
        all_files = self._get_directory_files(directory, recursive)
        
        filtered_files = self._filter_files_by_patterns(all_files, file_patterns, exclude_patterns)
        
        for file_path in filtered_files:
            try:
                loader = self.get_loader_for_file(str(file_path), **config_kwargs)
                loaders[str(file_path)] = loader
            except Exception as e:
                logger.warning(f"Could not create loader for {file_path}: {e}")
                continue
        
        return loaders
    
    def _get_directory_files(self, directory: Path, recursive: bool) -> List[Path]:
        """Get all files from a directory."""
        if recursive:
            return [f for f in directory.rglob("*") if f.is_file()]
        else:
            return [f for f in directory.glob("*") if f.is_file()]
    
    def _filter_files_by_patterns(
        self, 
        files: List[Path], 
        include_patterns: Optional[List[str]], 
        exclude_patterns: Optional[List[str]]
    ) -> List[Path]:
        """Filter files based on include/exclude patterns."""
        filtered_files = files
        
        if include_patterns:
            filtered_files = [
                f for f in filtered_files 
                if any(f.match(pattern) for pattern in include_patterns)
            ]
        
        if exclude_patterns:
            filtered_files = [
                f for f in filtered_files 
                if not any(f.match(pattern) for pattern in exclude_patterns)
            ]
        
        return filtered_files
    
    def get_supported_extensions(self) -> List[str]:
        return list(self._extensions.keys())
    
    def get_supported_loaders(self) -> List[str]:
        return list(self._loaders.keys())
    
    def can_handle(self, source: str) -> bool:
        """Check if the factory can handle the given source."""
        validation = self.validate_source(source)
        return validation['can_handle']
    
    def get_loader_info(self, loader_type: str) -> Dict[str, Any]:
        if loader_type not in self._loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        loader_class = self._loaders[loader_type]
        
        info = {
            'name': loader_type,
            'class': loader_class.__name__,
            'extensions': [ext for ext, name in self._extensions.items() if name == loader_type],
            'description': getattr(loader_class, '__doc__', 'No description available'),
            'has_config': loader_type in self._configs
        }
        
        return info
    
    def list_loaders(self) -> List[Dict[str, Any]]:
        return [self.get_loader_info(loader_type) for loader_type in self._loaders.keys()]
    
    def clear_cache(self):
        """Clear any cached data."""
        self._detect_loader_type.cache_clear()
        logger.debug("Factory cache cleared")
    
    def get_loader_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered loaders."""
        return {
            'total_loaders': len(self._loaders),
            'total_extensions': len(self._extensions),
            'loaders': list(self._loaders.keys()),
            'extensions': list(self._extensions.keys()),
            'configs_available': len(self._configs),
            'extension_conflicts': self._find_extension_conflicts()
        }
    
    def _find_extension_conflicts(self) -> Dict[str, List[str]]:
        """Find extensions that are handled by multiple loaders."""
        extension_to_loaders = {}
        for ext, loader_name in self._extensions.items():
            if ext not in extension_to_loaders:
                extension_to_loaders[ext] = []
            extension_to_loaders[ext].append(loader_name)
        
        return {ext: loaders for ext, loaders in extension_to_loaders.items() if len(loaders) > 1}
    
    def validate_source(self, source: str) -> Dict[str, Any]:
        """Validate a source and return information about it."""
        result = {
            'source': source,
            'is_url': source.startswith(('http://', 'https://', 'ftp://')),
            'is_file': os.path.exists(source) and os.path.isfile(source),
            'detected_type': None,
            'can_handle': False,
            'recommended_loader': None
        }
        
        try:
            if result['is_url']:
                detected_type = 'html'
                result['detected_type'] = detected_type
                result['can_handle'] = detected_type in self._loaders
                result['recommended_loader'] = detected_type if result['can_handle'] else None
            elif result['is_file']:
                detected_type = self._detect_loader_type(source)
                result['detected_type'] = detected_type
                result['can_handle'] = detected_type in self._loaders
                result['recommended_loader'] = detected_type if result['can_handle'] else None
            elif source.startswith(('{', '[', '<', '#', '---')):
                detected_type = self._detect_loader_type(source)
                result['detected_type'] = detected_type
                result['can_handle'] = detected_type in self._loaders
                result['recommended_loader'] = detected_type if result['can_handle'] else None
            else:
                result['detected_type'] = None
                result['can_handle'] = False
                result['recommended_loader'] = None
        except Exception as e:
            result['error'] = str(e)
            result['can_handle'] = False
        
        return result
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clear cache on exit."""
        self.clear_cache()


_global_factory: Optional[LoaderFactory] = None


def get_factory() -> LoaderFactory:
    global _global_factory
    if _global_factory is None:
        _global_factory = LoaderFactory()
    return _global_factory


def create_loader(source: str, loader_type: Optional[str] = None, **config_kwargs) -> BaseLoader:
    """Create a loader for the given source."""
    return get_factory().get_loader(source, loader_type, **config_kwargs)


def create_loader_for_file(file_path: str, **config_kwargs) -> BaseLoader:
    """Create a loader for a specific file."""
    return get_factory().get_loader_for_file(file_path, **config_kwargs)


def create_loader_for_content(content: str, content_type: str, **config_kwargs) -> BaseLoader:
    """Create a loader for content with a specific type."""
    return get_factory().get_loader_for_content(content, content_type, **config_kwargs)


def create_intelligent_loaders(sources: List[Union[str, Path]], **config_kwargs) -> List[BaseLoader]:
    """Create intelligent loaders for multiple sources."""
    return get_factory().create_intelligent_loaders(sources, **config_kwargs)


def can_handle_file(file_path: str) -> bool:
    """Check if the factory can handle a specific file."""
    return get_factory().can_handle(file_path)


def get_supported_extensions() -> List[str]:
    """Get all supported file extensions."""
    return get_factory().get_supported_extensions()


def get_supported_loaders() -> List[str]:
    """Get all supported loader types."""
    return get_factory().get_supported_loaders()


def load_document(source: str, **config_kwargs) -> List['Document']:
    loader = create_loader(source, **config_kwargs)
    return loader.load(source)


def load_documents_batch(
    sources: List[str], 
    **config_kwargs
) -> Dict[str, List['Document']]:
    """
    Load documents from multiple sources in batch.
    
    Args:
        sources: List of source paths or content strings
        **config_kwargs: Configuration parameters for loaders
        
    Returns:
        Dictionary mapping sources to their loaded documents
    """
    results = {}
    factory = get_factory()
    
    for source in sources:
        try:
            loader = factory.get_loader(source, **config_kwargs)
            documents = loader.load(source)
            results[source] = documents
        except Exception as e:
            factory._logger.error(f"Failed to load {source}: {e}")
            results[source] = []
    
    return results


def validate_source(source: str) -> Dict[str, Any]:
    """Validate a source and return information about it."""
    return get_factory().validate_source(source)


def get_loader_statistics() -> Dict[str, Any]:
    """Get statistics about registered loaders."""
    return get_factory().get_loader_statistics()


def list_available_loaders() -> List[Dict[str, Any]]:
    """List all available loaders with their information."""
    return get_factory().list_loaders()


def check_extension_conflicts() -> Dict[str, List[str]]:
    """Check for extension conflicts between loaders."""
    return get_factory()._find_extension_conflicts()


def create_factory() -> LoaderFactory:
    """Create a new factory instance."""
    return LoaderFactory()


def with_factory(func):
    """Decorator to provide a factory instance to a function."""
    def wrapper(*args, **kwargs):
        with create_factory() as factory:
            return func(factory, *args, **kwargs)
    return wrapper
