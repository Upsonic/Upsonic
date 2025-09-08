from __future__ import annotations

from .base import DocumentLoader, LoadingResult, LoadingProgress
from .config import (
    LoaderConfig, TextLoaderConfig, CSVLoaderConfig, PDFLoaderConfig,
    DOCXLoaderConfig, JSONLoaderConfig, XMLLoaderConfig, YAMLLoaderConfig,
    MarkdownLoaderConfig, HTMLLoaderConfig, LoaderConfigFactory, simple_config, advanced_config
)

from .text import TextLoader
from .csv import CSVLoader
from .pdf import PDFLoader
from .docx import DOCXLoader
from .json import JSONLoader
from .xml import XMLLoader
from .yaml import YAMLLoader
from .markdown import MarkdownLoader
from .html import HTMLLoader

from .factory import (
    LoaderFactory, get_factory, create_loader, create_loader_for_file,
    create_loader_for_content, can_handle_file, get_supported_extensions,
    get_supported_loaders, load_document, load_documents_batch,
    create_intelligent_loaders
)


__all__ = [
    'DocumentLoader', 'LoadingResult', 'LoadingProgress',
    
    'LoaderConfig', 'TextLoaderConfig', 'CSVLoaderConfig', 'PDFLoaderConfig',
    'DOCXLoaderConfig', 'JSONLoaderConfig', 'XMLLoaderConfig', 'YAMLLoaderConfig',
    'MarkdownLoaderConfig', 'HTMLLoaderConfig', 'LoaderConfigFactory', 'simple_config', 'advanced_config',
    
    'TextLoader', 'CSVLoader', 'PDFLoader', 'DOCXLoader',
    'JSONLoader', 'XMLLoader', 'YAMLLoader', 'MarkdownLoader', 'HTMLLoader',
    
    'LoaderFactory', 'get_factory', 'create_loader', 'create_loader_for_file',
    'create_loader_for_content', 'can_handle_file', 'get_supported_extensions',
    'get_supported_loaders', 'load_document', 'load_documents_batch',
    'create_intelligent_loaders',
]
