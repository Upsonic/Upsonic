from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseLoader
    from .config import LoaderConfig, LoaderConfigFactory

if TYPE_CHECKING:
    from .config import (
        TextLoaderConfig, CSVLoaderConfig, PdfLoaderConfig, PyMuPDFLoaderConfig, PdfPlumberLoaderConfig,
        DOCXLoaderConfig, JSONLoaderConfig, XMLLoaderConfig, YAMLLoaderConfig,
        MarkdownLoaderConfig, HTMLLoaderConfig, DoclingLoaderConfig, simple_config, advanced_config
    )
    from .text import TextLoader
    from .csv import CSVLoader
    from .pdf import PdfLoader
    from .pymupdf import PyMuPDFLoader
    from .pdfplumber import PdfPlumberLoader
    from .docx import DOCXLoader
    from .json import JSONLoader
    from .xml import XMLLoader
    from .yaml import YAMLLoader
    from .markdown import MarkdownLoader
    from .html import HTMLLoader
    from .docling import DoclingLoader
    from .factory import (
        LoaderFactory, get_factory, create_loader, create_loader_for_file,
        create_loader_for_content, can_handle_file, get_supported_extensions,
        get_supported_loaders, load_document, load_documents_batch,
        create_intelligent_loaders, validate_source, get_loader_statistics,
        list_available_loaders, check_extension_conflicts, create_factory,
        with_factory
    )

def _get_loader_classes():
    """Lazy import of loader classes."""
    from .text import TextLoader
    from .csv import CSVLoader
    from .pdf import PdfLoader
    from .pymupdf import PyMuPDFLoader
    from .pdfplumber import PdfPlumberLoader
    from .docx import DOCXLoader
    from .json import JSONLoader
    from .xml import XMLLoader
    from .yaml import YAMLLoader
    from .markdown import MarkdownLoader
    from .html import HTMLLoader
    
    loaders = {
        'TextLoader': TextLoader,
        'CSVLoader': CSVLoader,
        'PdfLoader': PdfLoader,
        'PyMuPDFLoader': PyMuPDFLoader,
        'PdfPlumberLoader': PdfPlumberLoader,
        'DOCXLoader': DOCXLoader,
        'JSONLoader': JSONLoader,
        'XMLLoader': XMLLoader,
        'YAMLLoader': YAMLLoader,
        'MarkdownLoader': MarkdownLoader,
        'HTMLLoader': HTMLLoader,
    }
    
    # Try to import DoclingLoader (optional dependency)
    try:
        from .docling import DoclingLoader
        loaders['DoclingLoader'] = DoclingLoader
    except ImportError:
        pass  # Docling not installed
    
    return loaders

def _get_config_classes():
    """Lazy import of config classes."""
    from .config import (
        TextLoaderConfig, CSVLoaderConfig, PdfLoaderConfig, PyMuPDFLoaderConfig, PdfPlumberLoaderConfig,
        DOCXLoaderConfig, JSONLoaderConfig, XMLLoaderConfig, YAMLLoaderConfig,
        MarkdownLoaderConfig, HTMLLoaderConfig, DoclingLoaderConfig, simple_config, advanced_config
    )
    
    return {
        'TextLoaderConfig': TextLoaderConfig,
        'CSVLoaderConfig': CSVLoaderConfig,
        'PdfLoaderConfig': PdfLoaderConfig,
        'PyMuPDFLoaderConfig': PyMuPDFLoaderConfig,
        'PdfPlumberLoaderConfig': PdfPlumberLoaderConfig,
        'DOCXLoaderConfig': DOCXLoaderConfig,
        'JSONLoaderConfig': JSONLoaderConfig,
        'XMLLoaderConfig': XMLLoaderConfig,
        'YAMLLoaderConfig': YAMLLoaderConfig,
        'MarkdownLoaderConfig': MarkdownLoaderConfig,
        'HTMLLoaderConfig': HTMLLoaderConfig,
        'DoclingLoaderConfig': DoclingLoaderConfig,
        'simple_config': simple_config,
        'advanced_config': advanced_config,
    }

def _get_factory_functions():
    """Lazy import of factory functions."""
    from .factory import (
        LoaderFactory, get_factory, create_loader, create_loader_for_file,
        create_loader_for_content, can_handle_file, get_supported_extensions,
        get_supported_loaders, load_document, load_documents_batch,
        create_intelligent_loaders, validate_source, get_loader_statistics,
        list_available_loaders, check_extension_conflicts, create_factory,
        with_factory
    )
    
    return {
        'LoaderFactory': LoaderFactory,
        'get_factory': get_factory,
        'create_loader': create_loader,
        'create_loader_for_file': create_loader_for_file,
        'create_loader_for_content': create_loader_for_content,
        'can_handle_file': can_handle_file,
        'get_supported_extensions': get_supported_extensions,
        'get_supported_loaders': get_supported_loaders,
        'load_document': load_document,
        'load_documents_batch': load_documents_batch,
        'create_intelligent_loaders': create_intelligent_loaders,
        'validate_source': validate_source,
        'get_loader_statistics': get_loader_statistics,
        'list_available_loaders': list_available_loaders,
        'check_extension_conflicts': check_extension_conflicts,
        'create_factory': create_factory,
        'with_factory': with_factory,
    }

def _get_base_classes():
    """Lazy import of base loader classes."""
    from .base import BaseLoader
    from .config import LoaderConfig, LoaderConfigFactory
    
    return {
        'BaseLoader': BaseLoader,
        'LoaderConfig': LoaderConfig,
        'LoaderConfigFactory': LoaderConfigFactory,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    # Base classes
    base_classes = _get_base_classes()
    if name in base_classes:
        return base_classes[name]
    
    # Loader classes
    loader_classes = _get_loader_classes()
    if name in loader_classes:
        return loader_classes[name]
    
    # Config classes
    config_classes = _get_config_classes()
    if name in config_classes:
        return config_classes[name]
    
    # Factory functions
    factory_functions = _get_factory_functions()
    if name in factory_functions:
        return factory_functions[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )


__all__ = [
    'BaseLoader',

    'LoaderConfig', 'TextLoaderConfig', 'CSVLoaderConfig', 'PdfLoaderConfig', 'PyMuPDFLoaderConfig', 'PdfPlumberLoaderConfig',
    'DOCXLoaderConfig', 'JSONLoaderConfig', 'XMLLoaderConfig', 'YAMLLoaderConfig',
    'MarkdownLoaderConfig', 'HTMLLoaderConfig', 'DoclingLoaderConfig', 'LoaderConfigFactory', 'simple_config', 'advanced_config',
    
    'TextLoader', 'CSVLoader', 'PdfLoader', 'PyMuPDFLoader', 'PdfPlumberLoader', 'DOCXLoader',
    'JSONLoader', 'XMLLoader', 'YAMLLoader', 'MarkdownLoader', 'HTMLLoader', 'DoclingLoader',
    
    'LoaderFactory', 'get_factory', 'create_loader', 'create_loader_for_file',
    'create_loader_for_content', 'can_handle_file', 'get_supported_extensions',
    'get_supported_loaders', 'load_document', 'load_documents_batch',
    'create_intelligent_loaders', 'validate_source', 'get_loader_statistics',
    'list_available_loaders', 'check_extension_conflicts', 'create_factory',
    'with_factory',
]
