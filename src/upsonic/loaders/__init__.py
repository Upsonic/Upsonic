from __future__ import annotations

from .base import DocumentLoader, LoadingResult, LoadingProgress
from .config import (
    LoaderConfig, TextLoaderConfig, CSVLoaderConfig, PDFLoaderConfig,
    DOCXLoaderConfig, JSONLoaderConfig, XMLLoaderConfig, YAMLLoaderConfig,
    MarkdownLoaderConfig, HTMLLoaderConfig, LoaderConfigFactory, simple_config, advanced_config
)
from .factory import (
    LoaderFactory, AutoLoader, LoaderRegistry,
    create_simple_loader, create_configured_loader,
    load_file, load_files, load_files_async
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

def _initialize_registry():
    registry = LoaderFactory
    
    registry.register_loader("text", TextLoader, [".txt", ".text"], ["txt"])
    registry.register_loader("csv", CSVLoader, [".csv"], ["csv"])
    registry.register_loader("pdf", PDFLoader, [".pdf"], ["pdf"])
    registry.register_loader("docx", DOCXLoader, [".docx"], ["docx", "word"])
    registry.register_loader("json", JSONLoader, [".json", ".jsonl"], ["json", "jsonl"])
    registry.register_loader("xml", XMLLoader, [".xml"], ["xml"])
    registry.register_loader("yaml", YAMLLoader, [".yaml", ".yml"], ["yaml", "yml"])
    registry.register_loader("markdown", MarkdownLoader, [".md", ".markdown"], ["md", "markdown"])
    registry.register_loader("html", HTMLLoader, [".html", ".htm", ".xhtml"], ["html", "htm"])

_initialize_registry()

def create_knowledge_base_loader(sources, **config_overrides):
    """
    Create a loader suitable for KnowledgeBase with sensible defaults.
    
    This function creates an AutoLoader that can handle multiple file types
    with configurations optimized for knowledge base ingestion.
    
    Args:
        sources: List of source files or single source file
        **config_overrides: Configuration overrides for specific loader types
        
    Returns:
        AutoLoader configured for knowledge base usage
    """
    if isinstance(sources, str):
        sources = [sources]
    
    default_configs = {
        "pdf": {
            "load_strategy": "one_document_per_page",
            "use_ocr": False,
            "error_handling": "warn"
        },
        "csv": {
            "content_synthesis_mode": "concatenated",
            "row_as_document": True,
            "error_handling": "warn"
        },
        "text": {
            "error_handling": "warn"
        },
        "docx": {
            "include_tables": True,
            "error_handling": "warn"
        },
        "json": {
            "jq_schema": ".",
            "flatten_metadata": True,
            "error_handling": "warn",
            "validation_level": "basic",
            "enable_streaming": True,
            "max_memory_mb": 1024
        },
        "xml": {
            "content_synthesis_mode": "smart_text",
            "strip_namespaces": True,
            "include_attributes": True,
            "error_handling": "warn"
        },
        "yaml": {
            "content_synthesis_mode": "canonical_yaml",
            "flatten_metadata": True,
            "error_handling": "warn"
        },
        "markdown": {
            "parse_front_matter": True,
            "include_code_blocks": True,
            "error_handling": "warn"
        },
        "html": {
            "extract_text": True,
            "preserve_structure": True,
            "include_links": True,
            "remove_scripts": True,
            "remove_styles": True,
            "extract_metadata": True,
            "clean_whitespace": True,
            "error_handling": "warn"
        }
    }
    
    for loader_type, overrides in config_overrides.items():
        if loader_type in default_configs:
            default_configs[loader_type].update(overrides)
        else:
            default_configs[loader_type] = overrides
    
    loader_configs = {}
    for loader_type, config_dict in default_configs.items():
        try:
            loader_configs[loader_type] = LoaderConfigFactory.create_config(loader_type, **config_dict)
        except ValueError:
            continue
    
    return AutoLoader(sources, None, loader_configs)

__all__ = [
    'DocumentLoader', 'LoadingResult', 'LoadingProgress',
    
    'LoaderConfig', 'TextLoaderConfig', 'CSVLoaderConfig', 'PDFLoaderConfig',
    'DOCXLoaderConfig', 'JSONLoaderConfig', 'XMLLoaderConfig', 'YAMLLoaderConfig',
    'MarkdownLoaderConfig', 'HTMLLoaderConfig', 'LoaderConfigFactory', 'simple_config', 'advanced_config',
    
    'LoaderFactory', 'AutoLoader', 'LoaderRegistry',
    
    'TextLoader', 'CSVLoader', 'PDFLoader', 'DOCXLoader',
    'JSONLoader', 'XMLLoader', 'YAMLLoader', 'MarkdownLoader', 'HTMLLoader',
    
    'create_simple_loader', 'create_configured_loader',
    'load_file', 'load_files', 'load_files_async',
    'create_knowledge_base_loader'
]
