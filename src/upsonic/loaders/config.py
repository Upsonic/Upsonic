from __future__ import annotations
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class LoaderConfig(BaseModel, ABC):
    """Base configuration class for all document loaders."""
    
    encoding: Optional[str] = Field(default=None, description="File encoding (auto-detected if None)")
    error_handling: Literal["ignore", "warn", "raise"] = Field(default="warn", description="How to handle errors")
    include_metadata: bool = Field(default=True, description="Whether to include file metadata")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata to include")
    
    max_file_size: Optional[int] = Field(default=None, description="Maximum file size in bytes")
    skip_empty_content: bool = Field(default=True, description="Skip documents with empty content")
    
    class Config:
        arbitrary_types_allowed = True


class TextLoaderConfig(LoaderConfig):
    """Configuration for text file loading."""
    pass


class CSVLoaderConfig(LoaderConfig):
    """Configuration for CSV file loading."""
    
    content_synthesis_mode: Literal["concatenated", "json"] = Field(
        default="concatenated", 
        description="How to create document content from rows"
    )
    include_columns: Optional[List[str]] = Field(
        default=None, 
        description="Only include these columns"
    )
    exclude_columns: Optional[List[str]] = Field(
        default=None, 
        description="Exclude these columns"
    )
    delimiter: str = Field(default=",", description="CSV delimiter")
    quotechar: str = Field(default='"', description="CSV quote character")
    has_header: bool = Field(default=True, description="Whether CSV has header row")
    row_as_document: bool = Field(default=True, description="Each row becomes a document")


class PDFLoaderConfig(LoaderConfig):
    """Configuration for PDF file loading."""
    
    load_strategy: Literal["one_document_per_page", "one_document_for_the_whole_file"] = Field(
        default="one_document_per_page",
        description="How to split PDF into documents"
    )
    use_ocr: bool = Field(default=False, description="Enable OCR for scanned PDFs")
    ocr_text_threshold: int = Field(default=50, description="Minimum text chars before triggering OCR")
    force_ocr: bool = Field(default=False, description="Force OCR even when text is available")
    ocr_dpi: int = Field(default=200, description="DPI for OCR image conversion (higher = better quality, slower)")
    ocr_language: str = Field(default="eng", description="OCR language code")
    extract_images: bool = Field(default=False, description="Extract and process images")
    preserve_formatting: bool = Field(default=False, description="Attempt to preserve text formatting")


class DOCXLoaderConfig(LoaderConfig):
    """Configuration for DOCX file loading."""
    
    include_tables: bool = Field(default=True, description="Include table content")
    include_headers: bool = Field(default=True, description="Include header content")
    include_footers: bool = Field(default=True, description="Include footer content")
    table_format: Literal["text", "markdown", "html"] = Field(
        default="text", 
        description="How to format tables"
    )


class JSONLoaderConfig(LoaderConfig):
    """Configuration for JSON file loading."""
    
    jq_schema: str = Field(default=".", description="JQ-style query for content extraction")
    is_jsonl: bool = Field(default=False, description="Treat as JSON Lines format")
    content_key: Optional[str] = Field(default=None, description="Key to use as content")
    flatten_metadata: bool = Field(default=False, description="Flatten nested metadata")
    json_indent: int = Field(default=2, description="JSON formatting indent")
    array_handling: Literal["separate_documents", "single_document", "flatten"] = Field(
        default="separate_documents",
        description="How to handle JSON arrays"
    )
    
    validation_level: Literal["none", "basic", "strict", "schema"] = Field(
        default="basic", 
        description="Level of JSON validation to perform"
    )
    schema_path: Optional[str] = Field(default=None, description="Path to JSON schema file for validation")
    custom_validators: Optional[List[Any]] = Field(default=None, description="List of custom validation functions")
    transformers: Optional[List[Any]] = Field(default=None, description="List of transformation functions to apply")
    filters: Optional[List[Any]] = Field(default=None, description="List of filter functions to apply")
    max_memory_mb: int = Field(default=1024, description="Maximum memory usage in MB")
    enable_streaming: bool = Field(default=True, description="Enable streaming processing for large files")
    enable_compression_detection: bool = Field(default=True, description="Auto-detect compression")
    enable_url_support: bool = Field(default=True, description="Enable loading from URLs")
    chunk_size: int = Field(default=1000, description="Number of objects to process in each chunk")


class XMLLoaderConfig(LoaderConfig):
    """Configuration for XML file loading."""
    
    split_by_xpath: Optional[str] = Field(default=None, description="XPath to split documents")
    content_synthesis_mode: Literal["smart_text", "xml_snippet"] = Field(
        default="smart_text",
        description="How to extract content"
    )
    strip_namespaces: bool = Field(default=True, description="Remove XML namespaces")
    include_attributes: bool = Field(default=True, description="Include XML attributes in metadata")


class YAMLLoaderConfig(LoaderConfig):
    """Configuration for YAML file loading."""
    
    content_synthesis_mode: Literal["canonical_yaml", "json"] = Field(
        default="canonical_yaml",
        description="How to serialize content"
    )
    flatten_metadata: bool = Field(default=True, description="Flatten nested metadata")
    yaml_indent: int = Field(default=2, description="YAML formatting indent")
    handle_multiple_docs: bool = Field(default=True, description="Handle multi-document YAML")


class MarkdownLoaderConfig(LoaderConfig):
    """Configuration for Markdown file loading."""
    
    parse_front_matter: bool = Field(default=True, description="Parse YAML front matter")
    include_code_blocks: bool = Field(default=True, description="Include code block content")
    code_block_language_metadata: bool = Field(default=True, description="Add code languages to metadata")
    table_format: Literal["text", "preserve_markdown", "html"] = Field(
        default="text",
        description="How to format tables"
    )
    heading_metadata: bool = Field(default=True, description="Extract headings as metadata")


class HTMLLoaderConfig(LoaderConfig):
    """Configuration for HTML file loading."""
    
    extract_text: bool = Field(default=True, description="Extract text content from HTML")
    preserve_structure: bool = Field(default=True, description="Preserve document structure in output")
    include_links: bool = Field(default=True, description="Include links in extracted content")
    include_images: bool = Field(default=False, description="Include image information")
    remove_scripts: bool = Field(default=True, description="Remove script tags")
    remove_styles: bool = Field(default=True, description="Remove style tags")
    extract_metadata: bool = Field(default=True, description="Extract metadata from HTML head")
    clean_whitespace: bool = Field(default=True, description="Clean up whitespace in output")
    
    extract_headers: bool = Field(default=True, description="Extract heading elements")
    extract_paragraphs: bool = Field(default=True, description="Extract paragraph content")
    extract_lists: bool = Field(default=True, description="Extract list content")
    extract_tables: bool = Field(default=True, description="Extract table content")
    
    table_format: Literal["text", "markdown", "html"] = Field(
        default="text",
        description="How to format extracted tables"
    )
    
    user_agent: str = Field(default="Upsonic HTML Loader 1.0", description="User agent for web requests")


class LoaderConfigFactory:
    """Factory for creating loader configurations."""
    
    _config_map: Dict[str, type] = {
        'text': TextLoaderConfig,
        'csv': CSVLoaderConfig,
        'pdf': PDFLoaderConfig,
        'docx': DOCXLoaderConfig,
        'json': JSONLoaderConfig,
        'jsonl': JSONLoaderConfig,
        'xml': XMLLoaderConfig,
        'yaml': YAMLLoaderConfig,
        'yml': YAMLLoaderConfig,
        'markdown': MarkdownLoaderConfig,
        'md': MarkdownLoaderConfig,
        'html': HTMLLoaderConfig,
        'htm': HTMLLoaderConfig,
    }
    
    @classmethod
    def create_config(
        self, 
        loader_type: str, 
        **kwargs
    ) -> LoaderConfig:
        """Create a configuration for the specified loader type."""
        config_class = self._config_map.get(loader_type.lower())
        if not config_class:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        return config_class(**kwargs)
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported loader types."""
        return list(cls._config_map.keys())


def simple_config(loader_type: str) -> LoaderConfig:
    """Create a simple configuration with defaults."""
    return LoaderConfigFactory.create_config(loader_type)


def advanced_config(loader_type: str, **kwargs) -> LoaderConfig:
    """Create an advanced configuration with custom settings."""
    return LoaderConfigFactory.create_config(loader_type, **kwargs)
