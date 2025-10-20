from __future__ import annotations
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, ConfigDict
from abc import ABC, abstractmethod


class LoaderConfig(BaseModel, ABC):
    """Base configuration class for all document loaders."""

    encoding: Optional[str] = Field(
        default=None, description="File encoding (auto-detected if None)"
    )
    error_handling: Literal["ignore", "warn", "raise"] = Field(
        default="warn", description="How to handle loading errors"
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include file metadata"
    )
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata to include"
    )
    max_file_size: Optional[int] = Field(
        default=None, description="Maximum file size in bytes"
    )
    skip_empty_content: bool = Field(
        default=True, description="Skip documents with empty content"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TextLoaderConfig(LoaderConfig):
    """
    An enhanced configuration for loading and processing plain text files.

    Provides options to structure the content by file, paragraph, or line,
    and to perform basic cleaning operations.
    """


    strip_whitespace: bool = Field(
        default=True,
        description="If True, removes leading/trailing whitespace from each chunk.",
    )
    min_chunk_length: int = Field(
        default=1,
        description="The minimum character length for a chunk to be kept after cleaning.",
        ge=0
    )


class CSVLoaderConfig(LoaderConfig):
    """Configuration for CSV file loading."""

    content_synthesis_mode: Literal["concatenated", "json"] = Field(
        default="concatenated",
        description="How to create document content from rows",
    )
    split_mode: Literal["single_document", "per_row", "per_chunk"] = Field(
        default="single_document",
        description="How to split CSV into documents: 'single_document' (all rows in one), 'per_row' (each row as document), 'per_chunk' (groups of rows)"
    )
    rows_per_chunk: int = Field(
        default=100,
        description="Number of rows per document when split_mode='per_chunk'",
        gt=0
    )
    include_columns: Optional[List[str]] = Field(
        default=None, description="Only include these columns"
    )
    exclude_columns: Optional[List[str]] = Field(
        default=None, description="Exclude these columns"
    )
    delimiter: str = Field(default=",", description="CSV delimiter")
    quotechar: str = Field(default='"', description="CSV quote character")
    has_header: bool = Field(default=True, description="Whether CSV has a header row")


class PdfLoaderConfig(LoaderConfig):
    """
    An advanced configuration model for loading and processing PDF documents.

    This configuration provides granular control over text extraction, OCR,
    content structuring, and preprocessing to handle a wide variety of
    PDF formats and quality levels.
    """

    extraction_mode: Literal["hybrid", "text_only", "ocr_only"] = Field(
        default="hybrid",
        description=(
            "The core strategy for content extraction. "
            "'hybrid': Extracts digital text and runs OCR on embedded images. "
            "'text_only': Fastest mode, extracts only digital text. "
            "'ocr_only': For scanned documents, runs OCR on the entire page."
        ),
    )

    start_page: Optional[int] = Field(
        default=None,
        description="The first page number to process (1-indexed). If None, starts from the beginning.",
        ge=1
    )
    end_page: Optional[int] = Field(
        default=None,
        description="The last page number to process (inclusive). If None, processes to the end.",
        ge=1
    )

    clean_page_numbers: bool = Field(
        default=True,
        description="If True, intelligently identifies and removes page numbers from page headers/footers.",
    )
    page_num_start_format: Optional[str] = Field(
        default=None,
        description="A Python f-string to prepend to each page's content if page numbers are cleaned. "
                    "Example: '<start page {page_nr}>'. If None, nothing is prepended."
    )
    page_num_end_format: Optional[str] = Field(
        default=None,
        description="A Python f-string to append to each page's content if page numbers are cleaned. "
                    "Example: '<end page {page_nr}>'. If None, nothing is appended."
    )
    extra_whitespace_removal: bool = Field(
        default=True,
        description="If True, normalizes whitespace by collapsing multiple newlines and spaces, cleaning up layout artifacts.",
    )

    pdf_password: Optional[str] = Field(
        default=None,
        description="Password to use for decrypting protected PDF files.",
        min_length=1
    )


class PyMuPDFLoaderConfig(LoaderConfig):
    """
    Advanced configuration for PyMuPDF-based PDF document loading.

    This configuration leverages PyMuPDF's superior performance and features
    for text extraction, OCR, image handling, and content structuring.
    """

    extraction_mode: Literal["hybrid", "text_only", "ocr_only"] = Field(
        default="hybrid",
        description=(
            "The core strategy for content extraction. "
            "'hybrid': Extracts digital text and runs OCR on embedded images. "
            "'text_only': Fastest mode, extracts only digital text. "
            "'ocr_only': For scanned documents, runs OCR on the entire page."
        ),
    )

    start_page: Optional[int] = Field(
        default=None,
        description="The first page number to process (1-indexed). If None, starts from the beginning.",
        ge=1
    )
    end_page: Optional[int] = Field(
        default=None,
        description="The last page number to process (inclusive). If None, processes to the end.",
        ge=1
    )

    clean_page_numbers: bool = Field(
        default=True,
        description="If True, intelligently identifies and removes page numbers from page headers/footers.",
    )
    page_num_start_format: Optional[str] = Field(
        default=None,
        description="A Python f-string to prepend to each page's content if page numbers are cleaned. "
                    "Example: '<start page {page_nr}>'. If None, nothing is prepended."
    )
    page_num_end_format: Optional[str] = Field(
        default=None,
        description="A Python f-string to append to each page's content if page numbers are cleaned. "
                    "Example: '<end page {page_nr}>'. If None, nothing is appended."
    )
    extra_whitespace_removal: bool = Field(
        default=True,
        description="If True, normalizes whitespace by collapsing multiple newlines and spaces, cleaning up layout artifacts.",
    )

    pdf_password: Optional[str] = Field(
        default=None,
        description="Password to use for decrypting protected PDF files.",
        min_length=1
    )

    # PyMuPDF-specific configurations
    text_extraction_method: Literal["text", "dict", "html", "xml"] = Field(
        default="text",
        description=(
            "Method for text extraction from pages. "
            "'text': Plain text extraction. "
            "'dict': Structured text with positioning. "
            "'html': HTML formatted text. "
            "'xml': XML formatted text."
        ),
    )

    include_images: bool = Field(
        default=False,
        description="If True, extracts and includes image information in metadata.",
    )

    image_dpi: int = Field(
        default=150,
        description="DPI for image rendering when performing OCR.",
        ge=72,
        le=600
    )

    preserve_layout: bool = Field(
        default=True,
        description="If True, preserves text layout and positioning information.",
    )

    extract_annotations: bool = Field(
        default=False,
        description="If True, extracts annotations and comments from the PDF.",
    )

    annotation_format: Literal["text", "json"] = Field(
        default="text",
        description="Format for extracted annotations.",
    )



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
    """
    Advanced configuration for loading and mapping structured JSON and JSONL files.
    """
    mode: Literal["single", "multi"] = Field(
        default="single",
        description="Processing mode: 'single' for one document per file, 'multi' to extract multiple documents from records within a file."
    )
    record_selector: Optional[str] = Field(
        default=None,
        description="A JQ query to select a list of records from the JSON object (e.g., '.articles[]'). Required for 'multi' mode."
    )
    content_mapper: str = Field(
        default=".",
        description="A JQ query to extract the content from a single record. The default '.' uses the entire record."
    )
    metadata_mapper: Optional[Dict[str, str]] = Field(
        default=None,
        description="A dictionary mapping metadata keys to JQ queries for extracting metadata from each record."
    )
    content_synthesis_mode: Literal["json", "text"] = Field(
        default="json",
        description="How to format the extracted content: 'json' for a compact JSON string, 'text' for the raw text value."
    )
    json_lines: bool = Field(
        default=False,
        description="Set to True if the file is in JSON Lines format (.jsonl), where each line is a separate JSON object."
    )


class XMLLoaderConfig(LoaderConfig):
    """
    An advanced configuration for parsing and structuring XML files.

    Provides powerful XPath-based controls for splitting a single XML file
    into multiple documents and for extracting specific content and metadata.
    """

    split_by_xpath: str = Field(
        default="//*[not(*)] | //item | //product | //book",
        description=(
            "An XPath expression that identifies the elements to be treated as individual documents. "
            "Defaults to a flexible pattern that matches common element types. If you want the whole file, use '/'."
        ),
    )
    content_xpath: Optional[str] = Field(
        default=None,
        description=(
            "An optional, relative XPath from a split element to select the specific content. "
            "Example: './description'. If None, the content of the entire split element is used."
        ),
    )

    content_synthesis_mode: Literal["smart_text", "xml_snippet"] = Field(
        default="smart_text",
        description=(
            "Defines the content for the Document. 'smart_text' extracts and concatenates all text. "
            "'xml_snippet' preserves the original XML structure as a string."
        ),
    )
    include_attributes: bool = Field(
        default=True,
        description="If True, automatically includes the attributes of the split element in the metadata.",
    )
    metadata_xpaths: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "A dictionary mapping metadata keys to XPath expressions to extract targeted metadata. "
            "Example: {'author': './/authorName', 'date': './@published'}."
        ),
    )
    strip_namespaces: bool = Field(
        default=True,
        description="If True, removes all XML namespaces from the document, which can simplify XPath expressions.",
    )
    recover_mode: bool = Field(
        default=False,
        description="If True, attempts to parse malformed or broken XML files instead of raising an error.",
    )


class YAMLLoaderConfig(LoaderConfig):
    """
    An advanced configuration for parsing and structuring YAML files.

    Provides a powerful jq-based query system for splitting a single YAML file
    into multiple documents and for extracting specific content and metadata.
    """

    split_by_jq_query: str = Field(
        default=".",
        description=(
            "A jq-style query to select objects to be treated as individual documents. "
            "Example: '.articles[]' to create a document for each item in the 'articles' list. "
            "Defaults to '.' to treat the whole YAML document as one."
        ),
    )
    handle_multiple_docs: bool = Field(
        default=True,
        description="If True, processes YAML files containing multiple documents separated by '---'.",
    )

    content_synthesis_mode: Literal["canonical_yaml", "json", "smart_text"] = Field(
        default="canonical_yaml",
        description=(
            "Defines the content for the Document. 'canonical_yaml' or 'json' serializes the data. "
            "'smart_text' recursively extracts and joins all string values into a single text block."
        ),
    )
    yaml_indent: int = Field(
        default=2, 
        description="The indentation level to use when `content_synthesis_mode` is 'canonical_yaml'.",
        ge=1
    )
    json_indent: Optional[int] = Field(
        default=2,
        description="The indentation level for JSON output. Set to None for compact JSON.",
    )

    flatten_metadata: bool = Field(
        default=True,
        description=(
            "If True, flattens the nested structure of the selected YAML object into the metadata. "
            "E.g., {'user': {'name': 'John'}} becomes {'user.name': 'John'}."
        ),
    )
    metadata_jq_queries: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "A dictionary mapping metadata keys to jq queries to extract targeted metadata. "
            "Example: {'author': '.info.authorName', 'title': '.title'}."
        ),
    )


class MarkdownLoaderConfig(LoaderConfig):
    """Configuration for Markdown file loading."""
    parse_front_matter: bool = Field(default=True, description="Parse YAML front matter from the top of the file.")
    include_code_blocks: bool = Field(default=True, description="Include code block content in the document.")
    code_block_language_metadata: bool = Field(default=True, description="Add code block language as metadata.")
    heading_metadata: bool = Field(default=True, description="Extract headings and add them to the metadata.")
    split_by_heading: Optional[Literal["h1", "h2", "h3"]] = Field(
        default=None, description="If set, splits the file into multiple documents based on the specified heading level."
    )


class HTMLLoaderConfig(LoaderConfig):
    """Configuration for HTML file and URL loading."""

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


class PdfPlumberLoaderConfig(LoaderConfig):
    """
    Advanced configuration for pdfplumber-based PDF document loading.
    
    pdfplumber provides superior table extraction, layout preservation, and 
    character-level text analysis. It excels at structured document processing
    and complex layout handling for AI agent frameworks requiring high-quality
    extraction from PDFs with tables, forms, and complex formatting.
    """

    extraction_mode: Literal["hybrid", "text_only", "ocr_only"] = Field(
        default="hybrid",
        description=(
            "The core strategy for content extraction. "
            "'hybrid': Extracts digital text and runs OCR on embedded images. "
            "'text_only': Fastest mode, extracts only digital text using pdfplumber. "
            "'ocr_only': For scanned documents, runs OCR on the entire page."
        ),
    )

    start_page: Optional[int] = Field(
        default=None,
        description="The first page number to process (1-indexed). If None, starts from the beginning.",
        ge=1
    )
    end_page: Optional[int] = Field(
        default=None,
        description="The last page number to process (inclusive). If None, processes to the end.",
        ge=1
    )

    clean_page_numbers: bool = Field(
        default=True,
        description="If True, intelligently identifies and removes page numbers from page headers/footers.",
    )
    page_num_start_format: Optional[str] = Field(
        default=None,
        description="A Python f-string to prepend to each page's content if page numbers are cleaned. "
                    "Example: '<start page {page_nr}>'. If None, nothing is prepended."
    )
    page_num_end_format: Optional[str] = Field(
        default=None,
        description="A Python f-string to append to each page's content if page numbers are cleaned. "
                    "Example: '<end page {page_nr}>'. If None, nothing is appended."
    )
    extra_whitespace_removal: bool = Field(
        default=True,
        description="If True, normalizes whitespace by collapsing multiple newlines and spaces, cleaning up layout artifacts.",
    )

    pdf_password: Optional[str] = Field(
        default=None,
        description="Password to use for decrypting protected PDF files.",
        min_length=1
    )

    # pdfplumber-specific configurations
    extract_tables: bool = Field(
        default=True,
        description="If True, extracts tables and includes them in the content. pdfplumber excels at table extraction.",
    )
    
    table_format: Literal["text", "markdown", "csv", "grid"] = Field(
        default="markdown",
        description=(
            "Format for extracted tables. "
            "'markdown': Tables as markdown format (best for AI agents). "
            "'text': Plain text representation. "
            "'csv': CSV format. "
            "'grid': Grid-style ASCII table."
        ),
    )
    
    table_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
        },
        description=(
            "Advanced table detection settings for pdfplumber. "
            "Strategies: 'lines' (uses ruling lines), 'lines_strict', 'text' (infers from text positioning). "
            "Tolerances control how strictly boundaries are detected."
        ),
    )

    extract_images: bool = Field(
        default=False,
        description="If True, extracts image information and includes metadata about images found on each page.",
    )

    layout_mode: Literal["default", "layout", "simple"] = Field(
        default="layout",
        description=(
            "Text extraction layout mode. "
            "'layout': Preserves original layout and spacing (best for structured docs). "
            "'default': Standard extraction. "
            "'simple': Simplified extraction without layout preservation."
        ),
    )

    use_text_flow: bool = Field(
        default=True,
        description="If True, uses pdfplumber's text flow analysis to maintain reading order in complex layouts.",
    )

    char_margin: float = Field(
        default=3.0,
        description="The minimum distance between characters for them to be considered separate words.",
        ge=0.0
    )

    line_margin: float = Field(
        default=0.5,
        description="The minimum distance between lines for them to be considered separate.",
        ge=0.0
    )

    word_margin: float = Field(
        default=0.1,
        description="The minimum distance between words.",
        ge=0.0
    )

    extract_page_dimensions: bool = Field(
        default=False,
        description="If True, includes page dimensions (width, height) in metadata.",
    )

    crop_box: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description="Optional crop box (x0, y0, x1, y1) to extract only a specific region of each page.",
    )

    extract_annotations: bool = Field(
        default=False,
        description="If True, extracts annotations and hyperlinks from the PDF.",
    )

    keep_blank_chars: bool = Field(
        default=False,
        description="If True, preserves blank characters in extracted text for layout fidelity.",
    )


class DoclingLoaderConfig(LoaderConfig):
    """
    Advanced configuration for Docling-based document loading.
    
    Docling provides enterprise-grade document processing with support for
    PDF, DOCX, XLSX, PPTX, HTML, Markdown, AsciiDoc, CSV, and various image formats.
    It leverages advanced ML models for layout understanding and content extraction.
    """

    extraction_mode: Literal["markdown", "chunks"] = Field(
        default="chunks",
        description=(
            "Content extraction strategy: "
            "'markdown' exports entire document as formatted markdown. "
            "'chunks' intelligently segments document into semantic chunks for RAG."
        ),
    )

    chunker_type: Literal["hybrid", "hierarchical"] = Field(
        default="hybrid",
        description=(
            "Chunking algorithm when extraction_mode='chunks': "
            "'hybrid' combines semantic and structural chunking for optimal retrieval. "
            "'hierarchical' preserves document hierarchy (sections, subsections)."
        ),
    )

    allowed_formats: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict input formats. Options: 'pdf', 'docx', 'xlsx', 'pptx', 'html', "
            "'md', 'asciidoc', 'csv', 'image'. If None, all formats are allowed."
        ),
    )

    # Markdown export options (used when extraction_mode='markdown')
    markdown_image_placeholder: str = Field(
        default="",
        description="Placeholder text for images in markdown export. Empty string removes images.",
    )

    # OCR Configuration (for PDFs and images)
    ocr_enabled: bool = Field(
        default=True,
        description="Enable OCR for scanned documents and images. Uses hybrid mode (smart OCR + text extraction).",
    )

    ocr_force_full_page: bool = Field(
        default=False,
        description="Force full-page OCR instead of hybrid mode. Slower but better for scanned documents.",
    )

    ocr_backend: Literal["rapidocr", "tesseract"] = Field(
        default="rapidocr",
        description="OCR engine to use. 'rapidocr' is fast and accurate, 'tesseract' supports more languages.",
    )

    ocr_lang: List[str] = Field(
        default_factory=lambda: ["english"],
        description="OCR languages. For RapidOCR: ['english', 'chinese']. For Tesseract: ['eng', 'fra', etc.] or ['auto'] for auto-detection.",
    )

    ocr_backend_engine: Literal["onnxruntime", "openvino", "paddle", "torch"] = Field(
        default="onnxruntime",
        description="Backend engine for RapidOCR. 'onnxruntime' recommended for best compatibility.",
    )

    ocr_text_score: float = Field(
        default=0.5,
        description="Minimum confidence score for OCR text (0.0-1.0). Lower values include more uncertain text.",
        ge=0.0,
        le=1.0
    )

    # Table Structure Detection
    enable_table_structure: bool = Field(
        default=True,
        description="Enable table structure detection and parsing.",
    )

    table_structure_cell_matching: bool = Field(
        default=True,
        description="Enable cell-level matching in tables for better structure preservation.",
    )

    # Performance and resource management
    max_pages: Optional[int] = Field(
        default=None,
        description="Maximum number of pages to process per document. None means no limit.",
        ge=1
    )

    page_range: Optional[tuple[int, int]] = Field(
        default=None,
        description="Specific page range to process (start, end) - 1-indexed, inclusive.",
    )

    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing for batch operations.",
    )

    batch_size: int = Field(
        default=10,
        description="Number of documents to process in parallel during batch operations.",
        ge=1,
        le=100
    )

    # Metadata extraction
    extract_document_metadata: bool = Field(
        default=True,
        description="Extract document properties (title, author, creation date, etc.).",
    )

    confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence score for extracted chunks (0.0-1.0).",
        ge=0.0,
        le=1.0
    )

    # URL handling
    support_urls: bool = Field(
        default=True,
        description="Allow loading documents from HTTP/HTTPS URLs.",
    )

    url_timeout: int = Field(
        default=30,
        description="Timeout in seconds for URL downloads.",
        ge=1,
        le=300
    )


class LoaderConfigFactory:
    """Factory for creating loader configurations."""
    
    _config_map: Dict[str, type] = {
        'text': TextLoaderConfig,
        'csv': CSVLoaderConfig,
        'pdf': PdfLoaderConfig,
        'pymupdf': PyMuPDFLoaderConfig,
        'pdfplumber': PdfPlumberLoaderConfig,
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
        'docling': DoclingLoaderConfig,
    }
    
    @classmethod
    def create_config(
        cls, 
        loader_type: str, 
        **kwargs
    ) -> LoaderConfig:
        """Create a configuration for the specified loader type."""
        config_class = cls._config_map.get(loader_type.lower())
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
