from __future__ import annotations
import asyncio
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import DoclingLoaderConfig
from upsonic.schemas.data_models import Document

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import DoclingDocument
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        RapidOcrOptions,
        TesseractCliOcrOptions,
    )
    from docling.exceptions import ConversionError
    
    try:
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
        CHUNKING_AVAILABLE = True
    except ImportError:
        try:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
            from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
            CHUNKING_AVAILABLE = True
        except ImportError:
            CHUNKING_AVAILABLE = False
    
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    ConversionError = Exception  # Fallback
    PdfFormatOption = None
    PdfPipelineOptions = None
    RapidOcrOptions = None
    TesseractCliOcrOptions = None


class DoclingLoader(BaseLoader):
    """
    Advanced document loader using Docling for enterprise-grade document processing.
    
    Docling provides state-of-the-art document understanding capabilities with support for:
    - PDF, DOCX, XLSX, PPTX (Office formats)
    - HTML, Markdown, AsciiDoc (web and markup)
    - CSV (tabular data)
    - Images (PNG, JPEG, TIFF, BMP, WEBP)
    - URLs (remote document loading)
    
    The loader supports two extraction modes:
    1. Markdown: Exports the entire document as formatted markdown
    2. Chunks: Intelligently segments documents into semantic chunks for RAG applications
    
    Example:
        >>> from upsonic.loaders import DoclingLoader, DoclingLoaderConfig
        >>> 
        >>> config = DoclingLoaderConfig(extraction_mode="chunks")
        >>> loader = DoclingLoader(config)
        >>> documents = loader.load("path/to/document.pdf")
    """

    def __init__(self, config: Optional[DoclingLoaderConfig] = None):
        """
        Initialize the Docling loader.
        
        Args:
            config: Configuration object specifying extraction mode, chunking strategy,
                   and other processing options.
                   
        Raises:
            ImportError: If the docling package is not installed.
        """
        if config is None:
            config = DoclingLoaderConfig()
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "The docling package is required to use DoclingLoader. "
                "Please install it using: pip install docling"
            )
        
        super().__init__(config)
        self.config: DoclingLoaderConfig = config
        self._converter = self._create_converter()
        self._chunker = self._create_chunker() if config.extraction_mode == "chunks" else None
        
    def _create_converter(self) -> 'DocumentConverter':
        """
        Create and configure a DocumentConverter instance with OCR and pipeline options.
        
        Returns:
            Configured DocumentConverter instance.
        """
        # Build allowed formats list
        allowed_formats = []
        if self.config.allowed_formats:
            format_mapping = {
                'pdf': InputFormat.PDF,
                'docx': InputFormat.DOCX,
                'xlsx': InputFormat.XLSX,
                'pptx': InputFormat.PPTX,
                'html': InputFormat.HTML,
                'md': InputFormat.MD,
                'markdown': InputFormat.MD,
                'asciidoc': InputFormat.ASCIIDOC,
                'csv': InputFormat.CSV,
                'image': InputFormat.IMAGE,
            }
            for fmt in self.config.allowed_formats:
                if fmt.lower() in format_mapping:
                    allowed_formats.append(format_mapping[fmt.lower()])
        else:
            # Allow all formats by default
            allowed_formats = [
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.XLSX,
                InputFormat.PPTX,
                InputFormat.HTML,
                InputFormat.MD,
                InputFormat.ASCIIDOC,
                InputFormat.CSV,
                InputFormat.IMAGE,
            ]
        
        # Configure PDF pipeline options (for PDFs and images)
        format_options = {}
        
        # Only configure PDF pipeline if PDF or IMAGE is in allowed formats
        if InputFormat.PDF in allowed_formats or InputFormat.IMAGE in allowed_formats:
            pdf_pipeline_options = self._create_pdf_pipeline_options()
            pdf_format_option = PdfFormatOption(pipeline_options=pdf_pipeline_options)
            format_options[InputFormat.PDF] = pdf_format_option
            format_options[InputFormat.IMAGE] = pdf_format_option
        
        # Create converter with format options
        converter = DocumentConverter(
            allowed_formats=allowed_formats,
            format_options=format_options if format_options else None
        )
        return converter
    
    def _create_pdf_pipeline_options(self) -> 'PdfPipelineOptions':
        """
        Create PDF pipeline options with OCR configuration.
        
        Returns:
            Configured PdfPipelineOptions instance.
        """
        # Create base pipeline options
        pipeline_options = PdfPipelineOptions()
        
        # Configure OCR (using config.ocr_enabled)
        pipeline_options.do_ocr = self.config.ocr_enabled
        
        # Configure table structure (using config.enable_table_structure)
        pipeline_options.do_table_structure = self.config.enable_table_structure
        if self.config.enable_table_structure:
            # Configure cell matching (using config.table_structure_cell_matching)
            pipeline_options.table_structure_options.do_cell_matching = self.config.table_structure_cell_matching
        
        # Create and configure OCR options if OCR is enabled
        if self.config.ocr_enabled:
            ocr_options = self._create_ocr_options()
            pipeline_options.ocr_options = ocr_options
        
        return pipeline_options
    
    def _create_ocr_options(self) -> Union['RapidOcrOptions', 'TesseractCliOcrOptions']:
        """
        Create OCR options based on configured backend.
        
        Returns:
            Configured OCR options instance.
        """
        # Use config.ocr_force_full_page for full-page OCR
        force_full_page = self.config.ocr_force_full_page
        
        # Create OCR options based on backend (using config.ocr_backend)
        if self.config.ocr_backend == "rapidocr":
            # RapidOCR options (using config.ocr_lang, config.ocr_backend_engine, config.ocr_text_score)
            ocr_options = RapidOcrOptions(
                force_full_page_ocr=force_full_page,
                lang=self.config.ocr_lang,
                backend=self.config.ocr_backend_engine,
                text_score=self.config.ocr_text_score,
            )
        else:  # tesseract
            # Tesseract options (using config.ocr_lang)
            ocr_options = TesseractCliOcrOptions(
                force_full_page_ocr=force_full_page,
                lang=self.config.ocr_lang,
            )
        
        return ocr_options
    
    def _create_chunker(self) -> Optional[any]:
        """
        Create and configure a chunker instance based on config.
        
        Returns:
            Configured chunker instance or None if chunking is not available.
        """
        if not CHUNKING_AVAILABLE:
            self._logger.warning(
                "Docling chunking modules not available. "
                "Falling back to full document extraction."
            )
            return None
        
        # Create chunker based on type
        if self.config.chunker_type == "hierarchical":
            chunker = HierarchicalChunker()
        else:  # hybrid
            chunker = HybridChunker()
        
        return chunker
    
    def _is_url(self, source: Union[str, Path]) -> bool:
        """Check if the source is a URL."""
        if isinstance(source, Path):
            return False
        try:
            result = urlparse(str(source))
            return result.scheme in ('http', 'https') and bool(result.netloc)
        except Exception:
            return False
    
    def _validate_source(self, source: Union[str, Path]) -> Union[str, Path]:
        """
        Validate a source path or URL.
        
        Args:
            source: File path or URL to validate.
            
        Returns:
            Validated source.
            
        Raises:
            ValueError: If URL support is disabled but URL is provided.
            FileNotFoundError: If local file doesn't exist.
        """
        if self._is_url(source):
            if not self.config.support_urls:
                raise ValueError(
                    f"URL loading is disabled. Enable it with support_urls=True. Source: {source}"
                )
            return str(source)
        
        # Handle local file
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        
        # Check file size using base class method
        if not self._check_file_size(path):
            raise ValueError(f"File size exceeds maximum allowed size: {path}")
        
        return path
    
    def _convert_document(self, source: Union[str, Path]) -> Optional['DoclingDocument']:
        """
        Convert a single document using Docling.
        
        Args:
            source: Path or URL to the document.
            
        Returns:
            DoclingDocument instance or None if conversion failed.
        """
        try:
            # Validate source
            validated_source = self._validate_source(source)
            
            # Build conversion kwargs
            convert_kwargs = {}
            
            # Add page range if specified (using config.page_range)
            if self.config.page_range:
                start_page, end_page = self.config.page_range
                convert_kwargs['pages'] = list(range(start_page - 1, end_page))  # Convert to 0-indexed
            elif self.config.max_pages:
                # Use config.max_pages
                convert_kwargs['pages'] = list(range(self.config.max_pages))
            
            # Add timeout for URLs (using config.url_timeout)
            if self._is_url(validated_source):
                convert_kwargs['timeout'] = self.config.url_timeout
            
            # Convert document
            conversion_result = self._converter.convert(
                source=str(validated_source),
                **convert_kwargs
            )
            
            return conversion_result.document
            
        except ConversionError as e:
            # Use base class error handling (config.error_handling)
            return self._handle_conversion_error(source, e)
        except Exception as e:
            return self._handle_conversion_error(source, e)
    
    def _handle_conversion_error(self, source: Union[str, Path], error: Exception):
        """Handle conversion errors using base class method."""
        # This will respect config.error_handling (raise/warn/ignore)
        docs = self._handle_loading_error(str(source), error)
        return None  # Return None for document conversion failures
    
    def _extract_markdown(self, dl_doc: 'DoclingDocument', source: Union[str, Path], document_id: str) -> List[Document]:
        """
        Extract document as markdown.
        
        Args:
            dl_doc: DoclingDocument instance.
            source: Original source path/URL.
            document_id: Document ID from base class.
            
        Returns:
            List containing a single Document with markdown content.
        """
        try:
            # Build markdown export kwargs (using config.markdown_image_placeholder)
            md_kwargs = {
                'image_placeholder': self.config.markdown_image_placeholder,
            }
            
            # Export to markdown
            markdown_content = dl_doc.export_to_markdown(**md_kwargs)
            
            # Use config.skip_empty_content from base class
            if self.config.skip_empty_content and not markdown_content.strip():
                return []
            
            # Create metadata using base class method
            if isinstance(source, Path):
                metadata = self._create_metadata(source)
            else:
                # For URLs, create minimal metadata
                metadata = {
                    "source": str(source),
                    "loader_type": "docling",
                }
                # Add custom_metadata from base config
                if self.config.custom_metadata:
                    metadata.update(self.config.custom_metadata)
            
            metadata['extraction_mode'] = 'markdown'
            
            # Add document metadata if enabled (using config.extract_document_metadata)
            if self.config.extract_document_metadata:
                self._add_docling_metadata(metadata, dl_doc)
            
            # Create document with proper fields (content, document_id, metadata)
            doc = Document(
                content=markdown_content,
                document_id=document_id,
                metadata=metadata
            )
            
            return [doc]
            
        except Exception as e:
            self._logger.error(f"Failed to export markdown for {source}: {e}")
            return []
    
    def _extract_chunks(self, dl_doc: 'DoclingDocument', source: Union[str, Path], document_id: str) -> List[Document]:
        """
        Extract document as semantic chunks.
        
        Args:
            dl_doc: DoclingDocument instance.
            source: Original source path/URL.
            document_id: Document ID from base class.
            
        Returns:
            List of Documents, one per chunk.
        """
        if not self._chunker:
            # Fallback to markdown if chunker not available
            self._logger.warning(f"Chunker not available, extracting as markdown: {source}")
            return self._extract_markdown(dl_doc, source, document_id)
        
        documents = []
        
        try:
            # Chunk the document
            chunk_iter = self._chunker.chunk(dl_doc)
            
            for idx, chunk in enumerate(chunk_iter):
                # Serialize chunk to text
                try:
                    if hasattr(self._chunker, 'serialize'):
                        chunk_text = self._chunker.serialize(chunk=chunk)
                    else:
                        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                except Exception:
                    chunk_text = str(chunk)
                
                # Use config.skip_empty_content from base class
                if self.config.skip_empty_content and not chunk_text.strip():
                    continue
                
                # Create metadata using base class method
                if isinstance(source, Path):
                    metadata = self._create_metadata(source)
                else:
                    metadata = {
                        "source": str(source),
                        "loader_type": "docling",
                    }
                    # Add custom_metadata from base config
                    if self.config.custom_metadata:
                        metadata.update(self.config.custom_metadata)
                
                metadata['extraction_mode'] = 'chunks'
                metadata['chunker_type'] = self.config.chunker_type
                metadata['chunk_index'] = idx
                
                # Add document metadata if enabled (using config.extract_document_metadata)
                if self.config.extract_document_metadata:
                    self._add_docling_metadata(metadata, dl_doc)
                
                # Add chunk-specific metadata
                if hasattr(chunk, 'meta'):
                    try:
                        chunk_meta = chunk.meta.export_json_dict() if hasattr(chunk.meta, 'export_json_dict') else {}
                        metadata['chunk_meta'] = chunk_meta
                    except Exception:
                        pass
                
                # Check confidence threshold (using config.confidence_threshold)
                if hasattr(chunk, 'confidence'):
                    if chunk.confidence < self.config.confidence_threshold:
                        continue
                    metadata['confidence'] = chunk.confidence
                
                # Create document with unique ID for each chunk
                chunk_doc_id = f"{document_id}_chunk_{idx}"
                doc = Document(
                    content=chunk_text,
                    document_id=chunk_doc_id,
                    metadata=metadata
                )
                
                documents.append(doc)
            
        except Exception as e:
            self._logger.error(f"Failed to chunk document {source}: {e}")
            return []
        
        return documents
    
    def _add_docling_metadata(self, metadata: dict, dl_doc: 'DoclingDocument'):
        """Add Docling-specific metadata to the metadata dict."""
        try:
            if hasattr(dl_doc, 'name'):
                metadata['document_name'] = dl_doc.name
            
            if hasattr(dl_doc, 'metadata'):
                doc_meta = dl_doc.metadata
                if hasattr(doc_meta, 'export_json_dict'):
                    metadata['docling_metadata'] = doc_meta.export_json_dict()
            
            if hasattr(dl_doc, 'pages'):
                metadata['page_count'] = len(dl_doc.pages)
                
        except Exception as e:
            self._logger.debug(f"Could not extract document metadata: {e}")
    
    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        Load and process documents from the given source(s).
        
        Args:
            source: Single file path/URL or list of file paths/URLs.
            
        Returns:
            List of processed Document objects.
        """
        # Handle single source vs list
        if isinstance(source, list):
            sources = source
        else:
            sources = [source]
        
        all_documents = []
        
        for src in sources:
            # Generate document ID using base class method
            if isinstance(src, Path):
                doc_path = src
            elif self._is_url(src):
                # For URLs, create a path-like ID
                doc_path = Path(f"url_{abs(hash(str(src)))}")
            else:
                doc_path = Path(src)
            
            document_id = self._generate_document_id(doc_path)
            
            # Check if already processed using base class attribute
            if document_id in self._processed_document_ids:
                self._logger.warning(f"Document already processed, skipping: {src}")
                continue
            
            # Convert document
            dl_doc = self._convert_document(src)
            if dl_doc is None:
                continue
            
            # Mark as processed
            self._processed_document_ids.add(document_id)
            
            # Extract content based on mode (using config.extraction_mode)
            if self.config.extraction_mode == "markdown":
                documents = self._extract_markdown(dl_doc, src, document_id)
            else:  # chunks
                documents = self._extract_chunks(dl_doc, src, document_id)
            
            all_documents.extend(documents)
        
        return all_documents
    
    async def aload(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        Asynchronously load and process documents.
        
        Args:
            source: Single file path/URL or list of file paths/URLs.
            
        Returns:
            List of processed Document objects.
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, source)
    
    def batch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """
        Load documents from multiple sources.
        
        Args:
            sources: List of file paths/URLs.
            
        Returns:
            List of processed Document objects from all sources.
        """
        return self.load(sources)
    
    async def abatch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """
        Asynchronously load documents from multiple sources with parallel processing.
        
        Args:
            sources: List of file paths/URLs.
            
        Returns:
            List of processed Document objects from all sources.
        """
        # Use config.parallel_processing to determine strategy
        if not self.config.parallel_processing:
            # Sequential processing
            return await self.aload(sources)
        
        # Process in batches using config.batch_size
        batch_size = self.config.batch_size
        all_documents = []
        
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size]
            
            # Create tasks for batch
            tasks = [self.aload([src]) for src in batch]
            
            # Wait for all tasks in batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect documents from successful tasks
            for result in batch_results:
                if isinstance(result, list):
                    all_documents.extend(result)
                elif isinstance(result, Exception):
                    self._logger.error(f"Batch processing error: {result}")
        
        return all_documents
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Get list of file extensions supported by Docling.
        
        Returns:
            List of supported extensions including dot (e.g., '.pdf').
        """
        return [
            '.pdf',
            '.docx',
            '.xlsx',
            '.pptx',
            '.html', '.htm',
            '.md', '.markdown',
            '.adoc', '.asciidoc',
            '.csv',
            '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp',
        ]
    
    @classmethod
    def can_load(cls, source: Union[str, Path]) -> bool:
        """
        Check if this loader can handle the given source.
        
        Args:
            source: File path or URL to check.
            
        Returns:
            True if the source can be loaded, False otherwise.
        """
        # Check if it's a URL
        if isinstance(source, str):
            try:
                result = urlparse(source)
                if result.scheme in ('http', 'https') and result.netloc:
                    return True  # URLs are supported
            except Exception:
                pass
        
        # Check file extension using base class method
        return super().can_load(source)
