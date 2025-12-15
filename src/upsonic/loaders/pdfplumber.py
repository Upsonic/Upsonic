import asyncio
import re
import csv
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any
import concurrent.futures

from upsonic.schemas.data_models import Document
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import PdfPlumberLoaderConfig

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None
    _PDFPLUMBER_AVAILABLE = False


try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_ENGINE = RapidOCR()
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    RapidOCR = None
    OCR_ENGINE = None
    _RAPIDOCR_AVAILABLE = False




class PdfPlumberLoader(BaseLoader):
    """
    An advanced, high-performance loader for PDF documents using pdfplumber.

    This loader is specifically designed for AI agent frameworks, providing:
    - Superior table extraction and preservation
    - Advanced layout analysis and preservation
    - Character-level text extraction precision
    - OCR support for scanned documents (hybrid mode)
    - Complex document structure handling
    - High-quality extraction for RAG applications

    pdfplumber excels at structured documents with tables, forms, and complex
    layouts, making it ideal for enterprise document processing in AI systems.
    """

    def __init__(self, config: Optional[PdfPlumberLoaderConfig] = None):
        """
        Initializes the PdfPlumberLoader with its specific configuration.

        Args:
            config: A PdfPlumberLoaderConfig object with settings for PDF processing.
        """
        if config is None:
            config = PdfPlumberLoaderConfig()
        if not _PDFPLUMBER_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pdfplumber",
                install_command='pip install "upsonic[loaders]"',
                feature_name="pdfplumber PDF loader"
            )
        if config.extraction_mode in ("ocr_only", "hybrid") and not _RAPIDOCR_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="rapidocr-onnxruntime",
                install_command='pip install "upsonic[loaders]"',
                feature_name="PDF OCR functionality"
            )
        super().__init__(config)
        self.config: PdfPlumberLoaderConfig = config


    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Gets a list of file extensions supported by this loader."""
        return [".pdf"]

    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        Loads all PDF documents from the given source synchronously.

        This is a convenience wrapper around the async `aload` method.
        """
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.aload(source))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.aload(source))

    async def aload(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        Loads all PDF documents from the given source asynchronously and concurrently.
        """
        pdf_paths = self._resolve_sources(source)
        if not pdf_paths:
            return []

        tasks = [self._process_single_pdf(path) for path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents = []
        for result in results:
            if isinstance(result, Exception):
                if self.config.error_handling == "raise":
                    raise result
                elif self.config.error_handling == "warn":
                    self._logger.warning("Failed to process PDF: %s", result)
            else:
                documents.extend(result)

        return documents
    
    def batch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Loads documents from a list of sources, leveraging the core `load` method."""
        return self.load(sources)

    async def abatch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Loads documents from a list of sources asynchronously, leveraging the core `aload` method."""
        return await self.aload(sources)



    async def _process_single_pdf(self, path: Path) -> List[Document]:
        """
        Processes a single PDF file, consolidating all page content into a single Document.
        
        This method orchestrates the entire processing pipeline:
        1. Document ID generation and duplicate checking
        2. File size validation
        3. PDF opening and password handling
        4. Page-by-page content extraction
        5. Table extraction and formatting
        6. OCR for scanned content (if configured)
        7. Content cleaning and normalization
        8. Metadata creation
        """
        try:
            document_id = self._generate_document_id(path)
            if document_id in self._processed_document_ids:
                raise FileExistsError(
                    f"Source file '{path.resolve()}' has already been processed by this loader instance."
                )
            self._processed_document_ids.add(document_id)
            
            if not self._check_file_size(path):
                return []
            
            # Open PDF with pdfplumber
            pdf_handle = await asyncio.to_thread(
                pdfplumber.open, 
                str(path),
                password=self.config.pdf_password
            )

            try:
                pages = pdf_handle.pages
                
                # Determine page range
                start_idx = (self.config.start_page - 1) if self.config.start_page else 0
                end_idx = self.config.end_page if self.config.end_page else len(pages)
                pages_to_process = pages[start_idx:end_idx]

                # Process pages concurrently
                page_tasks = [
                    self._extract_page_content(page, page_num=start_idx + i + 1) 
                    for i, page in enumerate(pages_to_process)
                ]
                page_contents_with_nums = await asyncio.gather(*page_tasks)
                
                page_contents = [content for content, _num in page_contents_with_nums]

                # Apply content cleaning
                if self.config.extra_whitespace_removal:
                    page_contents = [self._normalize_whitespace(content) for content in page_contents]

                if self.config.clean_page_numbers:
                    page_contents, _ = self._clean_page_numbers(page_contents)
                
                # Consolidate into single document
                full_content = "\n\n".join(page_contents).strip()

                if self.config.skip_empty_content and not full_content:
                    return []
                
                # Create metadata
                metadata = self._create_metadata(path)
                metadata["page_count"] = len(page_contents)
                
                # Add pdfplumber-specific metadata
                if self.config.include_metadata and pages_to_process:
                    first_page = pages_to_process[0]
                    if self.config.extract_page_dimensions:
                        metadata["page_width"] = first_page.width
                        metadata["page_height"] = first_page.height
                    
                    # Add PDF metadata if available
                    if hasattr(pdf_handle, 'metadata') and pdf_handle.metadata:
                        pdf_metadata = pdf_handle.metadata
                        if pdf_metadata.get('Title'):
                            metadata["pdf_title"] = pdf_metadata['Title']
                        if pdf_metadata.get('Author'):
                            metadata["pdf_author"] = pdf_metadata['Author']
                        if pdf_metadata.get('Subject'):
                            metadata["pdf_subject"] = pdf_metadata['Subject']
                        if pdf_metadata.get('Creator'):
                            metadata["pdf_creator"] = pdf_metadata['Creator']
                
                if self.config.include_metadata:
                    return [Document(document_id=document_id, content=full_content, metadata=metadata)]
                else:
                    return [Document(document_id=document_id, content=full_content)]

            finally:
                # Always close the PDF handle
                await asyncio.to_thread(pdf_handle.close)

        except (PermissionError, FileExistsError) as e:
            raise e
        except Exception as e:
            return self._handle_loading_error(str(path), e)


    async def _extract_page_content(self, page, page_num: int) -> Tuple[str, int]:
        """
        Extracts content from a single page based on the `extraction_mode`.
        
        This method leverages pdfplumber's advanced features:
        - Text extraction with layout preservation
        - Table detection and extraction
        - Image information extraction
        - Character-level precision
        - OCR fallback for scanned content
        
        Args:
            page: A pdfplumber page object
            page_num: The page number (1-indexed)
            
        Returns:
            Tuple of (extracted_content, page_num)
        """
        content_parts = []
        
        # Apply crop box if specified
        working_page = page
        if self.config.crop_box:
            working_page = await asyncio.to_thread(
                page.crop, 
                self.config.crop_box
            )

        # Extract text based on mode
        if self.config.extraction_mode in ("text_only", "hybrid"):
            text_content = await self._extract_text_from_page(working_page)
            if text_content:
                content_parts.append(text_content)
        
        # Extract tables if configured
        if self.config.extract_tables and self.config.extraction_mode in ("text_only", "hybrid"):
            tables_content = await self._extract_tables_from_page(working_page)
            if tables_content:
                content_parts.append(tables_content)
        
        # Extract image information if configured
        if self.config.extract_images:
            images_info = await self._extract_images_info(working_page)
            if images_info:
                content_parts.append(images_info)
        
        # Extract annotations if configured
        if self.config.extract_annotations:
            annotations = await self._extract_annotations(working_page)
            if annotations:
                content_parts.append(annotations)
        
        # Perform OCR if needed
        if self.config.extraction_mode in ("ocr_only", "hybrid"):
            ocr_text = await self._perform_ocr(working_page)
            if ocr_text:
                content_parts.append(ocr_text)
        
        full_content = "\n\n".join(content_parts).strip()
        return full_content, page_num


    async def _extract_text_from_page(self, page) -> str:
        """
        Extracts text from a page using pdfplumber's advanced text extraction.
        
        Uses configuration settings for:
        - Layout preservation
        - Character/word/line margins
        - Text flow analysis
        """
        extract_text_kwargs = {
            "x_tolerance": self.config.char_margin,
            "y_tolerance": self.config.line_margin,
            "layout": self.config.layout_mode == "layout",
            "keep_blank_chars": self.config.keep_blank_chars,
        }
        
        if self.config.use_text_flow:
            extract_text_kwargs["use_text_flow"] = True
        
        text = await asyncio.to_thread(
            page.extract_text,
            **extract_text_kwargs
        )
        
        return text or ""


    async def _extract_tables_from_page(self, page) -> str:
        """
        Extracts and formats tables from a page using pdfplumber's superior table detection.
        
        pdfplumber excels at table extraction, making it ideal for structured documents.
        Tables are formatted according to the configured table_format.
        """
        tables = await asyncio.to_thread(
            page.extract_tables,
            table_settings=self.config.table_settings
        )
        
        if not tables:
            return ""
        
        formatted_tables = []
        for i, table in enumerate(tables):
            if not table or not any(table):
                continue
            
            table_title = f"\n[Table {i+1}]\n"
            formatted_table = self._format_table(table, self.config.table_format)
            formatted_tables.append(table_title + formatted_table)
        
        return "\n\n".join(formatted_tables)


    def _format_table(self, table: List[List[Any]], format_type: str) -> str:
        """
        Formats a table according to the specified format type.
        
        Args:
            table: 2D list representing the table
            format_type: One of "text", "markdown", "csv", "grid"
            
        Returns:
            Formatted table as a string
        """
        if not table:
            return ""
        
        # Clean None values and convert to strings
        cleaned_table = [
            [str(cell) if cell is not None else "" for cell in row]
            for row in table
        ]
        
        if format_type == "markdown":
            return self._format_table_markdown(cleaned_table)
        elif format_type == "csv":
            return self._format_table_csv(cleaned_table)
        elif format_type == "grid":
            return self._format_table_grid(cleaned_table)
        else:  # "text"
            return self._format_table_text(cleaned_table)


    def _format_table_markdown(self, table: List[List[str]]) -> str:
        """Formats a table as Markdown (ideal for AI agents and RAG)."""
        if not table or len(table) < 1:
            return ""
        
        lines = []
        
        # Header row
        if table:
            header = table[0]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---" for _ in header]) + "|")
            
            # Data rows
            for row in table[1:]:
                # Pad row if it has fewer columns than header
                padded_row = row + [""] * (len(header) - len(row))
                lines.append("| " + " | ".join(padded_row[:len(header)]) + " |")
        
        return "\n".join(lines)


    def _format_table_csv(self, table: List[List[str]]) -> str:
        """Formats a table as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(table)
        return output.getvalue().strip()


    def _format_table_grid(self, table: List[List[str]]) -> str:
        """Formats a table as ASCII grid."""
        if not table:
            return ""
        
        # Calculate column widths
        col_widths = []
        max_cols = max(len(row) for row in table)
        
        for col_idx in range(max_cols):
            max_width = max(
                len(row[col_idx]) if col_idx < len(row) else 0
                for row in table
            )
            col_widths.append(max(max_width, 3))
        
        # Build grid
        lines = []
        separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
        
        for row in table:
            padded_row = row + [""] * (max_cols - len(row))
            line = "|" + "|".join([
                f" {cell:<{col_widths[i]}} "
                for i, cell in enumerate(padded_row)
            ]) + "|"
            lines.append(separator)
            lines.append(line)
        lines.append(separator)
        
        return "\n".join(lines)


    def _format_table_text(self, table: List[List[str]]) -> str:
        """Formats a table as plain text with spacing."""
        if not table:
            return ""
        
        # Calculate column widths
        max_cols = max(len(row) for row in table)
        col_widths = []
        
        for col_idx in range(max_cols):
            max_width = max(
                len(row[col_idx]) if col_idx < len(row) else 0
                for row in table
            )
            col_widths.append(max_width)
        
        # Format rows
        lines = []
        for row in table:
            padded_row = row + [""] * (max_cols - len(row))
            line = "  ".join([
                cell.ljust(col_widths[i])
                for i, cell in enumerate(padded_row)
            ])
            lines.append(line)
        
        return "\n".join(lines)


    async def _extract_images_info(self, page) -> str:
        """
        Extracts information about images on the page.
        
        Returns a formatted string with image metadata.
        """
        images = await asyncio.to_thread(lambda: page.images)
        
        if not images:
            return ""
        
        info_parts = [f"\n[Images: {len(images)} found on page]"]
        
        for i, img in enumerate(images):
            img_info = f"Image {i+1}: "
            if 'width' in img and 'height' in img:
                img_info += f"{img['width']}x{img['height']}px"
            if 'x0' in img and 'y0' in img:
                img_info += f" at ({img['x0']:.1f}, {img['y0']:.1f})"
            info_parts.append(img_info)
        
        return "\n".join(info_parts)


    async def _extract_annotations(self, page) -> str:
        """
        Extracts annotations and hyperlinks from the page.
        """
        annotations = await asyncio.to_thread(lambda: page.annots)
        hyperlinks = await asyncio.to_thread(lambda: page.hyperlinks)
        
        parts = []
        
        if annotations:
            parts.append(f"\n[Annotations: {len(annotations)}]")
            for i, annot in enumerate(annotations[:10]):  # Limit to 10
                if 'contents' in annot and annot['contents']:
                    parts.append(f"Note {i+1}: {annot['contents']}")
        
        if hyperlinks:
            parts.append(f"\n[Hyperlinks: {len(hyperlinks)}]")
            for i, link in enumerate(hyperlinks[:10]):  # Limit to 10
                if 'uri' in link:
                    parts.append(f"Link {i+1}: {link['uri']}")
        
        return "\n".join(parts) if parts else ""


    async def _perform_ocr(self, page) -> str:
        """
        Performs OCR on the page by converting it to an image.
        
        This is used when extraction_mode is "ocr_only" or "hybrid" (as fallback).
        For hybrid mode, OCR is performed on the entire page when text extraction yields little content.
        """
        try:
            # Convert page to image using pdfplumber's built-in functionality
            img = await asyncio.to_thread(
                page.to_image,
                resolution=150  # Good balance between quality and performance
            )
            
            # Get PIL image
            pil_image = await asyncio.to_thread(lambda: img.original)
            
            # Convert to bytes for OCR
            img_bytes_io = io.BytesIO()
            await asyncio.to_thread(pil_image.save, img_bytes_io, format='PNG')
            img_bytes = img_bytes_io.getvalue()
            
            # Run OCR
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                ocr_result = await loop.run_in_executor(
                    pool, 
                    self._run_single_ocr, 
                    img_bytes
                )
            
            return ocr_result
            
        except (ImportError, ValueError, OSError) as e:
            self._logger.warning("OCR failed for page: %s", e)
            return ""


    @staticmethod
    def _run_single_ocr(image_data: bytes) -> str:
        """Helper function that runs the synchronous OCR engine."""
        if not OCR_ENGINE:
            return ""
        result, _ = OCR_ENGINE(image_data)
        if result:
            return "\n".join([item[1] for item in result])
        return ""


    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapses multiple spaces/newlines and trims."""
        text = re.sub(r"\s*\n\s*", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
    

    def _clean_page_numbers(self, page_content_list: List[str]) -> Tuple[List[str], Optional[int]]:
        """
        Identifies and removes sequential page numbers from the top or bottom of pages.
        
        This method uses intelligent pattern matching to detect page numbers and removes them
        while optionally adding custom markers based on configuration.
        
        Args:
            page_content_list: List of page contents as strings
            
        Returns:
            Tuple of (cleaned_page_list, starting_page_number)
        """
        page_number_regex = re.compile(r"^\s*(\d+)\s*$")

        def find_page_number(content):
            lines = content.strip().split('\n')
            if not lines: 
                return None
            # Check first and last lines for standalone page numbers
            for line in [lines[0], lines[-1]]:
                match = page_number_regex.search(line)
                if match:
                    return int(match.group(1))
            return None

        page_numbers = [find_page_number(content) for content in page_content_list]
        
        # Find the best sequence match
        best_match, best_correct_count, best_shift = None, 0, None
        start_page = self.config.start_page or 1
        
        for shift in range(start_page, start_page + len(page_content_list) + 1):
            expected_numbers = list(range(shift, shift + len(page_numbers)))
            correct_count = sum(1 for actual, expected in zip(page_numbers, expected_numbers) if actual == expected)
            if correct_count > best_correct_count:
                best_correct_count, best_match, best_shift = correct_count, expected_numbers, shift
        
        # Apply cleaning if we found a good match (40% threshold)
        if best_match and best_correct_count / len(page_numbers) >= 0.4:
            cleaned_pages = []
            for i, content in enumerate(page_content_list):
                lines = content.strip().split('\n')
                expected_number = str(best_match[i])
                
                # Remove page number from top
                if lines and lines[0].strip() == expected_number:
                    lines.pop(0)
                # Remove page number from bottom
                if lines and lines[-1].strip() == expected_number:
                    lines.pop(-1)
                
                cleaned_content = '\n'.join(lines)

                # Add custom markers if configured
                start_marker = self.config.page_num_start_format.format(page_nr=expected_number) + "\n" if self.config.page_num_start_format else ""
                end_marker = "\n" + self.config.page_num_end_format.format(page_nr=expected_number) if self.config.page_num_end_format else ""

                cleaned_pages.append(f"{start_marker}{cleaned_content}{end_marker}")
            return cleaned_pages, best_shift
        
        return page_content_list, None

