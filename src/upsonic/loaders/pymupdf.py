import asyncio
import re
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import concurrent.futures

from upsonic.schemas.data_models import Document
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import PyMuPDFLoaderConfig

try:
    import pymupdf
    _PYMUPDF_AVAILABLE = True
except ImportError:
    pymupdf = None
    _PYMUPDF_AVAILABLE = False


try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_ENGINE = RapidOCR()
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    RapidOCR = None
    OCR_ENGINE = None
    _RAPIDOCR_AVAILABLE = False


class PyMuPDFLoader(BaseLoader):
    """
    A high-performance, comprehensive PDF loader using PyMuPDF (fitz).

    This loader leverages PyMuPDF's superior performance and features for
    text extraction, OCR, image handling, and content structuring. It supports
    digital text extraction, OCR for scanned images, and advanced content
    processing options driven by the PyMuPDFLoaderConfig.
    """

    def __init__(self, config: Optional[PyMuPDFLoaderConfig] = None):
        """
        Initializes the PyMuPDFLoader with its specific configuration.

        Args:
            config: A PyMuPDFLoaderConfig object with settings for PDF processing.
        """
        if config is None:
            config = PyMuPDFLoaderConfig()
        if not _PYMUPDF_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pymupdf",
                install_command='pip install "upsonic[loaders]"',
                feature_name="PyMuPDF loader"
            )
        super().__init__(config)
        self.config: PyMuPDFLoaderConfig = config

        if "ocr" in self.config.extraction_mode and not _RAPIDOCR_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="rapidocr-onnxruntime",
                install_command='pip install "upsonic[loaders]"',
                feature_name="PyMuPDF OCR functionality"
            )

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
            loop = asyncio.get_running_loop()
            import concurrent.futures
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
                    self._logger.warning(f"Failed to process PDF: {result}")
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
            
            # Open PDF document
            doc = await asyncio.to_thread(pymupdf.open, str(path))

            # Handle password-protected PDFs
            if doc.needs_pass:
                if not self.config.pdf_password:
                    doc.close()
                    raise PermissionError(f"PDF {path} is encrypted but no password provided.")
                
                auth_result = await asyncio.to_thread(doc.authenticate, self.config.pdf_password)
                if not auth_result:
                    doc.close()
                    raise PermissionError(f"Could not decrypt {path}. Invalid password provided.")
            
            # Determine page range
            total_pages = len(doc)
            start_idx = (self.config.start_page - 1) if self.config.start_page else 0
            end_idx = self.config.end_page if self.config.end_page else total_pages
            end_idx = min(end_idx, total_pages)
            
            # Process pages
            page_tasks = []
            for page_num in range(start_idx, end_idx):
                page = await asyncio.to_thread(doc.load_page, page_num)
                page_tasks.append(self._extract_page_content(page, page_num + 1))
            
            page_contents_with_nums = await asyncio.gather(*page_tasks)
            
            page_contents = [content for content, num in page_contents_with_nums]
            page_numbers = [num for content, num in page_contents_with_nums]

            # Apply content processing
            if self.config.extra_whitespace_removal:
                page_contents = [self._normalize_whitespace(content) for content in page_contents]

            if self.config.clean_page_numbers:
                page_contents, _ = self._clean_page_numbers(page_contents)
            
            full_content = "\n\n".join(page_contents).strip()

            if self.config.skip_empty_content and not full_content:
                doc.close()
                return []
            
            # Create metadata using base class method (includes custom_metadata from base config)
            metadata = self._create_metadata(path)
            metadata["page_count"] = len(page_contents)
            metadata["total_pages"] = total_pages
            metadata["processed_pages"] = f"{start_idx + 1}-{end_idx}"
            
            # Add PyMuPDF-specific metadata if include_metadata is enabled
            if self.config.include_metadata:
                metadata.update({
                    "extraction_method": self.config.text_extraction_method,
                    "preserve_layout": self.config.preserve_layout,
                    "image_dpi": self.config.image_dpi,
                })
                
                # Extract additional metadata from document
                doc_metadata = await self._extract_document_metadata(doc)
                metadata.update(doc_metadata)
                
                # Extract annotations if requested
                if self.config.extract_annotations:
                    annotations = await self._extract_annotations(doc, start_idx, end_idx)
                    if annotations:
                        metadata["annotations"] = annotations

            doc.close()
            
            # Return document with proper fields (always include metadata since _create_metadata handles include_metadata)
            return [Document(document_id=document_id, content=full_content, metadata=metadata)]

        except (PermissionError, FileExistsError) as e:
            raise e
        except Exception as e:
            return self._handle_loading_error(str(path), e)

    async def _extract_page_content(self, page: "pymupdf.Page", page_num: int) -> Tuple[str, int]:
        """
        Extracts content from a single page based on the `extraction_mode` and `text_extraction_method`.
        """
        text = ""
        ocr_text = ""

        if self.config.extraction_mode in ("text_only", "hybrid"):
            text = await asyncio.to_thread(self._extract_text_from_page, page)

        if self.config.extraction_mode in ("ocr_only", "hybrid"):
            ocr_text = await self._perform_ocr(page)
        
        full_content = f"{text}\n\n{ocr_text}".strip()
        return full_content, page_num

    def _extract_text_from_page(self, page: "pymupdf.Page") -> str:
        """
        Extracts text from a page using the configured extraction method.
        """
        if self.config.text_extraction_method == "text":
            return page.get_text()
        elif self.config.text_extraction_method == "dict":
            text_dict = page.get_text("dict")
            return self._process_text_dict(text_dict)
        elif self.config.text_extraction_method == "html":
            return page.get_text("html")
        elif self.config.text_extraction_method == "xml":
            return page.get_text("xml")
        else:
            return page.get_text()

    def _process_text_dict(self, text_dict: Dict[str, Any]) -> str:
        """
        Processes PyMuPDF's text dictionary format into readable text.
        """
        text_parts = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:  # Text block
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    if line_text.strip():
                        text_parts.append(line_text)
        
        return "\n".join(text_parts)

    async def _perform_ocr(self, page: "pymupdf.Page") -> str:
        """
        Performs OCR on a PDF page using PyMuPDF's image extraction and RapidOCR.
        """
        if not OCR_ENGINE:
            return ""

        # Get page as image
        mat = pymupdf.Matrix(self.config.image_dpi / 72, self.config.image_dpi / 72)
        pix = await asyncio.to_thread(page.get_pixmap, matrix=mat)
        
        # Convert to bytes
        img_data = await asyncio.to_thread(pix.tobytes, "png")
        pix = None  # Free memory
        
        # Perform OCR
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            ocr_result = await loop.run_in_executor(pool, self._run_single_ocr, img_data)
        
        return ocr_result

    @staticmethod
    def _run_single_ocr(image_data: bytes) -> str:
        """Helper function that runs the synchronous OCR engine."""
        if not OCR_ENGINE:
            return ""
        result, _ = OCR_ENGINE(image_data)
        if result:
            return "\n".join([item[1] for item in result])
        return ""

    async def _extract_document_metadata(self, doc: "pymupdf.Document") -> Dict[str, Any]:
        """
        Extracts metadata from the PDF document.
        """
        metadata = {}
        
        # Get document metadata
        doc_metadata = doc.metadata
        if doc_metadata:
            for key, value in doc_metadata.items():
                if value and value.strip():
                    metadata[f"pdf_{key.lower()}"] = value
        
        # Get document info
        try:
            doc_info = await asyncio.to_thread(doc.get_pdf_metadata)
            if doc_info:
                metadata.update(doc_info)
        except:
            # get_pdf_metadata might not be available in all versions
            pass
        
        # Extract images if requested
        if self.config.include_images:
            try:
                images_info = await self._extract_images_info(doc)
                if images_info:
                    metadata["images"] = images_info
            except Exception as e:
                # Skip image extraction if it fails
                pass
        
        return metadata

    async def _extract_images_info(self, doc: "pymupdf.Document") -> List[Dict[str, Any]]:
        """
        Extracts information about images in the document.
        """
        images_info = []
        
        for page_num in range(len(doc)):
            page = await asyncio.to_thread(doc.load_page, page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    img_info = {
                        "page": page_num + 1,
                        "index": img_index,
                        "xref": img[0],
                        "smask": img[1],
                        "width": img[2],
                        "height": img[3],
                        "bpc": img[4],
                        "colorspace": img[5],
                        "alt": img[6],
                        "name": img[7],
                        "filter": img[8],
                    }
                    images_info.append(img_info)
                except (IndexError, TypeError) as e:
                    # Skip malformed image entries
                    continue
        
        return images_info

    async def _extract_annotations(self, doc: "pymupdf.Document", start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """
        Extracts annotations from the specified page range.
        """
        annotations = []
        
        for page_num in range(start_idx, end_idx):
            page = await asyncio.to_thread(doc.load_page, page_num)
            page_annotations = page.annots()
            
            for annot in page_annotations:
                try:
                    annot_info = {
                        "page": page_num + 1,
                        "type": annot.type[1] if isinstance(annot.type, tuple) else annot.type,
                        "content": getattr(annot, 'content', ''),
                        "rect": str(annot.rect),
                    }
                    
                    if self.config.annotation_format == "json":
                        try:
                            annot_info["json"] = annot.get_text()
                        except:
                            annot_info["json"] = ""
                    
                    annotations.append(annot_info)
                except Exception as e:
                    # Skip malformed annotations
                    continue
        
        return annotations

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapses multiple spaces/newlines and trims."""
        text = re.sub(r"\s*\n\s*", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
    
    def _clean_page_numbers(self, page_content_list: List[str]) -> Tuple[List[str], Optional[int]]:
        """
        Identifies and removes sequential page numbers from the top or bottom of pages.
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
        
        if best_match and best_correct_count / len(page_numbers) >= 0.4:
            cleaned_pages = []
            for i, content in enumerate(page_content_list):
                lines = content.strip().split('\n')
                expected_number = str(best_match[i])
                
                if lines and lines[0].strip() == expected_number:
                    lines.pop(0)
                if lines and lines[-1].strip() == expected_number:
                    lines.pop(-1)
                
                cleaned_content = '\n'.join(lines)

                start_marker = self.config.page_num_start_format.format(page_nr=expected_number) + "\n" if self.config.page_num_start_format else ""
                end_marker = "\n" + self.config.page_num_end_format.format(page_nr=expected_number) if self.config.page_num_end_format else ""

                cleaned_pages.append(f"{start_marker}{cleaned_content}{end_marker}")
            return cleaned_pages, best_shift
        
        return page_content_list, None
