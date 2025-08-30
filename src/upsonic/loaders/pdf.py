from __future__ import annotations
from typing import List, Any, Dict, Literal, Optional
import os
import re
from datetime import datetime

from .base import DocumentLoader
from .config import PDFLoaderConfig
from ..schemas.data_models import Document

try:
    import pypdf
except ImportError:
    raise ImportError(
        "pypdf is not installed. It is required for the PDFLoader. "
    )

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    pytesseract = None
    OCR_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


class PDFLoader(DocumentLoader):
    """
    A master-class, multi-modal ingestion engine for PDF documents.

    This loader is a pipeline that intelligently handles both
    digitally-native and scanned PDFs. It is built on four pillars:
    1.  **Multi-Modal Extraction:** Uses direct text extraction for digital PDFs
        and automatically falls back to Optical Character Recognition (OCR) for
        scanned or image-based pages.
    2.  **Granular Loading:** Can load a PDF as one document per page (default) or
        as a single document for the entire file.
    3.  **Deep Metadata Archaeology:** Extracts both filesystem metadata and the
        PDF's internal document information dictionary (e.g., author, title).
    4.  **Intelligent Text Cleaning:** Includes a post-processing step to fix
        common PDF text extraction artifacts like unwanted hyphenation and line breaks.
    """
    def __init__(
        self,
        config: Optional[PDFLoaderConfig] = None,
    ):
        """
        Initializes the PDFLoader.

        Args:
            config: PDFLoaderConfig object with all settings
        """
        if config is None:
            config = PDFLoaderConfig()
        
        super().__init__(config)
        
        if config.use_ocr:
            if not OCR_AVAILABLE:
                raise ImportError(
                    "OCR dependencies are not installed. Please run 'pip install pytesseract' "
                    "and ensure you have a system-level Tesseract installation."
                )
            if not PDF2IMAGE_AVAILABLE:
                raise ImportError(
                    "pdf2image is not installed. Please run 'pip install pdf2image' "
                    "and ensure you have poppler-utils installed."
                )
        
        self.config = config

    def load(self, source: str) -> List[Document]:
        """
        Loads a PDF file, applying the configured extraction and loading strategy.
        """
        try:
            file_path = os.path.abspath(source)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Source path '{source}' is not a valid file.")

            stats = os.stat(file_path)
            base_metadata: Dict[str, Any] = {
                "source": source, "file_name": os.path.basename(file_path),
                "file_path": file_path, "file_size": stats.st_size,
                "creation_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "last_modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            }

            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                
                doc_info = reader.metadata
                if doc_info:
                    for key, value in doc_info.items():
                        clean_key = key[1:].lower() if key.startswith('/') else key.lower()
                        base_metadata[clean_key] = value

                documents: List[Document] = []
                all_page_texts: List[str] = []

                for i, page in enumerate(reader.pages):
                    page_number = i + 1
                    page_metadata = base_metadata.copy()
                    page_metadata["page_number"] = page_number
                    
                    text = page.extract_text()
                    extraction_method = "direct"

                    should_use_ocr = (
                        self.config.use_ocr and 
                        (self.config.force_ocr or not text or len(text.strip()) < self.config.ocr_text_threshold)
                    )
                    
                    if should_use_ocr:
                        try:
                            page_images = convert_from_path(
                                file_path, 
                                first_page=page_number, 
                                last_page=page_number,
                                dpi=self.config.ocr_dpi
                            )
                            
                            if page_images:
                                page_image = page_images[0]
                                ocr_text = pytesseract.image_to_string(
                                    page_image, 
                                    lang=self.config.ocr_language
                                )
                                
                                if ocr_text.strip():
                                    text = ocr_text
                                    extraction_method = "ocr"
                                    
                        except Exception as ocr_error:
                            print(f"Warning: OCR failed on page {page_number} of '{source}': {ocr_error}")
                    
                    page_metadata["extraction_method"] = extraction_method
                    
                    cleaned_text = self._clean_text(text)

                    if self.config.load_strategy == "one_document_per_page":
                        documents.append(Document(content=cleaned_text, metadata=page_metadata))
                    else:
                        all_page_texts.append(cleaned_text)

                if self.config.load_strategy == "one_document_for_the_whole_file":
                    full_content = "\n\n".join(all_page_texts)
                    documents.append(Document(content=full_content, metadata=base_metadata))
                
                return documents

        except FileNotFoundError as e:
            print(f"Error: [PDFLoader] File not found at path: {e}")
            return []
        except Exception as e:
            print(f"Error: [PDFLoader] An unexpected error occurred while loading '{source}': {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """Applies a series of cleaning steps to raw extracted PDF text."""
        if not text:
            return ""
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of file extensions supported by this loader."""
        return [".pdf"]