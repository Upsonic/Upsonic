import asyncio
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
import concurrent.futures

from upsonic.schemas.data_models import Document
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import PdfLoaderConfig

try:
    from pypdf import PdfReader, PageObject
    _PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None
    PageObject = None
    _PYPDF_AVAILABLE = False


try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_ENGINE = RapidOCR()
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    RapidOCR = None
    OCR_ENGINE = None
    _RAPIDOCR_AVAILABLE = False




class PdfLoader(BaseLoader):
    """
    A high-performance, comprehensive loader for PDF documents.

    This loader supports digital text extraction, OCR for scanned images,
    and a wide range of content structuring and cleaning options driven by
    the PdfLoaderConfig. It is built with an async-first design to efficiently
    process large numbers of documents concurrently.
    """

    def __init__(self, config: Optional[PdfLoaderConfig] = None):
        """
        Initializes the PdfLoader with its specific configuration.

        Args:
            config: A PdfLoaderConfig object with settings for PDF processing.
        """
        if config is None:
            config = PdfLoaderConfig()
        if not _PYPDF_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pypdf",
                install_command='pip install "upsonic[loaders]"',
                feature_name="PDF loader"
            )
        if config.extraction_mode in ("ocr_only", "hybrid") and not _RAPIDOCR_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="rapidocr-onnxruntime",
                install_command='pip install "upsonic[loaders]"',
                feature_name="PDF OCR functionality"
            )
        super().__init__(config)
        self.config: PdfLoaderConfig = config


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
            
            reader = await asyncio.to_thread(PdfReader, str(path))

            if reader.is_encrypted:
                if not self.config.pdf_password:
                    raise PermissionError(f"PDF {path} is encrypted but no password provided.")
                decrypt_result = await asyncio.to_thread(reader.decrypt, self.config.pdf_password)
                if decrypt_result == 0:
                    raise PermissionError(f"Could not decrypt {path}. Invalid password provided.")
            
            start_idx = (self.config.start_page - 1) if self.config.start_page else 0
            end_idx = self.config.end_page if self.config.end_page else len(reader.pages)
            pages_to_process = reader.pages[start_idx:end_idx]

            page_tasks = [self._extract_page_content(page, page_num=start_idx + i + 1) for i, page in enumerate(pages_to_process)]
            page_contents_with_nums = await asyncio.gather(*page_tasks)
            
            page_contents = [content for content, num in page_contents_with_nums]
            page_numbers = [num for content, num in page_contents_with_nums]

            if self.config.extra_whitespace_removal:
                page_contents = [self._normalize_whitespace(content) for content in page_contents]

            if self.config.clean_page_numbers:
                page_contents, _ = self._clean_page_numbers(page_contents)
            
            full_content = "\n\n".join(page_contents).strip()

            if self.config.skip_empty_content and not full_content:
                return []
            metadata = self._create_metadata(path)
            metadata["page_count"] = len(page_contents)
            if self.config.include_metadata:
                
                return [Document(document_id=document_id, content=full_content, metadata=metadata)]
            else:
                return [Document(document_id=document_id, content=full_content)]

        except (PermissionError, FileExistsError) as e:

            raise e
        except Exception as e:
            return self._handle_loading_error(str(path), e)


    async def _extract_page_content(self, page: PageObject, page_num: int) -> Tuple[str, int]:
        """
        Extracts content from a single page based on the `extraction_mode`.
        """
        text = ""
        ocr_text = ""

        if self.config.extraction_mode in ("text_only", "hybrid"):
            text = await asyncio.to_thread(page.extract_text) or ""

        if self.config.extraction_mode in ("ocr_only", "hybrid"):
            ocr_text = await self._perform_ocr(page)
        
        full_content = f"{text}\n\n{ocr_text}".strip()
        return full_content, page_num



    async def _perform_ocr(self, page: PageObject) -> str:
        """
        Performs OCR on all images within a single PDF page.
        """
        if not page.images:
            return ""

        image_data_list = [img.data for img in page.images]

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            ocr_tasks = [
                loop.run_in_executor(pool, self._run_single_ocr, img_data)
                for img_data in image_data_list
            ]
            ocr_results = await asyncio.gather(*ocr_tasks)
        
        return "\n".join(filter(None, ocr_results))

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
        Adapted from the provided example to use instance configuration.
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