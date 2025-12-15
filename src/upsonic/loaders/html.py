import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    _AIOHTTP_AVAILABLE = False


try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    _REQUESTS_AVAILABLE = False


try:
    from bs4 import BeautifulSoup, Tag
    _BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    Tag = None
    _BS4_AVAILABLE = False


from upsonic.schemas.data_models import Document
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import HTMLLoaderConfig


class HTMLLoader(BaseLoader):
    """
    A loader for HTML files and web URLs.

    This loader can fetch content from the web or read local files. It uses
    BeautifulSoup4 to parse the HTML and provides extensive options for cleaning,
    filtering, and formatting the extracted content.
    """

    def __init__(self, config: Optional[HTMLLoaderConfig] = None):
        """Initializes the HTMLLoader with its specific configuration."""
        if config is None:
            config = HTMLLoaderConfig()
        if not _AIOHTTP_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="aiohttp",
                install_command='pip install "upsonic[loaders]"',
                feature_name="HTML loader"
            )
        if not _REQUESTS_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="requests",
                install_command='pip install "upsonic[loaders]"',
                feature_name="HTML loader"
            )
        if not _BS4_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="beautifulsoup4",
                install_command='pip install "upsonic[loaders]"',
                feature_name="HTML loader"
            )
        super().__init__(config)
        self.config: HTMLLoaderConfig = config

    def _generate_document_id(self, source_identifier: str) -> str:
        """Creates a universal MD5 hash for any source identifier string."""
        return hashlib.md5(source_identifier.encode("utf-8")).hexdigest()

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Gets the list of supported file extensions for local files."""
        return [".html", ".htm", ".xhtml"]
    
    @classmethod
    def can_load(cls, source: Union[str, Path]) -> bool:
        """Checks if this loader can handle a file path or a URL."""
        source_str = str(source)
        if source_str.startswith(("http://", "https://")):
            return True
        return super().can_load(source)

    def _format_table(self, table: Tag) -> str:
        """Formats a BeautifulSoup table object into a string."""
        if self.config.table_format == "html":
            return str(table)
        
        if self.config.table_format == "markdown":
            markdown = []
            header_row = table.find('tr')
            if header_row:
                headers = [cell.get_text(strip=True) for cell in header_row.find_all(['th', 'td'])]
                markdown.append("| " + " | ".join(headers) + " |")
                markdown.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            for row in table.find_all('tr')[1:]:
                row_text = [cell.get_text(strip=True) for cell in row.find_all('td')]
                markdown.append("| " + " | ".join(row_text) + " |")
            return "\n".join(markdown)

        text = []
        for row in table.find_all('tr'):
            row_text = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
            text.append("\t".join(row_text))
        return "\n".join(text)

    def _extract_html_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extracts metadata from the HTML <head> section."""
        metadata = {}
        if title_tag := soup.find("title"):
            metadata["title"] = title_tag.get_text(strip=True)

        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property")
            if name and (content := meta.get("content")):
                metadata[name.lower()] = content
        return metadata

    def _extract_structured_content(self, soup: BeautifulSoup) -> str:
        """Extracts and structures content based on config."""
        content_parts = []
        
        main_content = soup.find("main") or soup.find("article") or soup.body or soup

        element_selectors = []
        if self.config.extract_headers:
            element_selectors.extend([f"h{i}" for i in range(1, 7)])
        if self.config.extract_paragraphs:
            element_selectors.append("p")
        if self.config.extract_lists:
            element_selectors.extend(["ul", "ol"])
        if self.config.extract_tables:
            element_selectors.append("table")

        for element in main_content.find_all(element_selectors, recursive=True):
            if element.name.startswith("h") and self.config.preserve_structure:
                level = int(element.name[1])
                content_parts.append(f"{'#' * level} {element.get_text(strip=True)}")
            elif element.name in ["ul", "ol"]:
                items = [f"- {li.get_text(strip=True)}" for li in element.find_all("li", recursive=False)]
                content_parts.append("\n".join(items))
            elif element.name == "table":
                content_parts.append(self._format_table(element))
            else:
                content_parts.append(element.get_text(strip=True))
        
        full_text = "\n\n".join(part for part in content_parts if part.strip())

        if self.config.include_links:
            links = [f"- [{a.get_text(strip=True)}]: {a.get('href', '')}" for a in soup.find_all("a", href=True)]
            if links:
                full_text += "\n\n--- Links ---\n" + "\n".join(links)
        
        if self.config.include_images:
            images = [f"- Alt: '{img.get('alt', '')}', Src: {img.get('src', '')}" for img in soup.find_all("img", src=True)]
            if images:
                full_text += "\n\n--- Images ---\n" + "\n".join(images)
        
        return full_text

    def _parse_html(self, html_content: str, base_metadata: Dict[str, Any], document_id: str) -> List[Document]:
        """The central parsing engine for HTML content."""
        soup = BeautifulSoup(html_content, "html.parser")

        if self.config.remove_scripts:
            for script in soup.find_all("script"):
                script.decompose()
        if self.config.remove_styles:
            for style in soup.find_all("style"):
                style.decompose()

        metadata = base_metadata
        if self.config.extract_metadata:
            metadata.update(self._extract_html_metadata(soup))

        if self.config.extract_text:
            content = self._extract_structured_content(soup)
        else:
            content = str(soup)
        
        if self.config.clean_whitespace:
            content = re.sub(r"\s+\n", "\n", content)
            content = re.sub(r"\n\n+", "\n\n", content).strip()

        if self.config.skip_empty_content and not content:
            return []
        if self.config.include_metadata:
            doc = Document(document_id=document_id, content=content, metadata=metadata)
        else:
            doc = Document(document_id=document_id, content=content)
        return [doc]
    
    def _load_from_file(self, file_path: Path) -> List[Document]:
        """Loads and parses a single local HTML file."""
        if not self._check_file_size(file_path):
            return []
        try:
            source_identifier = str(file_path.resolve())
            document_id = self._generate_document_id(source_identifier)
            if document_id in self._processed_document_ids:
                raise FileExistsError(f"Source file '{source_identifier}' has already been processed.")
            self._processed_document_ids.add(document_id)
            
            with open(file_path, "r", encoding=self.config.encoding or "utf-8") as f:
                html_content = f.read()
            metadata = self._create_metadata(file_path)
            return self._parse_html(html_content, metadata, document_id)
        except Exception as e:
            return self._handle_loading_error(str(file_path), e)
        
    def _load_from_url(self, url: str) -> List[Document]:
        """Fetches and parses a single URL."""
        try:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL scheme: {url}")
            
            document_id = self._generate_document_id(url)
            if document_id in self._processed_document_ids:
                raise FileExistsError(f"Source URL '{url}' has already been processed.")
            self._processed_document_ids.add(document_id)

            headers = {"User-Agent": self.config.user_agent}
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            metadata = {
                "source": url, "final_url": response.url, "status_code": response.status_code
            }
            return self._parse_html(response.text, metadata, document_id)
        except Exception as e:
            return self._handle_loading_error(url, e)

    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads HTML from file paths, directories, or URLs."""
        sources = [source] if not isinstance(source, list) else source
        
        urls_to_process = [s for s in sources if isinstance(s, str) and s.startswith(("http:", "https:"))]
        paths_to_process = [s for s in sources if s not in urls_to_process]
        
        all_documents = []
        if paths_to_process:
            resolved_files = self._resolve_sources(paths_to_process)
            for file_path in resolved_files:
                all_documents.extend(self._load_from_file(file_path))
        
        for url in urls_to_process:
            all_documents.extend(self._load_from_url(url))
            
        return all_documents
    
    async def _aload_from_file(self, file_path: Path) -> List[Document]:
        """Async: Loads and parses a single local HTML file."""
        return await asyncio.to_thread(self._load_from_file, file_path)

    async def _aload_from_url(self, url: str, session: aiohttp.ClientSession) -> List[Document]:
        """Async: Fetches and parses a single URL."""
        try:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL scheme: {url}")
            
            document_id = self._generate_document_id(url)
            if document_id in self._processed_document_ids:
                raise FileExistsError(f"Source URL '{url}' has already been processed.")
            self._processed_document_ids.add(document_id)

            async with session.get(url, timeout=20) as response:
                response.raise_for_status()
                html_content = await response.text()
                metadata = {
                    "source": url, "final_url": str(response.url), "status_code": response.status
                }
                return await asyncio.to_thread(self._parse_html, html_content, metadata, document_id)
        except Exception as e:
            return self._handle_loading_error(url, e)

    async def aload(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Async: Loads HTML from file paths, directories, or URLs."""
        sources = [source] if not isinstance(source, list) else source
        
        urls = [s for s in sources if isinstance(s, str) and s.startswith(("http:", "https:"))]
        paths = [s for s in sources if s not in urls]
        
        tasks = []
        if paths:
            resolved_files = await asyncio.to_thread(self._resolve_sources, paths)
            for file_path in resolved_files:
                tasks.append(self._aload_from_file(file_path))
        
        if urls:
            headers = {"User-Agent": self.config.user_agent}
            async with aiohttp.ClientSession(headers=headers) as session:
                for url in urls:
                    tasks.append(self._aload_from_url(url, session))
        
        results = await asyncio.gather(*tasks)
        return [doc for sublist in results for doc in sublist]

    def batch(self, sources: List[Union[str, Path]]) -> List[Document]:
        return self.load(sources)

    async def abatch(self, sources: List[Union[str, Path]]) -> List[Document]:
        return await self.aload(sources)