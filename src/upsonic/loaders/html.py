from __future__ import annotations
from typing import List, Optional, Dict, Any
import os
import re
import time
from datetime import datetime
from pathlib import Path

from .base import DocumentLoader
from ..schemas.data_models import Document
from .config import HTMLLoaderConfig

try:
    from bs4 import BeautifulSoup
    import requests
except ImportError:
    BeautifulSoup = None
    requests = None




class HTMLLoader(DocumentLoader):
    """
    A master-class, structure-aware loader for HTML files and web pages.

    This loader intelligently processes HTML content while preserving meaningful
    structure and extracting relevant information. It supports both local HTML files
    and web URLs, with configurable extraction options.

    Key features include:
    - Intelligent text extraction while preserving structure
    - Metadata extraction from HTML head elements
    - Table processing and formatting
    - Link and image extraction
    - Configurable content filtering
    - Web page fetching with user-agent support
    - Clean text output with optional structure preservation
    """

    def __init__(self, config: Optional[HTMLLoaderConfig] = None):
        """Initialize HTMLLoader with configuration."""
        super().__init__(config)
        self.config = config or HTMLLoaderConfig()
        
        if BeautifulSoup is None:
            raise ImportError(
                "beautifulsoup4 is not installed. It is required for the HTMLLoader. "
                "Please run: 'pip install beautifulsoup4'"
            )

    def load(self, source: str) -> List[Document]:
        """
        Loads HTML content from a file or URL.

        Args:
            source: File path or URL to HTML content

        Returns:
            List containing a single Document object with processed HTML content
        """
        return self._load_with_error_handling(source)

    def _load_html_content(self, source: str) -> List[Document]:
        """
        Internal method to load HTML content with proper error handling.
        """
        try:
            if source.startswith(('http://', 'https://')):
                html_content, base_metadata = self._load_from_url(source)
            else:
                html_content, base_metadata = self._load_from_file(source)

            if not html_content:
                return []

            soup = BeautifulSoup(html_content, 'html.parser')

            if self.config.remove_scripts:
                for script in soup(["script"]):
                    script.decompose()
            
            if self.config.remove_styles:
                for style in soup(["style"]):
                    style.decompose()

            html_metadata = {}
            if self.config.extract_metadata:
                html_metadata = self._extract_html_metadata(soup)
            
            metadata = {**base_metadata, **html_metadata}

            if self.config.extract_text:
                content = self._extract_structured_content(soup)
            else:
                content = str(soup)

            if self.config.clean_whitespace:
                content = self._clean_whitespace(content)

            if self.config.custom_metadata:
                metadata.update(self.config.custom_metadata)

            document = Document(content=content, metadata=metadata)
            return [document]

        except Exception as e:
            return self._handle_error(f"Error loading HTML from '{source}': {e}", e)

    def _load_with_error_handling(self, source: str) -> List[Document]:
        """Override to use our internal HTML loading method."""
        start_time = time.time()
        
        try:
            if not self._validate_source(source):
                raise ValueError(f"Invalid source: {source}")
            
            if self.config and self.config.max_file_size:
                if os.path.exists(source):
                    file_size = os.path.getsize(source)
                    if file_size > self.config.max_file_size:
                        raise ValueError(f"File size {file_size} exceeds limit {self.config.max_file_size}")
            
            documents = self._load_html_content(source)
            
            documents = self._post_process_documents(documents, source)
            
            processing_time = time.time() - start_time
            self._update_stats(len(documents), processing_time, success=True)
            
            return documents
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(0, processing_time, success=False)
            
            if self.config:
                if self.config.error_handling == "ignore":
                    return []
                elif self.config.error_handling == "warn":
                    print(f"Warning: Failed to load {source}: {e}")
                    return []
                else:
                    raise
            else:
                raise

    def _load_from_file(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Load HTML content from a local file."""
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Source path '{file_path}' is not a valid file.")

        stats = os.stat(file_path)
        metadata = {
            "source": file_path,
            "source_type": "file",
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size": stats.st_size,
            "creation_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "last_modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        }

        encoding = self.config.encoding or "utf-8"
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            metadata["detected_encoding"] = encoding
        except UnicodeDecodeError:
            for fallback_encoding in ["latin1", "cp1252", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=fallback_encoding) as f:
                        content = f.read()
                    metadata["detected_encoding"] = fallback_encoding
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError(f"Could not decode file '{file_path}' with any standard encoding")

        return content, metadata

    def _load_from_url(self, url: str) -> tuple[str, Dict[str, Any]]:
        """Load HTML content from a web URL."""
        if requests is None:
            raise ImportError(
                "requests is not installed. It is required for loading URLs. "
                "Please run: 'pip install requests'"
            )

        headers = {"User-Agent": self.config.user_agent}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            metadata = {
                "source": url,
                "source_type": "url",
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "encoding": response.encoding or "utf-8",
                "url": url,
                "final_url": response.url,
                "fetch_time": datetime.now().isoformat(),
            }
            
            return response.text, metadata
            
        except Exception as e:
            raise ConnectionError(f"Failed to fetch URL '{url}': {e}")

    def _extract_html_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML head elements."""
        metadata = {}
        
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            if name and content:
                metadata[f'meta_{name}'] = content
        
        description_meta = soup.find('meta', attrs={'name': 'description'})
        if description_meta:
            metadata['description'] = description_meta.get('content')
        
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_meta:
            metadata['keywords'] = keywords_meta.get('content')
        
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            metadata['author'] = author_meta.get('content')
        
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')
        
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                headings.append({
                    'level': level,
                    'text': heading.get_text().strip()
                })
        if headings:
            metadata['headings'] = headings[:10]
        
        metadata['element_counts'] = {
            'paragraphs': len(soup.find_all('p')),
            'headings': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
            'links': len(soup.find_all('a')),
            'images': len(soup.find_all('img')),
            'tables': len(soup.find_all('table')),
            'lists': len(soup.find_all(['ul', 'ol']))
        }
        
        return metadata

    def _extract_structured_content(self, soup: BeautifulSoup) -> str:
        """Extract content while preserving meaningful structure."""
        content_parts = []
        
        title = soup.find('title')
        if title:
            content_parts.append(f"Title: {title.get_text().strip()}")
        
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_=re.compile(r'content|main|body', re.I)) or
            soup.find('body') or
            soup
        )
        
        if self.config.extract_headers:
            content_parts.extend(self._extract_headings(main_content))
        
        if self.config.extract_paragraphs:
            content_parts.extend(self._extract_paragraphs(main_content))
        
        if self.config.extract_lists:
            content_parts.extend(self._extract_lists(main_content))
        
        if self.config.extract_tables:
            content_parts.extend(self._extract_tables(main_content))
        
        if not any([self.config.extract_headers, self.config.extract_paragraphs, 
                   self.config.extract_lists, self.config.extract_tables]):
            content_parts.append(main_content.get_text())
        
        content = "\n\n".join(part for part in content_parts if part and part.strip())
        
        if self.config.include_links:
            links = self._extract_links(soup)
            if links:
                content += f"\n\nLinks:\n{links}"
        
        if self.config.include_images:
            images = self._extract_images(soup)
            if images:
                content += f"\n\nImages:\n{images}"
        
        return content

    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract and format headings."""
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                text = heading.get_text().strip()
                if text:
                    if self.config.preserve_structure:
                        prefix = "#" * level
                        headings.append(f"{prefix} {text}")
                    else:
                        headings.append(text)
        return headings

    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract paragraph content."""
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if text:
                paragraphs.append(text)
        return paragraphs

    def _extract_lists(self, soup: BeautifulSoup) -> List[str]:
        """Extract list content."""
        lists = []
        for list_tag in soup.find_all(['ul', 'ol']):
            items = []
            for li in list_tag.find_all('li'):
                text = li.get_text().strip()
                if text:
                    if self.config.preserve_structure:
                        prefix = "- " if list_tag.name == 'ul' else f"{len(items) + 1}. "
                        items.append(f"{prefix}{text}")
                    else:
                        items.append(text)
            if items:
                lists.append("\n".join(items))
        return lists

    def _extract_tables(self, soup: BeautifulSoup) -> List[str]:
        """Extract and format table content."""
        tables = []
        for table in soup.find_all('table'):
            if self.config.table_format == "html":
                tables.append(str(table))
            else:
                table_text = self._parse_table_to_text(table)
                if table_text:
                    tables.append(table_text)
        return tables

    def _parse_table_to_text(self, table) -> str:
        """Convert HTML table to text format."""
        try:
            rows = []
            for tr in table.find_all('tr'):
                cells = []
                for td in tr.find_all(['td', 'th']):
                    cell_text = td.get_text().strip()
                    cells.append(cell_text)
                if cells:
                    if self.config.table_format == "markdown":
                        rows.append(" | ".join(cells))
                        if not rows or len(rows) == 1:
                            rows.append(" | ".join(["---"] * len(cells)))
                    else:
                        rows.append(", ".join(f"{cell}" for cell in cells))
            
            if rows:
                return f"[Table]\n" + "\n".join(rows)
            return ""
        except Exception:
            return "[Unable to parse table]"

    def _extract_links(self, soup: BeautifulSoup) -> str:
        """Extract links from the HTML."""
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text().strip()
            if href and text:
                links.append(f"- {text}: {href}")
        return "\n".join(links[:20])

    def _extract_images(self, soup: BeautifulSoup) -> str:
        """Extract image information."""
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            title = img.get('title', '')
            if src:
                img_info = f"- {src}"
                if alt:
                    img_info += f" (alt: {alt})"
                if title:
                    img_info += f" (title: {title})"
                images.append(img_info)
        return "\n".join(images[:10])

    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace in extracted text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def _handle_error(self, message: str, exception: Exception) -> List[Document]:
        """Handle errors based on configuration."""
        if self.config.error_handling == "ignore":
            return []
        elif self.config.error_handling == "warn":
            print(message)
            return []
        else:
            raise exception

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of file extensions supported by this loader."""
        return ['.html', '.htm', '.xhtml']

    @classmethod
    def can_load(cls, source: str) -> bool:
        """Check if this loader can handle the given source."""
        if not source:
            return False
        
        if source.startswith(('http://', 'https://')):
            return True
        
        extensions = cls.get_supported_extensions()
        if extensions:
            source_ext = Path(source).suffix.lower()
            return source_ext in extensions
        
        return True
