from __future__ import annotations
from typing import Any, List, Dict, Tuple, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from pydantic import Field

from upsonic.text_splitter.base import ChunkingStrategy, ChunkingConfig
from upsonic.text_splitter.recursive import RecursiveCharacterChunkingStrategy
from upsonic.schemas.data_models import Document, Chunk

try:
    from bs4 import BeautifulSoup, Tag
    from bs4.element import Comment
except ImportError:
    raise ImportError(
        "BeautifulSoup4 is not installed. Please install it to use the HTML strategies by running: "
        "'pip install beautifulsoup4'"
    )

logger = logging.getLogger(__name__)


class ChunkingMode(Enum):
    """Enumeration of different chunking approaches."""
    TEXT_ONLY = "text_only"
    HEADER_BASED = "header_based"
    SEMANTIC_PRESERVING = "semantic_preserving"
    ADAPTIVE = "adaptive"


class ContentType(Enum):
    """Enumeration of different content types found in HTML."""
    TEXT = "text"
    CODE = "code"
    LIST = "list"
    TABLE = "table"
    QUOTE = "quote"
    MEDIA = "media"
    NAVIGATION = "navigation"
    HEADER = "header"
    FOOTER = "footer"
    LINK = "link"


@dataclass
class HTMLElement:
    """Represents a parsed HTML element with metadata."""
    tag: str
    text: str
    attributes: Dict[str, str] = field(default_factory=dict)
    content_type: ContentType = ContentType.TEXT
    depth: int = 0
    parent_chain: List[str] = field(default_factory=list)
    word_count: int = 0
    dom_depth: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.text.split()) if self.text else 0


class HTMLChunkingConfig(ChunkingConfig):
    """Configuration for HTML chunking strategy."""
    
    mode: ChunkingMode = Field(ChunkingMode.ADAPTIVE, description="Chunking mode")
    max_chunk_size: int = Field(2000, description="Maximum characters per chunk")
    min_chunk_size: int = Field(100, description="Minimum characters per chunk")
    chunk_overlap: int = Field(50, description="Character overlap between chunks")
    return_each_element: bool = Field(False, description="Whether to return each HTML element as separate chunk")
    
    preserve_formatting: bool = Field(True, description="Whether to preserve basic formatting")
    preserve_links: bool = Field(True, description="Whether to convert links to markdown format")
    preserve_images: bool = Field(True, description="Whether to convert images to markdown format")
    preserve_videos: bool = Field(False, description="Whether to convert videos to markdown format")
    preserve_audio: bool = Field(False, description="Whether to convert audio to markdown format")
    preserve_hierarchy: bool = Field(True, description="Whether to maintain hierarchical context")
    include_element_metadata: bool = Field(True, description="Whether to include detailed element metadata")
    normalize_text: bool = Field(False, description="Whether to normalize text")
    
    skip_elements: Set[str] = Field(default_factory=lambda: {'script', 'style', 'noscript', 'template'}, description="Elements to skip during extraction")
    elements_to_preserve: Set[str] = Field(default_factory=lambda: {'table', 'ul', 'ol', 'pre', 'code', 'blockquote'}, description="Elements to keep intact during splitting")
    allowlist_tags: Optional[List[str]] = Field(None, description="Only these tags will be retained")
    denylist_tags: Optional[List[str]] = Field(None, description="These tags will be removed")
    
    headers_to_split_on: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            ("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"),
            ("h4", "Header 4"), ("h5", "Header 5"), ("h6", "Header 6"),
        ],
        description="Header tags and their metadata keys"
    )
    
    custom_separators: Dict[str, str] = Field(default_factory=dict, description="Custom separators for different elements")
    custom_handlers: Dict[str, Callable[[Any], str]] = Field(default_factory=dict, description="Custom handlers for specific HTML tags")
    content_type_weights: Dict[ContentType, float] = Field(default_factory=dict, description="Weights for different content types")
    
    optimization_target: str = Field("balanced", description="Optimization focus for adaptive mode: speed, accuracy, balanced")
    analysis_threshold: int = Field(500, description="Minimum content length for advanced analysis")
    external_metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata to attach")
    keep_separator: Union[bool, str] = Field(True, description="Whether to keep separators in chunks")


class HTMLChunkingStrategy(ChunkingStrategy):
    """
    HTML chunking strategy that processes HTML documents
    with multiple modes and comprehensive content preservation capabilities.
    
    Inspired by LangChain's HTML splitters but adapted for the Upsonic framework.
    """
    
    SEMANTIC_ELEMENTS = {
        'article', 'section', 'aside', 'nav', 'header', 'footer', 'main',
        'figure', 'figcaption', 'details', 'summary', 'mark', 'time'
    }
    
    CODE_ELEMENTS = {'code', 'pre', 'kbd', 'samp', 'var'}
    LIST_ELEMENTS = {'ul', 'ol', 'dl', 'li', 'dt', 'dd'}
    TABLE_ELEMENTS = {'table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th', 'caption'}
    MEDIA_ELEMENTS = {'img', 'video', 'audio', 'canvas', 'svg', 'picture', 'source'}
    QUOTE_ELEMENTS = {'blockquote', 'cite', 'q'}
    SKIP_ELEMENTS = {'script', 'style', 'noscript', 'template'}
    ELEMENTS_TO_PRESERVE = {'table', 'ul', 'ol', 'pre', 'code', 'blockquote'}
    
    def __init__(self, config: Optional[HTMLChunkingConfig] = None):
        """
        Initialize the HTML chunking strategy with comprehensive configuration.
        
        Args:
            config: Configuration object with all settings
        """
        if config is None:
            config = HTMLChunkingConfig()
        
        super().__init__(config)
        
        self.text_chunker = RecursiveCharacterChunkingStrategy()
        
        self.headers_to_split_on = sorted(self.config.headers_to_split_on, key=lambda x: int(x[0][1:]))
        self.header_mapping = dict(self.headers_to_split_on)
        self.header_tags = [tag for tag, _ in self.headers_to_split_on]
        
        self.skip_elements = self.config.skip_elements | self.SKIP_ELEMENTS
        self.elements_to_preserve = self.config.elements_to_preserve | self.ELEMENTS_TO_PRESERVE
        self.allowlist_tags = self.config.allowlist_tags
        self.denylist_tags = self.config.denylist_tags
        
        if self.denylist_tags:
            self.denylist_tags = [
                tag for tag in self.denylist_tags
                if tag not in [header[0] for header in self.headers_to_split_on]
            ]
        
        self.separators = {
            'paragraph': '\n\n',
            'break': '\n',
            'list_item': '\n• ',
            'table_cell': ' | ',
            'table_row': '\n',
            'header': '\n\n',
            **self.config.custom_separators
        }
        
        self.custom_handlers = self.config.custom_handlers
        
        self.content_type_weights = self.config.content_type_weights or {
            ContentType.HEADER: 1.0,
            ContentType.TEXT: 0.8,
            ContentType.LIST: 0.6,
            ContentType.TABLE: 0.9,
            ContentType.CODE: 1.0,
            ContentType.QUOTE: 0.7,
        }
        
        self.external_metadata = self.config.external_metadata
        self.keep_separator = self.config.keep_separator

    def _classify_content(self, element: Tag) -> ContentType:
        """Classify the content type of an HTML element."""
        tag_name = element.name.lower()
        
        if tag_name in self.CODE_ELEMENTS:
            return ContentType.CODE
        elif tag_name in self.LIST_ELEMENTS:
            return ContentType.LIST
        elif tag_name in self.TABLE_ELEMENTS:
            return ContentType.TABLE
        elif tag_name in self.QUOTE_ELEMENTS:
            return ContentType.QUOTE
        elif tag_name in self.MEDIA_ELEMENTS:
            return ContentType.MEDIA
        elif tag_name == 'nav':
            return ContentType.NAVIGATION
        elif tag_name in ['header', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return ContentType.HEADER
        elif tag_name == 'footer':
            return ContentType.FOOTER
        elif tag_name == 'a':
            return ContentType.LINK
        else:
            return ContentType.TEXT

    def _process_media_elements(self, soup: BeautifulSoup) -> None:
        """Process media elements by converting them to markdown format."""
        if self.config.preserve_images:
            for img_tag in soup.find_all("img"):
                img_src = img_tag.get("src", "")
                img_alt = img_tag.get("alt", "image")
                markdown_img = f"![{img_alt}]({img_src})"
                wrapper = soup.new_tag("media-wrapper")
                wrapper.string = markdown_img
                img_tag.replace_with(wrapper)
        else:
            for img_tag in soup.find_all("img"):
                alt_text = img_tag.get("alt", "")
                img_tag.replace_with(alt_text)
                
        if self.config.preserve_videos:
            for video_tag in soup.find_all("video"):
                video_src = video_tag.get("src", "")
                markdown_video = f"![video]({video_src})"
                wrapper = soup.new_tag("media-wrapper")
                wrapper.string = markdown_video
                video_tag.replace_with(wrapper)
        else:
            for video_tag in soup.find_all("video"):
                video_tag.decompose()

        if self.config.preserve_audio:
            for audio_tag in soup.find_all("audio"):
                audio_src = audio_tag.get("src", "")
                markdown_audio = f"![audio]({audio_src})"
                wrapper = soup.new_tag("media-wrapper")
                wrapper.string = markdown_audio
                audio_tag.replace_with(wrapper)
        else:
            for audio_tag in soup.find_all("audio"):
                audio_tag.decompose()

    def _process_links(self, soup: BeautifulSoup) -> None:
        """Process links by converting them to markdown format."""
        if self.config.preserve_links:
            for a_tag in soup.find_all("a"):
                a_href = a_tag.get("href", "")
                a_text = a_tag.get_text(strip=True)
                markdown_link = f"[{a_text}]({a_href})"
                a_tag.replace_with(markdown_link)

    def _filter_tags(self, soup: BeautifulSoup) -> None:
            """Filter HTML content based on allowlist and denylist tags."""
            if self.allowlist_tags:
                effective_allowlist = set(self.allowlist_tags)
                
                structural_tags = {'html', 'head', 'body'}
                changed = True
                while changed:
                    changed = False
                    for tag in soup.find_all(True):
                        if tag.name not in effective_allowlist and tag.name not in structural_tags:
                            tag.decompose()
                            changed = True
                            break

            if self.denylist_tags:
                for tag in soup.find_all(self.denylist_tags):
                    tag.decompose()

    def _normalize_and_clean_text(self, text: str) -> str:
        """Normalize and clean text content."""
        if self.config.normalize_text:
            text = text.lower()
        
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text

    def _get_element_text(self, element: Any) -> str:
        """Extract text from an element, preserving structure for certain tags."""
        if element.name in self.custom_handlers:
            return self.custom_handlers[element.name](element)

        if element.name in self.SKIP_ELEMENTS:
            return ""

        if element.name in ['pre', 'code']:
            return element.get_text()

        if element.name == 'li':
            return self.separators.get('list_item', '\n• ') + self._get_inner_text(element)

        if element.name == 'table':
            rows = []
            for row in element.find_all('tr'):
                cells = [self._get_inner_text(cell) for cell in row.find_all(['td', 'th'])]
                rows.append(self.separators.get('table_cell', ' | ').join(cells))
            return self.separators.get('table_row', '\n').join(rows)

        if element.name in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'section', 'article']:
            return self.separators.get('paragraph', '\n\n') + self._get_inner_text(element)
        
        return self._get_inner_text(element)

    def _get_inner_text(self, element: Any) -> str:
        """A helper to recursively get text from children."""
        text = ""
        for child in element.children:
            if isinstance(child, str):
                text += child
            elif hasattr(child, 'name'):
                text += self._get_element_text(child)
        
        return self._normalize_and_clean_text(text)

    def _analyze_document_complexity(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze document to determine optimal chunking approach."""
        analysis = {
            'total_elements': len(soup.find_all(True)),
            'text_length': len(soup.get_text()),
            'header_count': len(soup.find_all(self.header_tags)),
            'semantic_elements': len(soup.find_all(list(self.SEMANTIC_ELEMENTS))),
            'table_count': len(soup.find_all('table')),
            'list_count': len(soup.find_all(['ul', 'ol'])),
            'code_blocks': len(soup.find_all(['pre', 'code'])),
            'link_count': len(soup.find_all('a')),
            'complexity_score': 0.0
        }
        
        if analysis['text_length'] > 0:
            analysis['complexity_score'] = (
                (analysis['semantic_elements'] / max(1, analysis['total_elements'])) * 0.3 +
                (analysis['header_count'] / max(1, analysis['text_length'] / 1000)) * 0.3 +
                (analysis['table_count'] * 0.2) +
                (analysis['code_blocks'] * 0.2)
            )
        
        return analysis

    def _select_chunking_approach(self, analysis: Dict[str, Any]) -> ChunkingMode:
        """Select the best chunking approach based on document analysis."""
        if analysis['table_count'] > 0 or analysis['code_blocks'] > 0 or analysis['list_count'] > 2:
            return ChunkingMode.SEMANTIC_PRESERVING
        
        if analysis['header_count'] > 1:
            return ChunkingMode.HEADER_BASED

        return ChunkingMode.TEXT_ONLY


    def _chunk_text_only(self, soup: BeautifulSoup, document: Document) -> List[Chunk]:
        """Execute simple text extraction and chunking."""
        for element in soup(list(self.skip_elements)):
            element.decompose()
        
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        if self.allowlist_tags:
            text_parts = []
            for tag_name in self.allowlist_tags:
                for tag in soup.find_all(tag_name):
                    text_parts.append(tag.get_text(strip=True))
            clean_text = " ".join(text_parts)
        else:
            clean_text = soup.get_text(separator=' ', strip=True)
        
        clean_text = self._normalize_and_clean_text(clean_text)
        
        if not clean_text:
            return []
        
        enhanced_metadata = document.metadata.copy()
        enhanced_metadata.update(self.external_metadata)
        enhanced_metadata['extraction_method'] = 'text_only'
        
        text_document = Document(
            content=clean_text,
            metadata=enhanced_metadata,
            document_id=document.document_id
        )
        
        text_chunks = self.text_chunker.chunk(text_document)
        
        chunks = []
        for i, text_chunk in enumerate(text_chunks):
            chunk = Chunk(
                text_content=text_chunk.text_content,
                metadata=text_chunk.metadata,
                document_id=document.document_id
            )
            chunk.metadata['chunk_index'] = i
            chunks.append(chunk)
        
        return chunks

    def _chunk_header_based(self, soup: BeautifulSoup, document: Document) -> List[Chunk]:
            """
            Execute header-based chunking by splitting the document by header tags.
            This is the definitive, corrected version.
            """
            chunks = []
            body = soup.body if soup.body else soup
            if not body:
                return []

            if self.config.return_each_element:
                block_tags = self.header_tags + ['p', 'li', 'td', 'th', 'div']
                current_headers = {}
                
                for element in body.find_all(block_tags):
                    text = self._normalize_and_clean_text(element.get_text(strip=True))
                    if text:
                        if element.name in self.header_tags:
                            level = int(element.name[1:])
                            headers_to_remove = [k for k, (_, lvl) in current_headers.items() if lvl >= level]
                            for k in headers_to_remove:
                                del current_headers[k]
                            current_headers[self.header_mapping[element.name]] = (text, level)
                        
                        metadata = document.metadata.copy()
                        metadata.update(self.external_metadata)
                        metadata['element_tag'] = element.name
                        metadata.update({k: v[0] for k, v in current_headers.items()})
                        chunks.append(Chunk(text_content=text, metadata=metadata, document_id=document.document_id))
                return chunks

            current_headers = {}
            
            split_points = body.find_all(self.header_tags)
            if not split_points:
                full_text = self._normalize_and_clean_text(body.get_text(strip=True))
                if full_text:
                    chunks.append(Chunk(text_content=full_text, metadata=document.metadata, document_id=document.document_id))
                return chunks

            for i, header in enumerate(split_points):
                content_elements = []
                current_level = int(header.name[1:])
                
                current_element = header.next_sibling
                while current_element:
                    if hasattr(current_element, 'name') and current_element.name in self.header_tags:
                        sibling_level = int(current_element.name[1:])
                        if sibling_level <= current_level:
                            break
                    elif hasattr(current_element, 'name') and hasattr(current_element, 'find_all'):
                        nested_headers = current_element.find_all(self.header_tags, recursive=True)
                        if not nested_headers:
                            content_elements.append(current_element)
                    elif isinstance(current_element, str):
                        if current_element.strip():
                            content_elements.append(current_element)
                    current_element = current_element.next_sibling
                
                content_parts = []
                for elem in content_elements:
                    if isinstance(elem, str):
                        content_parts.append(elem.strip())
                    else:
                        content_parts.append(elem.get_text(strip=True))
                
                content_text = self._normalize_and_clean_text(" ".join(content_parts))

                header_text = self._normalize_and_clean_text(header.get_text(strip=True))
                level = int(header.name[1:])

                headers_to_remove = [k for k, (_, lvl) in current_headers.items() if lvl >= level]
                for k in headers_to_remove:
                    del current_headers[k]
                current_headers[self.header_mapping[header.name]] = (header_text, level)

                final_text = f"{header_text}\n\n{content_text}".strip()
                metadata = document.metadata.copy()
                metadata.update(self.external_metadata)
                metadata.update({k: v[0] for k, v in current_headers.items()})
                chunks.append(Chunk(text_content=final_text, metadata=metadata, document_id=document.document_id))
                
            return chunks

    def _chunk_semantic_preserving(self, soup: BeautifulSoup, document: Document) -> List[Chunk]:
        """
        Execute semantic-preserving chunking by traversing the entire tree
        and isolating preserved elements. This is the definitive, corrected version.
        """
        chunks = []
        current_text_group = []
        body = soup.body if soup.body else soup
        if not body:
            return []

        def finalize_text_group():
            """Chunks any accumulated regular text."""
            if not current_text_group:
                return
            full_text = self._normalize_and_clean_text(" ".join(current_text_group))
            current_text_group.clear()
            if full_text:
                if len(full_text) > self.config.max_chunk_size:
                    text_doc = Document(content=full_text, metadata=document.metadata, document_id=document.document_id)
                    text_chunks = self.text_chunker.chunk(text_doc)
                    for text_chunk in text_chunks:
                        metadata = document.metadata.copy()
                        metadata.update(self.external_metadata)
                        chunks.append(Chunk(text_content=text_chunk.text_content, metadata=metadata, document_id=document.document_id))
                else:
                    metadata = document.metadata.copy()
                    metadata.update(self.external_metadata)
                    chunks.append(Chunk(text_content=full_text, metadata=metadata, document_id=document.document_id))

        for element in body.find_all(True):
            if any(p.name in self.elements_to_preserve for p in element.find_parents()):
                continue

            if element.name in self.elements_to_preserve:
                finalize_text_group()
                
                if element.name == 'table':
                    caption_text = ""
                    caption = element.find('caption')
                    if caption:
                        caption_text = caption.get_text(strip=True) + "\n"
                    
                    rows = []
                    for row in element.find_all('tr'):
                        cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                        rows.append(' | '.join(cells))
                    element_text = caption_text + '\n'.join(rows)
                elif element.name in self.CODE_ELEMENTS:
                    element_text = element.get_text()
                else:
                    element_text = self._normalize_and_clean_text(element.get_text(strip=True))

                if element_text:
                    metadata = document.metadata.copy()
                    metadata.update(self.external_metadata)
                    chunks.append(Chunk(text_content=element_text, metadata=metadata, document_id=document.document_id))
            
            else:
                direct_text = " ".join(element.find_all(string=True, recursive=False)).strip()
                if direct_text:
                    current_text_group.append(direct_text)
        
        finalize_text_group()
        return chunks

    def _process_semantic_element(self, elem, chunks, current_headers, current_content, 
                                preserved_elements, placeholder_count, create_documents_func):
        """Process a single semantic element."""
        if elem.name in self.header_tags:
            if current_content:
                final_chunks = create_documents_func(
                    current_headers,
                    " ".join(current_content),
                    preserved_elements
                )
                chunks.extend(final_chunks)
                current_content.clear()
                preserved_elements.clear()
            
            header_name = elem.get_text(strip=True)
            current_headers = {
                self.header_mapping[elem.name]: header_name
            }
        
        elif elem.name in self.elements_to_preserve:
            placeholder = f"PRESERVED_{placeholder_count}"
            preserved_elements[placeholder] = self._get_element_text(elem)
            current_content.append(placeholder)
            placeholder_count += 1
        
        else:
            content = self._get_element_text(elem)
            if content:
                current_content.append(content)

    def _reinsert_preserved_elements(self, content: str, preserved_elements: Dict[str, str]) -> str:
        """Reinsert preserved elements into content."""
        for placeholder, preserved_content in preserved_elements.items():
            content = content.replace(placeholder, preserved_content.strip())
        return content

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Chunk HTML document using the configured approach.
        
        Args:
            document: The HTML document to chunk
            
        Returns:
            List[Chunk]: List of processed chunks with metadata
        """
        try:
            soup = BeautifulSoup(document.content, "html.parser")
            
            self._process_media_elements(soup)
            self._process_links(soup)
            
            if self.allowlist_tags or self.denylist_tags:
                self._filter_tags(soup)
            
            processing_mode = self.config.mode
            analysis = None
            
            if self.config.mode == ChunkingMode.ADAPTIVE:
                analysis = self._analyze_document_complexity(soup)
                processing_mode = self._select_chunking_approach(analysis)
                logger.info(f"Adaptive mode selected {processing_mode.value} for document {document.document_id}")
            
            if processing_mode == ChunkingMode.TEXT_ONLY:
                chunks = self._chunk_text_only(soup, document)
            elif processing_mode == ChunkingMode.HEADER_BASED:
                chunks = self._chunk_header_based(soup, document)
            elif processing_mode == ChunkingMode.SEMANTIC_PRESERVING:
                chunks = self._chunk_semantic_preserving(soup, document)
            else:
                chunks = self._chunk_text_only(soup, document)
            
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_type'] = f'html_{processing_mode.value}'
                chunk.metadata['chunking_strategy'] = 'HTMLChunkingStrategy'
                if 'chunk_index' not in chunk.metadata:
                    chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(chunks)
                chunk.metadata['processing_mode'] = processing_mode.value
                
                if analysis:
                    chunk.metadata['document_analysis'] = analysis
                    chunk.metadata['optimization_target'] = self.config.optimization_target
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing HTML document {document.document_id}: {e}")
            try:
                soup = BeautifulSoup(document.content, "html.parser")
                plain_text = soup.get_text(separator=' ', strip=True)
                if plain_text:
                    text_doc = Document(
                        content=plain_text,
                        metadata=document.metadata,
                        document_id=document.document_id
                    )
                    text_chunks = self.text_chunker.chunk(text_doc)
                    
                    fallback_chunks = []
                    for i, text_chunk in enumerate(text_chunks):
                        chunk = Chunk(
                            text_content=text_chunk.content,
                            metadata=text_chunk.metadata,
                            document_id=document.document_id
                        )
                        chunk.metadata['chunk_type'] = 'html_fallback'
                        chunk.metadata['chunk_index'] = i
                        fallback_chunks.append(chunk)
                    
                    return fallback_chunks
            except:
                pass
            return []