from __future__ import annotations

from typing import List, Optional, Dict, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.text_splitter.base import BaseChunker

from pydantic import Field, ConfigDict
try:
    from bs4 import BeautifulSoup, NavigableString, Tag
    _BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    _BEAUTIFULSOUP_AVAILABLE = False
    class BeautifulSoup:
        pass
    class NavigableString:
        pass
    class Tag:
        pass

from upsonic.schemas.data_models import Chunk, Document
from upsonic.text_splitter.base import BaseChunker, BaseChunkingConfig
from upsonic.text_splitter.recursive import RecursiveChunker, RecursiveChunkingConfig
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)

def get_default_text_chunker() -> BaseChunker:
    """A factory function to create a default text chunker instance."""
    config = RecursiveChunkingConfig(chunk_size=512, chunk_overlap=50)
    return RecursiveChunker(config)


class HTMLChunkingConfig(BaseChunkingConfig):
    """
    A comprehensive configuration model for the structure-aware HTML Chunker.

    This config provides fine-grained control over the entire HTML processing
    pipeline, from parsing and cleaning to semantic segmentation and final
    text-level chunking.
    """
    split_on_tags: List[str] = Field(
        default_factory=lambda: ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"],
        description=(
            "A list of HTML tags that signify a semantic boundary. A new logical "
            "block of content is started when one of these tags is encountered."
        )
    )
    tags_to_ignore: List[str] = Field(
        default_factory=lambda: ["script", "style", "nav", "footer", "aside", "header", "form", "head", "meta", "link"],
        description=(
            "A denylist of tags to be completely removed from the HTML tree before "
            "any processing. This is critical for removing irrelevant noise."
        )
    )
    tags_to_extract: Optional[List[str]] = Field(
        default=None,
        description=(
            "An optional allowlist of tags. If specified, the chunker will *only* "
            "process the content within these tags and discard everything else. This "
            "provides a more precise way to extract specific content from a page."
        )
    )
    preserve_whole_tags: List[str] = Field(
        default_factory=lambda: ["table", "pre", "code", "ul", "ol"],
        description=(
            "A list of tags whose entire inner content should be treated as an "
            "indivisible, atomic unit. The chunker will not split the content "
            "inside these tags, creating an oversized chunk if necessary."
        )
    )
    extract_link_info: bool = Field(
        default=True,
        description=(
            "If True, hyperlink text will be transformed to include the URL in a "
            "Markdown-style format, e.g., 'Link Text (https://example.com)', "
            "providing more context to the language model."
        )
    )
    preserve_html_content: bool = Field(
        default=False,
        description=(
            "If True, preserves the original HTML content in chunks for accurate indexing. "
            "If False, extracts clean text but indices may not correspond to chunk content."
        )
    )
    text_chunker_to_use: "BaseChunker" = Field(
        default_factory=get_default_text_chunker,
        description=(
            "An instance of another chunker (e.g., RecursiveChunker) to be used "
            "for splitting the clean text content *within* a semantic block if that "
            "text exceeds the main `chunk_size`."
        ),
        exclude=True
    )
    merge_small_chunks: bool = Field(
        default=True,
        description=(
            "If True, merges small chunks with adjacent chunks to reduce over-chunking. "
            "This helps prevent creating too many tiny chunks from large documents."
        )
    )
    min_chunk_size_ratio: float = Field(
        default=0.3,
        description=(
            "Minimum ratio of chunk size to target chunk size for merging. "
            "Chunks smaller than this ratio will be merged with adjacent chunks."
        ),
        ge=0.0,
        le=1.0
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

# Rebuild the model to resolve forward references
HTMLChunkingConfig.model_rebuild()

class _SemanticBlock(NamedTuple):
    """An internal data structure to hold a semantically grouped block of content."""
    text_content: str
    metadata: Dict[str, str]
    start_index: int
    end_index: int
    tag_name: str

class HTMLChunker(BaseChunker[HTMLChunkingConfig]):
    """
    A structure-aware chunker for HTML documents that segments content semantically.

    This advanced chunker parses the HTML's DOM to intelligently group content.
    It follows a multi-stage pipeline:
    1.  **Parse & Sanitize:** Cleans the HTML.
    2.  **Segment:** Divides the document into logical blocks.
    3.  **Chunk:** Splits the text within each block.
    4.  **Finalize:** Merges any trailing small chunks.

    This process ensures chunks are contextually coherent and retains rich
    metadata about the document's structure.
    """

    def __init__(self, config: Optional[HTMLChunkingConfig] = None):
        """Initializes the chunker with a specific or default configuration."""
        if not _BEAUTIFULSOUP_AVAILABLE:
            raise ImportError(
                "BeautifulSoup4 is required for the HTMLChunker. "
                "Please install it with 'pip install beautifulsoup4 lxml'."
            )
        super().__init__(config or HTMLChunkingConfig())
        if self.config.tags_to_extract:
            for tag in self.config.split_on_tags:
                if tag not in self.config.tags_to_extract:
                    self.config.tags_to_extract.append(tag)

    def _chunk_document(self, document: Document) -> List[Chunk]:
        """
        The core implementation for chunking a single HTML document.

        Args:
            document: The document containing the raw HTML content.

        Returns:
            A list of `Chunk` objects derived from the HTML structure.
        """
        soup = self._parse_and_sanitize(document.content)

        semantic_blocks = self._segment_dom(soup, document.content)
        all_chunks: List[Chunk] = []
        for block in semantic_blocks:
            original_html_content = document.content[block.start_index:block.end_index]
            
            if block.tag_name in self.config.preserve_whole_tags:
                if self.config.length_function(block.text_content) > self.config.chunk_size:
                    logger.warning(
                        f"A preserved tag '{block.tag_name}' of length "
                        f"{len(block.text_content)} exceeds the chunk size. Creating an "
                        f"oversized chunk to maintain its integrity."
                    )
                
                all_chunks.append(
                    self._create_chunk(
                        parent_document=document, text_content=original_html_content,
                        start_index=block.start_index, end_index=block.end_index,
                        extra_metadata=block.metadata,
                    )
                )
                continue

            if self.config.length_function(block.text_content) <= self.config.chunk_size:
                all_chunks.append(
                    self._create_chunk(
                        parent_document=document, text_content=original_html_content,
                        start_index=block.start_index, end_index=block.end_index,
                        extra_metadata=block.metadata,
                    )
                )
            else:
                temp_doc = Document(content=original_html_content, document_id=document.document_id)
                sub_chunks = self.config.text_chunker_to_use.chunk([temp_doc])
                
                for sub_chunk in sub_chunks:
                    combined_metadata = {**block.metadata, **sub_chunk.metadata}
                    
                    abs_start = block.start_index + (sub_chunk.start_index or 0)
                    abs_end = block.start_index + (sub_chunk.end_index or 0)
                    
                    all_chunks.append(
                        self._create_chunk(
                            parent_document=document, text_content=sub_chunk.text_content,
                            start_index=abs_start, end_index=abs_end,
                            extra_metadata=combined_metadata,
                        )
                    )
        
        if not all_chunks:
            return []

        if self.config.merge_small_chunks:
            all_chunks = self._merge_small_chunks(all_chunks, document)

        return all_chunks

    def _parse_and_sanitize(self, html_content: str) -> BeautifulSoup:
        soup = BeautifulSoup(html_content, "lxml")
        
        for tag in self.config.tags_to_ignore:
            for match in soup.find_all(tag):
                match.decompose()
        
        if self.config.tags_to_extract:
            new_soup = BeautifulSoup("", "lxml")
            extracted_tags = soup.find_all(self.config.tags_to_extract)
            for tag in extracted_tags:
                new_soup.append(tag)
            return new_soup

        return soup

    def _segment_dom(self, soup: BeautifulSoup, raw_html: str) -> List[_SemanticBlock]:
        blocks: List[_SemanticBlock] = []
        all_split_tags = soup.find_all(self.config.split_on_tags)
        
        for i, start_tag in enumerate(all_split_tags):
            content_nodes = []
            end_tag = all_split_tags[i + 1] if i + 1 < len(all_split_tags) else None
            
            for sibling in start_tag.next_siblings:
                if sibling is end_tag:
                    break
                content_nodes.append(sibling)

            all_nodes_in_block = [start_tag] + content_nodes
            
            if self.config.preserve_html_content:
                block_contents = [self._get_html_from_node(node) for node in all_nodes_in_block]
                full_text = "".join(filter(None, block_contents))
            else:
                block_texts = [self._get_text_from_node(node) for node in all_nodes_in_block]
                full_text = " ".join(filter(None, block_texts)).strip()
            
            if not full_text:
                continue

            headers = {}
            for header_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                prev_header = start_tag.find_previous(header_tag)
                if prev_header:
                    headers[header_tag] = prev_header.get_text(strip=True)
            
            if start_tag.name in headers or start_tag.name.startswith('h'):
                 headers[start_tag.name] = start_tag.get_text(strip=True)
            
            start_index, end_index = self._calculate_tag_indices(start_tag, all_nodes_in_block, raw_html)
            
            blocks.append(
                _SemanticBlock(
                    text_content=full_text, metadata=headers,
                    start_index=start_index, end_index=end_index,
                    tag_name=start_tag.name
                )
            )
        return blocks

    def _calculate_tag_indices(self, start_tag, all_nodes_in_block, raw_html: str) -> tuple[int, int]:
        import re
        
        tag_name = start_tag.name
        tag_content = start_tag.get_text(strip=True)
        
        start_index = 0
        if tag_content:
            pattern = f"<{tag_name}[^>]*>{re.escape(tag_content)}</{tag_name}>"
            match = re.search(pattern, raw_html)
            if match:
                start_index = match.start()
                end_index = match.end()
                return start_index, end_index
        
        pattern = f"<{tag_name}[^>]*>"
        match = re.search(pattern, raw_html)
        if match:
            start_index = match.start()
            closing_pattern = f"</{tag_name}>"
            closing_match = re.search(closing_pattern, raw_html[start_index:])
            if closing_match:
                end_index = start_index + closing_match.end()
                return start_index, end_index
        
        end_index = start_index + len("".join(str(node) for node in all_nodes_in_block))
        return start_index, end_index

    def _merge_small_chunks(self, chunks: List[Chunk], document: Document) -> List[Chunk]:
        """Merge small chunks to reduce over-chunking."""
        if len(chunks) <= 1:
            return chunks
        
        min_chunk_size = self.config.chunk_size * self.config.min_chunk_size_ratio
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_size = self.config.length_function(current_chunk.text_content)
            
            if current_size >= min_chunk_size:
                merged_chunks.append(current_chunk)
                i += 1
                continue
            merged_text = current_chunk.text_content
            merged_start = current_chunk.start_index
            merged_end = current_chunk.end_index
            merged_metadata = current_chunk.metadata.copy()
            j = i + 1
            
            while j < len(chunks):
                next_chunk = chunks[j]
                next_size = self.config.length_function(next_chunk.text_content)
                
                if self.config.preserve_html_content:
                    separator = ""
                    combined_size = self.config.length_function(merged_text + separator + next_chunk.text_content)
                else:
                    separator = " "
                    combined_size = self.config.length_function(merged_text + separator + next_chunk.text_content)
                
                if combined_size > self.config.chunk_size * 1.5:
                    break
                
                if self.config.preserve_html_content:
                    merged_text = merged_text + next_chunk.text_content
                else:
                    merged_text = merged_text.rstrip() + " " + next_chunk.text_content.lstrip()
                merged_end = next_chunk.end_index
                merged_metadata.update(next_chunk.metadata)
                j += 1
            merged_chunk = self._create_chunk(
                parent_document=document,
                text_content=merged_text,
                start_index=merged_start,
                end_index=merged_end,
                extra_metadata=merged_metadata
            )
            merged_chunks.append(merged_chunk)
            i = j
        
        return merged_chunks

    def _get_text_from_node(self, node: Tag | NavigableString) -> str:
        if isinstance(node, NavigableString):
            return node.strip()
        if not hasattr(node, 'name'):
            return ""
        if self.config.extract_link_info and node.name == 'a':
            text = node.get_text(strip=True)
            href = node.get('href', '')
            return f"{text} ({href})" if href else text
        return node.get_text(strip=True)
    
    def _get_html_from_node(self, node: Tag | NavigableString) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        if not hasattr(node, 'name'):
            return ""
        return str(node)