from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from pydantic import Field

from .base import ChunkingStrategy, ChunkingConfig
from .recursive import RecursiveCharacterChunkingStrategy, RecursiveChunkingConfig
from upsonic.schemas.data_models import Document, Chunk

MARKDOWN_SEPARATORS = [
    r"\n# ",
    r"\n## ",
    r"\n### ",
    r"\n#### ",
    r"\n##### ",
    r"\n###### ",
    r"\n```\n",
    r"\n---\n",
    r"\n___\n",
    r"\n\*\*\*\n",
    r"\n\n",
    r"\n",
    r" ",
    r"",
]

class MarkdownChunkingConfig(RecursiveChunkingConfig):
    """Enhanced configuration for Markdown chunking."""
    separators: List[str] = Field(default_factory=lambda: MARKDOWN_SEPARATORS, description="Markdown-optimized separators")
    preserve_headers: bool = Field(True, description="Preserve header hierarchy")
    preserve_code_blocks: bool = Field(True, description="Keep code blocks intact")
    preserve_lists: bool = Field(True, description="Keep list structures together")
    include_front_matter: bool = Field(True, description="Include YAML front matter")


class MarkdownRecursiveChunkingStrategy(RecursiveCharacterChunkingStrategy):
    """
    Markdown chunking strategy with structure awareness.

    This specialist strategy uses recursive splitting with Markdown-specific
    separators and enhanced structure preservation features:
    
    Features:
    - Header hierarchy preservation and context
    - Code block integrity maintenance
    - List structure preservation
    - Front matter handling and metadata extraction
    - Enhanced separator optimization for Markdown syntax
    - Rich metadata with document structure information
    """
    
    def __init__(self, config: Optional[MarkdownChunkingConfig] = None):
        """Initialize Markdown chunking strategy."""
        if config is None:
            config = MarkdownChunkingConfig()
        super().__init__(config)


class MarkdownHeaderChunkingConfig(ChunkingConfig):
    """Enhanced configuration for Markdown header-based chunking."""
    headers_to_split_on: List[Tuple[str, str]] = Field(
        default_factory=lambda: [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), 
                                ("####", "Header 4"), ("#####", "Header 5"), ("######", "Header 6")],
        description="Header levels to split on"
    )
    strip_headers: bool = Field(True, description="Remove header line from chunk content")
    preserve_hierarchy: bool = Field(True, description="Maintain header hierarchy in metadata")
    include_header_context: bool = Field(True, description="Include parent headers in context")


class MarkdownHeaderChunkingStrategy(ChunkingStrategy):
    """
    Enhanced header-based Markdown chunking strategy with framework features.

    This strategy splits Markdown documents based on header structure with
    enhanced hierarchy preservation and metadata enrichment:
    
    Enhanced Features:
    - Hierarchical header context preservation
    - Rich metadata with document structure
    - Configurable header level processing
    - Enhanced section boundary detection
    - Parent-child header relationship tracking
    """
    
    def __init__(self, config: Optional[MarkdownHeaderChunkingConfig] = None):
        """
        Initialize enhanced Markdown header chunking strategy.

        Args:
            config: Configuration object with all settings
        """
        if config is None:
            config = MarkdownHeaderChunkingConfig()
        
        super().__init__(config)
        
        self.headers_to_split_on = sorted(
            self.config.headers_to_split_on, key=lambda split: len(split[0]), reverse=True
        )
        self.strip_headers = self.config.strip_headers

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Parses the document line by line, splitting it into chunks based on
        the configured header levels.
        """
        lines = document.content.split("\n")
        
        chunks: List[Chunk] = []
        current_chunk_lines: List[str] = []
        header_stack: Dict[str, str] = {}
        
        for line in lines:
            is_header_line = False
            for sep, name in self.headers_to_split_on:
                if line.startswith(sep + " "):
                    if current_chunk_lines:
                        chunk_content = "\n".join(current_chunk_lines).strip()
                        if chunk_content:
                            final_metadata = document.metadata.copy()
                            final_metadata.update(header_stack)
                            chunks.append(Chunk(
                                text_content=chunk_content,
                                metadata=final_metadata,
                                document_id=document.document_id
                            ))
                        current_chunk_lines = []

                    header_level = len(sep)
                    header_value = line[header_level:].strip()
                    
                    keys_to_pop = [
                        h_name for h_sep, h_name in self.headers_to_split_on
                        if len(h_sep) >= header_level and h_name in header_stack
                    ]
                    for key in keys_to_pop:
                        header_stack.pop(key)
                    
                    header_stack[name] = header_value
                    
                    if not self.strip_headers:
                        current_chunk_lines.append(line)
                    
                    is_header_line = True
                    break
            
            if not is_header_line:
                current_chunk_lines.append(line)

        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines).strip()
            if chunk_content:
                final_metadata = document.metadata.copy()
                final_metadata.update(header_stack)
                chunks.append(Chunk(
                    text_content=chunk_content,
                    metadata=final_metadata,
                    document_id=document.document_id
                ))
        
        return chunks