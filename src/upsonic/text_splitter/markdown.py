from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional
import re

from pydantic import Field, ConfigDict

from upsonic.text_splitter.base import BaseChunker, BaseChunkingConfig
from upsonic.schemas.data_models import Chunk, Document
from upsonic.text_splitter.recursive import RecursiveChunker, RecursiveChunkingConfig
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)

def get_default_text_chunker() -> BaseChunker:
    """A factory function to create a default text chunker instance."""
    config = RecursiveChunkingConfig(chunk_size=512, chunk_overlap=50)
    return RecursiveChunker(config)


class MarkdownChunkingConfig(BaseChunkingConfig):
    """
    A specialized configuration model for the syntax-aware Markdown Chunker.

    This config provides fine-grained control over which Markdown elements are
    treated as semantic boundaries and how the final text content is processed.
    """
    split_on_elements: List[str] = Field(
        default_factory=lambda: ["h1", "h2", "h3", "code_block", "table", "horizontal_rule"],
        description=(
            "A list of Markdown element types that signify a major semantic boundary. "
            "A new logical block of content is started when one of these is encountered."
        )
    )
    preserve_whole_elements: List[str] = Field(
        default_factory=lambda: ["code_block", "table"],
        description=(
            "A list of element types that should be treated as indivisible, atomic "
            "units. The chunker will not split the content inside these elements, "
            "creating an oversized chunk if necessary to preserve their integrity."
        )
    )
    strip_elements: bool = Field(
        default=True,
        description=(
            "If True, the Markdown syntax characters (e.g., '## ', '|', '---') are "
            "stripped from the final text_content of the chunks. If False, the "
            "original Markdown formatting is preserved."
        )
    )
    preserve_original_content: bool = Field(
        default=False,
        description=(
            "If True, preserves the original markdown content in chunks for accurate indexing. "
            "If False, strips elements but indices may not correspond to chunk content."
        )
    )
    text_chunker_to_use: BaseChunker = Field(
        default_factory=get_default_text_chunker,
        description=(
            "An instance of another chunker (e.g., RecursiveChunker) to be used "
            "for splitting the clean text content *within* a semantic block if that "
            "text exceeds the main `chunk_size`."
        ),
        exclude=True
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

# Rebuild the model to resolve forward references
MarkdownChunkingConfig.model_rebuild()

class _SemanticBlock(NamedTuple):
    """An internal data structure for a semantically distinct block of Markdown."""
    type: str
    raw_content: str
    start_index: int
    end_index: int
    metadata: Dict[str, str]



class MarkdownChunker(BaseChunker[MarkdownChunkingConfig]):
    """
    A syntax-aware chunker for Markdown documents that segments content structurally.

    This advanced chunker parses the raw Markdown text to identify the boundaries
    of its structural elements like headers, code blocks, tables, and lists.
    """

    def __init__(self, config: Optional[MarkdownChunkingConfig] = None):
        """Initializes the chunker with a specific or default configuration."""
        super().__init__(config or MarkdownChunkingConfig())

    def _chunk_document(self, document: Document) -> List[Chunk]:
        semantic_blocks = self._segment_markdown(document.content)
        all_chunks: List[Chunk] = []
        current_chunk_blocks: List[_SemanticBlock] = []
        header_stack: List[Dict[str, str]] = []

        for block in semantic_blocks:
            is_boundary = block.type in self.config.split_on_elements
            is_preserved = block.type in self.config.preserve_whole_elements

            if (is_boundary or is_preserved) and current_chunk_blocks:
                all_chunks.extend(self._finalize_chunk(current_chunk_blocks, header_stack, document))
                current_chunk_blocks = []

            if block.type.startswith("h") and block.type[1:].isdigit():
                level = int(block.type[1:])
                header_stack = [h for h in header_stack if int(list(h.keys())[0][1:]) < level]
                header_stack.append({block.type: block.metadata.get("header_text", "")})

            if is_preserved:
                all_chunks.extend(self._finalize_chunk([block], header_stack, document))
                continue

            current_chunk_blocks.append(block)

        if current_chunk_blocks:
            all_chunks.extend(self._finalize_chunk(current_chunk_blocks, header_stack, document))
        
        if not all_chunks:
            return []

        min_chunk_size = self._get_effective_min_chunk_size()

        if len(all_chunks) > 1 and self.config.length_function(all_chunks[-1].text_content) < min_chunk_size:
            last_chunk = all_chunks.pop()
            previous_chunk = all_chunks.pop()

            merged_text = previous_chunk.text_content.rstrip() + "\n\n" + last_chunk.text_content.lstrip()

            merged_chunk = self._create_chunk(
                parent_document=document,
                text_content=merged_text,
                start_index=previous_chunk.start_index,
                end_index=last_chunk.end_index,
                extra_metadata=previous_chunk.metadata,
            )
            all_chunks.append(merged_chunk)

        return all_chunks

    def _segment_markdown(self, text: str) -> List[_SemanticBlock]:
        patterns = [
            r"(?P<h1_h6>^(#{1,6})\s+.*)",
            r"(?P<code_block>^```\w*\n[\s\S]*?\n```)",
            r"(?P<list_item>^(?:[ \t]*)(?:[\*\-\+]|\d+\.)[ \t]+.*)",
            r"(?P<blockquote>^(?:> .*(?:\n|$))+)",
            r"(?P<table>(?:\|.*\|\n)+(?:\|(?:[-: ]+)\|.*))",
            r"(?P<horizontal_rule>^\s*(?:---|\*\*\*|___)\s*$)",
        ]
        master_pattern = re.compile("|".join(patterns), re.MULTILINE)

        blocks: List[_SemanticBlock] = []
        last_end = 0

        for match in master_pattern.finditer(text):
            if match.start() > last_end:
                blocks.append(_SemanticBlock(
                    type="paragraph", raw_content=text[last_end:match.start()],
                    start_index=last_end, end_index=match.start(), metadata={}
                ))

            block_type = match.lastgroup or "unknown"
            raw_content = match.group(0)
            metadata = {}
            if block_type == "h1_h6":
                level = len(match.group(2))
                block_type = f"h{level}"
                metadata["header_text"] = raw_content.lstrip("# ").strip()

            blocks.append(_SemanticBlock(
                type=block_type, raw_content=raw_content,
                start_index=match.start(), end_index=match.end(), metadata=metadata
            ))
            last_end = match.end()

        if last_end < len(text):
            blocks.append(_SemanticBlock(
                type="paragraph", raw_content=text[last_end:],
                start_index=last_end, end_index=len(text), metadata={}
            ))
        return blocks

    def _finalize_chunk(
        self,
        blocks: List[_SemanticBlock],
        header_stack: List[Dict[str, str]],
        document: Document
    ) -> List[Chunk]:
        """Assembles a list of blocks into one or more final Chunk objects."""
        if not blocks:
            return []

        full_text = "".join(block.raw_content for block in blocks)
        start_index = blocks[0].start_index
        end_index = blocks[-1].end_index
        
        original_text = full_text
        
        if self.config.strip_elements and not self.config.preserve_original_content:
            full_text = re.sub(r"^(#{1,6})\s+", "", full_text, flags=re.MULTILINE)

        final_metadata = {}
        for header in header_stack:
            final_metadata.update(header)
            
        if self.config.length_function(full_text) <= self.config.chunk_size:
            if full_text.strip():
                chunk_content = original_text if self.config.preserve_original_content else full_text
                return [self._create_chunk(
                    parent_document=document, text_content=chunk_content,
                    start_index=start_index, end_index=end_index,
                    extra_metadata=final_metadata
                )]
            else:
                return []
        else:
            sub_chunks: List[Chunk] = []
            content_to_split = original_text if self.config.preserve_original_content else full_text
            temp_doc = Document(content=content_to_split, document_id=document.document_id)
            split_sub_chunks = self.config.text_chunker_to_use.chunk([temp_doc])
            
            for sub_chunk in split_sub_chunks:
                if sub_chunk.text_content.strip():
                    abs_start = start_index + (sub_chunk.start_index or 0)
                    abs_end = start_index + (sub_chunk.end_index or 0)
                    sub_chunks.append(self._create_chunk(
                        parent_document=document, text_content=sub_chunk.text_content,
                        start_index=abs_start, end_index=abs_end,
                        extra_metadata={**final_metadata, **sub_chunk.metadata}
                    ))
            return sub_chunks