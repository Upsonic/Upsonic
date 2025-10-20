from __future__ import annotations

import re
from typing import List, Optional, Tuple

from pydantic import Field

from upsonic.schemas.data_models import Chunk, Document
from upsonic.text_splitter.base import BaseChunkingConfig, BaseChunker
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)

class CharacterChunkingConfig(BaseChunkingConfig):
    """
    A specialized configuration model for the Character Chunker strategy.

    This configuration extends the base settings with parameters that control
    the splitting behavior based on a single, user-defined separator.
    """
    separator: str = Field(
        default="\n\n",
        description=(
            "The single, definitive string or regex pattern that will be used to "
            "split the document text. This acts as the primary boundary marker."
        )
    )
    is_separator_regex: bool = Field(
        default=False,
        description=(
            "If True, the separator is treated as a regular expression, enabling "
            "more complex and powerful splitting rules. If False, it is treated "
            "as a simple string literal."
        )
    )
    keep_separator: bool = Field(
        default=True,
        description=(
            "Determines whether the separator itself is kept as part of the chunks. "
            "Keeping the separator is often useful for preserving the original "
            "structure and context of the document."
        )
    )


class CharacterChunker(BaseChunker[CharacterChunkingConfig]):
    """
    A foundational chunker that splits text based on a single, specified character separator.

    This chunker is a workhorse for documents with a clear and consistent delimiter.
    It operates with a direct, "Split and Merge" process to ensure both
    efficiency and perfect positional integrity of the final chunks.
    """

    def __init__(self, config: Optional[CharacterChunkingConfig] = None):
        """Initializes the chunker with a specific or default configuration."""
        super().__init__(config or CharacterChunkingConfig())

    def _chunk_document(self, document: Document) -> List[Chunk]:
        """
        The core implementation for splitting a single document by a character separator.


        Args:
            document: The document to be chunked.

        Returns:
            A list of `Chunk` objects with accurate content and positional data.
        """
        content = document.content
        if not content or content.isspace():
            return []

        atomic_splits: List[Tuple[str, int, int]] = []
        separator = self.config.separator

        if not separator:
            for i, char in enumerate(content):
                atomic_splits.append((char, i, i + 1))
        else:
            pattern = re.escape(separator) if not self.config.is_separator_regex else separator
            cursor = 0
            for match in re.finditer(pattern, content):
                if match.start() > cursor:
                    atomic_splits.append((content[cursor:match.start()], cursor, match.start()))
                if self.config.keep_separator:
                    atomic_splits.append((match.group(0), match.start(), match.end()))
                cursor = match.end()
            if cursor < len(content):
                atomic_splits.append((content[cursor:], cursor, len(content)))

        if not atomic_splits:
            atomic_splits.append((content, 0, len(content)))
        chunks: List[Chunk] = []
        current_chunk_parts: List[Tuple[str, int, int]] = []
        current_length = 0
        length_func = self.config.length_function

        for text, start_idx, end_idx in atomic_splits:
            part_length = length_func(text)
            
            if part_length > self.config.chunk_size:
                if current_chunk_parts:
                    chunk_start_idx = current_chunk_parts[0][1]
                    chunk_end_idx = current_chunk_parts[-1][2]
                    final_text = content[chunk_start_idx:chunk_end_idx]
                    chunks.append(
                        self._create_chunk(document, final_text, chunk_start_idx, chunk_end_idx)
                    )
                    current_chunk_parts = []
                    current_length = 0

                logger.warning(
                    f"A single text segment of length {part_length} from document ID "
                    f"'{document.document_id}' exceeds the chunk size of {self.config.chunk_size}. "
                    f"Creating an oversized chunk."
                )
                chunks.append(self._create_chunk(document, text, start_idx, end_idx))
                continue

            if current_length + part_length > self.config.chunk_size and current_chunk_parts:
                chunk_start_idx = current_chunk_parts[0][1]
                chunk_end_idx = current_chunk_parts[-1][2]
                final_text = content[chunk_start_idx:chunk_end_idx]
                chunks.append(
                    self._create_chunk(document, final_text, chunk_start_idx, chunk_end_idx)
                )

                overlap_len = 0
                overlap_start_index = len(current_chunk_parts)
                for j in range(len(current_chunk_parts) - 1, -1, -1):
                    part_text, _, _ = current_chunk_parts[j]
                    if overlap_len + length_func(part_text) > self.config.chunk_overlap:
                        break
                    overlap_len += length_func(part_text)
                    overlap_start_index = j
                
                current_chunk_parts = current_chunk_parts[overlap_start_index:]
                current_length = sum(length_func(part[0]) for part in current_chunk_parts)

            current_chunk_parts.append((text, start_idx, end_idx))
            current_length += part_length

        if current_chunk_parts:
            chunk_start_idx = current_chunk_parts[0][1]
            chunk_end_idx = current_chunk_parts[-1][2]
            final_text = content[chunk_start_idx:chunk_end_idx]

            min_chunk_size = self._get_effective_min_chunk_size()

            if chunks and length_func(final_text) < min_chunk_size:
                last_chunk = chunks.pop()
                merged_text = content[last_chunk.start_index:chunk_end_idx]
                merged_chunk = self._create_chunk(
                    document,
                    merged_text,
                    last_chunk.start_index,
                    chunk_end_idx
                )
                chunks.append(merged_chunk)
            else:
                chunks.append(
                    self._create_chunk(document, final_text, chunk_start_idx, chunk_end_idx)
                )

        return chunks