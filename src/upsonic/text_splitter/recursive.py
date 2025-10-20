from __future__ import annotations

from enum import Enum
import re

from typing import List, Optional, Tuple

from pydantic import Field

from upsonic.text_splitter.base import BaseChunkingConfig, BaseChunker
from upsonic.schemas.data_models import Chunk, Document
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)

class Language(str, Enum):
    """
    An enumeration of supported programming and markup languages for specialized,
    semantically-aware chunking.
    """
    PYTHON = "python"
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"
    JAVA = "java"
    JS = "js"


RECURSIVE_SEPARATORS = {
    Language.PYTHON: [
        r"\nclass [a-zA-Z_][a-zA-Z0-9_]*:",
        r"\ndef [a-zA-Z_][a-zA-Z0-9_]*\([^)]*\):",
        r"\n\tdef [a-zA-Z_][a-zA-Z0-9_]*\([^)]*\):",
        "\n\n", "\n", " ", "",
    ],
    Language.MARKDOWN: [
        "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",
        "```\n", "\n\n***\n\n", "\n\n---\n\n", "\n\n___\n\n",
        "\n\n", "\n", " ", "",
    ],
    Language.HTML: [
        r"(?i)(?=</body>)", r"(?i)(?=</div>)", r"(?i)(?=</p>)", r"(?i)(?=</table>)",
        r"(?i)(?=</header>)", r"(?i)(?=</footer>)", r"(?i)(?=</main>)", r"(?i)(?=</section>)",
        r"(?i)(?=</h1>)", r"(?i)(?=</h2>)", r"(?i)(?=</h3>)", r"(?i)(?=</h4>)", r"(?i)(?=</h5>)", r"(?i)(?=</h6>)",
        "\n\n", "\n", " ", "",
    ],
    Language.LATEX: [
        r"\n\\chapter{", r"\n\\section{", r"\n\\subsection{", r"\n\\subsubsection{",
        r"\n\n\\begin{enumerate}", r"\n\n\\begin{itemize}", r"\n\n\\begin{verbatim}",
        "\n\n", "\n", " ", "",
    ],
    Language.JAVA: [
        r"\nclass ", r"\npublic class ", r"\ninterface ",
        r"\npublic void ", r"\nprotected void ", r"\nprivate void ", r"\nstatic void ",
        "\n\n", "\n", " ", ""
    ],
    Language.JS: [
        r"\nfunction ", r"\nconst ", r"\nlet ", r"\nvar ", r"\nclass ",
        "\n\n", "\n", " ", ""
    ],
}



class RecursiveChunkingConfig(BaseChunkingConfig):
    """
    A specialized configuration model for the Recursive Chunker strategy.

    This configuration extends the base settings with parameters that control
    the recursive splitting behavior, such as the prioritized list of separators
    to use.
    """
    separators: List[str] = Field(
        default=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        description=(
            "A prioritized list of separators or regex patterns. The chunker "
            "tries to split the text by the first separator in the list. If any "
            "resulting segment is still too large, it moves to the next separator "
            "to split that specific segment."
        )
    )
    keep_separator: bool = Field(
        default=True,
        description=(
            "Determines whether the separator is kept as part of the chunks. "
            "When True, the separator is attached to the end of the text segment "
            "that precedes it, preserving context. When False, separators are discarded."
        )
    )
    is_separator_regex: bool = Field(
        default=False,
        description=(
            "If True, the separators are treated as regular expressions, enabling "
            "more complex and powerful splitting rules. If False, they are treated "
            "as simple string literals."
        )
    )

# Rebuild the model to resolve forward references
RecursiveChunkingConfig.model_rebuild()




class RecursiveChunker(BaseChunker[RecursiveChunkingConfig]):
    """
    A sophisticated chunker that recursively splits text to preserve semantic boundaries.

    This chunker works by trying to split text using a prioritized list of separators.
    If a resulting segment is still too large, it moves to the next separator in the
    list and tries to split that segment again. This is highly effective for structured
    text like code and markdown, ensuring logical units of information are kept together.
    """

    def __init__(self, config: Optional[RecursiveChunkingConfig] = None):
        """Initializes the chunker with a specific or default configuration."""
        super().__init__(config or RecursiveChunkingConfig())

    def _chunk_document(self, document: Document) -> List[Chunk]:
        """
        The core implementation for recursively chunking a single document.
        """
        content = document.content
        if not content or content.isspace():
            return []

        atomic_splits = self._recursive_split(
            text=content,
            separators=self.config.separators,
            offset=0
        )

        if not atomic_splits:
            return []
        chunks: List[Chunk] = []
        current_chunk_parts: List[Tuple[str, int, int]] = []
        current_length = 0
        length_func = self.config.length_function

        for i, (text, start_idx, end_idx) in enumerate(atomic_splits):
            part_length = length_func(text)

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
                    document, merged_text, last_chunk.start_index, chunk_end_idx
                )
                chunks.append(merged_chunk)
            else:
                chunks.append(
                    self._create_chunk(document, final_text, chunk_start_idx, chunk_end_idx)
                )

        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        offset: int
    ) -> List[Tuple[str, int, int]]:
        if not text:
            return []

        current_separator = ""
        next_separators = []
        for i, sep in enumerate(separators):
            if sep == "" and i < len(separators) - 1:
                continue
            
            pattern = sep if self.config.is_separator_regex else re.escape(sep)
            if pattern and re.search(pattern, text):
                current_separator = sep
                next_separators = separators[i + 1:]
                break
        
        if not current_separator and separators:
            current_separator = separators[-1]

        if not current_separator:
            return [(text, offset, offset + len(text))]
        final_splits: List[Tuple[str, int, int]] = []
        pattern = current_separator if self.config.is_separator_regex else re.escape(current_separator)
        
        cursor = 0
        for match in re.finditer(pattern, text):
            if self.config.keep_separator:
                segment_text = text[cursor:match.end()]
                start_idx, end_idx = offset + cursor, offset + match.end()
            else:
                segment_text = text[cursor:match.start()]
                start_idx, end_idx = offset + cursor, offset + match.start()

            if self.config.length_function(segment_text) > self.config.chunk_size:
                if next_separators:
                    final_splits.extend(self._recursive_split(segment_text, next_separators, start_idx))
                else:
                    final_splits.append((segment_text, start_idx, end_idx))
            elif segment_text:
                final_splits.append((segment_text, start_idx, end_idx))

            cursor = match.end()
        if cursor < len(text):
            remaining_text = text[cursor:]
            start_idx, end_idx = offset + cursor, offset + len(text)

            if self.config.length_function(remaining_text) > self.config.chunk_size:
                if next_separators:
                    final_splits.extend(self._recursive_split(remaining_text, next_separators, start_idx))
                else:
                    final_splits.append((remaining_text, start_idx, end_idx))
            elif remaining_text:
                final_splits.append((remaining_text, start_idx, end_idx))
                
        return final_splits

    @classmethod
    def from_language(
        cls,
        language: Language,
        config: Optional[RecursiveChunkingConfig] = None
    ) -> "RecursiveChunker":
        if language not in RECURSIVE_SEPARATORS:
            raise ValueError(
                f"Language '{language.value}' is not supported for specialized chunking. "
                f"Please choose from {[lang.value for lang in RECURSIVE_SEPARATORS.keys()]}"
            )

        active_config = config or RecursiveChunkingConfig()
        active_config.separators = RECURSIVE_SEPARATORS[language]
        active_config.is_separator_regex = True
        active_config.keep_separator = True

        logger.info(f"Created a RecursiveChunker instance configured for '{language.value}'.")
        return cls(active_config)