from __future__ import annotations

import asyncio
import re
from enum import Enum
from typing import Callable, List

from pydantic import Field, field_validator, ValidationInfo, ConfigDict
try:
    import numpy as np
except ImportError:
    raise ImportError(
        "numpy is required for the SemanticChunker. "
        "Please install it with 'pip install numpy'."
    )

from upsonic.schemas.data_models import Chunk, Document
from upsonic.embeddings.base import EmbeddingProvider
from upsonic.text_splitter.base import BaseChunkingConfig, BaseChunker
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)

class BreakpointThresholdType(str, Enum):
    """
    Defines the statistical method used to determine the threshold for a topic break.
    """
    PERCENTILE = "percentile"
    STD_DEV = "standard_deviation"
    INTERQUARTILE = "interquartile"
    GRADIENT = "gradient"


def default_sentence_splitter(text: str) -> List[str]:
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
    sentences = re.split(pattern, text)
    return [s for s in sentences if s and s.strip()]


class SemanticChunkingConfig(BaseChunkingConfig):
    """
    A specialized configuration model for the data-driven Semantic Chunker.

    This config requires an embedding provider and provides fine-grained control
    over the statistical methods used to identify semantic boundaries in text.
    """
    embedding_provider: EmbeddingProvider = Field(
        ...,
        description=(
            "A required, initialized instance of an EmbeddingProvider (e.g., "
            "OpenAIEmbedding). This will be used to generate the embeddings for "
            "semantic analysis."
        ),
        exclude=True
    )
    breakpoint_threshold_type: BreakpointThresholdType = Field(
        default=BreakpointThresholdType.PERCENTILE,
        description=(
            "The statistical method to use for identifying topic breaks. 'PERCENTILE' "
            "is a robust default."
        )
    )
    breakpoint_threshold_amount: float = Field(
        default=95.0,
        description=(
            "The numeric value for the chosen threshold type. For PERCENTILE, this "
            "is the percentile (0-100). For STD_DEV, it's the number of standard "
            "deviations above the mean."
        )
    )
    sentence_splitter: Callable[[str], List[str]] = Field(
        default_factory=lambda: default_sentence_splitter,
        description=(
            "A function that takes a string and returns a list of sentence strings. "
            "Defaults to a regex-based splitter, but can be replaced with more "
            "advanced tokenizers like from NLTK or spaCy."
        ),
        exclude=True
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("breakpoint_threshold_amount")
    @classmethod
    def validate_threshold_amount(
        cls, v: float, info: ValidationInfo
    ) -> float:
        if "breakpoint_threshold_type" in info.data:
            threshold_type = info.data["breakpoint_threshold_type"]
            if threshold_type == BreakpointThresholdType.PERCENTILE and not (0 <= v <= 100):
                raise ValueError("PERCENTILE threshold amount must be between 0 and 100.")
        return v


class _Sentence:
    def __init__(self, text: str, start_index: int, end_index: int):
        self.text = text
        self.start_index = start_index
        self.end_index = end_index
        self.embedding: List[float] = []



class SemanticChunker(BaseChunker[SemanticChunkingConfig]):
    """
    An advanced chunker that splits text based on semantic topic shifts.

    This chunker identifies boundaries by embedding individual sentences and finding
    points of high cosine distance between adjacent sentences, indicating a change
    in topic. It is an asynchronous process that leverages an `EmbeddingProvider`.
    """

    def __init__(self, config: SemanticChunkingConfig):
        """
        Initializes the chunker with a specific configuration.
        Note: A config with an `embedding_provider` is required for this chunker.
        """
        super().__init__(config)

    def _chunk_document(self, document: Document) -> List[Chunk]:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._achunk_document(document))
                return future.result()
        except RuntimeError:
            return asyncio.run(self._achunk_document(document))

    async def _achunk_document(self, document: Document) -> List[Chunk]:
        sentences = self._segment_into_sentences(document.content)
        if len(sentences) < 2:
            return [self._create_chunk(document, document.content, 0, len(document.content))]
            
        sentence_texts = [s.text for s in sentences]
        embeddings = await self.config.embedding_provider.embed_texts(sentence_texts, show_progress=False)
        for i, sentence in enumerate(sentences):
            sentence.embedding = embeddings[i]

        distances = self._calculate_distances(sentences)
        threshold = self._calculate_breakpoint_threshold(distances)
        breakpoints = [i for i, dist in enumerate(distances) if dist > threshold]
        all_chunks: List[Chunk] = []
        start_sentence_idx = 0
        for breakpoint_idx in breakpoints:
            end_sentence_idx = breakpoint_idx + 1
            chunk_sentences = sentences[start_sentence_idx:end_sentence_idx]
            if chunk_sentences:
                text = " ".join(s.text for s in chunk_sentences)
                all_chunks.append(self._create_chunk(
                    document, text, chunk_sentences[0].start_index, chunk_sentences[-1].end_index
                ))
            start_sentence_idx = end_sentence_idx

        final_group = sentences[start_sentence_idx:]
        if final_group:
            text = " ".join(s.text for s in final_group)
            all_chunks.append(self._create_chunk(
                document, text, final_group[0].start_index, final_group[-1].end_index
            ))

        if not all_chunks:
            return []

        min_chunk_size = self._get_effective_min_chunk_size()

        if len(all_chunks) > 1 and self.config.length_function(all_chunks[-1].text_content) < min_chunk_size:
            last_chunk = all_chunks.pop()
            previous_chunk = all_chunks.pop()
            
            merged_text = previous_chunk.text_content.rstrip() + " " + last_chunk.text_content.lstrip()
            
            merged_chunk = self._create_chunk(
                parent_document=document,
                text_content=merged_text,
                start_index=previous_chunk.start_index,
                end_index=last_chunk.end_index,
                extra_metadata=previous_chunk.metadata
            )
            all_chunks.append(merged_chunk)
            
        return all_chunks

    def _segment_into_sentences(self, text: str) -> List[_Sentence]:
        sentence_strings = self.config.sentence_splitter(text)
        sentences: List[_Sentence] = []
        cursor = 0
        
        for s_text in sentence_strings:
            try:
                start_index = text.find(s_text, cursor)
                if start_index == -1:
                    start_index = text.find(s_text)
                    if start_index == -1:
                        logger.warning(f"Could not find sentence '{s_text[:30]}...' in the original text. Skipping.")
                        continue
                
                end_index = start_index + len(s_text)
                sentences.append(_Sentence(s_text, start_index, end_index))
                cursor = end_index
                
            except Exception as e:
                logger.warning(f"Error processing sentence '{s_text[:30]}...': {e}. Skipping.")
                continue
                
        return sentences
        
    def _calculate_distances(self, sentences: List[_Sentence]) -> List[float]:
        distances: List[float] = []
        for i in range(len(sentences) - 1):
            emb_curr = np.array(sentences[i].embedding)
            emb_next = np.array(sentences[i+1].embedding)
            similarity = np.dot(emb_curr, emb_next) / (np.linalg.norm(emb_curr) * np.linalg.norm(emb_next))
            distances.append(1 - similarity)
        return distances

    def _calculate_breakpoint_threshold(self, distances: List[float]) -> float:
        if not distances:
            return 0.0

        amount = self.config.breakpoint_threshold_amount
        method = self.config.breakpoint_threshold_type

        if method == BreakpointThresholdType.PERCENTILE:
            return float(np.percentile(distances, amount))
        elif method == BreakpointThresholdType.STD_DEV:
            return float(np.mean(distances) + amount * np.std(distances))
        elif method == BreakpointThresholdType.INTERQUARTILE:
            q1, q3 = np.percentile(distances, [25, 75])
            return float(np.mean(distances) + amount * (q3 - q1))
        elif method == BreakpointThresholdType.GRADIENT:
            return float(np.percentile(np.gradient(distances), amount))
        else:
            raise ValueError(f"Unknown breakpoint threshold type: {method}")