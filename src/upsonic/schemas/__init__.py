from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .user import UserTraits
    from .agentic import PropositionList, Topic, TopicAssignmentList, RefinedTopic
    from .data_models import Document, Chunk, RAGSearchResult
    from .vector_schemas import VectorSearchResult

def _get_schema_classes():
    """Lazy import of schema classes."""
    from .user import UserTraits
    from .agentic import PropositionList, Topic, TopicAssignmentList, RefinedTopic
    from .data_models import Document, Chunk, RAGSearchResult
    from .vector_schemas import VectorSearchResult
    
    return {
        'UserTraits': UserTraits,
        'PropositionList': PropositionList,
        'Topic': Topic,
        'TopicAssignmentList': TopicAssignmentList,
        'RefinedTopic': RefinedTopic,
        'Document': Document,
        'Chunk': Chunk,
        'RAGSearchResult': RAGSearchResult,
        'VectorSearchResult': VectorSearchResult,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    schema_classes = _get_schema_classes()
    if name in schema_classes:
        return schema_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "UserTraits",
    "PropositionList",
    "Topic",
    "TopicAssignmentList",
    "RefinedTopic",
    "Document",
    "Chunk",
    "RAGSearchResult",
    "VectorSearchResult",
]