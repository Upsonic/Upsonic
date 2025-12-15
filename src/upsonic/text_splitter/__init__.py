from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import (
        BaseChunker,
        BaseChunkingConfig
    )

if TYPE_CHECKING:
    from .character import CharacterChunker, CharacterChunkingConfig
    from .recursive import RecursiveChunker, RecursiveChunkingConfig
    from .html_chunker import HTMLChunker, HTMLChunkingConfig
    from .json_chunker import JSONChunker, JSONChunkingConfig
    from .markdown import MarkdownChunker, MarkdownChunkingConfig
    from .python import PythonChunker, PythonChunkingConfig
    from .semantic import SemanticChunker, SemanticChunkingConfig
    from .agentic import AgenticChunker, AgenticChunkingConfig
    from .factory import (
        create_chunking_strategy,
        create_adaptive_strategy,
        create_rag_strategy,
        create_semantic_search_strategy,
        create_fast_strategy,
        create_quality_strategy,
        create_intelligent_splitters,
        list_available_strategies,
        get_strategy_info,
        detect_content_type,
        recommend_strategy_for_content,
        ContentType,
        ChunkingUseCase
    )

def _get_chunker_classes():
    """Lazy import of chunker classes."""
    from .character import CharacterChunker, CharacterChunkingConfig
    from .recursive import RecursiveChunker, RecursiveChunkingConfig
    from .json_chunker import JSONChunker, JSONChunkingConfig
    from .markdown import MarkdownChunker, MarkdownChunkingConfig
    from .python import PythonChunker, PythonChunkingConfig
    from .semantic import SemanticChunker, SemanticChunkingConfig
    from .agentic import AgenticChunker, AgenticChunkingConfig
    
    try:
        from .html_chunker import HTMLChunker, HTMLChunkingConfig
    except ImportError:
        HTMLChunker = None
        HTMLChunkingConfig = None
    
    return {
        'CharacterChunker': CharacterChunker,
        'CharacterChunkingConfig': CharacterChunkingConfig,
        'RecursiveChunker': RecursiveChunker,
        'RecursiveChunkingConfig': RecursiveChunkingConfig,
        'HTMLChunker': HTMLChunker,
        'HTMLChunkingConfig': HTMLChunkingConfig,
        'JSONChunker': JSONChunker,
        'JSONChunkingConfig': JSONChunkingConfig,
        'MarkdownChunker': MarkdownChunker,
        'MarkdownChunkingConfig': MarkdownChunkingConfig,
        'PythonChunker': PythonChunker,
        'PythonChunkingConfig': PythonChunkingConfig,
        'SemanticChunker': SemanticChunker,
        'SemanticChunkingConfig': SemanticChunkingConfig,
        'AgenticChunker': AgenticChunker,
        'AgenticChunkingConfig': AgenticChunkingConfig,
    }

def _get_factory_functions():
    """Lazy import of factory functions."""
    from .factory import (
        create_chunking_strategy,
        create_adaptive_strategy,
        create_rag_strategy,
        create_semantic_search_strategy,
        create_fast_strategy,
        create_quality_strategy,
        create_intelligent_splitters,
        list_available_strategies,
        get_strategy_info,
        detect_content_type,
        recommend_strategy_for_content,
        ContentType,
        ChunkingUseCase
    )
    
    return {
        'create_chunking_strategy': create_chunking_strategy,
        'create_adaptive_strategy': create_adaptive_strategy,
        'create_rag_strategy': create_rag_strategy,
        'create_semantic_search_strategy': create_semantic_search_strategy,
        'create_fast_strategy': create_fast_strategy,
        'create_quality_strategy': create_quality_strategy,
        'create_intelligent_splitters': create_intelligent_splitters,
        'list_available_strategies': list_available_strategies,
        'get_strategy_info': get_strategy_info,
        'detect_content_type': detect_content_type,
        'recommend_strategy_for_content': recommend_strategy_for_content,
        'ContentType': ContentType,
        'ChunkingUseCase': ChunkingUseCase,
    }

def _get_base_classes():
    """Lazy import of base chunker classes."""
    from .base import (
        BaseChunker,
        BaseChunkingConfig
    )
    
    return {
        'BaseChunker': BaseChunker,
        'BaseChunkingConfig': BaseChunkingConfig,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    # Base classes
    base_classes = _get_base_classes()
    if name in base_classes:
        return base_classes[name]
    
    # Chunker classes
    chunker_classes = _get_chunker_classes()
    if name in chunker_classes:
        return chunker_classes[name]
    
    # Factory functions
    factory_functions = _get_factory_functions()
    if name in factory_functions:
        return factory_functions[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "BaseChunker",
    "BaseChunkingConfig",
    "CharacterChunker",
    "CharacterChunkingConfig",
    "RecursiveChunker",
    "RecursiveChunkingConfig",
    "HTMLChunker",
    "HTMLChunkingConfig",
    "JSONChunker",
    "JSONChunkingConfig",
    "MarkdownChunker",
    "MarkdownChunkingConfig",
    "PythonChunker",
    "PythonChunkingConfig",
    "SemanticChunker",
    "SemanticChunkingConfig",
    "AgenticChunker",
    "AgenticChunkingConfig",
    "create_chunking_strategy",
    "create_adaptive_strategy",
    "create_rag_strategy",
    "create_semantic_search_strategy",
    "create_fast_strategy",
    "create_quality_strategy",
    "create_intelligent_splitters",
    "list_available_strategies",
    "get_strategy_info",
    "detect_content_type",
    "recommend_strategy_for_content",
    "ContentType",
    "ChunkingUseCase",
]