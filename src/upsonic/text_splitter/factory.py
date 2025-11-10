from __future__ import annotations
from typing import Dict, Type, List, Any, Optional, Union
import re
from enum import Enum
from pathlib import Path

from .base import BaseChunker, BaseChunkingConfig
from ..schemas.data_models import Document
from ..utils.package.exception import ConfigurationError
from upsonic.utils.printing import warning_log, info_log


class ContentType(Enum):
    """Detected content types for chunking."""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown" 
    HTML = "html"
    CODE = "code"
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TECHNICAL_DOC = "technical_doc"
    NARRATIVE = "narrative"


class ChunkingUseCase(Enum):
    """Different use cases for chunking optimization."""
    RAG_RETRIEVAL = "rag_retrieval"
    SEMANTIC_SEARCH = "semantic_search"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CLASSIFICATION = "classification"
    GENERAL = "general"


_STRATEGY_REGISTRY: Dict[str, Type[BaseChunker]] = {}
_CONFIG_REGISTRY: Dict[str, Type[BaseChunkingConfig]] = {}


def register_chunking_strategy(name: str, strategy_class: Type[BaseChunker], config_class: Type[BaseChunkingConfig] = None):
    """
    Register a chunking strategy for use with the factory.
    
    Args:
        name: Strategy name (must be unique)
        strategy_class: Chunker class that inherits from BaseChunker
        config_class: Optional config class that inherits from BaseChunkingConfig
        
    Raises:
        ValueError: If name is already registered or invalid
        TypeError: If strategy_class doesn't inherit from BaseChunker
    """
    if not name or not isinstance(name, str):
        raise ValueError("Strategy name must be a non-empty string")
    
    if not issubclass(strategy_class, BaseChunker):
        raise TypeError(f"Strategy class must inherit from BaseChunker, got {strategy_class}")
    
    if config_class and not issubclass(config_class, BaseChunkingConfig):
        raise TypeError(f"Config class must inherit from BaseChunkingConfig, got {config_class}")
    
    if name in _STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{name}' is already registered")
    
    _STRATEGY_REGISTRY[name] = strategy_class
    if config_class:
        _CONFIG_REGISTRY[name] = config_class


def unregister_chunking_strategy(name: str) -> bool:
    """
    Unregister a chunking strategy from the factory.
    
    Args:
        name: Strategy name to unregister
        
    Returns:
        True if strategy was unregistered, False if it wasn't found
    """
    if name in _STRATEGY_REGISTRY:
        del _STRATEGY_REGISTRY[name]
        if name in _CONFIG_REGISTRY:
            del _CONFIG_REGISTRY[name]
        return True
    return False


def clear_strategy_registry():
    """Clear all registered strategies (useful for testing)."""
    global _STRATEGY_REGISTRY, _CONFIG_REGISTRY
    _STRATEGY_REGISTRY.clear()
    _CONFIG_REGISTRY.clear()


def _lazy_import_strategies():
    """Lazy import all chunking strategies to populate registry."""
    global _STRATEGY_REGISTRY, _CONFIG_REGISTRY
    
    if _STRATEGY_REGISTRY:
        return
    
    # Import and register all available strategies using try/except blocks
    try:
        from .character import CharacterChunker, CharacterChunkingConfig
        register_chunking_strategy("character", CharacterChunker, CharacterChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .recursive import RecursiveChunker, RecursiveChunkingConfig
        register_chunking_strategy("recursive", RecursiveChunker, RecursiveChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .semantic import SemanticChunker, SemanticChunkingConfig
        register_chunking_strategy("semantic", SemanticChunker, SemanticChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .markdown import MarkdownChunker, MarkdownChunkingConfig
        register_chunking_strategy("markdown", MarkdownChunker, MarkdownChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .html_chunker import HTMLChunker, HTMLChunkingConfig
        register_chunking_strategy("html", HTMLChunker, HTMLChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .json_chunker import JSONChunker, JSONChunkingConfig
        register_chunking_strategy("json", JSONChunker, JSONChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .python import PythonChunker, PythonChunkingConfig
        register_chunking_strategy("python", PythonChunker, PythonChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .agentic import AgenticChunker, AgenticChunkingConfig
        register_chunking_strategy("agentic", AgenticChunker, AgenticChunkingConfig)
    except ImportError:
        pass
    
    # Register aliases for common use cases
    try:
        if "recursive" in _STRATEGY_REGISTRY:
            register_chunking_strategy("recursive_character", _STRATEGY_REGISTRY["recursive"], _CONFIG_REGISTRY.get("recursive"))
    except Exception:
        pass
    
    try:
        if "markdown" in _STRATEGY_REGISTRY:
            register_chunking_strategy("markdown_header", _STRATEGY_REGISTRY["markdown"], _CONFIG_REGISTRY.get("markdown"))
            register_chunking_strategy("markdown_recursive", _STRATEGY_REGISTRY["markdown"], _CONFIG_REGISTRY.get("markdown"))
    except Exception:
        pass
    
    try:
        if "python" in _STRATEGY_REGISTRY:
            register_chunking_strategy("code", _STRATEGY_REGISTRY["python"], _CONFIG_REGISTRY.get("python"))
            register_chunking_strategy("py", _STRATEGY_REGISTRY["python"], _CONFIG_REGISTRY.get("python"))
            register_chunking_strategy("python_code", _STRATEGY_REGISTRY["python"], _CONFIG_REGISTRY.get("python"))
    except Exception:
        pass
    
    try:
        if "agentic" in _STRATEGY_REGISTRY:
            register_chunking_strategy("ai", _STRATEGY_REGISTRY["agentic"], _CONFIG_REGISTRY.get("agentic"))
    except Exception:
        pass
    



def list_available_strategies() -> List[str]:
    """List all available chunking strategies."""
    _lazy_import_strategies()
    return list(_STRATEGY_REGISTRY.keys())


def get_strategy_info() -> Dict[str, Dict[str, Any]]:
    """Get detailed information about all available strategies."""
    _lazy_import_strategies()
    
    strategy_info = {
        "character": {
            "description": "Simple character-based splitting with configurable separator",
            "best_for": ["Simple text", "Fixed-size chunks", "Consistent delimiters"],
            "features": ["Fast", "Predictable sizes", "Regex support", "Separator control"],
            "use_cases": ["Basic RAG", "Simple search", "Log files"],
            "config_params": ["separator", "is_separator_regex", "keep_separator"]
        },
        "recursive": {
            "description": "Intelligent recursive splitting with separator prioritization",
            "best_for": ["General text", "Mixed content", "Structured documents"],
            "features": ["Adaptive", "Content-aware", "Boundary preservation", "Language-specific"],
            "use_cases": ["RAG", "Semantic search", "General purpose", "Code chunking"],
            "config_params": ["separators", "keep_separator", "is_separator_regex"]
        },
        "semantic": {
            "description": "Semantic similarity-based chunking using embeddings",
            "best_for": ["Narrative text", "Topic-based splitting", "Coherent content"],
            "features": ["Topic coherence", "Semantic boundaries", "Statistical thresholds"],
            "use_cases": ["High-quality RAG", "Topic analysis", "Research documents"],
            "config_params": ["embedding_provider", "breakpoint_threshold_type", "breakpoint_threshold_amount", "sentence_splitter"]
        },
        "markdown": {
            "description": "Markdown-aware chunking with header structure preservation",
            "best_for": ["Documentation", "Structured markdown", "Technical docs"],
            "features": ["Header preservation", "Structure-aware", "Element filtering"],
            "use_cases": ["Documentation RAG", "Knowledge bases", "Wiki content"],
            "config_params": ["split_on_elements", "preserve_whole_elements", "strip_elements", "preserve_original_content"]
        },
        "html": {
            "description": "HTML-aware chunking with element preservation and content extraction",
            "best_for": ["Web content", "HTML documents", "Structured web pages"],
            "features": ["Tag-aware", "Content extraction", "Multiple modes", "Link processing"],
            "use_cases": ["Web scraping", "Content extraction", "Website analysis"],
            "config_params": ["split_on_tags", "tags_to_ignore", "tags_to_extract", "preserve_whole_tags", "extract_link_info", "preserve_html_content", "merge_small_chunks"]
        },
        "json": {
            "description": "JSON structure-preserving chunking with path tracking",
            "best_for": ["JSON data", "Structured data", "API responses"],
            "features": ["Structure preservation", "Path tracking", "List conversion"],
            "use_cases": ["API data", "Configuration files", "Data processing"],
            "config_params": ["convert_lists_to_dicts", "max_depth", "json_encoder_options"]
        },
        "python": {
            "description": "Python code-aware chunking using AST parsing",
            "best_for": ["Python code", "Source code", "Code documentation"],
            "features": ["Syntax-aware", "Function/class boundaries", "AST-based"],
            "use_cases": ["Code analysis", "Documentation generation", "Code search"],
            "config_params": ["split_on_nodes", "min_chunk_lines", "include_docstrings", "strip_decorators"]
        },
        "agentic": {
            "description": "AI-powered intelligent chunking with proposition extraction",
            "best_for": ["Complex documents", "Maximum quality", "Research content"],
            "features": ["AI analysis", "Thematic coherence", "Adaptive", "Quality scoring"],
            "use_cases": ["Premium RAG", "Research documents", "High-quality search"],
            "config_params": ["max_agent_retries", "min_proposition_length", "max_propositions_per_chunk", "enable_caching", "fallback_to_recursive"]
        }
    }
    
    available_info = {}
    for strategy in list_available_strategies():
        base_strategy = strategy.split("_")[0] if "_" in strategy else strategy
        if base_strategy in strategy_info:
            available_info[strategy] = strategy_info[base_strategy]
    
    return available_info


def detect_content_type(content: str, metadata: Optional[Dict[str, Any]] = None) -> ContentType:
    """
    Detect content type from text content and metadata.
    
    Args:
        content: Text content to analyze
        metadata: Optional metadata with hints like file extension
        
    Returns:
        Detected ContentType
    """
    if not content.strip():
        return ContentType.PLAIN_TEXT
    
    if metadata:
        source = metadata.get('source', '').lower()
        file_name = metadata.get('file_name', '').lower()
        
        if any(ext in source or ext in file_name for ext in ['.md', '.markdown']):
            return ContentType.MARKDOWN
        elif any(ext in source or ext in file_name for ext in ['.html', '.htm']):
            return ContentType.HTML
        elif any(ext in source or ext in file_name for ext in ['.json']):
            return ContentType.JSON
        elif any(ext in source or ext in file_name for ext in ['.csv']):
            return ContentType.CSV
        elif any(ext in source or ext in file_name for ext in ['.xml']):
            return ContentType.XML
        elif any(ext in source or ext in file_name for ext in ['.py']):
            return ContentType.PYTHON
        elif any(ext in source or ext in file_name for ext in ['.js', '.jsx', '.ts', '.tsx']):
            return ContentType.JAVASCRIPT
    
    content_sample = content[:2000]
    
    if re.search(r'<[^>]+>', content_sample) and any(tag in content_sample.lower() for tag in ['<html', '<div', '<p>', '<span']):
        return ContentType.HTML
    
    markdown_patterns = [
        r'^#{1,6}\s',
        r'\*\*.*?\*\*',
        r'^\s*[-*+]\s',
        r'```',
        r'\[.*?\]\(.*?\)'
    ]
    if any(re.search(pattern, content_sample, re.MULTILINE) for pattern in markdown_patterns):
        return ContentType.MARKDOWN
    
    try:
        import json
        json.loads(content_sample)
        return ContentType.JSON
    except:
        pass
    
    if content_sample.strip().startswith('<?xml') or re.search(r'<\w+[^>]*>.*?</\w+>', content_sample):
        return ContentType.XML
    
    python_keywords = ['def ', 'class ', 'import ', 'from ', 'if __name__']
    if any(keyword in content_sample for keyword in python_keywords):
        return ContentType.PYTHON
    
    js_keywords = ['function ', 'const ', 'let ', 'var ', '=>', 'console.log']
    if any(keyword in content_sample for keyword in js_keywords):
        return ContentType.JAVASCRIPT
    
    tech_indicators = ['API', 'endpoint', 'parameter', 'response', 'documentation', 'specification']
    if sum(content_sample.lower().count(indicator.lower()) for indicator in tech_indicators) > 3:
        return ContentType.TECHNICAL_DOC
    
    sentence_count = len(re.findall(r'[.!?]+', content_sample))
    word_count = len(content_sample.split())
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    if avg_sentence_length > 15 and sentence_count > 3:
        return ContentType.NARRATIVE
    
    return ContentType.PLAIN_TEXT


def recommend_strategy_for_content(
    content_type: ContentType,
    use_case: ChunkingUseCase = ChunkingUseCase.GENERAL,
    content_length: int = 0,
    quality_preference: str = "balanced"
) -> str:
    """
    Recommend the best chunking strategy based on content analysis.
    
    Args:
        content_type: Detected content type
        use_case: Intended use case
        content_length: Length of content in characters
        quality_preference: Speed vs quality preference ("fast", "balanced", "quality")
        
    Returns:
        Recommended strategy name
        
    Quality Preference Behavior:
        - "fast": Prioritizes speed, uses character-based chunking
        - "balanced": Balances quality and performance, uses semantic or recursive
        - "quality": Maximizes quality, uses agentic or semantic strategies
        
    Note:
        This function only recommends strategies. It does not check for embedding provider
        availability. The caller must provide embedding_provider if semantic strategy is selected.
    """
    _lazy_import_strategies()
    available = set(list_available_strategies())
    
    # Content-specific strategies (override quality preference for structured content)
    if content_type == ContentType.MARKDOWN and "markdown" in available:
        return "markdown"
    elif content_type == ContentType.HTML and "html" in available:
        return "html"
    elif content_type == ContentType.JSON and "json" in available:
        return "json"
    elif content_type in [ContentType.PYTHON, ContentType.JAVASCRIPT] and "python" in available:
        return "python"
    
    # Fast preference: prioritize speed over quality
    if quality_preference == "fast" or content_length > 100000:
        if "character" in available:
            return "character"
        elif "recursive" in available:
            return "recursive"
    
    # Quality preference: maximize quality regardless of cost
    if quality_preference == "quality":
        if use_case == ChunkingUseCase.SEMANTIC_SEARCH and "semantic" in available:
            return "semantic"
        
        if use_case in [ChunkingUseCase.RAG_RETRIEVAL, ChunkingUseCase.QUESTION_ANSWERING]:
            # Try agentic first for highest quality (if content is not too large)
            if "agentic" in available and content_length < 50000:
                return "agentic"
            # Fall back to semantic for quality
            if "semantic" in available:
                return "semantic"
    
    # Balanced preference: good quality without expensive operations
    if quality_preference == "balanced":
        if use_case == ChunkingUseCase.SEMANTIC_SEARCH and "semantic" in available:
            # Use semantic for semantic search but with reasonable content size limits
            if content_length < 75000:
                return "semantic"
        
        if use_case in [ChunkingUseCase.RAG_RETRIEVAL, ChunkingUseCase.QUESTION_ANSWERING]:
            # Skip expensive agentic, use semantic for balanced approach
            if "semantic" in available and content_length < 75000:
                return "semantic"
        
        # For balanced, prefer recursive over character
        if "recursive" in available:
            return "recursive"
    
    # Default fallback strategy
    if "recursive" in available:
        return "recursive"
    elif "character" in available:
        return "character"
    
    return list(available)[0] if available else "recursive"


def create_chunking_strategy(
    strategy: str,
    config: Optional[Union[BaseChunkingConfig, Dict[str, Any]]] = None,
    **kwargs
) -> BaseChunker:
    """
    Create a chunking strategy using the factory pattern.
    
    Args:
        strategy: Strategy name
        config: Configuration object or dictionary
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured chunking strategy instance
    """
    _lazy_import_strategies()
    
    strategy = strategy.lower().replace("-", "_")
    
    if strategy not in _STRATEGY_REGISTRY:
        available = ", ".join(list_available_strategies())
        raise ConfigurationError(
            f"Unknown chunking strategy '{strategy}'. Available strategies: {available}",
            error_code="UNKNOWN_STRATEGY"
        )
    
    strategy_class = _STRATEGY_REGISTRY[strategy]
    config_class = _CONFIG_REGISTRY.get(strategy)
    
    # Handle special cases that require additional parameters
    if strategy == "semantic":
        embedding_provider = kwargs.pop("embedding_provider", None)
        if embedding_provider is None:
            raise ConfigurationError(
                "Semantic strategy requires an embedding_provider. Please provide one via the embedding_provider parameter.",
                error_code="MISSING_EMBEDDING_PROVIDER"
            )
        kwargs["embedding_provider"] = embedding_provider
    
    elif strategy in ["agentic", "ai"]:
        agent = kwargs.pop("agent", None)
        if agent is None:
            # Create a default agent if none is provided
            try:
                from upsonic.agent.agent import Agent
                agent = Agent("openai/gpt-4o")
                info_log("Created default agent for agentic chunking strategy", context="TextSplitterFactory")
            except (ImportError, Exception) as e:
                raise ConfigurationError(
                    f"Agentic strategy requires an agent. Failed to create default agent: {str(e)}",
                    error_code="MISSING_AGENT"
                )
        
        # Create config for agentic strategy
        final_config = _create_final_config(config, config_class, kwargs)
        return strategy_class(agent, config=final_config)
    
    # Create config for standard strategies
    final_config = _create_final_config(config, config_class, kwargs)
    return strategy_class(config=final_config)


def _create_final_config(
    config: Optional[Union[BaseChunkingConfig, Dict[str, Any]]],
    config_class: Optional[Type[BaseChunkingConfig]],
    kwargs: Dict[str, Any]
) -> BaseChunkingConfig:
    """Helper function to create final configuration object."""
    if config is None:
        if config_class:
            return config_class(**kwargs)
        else:
            return BaseChunkingConfig(**kwargs)
    elif isinstance(config, dict):
        merged_config = {**config, **kwargs}
        if config_class:
            return config_class(**merged_config)
        else:
            return BaseChunkingConfig(**merged_config)
    elif kwargs:
        warning_log(f"Both config object and kwargs provided. Using config object, ignoring kwargs: {list(kwargs.keys())}", context="TextSplitterFactory")
    
    return config


def create_adaptive_strategy(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    use_case: ChunkingUseCase = ChunkingUseCase.GENERAL,
    quality_preference: str = "balanced",
    embedding_provider: Optional[Any] = None,
    agent: Optional[Any] = None,
    **kwargs
) -> BaseChunker:
    """
    Create an adaptive chunking strategy based on content analysis.
    
    Args:
        content: Text content to analyze
        metadata: Optional metadata
        use_case: Intended use case
        quality_preference: Speed vs quality preference
        embedding_provider: Optional embedding provider (required if semantic strategy is selected)
        agent: Optional agent (will be created if agentic strategy is selected and none provided)
        **kwargs: Additional configuration
        
    Returns:
        Optimally configured chunking strategy
    """
    content_type = detect_content_type(content, metadata)
    
    strategy_name = recommend_strategy_for_content(
        content_type=content_type,
        use_case=use_case,
        content_length=len(content),
        quality_preference=quality_preference
    )
    
    optimized_config = _create_optimized_config(content, content_type, strategy_name, **kwargs)
    
    # Pass embedding_provider if semantic strategy is selected
    if strategy_name == "semantic" and embedding_provider is not None:
        optimized_config["embedding_provider"] = embedding_provider
    
    # Pass agent if agentic strategy is selected
    if strategy_name == "agentic" and agent is not None:
        optimized_config["agent"] = agent
    
    info_log(f"Auto-selected {strategy_name} strategy for {content_type.value} content", context="TextSplitterFactory")
    
    return create_chunking_strategy(strategy_name, config=optimized_config)


def _create_optimized_config(
    content: str,
    content_type: ContentType,
    strategy_name: str,
    **kwargs
) -> Dict[str, Any]:
    """Create optimized configuration based on content analysis."""
    config = kwargs.copy()
    
    # Set chunk size based on content type
    if content_type in [ContentType.CODE, ContentType.PYTHON, ContentType.JAVASCRIPT]:
        config.setdefault('chunk_size', 1500)
    elif content_type == ContentType.TECHNICAL_DOC:
        config.setdefault('chunk_size', 1200)
    elif content_type == ContentType.NARRATIVE:
        config.setdefault('chunk_size', 800)
    else:
        config.setdefault('chunk_size', 1000)
    
    # Set chunk overlap based on content type
    if content_type in [ContentType.NARRATIVE, ContentType.TECHNICAL_DOC]:
        config.setdefault('chunk_overlap', int(config['chunk_size'] * 0.25))
    else:
        config.setdefault('chunk_overlap', int(config['chunk_size'] * 0.15))
    
    # Set strategy-specific parameters
    if strategy_name == "recursive":
        if content_type == ContentType.MARKDOWN:
            config.setdefault('separators', ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""])
        elif content_type in [ContentType.CODE, ContentType.PYTHON]:
            config.setdefault('separators', ["\nclass ", "\ndef ", "\n    def ", "\n\n", "\n", " ", ""])
    
    return config


def create_intelligent_splitters(
    sources: List[Union[str, Path]],
    content_samples: Optional[List[str]] = None,
    use_case: ChunkingUseCase = ChunkingUseCase.RAG_RETRIEVAL,
    quality_preference: str = "balanced",
    embedding_provider: Optional[Any] = None,
    agent: Optional[Any] = None,
    **global_config_kwargs
) -> List[BaseChunker]:
    """
    Intelligently create appropriate chunking strategies for multiple sources.
    
    This method analyzes each source and creates the most appropriate chunking strategy
    with optimized configuration based on the source type, content, and use case.
    Handles both file paths (Path objects) and string content.
    
    Args:
        sources: List of source paths (Path objects) or content strings
        content_samples: Optional list of content samples for analysis (if sources are file paths)
        use_case: Intended use case for chunking optimization
        quality_preference: Speed vs quality preference
        embedding_provider: Optional embedding provider (required if semantic strategy is selected)
        agent: Optional agent (will be created if agentic strategy is selected and none provided)
        **global_config_kwargs: Global configuration options to apply to all strategies
        
    Returns:
        List of configured ChunkingStrategy instances
    """
    _lazy_import_strategies()
    
    if not sources:
        raise ValueError("At least one source must be provided")
    
    splitters = []
    
    for i, source in enumerate(sources):
        try:
            content_sample = ""
            source_str = str(source)
            
            if content_samples and i < len(content_samples):
                content_sample = content_samples[i]
            elif isinstance(source, str):
                content_sample = source_str
            elif isinstance(source, Path) and source.exists() and source.is_file():
                try:
                    with open(source, 'r', encoding='utf-8', errors='ignore') as f:
                        content_sample = f.read(5000)
                except Exception:
                    content_sample = source.name
            else:
                content_sample = source_str
            
            source_config = _create_source_optimized_config(
                source_str, content_sample, use_case, **global_config_kwargs
            )
            
            splitter = create_adaptive_strategy(
                content=content_sample,
                metadata={'source': source_str},
                use_case=use_case,
                quality_preference=quality_preference,
                embedding_provider=embedding_provider,
                agent=agent,
                **source_config
            )
            
            splitters.append(splitter)
            info_log(f"Created {splitter.__class__.__name__} for {source}", context="TextSplitterFactory")
            
        except Exception as e:
            warning_log(f"Failed to create intelligent splitter for {source}: {e}", context="TextSplitterFactory")
            try:
                fallback_config = _create_source_optimized_config(
                    source_str, "", use_case, **global_config_kwargs
                )
                fallback_splitter = create_chunking_strategy("recursive", **fallback_config)
                splitters.append(fallback_splitter)
                info_log(f"Using recursive strategy fallback for {source}", context="TextSplitterFactory")
            except Exception as fallback_error:
                warning_log(f"Fallback splitter also failed for {source}: {fallback_error}", context="TextSplitterFactory")
                raise
    
    return splitters


def _create_source_optimized_config(
    source: Union[str, Path],
    content_sample: str,
    use_case: ChunkingUseCase,
    **global_config_kwargs
) -> Dict[str, Any]:
    """
    Create optimized configuration for a specific source and content.
    
    Args:
        source: Source path (Path object or string) or content string
        content_sample: Sample of content for analysis
        use_case: Intended use case
        **global_config_kwargs: Global configuration options
        
    Returns:
        Optimized configuration dictionary
    """
    config = global_config_kwargs.copy()
    
    source_path = Path(source) if not isinstance(source, Path) else source
    
    if source_path.exists() and source_path.is_file():
        file_size = source_path.stat().st_size
        extension = source_path.suffix.lower()
        
        # Set chunk size based on file size
        if file_size > 100 * 1024 * 1024:  # > 100MB
            config.setdefault('chunk_size', 2000)
        elif file_size > 10 * 1024 * 1024:  # > 10MB
            config.setdefault('chunk_size', 1500)
        else:
            config.setdefault('chunk_size', 1000)
        
        # Set chunk overlap based on file type
        if extension in ['.md', '.markdown']:
            config.setdefault('chunk_overlap', int(config.get('chunk_size', 1000) * 0.2))
        elif extension in ['.py', '.js', '.ts', '.java', '.cpp', '.c']:
            config.setdefault('chunk_overlap', int(config.get('chunk_size', 1000) * 0.15))
        elif extension in ['.json', '.xml', '.yaml', '.yml']:
            config.setdefault('chunk_overlap', int(config.get('chunk_size', 1000) * 0.1))
        elif extension in ['.html', '.htm']:
            config.setdefault('chunk_overlap', int(config.get('chunk_size', 1000) * 0.25))
        elif extension in ['.pdf', '.docx']:
            config.setdefault('chunk_overlap', int(config.get('chunk_size', 1000) * 0.3))
        else:  # .txt, .csv, etc.
            config.setdefault('chunk_overlap', int(config.get('chunk_size', 1000) * 0.2))
    
    # Adjust chunk size based on content characteristics
    if content_sample:
        content_length = len(content_sample)
        line_count = content_sample.count('\n')
        avg_line_length = content_length / max(line_count, 1)
        
        if line_count > 0 and avg_line_length < 50:
            config.setdefault('chunk_size', min(config.get('chunk_size', 1000), 800))
        elif line_count > 0 and avg_line_length > 200:
            config.setdefault('chunk_size', max(config.get('chunk_size', 1000), 1200))
    
    # Adjust overlap based on use case
    if use_case == ChunkingUseCase.SEMANTIC_SEARCH:
        config.setdefault('chunk_overlap', int(config.get('chunk_size', 1000) * 0.3))
    elif use_case == ChunkingUseCase.SUMMARIZATION:
        config.setdefault('chunk_overlap', int(config.get('chunk_size', 1000) * 0.1))
    
    return config


def create_rag_strategy(content: str = "", embedding_provider: Optional[Any] = None, agent: Optional[Any] = None, **kwargs) -> BaseChunker:
    """Create optimal strategy for RAG use case."""
    if content:
        return create_adaptive_strategy(
            content, 
            use_case=ChunkingUseCase.RAG_RETRIEVAL,
            quality_preference="balanced",
            embedding_provider=embedding_provider,
            agent=agent,
            **kwargs
        )
    else:
        return create_chunking_strategy("recursive", **kwargs)


def create_semantic_search_strategy(content: str = "", embedding_provider: Optional[Any] = None, **kwargs) -> BaseChunker:
    """
    Create optimal strategy for semantic search.
    
    Args:
        content: Optional content string for adaptive strategy selection
        embedding_provider: Embedding provider (required if semantic strategy is selected)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured chunking strategy
        
    Raises:
        ConfigurationError: If semantic strategy is selected but embedding_provider is not provided
    """
    if content:
        return create_adaptive_strategy(
            content,
            use_case=ChunkingUseCase.SEMANTIC_SEARCH,
            quality_preference="quality",
            embedding_provider=embedding_provider,
            **kwargs
        )
    else:
        if embedding_provider is None:
            warning_log("No embedding provider provided, falling back to recursive strategy", context="TextSplitterFactory")
            return create_chunking_strategy("recursive", **kwargs)
        kwargs["embedding_provider"] = embedding_provider
        return create_chunking_strategy("semantic", **kwargs)


def create_fast_strategy(**kwargs) -> BaseChunker:
    """Create fast chunking strategy for large documents."""
    return create_chunking_strategy("character", **kwargs)


def create_quality_strategy(content: str = "", embedding_provider: Optional[Any] = None, agent: Optional[Any] = None, **kwargs) -> BaseChunker:
    """
    Create highest quality strategy available.
    
    Args:
        content: Optional content string for adaptive strategy selection
        embedding_provider: Optional embedding provider (required if semantic strategy is selected)
        agent: Optional agent (will be created if agentic strategy is selected and none provided)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured chunking strategy
    """
    if content:
        return create_adaptive_strategy(
            content,
            quality_preference="quality",
            embedding_provider=embedding_provider,
            agent=agent,
            **kwargs
        )
    else:
        available = list_available_strategies()
        if "agentic" in available:
            # Agent will be created automatically if not provided
            if agent is not None:
                kwargs["agent"] = agent
            try:
                return create_chunking_strategy("agentic", **kwargs)
            except ConfigurationError as e:
                error_code = getattr(e, 'error_code', '')
                if error_code == 'MISSING_AGENT' or "agent" in str(e).lower():
                    warning_log("Failed to create agent, trying semantic strategy", context="TextSplitterFactory")
                else:
                    raise
        
        if "semantic" in available:
            if embedding_provider is not None:
                kwargs["embedding_provider"] = embedding_provider
                try:
                    return create_chunking_strategy("semantic", **kwargs)
                except ConfigurationError as e:
                    raise
            else:
                warning_log("No embedding provider provided, falling back to recursive strategy", context="TextSplitterFactory")
        
        return create_chunking_strategy("recursive", **kwargs)
