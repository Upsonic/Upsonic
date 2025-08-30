from __future__ import annotations
from typing import Dict, Type, List, Any, Optional, Union
import re
from enum import Enum

from .base import ChunkingStrategy, ChunkingConfig, ChunkingMode
from ..schemas.data_models import Document
from ..utils.package.exception import ConfigurationError


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
    STRUCTURED_DATA = "structured_data"


class ChunkingUseCase(Enum):
    """Different use cases for chunking optimization."""
    RAG_RETRIEVAL = "rag_retrieval"
    SEMANTIC_SEARCH = "semantic_search"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CLASSIFICATION = "classification"
    GENERAL = "general"


_STRATEGY_REGISTRY: Dict[str, Type[ChunkingStrategy]] = {}
_CONFIG_REGISTRY: Dict[str, Type[ChunkingConfig]] = {}


def register_chunking_strategy(name: str, strategy_class: Type[ChunkingStrategy], config_class: Type[ChunkingConfig] = None):
    """Register a chunking strategy for use with the factory."""
    _STRATEGY_REGISTRY[name] = strategy_class
    if config_class:
        _CONFIG_REGISTRY[name] = config_class


def _lazy_import_strategies():
    """Lazy import all chunking strategies to populate registry."""
    global _STRATEGY_REGISTRY, _CONFIG_REGISTRY
    
    if _STRATEGY_REGISTRY:
        return
    
    try:
        from .character import CharacterChunkingStrategy, CharacterChunkingConfig
        register_chunking_strategy("character", CharacterChunkingStrategy, CharacterChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .recursive import RecursiveCharacterChunkingStrategy, RecursiveChunkingConfig
        register_chunking_strategy("recursive", RecursiveCharacterChunkingStrategy, RecursiveChunkingConfig)
        register_chunking_strategy("recursive_character", RecursiveCharacterChunkingStrategy, RecursiveChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .semantic import SemanticSimilarityChunkingStrategy, SemanticChunkingConfig
        register_chunking_strategy("semantic", SemanticSimilarityChunkingStrategy, SemanticChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .markdown import MarkdownHeaderChunkingStrategy, MarkdownRecursiveChunkingStrategy, MarkdownHeaderChunkingConfig, MarkdownChunkingConfig
        register_chunking_strategy("markdown_header", MarkdownHeaderChunkingStrategy, MarkdownHeaderChunkingConfig)
        register_chunking_strategy("markdown_recursive", MarkdownRecursiveChunkingStrategy, MarkdownChunkingConfig)
        register_chunking_strategy("markdown", MarkdownRecursiveChunkingStrategy, MarkdownChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .html import HTMLChunkingStrategy, HTMLChunkingConfig
        register_chunking_strategy("html", HTMLChunkingStrategy, HTMLChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .json import JSONChunkingStrategy, JSONChunkingConfig
        register_chunking_strategy("json", JSONChunkingStrategy, JSONChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .python import PythonCodeChunkingStrategy, PythonCodeChunkingConfig
        register_chunking_strategy("python", PythonCodeChunkingStrategy, PythonCodeChunkingConfig)
        register_chunking_strategy("code", PythonCodeChunkingStrategy, PythonCodeChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .agentic import AgenticChunkingStrategy, AgenticChunkingConfig
        register_chunking_strategy("agentic", AgenticChunkingStrategy, AgenticChunkingConfig)
        register_chunking_strategy("ai", AgenticChunkingStrategy, AgenticChunkingConfig)
    except ImportError:
        pass
    
    try:
        from .rule import RuleBasedChunkingStrategy, RuleBasedChunkingConfig
        register_chunking_strategy("rule_based", RuleBasedChunkingStrategy, RuleBasedChunkingConfig)
        register_chunking_strategy("rule", RuleBasedChunkingStrategy, RuleBasedChunkingConfig)
        globals()["RuleBasedChunkingConfig"] = RuleBasedChunkingConfig
    except ImportError:
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
            "description": "Simple character-based splitting",
            "best_for": ["Simple text", "Fixed-size chunks"],
            "features": ["Fast", "Predictable sizes"],
            "use_cases": ["Basic RAG", "Simple search"]
        },
        "recursive": {
            "description": "Intelligent recursive splitting with separator prioritization",
            "best_for": ["General text", "Mixed content"],
            "features": ["Adaptive", "Content-aware", "Boundary preservation"],
            "use_cases": ["RAG", "Semantic search", "General purpose"]
        },
        "semantic": {
            "description": "Semantic similarity-based chunking",
            "best_for": ["Narrative text", "Topic-based splitting"],
            "features": ["Topic coherence", "Semantic boundaries"],
            "use_cases": ["High-quality RAG", "Topic analysis"]
        },
        "markdown": {
            "description": "Markdown-aware chunking with header structure",
            "best_for": ["Documentation", "Structured markdown"],
            "features": ["Header preservation", "Structure-aware"],
            "use_cases": ["Documentation RAG", "Knowledge bases"]
        },
        "html": {
            "description": "HTML-aware chunking with element preservation",
            "best_for": ["Web content", "HTML documents"],
            "features": ["Tag-aware", "Content extraction", "Multiple modes"],
            "use_cases": ["Web scraping", "Content extraction"]
        },
        "json": {
            "description": "JSON structure-preserving chunking",
            "best_for": ["JSON data", "Structured data"],
            "features": ["Structure preservation", "Path tracking"],
            "use_cases": ["API data", "Configuration files"]
        },
        "python": {
            "description": "Python code-aware chunking",
            "best_for": ["Python code", "Source code"],
            "features": ["Syntax-aware", "Function/class boundaries"],
            "use_cases": ["Code analysis", "Documentation generation"]
        },
        "agentic": {
            "description": "AI-powered intelligent chunking",
            "best_for": ["Complex documents", "Maximum quality"],
            "features": ["AI analysis", "Thematic coherence", "Adaptive"],
            "use_cases": ["Premium RAG", "Research documents"]
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
        quality_preference: Speed vs quality preference
        
    Returns:
        Recommended strategy name
    """
    _lazy_import_strategies()
    available = set(list_available_strategies())
    
    if content_type == ContentType.MARKDOWN and "markdown" in available:
        return "markdown"
    elif content_type == ContentType.HTML and "html" in available:
        return "html"
    elif content_type == ContentType.JSON and "json" in available:
        return "json"
    elif content_type in [ContentType.PYTHON, ContentType.JAVASCRIPT] and "python" in available:
        return "python"
    
    if use_case == ChunkingUseCase.SEMANTIC_SEARCH and "semantic" in available:
        if quality_preference == "quality":
            try:
                from upsonic.embeddings.factory import create_embedding_provider
                create_embedding_provider("openai")
                return "semantic"
            except (ImportError, ConfigurationError):
                pass
    
    if use_case in [ChunkingUseCase.RAG_RETRIEVAL, ChunkingUseCase.QUESTION_ANSWERING]:
        if quality_preference == "quality" and "agentic" in available and content_length < 50000:
            try:
                return "agentic"
            except:
                pass
        elif "semantic" in available and quality_preference != "fast":
            try:
                from upsonic.embeddings.factory import create_embedding_provider
                create_embedding_provider("openai")
                return "semantic"
            except (ImportError, ConfigurationError):
                pass
    
    if quality_preference == "fast" or content_length > 100000:
        if "character" in available:
            return "character"
    
    if "recursive" in available:
        return "recursive"
    elif "character" in available:
        return "character"
    
    return list(available)[0] if available else "recursive"


def create_chunking_strategy(
    strategy: str,
    config: Optional[Union[ChunkingConfig, Dict[str, Any]]] = None,
    **kwargs
) -> ChunkingStrategy:
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
    
    if strategy in ["rule_based", "rule"]:
        if config is None:
            config = {}
        
        if isinstance(config, dict):
            if "rules" not in config and "rules" in kwargs:
                config["rules"] = kwargs.pop("rules")
            if "default_strategy" not in config and "default_strategy" in kwargs:
                config["default_strategy"] = kwargs.pop("default_strategy")
            
            if "rules" not in config:
                config["rules"] = []
            if "default_strategy" not in config:
                config["default_strategy"] = "recursive"
            
            merged_config = {**config, **kwargs}
            config = config_class(**merged_config) if config_class else RuleBasedChunkingConfig(**merged_config)
        elif kwargs:
            print(f"Warning: Both config object and kwargs provided. Using config object, ignoring kwargs: {list(kwargs.keys())}")
        
        return strategy_class(config=config)
    
    if strategy == "semantic":
        embedding_provider = kwargs.pop("embedding_provider", None)
        if embedding_provider is None:
            try:
                from upsonic.embeddings.factory import create_embedding_provider
                embedding_provider = create_embedding_provider("openai")
            except (ImportError, ConfigurationError) as e:
                error_code = getattr(e, 'error_code', 'MISSING_EMBEDDING_PROVIDER')
                raise ConfigurationError(
                    f"Semantic strategy requires an embedding_provider. Original error: {str(e)}",
                    error_code=error_code
                )
        
        if config is None:
            if config_class:
                config = config_class(**kwargs)
            else:
                config = ChunkingConfig(**kwargs)
        elif isinstance(config, dict):
            merged_config = {**config, **kwargs}
            if config_class:
                config = config_class(**merged_config)
            else:
                config = ChunkingConfig(**merged_config)
        elif kwargs:
            print(f"Warning: Both config object and kwargs provided. Using config object, ignoring kwargs: {list(kwargs.keys())}")
        
        return strategy_class(embedding_provider, config=config)
    
    if strategy in ["agentic", "ai"]:
        agent = kwargs.pop("agent", None)
        if agent is None:
            raise ConfigurationError(
                "Agentic strategy requires an agent. Please provide one.",
                error_code="MISSING_AGENT"
            )
        
        if config is None:
            if config_class:
                config = config_class(**kwargs)
            else:
                config = ChunkingConfig(**kwargs)
        elif isinstance(config, dict):
            merged_config = {**config, **kwargs}
            if config_class:
                config = config_class(**merged_config)
            else:
                config = ChunkingConfig(**merged_config)
        elif kwargs:
            print(f"Warning: Both config object and kwargs provided. Using config object, ignoring kwargs: {list(kwargs.keys())}")
        
        return strategy_class(agent, config=config)
    
    if config is None:
        if config_class:
            config = config_class(**kwargs)
        else:
            config = ChunkingConfig(**kwargs)
    elif isinstance(config, dict):
        merged_config = {**config, **kwargs}
        if config_class:
            config = config_class(**merged_config)
        else:
            config = ChunkingConfig(**merged_config)
    elif kwargs:
        print(f"Warning: Both config object and kwargs provided. Using config object, ignoring kwargs: {list(kwargs.keys())}")
    
    return strategy_class(config=config)


def create_adaptive_strategy(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    use_case: ChunkingUseCase = ChunkingUseCase.GENERAL,
    quality_preference: str = "balanced",
    **kwargs
) -> ChunkingStrategy:
    """
    Create an adaptive chunking strategy based on content analysis.
    
    Args:
        content: Text content to analyze
        metadata: Optional metadata
        use_case: Intended use case
        quality_preference: Speed vs quality preference
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
    
    print(f"Auto-selected {strategy_name} strategy for {content_type.value} content")
    
    return create_chunking_strategy(strategy_name, config=optimized_config)


def _create_optimized_config(
    content: str,
    content_type: ContentType,
    strategy_name: str,
    **kwargs
) -> Dict[str, Any]:
    """Create optimized configuration based on content analysis."""
    config = kwargs.copy()
    
    content_length = len(content)
    line_count = content.count('\n')
    avg_line_length = content_length / max(line_count, 1)
    
    if content_type in [ContentType.CODE, ContentType.PYTHON, ContentType.JAVASCRIPT]:
        config.setdefault('chunk_size', 1500)
        config.setdefault('preserve_sentences', False)
    elif content_type == ContentType.TECHNICAL_DOC:
        config.setdefault('chunk_size', 1200)
        config.setdefault('preserve_sentences', True)
    elif content_type == ContentType.NARRATIVE:
        config.setdefault('chunk_size', 800)
        config.setdefault('preserve_sentences', True)
    else:
        config.setdefault('chunk_size', 1000)
    
    if content_type in [ContentType.NARRATIVE, ContentType.TECHNICAL_DOC]:
        config.setdefault('chunk_overlap', int(config['chunk_size'] * 0.25))
    else:
        config.setdefault('chunk_overlap', int(config['chunk_size'] * 0.15))
    
    if strategy_name == "recursive":
        if content_type == ContentType.MARKDOWN:
            config.setdefault('separators', ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""])
        elif content_type in [ContentType.CODE, ContentType.PYTHON]:
            config.setdefault('separators', ["\nclass ", "\ndef ", "\n    def ", "\n\n", "\n", " ", ""])
    
    return config


def create_rag_strategy(content: str = "", **kwargs) -> ChunkingStrategy:
    """Create optimal strategy for RAG use case."""
    if content:
        return create_adaptive_strategy(
            content, 
            use_case=ChunkingUseCase.RAG_RETRIEVAL,
            quality_preference="balanced",
            **kwargs
        )
    else:
        return create_chunking_strategy("recursive", **kwargs)


def create_semantic_search_strategy(content: str = "", **kwargs) -> ChunkingStrategy:
    """Create optimal strategy for semantic search."""
    if content:
        return create_adaptive_strategy(
            content,
            use_case=ChunkingUseCase.SEMANTIC_SEARCH,
            quality_preference="quality",
            **kwargs
        )
    else:
        try:
            return create_chunking_strategy("semantic", **kwargs)
        except ConfigurationError as e:
            error_code = getattr(e, 'error_code', '')
            if (error_code in ['MISSING_EMBEDDING_PROVIDER', 'API_KEY_MISSING', 'DEPENDENCY_MISSING'] or 
                "OpenAI API key not found" in str(e) or "embedding_provider" in str(e)):
                print("Warning: No embedding provider available, falling back to recursive strategy")
                return create_chunking_strategy("recursive", **kwargs)
            else:
                raise


def create_fast_strategy(**kwargs) -> ChunkingStrategy:
    """Create fast chunking strategy for large documents."""
    return create_chunking_strategy("character", **kwargs)


def create_quality_strategy(content: str = "", **kwargs) -> ChunkingStrategy:
    """Create highest quality strategy available."""
    if content:
        return create_adaptive_strategy(
            content,
            quality_preference="quality",
            **kwargs
        )
    else:
        available = list_available_strategies()
        if "agentic" in available:
            try:
                return create_chunking_strategy("agentic", **kwargs)
            except ConfigurationError as e:
                error_code = getattr(e, 'error_code', '')
                if error_code == 'MISSING_AGENT' or "agent" in str(e).lower():
                    print("Warning: No agent available, trying semantic strategy")
                else:
                    raise
        
        if "semantic" in available:
            try:
                return create_chunking_strategy("semantic", **kwargs)
            except ConfigurationError as e:
                error_code = getattr(e, 'error_code', '')
                if (error_code in ['MISSING_EMBEDDING_PROVIDER', 'API_KEY_MISSING', 'DEPENDENCY_MISSING'] or 
                    "OpenAI API key not found" in str(e) or "embedding_provider" in str(e)):
                    print("Warning: No embedding provider available, falling back to recursive strategy")
                else:
                    raise
        
        return create_chunking_strategy("recursive", **kwargs)
