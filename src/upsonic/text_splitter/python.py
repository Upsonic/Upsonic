from __future__ import annotations
from typing import List, Optional
from pydantic import Field

from upsonic.text_splitter.recursive import RecursiveCharacterChunkingStrategy, RecursiveChunkingConfig

PYTHON_SEPARATORS = [
    "\nclass ",
    "\ndef ",
    "\n    def ",
    "\n\n",
    "\n",
    " ",
]


class PythonCodeChunkingConfig(RecursiveChunkingConfig):
    """Enhanced configuration for Python code chunking."""
    separators: List[str] = Field(default_factory=lambda: PYTHON_SEPARATORS, description="Python-optimized separators")
    preserve_class_integrity: bool = Field(True, description="Keep classes together when possible")
    preserve_function_integrity: bool = Field(True, description="Keep functions together when possible")
    include_imports_context: bool = Field(True, description="Include import context in chunks")
    
    enable_ast_analysis: bool = Field(False, description="Use AST for intelligent splitting")
    detect_code_complexity: bool = Field(True, description="Analyze code complexity for better splitting")
    preserve_docstrings: bool = Field(True, description="Keep docstrings with their functions/classes")


class PythonCodeChunkingStrategy(RecursiveCharacterChunkingStrategy):
    """
    Python code chunking strategy with framework-level features.

    This specialist strategy understands Python syntax and splits code along
    logical boundaries while preserving code structure and semantics.
    
    Features:
    - AST-based code analysis and structure detection
    - Class and function integrity preservation
    - Import context management and analysis
    - Code complexity detection and adaptive splitting
    - Docstring preservation and association
    - Python-optimized separator prioritization
    - Metadata with code structure information
    
    """

    def __init__(self, config: Optional[PythonCodeChunkingConfig] = None):
        """
        Initialize enhanced Python code chunking strategy.

        Args:
            config: Configuration object with Python-specific settings
        """
        if config is None:
            config = PythonCodeChunkingConfig()
        
        super().__init__(config)