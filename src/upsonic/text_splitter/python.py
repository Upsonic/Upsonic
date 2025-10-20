from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Any
import ast
import re

from pydantic import Field, ConfigDict

from upsonic.text_splitter.base import BaseChunker, BaseChunkingConfig
from upsonic.schemas.data_models import Chunk, Document
from upsonic.text_splitter.recursive import RecursiveChunker, RecursiveChunkingConfig
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)

def get_default_text_chunker() -> BaseChunker:
    """A factory function to create a default text chunker instance."""
    config = RecursiveChunkingConfig(
        chunk_size=512, 
        chunk_overlap=50, 
        separators=["\n\n", "\n", " ", ""]
    )
    return RecursiveChunker(config)


class PythonChunkingConfig(BaseChunkingConfig):
    """
    A specialized configuration model for the AST-powered Python Chunker.

    This config provides fine-grained control over how Python source code is
    parsed, segmented by its grammatical structure, and chunked.
    """
    split_on_nodes: List[str] = Field(
        default_factory=lambda: ["ClassDef", "FunctionDef", "AsyncFunctionDef"],
        description=(
            "A list of Python AST node types that signify a chunk boundary. By default, "
            "every class and function will be treated as a primary semantic block."
        )
    )
    min_chunk_lines: int = Field(
        default=1,
        description=(
            "The minimum number of lines a code block must have to be considered a "
            "standalone chunk. Helps filter out trivial one-liners or empty constructs."
        ),
        ge=1,
    )
    include_docstrings: bool = Field(
        default=True,
        description=(
            "If True, the docstring of a class or function will be included in the "
            "chunk's content, which is highly recommended for semantic context."
        )
    )
    strip_decorators: bool = Field(
        default=False,
        description=(
            "If True, decorator syntax (e.g., '@my_decorator') will be stripped "
            "from the chunk's content, focusing only on the core function/class logic."
        )
    )
    text_chunker_to_use: BaseChunker = Field(
        default_factory=get_default_text_chunker,
        description=(
            "An instance of another chunker (e.g., RecursiveChunker) to be used "
            "for splitting the body of an oversized function or class if it exceeds "
            "the main `chunk_size`."
        ),
        exclude=True
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

# Rebuild the model to resolve forward references
PythonChunkingConfig.model_rebuild()


class _SemanticBlock(NamedTuple):
    """
    An internal data structure for a semantically distinct block of Python code.
    """
    type: str
    name: str
    full_name_path: str
    raw_content: str
    start_line: int
    end_line: int
    docstring: Optional[str]



class PythonChunker(BaseChunker[PythonChunkingConfig]):
    """
    A syntax-aware chunker for Python source code that uses the Abstract Syntax Tree (AST).

    This chunker parses Python code into an AST to identify the precise boundaries
    of logical blocks like classes and functions. This allows for highly accurate,
    semantically meaningful chunking that is robust to formatting variations.
    """

    def __init__(self, config: Optional[PythonChunkingConfig] = None):
        """Initializes the chunker with a specific or default configuration."""
        super().__init__(config or PythonChunkingConfig())

    def _chunk_document(self, document: Document) -> List[Chunk]:
        """
        The core implementation for chunking a single Python document.
        """
        try:
            semantic_blocks = self._segment_python_code(document.content)
        except SyntaxError as e:
            logger.error(f"Invalid Python syntax in document {document.document_id}: {e}")
            return []
        all_chunks: List[Chunk] = []
        line_start_indices = [0] + [m.end() for m in re.finditer(r'\n', document.content)]

        for block in semantic_blocks:
            if (block.end_line - block.start_line) + 1 < self.config.min_chunk_lines:
                continue
            
            content = block.raw_content
            
            if self.config.strip_decorators:
                content = self._strip_decorators_from_string(content)
            if not self.config.include_docstrings and block.docstring:
                content = content.replace(f'"""{block.docstring}"""', '', 1)
                content = content.replace(f"'''{block.docstring}'''", '', 1)

            metadata = {
                "path": block.full_name_path, "type": block.type,
                "start_line": block.start_line, "end_line": block.end_line,
            }
            
            start_index = line_start_indices[block.start_line - 1]
            try:
                end_index = line_start_indices[block.end_line] - 1
            except IndexError:
                end_index = len(document.content)

            if self.config.length_function(content) > self.config.chunk_size:
                temp_doc = Document(content=content, document_id=document.document_id)
                sub_chunks = self.config.text_chunker_to_use.chunk([temp_doc])
                for sub_chunk in sub_chunks:
                    abs_start = start_index + (sub_chunk.start_index or 0)
                    abs_end = start_index + (sub_chunk.end_index or 0)
                    all_chunks.append(self._create_chunk(
                        document, sub_chunk.text_content, abs_start, abs_end, {**metadata, **sub_chunk.metadata}
                    ))
            else:
                all_chunks.append(self._create_chunk(
                    document, content, start_index, end_index, metadata
                ))

        if not all_chunks:
            return []
        
        min_chunk_size = self._get_effective_min_chunk_size()

        if len(all_chunks) > 1 and self.config.length_function(all_chunks[-1].text_content) < min_chunk_size:
            last_chunk = all_chunks.pop()
            previous_chunk = all_chunks.pop()

            merged_text = document.content[previous_chunk.start_index:last_chunk.end_index]

            merged_chunk = self._create_chunk(
                parent_document=document,
                text_content=merged_text,
                start_index=previous_chunk.start_index,
                end_index=last_chunk.end_index,
                extra_metadata=previous_chunk.metadata,
            )
            all_chunks.append(merged_chunk)

        return all_chunks

    def _segment_python_code(self, code: str) -> List[_SemanticBlock]:
        tree = ast.parse(code)
        visitor = self._CodeVisitor(code, self.config)
        visitor.visit(tree)
        return visitor.blocks
    
    def _strip_decorators_from_string(self, code: str) -> str:
        lines = code.split('\n')
        non_decorator_lines = [line for line in lines if not line.strip().startswith('@')]
        return '\n'.join(non_decorator_lines)

    class _CodeVisitor(ast.NodeVisitor):
        """
        An AST visitor that traverses the code tree to find and extract
        semantic blocks like classes and functions.
        """
        def __init__(self, source_code: str, config: PythonChunkingConfig):
            self.source_lines = source_code.splitlines(keepends=True)
            self.config = config
            self.blocks: List[_SemanticBlock] = []
            self.context_stack: List[str] = []

        def _get_node_source(self, node: ast.AST) -> str:
            start_line = node.lineno - 1
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
            return "".join(self.source_lines[start_line:end_line])

        def _visit_node(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            node_type_name = type(node).__name__
            if node_type_name not in self.config.split_on_nodes:
                self.generic_visit(node)
                return

            self.context_stack.append(node.name)
            
            full_name_path = ".".join(self.context_stack)
            docstring = ast.get_docstring(node)
            raw_content = self._get_node_source(node)
            
            block_type = "class"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                block_type = "method" if len(self.context_stack) > 1 else "function"

            self.blocks.append(_SemanticBlock(
                type=block_type, name=node.name,
                full_name_path=full_name_path, raw_content=raw_content,
                start_line=node.lineno, end_line=node.end_lineno,
                docstring=docstring
            ))

            self.generic_visit(node)
            self.context_stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef):
            self._visit_node(node)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._visit_node(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._visit_node(node)