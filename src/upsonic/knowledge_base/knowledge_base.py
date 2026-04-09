from __future__ import annotations
import asyncio
import hashlib
import json
import re
from enum import Enum
from typing import TYPE_CHECKING, List, Literal, Optional, Dict, Any, Union
from pathlib import Path

if TYPE_CHECKING:
    from upsonic.storage.base import Storage
    from upsonic.storage.schemas import KnowledgeRow
    from upsonic.tools.base import Tool

from ..text_splitter.base import BaseChunker
from ..embeddings.base import EmbeddingProvider
from ..vectordb.base import BaseVectorDBProvider
from ..loaders.base import BaseLoader
from ..schemas.data_models import Document, RAGSearchResult, Chunk
from ..schemas.vector_schemas import VectorSearchResult
from ..loaders.factory import create_intelligent_loaders
from ..text_splitter.factory import create_intelligent_splitters
from ..utils.printing import info_log, debug_log, warning_log, error_log, success_log
from upsonic.utils.package.exception import (
    VectorDBConnectionError, 
    UpsertError,
)


class KBState(str, Enum):
    UNINITIALIZED = "uninitialized"
    CONNECTED = "connected"
    INDEXED = "indexed"
    CLOSED = "closed"


class KnowledgeBase:
    """
    The central, intelligent orchestrator for a collection of knowledge in an AI Agent Framework.

    This class manages the entire lifecycle of documents for RAG (Retrieval-Augmented Generation) 
    pipelines, from ingestion and processing to vector storage and retrieval.

    Key Features:
    - **Intelligent Document Processing**: Automatic loader and splitter detection
    - **Idempotent Operations**: Expensive processing done only once per configuration
    - **Async-First Architecture**: High-performance async operations with sync fallbacks
    - **Flexible Search**: Dense, full-text, and hybrid search capabilities
    - **Document Management**: Track, update, and delete documents by various identifiers
    - **Health Monitoring**: Comprehensive health checks and diagnostics
    - **Resource Management**: Proper connection lifecycle and cleanup
    - **Tool Provider Protocol**: Exposes ``get_tools()`` for agent integration without inheritance

    This class serves as the bridge between raw documents and the vector database,
    providing a high-level, framework-agnostic interface for knowledge management.
    """
    
    def __init__(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        vectordb: BaseVectorDBProvider,
        embedding_provider: Optional[EmbeddingProvider] = None,
        splitters: Optional[Union[BaseChunker, List[BaseChunker]]] = None,
        loaders: Optional[Union[BaseLoader, List[BaseLoader]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        topics: Optional[List[str]] = None,
        use_case: str = "rag_retrieval",
        quality_preference: str = "balanced",
        loader_config: Optional[Dict[str, Any]] = None,
        splitter_config: Optional[Dict[str, Any]] = None,
        isolate_search: bool = True,
        storage: Optional["Storage"] = None,
        **config_kwargs
    ):
        """
        Initializes the KnowledgeBase with all necessary components.

        This is a lightweight initialization that:
        - Resolves and validates sources
        - Sets up or auto-detects loaders and splitters
        - Generates a unique, deterministic knowledge ID
        - Prepares for async operations

        No data processing or I/O occurs at this stage. All expensive operations
        are deferred to the `setup_async()` method for just-in-time execution.

        Args:
            sources: Source identifiers (file paths, directory paths, or direct content strings).
                    Can be a single source or a list of sources.
            vectordb: An instance of BaseVectorDBProvider for vector storage and retrieval.
            embedding_provider: An instance of EmbeddingProvider for converting text to vectors.
                    Optional for providers that handle their own embeddings (e.g. SuperMemory).
            splitters: Optional text chunking strategy. If None, intelligent auto-detection is used.
                      Can be a single BaseChunker or a list matching source count.
            loaders: Optional document loaders for various file types. If None, auto-detected.
                    Can be a single BaseLoader or a list matching file source count.
            name: Optional human-readable name. If None, uses the knowledge_id.
            use_case: Intended use case for chunking optimization 
                     ("rag_retrieval", "semantic_search", "question_answering", etc.).
            quality_preference: Speed vs quality trade-off ("fast", "balanced", "quality").
            loader_config: Specific configuration for document loaders.
            splitter_config: Specific configuration for text splitters.
            **config_kwargs: Legacy global config options (use specific configs instead).

        Raises:
            ValueError: If sources is empty or component counts are incompatible.

        Example:
            ```python
            kb = KnowledgeBase(
                sources=["docs/", "README.md"],
                vectordb=ChromaProvider(config=chroma_config),
                embedding_provider=OpenAIEmbedding(),
                use_case="rag_retrieval"
            )
            await kb.setup_async()  # Process and index documents
            results = await kb.query_async("What is the project about?")
            ```
        """
        # Validate inputs
        if not sources:
            raise ValueError("KnowledgeBase must be initialized with at least one source.")

        # Validate that all file/directory sources exist before processing
        self._validate_sources_exist(sources)
        self.description: str = description or f"Knowledge base for {name}"
        self.topics: List[str] = topics or []

        # Core components
        self.sources: List[Union[str, Path]] = self._resolve_sources(sources)
        self.embedding_provider: Optional[EmbeddingProvider] = embedding_provider
        self.vectordb: BaseVectorDBProvider = vectordb
        self.isolate_search: bool = isolate_search
        self.storage: Optional["Storage"] = storage
        
        # Setup loaders with intelligent auto-detection
        self.loaders: List[BaseLoader] = self._setup_loaders(
            loaders, loader_config or config_kwargs
        )
        
        # Setup splitters with intelligent auto-detection
        self.splitters: List[BaseChunker] = self._setup_splitters(
            splitters, splitter_config or config_kwargs, use_case, quality_preference
        )

        # Validate component compatibility
        self._validate_component_counts()

        # Knowledge base identification
        self.knowledge_id: str = self._generate_knowledge_id()
        self.name: str = name or self.knowledge_id[:16]  # Use first 16 chars of ID if no name
        
        # State management
        self._state: KBState = KBState.UNINITIALIZED
        self._setup_lock: asyncio.Lock = asyncio.Lock()
        self._processing_stats: Dict[str, Any] = {}
        
        # Auto-derive collection_name when the user didn't explicitly set one
        if self.vectordb._config.collection_name == "default_collection":
            sanitized: str = re.sub(r'[^a-zA-Z0-9_]', '_', self.name)[:50]
            derived_name: str = f"kb_{sanitized}_{self.knowledge_id[:8]}"
            object.__setattr__(self.vectordb._config, 'collection_name', derived_name)
            info_log(
                f"Auto-derived collection name: '{derived_name}' for KnowledgeBase '{self.name}'",
                context="KnowledgeBase",
            )

        # Precompute the search tool name for get_tools() / build_context()
        self._search_tool_name: str = f"search_{self._sanitize_tool_name(self.name)}"

        info_log(
            f"Initialized KnowledgeBase '{self.name}' with {len(self.sources)} sources, "
            f"{len(self.loaders)} loaders, {len(self.splitters)} splitters",
            context="KnowledgeBase"
        )

    def _sanitize_tool_name(self, name: str) -> str:
        """
        Sanitize a string for use as a tool name component.
        
        Tool names must be valid Python identifiers (alphanumeric + underscores).
        
        Args:
            name: The name to sanitize
            
        Returns:
            A sanitized string suitable for use in a tool name
        """
        # Replace non-alphanumeric characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Collapse multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"kb_{sanitized}"
        return sanitized.lower() if sanitized else "unnamed"
    
    # ------------------------------------------------------------------
    # Tool Provider Protocol
    # ------------------------------------------------------------------

    def get_tools(self) -> List["Tool"]:
        """Produce fully-formed ``FunctionTool`` objects for agent registration.

        Creates an async search tool whose name is derived from
        ``self.name`` (e.g. ``KnowledgeBase(name="technical_docs")`` →
        ``search_technical_docs``), guaranteeing uniqueness when multiple
        KnowledgeBase instances are registered with the same agent.

        Uses ``FunctionTool.from_callable`` — the framework's standard
        one-step tool creation — so no manual schema or attribute
        manipulation is needed.

        Returns:
            List containing a single ``FunctionTool`` wrapping the search
            callable.
        """
        from upsonic.tools.wrappers import FunctionTool

        kb_instance: KnowledgeBase = self
        tool_name: str = self._search_tool_name
        topics_str: str = ", ".join(self.topics) if self.topics else "general"

        async def search_knowledge_base(query: str) -> str:
            """Search the knowledge base for information about a query.

            Args:
                query: The query to search for.

            Returns:
                A string containing the response from the knowledge base.
            """
            return await kb_instance._search_impl(query)

        return [
            FunctionTool.from_callable(
                search_knowledge_base,
                name=tool_name,
                description=(
                    f"Search the '{self.name}' knowledge base for relevant information.\n\n"
                    f"This tool performs intelligent retrieval from the '{self.name}' "
                    f"knowledge base.\n"
                    f"Topics covered: {topics_str}\n"
                    f"Description: {self.description}"
                ),
            )
        ]

    async def aget_tools(self) -> List["Tool"]:
        """Async version of get_tools."""
        return self.get_tools()

    def build_context(self) -> str:
        """Build context instructions for the agent's system prompt.

        Returns a structured instruction block telling the model about this
        knowledge base, its search tool, and best-practice usage guidance.

        Returns:
            Context string to inject into the system prompt.
        """
        parts: List[str] = [
            f"You have access to a knowledge base called '{self.name}' "
            f"that you can search using the {self._search_tool_name} tool.",
        ]

        if self.description:
            parts.append(f"Knowledge base description: {self.description}")

        if self.topics:
            parts.append(f"Topics covered: {', '.join(self.topics)}")

        parts.append(
            "Always search this knowledge base before answering questions related to "
            "its topics — do not assume you already know the answer. "
            "For ambiguous questions, search first rather than asking for clarification."
        )

        return "<knowledge_base>\n" + "\n".join(parts) + "\n</knowledge_base>"

    async def abuild_context(self) -> str:
        """Async version of build_context."""
        return self.build_context()
    
    async def _search_impl(self, query: str) -> str:
        """
        Internal search implementation called by the dynamic search tool.
        
        This method contains the actual search logic, separate from the tool interface.
        
        Args:
            query: The search query
            
        Returns:
            Formatted search results as a string
        """
        results = await self.query_async(query)
        if not results:
            return "No relevant information found in the knowledge base."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(f"Result {i}:\n{result.text}")
            
        return "\n\n".join(formatted_results)

    def _validate_sources_exist(self, sources: Union[str, Path, List[Union[str, Path]]]) -> None:
        """
        Validate that all file and directory sources exist before processing.
        
        This method checks that:
        - File paths exist and are files
        - Directory paths exist and are directories
        - String content sources are skipped (they don't need to exist as files)
        
        Args:
            sources: Single source or list of sources to validate
            
        Raises:
            FileNotFoundError: If any file or directory source doesn't exist
            ValueError: If a path exists but is not the expected type (file vs directory)
        """
        if not isinstance(sources, list):
            source_list = [sources]
        else:
            source_list = sources
        
        missing_sources = []
        
        for item in source_list:
            # Skip string content sources (they don't need to exist as files)
            if isinstance(item, str) and self._is_direct_content(item):
                continue
            
            try:
                path_item = Path(item)
                
                # Check if path exists
                if not path_item.exists():
                    missing_sources.append(str(item))
                    continue
                
                # Validate that files are actually files and directories are actually directories
                if path_item.is_file():
                    # File exists and is a file - valid
                    continue
                elif path_item.is_dir():
                    # Directory exists and is a directory - valid
                    continue
                else:
                    # Path exists but is neither file nor directory (e.g., symlink to nowhere)
                    missing_sources.append(str(item))
                    
            except (OSError, ValueError) as e:
                # If we can't even create a Path, it's invalid
                missing_sources.append(str(item))
        
        if missing_sources:
            raise FileNotFoundError(
                f"The following source(s) do not exist: {', '.join(missing_sources)}. "
                f"Please ensure all file and directory paths are valid and exist."
            )

    def _resolve_sources(self, sources: Union[str, Path, List[Union[str, Path]]]) -> List[Union[str, Path]]:
        """
        Resolves a flexible source input into a definitive list of sources.
        Handles mixed types: file paths, directory paths, and string content.
        
        Args:
            sources: Single source or list of sources (can be paths or string content)
            
        Returns:
            List of resolved sources (Path objects for files/directories, strings for content)
        """
        if not isinstance(sources, list):
            source_list = [sources]
        else:
            source_list = sources

        resolved_sources: List[Union[str, Path]] = []
        added_paths: set[Path] = set()
        
        for item in source_list:
            if isinstance(item, str) and self._is_direct_content(item):
                resolved_sources.append(item)
                continue
            
            try:
                path_item = Path(item)
                
                if not path_item.exists():
                    resolved_sources.append(str(item))
                    continue

                if path_item.is_file():
                    if path_item not in added_paths:
                        resolved_sources.append(path_item)
                        added_paths.add(path_item)
                elif path_item.is_dir():
                    supported_files = self._get_supported_files_from_directory(path_item)
                    for file_path in supported_files:
                        if file_path not in added_paths:
                            resolved_sources.append(file_path)
                            added_paths.add(file_path)
                            
            except (OSError, ValueError):
                resolved_sources.append(str(item))

        return resolved_sources

    def _get_supported_files_from_directory(self, directory: Path) -> List[Path]:
        """Recursively finds all supported files within a directory."""
        supported_extensions = {
            '.txt', '.md', '.rst', '.log', '.py', '.js', '.ts', '.java', '.c', '.cpp', 
            '.h', '.cs', '.go', '.rs', '.php', '.rb', '.html', '.css', '.xml', '.json', 
            '.yaml', '.yml', '.ini', '.csv', '.pdf', '.docx', '.jsonl', '.markdown', 
            '.htm', '.xhtml'
        }
        
        supported_files = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                supported_files.append(file_path)
        return supported_files

    def _setup_loaders(
        self, 
        loaders: Optional[Union[BaseLoader, List[BaseLoader]]], 
        config: Dict[str, Any]
    ) -> List[BaseLoader]:
        """
        Setup document loaders with intelligent auto-detection if not provided.
        
        Args:
            loaders: Optional loader(s) to use
            config: Configuration for loader creation
            
        Returns:
            List of BaseLoader instances
        """
        if loaders is not None:
            return self._normalize_loaders(loaders)
        
        # Auto-detect loaders
        info_log(
            f"Auto-detecting loaders for {len(self.sources)} sources...", 
            context="KnowledgeBase"
        )
        try:
            detected_loaders = create_intelligent_loaders(self.sources, **config)
            info_log(
                f"Created {len(detected_loaders)} intelligent loaders", 
                context="KnowledgeBase"
            )
            return detected_loaders
        except Exception as e:
            warning_log(
                f"Auto-detection failed: {e}, proceeding without loaders", 
                context="KnowledgeBase"
            )
            return []
    
    def _setup_splitters(
        self, 
        splitters: Optional[Union[BaseChunker, List[BaseChunker]]], 
        config: Dict[str, Any],
        use_case: str,
        quality_preference: str
    ) -> List[BaseChunker]:
        """
        Setup text splitters with intelligent auto-detection if not provided.
        
        Args:
            splitters: Optional splitter(s) to use
            config: Configuration for splitter creation
            use_case: The intended use case
            quality_preference: Quality vs speed preference
            
        Returns:
            List of BaseChunker instances
        """
        if splitters is not None:
            return self._normalize_splitters(splitters)
        
        # Auto-detect splitters
        info_log(
            f"Auto-detecting splitters for {len(self.sources)} sources...", 
            context="KnowledgeBase"
        )
        try:
            detected_splitters = create_intelligent_splitters(
                self.sources,
                use_case=use_case,
                quality_preference=quality_preference,
                embedding_provider=self.embedding_provider,
                **config
            )
            info_log(
                f"Created {len(detected_splitters)} intelligent splitters", 
                context="KnowledgeBase"
            )
            return detected_splitters
        except Exception as e:
            warning_log(
                f"Auto-detection failed: {e}, using default recursive strategy", 
                context="KnowledgeBase"
            )
            from ..text_splitter.factory import create_chunking_strategy
            return [create_chunking_strategy("recursive")]

    def _normalize_splitters(self, splitters: Union[BaseChunker, List[BaseChunker]]) -> List[BaseChunker]:
        """
        Normalize splitters to always be a list.
        
        Args:
            splitters: Single splitter or list of splitters
            
        Returns:
            List of BaseChunker instances
            
        Raises:
            ValueError: If splitters is not the correct type
        """
        if isinstance(splitters, list):
            return splitters
        elif isinstance(splitters, BaseChunker):
            return [splitters]
        else:
            raise ValueError("Splitters must be a BaseChunker or list of BaseChunker instances")

    def _normalize_loaders(self, loaders: Optional[Union[BaseLoader, List[BaseLoader]]]) -> List[BaseLoader]:
        """
        Normalize loaders to always be a list.
        
        Args:
            loaders: Single loader, list of loaders, or None
            
        Returns:
            List of BaseLoader instances (empty list if None)
            
        Raises:
            ValueError: If loaders is not the correct type
        """
        if loaders is None:
            return []
        elif isinstance(loaders, list):
            return loaders
        elif isinstance(loaders, BaseLoader):
            return [loaders]
        else:
            raise ValueError("Loaders must be a BaseLoader or list of BaseLoader instances")

    def _validate_component_counts(self):
        """Validate that component counts are compatible for indexed processing."""
        source_count = len(self.sources)
        splitter_count = len(self.splitters)
        loader_count = len(self.loaders) if self.loaders else 0
        
        file_source_count = sum(1 for source in self.sources if isinstance(source, Path))
        
        if source_count > 1:
            if splitter_count > 1 and splitter_count != source_count:
                raise ValueError(
                    f"Number of splitters ({splitter_count}) must match number of sources ({source_count}) "
                    "for indexed processing"
                )
            
            if loader_count > 1 and loader_count != file_source_count:
                raise ValueError(
                    f"Number of loaders ({loader_count}) must match number of file sources ({file_source_count}) "
                    "for indexed processing. String content sources don't need loaders."
                )


    def _is_direct_content(self, source: str) -> bool:
        """
        Check if a source is direct content (not a file path).
        
        Args:
            source: The source string to check
            
        Returns:
            True if the source appears to be direct content, False if it's a file path
        """
        if len(source) > 200:
            return True
            
        if '\n' in source:
            return True
            
        if source.count('.') > 2:
            return True
            
        if len(source) > 100 and not any(ext in source.lower() for ext in ['.txt', '.pdf', '.docx', '.csv', '.json', '.xml', '.yaml', '.md', '.html']):
            return True
            
        words = source.split()
        if len(words) > 5 and not any(word.startswith('/') or word.startswith('.') for word in words):
            return True
        
        try:
            source_path = Path(source)
            
            if source_path.exists():
                return False
                
            if source_path.suffix and not source_path.exists():
                return True
                
        except (OSError, ValueError):
            return True
            
        return False

    def _create_document_from_content(self, content: str, source_index: int) -> Document:
        """
        Create a Document object from direct content string.
        
        Args:
            content: The direct content string
            source_index: Index of the source for metadata
            
        Returns:
            Document object created from the content
        """
        import hashlib
        import time
        
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        
        current_time = time.time()
        metadata = {
            "source": f"direct_content_{source_index}",
            "document_name": f"direct_content_{source_index}.txt",
            "file_path": f"direct_content_{source_index}",
            "file_size": len(content.encode("utf-8")),
            "creation_datetime_utc": current_time,
            "last_modified_datetime_utc": current_time,
        }
        
        return Document(
            content=content,
            metadata=metadata,
            document_id=content_hash,
            doc_content_hash=content_hash,
        )

    def _get_component_for_source(self, source_index: int, component_list: List, component_name: str):
        """
        Get the component for a specific source index.
        
        Args:
            source_index: Index of the source
            component_list: List of components (loaders or splitters)
            component_name: Name of the component type for error messages
            
        Returns:
            Component at the specified index, or the first component if list is shorter
        """
        if not component_list:
            raise ValueError(f"No {component_name}s provided")
        
        if len(component_list) == 1:
            return component_list[0]
        elif source_index < len(component_list):
            return component_list[source_index]
        else:
            from upsonic.utils.printing import warning_log
            warning_log(f"{component_name} index {source_index} out of range, using first {component_name}", "KnowledgeBase")
            return component_list[0]

    def _generate_knowledge_id(self) -> str:
        """
        Creates a unique, deterministic hash for this specific knowledge configuration.

        This ID is used as the collection name in the vector database. By hashing the
        source identifiers and the class names of the components, we ensure that
        if the data or the way it's processed changes, a new, separate collection
        will be created.

        Returns:
            A SHA256 hash string representing this unique knowledge configuration.
        """
        sources_serializable = [str(source) for source in self.sources]
        
        config_representation = {
            "sources": sorted(sources_serializable),
            "loaders": [loader.__class__.__name__ for loader in self.loaders] if self.loaders else [],
            "splitters": [splitter.__class__.__name__ for splitter in self.splitters],
            "embedding_provider": self.embedding_provider.__class__.__name__ if self.embedding_provider else "none",
        }
        
        config_string = json.dumps(config_representation, sort_keys=True)
        
        return hashlib.sha256(config_string.encode('utf-8')).hexdigest()

    # ============================================================================
    # Storage Helpers
    # ============================================================================

    def _build_knowledge_row(
        self,
        doc: Document,
        chunk_count: int,
        status: str = "indexed",
    ) -> "KnowledgeRow":
        """Build a KnowledgeRow from a Document for storage persistence.

        Args:
            doc: The source Document.
            chunk_count: Number of chunks produced from this document.
            status: Processing status (e.g. "indexed", "failed").

        Returns:
            A populated KnowledgeRow instance.
        """
        from upsonic.storage.schemas import KnowledgeRow

        file_path: Optional[str] = doc.metadata.get("file_path")
        file_ext: Optional[str] = None
        if file_path:
            file_ext = Path(file_path).suffix.lstrip(".") or None

        return KnowledgeRow(
            id=doc.document_id,
            name=doc.metadata.get("document_name", doc.document_id),
            description=doc.metadata.get("description"),
            metadata=doc.metadata,
            type=file_ext,
            size=doc.metadata.get("file_size"),
            knowledge_base_id=self.knowledge_id,
            content_hash=doc.doc_content_hash,
            chunk_count=chunk_count,
            source=str(file_path) if file_path else None,
            status=status,
            status_message=None,
            access_count=0,
        )

    def _remove_source_for_document(self, document_id: str) -> None:
        """Remove the source entry corresponding to a document from self.sources.

        Looks up the document's source path from storage first, then falls
        back to a best-effort match against resolved source paths.

        Args:
            document_id: The document ID being removed.
        """
        if self.storage is not None:
            row = self.storage.get_knowledge_content(document_id)
            if row is not None and row.source:
                source_path = Path(row.source)
                self.sources = [
                    s for s in self.sources
                    if Path(str(s)).resolve() != source_path.resolve()
                ]
                return



    async def setup_async(self, force: bool = False) -> None:
        """
        The main just-in-time engine for processing and indexing knowledge.

        This method is **idempotent** and **thread-safe**. It:
        1. Connects to the vector database
        2. Loads documents from all sources
        3. Computes content hashes for change detection
        4. Filters out unchanged documents (skips), deletes stale chunks (edits)
        5. Chunks only new/changed documents
        6. Generates embeddings
        7. Stores everything in the vector database

        On first run the full pipeline executes.  On subsequent runs only
        documents whose content has actually changed are re-processed, saving
        embedding and storage costs.

        Args:
            force: If True, re-runs the full pipeline even if already indexed.

        Raises:
            VectorDBConnectionError: If database connection fails
            UpsertError: If data ingestion fails
            RuntimeError: If the KnowledgeBase has been closed
        """
        async with self._setup_lock:
            if self._state == KBState.INDEXED and not force:
                debug_log(
                    f"KnowledgeBase '{self.name}' already set up. Skipping.",
                    context="KnowledgeBase",
                )
                return

            if self._state == KBState.CLOSED:
                raise RuntimeError("Cannot setup a closed KnowledgeBase. Create a new instance.")

            collection_existed: bool = False

            try:
                # Step 0: Connect to vector database
                await self._ensure_connection()
                self._state = KBState.CONNECTED

                # Step 1: Load documents from all sources
                all_documents, processing_metadata = await self._load_documents()

                if not all_documents:
                    warning_log(
                        "No documents loaded. Marking as indexed but empty.",
                        context="KnowledgeBase",
                    )
                    self._state = KBState.INDEXED
                    return

                for source_docs in processing_metadata['source_to_documents'].values():
                    for doc in source_docs:
                        if not doc.doc_content_hash:
                            doc.doc_content_hash = hashlib.md5(
                                doc.content.encode("utf-8")
                            ).hexdigest()

                # Step 1.5: Determine which documents actually need processing
                collection_existed = await self.vectordb.acollection_exists()

                if collection_existed:
                    documents_to_process, processing_metadata = (
                        await self._filter_changed_documents(
                            all_documents, processing_metadata
                        )
                    )
                    if not documents_to_process:
                        info_log(
                            f"All documents in '{self.name}' are up-to-date. "
                            f"Nothing to re-index.",
                            context="KnowledgeBase",
                        )
                        self._state = KBState.INDEXED
                        return
                else:
                    documents_to_process = all_documents

                info_log(
                    f"Processing {len(documents_to_process)} document(s) for "
                    f"'{self.name}'...",
                    context="KnowledgeBase",
                )

                # Step 2: Chunk documents
                all_chunks = await self._chunk_documents(
                    documents_to_process, processing_metadata
                )

                if not all_chunks:
                    warning_log(
                        "No chunks created. Marking as indexed but empty.",
                        context="KnowledgeBase",
                    )
                    self._state = KBState.INDEXED
                    return

                # Step 3: Generate embeddings
                vectors = await self._generate_embeddings(all_chunks)

                # Step 4: Store in vector database
                await self._store_in_vectordb(all_chunks, vectors)

                if self.storage is not None:
                    doc_chunk_counts: Dict[str, int] = {}
                    for chunk in all_chunks:
                        doc_chunk_counts[chunk.document_id] = doc_chunk_counts.get(chunk.document_id, 0) + 1

                    for doc in documents_to_process:
                        row = self._build_knowledge_row(
                            doc=doc,
                            chunk_count=doc_chunk_counts.get(doc.document_id, 0),
                            status="indexed",
                        )
                        self.storage.upsert_knowledge_content(row)

                # Update stats
                self._processing_stats = {
                    "sources_count": len(self.sources),
                    "documents_count": len(documents_to_process),
                    "chunks_count": len(all_chunks),
                    "vectors_count": len(vectors),
                    "indexed_at": __import__('datetime').datetime.now().isoformat(),
                }

                self._state = KBState.INDEXED
                success_log(
                    f"KnowledgeBase '{self.name}' indexing completed! "
                    f"{len(documents_to_process)} docs → {len(all_chunks)} chunks",
                    context="KnowledgeBase",
                )

            except Exception as e:
                error_log(
                    f"Setup failed for '{self.name}': {e}",
                    context="KnowledgeBase",
                )
                if not collection_existed:
                    try:
                        if await self.vectordb.acollection_exists():
                            warning_log(
                                "Cleaning up partially created collection...",
                                context="KnowledgeBase",
                            )
                            await self.vectordb.adelete_collection()
                    except Exception:
                        pass
                raise

    def setup(self, force: bool = False) -> None:
        """Synchronous wrapper for setup_async."""
        import asyncio
        try:
            asyncio.get_running_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                def _run() -> None:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.setup_async(force))
                executor.submit(_run).result()
        except RuntimeError:
            asyncio.new_event_loop().run_until_complete(self.setup_async(force))

    async def _ensure_connection(self) -> None:
        """Ensures the vector database is connected."""
        if self.vectordb._is_connected:
            return

        try:
            await self.vectordb.aconnect()
            info_log("Vector database connected successfully", context="KnowledgeBase")
        except Exception as e:
            error_log(f"Failed to connect to vector database: {e}", context="KnowledgeBase")
            raise VectorDBConnectionError(f"Connection failed: {e}")

    async def _load_documents(self) -> tuple[List[Document], Dict[int, Any]]:
        """
        Load documents from all sources using appropriate loaders.
        
        Returns:
            Tuple of (all_documents, processing_metadata)
            where processing_metadata tracks loader/source relationships
        """
        info_log(f"[Step 1/4] Loading documents from {len(self.sources)} sources...", context="KnowledgeBase")
        
        all_documents = []
        processing_metadata = {
            'source_to_documents': {},
            'source_to_loader': {},
        }
        
        for source_index, source in enumerate(self.sources):
            source_str = str(source)[:100] + ('...' if len(str(source)) > 100 else '')
            debug_log(f"Processing source {source_index}: {source_str}", context="KnowledgeBase")
            
            try:
                if isinstance(source, str) and self._is_direct_content(source):
                    # Direct content string
                    document = self._create_document_from_content(source, source_index)
                    source_documents = [document]
                    processing_metadata['source_to_loader'][source_index] = None
                    debug_log(f"Created document from direct content (length: {len(source)})", context="KnowledgeBase")
                else:
                    # File source - use loader
                    if not self.loaders:
                        warning_log(f"No loaders available for file source {source}", context="KnowledgeBase")
                        continue
                    
                    loader = self._get_component_for_source(source_index, self.loaders, "loader")
                    
                    if not loader.can_load(source):
                        warning_log(
                            f"Loader {loader.__class__.__name__} cannot handle {source}",
                            context="KnowledgeBase"
                        )
                        continue
                    
                    source_documents = loader.load(source)
                    processing_metadata['source_to_loader'][source_index] = loader
                    debug_log(
                        f"Loaded {len(source_documents)} documents from {source} using {loader.__class__.__name__}",
                        context="KnowledgeBase"
                    )
                
                all_documents.extend(source_documents)
                processing_metadata['source_to_documents'][source_index] = source_documents
                
            except Exception as e:
                error_log(f"Error processing source {source_index} ({source}): {e}", context="KnowledgeBase")
                continue
        
        info_log(f"Loaded {len(all_documents)} documents from {len(processing_metadata['source_to_documents'])} sources", context="KnowledgeBase")
        return all_documents, processing_metadata

    async def _filter_changed_documents(
        self,
        documents: List[Document],
        processing_metadata: Dict[int, Any],
    ) -> tuple[List[Document], Dict[int, Any]]:
        """
        Filter out unchanged documents and delete stale chunks for edited ones.

        For each loaded document this method:
        1. Checks if its content_hash already exists in the vector DB (unchanged → skip).
        2. If the hash is new but the document_id exists, deletes old chunks (edited → replace).
        3. If neither exists, the document is new.

        Args:
            documents: All loaded documents (with content_hash already computed).
            processing_metadata: Source-to-document mapping from the loading phase.

        Returns:
            Tuple of (documents_that_need_processing, filtered_processing_metadata).
        """
        filtered_docs: List[Document] = []
        filtered_source_to_docs: Dict[int, List[Document]] = {}

        source_to_documents: Dict[int, List[Document]] = processing_metadata.get(
            'source_to_documents', {}
        )

        for source_index, source_docs in source_to_documents.items():
            changed_docs: List[Document] = []

            for doc in source_docs:
                if await self.vectordb.adoc_content_hash_exists(doc.doc_content_hash):
                    debug_log(
                        f"Document '{doc.document_id[:16]}...' unchanged (hash match), skipping.",
                        context="KnowledgeBase",
                    )
                    continue

                if await self.vectordb.adocument_id_exists(doc.document_id):
                    await self.vectordb.adelete_by_document_id(doc.document_id)
                    info_log(
                        f"Document '{doc.document_id[:16]}...' changed, deleted old chunks.",
                        context="KnowledgeBase",
                    )

                changed_docs.append(doc)

            if changed_docs:
                filtered_docs.extend(changed_docs)
                filtered_source_to_docs[source_index] = changed_docs

        filtered_metadata: Dict[int, Any] = {
            'source_to_documents': filtered_source_to_docs,
            'source_to_loader': processing_metadata.get('source_to_loader', {}),
        }

        info_log(
            f"Document filter: {len(documents)} total, {len(filtered_docs)} need processing, "
            f"{len(documents) - len(filtered_docs)} unchanged.",
            context="KnowledgeBase",
        )

        return filtered_docs, filtered_metadata

    async def _chunk_documents(
        self, 
        documents: List[Document], 
        processing_metadata: Dict[int, Any]
    ) -> List[Chunk]:
        """
        Chunk all documents using appropriate splitters.
        
        Handles fallback to RecursiveChunker if the primary splitter fails
        (e.g., PythonChunker on non-Python content).
        
        Args:
            documents: List of documents to chunk
            processing_metadata: Metadata from loading phase
            
        Returns:
            List of Chunk objects
        """
        info_log(f"[Step 2/4] Chunking {len(documents)} documents...", context="KnowledgeBase")
        
        all_chunks = []
        source_to_documents = processing_metadata['source_to_documents']
        chunks_per_source = {}
        
        for source_index in sorted(source_to_documents.keys()):
            source_docs = source_to_documents[source_index]
            splitter = self._get_component_for_source(source_index, self.splitters, "splitter")
            
            source_chunks = []
            for doc in source_docs:
                try:
                    doc_chunks = splitter.chunk([doc])
                    
                    # If no chunks created (e.g., PythonChunker failed), try fallback
                    if not doc_chunks and splitter.__class__.__name__ != "RecursiveChunker":
                        warning_log(
                            f"Primary splitter {splitter.__class__.__name__} produced 0 chunks. "
                            f"Trying RecursiveChunker as fallback...",
                            context="KnowledgeBase"
                        )
                        # Fallback to RecursiveChunker
                        from ..text_splitter.factory import create_chunking_strategy
                        fallback_splitter = create_chunking_strategy("recursive")
                        doc_chunks = fallback_splitter.chunk([doc])
                        debug_log(
                            f"Fallback splitter created {len(doc_chunks)} chunks",
                            context="KnowledgeBase"
                        )
                    
                    source_chunks.extend(doc_chunks)
                    debug_log(
                        f"Document '{doc.document_id[:16]}...' → {len(doc_chunks)} chunks",
                        context="KnowledgeBase"
                    )
                except Exception as e:
                    error_log(
                        f"Error chunking document {doc.document_id}: {e}",
                        context="KnowledgeBase"
                    )
                    # Try fallback splitter on error
                    try:
                        warning_log(
                            f"Primary splitter failed with error. Trying RecursiveChunker...",
                            context="KnowledgeBase"
                        )
                        from ..text_splitter.factory import create_chunking_strategy
                        fallback_splitter = create_chunking_strategy("recursive")
                        doc_chunks = fallback_splitter.chunk([doc])
                        source_chunks.extend(doc_chunks)
                        debug_log(
                            f"Fallback splitter created {len(doc_chunks)} chunks",
                            context="KnowledgeBase"
                        )
                    except Exception as fallback_error:
                        error_log(
                            f"Fallback splitter also failed: {fallback_error}",
                            context="KnowledgeBase"
                        )
                        continue
            
            chunks_per_source[source_index] = source_chunks
            all_chunks.extend(source_chunks)
            debug_log(
                f"Source {source_index}: {len(source_chunks)} chunks using {splitter.__class__.__name__}",
                context="KnowledgeBase"
            )
        
        info_log(f"Created {len(all_chunks)} chunks from {len(documents)} documents", context="KnowledgeBase")
        return all_chunks

    async def _generate_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Generate embeddings for all chunks.
        
        When no embedding_provider is configured (e.g. SuperMemory which embeds
        internally), returns placeholder zero-vectors so the vectordb upsert
        interface contract is satisfied.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of embedding vectors
        """
        if self.embedding_provider is None:
            vector_size: int = getattr(self.vectordb._config, "vector_size", 0) or 1
            info_log(
                f"[Step 3/4] No embedding provider — generating {len(chunks)} placeholder vectors "
                f"(vectordb handles embeddings internally)",
                context="KnowledgeBase",
            )
            return [[0.0] * vector_size for _ in chunks]

        info_log(f"[Step 3/4] Generating embeddings for {len(chunks)} chunks...", context="KnowledgeBase")
        
        try:
            vectors = await self.embedding_provider.embed_documents(chunks)
            
            if len(vectors) != len(chunks):
                raise ValueError(
                    f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks"
                )
            
            info_log(f"Generated {len(vectors)} embeddings", context="KnowledgeBase")
            return vectors
            
        except Exception as e:
            error_log(f"Failed to generate embeddings: {e}", context="KnowledgeBase")
            raise

    async def _store_in_vectordb(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """
        Store chunks and their vectors in the vector database.
        
        Args:
            chunks: List of chunks to store
            vectors: Corresponding embedding vectors
        """
        info_log(f"[Step 4/4] Storing {len(chunks)} chunks in vector database...", context="KnowledgeBase")
        
        try:
            if not await self.vectordb.acollection_exists():
                await self.vectordb.acreate_collection()

            chunk_texts: List[str] = [chunk.text_content for chunk in chunks]
            chunk_ids: List[str] = [chunk.chunk_id for chunk in chunks]
            doc_ids: List[str] = [chunk.document_id for chunk in chunks]
            doc_hashes: List[str] = [chunk.doc_content_hash for chunk in chunks]
            chunk_hashes: List[str] = [chunk.chunk_content_hash for chunk in chunks]
            chunk_payloads: List[Dict[str, Any]] = [dict(chunk.metadata) for chunk in chunks]

            if self.isolate_search:
                for payload in chunk_payloads:
                    payload["knowledge_base_id"] = self.knowledge_id

            await self.vectordb.aupsert(
                vectors=vectors,
                payloads=chunk_payloads,
                ids=chunk_ids,
                chunks=chunk_texts,
                document_ids=doc_ids,
                doc_content_hashes=doc_hashes,
                chunk_content_hashes=chunk_hashes,
            )
            
            success_log(f"Stored {len(chunks)} chunks successfully", context="KnowledgeBase")
            
        except Exception as e:
            error_log(f"Failed to store in vector database: {e}", context="KnowledgeBase")
            raise UpsertError(f"Storage failed: {e}")



    # ============================================================================
    # Query and Retrieval Methods
    # ============================================================================

    async def query_async(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        task: Optional[Any] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[RAGSearchResult]:
        """
        Performs a search to retrieve relevant knowledge chunks.

        This is the primary retrieval method. It automatically triggers setup
        if not done yet, embeds the query, and searches the vector database.
        The vectordb's asearch method internally determines the best search
        strategy (dense, full-text, or hybrid) based on provider configuration.

        Args:
            query: The user's query string.
            top_k: Number of results to return. If None, uses provider's default or Task's vector_search_top_k.
            filter: Optional metadata filter to apply. If None, uses Task's vector_search_filter if provided.
            task: Optional Task object. If provided, uses Task's vector search parameters to override config defaults.
            alpha: Balance between dense and sparse search (0.0 = pure sparse, 1.0 = pure dense).
            fusion_method: Method for combining search results ('rrf' or 'weighted').
            similarity_threshold: Minimum similarity score for results.
            apply_reranking: Whether to apply reranking if configured on the provider.
            sparse_query_vector: Pre-computed sparse vector for providers that support sparse search.

        Returns:
            List of RAGSearchResult objects containing text content and metadata.

        Raises:
            ValueError: If search results are invalid
        """
        await self.setup_async()

        if self._state != KBState.INDEXED:
            warning_log(
                f"KnowledgeBase '{self.name}' is not ready. Returning empty results.",
                context="KnowledgeBase"
            )
            return []

        info_log(f"Querying '{self.name}': '{query[:100]}...'", context="KnowledgeBase")
        
        try:
            if self.embedding_provider is not None:
                query_vector: List[float] = await self.embedding_provider.embed_query(query)
            else:
                vector_size: int = getattr(self.vectordb._config, "vector_size", 0) or 1
                query_vector = [0.0] * vector_size

            search_results = await self._perform_search(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                filter=filter,
                task=task,
                alpha=alpha,
                fusion_method=fusion_method,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
                sparse_query_vector=sparse_query_vector,
            )
            # Convert to RAG results
            rag_results = self._convert_to_rag_results(search_results)

            if not rag_results:
                warning_log(
                    f"No results found for query: '{query[:50]}...'",
                    context="KnowledgeBase"
                )
            else:
                success_log(
                    f"Retrieved {len(rag_results)} results",
                    context="KnowledgeBase"
                )
            
            return rag_results
            
        except Exception as e:
            error_log(f"Query failed: {e}", context="KnowledgeBase")
            raise

    async def search(self, query: str) -> str:
        """
        Search the knowledge base for relevant information using semantic similarity.
        
        This is a convenience method that wraps the internal search implementation.
        When used as a tool, the dynamically named method (e.g., search_technical_docs)
        is used instead to avoid name collisions with other KnowledgeBase instances.

        Args:
            query: The question, topic, or search query to find relevant information.
                  Can be a natural language question, a topic description, or keywords.

        Returns:
            A formatted string containing the most relevant information found in the
            knowledge base. Results are ranked by relevance and presented in a readable
            format. Returns "No relevant information found in the knowledge base."
            if no matches are found.
        """
        return await self._search_impl(query)

    async def _perform_search(
        self,
        query: str,
        query_vector: List[float],
        top_k: Optional[int],
        filter: Optional[Dict[str, Any]],
        task: Optional[Any] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        if task is not None:
            top_k = top_k if top_k is not None else getattr(task, 'vector_search_top_k', None)
            alpha = alpha if alpha is not None else getattr(task, 'vector_search_alpha', None)
            fusion_method = fusion_method if fusion_method is not None else getattr(task, 'vector_search_fusion_method', None)
            similarity_threshold = similarity_threshold if similarity_threshold is not None else getattr(task, 'vector_search_similarity_threshold', None)
            filter = filter if filter is not None else getattr(task, 'vector_search_filter', None)

        if self.isolate_search:
            isolation_filter: Dict[str, str] = {"knowledge_base_id": self.knowledge_id}
            if filter is None:
                filter = isolation_filter
            else:
                filter = {**filter, **isolation_filter}

        return await self.vectordb.asearch(
            query_vector=query_vector,
            query_text=query,
            top_k=top_k,
            filter=filter,
            alpha=alpha,
            fusion_method=fusion_method,
            similarity_threshold=similarity_threshold,
            apply_reranking=apply_reranking,
            sparse_query_vector=sparse_query_vector,
        )

    def _convert_to_rag_results(self, search_results: List[VectorSearchResult]) -> List[RAGSearchResult]:
        """
        Convert VectorSearchResult objects to RAGSearchResult objects.
        
        Args:
            search_results: Results from vector database search
            
        Returns:
            List of RAGSearchResult objects
            
        Raises:
            ValueError: If results are missing required fields
        """
        rag_results = []
        
        for result in search_results:
            # Extract text content
            text_content = result.text
            
            # If text is not in result object, try to get it from payload
            if not text_content and result.payload:
                text_content = result.payload.get('content') or result.payload.get('chunk') or result.payload.get('text')
            
            if not text_content:
                warning_log(
                    f"Result {result.id} has no text content. Payload: {result.payload}",
                    context="KnowledgeBase"
                )
                continue
            
            # Create RAG result
            rag_result = RAGSearchResult(
                text=text_content,
                metadata=result.payload or {},
                score=result.score,
                chunk_id=str(result.id)
            )
            rag_results.append(rag_result)
        
        return rag_results

    # ============================================================================
    # Dynamic Content Management
    # ============================================================================

    async def aadd_source(
        self,
        source: Union[str, Path],
        loader: Optional[BaseLoader] = None,
        splitter: Optional[BaseChunker] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Add and process a new source after initial setup.

        Args:
            source: File path, directory path, or direct content string.
            loader: Optional loader override. If None, auto-detects.
            splitter: Optional splitter override. If None, auto-detects.
            metadata: Extra metadata to inject into every chunk.

        Returns:
            List of document_ids that were ingested.

        Raises:
            RuntimeError: If the KnowledgeBase has been closed.
        """
        if self._state == KBState.CLOSED:
            raise RuntimeError("Cannot add source to a closed KnowledgeBase.")

        await self._ensure_connection()

        resolved: List[Union[str, Path]] = self._resolve_sources(source)

        if loader is None:
            loader_list: List[BaseLoader] = create_intelligent_loaders(resolved)
        else:
            loader_list = [loader] * len(resolved)

        if splitter is None:
            splitter_list: List[BaseChunker] = create_intelligent_splitters(
                resolved,
                embedding_provider=self.embedding_provider,
            )
        else:
            splitter_list = [splitter] * len(resolved)

        document_ids: List[str] = []
        all_chunks: List[Chunk] = []
        processed_docs: List[Document] = []

        for i, src in enumerate(resolved):
            try:
                if isinstance(src, str) and self._is_direct_content(src):
                    docs: List[Document] = [self._create_document_from_content(src, len(self.sources) + i)]
                else:
                    current_loader: BaseLoader = loader_list[i] if i < len(loader_list) else loader_list[0]
                    docs = current_loader.load(src)

                for doc in docs:
                    if not doc.doc_content_hash:
                        doc.doc_content_hash = hashlib.md5(doc.content.encode("utf-8")).hexdigest()

                    if await self.vectordb.adoc_content_hash_exists(doc.doc_content_hash):
                        continue

                    if await self.vectordb.adocument_id_exists(doc.document_id):
                        await self.vectordb.adelete_by_document_id(doc.document_id)

                    current_splitter: BaseChunker = splitter_list[i] if i < len(splitter_list) else splitter_list[0]
                    chunks: List[Chunk] = current_splitter.chunk([doc])
                    all_chunks.extend(chunks)
                    document_ids.append(doc.document_id)
                    processed_docs.append(doc)

                    if metadata:
                        for chunk in chunks:
                            chunk.metadata.update(metadata)

            except Exception as e:
                error_log(f"Error adding source {src}: {e}", context="KnowledgeBase")
                continue

        if all_chunks:
            vectors: List[List[float]] = await self._generate_embeddings(all_chunks)
            await self._store_in_vectordb(all_chunks, vectors)
            self.sources.extend(resolved)

            if self.storage is not None:
                doc_chunk_counts: Dict[str, int] = {}
                for chunk in all_chunks:
                    doc_chunk_counts[chunk.document_id] = doc_chunk_counts.get(chunk.document_id, 0) + 1

                for doc in processed_docs:
                    row = self._build_knowledge_row(
                        doc=doc,
                        chunk_count=doc_chunk_counts.get(doc.document_id, 0),
                        status="indexed",
                    )
                    self.storage.upsert_knowledge_content(row)

        if self._state == KBState.UNINITIALIZED:
            self._state = KBState.INDEXED

        return document_ids

    def add_source(
        self,
        source: Union[str, Path],
        loader: Optional[BaseLoader] = None,
        splitter: Optional[BaseChunker] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Synchronous wrapper for aadd_source."""
        import asyncio
        try:
            asyncio.get_running_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                def _run() -> List[str]:
                    loop = asyncio.new_event_loop()
                    return loop.run_until_complete(self.aadd_source(source, loader, splitter, metadata))
                return executor.submit(_run).result()
        except RuntimeError:
            return asyncio.new_event_loop().run_until_complete(
                self.aadd_source(source, loader, splitter, metadata)
            )

    async def aadd_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_name: Optional[str] = None,
        splitter: Optional[BaseChunker] = None,
    ) -> str:
        """
        Insert raw text content directly into the knowledge base.

        Args:
            text: The text content to ingest.
            metadata: Optional metadata to attach to every chunk.
            document_name: Human-readable name. Defaults to "text_<hash>".
            splitter: Optional chunker override. Defaults to recursive.

        Returns:
            The document_id of the ingested (or deduplicated) document.

        Raises:
            RuntimeError: If the KnowledgeBase has been closed.
        """
        if self._state == KBState.CLOSED:
            raise RuntimeError("Cannot add text to a closed KnowledgeBase.")

        await self._ensure_connection()

        content_hash: str = hashlib.md5(text.encode("utf-8")).hexdigest()
        doc_name: str = document_name or f"text_{content_hash[:8]}"

        doc_metadata: Dict[str, Any] = {
            "source": doc_name,
            "document_name": doc_name,
            **(metadata or {}),
        }

        doc: Document = Document(
            content=text,
            metadata=doc_metadata,
            document_id=content_hash,
            doc_content_hash=content_hash,
        )

        if await self.vectordb.adoc_content_hash_exists(content_hash):
            return doc.document_id

        if splitter is None:
            from ..text_splitter.factory import create_chunking_strategy
            splitter = create_chunking_strategy("recursive")

        chunks: List[Chunk] = splitter.chunk([doc])

        if not chunks:
            warning_log(f"No chunks created for text '{doc_name}'", context="KnowledgeBase")
            return doc.document_id

        vectors: List[List[float]] = await self._generate_embeddings(chunks)
        await self._store_in_vectordb(chunks, vectors)

        if self.storage is not None:
            row = self._build_knowledge_row(doc=doc, chunk_count=len(chunks), status="indexed")
            self.storage.upsert_knowledge_content(row)

        if self._state == KBState.UNINITIALIZED:
            self._state = KBState.INDEXED

        return doc.document_id

    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_name: Optional[str] = None,
        splitter: Optional[BaseChunker] = None,
    ) -> str:
        """Synchronous wrapper for aadd_text."""
        import asyncio
        try:
            asyncio.get_running_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                def _run() -> str:
                    loop = asyncio.new_event_loop()
                    return loop.run_until_complete(self.aadd_text(text, metadata, document_name, splitter))
                return executor.submit(_run).result()
        except RuntimeError:
            return asyncio.new_event_loop().run_until_complete(
                self.aadd_text(text, metadata, document_name, splitter)
            )

    async def aremove_document(self, document_id: str) -> bool:
        """
        Remove a document and all its chunks from the knowledge base.

        Args:
            document_id: The document ID to remove.

        Returns:
            True if deletion was successful.

        Raises:
            RuntimeError: If the KnowledgeBase has been closed.
        """
        if self._state == KBState.CLOSED:
            raise RuntimeError("Cannot remove document from a closed KnowledgeBase.")

        await self._ensure_connection()

        self._remove_source_for_document(document_id)

        result: bool = await self.vectordb.adelete_by_document_id(document_id)
        if result:
            if self.storage is not None:
                self.storage.delete_knowledge_content(document_id)
            success_log(f"Removed document '{document_id}' from knowledge base", context="KnowledgeBase")
        return result

    def remove_document(self, document_id: str) -> bool:
        """Synchronous wrapper for aremove_document."""
        import asyncio
        try:
            asyncio.get_running_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                def _run() -> bool:
                    loop = asyncio.new_event_loop()
                    return loop.run_until_complete(self.aremove_document(document_id))
                return executor.submit(_run).result()
        except RuntimeError:
            return asyncio.new_event_loop().run_until_complete(self.aremove_document(document_id))

    async def arefresh(self) -> Dict[str, Any]:
        """
        Re-scan existing sources for changes and re-index modified documents.

        Returns:
            Processing stats dictionary.

        Raises:
            RuntimeError: If the KnowledgeBase has been closed.
        """
        if self._state == KBState.CLOSED:
            raise RuntimeError("Cannot refresh a closed KnowledgeBase.")

        for loader in self.loaders:
            loader.reset()

        self._state = KBState.CONNECTED
        await self.setup_async(force=True)
        return self._processing_stats

    def refresh(self) -> Dict[str, Any]:
        """Synchronous wrapper for arefresh."""
        import asyncio
        try:
            asyncio.get_running_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                def _run() -> Dict[str, Any]:
                    loop = asyncio.new_event_loop()
                    return loop.run_until_complete(self.arefresh())
                return executor.submit(_run).result()
        except RuntimeError:
            return asyncio.new_event_loop().run_until_complete(self.arefresh())

    async def adelete_by_filter(self, metadata_filter: Dict[str, Any]) -> bool:
        """
        Delete all chunks matching a metadata filter.

        Args:
            metadata_filter: Metadata filter to match for deletion.

        Returns:
            True if deletion was successful.
        """
        await self.setup_async()

        try:
            result: bool = await self.vectordb.adelete_by_metadata(metadata_filter)
            if result:
                success_log(f"Deleted chunks matching filter: {metadata_filter}", context="KnowledgeBase")
            return result
        except Exception as e:
            error_log(f"Failed to delete by filter: {e}", context="KnowledgeBase")
            return False

    def delete_by_filter(self, metadata_filter: Dict[str, Any]) -> bool:
        """Synchronous wrapper for adelete_by_filter."""
        import asyncio
        try:
            asyncio.get_running_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                def _run() -> bool:
                    loop = asyncio.new_event_loop()
                    return loop.run_until_complete(self.adelete_by_filter(metadata_filter))
                return executor.submit(_run).result()
        except RuntimeError:
            return asyncio.new_event_loop().run_until_complete(self.adelete_by_filter(metadata_filter))

    async def aupdate_document_metadata(
        self,
        document_id: str,
        metadata_updates: Dict[str, Any],
    ) -> bool:
        """
        Update metadata for all chunks of a document.

        Args:
            document_id: The document ID.
            metadata_updates: Metadata fields to update.

        Returns:
            True if update was successful.
        """
        await self.setup_async()

        try:
            chunks: List[VectorSearchResult] = await self.vectordb.asearch(
                query_vector=None,
                query_text=None,
                filter={"document_id": document_id},
            )

            success: bool = True
            for chunk in chunks:
                result: bool = await self.vectordb.aupdate_metadata(chunk.id, metadata_updates)
                if not result:
                    success = False

            if success:
                success_log(f"Updated metadata for document '{document_id}'", context="KnowledgeBase")
            return success

        except Exception as e:
            error_log(f"Failed to update metadata for document '{document_id}': {e}", context="KnowledgeBase")
            return False

    def update_document_metadata(
        self,
        document_id: str,
        metadata_updates: Dict[str, Any],
    ) -> bool:
        """Synchronous wrapper for aupdate_document_metadata."""
        import asyncio
        try:
            asyncio.get_running_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                def _run() -> bool:
                    loop = asyncio.new_event_loop()
                    return loop.run_until_complete(self.aupdate_document_metadata(document_id, metadata_updates))
                return executor.submit(_run).result()
        except RuntimeError:
            return asyncio.new_event_loop().run_until_complete(
                self.aupdate_document_metadata(document_id, metadata_updates)
            )



    def markdown(self) -> str:
        """Return a markdown representation of the knowledge base."""
        source_strs: List[str] = [str(source) for source in self.sources]
        return f"# Knowledge Base: {self.name}\n\nSources: {', '.join(source_strs)}"
    

    async def get_collection_info_async(self) -> Dict[str, Any]:
        """
        Get detailed information about the vector database collection.
        
        Returns:
            Dictionary containing collection metadata and statistics.
        """
        await self.setup_async()
        
        try:
            # Try provider-specific method
            if hasattr(self.vectordb, 'get_collection_info'):
                if asyncio.iscoroutinefunction(self.vectordb.get_collection_info):
                    return await self.vectordb.get_collection_info()
                else:
                    return self.vectordb.get_collection_info()
            
            exists: bool = await self.vectordb.acollection_exists()
            
            return {
                "collection_name": getattr(self.vectordb._config, 'collection_name', self.knowledge_id),
                "exists": exists,
                "provider": self.vectordb.__class__.__name__,
                "processing_stats": self._processing_stats
            }
            
        except Exception as e:
            error_log(f"Failed to get collection info: {e}", context="KnowledgeBase")
            return {
                "error": str(e),
                "provider": self.vectordb.__class__.__name__
            }

    async def optimize_vectordb(self) -> bool:
        """
        Optimize the vector database for better performance.
        
        Returns:
            True if optimization was successful, False otherwise
        """
        await self.setup_async()
        
        try:
            result: bool = await self.vectordb.aoptimize()

            if result:
                success_log("Vector database optimized successfully", context="KnowledgeBase")
            
            return result
            
        except Exception as e:
            error_log(f"Failed to optimize vector database: {e}", context="KnowledgeBase")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the KnowledgeBase configuration.
        
        Returns:
            Dictionary containing configuration details of all components.
        """
        vectordb_config = {}
        if hasattr(self.vectordb, '_config'):
            config = self.vectordb._config
            vectordb_config = {
                "provider": self.vectordb.__class__.__name__,
                "collection_name": getattr(config, 'collection_name', 'unknown'),
                "vector_size": getattr(config, 'vector_size', 'unknown'),
                "distance_metric": str(getattr(config, 'distance_metric', 'unknown')),
                "dense_search_enabled": getattr(config, 'dense_search_enabled', True),
                "full_text_search_enabled": getattr(config, 'full_text_search_enabled', False),
                "hybrid_search_enabled": getattr(config, 'hybrid_search_enabled', False),
            }
        else:
            vectordb_config = {
                "provider": self.vectordb.__class__.__name__
            }
        
        summary = {
            "knowledge_base": {
                "name": self.name,
                "knowledge_id": self.knowledge_id,
                "sources_count": len(self.sources),
                "state": self._state.value,
                "isolate_search": self.isolate_search,
            },
            "sources": [str(source) for source in self.sources],
            "loaders": {
                "classes": [loader.__class__.__name__ for loader in self.loaders] if self.loaders else [],
                "count": len(self.loaders),
                "indexed_processing": len(self.loaders) > 1
            },
            "splitters": {
                "classes": [splitter.__class__.__name__ for splitter in self.splitters],
                "count": len(self.splitters),
                "indexed_processing": len(self.splitters) > 1
            },
            "embedding_provider": {
                "class": self.embedding_provider.__class__.__name__ if self.embedding_provider else "none",
                "provider": getattr(self.embedding_provider, 'provider', 'unknown') if self.embedding_provider else "none"
            },
            "vectordb": vectordb_config,
            "processing_stats": self._processing_stats
        }
        
        return summary
    
    async def health_check_async(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the KnowledgeBase and its components.
        
        Returns:
            Dictionary containing health status and diagnostic information for all components.
        """
        health_status: Dict[str, Any] = {
            "name": self.name,
            "healthy": False,
            "state": self._state.value,
            "knowledge_id": self.knowledge_id,
            "sources_count": len(self.sources),
            "isolate_search": self.isolate_search,
            "components": {},
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        try:
            # Check embedding provider (skip if not configured)
            if self.embedding_provider is not None:
                health_status["components"]["embedding_provider"] = await self._check_embedding_provider_health()
            else:
                health_status["components"]["embedding_provider"] = {
                    "healthy": True,
                    "provider": "none",
                    "note": "Vectordb handles embeddings internally"
                }
            
            # Check splitters
            health_status["components"]["splitters"] = self._check_splitters_health()
            
            # Check loaders
            health_status["components"]["loaders"] = self._check_loaders_health()
            
            # Check vector database
            health_status["components"]["vectordb"] = await self._check_vectordb_health()
            
            # Add collection info if ready
            if self._state == KBState.INDEXED:
                try:
                    health_status["collection_info"] = await self.get_collection_info_async()
                except Exception as e:
                    health_status["collection_info"] = {"error": str(e)}
            
            # Add processing stats
            if self._processing_stats:
                health_status["processing_stats"] = self._processing_stats
            
            # Overall health determination
            all_healthy = all(
                comp.get("healthy", False)
                for comp in health_status["components"].values()
            )
            health_status["healthy"] = all_healthy and self._state == KBState.INDEXED
            
            return health_status
            
        except Exception as e:
            error_log(f"Health check failed: {e}", context="KnowledgeBase")
            health_status["healthy"] = False
            health_status["error"] = str(e)
            return health_status

    async def _check_embedding_provider_health(self) -> Dict[str, Any]:
        """Check embedding provider health."""
        try:
            if hasattr(self.embedding_provider, 'validate_connection'):
                is_healthy = await self.embedding_provider.validate_connection()
                return {
                    "healthy": is_healthy,
                    "provider": self.embedding_provider.__class__.__name__
                }
            else:
                return {
                    "healthy": True,  # Assume healthy if no validation method
                    "provider": self.embedding_provider.__class__.__name__,
                    "note": "No validation method available"
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "provider": self.embedding_provider.__class__.__name__
            }

    def _check_splitters_health(self) -> Dict[str, Any]:
        """Check splitters health."""
        try:
            splitter_details = [
                {
                    "index": i,
                    "strategy": splitter.__class__.__name__,
                    "healthy": True
                }
                for i, splitter in enumerate(self.splitters)
            ]
            
            return {
                "healthy": True,
                "count": len(self.splitters),
                "details": splitter_details
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    def _check_loaders_health(self) -> Dict[str, Any]:
        """Check loaders health."""
        try:
            if not self.loaders:
                return {
                    "healthy": True,
                    "count": 0,
                    "note": "No loaders configured"
                }
            
            loader_details = [
                {
                    "index": i,
                    "loader": loader.__class__.__name__,
                    "healthy": True
                }
                for i, loader in enumerate(self.loaders)
            ]
            
            return {
                "healthy": True,
                "count": len(self.loaders),
                "details": loader_details
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    async def _check_vectordb_health(self) -> Dict[str, Any]:
        """Check vector database health."""
        try:
            is_ready: bool = await self.vectordb.ais_ready()
            return {
                "healthy": is_ready,
                "provider": self.vectordb.__class__.__name__,
                "connected": self.vectordb._is_connected,
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "provider": self.vectordb.__class__.__name__,
            }
    



    async def close(self) -> None:
        """
        Clean up resources and close connections.
        
        This method should be called when the KnowledgeBase is no longer needed
        to prevent resource leaks. It is idempotent and safe to call multiple times.
        
        Example:
            ```python
            kb = KnowledgeBase(...)
            try:
                await kb.setup_async()
                results = await kb.query_async("query")
            finally:
                await kb.close()  # Always clean up
            ```
        """
        if self._state == KBState.CLOSED:
            debug_log(f"KnowledgeBase '{self.name}' already closed", context="KnowledgeBase")
            return
        
        debug_log(f"Closing KnowledgeBase '{self.name}'...", context="KnowledgeBase")
        
        try:
            if self.embedding_provider is not None and hasattr(self.embedding_provider, 'close'):
                try:
                    if asyncio.iscoroutinefunction(self.embedding_provider.close):
                        await self.embedding_provider.close()
                    else:
                        self.embedding_provider.close()
                    debug_log("Embedding provider closed", context="KnowledgeBase")
                except Exception as e:
                    warning_log(f"Error closing embedding provider: {e}", context="KnowledgeBase")
            
            try:
                await self.vectordb.adisconnect()
                debug_log("Vector database disconnected", context="KnowledgeBase")
            except Exception as e:
                warning_log(f"Error disconnecting vector database: {e}", context="KnowledgeBase")
            
            self._state = KBState.CLOSED
            success_log(
                f"KnowledgeBase '{self.name}' resources cleaned up successfully",
                context="KnowledgeBase"
            )
            
        except Exception as e:
            error_log(f"Error during cleanup: {e}", context="KnowledgeBase")
            self._state = KBState.CLOSED

    def __del__(self) -> None:
        """Best-effort cleanup when garbage collected. Prefer explicit close()."""
        try:
            if not hasattr(self, '_state'):
                return
            
            if self._state in (KBState.CONNECTED, KBState.INDEXED):
                if hasattr(self, 'vectordb'):
                    try:
                        self.vectordb.disconnect()
                    except Exception:
                        pass

                warning_log(
                    f"KnowledgeBase '{getattr(self, 'name', 'Unknown')}' was not explicitly closed. "
                    "Consider using 'async with' context manager or calling close() explicitly.",
                    context="KnowledgeBase"
                )
        except:
            pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False  # Don't suppress exceptions