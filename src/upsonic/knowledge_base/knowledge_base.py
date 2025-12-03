from __future__ import annotations
import asyncio
import hashlib
import json
from typing import List, Optional, Dict, Any, Union, Literal
from pathlib import Path

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
from upsonic.tools import ToolKit, tool


class KnowledgeBase(ToolKit):
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

    This class serves as the bridge between raw documents and the vector database,
    providing a high-level, framework-agnostic interface for knowledge management.
    """
    
    def __init__(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        embedding_provider: EmbeddingProvider,
        vectordb: BaseVectorDBProvider,
        splitters: Optional[Union[BaseChunker, List[BaseChunker]]] = None,
        loaders: Optional[Union[BaseLoader, List[BaseLoader]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        topics: Optional[List[str]] = None,
        use_case: str = "rag_retrieval",
        quality_preference: str = "balanced",
        loader_config: Optional[Dict[str, Any]] = None,
        splitter_config: Optional[Dict[str, Any]] = None,
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
            embedding_provider: An instance of EmbeddingProvider for converting text to vectors.
            vectordb: An instance of BaseVectorDBProvider for vector storage and retrieval.
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
                embedding_provider=OpenAIEmbedding(),
                vectordb=ChromaProvider(config=chroma_config),
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
        # Set dynamic docstring for search method with description
        self._update_search_docstring()

        # Core components
        self.sources: List[Union[str, Path]] = self._resolve_sources(sources)
        self.embedding_provider: EmbeddingProvider = embedding_provider
        self.vectordb: BaseVectorDBProvider = vectordb
        
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
        self.rag: bool = True  # Flag for RAG-enabled mode
        self._is_ready: bool = False
        self._is_closed: bool = False
        self._setup_lock: asyncio.Lock = asyncio.Lock()
        self._processing_stats: Dict[str, Any] = {}  # Track processing statistics

        info_log(
            f"Initialized KnowledgeBase '{self.name}' with {len(self.sources)} sources, "
            f"{len(self.loaders)} loaders, {len(self.splitters)} splitters",
            context="KnowledgeBase"
        )

    def _update_search_docstring(self) -> None:
        """
        Update the search method's docstring to include the knowledge base description.
        """
        topics_str = ", ".join(self.topics) if self.topics else "general"
        base_docstring = """Search the knowledge base for relevant information using semantic similarity.

        This tool performs intelligent retrieval from the indexed knowledge base for the topics: {topics}.

        Description about knowledge base which the informations will be retrieved from:
            {description}

        Args:
            query: The question, topic, or search query to find relevant information.
                  Can be a natural language question, a topic description, or keywords.
                  Examples: "What is machine learning?", "How does authentication work?",
                  "Python best practices", "API documentation for user management".

        Returns:
            A formatted string containing the most relevant information found in the
            knowledge base. Results are ranked by relevance and presented in a readable
            format. Returns "No relevant information found in the knowledge base."
            if no matches are found.
        """
        # Use __func__ to access the underlying function object for docstring update
        if hasattr(self.search, '__func__'):
            self.search.__func__.__doc__ = base_docstring.format(
                description=self.description,
                topics=topics_str
            )

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
            "file_name": f"direct_content_{source_index}.txt",
            "file_path": f"direct_content_{source_index}",
            "file_size": len(content.encode("utf-8")),
            "creation_datetime_utc": current_time,
            "last_modified_datetime_utc": current_time,
        }
        
        return Document(
            content=content,
            metadata=metadata,
            document_id=content_hash
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
            "embedding_provider": self.embedding_provider.__class__.__name__,
        }
        
        config_string = json.dumps(config_representation, sort_keys=True)
        
        return hashlib.sha256(config_string.encode('utf-8')).hexdigest()

    # ============================================================================
    # Lifecycle Management - Connection and Setup
    # ============================================================================

    async def setup_async(self) -> None:
        """
        The main just-in-time engine for processing and indexing knowledge.

        This method is **idempotent** and **thread-safe**. It:
        1. Checks if knowledge is already indexed (skip if yes)
        2. Connects to the vector database
        3. Loads documents from sources
        4. Chunks documents into embeddable pieces
        5. Generates embeddings
        6. Stores everything in the vector database

        The lock prevents race conditions in concurrent environments.
        Indexed processing is supported where each source uses its corresponding
        loader and splitter.

        Raises:
            VectorDBConnectionError: If database connection fails
            UpsertError: If data ingestion fails
            Exception: For various processing errors

        Example:
            ```python
            kb = KnowledgeBase(sources=["docs/"], ...)
            await kb.setup_async()  # Idempotent - safe to call multiple times
            ```
        """
        async with self._setup_lock:
            if self._is_ready:
                debug_log(
                    f"KnowledgeBase '{self.name}' already set up. Skipping.",
                    context="KnowledgeBase"
                )
                return

            try:
                # Step 0: Connect to vector database
                await self._ensure_connection()

                # Check if collection already exists (idempotency)
                if await self.vectordb.collection_exists():
                    info_log(
                        f"Collection for '{self.name}' already exists. Setup complete.",
                        context="KnowledgeBase"
                    )
                    self._is_ready = True
                    return

                info_log(
                    f"Collection not found. Starting indexing pipeline for '{self.name}'...",
                    context="KnowledgeBase"
                )

                # Step 1: Load documents
                all_documents, processing_metadata = await self._load_documents()
                
                if not all_documents:
                    warning_log(
                        "No documents loaded. Marking as ready but empty.",
                        context="KnowledgeBase"
                    )
                    self._is_ready = True
                    return

                # Step 2: Chunk documents
                all_chunks = await self._chunk_documents(all_documents, processing_metadata)
                
                if not all_chunks:
                    warning_log(
                        "No chunks created. Marking as ready but empty.",
                        context="KnowledgeBase"
                    )
                    self._is_ready = True
                    return

                # Step 3: Generate embeddings
                vectors = await self._generate_embeddings(all_chunks)

                # Step 4: Store in vector database
                await self._store_in_vectordb(all_chunks, vectors)

                # Update stats
                self._processing_stats = {
                    "sources_count": len(self.sources),
                    "documents_count": len(all_documents),
                    "chunks_count": len(all_chunks),
                    "vectors_count": len(vectors),
                    "indexed_at": __import__('datetime').datetime.now().isoformat()
                }

                self._is_ready = True
                success_log(
                    f"KnowledgeBase '{self.name}' indexing completed! "
                    f"{len(all_documents)} docs → {len(all_chunks)} chunks",
                    context="KnowledgeBase"
                )

            except Exception as e:
                error_log(f"Setup failed for '{self.name}': {e}", context="KnowledgeBase")
                # Clean up partial state if needed
                try:
                    if await self.vectordb.collection_exists():
                        warning_log(
                            "Cleaning up partially created collection...",
                            context="KnowledgeBase"
                        )
                        await self.vectordb.delete_collection()
                except:
                    pass  # Best effort cleanup
                raise

    async def _ensure_connection(self) -> None:
        """
        Ensures the vector database is connected.
        Uses async connection if available, falls back to sync.
        """
        if hasattr(self.vectordb, '_is_connected') and self.vectordb._is_connected:
            return
        
        try:
            # Prefer async connection
            if hasattr(self.vectordb, 'connect'):
                await self.vectordb.connect()
            elif hasattr(self.vectordb, 'connect_sync'):
                self.vectordb.connect_sync()
            else:
                # Some providers might auto-connect
                debug_log(
                    "No explicit connect method. Assuming auto-connection.",
                    context="KnowledgeBase"
                )
            
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
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of embedding vectors
        """
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
            # Create collection if it doesn't exist
            if not await self.vectordb.collection_exists():
                if hasattr(self.vectordb, 'create_collection'):
                    await self.vectordb.create_collection()
                elif hasattr(self.vectordb, 'create_collection_sync'):
                    self.vectordb.create_collection_sync()
                else:
                    raise VectorDBConnectionError("No create_collection method available")
            
            # Prepare data for upsert
            chunk_texts = [chunk.text_content for chunk in chunks]
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            chunk_payloads = [chunk.metadata for chunk in chunks]
            
            # Upsert data
            if hasattr(self.vectordb, 'upsert'):
                await self.vectordb.upsert(
                    vectors=vectors,
                    payloads=chunk_payloads,
                    ids=chunk_ids,
                    chunks=chunk_texts
                )
            elif hasattr(self.vectordb, 'upsert_sync'):
                self.vectordb.upsert_sync(
                    vectors=vectors,
                    payloads=chunk_payloads,
                    ids=chunk_ids,
                    chunks=chunk_texts
                )
            else:
                raise VectorDBConnectionError("No upsert method available")
            
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
        search_type: Literal['auto', 'dense', 'full_text', 'hybrid'] = 'auto',
        task: Optional[Any] = None,
        **search_kwargs
    ) -> List[RAGSearchResult]:
        """
        Performs a search to retrieve relevant knowledge chunks.

        This is the primary retrieval method. It automatically triggers setup
        if not done yet, embeds the query, and searches the vector database
        using the most appropriate search method.

        Args:
            query: The user's query string.
            top_k: Number of results to return. If None, uses provider's default or Task's vector_search_top_k.
            filter: Optional metadata filter to apply. If None, uses Task's vector_search_filter if provided.
            search_type: Type of search to perform:
                - 'auto': Automatically choose based on provider capabilities (default)
                - 'dense': Pure vector similarity search
                - 'full_text': Text-based search (if supported)
                - 'hybrid': Combined vector + text search (if supported)
            task: Optional Task object. If provided, uses Task's vector search parameters to override config defaults.
            **search_kwargs: Additional search parameters (alpha, fusion_method, etc.)

        Returns:
            List of RAGSearchResult objects containing text content and metadata.

        Raises:
            ValueError: If search results are invalid

        Example:
            ```python
            # Simple query
            results = await kb.query_async("What is machine learning?")
            
            # Advanced query with filtering
            results = await kb.query_async(
                "What is ML?",
                top_k=5,
                filter={"source": "ml_book.pdf"},
                search_type='hybrid',
                alpha=0.7
            )
            ```
        """
        # Ensure setup has completed
        await self.setup_async()

        if not self._is_ready:
            warning_log(
                f"KnowledgeBase '{self.name}' is not ready. Returning empty results.",
                context="KnowledgeBase"
            )
            return []

        info_log(f"Querying '{self.name}': '{query[:100]}...'", context="KnowledgeBase")
        
        try:
            # Generate query embedding
            query_vector = await self.embedding_provider.embed_query(query)

            # Perform search based on type
            search_results = await self._perform_search(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                filter=filter,
                search_type=search_type,
                task=task,
                **search_kwargs
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

    @tool
    async def search(self, query: str) -> str:
        """
        Search the knowledge base for relevant information using semantic similarity.

        This tool performs intelligent retrieval from the indexed knowledge base, using
        vector similarity search to find the most relevant chunks of information that
        match the query. The search leverages embeddings to understand semantic meaning,
        making it effective for finding information even when exact keywords don't match.

        Args:
            query: The question, topic, or search query to find relevant information.
                  Can be a natural language question, a topic description, or keywords.
                  Examples: "What is machine learning?", "How does authentication work?",
                  "Python best practices", "API documentation for user management".

        Returns:
            A formatted string containing the most relevant information found in the
            knowledge base. Results are ranked by relevance and presented in a readable
            format. Returns "No relevant information found in the knowledge base."
            if no matches are found.

        Use Cases:
            - **Question Answering**: Answer specific questions from documentation or
              knowledge sources. Example: "What are the system requirements?"
            
            - **Information Retrieval**: Find relevant sections or topics from large
              document collections. Example: "Find information about error handling"
            
            - **Code Documentation Lookup**: Search through codebases or technical docs
              for specific functionality. Example: "How to implement authentication?"
            
            - **Research Assistance**: Retrieve relevant information for research or
              analysis tasks. Example: "What are the best practices for database design?"
            
            - **Context Gathering**: Collect relevant context before generating responses
              or performing tasks. Example: "Find all mentions of security protocols"
        """
        results = await self.query_async(query)
        if not results:
            return "No relevant information found in the knowledge base."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(f"Result {i}:\n{result.text}")
            
        return "\n\n".join(formatted_results)

    async def _perform_search(
        self,
        query: str,
        query_vector: List[float],
        top_k: Optional[int],
        filter: Optional[Dict[str, Any]],
        search_type: str,
        task: Optional[Any] = None,
        **search_kwargs
    ) -> List[VectorSearchResult]:
        """
        Perform the appropriate search based on type and provider capabilities.
        
        Args:
            query: Query text
            query_vector: Query embedding
            top_k: Number of results
            filter: Metadata filter
            search_type: Type of search
            task: Optional Task object to get search parameters from
            **search_kwargs: Additional search parameters
            
        Returns:
            List of VectorSearchResult objects
        """
        # Get search parameters from Task if provided, otherwise use config defaults
        config = getattr(self.vectordb, '_config', None)
        
        # Extract Task parameters if provided (override method parameters and config defaults)
        alpha = search_kwargs.pop('alpha', None) if 'alpha' in search_kwargs else None
        fusion_method = search_kwargs.pop('fusion_method', None) if 'fusion_method' in search_kwargs else None
        similarity_threshold = search_kwargs.pop('similarity_threshold', None) if 'similarity_threshold' in search_kwargs else None
        
        # Override with Task parameters if provided and not already set
        if task is not None:
            top_k = top_k if top_k is not None else getattr(task, 'vector_search_top_k', None)
            alpha = alpha if alpha is not None else getattr(task, 'vector_search_alpha', None)
            fusion_method = fusion_method if fusion_method is not None else getattr(task, 'vector_search_fusion_method', None)
            similarity_threshold = similarity_threshold if similarity_threshold is not None else getattr(task, 'vector_search_similarity_threshold', None)
            filter = filter if filter is not None else getattr(task, 'vector_search_filter', None)
        
        # Determine search capabilities
        hybrid_enabled = getattr(config, 'hybrid_search_enabled', False) if config else False
        full_text_enabled = getattr(config, 'full_text_search_enabled', False) if config else False
        
        # Auto-select search type
        if search_type == 'auto':
            if hybrid_enabled:
                search_type = 'hybrid'
            elif full_text_enabled:
                search_type = 'dense'  # Prefer dense for auto
            else:
                search_type = 'dense'
        
        # Perform search
        if search_type == 'hybrid' and hybrid_enabled:
            debug_log("Performing hybrid search", context="KnowledgeBase")
            if hasattr(self.vectordb, 'search'):
                return await self.vectordb.search(
                    query_vector=query_vector,
                    query_text=query,
                    top_k=top_k,
                    filter=filter,
                    alpha=alpha,
                    fusion_method=fusion_method,
                    similarity_threshold=similarity_threshold,
                    **search_kwargs
                )
            elif hasattr(self.vectordb, 'search_sync'):
                return self.vectordb.search_sync(
                    query_vector=query_vector,
                    query_text=query,
                    top_k=top_k,
                    filter=filter,
                    alpha=alpha,
                    fusion_method=fusion_method,
                    similarity_threshold=similarity_threshold,
                    **search_kwargs
                )
        elif search_type == 'full_text' and full_text_enabled:
            debug_log("Performing full-text search", context="KnowledgeBase")
            if hasattr(self.vectordb, 'full_text_search'):
                return await self.vectordb.full_text_search(
                    query_text=query,
                    top_k=top_k or 10,
                    filter=filter,
                    similarity_threshold=similarity_threshold,
                    **search_kwargs
                )
            elif hasattr(self.vectordb, 'full_text_search_sync'):
                return self.vectordb.full_text_search_sync(
                    query_text=query,
                    top_k=top_k or 10,
                    filter=filter,
                    similarity_threshold=similarity_threshold,
                    **search_kwargs
                )
        else:
            # Default to dense search
            debug_log("Performing dense vector search", context="KnowledgeBase")
            if hasattr(self.vectordb, 'search'):
                return await self.vectordb.search(
                    query_vector=query_vector,
                    top_k=top_k,
                    filter=filter,
                    similarity_threshold=similarity_threshold,
                    **search_kwargs
                )
            elif hasattr(self.vectordb, 'search_sync'):
                return self.vectordb.search_sync(
                    query_vector=query_vector,
                    top_k=top_k,
                    filter=filter,
                    similarity_threshold=similarity_threshold,
                    **search_kwargs
                )
        
        raise VectorDBConnectionError("No search method available")

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
    # Document Management Methods
    # ============================================================================

    async def document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in the knowledge base.
        
        Args:
            document_id: The document ID to check
            
        Returns:
            True if the document exists, False otherwise
        """
        await self.setup_async()
        
        if hasattr(self.vectordb, 'async_document_id_exists'):
            return await self.vectordb.async_document_id_exists(document_id)
        elif hasattr(self.vectordb, 'document_id_exists'):
            return self.vectordb.document_id_exists(document_id)
        else:
            warning_log(
                "Vector database does not support document_id_exists check",
                context="KnowledgeBase"
            )
            return False

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks associated with a document.
        
        Args:
            document_id: The document ID to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        await self.setup_async()
        
        try:
            if hasattr(self.vectordb, 'async_delete_by_document_id'):
                result = await self.vectordb.async_delete_by_document_id(document_id)
            elif hasattr(self.vectordb, 'delete_by_document_id'):
                result = self.vectordb.delete_by_document_id(document_id)
            else:
                warning_log(
                    "Vector database does not support delete_by_document_id",
                    context="KnowledgeBase"
                )
                return False
            
            if result:
                success_log(
                    f"Deleted document '{document_id}' from knowledge base",
                    context="KnowledgeBase"
                )
            
            return result
            
        except Exception as e:
            error_log(f"Failed to delete document '{document_id}': {e}", context="KnowledgeBase")
            return False

    async def delete_by_filter(self, metadata_filter: Dict[str, Any]) -> bool:
        """
        Delete all chunks matching a metadata filter.
        
        Args:
            metadata_filter: Metadata filter to match for deletion
            
        Returns:
            True if deletion was successful, False otherwise
        """
        await self.setup_async()
        
        try:
            if hasattr(self.vectordb, 'async_delete_by_metadata'):
                result = await self.vectordb.async_delete_by_metadata(metadata_filter)
            elif hasattr(self.vectordb, 'delete_by_metadata'):
                result = self.vectordb.delete_by_metadata(metadata_filter)
            else:
                warning_log(
                    "Vector database does not support delete_by_metadata",
                    context="KnowledgeBase"
                )
                return False
            
            if result:
                success_log(
                    f"Deleted chunks matching filter: {metadata_filter}",
                    context="KnowledgeBase"
                )
            
            return result
            
        except Exception as e:
            error_log(f"Failed to delete by filter: {e}", context="KnowledgeBase")
            return False

    async def update_document_metadata(
        self, 
        document_id: str, 
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for all chunks of a document.
        
        Args:
            document_id: The document ID
            metadata_updates: Metadata fields to update
            
        Returns:
            True if update was successful, False otherwise
        """
        await self.setup_async()
        
        try:
            # Get all chunks for this document
            if hasattr(self.vectordb, 'search_sync'):
                chunks = self.vectordb.search_sync(
                    query_vector=None,
                    query_text=None,
                    filter={"document_id": document_id}
                )
            else:
                warning_log(
                    "Cannot update metadata: search not supported",
                    context="KnowledgeBase"
                )
                return False
            
            # Update each chunk's metadata
            success = True
            for chunk in chunks:
                if hasattr(self.vectordb, 'async_update_metadata'):
                    result = await self.vectordb.async_update_metadata(
                        chunk.id, metadata_updates
                    )
                elif hasattr(self.vectordb, 'update_metadata'):
                    result = self.vectordb.update_metadata(chunk.id, metadata_updates)
                else:
                    return False
                
                if not result:
                    success = False
            
            if success:
                success_log(
                    f"Updated metadata for document '{document_id}'",
                    context="KnowledgeBase"
                )
            
            return success
            
        except Exception as e:
            error_log(
                f"Failed to update metadata for document '{document_id}': {e}",
                context="KnowledgeBase"
            )
            return False

    # ============================================================================
    # Utility and Compatibility Methods
    # ============================================================================

    async def setup_rag(self) -> None:
        """
        Setup RAG functionality for the knowledge base.
        This method is called by the context manager when RAG is enabled.
        """
        await self.setup_async()

    def markdown(self) -> str:
        """
        Return a markdown representation of the knowledge base.
        Used when RAG is disabled.
        """
        # Convert sources to strings to handle Path objects
        source_strs = [str(source) for source in self.sources]
        return f"# Knowledge Base: {self.name}\n\nSources: {', '.join(source_strs)}"
    
    # ============================================================================
    # Collection and Health Management
    # ============================================================================

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
            
            # Fallback to basic info
            exists = await self.vectordb.collection_exists() if hasattr(self.vectordb, 'collection_exists') else False
            
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
            if hasattr(self.vectordb, 'async_optimize'):
                result = await self.vectordb.async_optimize()
            elif hasattr(self.vectordb, 'optimize'):
                result = self.vectordb.optimize()
            else:
                debug_log(
                    "Vector database does not support optimization",
                    context="KnowledgeBase"
                )
                return False
            
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
                "is_ready": self._is_ready,
                "is_closed": self._is_closed
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
                "class": self.embedding_provider.__class__.__name__,
                "provider": getattr(self.embedding_provider, 'provider', 'unknown')
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
        health_status = {
            "name": self.name,
            "healthy": False,
            "is_ready": self._is_ready,
            "is_closed": self._is_closed,
            "knowledge_id": self.knowledge_id,
            "sources_count": len(self.sources),
            "components": {},
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        try:
            # Check embedding provider
            health_status["components"]["embedding_provider"] = await self._check_embedding_provider_health()
            
            # Check splitters
            health_status["components"]["splitters"] = self._check_splitters_health()
            
            # Check loaders
            health_status["components"]["loaders"] = self._check_loaders_health()
            
            # Check vector database
            health_status["components"]["vectordb"] = await self._check_vectordb_health()
            
            # Add collection info if ready
            if self._is_ready:
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
            health_status["healthy"] = all_healthy and self._is_ready and not self._is_closed
            
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
            # Try provider-specific health check
            if hasattr(self.vectordb, 'is_ready'):
                if asyncio.iscoroutinefunction(self.vectordb.is_ready):
                    is_ready = await self.vectordb.is_ready()
                else:
                    is_ready = self.vectordb.is_ready()
                
                return {
                    "healthy": is_ready,
                    "provider": self.vectordb.__class__.__name__,
                    "connected": getattr(self.vectordb, '_is_connected', False)
                }
            else:
                # Fallback check
                return {
                    "healthy": getattr(self.vectordb, '_is_connected', False),
                    "provider": self.vectordb.__class__.__name__,
                    "note": "No is_ready method available"
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "provider": self.vectordb.__class__.__name__
            }
    

    # ============================================================================
    # Resource Management and Cleanup
    # ============================================================================

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
        if self._is_closed:
            debug_log(f"KnowledgeBase '{self.name}' already closed", context="KnowledgeBase")
            return
        
        debug_log(f"Closing KnowledgeBase '{self.name}'...", context="KnowledgeBase")
        
        try:
            # Close embedding provider
            if hasattr(self.embedding_provider, 'close'):
                try:
                    if asyncio.iscoroutinefunction(self.embedding_provider.close):
                        await self.embedding_provider.close()
                    else:
                        self.embedding_provider.close()
                    debug_log("Embedding provider closed", context="KnowledgeBase")
                except Exception as e:
                    warning_log(f"Error closing embedding provider: {e}", context="KnowledgeBase")
            
            # Close vector database
            if hasattr(self.vectordb, 'disconnect'):
                try:
                    if asyncio.iscoroutinefunction(self.vectordb.disconnect):
                        await self.vectordb.disconnect()
                    else:
                        self.vectordb.disconnect()
                    debug_log("Vector database disconnected", context="KnowledgeBase")
                except Exception as e:
                    warning_log(f"Error disconnecting vector database: {e}", context="KnowledgeBase")
            elif hasattr(self.vectordb, 'disconnect_sync'):
                try:
                    self.vectordb.disconnect_sync()
                    debug_log("Vector database disconnected (sync)", context="KnowledgeBase")
                except Exception as e:
                    warning_log(f"Error disconnecting vector database: {e}", context="KnowledgeBase")
            
            # Mark as closed
            self._is_closed = True
            success_log(
                f"KnowledgeBase '{self.name}' resources cleaned up successfully",
                context="KnowledgeBase"
            )
            
        except Exception as e:
            error_log(f"Error during cleanup: {e}", context="KnowledgeBase")
            self._is_closed = True  # Mark as closed even if cleanup had errors

    def __del__(self):
        """
        Destructor to ensure cleanup when object is garbage collected.
        
        Note: This is a best-effort cleanup. It's better to explicitly call close().
        """
        try:
            if not hasattr(self, '_is_closed'):
                return
            
            if self._is_ready and not self._is_closed:
                # Try sync disconnect as last resort
                if hasattr(self, 'vectordb'):
                    try:
                        if hasattr(self.vectordb, 'disconnect_sync'):
                            self.vectordb.disconnect_sync()
                        elif hasattr(self.vectordb, 'disconnect'):
                            # Only call if it's not async
                            if not asyncio.iscoroutinefunction(self.vectordb.disconnect):
                                self.vectordb.disconnect()
                    except Exception:
                        pass  # Ignore errors in destructor
                
                # Warn about improper cleanup
                warning_log(
                    f"KnowledgeBase '{getattr(self, 'name', 'Unknown')}' was not explicitly closed. "
                    "Consider using 'async with' context manager or calling close() explicitly.",
                    context="KnowledgeBase"
                )
        except:
            pass  # Never let destructor raise exceptions

    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False  # Don't suppress exceptions