from __future__ import annotations
import asyncio
import hashlib
import json
import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from ..text_splitter.base import BaseChunker
from ..embeddings.base import EmbeddingProvider
from ..vectordb.base import BaseVectorDBProvider
from ..loaders.base import BaseLoader
from ..loaders.config import LoaderConfig
from ..schemas.data_models import Document, Chunk, RAGSearchResult
from ..loaders.factory import create_intelligent_loaders
from ..text_splitter.factory import create_intelligent_splitters


class KnowledgeBase:
    """
    The central, intelligent orchestrator for a collection of knowledge.

    This class manages the entire lifecycle of documents for a RAG pipeline,
    from ingestion and processing to vector storage and retrieval. It is designed
    to be idempotent and efficient, ensuring that the expensive work of processing
    and embedding data is performed only once for a given set of sources and
    configurations.
    """
    
    def __init__(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        embedding_provider: EmbeddingProvider,
        vectordb: BaseVectorDBProvider,
        splitters: Optional[Union[BaseChunker, List[BaseChunker]]] = None,
        loaders: Optional[Union[BaseLoader, List[BaseLoader]]] = None,
        name: Optional[str] = None,
        use_case: str = "rag_retrieval",
        quality_preference: str = "balanced",
        loader_config: Optional[Dict[str, Any]] = None,
        splitter_config: Optional[Dict[str, Any]] = None,
        **config_kwargs
    ):
        """
        Initializes the KnowledgeBase configuration.

        This is a lightweight operation that sets up the components and calculates a
        unique, deterministic ID for this specific knowledge configuration. No
        data processing or I/O occurs at this stage.

        Args:
            sources: Source identifiers (file path, list of files, directory path, or string content).
            embedding_provider: An instance of a concrete EmbeddingProvider.
            splitters: A single BaseChunker or list of BaseChunker instances.
            vectordb: An instance of a concrete BaseVectorDBProvider.
            loaders: A single BaseLoader or list of BaseLoader instances for different file types.
            name: An optional human-readable name for this knowledge base.
            use_case: The intended use case for chunking optimization.
            quality_preference: Speed vs quality preference ("fast", "balanced", "quality").
            loader_config: Configuration options specifically for loaders.
            splitter_config: Configuration options specifically for splitters.
            **config_kwargs: Additional global configuration options (deprecated, use specific configs instead).
        """

        if not sources:
            raise ValueError("KnowledgeBase must be initialized with at least one source.")

        self.sources = self._resolve_sources(sources)
        
        self.embedding_provider = embedding_provider
        self.vectordb = vectordb
        
        if loaders is None:
            print(f"🔄 Auto-detecting loaders for {len(self.sources)} sources...")
            try:
                config_to_use = loader_config or config_kwargs
                self.loaders = create_intelligent_loaders(self.sources, **config_to_use)
                print(f"✅ Created {len(self.loaders)} intelligent loaders")
            except Exception as e:
                print(f"⚠️ Auto-detection failed: {e}, proceeding without loaders")
                self.loaders = []
        else:
            self.loaders = self._normalize_loaders(loaders)
        
        if splitters is None:
            print(f"🔄 Auto-detecting splitters for {len(self.sources)} sources...")
            try:
                config_to_use = splitter_config or config_kwargs
                self.splitters = create_intelligent_splitters(
                    self.sources,
                    use_case=use_case,
                    quality_preference=quality_preference,
                    **config_to_use
                )
                print(f"✅ Created {len(self.splitters)} intelligent splitters")
            except Exception as e:
                print(f"⚠️ Auto-detection failed: {e}, using default recursive strategy")
                from ..text_splitter.factory import create_chunking_strategy
                self.splitters = [create_chunking_strategy("recursive")]
        else:
            self.splitters = self._normalize_splitters(splitters)

        self._validate_component_counts()

        self.name = name or self._generate_knowledge_id()
        self.knowledge_id: str = self._generate_knowledge_id()
        self.rag = True  
        self._is_ready = False
        self._is_closed = False
        self._setup_lock = asyncio.Lock()

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

    def _normalize_splitters(self, splitters: Union[BaseChunker, List[BaseChunker]]) -> List[BaseChunker]:
        """Normalize splitters to always be a list."""
        if isinstance(splitters, list):
            return splitters
        elif isinstance(splitters, BaseChunker):
            return [splitters]
        else:
            raise ValueError("Splitters must be a BaseChunker or list of BaseChunker instances")

    def _normalize_loaders(self, loaders: Optional[Union[BaseLoader, List[BaseLoader]]]) -> List[BaseLoader]:
        """Normalize loaders to always be a list."""
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
            print(f"Warning: {component_name} index {source_index} out of range, using first {component_name}")
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

    async def setup_async(self) -> None:
        """
        The main just-in-time engine for processing and indexing knowledge.

        This method is idempotent. It checks if the knowledge has already been
        processed and indexed. If so, it does nothing. If not, it executes the
        full data pipeline: Load -> Chunk -> Embed -> Store. A lock is used to
        prevent race conditions in concurrent environments.
        
        Now supports indexed processing where each source uses its corresponding
        loader and splitter.
        """
        async with self._setup_lock:
            if self._is_ready:
                return

            self.vectordb.connect()

            if self.vectordb.collection_exists():
                print(f"KnowledgeBase '{self.name}' is already indexed. Setup is complete.")
                self._is_ready = True
                return

            print(f"KnowledgeBase '{self.name}' not found in vector store. Starting indexed indexing process...")

            all_documents = []
            source_to_documents = {}
            source_to_loader = {}
            source_to_splitter = {}
            
            for source_index, source in enumerate(self.sources):
                source_str = str(source)
                print(f"    Processing source {source_index}: {source_str[:100]}{'...' if len(source_str) > 100 else ''}")
                
                if isinstance(source, str):
                    print(f"      Detected direct content, creating document directly...")
                    try:
                        document = self._create_document_from_content(source, source_index)
                        source_documents = [document]
                        print(f"      ✓ Created document from direct content (length: {len(source)})")
                        all_documents.extend(source_documents)
                        source_to_documents[source_index] = source_documents
                        source_to_loader[source_index] = None
                    except Exception as e:
                        print(f"      ✗ Error creating document from direct content: {e}")
                        continue
                else:
                    if self.loaders:
                        loader = self._get_component_for_source(source_index, self.loaders, "loader")
                        print(f"      Using loader: {loader.__class__.__name__}")
                        
                        print(f"      Checking if {loader.__class__.__name__} can load {source}...")
                        can_load_result = loader.can_load(source)
                        print(f"      can_load result: {can_load_result}")
                        
                        if can_load_result:
                            try:
                                source_documents = loader.load(source)
                                print(f"      ✓ Loaded {len(source_documents)} documents from {source}")
                                all_documents.extend(source_documents)
                                source_to_documents[source_index] = source_documents
                                source_to_loader[source_index] = loader
                            except Exception as e:
                                print(f"      ✗ Error loading {source}: {e}")
                                continue
                        else:
                            print(f"      ✗ Loader {loader.__class__.__name__} cannot handle {source}")
                            continue
                    else:
                        print(f"      ✗ No loaders provided for {source}")
                        continue
            
            if not all_documents:
                self._is_ready = True
                return

            print(f"  [Step 2/4] Chunking {len(all_documents)} documents with indexed splitters...")
            all_chunks = []
            chunks_per_source = {}
            
            for source_index in sorted(source_to_documents.keys()):
                documents = source_to_documents[source_index]
                
                splitter = self._get_component_for_source(source_index, self.splitters, "splitter")
                source_to_splitter[source_index] = splitter
                
                source_chunks = []
                for doc in documents:
                    doc_chunks = splitter.chunk([doc])
                    source_chunks.extend(doc_chunks)
                    print(f"      Document '{doc.document_id}' split into {len(doc_chunks)} chunks")
                
                chunks_per_source[source_index] = source_chunks
                all_chunks.extend(source_chunks)
                print(f"    ✓ Source {source_index} total chunks: {len(source_chunks)}")
            
            print(f"  Summary: Total chunks created: {len(all_chunks)}")
            for source_index, chunks in chunks_per_source.items():
                print(f"    - Source {source_index}: {len(chunks)} chunks")

            print(f"  [Step 3/4] Creating embeddings for {len(all_chunks)} chunks...")
            vectors = await self.embedding_provider.embed_documents(all_chunks)
            print(f"  ✓ Created embeddings for {len(vectors)} chunks")
            
            print(f"  [Step 4/4] Storing {len(all_chunks)} chunks in vector database...")
            self.vectordb.create_collection()
            
            chunk_texts = [chunk.text_content for chunk in all_chunks]
            chunk_metadata = [chunk.metadata for chunk in all_chunks]
            chunk_ids = [chunk.chunk_id for chunk in all_chunks]
            
            self.vectordb.upsert(
                vectors=vectors,
                payloads=chunk_metadata,
                ids=chunk_ids,
                chunks=chunk_texts
            )
            
            self._is_ready = True
            print(f"KnowledgeBase '{self.name}' indexing completed successfully!")



    async def query_async(self, query: str) -> List[RAGSearchResult]:
        """
        Performs a similarity search to retrieve relevant knowledge.

        This is the primary retrieval method. It automatically triggers the setup
        process if it hasn't been run yet. It then embeds the user's query and
        searches the vector database for the most relevant chunks of text.

        Args:
            query: The user's query string.

        Returns:
            A list of RAGSearchResult objects, where each contains the text content
            and metadata of a retrieved chunk.
        """
        await self.setup_async()

        if not self._is_ready:
            return []

        print(f"Querying KnowledgeBase '{self.name}' with: '{query}'")
        
        query_vector = await self.embedding_provider.embed_query(query)

        if hasattr(self.vectordb, '_config') and hasattr(self.vectordb._config, 'search'):
            if self.vectordb._config.search.hybrid_search_enabled:
                search_results = self.vectordb.search(
                    query_vector=query_vector,
                    query_text=query
                )
            else:
                search_results = self.vectordb.search(
                    query_vector=query_vector
                )
        else:
            search_results = self.vectordb.search(
                query_vector=query_vector,
                query_text=query
            )

        rag_results = []
        for result in search_results:
            if not result.text:
                error_msg = f"ERROR: Vector search result {result.id} has no text content. Payload: {result.payload}"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            text_content = result.text
            metadata = result.payload or {}         
            rag_result = RAGSearchResult(
                text=text_content,
                metadata=metadata,
                score=result.score,
                chunk_id=result.id
            )
            rag_results.append(rag_result)

        if len(rag_results) == 0:
            print(f"Warning: No results found for KnowledgeBase '{self.name}' with query: '{query}'")
        
        print(f"✅ Successfully processed {len(rag_results)} results for RAG")
        return rag_results

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
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the KnowledgeBase configuration.
        
        Returns:
            Dictionary containing configuration details of all components.
        """
        summary = {
            "knowledge_base": {
                "name": self.name,
                "knowledge_id": self.knowledge_id,
                "sources": self.sources,
                "is_ready": self._is_ready
            },
            "loaders": {
                "classes": [loader.__class__.__name__ for loader in self.loaders] if self.loaders else [],
                "indexed_processing": len(self.loaders) > 1 if self.loaders else False
            },
            "splitters": {
                "classes": [splitter.__class__.__name__ for splitter in self.splitters],
                "indexed_processing": len(self.splitters) > 1
            },
            "embedding_provider": {
                "class": self.embedding_provider.__class__.__name__
            },
            "vectordb": self.vectordb.get_config_summary() if hasattr(self.vectordb, 'get_config_summary') else {
                "class": self.vectordb.__class__.__name__
            }
        }
        
        return summary
    
    async def health_check_async(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the KnowledgeBase.
        
        Returns:
            Dictionary containing health status and diagnostic information
        """
        health_status = {
            "name": self.name,
            "healthy": False,
            "is_ready": getattr(self, '_is_ready', False),
            "knowledge_id": getattr(self, 'knowledge_id', 'unknown'),
            "type": "rag" if getattr(self, 'rag', True) else "static",
            "sources_count": len(self.sources) if hasattr(self, 'sources') else 0,
            "components": {
                "embedding_provider": {"healthy": False, "error": "Not checked"},
                "splitters": {"healthy": False, "error": "Not checked"},
                "vectordb": {"healthy": False, "error": "Not checked"},
                "loaders": {"healthy": False, "error": "Not checked"}
            },
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        try:
            try:
                if hasattr(self.embedding_provider, 'validate_connection'):
                    embedding_health = await self.embedding_provider.validate_connection()
                    health_status["components"]["embedding_provider"] = {
                        "healthy": embedding_health,
                        "provider": self.embedding_provider.__class__.__name__
                    }
                else:
                    health_status["components"]["embedding_provider"] = {
                        "healthy": True,
                        "provider": self.embedding_provider.__class__.__name__
                    }
            except Exception as e:
                health_status["components"]["embedding_provider"] = {
                    "healthy": False,
                    "error": str(e)
                }
            
            try:
                splitter_health = []
                for i, splitter in enumerate(self.splitters):
                    splitter_health.append({
                        "index": i,
                        "healthy": True,
                        "strategy": splitter.__class__.__name__
                    })
                
                health_status["components"]["splitters"] = {
                    "healthy": True,
                    "count": len(self.splitters),
                    "splitters": splitter_health
                }
            except Exception as e:
                health_status["components"]["splitters"] = {
                    "healthy": False,
                    "error": str(e)
                }
            
            try:
                if hasattr(self.vectordb, 'health_check'):
                    vector_db_health = self.vectordb.health_check()
                    health_status["components"]["vectordb"] = vector_db_health
                else:
                    health_status["components"]["vectordb"] = {
                        "healthy": True,
                        "provider": self.vectordb.__class__.__name__
                    }
                
                if hasattr(self.vectordb, 'get_collection_info'):
                    try:
                        collection_info = self.vectordb.get_collection_info()
                        health_status["collection_info"] = collection_info
                    except Exception as e:
                        health_status["collection_info"] = {
                            "error": f"Failed to get collection info: {str(e)}"
                        }
                
            except Exception as e:
                health_status["components"]["vectordb"] = {
                    "healthy": False,
                    "error": str(e)
                }
            
            try:
                if self.loaders:
                    loader_health = []
                    for i, loader in enumerate(self.loaders):
                        loader_health.append({
                            "index": i,
                            "healthy": True,
                            "loader": loader.__class__.__name__
                        })
                    
                    health_status["components"]["loaders"] = {
                        "healthy": True,
                        "count": len(self.loaders),
                        "loaders": loader_health
                    }
                else:
                    health_status["components"]["loaders"] = {
                        "healthy": True,
                        "loaders": "None (manual setup)"
                    }
            except Exception as e:
                health_status["components"]["loaders"] = {
                    "healthy": False,
                    "error": str(e)
                }
            
            all_healthy = all(
                component.get("healthy", False) 
                for component in health_status["components"].values()
            )
            
            health_status["healthy"] = all_healthy
            
            return health_status
            
        except Exception as e:
            health_status["healthy"] = False
            health_status["error"] = str(e)
            return health_status
    

    async def get_collection_info_async(self) -> Dict[str, Any]:
        """
        Get detailed information about the vector database collection.
        
        Returns:
            Dictionary containing collection metadata and statistics.
        """
        await self.setup_async()
        
        if hasattr(self.vectordb, 'get_collection_info'):
            return self.vectordb.get_collection_info()
        else:
            return {
                "collection_name": self.knowledge_id,
                "exists": self.vectordb.collection_exists(),
                "provider": self.vectordb.__class__.__name__
            }
    
    async def close(self) -> None:
        """
        Clean up resources and close connections.
        
        This method should be called when the KnowledgeBase is no longer needed
        to prevent resource leaks.
        """
        if self._is_closed:
            return
            
        try:
            if hasattr(self.embedding_provider, 'close'):
                await self.embedding_provider.close()
            
            if hasattr(self.vectordb, 'close'):
                await self.vectordb.close()
            elif hasattr(self.vectordb, 'disconnect_async'):
                await self.vectordb.disconnect_async()
            elif hasattr(self.vectordb, 'disconnect'):
                self.vectordb.disconnect()
            
            self._is_closed = True
            print(f"✅ KnowledgeBase '{self.name}' resources cleaned up successfully")
        except Exception as e:
            print(f"⚠️ Warning: Error during KnowledgeBase cleanup: {e}")
            self._is_closed = True
    
    def __del__(self):
        """
        Destructor to ensure cleanup when object is garbage collected.
        """
        try:
            if hasattr(self, '_is_ready') and self._is_ready and not getattr(self, '_is_closed', False):
                try:
                    if hasattr(self, 'vectordb') and self.vectordb:
                        if hasattr(self.vectordb, 'disconnect'):
                            self.vectordb.disconnect()
                        elif hasattr(self.vectordb, 'disconnect_async'):
                            import asyncio
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    pass
                                else:
                                    loop.run_until_complete(self.vectordb.disconnect_async())
                            except RuntimeError:
                                asyncio.run(self.vectordb.disconnect_async())
                except Exception:
                    pass
                print(f"⚠️ Warning: KnowledgeBase '{getattr(self, 'name', 'Unknown')}' was not explicitly closed")
        except:
            pass