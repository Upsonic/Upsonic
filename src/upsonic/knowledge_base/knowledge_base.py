from __future__ import annotations
import asyncio
import hashlib
import json
import os
from typing import List, Optional, Dict, Any, Union

from ..text_splitter.base import ChunkingStrategy
from ..embeddings.base import EmbeddingProvider
from ..vectordb.base import BaseVectorDBProvider
from ..loaders.base import DocumentLoader
from ..loaders.config import LoaderConfig
from ..schemas.data_models import Document, Chunk, RAGSearchResult
from ..text_splitter.factory import create_intelligent_splitters, ChunkingUseCase
from ..loaders.factory import create_intelligent_loaders


class KnowledgeBase:
    """
    The central, intelligent orchestrator for a collection of knowledge.

    This class manages the entire lifecycle of documents for a RAG pipeline,
    from ingestion and processing to vector storage and retrieval. It is designed
    to be idempotent and efficient, ensuring that the expensive work of processing
    and embedding data is performed only once for a given set of sources and
    configurations.
    
    Enhanced with intelligent loader and splitter auto-detection and configuration:
    - Automatically detects file types and uses appropriate loaders
    - Supports both simple and advanced loader configuration
    - Provides framework-level loading capabilities
    - Backward compatible with existing loader parameter
    - Supports indexed processing where each source uses corresponding loader/splitter
    """
    
    def __init__(
        self,
        sources: Union[str, List[str]],
        embedding_provider: EmbeddingProvider,
        vectordb: BaseVectorDBProvider,
        splitters: Optional[Union[ChunkingStrategy, List[ChunkingStrategy]]] = None,
        loaders: Optional[Union[DocumentLoader, List[DocumentLoader]]] = None,
        name: Optional[str] = None,
        use_case: ChunkingUseCase = ChunkingUseCase.RAG_RETRIEVAL,
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
            sources: Source identifiers (file path, list of files, or directory path).
            embedding_provider: An instance of a concrete EmbeddingProvider.
            splitters: A single ChunkingStrategy or list of ChunkingStrategy instances.
            vectordb: An instance of a concrete BaseVectorDBProvider.
            loaders: A single DocumentLoader or list of DocumentLoader instances for different file types.
            name: An optional human-readable name for this knowledge base.
            use_case: The intended use case for chunking optimization.
            quality_preference: Speed vs quality preference ("fast", "balanced", "quality").
            loader_config: Configuration options specifically for loaders.
            splitter_config: Configuration options specifically for splitters.
            **config_kwargs: Additional global configuration options (deprecated, use specific configs instead).
        """

        if not sources:
            raise ValueError("KnowledgeBase must be initialized with at least one source.")

        self.sources = self._process_sources(sources)
        
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
        self._setup_lock = asyncio.Lock()

    def _normalize_splitters(self, splitters: Union[ChunkingStrategy, List[ChunkingStrategy]]) -> List[ChunkingStrategy]:
        """Normalize splitters to always be a list."""
        if isinstance(splitters, list):
            return splitters
        elif isinstance(splitters, ChunkingStrategy):
            return [splitters]
        else:
            raise ValueError("Splitters must be a ChunkingStrategy or list of ChunkingStrategy instances")

    def _normalize_loaders(self, loaders: Optional[Union[DocumentLoader, List[DocumentLoader]]]) -> List[DocumentLoader]:
        """Normalize loaders to always be a list."""
        if loaders is None:
            return []
        elif isinstance(loaders, list):
            return loaders
        elif isinstance(loaders, DocumentLoader):
            return [loaders]
        else:
            raise ValueError("Loaders must be a DocumentLoader or list of DocumentLoader instances")

    def _validate_component_counts(self):
        """Validate that component counts are compatible for indexed processing."""
        source_count = len(self.sources)
        splitter_count = len(self.splitters)
        loader_count = len(self.loaders) if self.loaders else 0
        
        
        if source_count > 1:
            if splitter_count > 1 and splitter_count != source_count:
                raise ValueError(
                    f"Number of splitters ({splitter_count}) must match number of sources ({source_count}) "
                    "for indexed processing"
                )
            
            if loader_count > 1 and loader_count != source_count:
                raise ValueError(
                    f"Number of loaders ({loader_count}) must match number of sources ({source_count}) "
                    "for indexed processing"
                )

    def _process_sources(self, sources: Union[str, List[str]]) -> List[str]:
        """
        Process sources to handle different input formats.
        
        Args:
            sources: Can be a single file path, list of file paths, or directory path
            
        Returns:
            List of file paths
        """
        if isinstance(sources, str):
            if os.path.isdir(sources):
                file_paths = []
                for root, dirs, files in os.walk(sources):
                    for file in files:
                        file_paths.append(os.path.join(root, file))
                return file_paths
            else:
                if not os.path.exists(sources):
                    raise ValueError(f"Source file does not exist: {sources}")
                return [sources]
        elif isinstance(sources, list):
            for source in sources:
                if not os.path.exists(source):
                    raise ValueError(f"Source file does not exist: {source}")
            return sources
        else:
            raise ValueError("Sources must be a string (file path or directory) or list of strings (file paths)")

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
        config_representation = {
            "sources": sorted(self.sources),
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
                print(f"    Processing source {source_index}: {source}")
                
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
                    doc_chunks = splitter.chunk(doc)
                    
                    for chunk in doc_chunks:
                        chunk.metadata.update({
                            'source_index': source_index,
                            'source_path': self.sources[source_index],
                            'source_file': os.path.basename(self.sources[source_index]),
                            'loader_type': source_to_loader[source_index].__class__.__name__,
                            'chunking_strategy': splitter.__class__.__name__,
                            'original_document_id': doc.document_id
                        })
                    
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

        search_results = self.vectordb.search(
            query_vector=query_vector,
            query_text=query
        )

        rag_results = []
        for result in search_results:
            text_content = result.text or result.payload.get('text_content', str(result.payload))
            
            metadata = result.payload or {}         
            rag_result = RAGSearchResult(
                text=text_content,
                metadata=metadata,
                score=result.score,
                chunk_id=result.id
            )
            rag_results.append(rag_result)

        return rag_results

    async def setup_rag(self, agent) -> None:
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
        return f"# Knowledge Base: {self.name}\n\nSources: {', '.join(self.sources)}"
    
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
    
    async def close(self):
        """
        Clean up resources and close connections.
        
        This method should be called when the KnowledgeBase is no longer needed
        to prevent resource leaks.
        """
        try:
            if hasattr(self.embedding_provider, 'close'):
                await self.embedding_provider.close()
            
            if hasattr(self.vectordb, 'close'):
                await self.vectordb.close()
            elif hasattr(self.vectordb, 'disconnect_async'):
                await self.vectordb.disconnect_async()
            elif hasattr(self.vectordb, 'disconnect'):
                self.vectordb.disconnect()
            
            print(f"✅ KnowledgeBase '{self.name}' resources cleaned up successfully")
        except Exception as e:
            print(f"⚠️ Warning: Error during KnowledgeBase cleanup: {e}")
    
    def __del__(self):
        """
        Destructor to ensure cleanup when object is garbage collected.
        """
        try:
            if hasattr(self, '_is_ready') and self._is_ready:
                print(f"⚠️ Warning: KnowledgeBase '{getattr(self, 'name', 'Unknown')}' was not explicitly closed")
        except:
            pass