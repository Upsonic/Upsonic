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
from ..loaders.factory import LoaderFactory, AutoLoader
from ..loaders.config import LoaderConfig
from ..schemas.data_models import Document, Chunk, RAGSearchResult


class KnowledgeBase:
    """
    The central, intelligent orchestrator for a collection of knowledge.

    This class manages the entire lifecycle of documents for a RAG pipeline,
    from ingestion and processing to vector storage and retrieval. It is designed
    to be idempotent and efficient, ensuring that the expensive work of processing
    and embedding data is performed only once for a given set of sources and
    configurations.
    
    Enhanced with intelligent loader auto-detection and configuration:
    - Automatically detects file types and uses appropriate loaders
    - Supports both simple and advanced loader configuration
    - Provides framework-level loading capabilities
    - Backward compatible with existing loader parameter
    """
    
    def __init__(
        self,
        sources: Union[str, List[str]],
        embedding_provider: EmbeddingProvider,
        splitter: ChunkingStrategy,
        vectordb: BaseVectorDBProvider,
        loaders: Optional[List[DocumentLoader]] = None,
        name: Optional[str] = None,
        loader_config: Optional[Union[Dict[str, Any], LoaderConfig]] = None,
        loader_configs: Optional[Dict[str, Union[Dict[str, Any], LoaderConfig]]] = None,
        auto_detect_loaders: bool = True,
    ):
        """
        Initializes the KnowledgeBase configuration.

        This is a lightweight operation that sets up the components and calculates a
        unique, deterministic ID for this specific knowledge configuration. No
        data processing or I/O occurs at this stage.

        Args:
            sources: Source identifiers (file path, list of files, or directory path).
            embedding_provider: An instance of a concrete EmbeddingProvider.
            splitter: An instance of a concrete ChunkingStrategy.
            vectordb: An instance of a concrete BaseVectorDBProvider.
            loaders: A list of DocumentLoader instances for different file types.
            name: An optional human-readable name for this knowledge base.
            loader_config: Default configuration for all loaders (when auto-detecting).
            loader_configs: Specific configurations for each loader type.
            auto_detect_loaders: If True, automatically detect file types and create appropriate loaders.
        """

        if not sources:
            raise ValueError("KnowledgeBase must be initialized with at least one source.")

        self.sources = self._process_sources(sources)
        
        self.embedding_provider = embedding_provider
        self.splitter = splitter
        self.vectordb = vectordb
        self.name = name or self._generate_knowledge_id()
        self.auto_detect_loaders = auto_detect_loaders
        
        self.loaders = self._create_intelligent_loaders(
            loaders, loader_config, loader_configs, auto_detect_loaders
        )

        self.knowledge_id: str = self._generate_knowledge_id()
        self.rag = True  
        self._is_ready = False
        self._setup_lock = asyncio.Lock()

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
                return [sources]
        elif isinstance(sources, list):
            return sources
        else:
            raise ValueError("Sources must be a string (file path or directory) or list of strings (file paths)")

    def _create_intelligent_loaders(
        self,
        loaders: Optional[List[DocumentLoader]],
        loader_config: Optional[Union[Dict[str, Any], LoaderConfig]],
        loader_configs: Optional[Dict[str, Union[Dict[str, Any], LoaderConfig]]],
        auto_detect_loaders: bool
    ) -> List[DocumentLoader]:
        """
        Create intelligent loaders based on the provided configuration.
        
        This method provides backward compatibility while enabling enhanced functionality.
        """
        if loaders is not None:
            return loaders
        
        if not auto_detect_loaders:
            from ..loaders import TextLoader
            return [TextLoader()]
        
        kb_optimized_configs = {
            "pdf": {
                "load_strategy": "one_document_per_page",
                "use_ocr": False,  
                "error_handling": "warn"
            },
            "csv": {
                "content_synthesis_mode": "concatenated",
                "row_as_document": True,
                "error_handling": "warn"
            },
            "text": {
                "error_handling": "warn"
            },
            "docx": {
                "include_tables": True,
                "error_handling": "warn"
            },
            "json": {
                "jq_schema": ".",
                "flatten_metadata": True,
                "error_handling": "warn"
            },
            "markdown": {
                "parse_front_matter": True,
                "include_code_blocks": True,
                "error_handling": "warn"
            },
            "xml": {
                "content_synthesis_mode": "smart_text",
                "strip_namespaces": True,
                "error_handling": "warn"
            },
            "yaml": {
                "content_synthesis_mode": "canonical_yaml",
                "flatten_metadata": True,
                "error_handling": "warn"
            },
            "html": {
                "extract_text": True,
                "preserve_structure": True,
                "error_handling": "warn"
            }
        }
        
        if loader_configs:
            for loader_type, config in loader_configs.items():
                if loader_type in kb_optimized_configs:
                    if isinstance(config, dict):
                        kb_optimized_configs[loader_type].update(config)
                    else:
                        kb_optimized_configs[loader_type] = config
                else:
                    kb_optimized_configs[loader_type] = config
        
        try:
            auto_loader = AutoLoader(
                sources=self.sources,
                default_config=loader_config,
                loader_configs=kb_optimized_configs
            )
            return [auto_loader]
        except Exception as e:
            print(f"Warning: Failed to create AutoLoader: {e}. Falling back to basic text loader.")
            from ..loaders import TextLoader
            return [TextLoader()]

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
            "loaders": [loader.__class__.__name__ for loader in self.loaders],
            "splitter": self.splitter.__class__.__name__,
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
        """
        async with self._setup_lock:
            if self._is_ready:
                return

            self.vectordb.connect()

            if self.vectordb.collection_exists():
                print(f"KnowledgeBase '{self.name}' is already indexed. Setup is complete.")
                self._is_ready = True
                return

            print(f"KnowledgeBase '{self.name}' not found in vector store. Starting indexing process...")

            print(f"  [Step 1/4] Loading {len(self.sources)} source(s)...")
            all_documents = []
            
            for source in self.sources:
                source_documents = []
                for loader in self.loaders:
                    if loader.can_load(source):
                        try:
                            source_documents = loader.load(source)
                            break
                        except Exception as e:
                            print(f"Warning: Failed to load {source} with {loader.__class__.__name__}: {e}")
                            continue
                
                if not source_documents:
                    print(f"Warning: No documents loaded from {source}")
                else:
                    all_documents.extend(source_documents)
            
            if not all_documents:
                self._is_ready = True
                return

            all_chunks = []
            for doc in all_documents:
                doc_chunks = self.splitter.chunk(doc)
                all_chunks.extend(doc_chunks)

            vectors = await self.embedding_provider.embed_documents(all_chunks)
            
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
            
            rag_result = RAGSearchResult(
                text=text_content,
                metadata=result.payload or {},
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
                "classes": [loader.__class__.__name__ for loader in self.loaders],
                "auto_detect_enabled": self.auto_detect_loaders
            },
            "splitter": {
                "class": self.splitter.__class__.__name__
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
                "splitter": {"healthy": False, "error": "Not checked"},
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
                health_status["components"]["splitter"] = {
                    "healthy": True,
                    "strategy": self.splitter.__class__.__name__
                }
            except Exception as e:
                health_status["components"]["splitter"] = {
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
                    health_status["components"]["loaders"] = {
                        "healthy": True,
                        "loaders": [loader.__class__.__name__ for loader in self.loaders]
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