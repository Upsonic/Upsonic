from __future__ import annotations
from typing import Dict, Type, List, Optional, Any
from pathlib import Path

from .base import DocumentLoader
from .config import LoaderConfig, LoaderConfigFactory
from upsonic.schemas.data_models import Document


class LoaderRegistry:
    """Registry for managing document loader classes."""
    
    def __init__(self):
        self._loaders: Dict[str, Type[DocumentLoader]] = {}
        self._extensions: Dict[str, str] = {}
        self._aliases: Dict[str, str] = {}
        
    def register(
        self, 
        name: str, 
        loader_class: Type[DocumentLoader], 
        extensions: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None
    ):
        """
        Register a loader class.
        
        Args:
            name: Unique name for the loader
            loader_class: The loader class to register
            extensions: File extensions this loader handles
            aliases: Alternative names for this loader
        """
        self._loaders[name] = loader_class
        
        if extensions:
            for ext in extensions:
                ext = ext.lower()
                if not ext.startswith('.'):
                    ext = f'.{ext}'
                self._extensions[ext] = name
        
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name
    
    def get_loader_class(self, name: str) -> Optional[Type[DocumentLoader]]:
        """Get a loader class by name."""
        if name in self._loaders:
            return self._loaders[name]
        
        if name in self._aliases:
            real_name = self._aliases[name]
            return self._loaders.get(real_name)
        
        return None
    
    def get_loader_for_extension(self, extension: str) -> Optional[Type[DocumentLoader]]:
        """Get a loader class for a file extension."""
        ext = extension.lower()
        if not ext.startswith('.'):
            ext = f'.{ext}'
        
        loader_name = self._extensions.get(ext)
        if loader_name:
            return self._loaders.get(loader_name)
        
        return None
    
    def get_loader_for_file(self, file_path: str) -> Optional[Type[DocumentLoader]]:
        """Get a loader class for a file path."""
        extension = Path(file_path).suffix
        return self.get_loader_for_extension(extension)
    
    def list_loaders(self) -> List[str]:
        """List all registered loader names."""
        return list(self._loaders.keys())
    
    def list_extensions(self) -> List[str]:
        """List all supported file extensions."""
        return list(self._extensions.keys())


_registry = LoaderRegistry()


class LoaderFactory:
    """
    Factory for creating and managing document loaders.
    
    This factory provides multiple ways to create loaders:
    1. High-level: Auto-detect loader type from file extension
    2. Medium-level: Specify loader type with optional configuration
    3. Low-level: Full control with custom loader classes and configs
    """
    
    @staticmethod
    def create_loader(
        loader_type: Optional[str] = None,
        config: Optional[LoaderConfig] = None
    ) -> DocumentLoader:
        """
        Create a loader instance.
        
        Args:
            loader_type: Type of loader (auto-detected if None)
            config: Loader configuration object (must inherit from LoaderConfig)
            
        Returns:
            Configured DocumentLoader instance
        """
        if loader_type is None:
            raise ValueError("loader_type must be specified when not auto-detecting")
        
        loader_class = _registry.get_loader_class(loader_type)
        if not loader_class:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        return loader_class(config)
    
    @staticmethod
    def create_loader_for_file(
        file_path: str,
        config: Optional[LoaderConfig] = None
    ) -> DocumentLoader:
        """
        Create a loader for a specific file by auto-detecting the type.
        
        Args:
            file_path: Path to the file
            config: Optional configuration object (must inherit from LoaderConfig)
            
        Returns:
            Configured DocumentLoader instance
        """
        loader_class = _registry.get_loader_for_file(file_path)
        if not loader_class:
            loader_class = _registry.get_loader_class('text')
            if not loader_class:
                raise ValueError(f"No loader available for file: {file_path}")
        
        loader_type = None
        for name, cls in _registry._loaders.items():
            if cls == loader_class:
                loader_type = name
                break
        
        if not loader_type:
            loader_type = 'text'
        
        return LoaderFactory.create_loader(loader_type, config)
    
    @staticmethod
    def create_auto_loader(
        sources: List[str],
        default_config: Optional[LoaderConfig] = None,
        loader_configs: Optional[Dict[str, LoaderConfig]] = None
    ) -> 'AutoLoader':
        """
        Create an automatic loader that handles multiple file types.
        
        Args:
            sources: List of source files
            default_config: Default configuration for all loaders
            loader_configs: Specific configurations for each loader type
            
        Returns:
            AutoLoader instance
        """
        return AutoLoader(sources, default_config, loader_configs)
    
    @staticmethod
    def register_loader(
        name: str,
        loader_class: Type[DocumentLoader],
        extensions: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None
    ):
        """Register a new loader class."""
        _registry.register(name, loader_class, extensions, aliases)
    
    @staticmethod
    def list_loaders() -> List[str]:
        """List all available loader types."""
        return _registry.list_loaders()
    
    @staticmethod
    def list_extensions() -> List[str]:
        """List all supported file extensions."""
        return _registry.list_extensions()


class AutoLoader(DocumentLoader):
    """
    Automatic loader that handles multiple file types intelligently.
    
    This loader analyzes each source file and automatically selects
    the appropriate specialized loader for each file type.
    """
    
    def __init__(
        self,
        sources: List[str],
        default_config: Optional[LoaderConfig] = None,
        loader_configs: Optional[Dict[str, LoaderConfig]] = None
    ):
        """
        Initialize AutoLoader.
        
        Args:
            sources: List of source files to handle
            default_config: Default configuration for all loaders
            loader_configs: Specific configurations per loader type
        """
        super().__init__()
        self.sources = sources
        self.default_config = default_config
        self.loader_configs = loader_configs or {}
        self._loaders: Dict[str, DocumentLoader] = {}
        
        self._initialize_loaders()
    
    def _initialize_loaders(self):
        """Pre-create loaders for all detected file types."""
        file_types = set()
        
        for source in self.sources:
            ext = Path(source).suffix.lower()
            loader_class = _registry.get_loader_for_extension(ext)
            if loader_class:
                for name, cls in _registry._loaders.items():
                    if cls == loader_class:
                        file_types.add(name)
                        break
            else:
                file_types.add('text')
        
        for loader_type in file_types:
            config = self.loader_configs.get(loader_type, self.default_config)
            self._loaders[loader_type] = LoaderFactory.create_loader(loader_type, config)
    
    def load(self, source: str) -> List[Document]:
        """Load a single source using the appropriate loader."""
        loader_class = _registry.get_loader_for_file(source)
        loader_type = 'text'
        
        if loader_class:
            for name, cls in _registry._loaders.items():
                if cls == loader_class:
                    loader_type = name
                    break
        
        if loader_type in self._loaders:
            return self._loaders[loader_type].load(source)
        else:
            config = self.loader_configs.get(loader_type, self.default_config)
            loader = LoaderFactory.create_loader(loader_type, config)
            self._loaders[loader_type] = loader
            return loader.load(source)
    
    def load_all(self) -> List[Document]:
        """Load all sources."""
        all_documents = []
        for source in self.sources:
            documents = self.load(source)
            all_documents.extend(documents)
        return all_documents
    
    async def load_all_async(self) -> List[Document]:
        """Load all sources asynchronously."""
        import asyncio
        
        async def load_source(source: str):
            return await self.load_async(source)
        
        tasks = [load_source(source) for source in self.sources]
        results = await asyncio.gather(*tasks)
        
        all_documents = []
        for documents in results:
            all_documents.extend(documents)
        return all_documents
    
    def get_loader_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all used loaders."""
        stats = {}
        for loader_type, loader in self._loaders.items():
            stats[loader_type] = loader.get_stats()
        return stats


def create_simple_loader(loader_type: str) -> DocumentLoader:
    """Create a simple loader with default configuration."""
    return LoaderFactory.create_loader(loader_type)


def create_configured_loader(loader_type: str, config: Optional[LoaderConfig] = None) -> DocumentLoader:
    """Create a loader with custom configuration."""
    return LoaderFactory.create_loader(loader_type, config)


def load_file(file_path: str, config: Optional[LoaderConfig] = None) -> List[Document]:
    """Load a single file using auto-detected loader."""
    loader = LoaderFactory.create_loader_for_file(file_path, config)
    return loader.load(file_path)


def load_files(file_paths: List[str], config: Optional[LoaderConfig] = None) -> List[Document]:
    """Load multiple files using auto-detected loaders."""
    auto_loader = LoaderFactory.create_auto_loader(file_paths, config)
    return auto_loader.load_all()


async def load_files_async(file_paths: List[str], config: Optional[LoaderConfig] = None) -> List[Document]:
    """Load multiple files asynchronously using auto-detected loaders."""
    auto_loader = LoaderFactory.create_auto_loader(file_paths, config)
    return await auto_loader.load_all_async()
