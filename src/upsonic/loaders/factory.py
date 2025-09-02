from __future__ import annotations
from typing import Dict, Type, Optional, List, Union, Any
import os
from pathlib import Path
import logging
from functools import lru_cache

from .base import DocumentLoader
from .config import LoaderConfig, LoaderConfigFactory
from .text import TextLoader
from .csv import CSVLoader
from .pdf import PDFLoader
from .docx import DOCXLoader
from .json import JSONLoader
from .xml import XMLLoader
from .yaml import YAMLLoader
from .markdown import MarkdownLoader
from .html import HTMLLoader


class LoaderFactory:
    def __init__(self):
        self._loaders: Dict[str, Type[DocumentLoader]] = {}
        self._extensions: Dict[str, str] = {}
        self._configs: Dict[str, Type[LoaderConfig]] = {}
        self._logger = logging.getLogger(__name__)
        
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        default_loaders = [
            (TextLoader, ['.txt']),
            (CSVLoader, ['.csv']),
            (PDFLoader, ['.pdf']),
            (DOCXLoader, ['.docx']),
            (JSONLoader, ['.json', '.jsonl', '.js']),
            (XMLLoader, ['.xml', '.xhtml', '.rss', '.atom']),
            (YAMLLoader, ['.yaml', '.yml']),
            (MarkdownLoader, ['.md', '.markdown']),
            (HTMLLoader, ['.html', '.htm', '.xhtml'])
        ]
        
        for loader_class, extensions in default_loaders:
            self.register_loader(loader_class, extensions)
    
    def register_loader(self, loader_class: Type[DocumentLoader], extensions: List[str]):
        loader_name = loader_class.__name__.lower().replace('loader', '')
        
        self._loaders[loader_name] = loader_class
        
        for ext in extensions:
            self._extensions[ext.lower()] = loader_name
        
        try:
            config_name = f"{loader_name.title()}Config"
            config_class = getattr(loader_class, 'config_class', None)
            if config_class:
                self._configs[loader_name] = config_class
        except AttributeError:
            pass
        
        self._logger.debug(f"Registered loader '{loader_name}' with extensions: {extensions}")
    
    def get_loader(self, source: str, loader_type: Optional[str] = None, **config_kwargs) -> DocumentLoader:
        try:
            if loader_type:
                loader_name = loader_type.lower()
            else:
                loader_name = self._detect_loader_type(source)
            
            if loader_name not in self._loaders:
                raise ValueError(f"No loader found for type '{loader_type}'")
            
            loader_class = self._loaders[loader_name]
            
            config = self._create_config(loader_name, **config_kwargs)
            
            return loader_class(config)
            
        except Exception as e:
            self._logger.error(f"Failed to create loader for '{source}': {e}")
            raise
    
    def _detect_loader_type(self, source: str) -> str:
        if source.startswith(('http://', 'https://', 'ftp://')):
            return 'html'
        
        if os.path.exists(source) and os.path.isfile(source):
            file_path = Path(source)
            extension = file_path.suffix.lower()
            
            if extension in self._extensions:
                return self._extensions[extension]
            
            if len(file_path.suffixes) >= 2:
                double_ext = ''.join(file_path.suffixes[-2:]).lower()
                if double_ext in self._extensions:
                    return self._extensions[double_ext]
        
        if isinstance(source, str):
            source_stripped = source.strip()
            
            if source_stripped.startswith('{') or source_stripped.startswith('['):
                return 'json'
            
            if source_stripped.startswith('<'):
                return 'xml'
            
            if source_stripped.startswith('---') or source_stripped.startswith('- '):
                return 'yaml'
            
            if any(source_stripped.startswith(prefix) for prefix in ['# ', '## ', '### ']):
                return 'markdown'
        
        return 'text'
    
    def _create_config(self, loader_name: str, **config_kwargs) -> LoaderConfig:
        try:
            if config_kwargs:
                return LoaderConfigFactory.create_config(loader_name, **config_kwargs)
            else:
                return LoaderConfigFactory.create_config(loader_name)
        except Exception as e:
            self._logger.warning(f"Failed to create config for '{loader_name}': {e}")
            return LoaderConfig(**config_kwargs)
    
    def get_loader_for_file(self, file_path: str, **config_kwargs) -> DocumentLoader:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
        
        return self.get_loader(file_path, **config_kwargs)
    
    def get_loader_for_content(self, content: str, content_type: str, **config_kwargs) -> DocumentLoader:
        return self.get_loader(content, content_type, **config_kwargs)
    
    def get_loaders_for_directory(
        self, 
        directory_path: str, 
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        **config_kwargs
    ) -> Dict[str, DocumentLoader]:
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        loaders = {}
        directory = Path(directory_path)
        
        if recursive:
            search_pattern = "**/*"
        else:
            search_pattern = "*"
        
        all_files = list(directory.glob(search_pattern))
        
        for file_path in all_files:
            if not file_path.is_file():
                continue
            
            file_path_str = str(file_path)
            
            if file_patterns:
                matches_pattern = False
                for pattern in file_patterns:
                    if file_path.match(pattern):
                        matches_pattern = True
                        break
                if not matches_pattern:
                    continue
            
            if exclude_patterns:
                should_exclude = False
                for pattern in exclude_patterns:
                    if file_path.match(pattern):
                        should_exclude = True
                        break
                if should_exclude:
                    continue
            
            try:
                loader = self.get_loader_for_file(file_path_str, **config_kwargs)
                loaders[file_path_str] = loader
            except Exception as e:
                self._logger.warning(f"Could not create loader for {file_path_str}: {e}")
                continue
        
        return loaders
    
    def get_supported_extensions(self) -> List[str]:
        return list(self._extensions.keys())
    
    def get_supported_loaders(self) -> List[str]:
        return list(self._loaders.keys())
    
    def can_handle(self, source: str) -> bool:
        try:
            loader_type = self._detect_loader_type(source)
            return loader_type in self._loaders
        except Exception:
            return False
    
    def get_loader_info(self, loader_type: str) -> Dict[str, Any]:
        if loader_type not in self._loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        loader_class = self._loaders[loader_type]
        
        info = {
            'name': loader_type,
            'class': loader_class.__name__,
            'extensions': [ext for ext, name in self._extensions.items() if name == loader_type],
            'description': getattr(loader_class, '__doc__', 'No description available'),
            'has_config': loader_type in self._configs
        }
        
        return info
    
    def list_loaders(self) -> List[Dict[str, Any]]:
        return [self.get_loader_info(loader_type) for loader_type in self._loaders.keys()]
    
    def clear_cache(self):
        pass


_global_factory: Optional[LoaderFactory] = None


def get_factory() -> LoaderFactory:
    global _global_factory
    if _global_factory is None:
        _global_factory = LoaderFactory()
    return _global_factory


def create_loader(source: str, loader_type: Optional[str] = None, **config_kwargs) -> DocumentLoader:
    return get_factory().get_loader(source, loader_type, **config_kwargs)


def create_loader_for_file(file_path: str, **config_kwargs) -> DocumentLoader:
    return get_factory().get_loader_for_file(file_path, **config_kwargs)


def create_loader_for_content(content: str, content_type: str, **config_kwargs) -> DocumentLoader:
    return get_factory().get_loader_for_content(content, content_type, **config_kwargs)


def can_handle_file(file_path: str) -> bool:
    return get_factory().can_handle(file_path)


def get_supported_extensions() -> List[str]:
    return get_factory().get_supported_extensions()


def get_supported_loaders() -> List[str]:
    return get_factory().get_supported_loaders()


def load_document(source: str, **config_kwargs) -> List[Any]:
    loader = create_loader(source, **config_kwargs)
    return loader.load(source)


def load_documents_batch(
    sources: List[str], 
    **config_kwargs
) -> Dict[str, List[Any]]:
    results = {}
    factory = get_factory()
    
    for source in sources:
        try:
            loader = factory.get_loader(source, **config_kwargs)
            documents = loader.load(source)
            results[source] = documents
        except Exception as e:
            factory._logger.error(f"Failed to load {source}: {e}")
            results[source] = []
    
    return results
