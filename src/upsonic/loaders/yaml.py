from __future__ import annotations
from typing import List, Any, Dict, Union, Literal, Optional
import os
import io
from datetime import datetime
import time

from .base import DocumentLoader
from .config import YAMLLoaderConfig
from ..schemas.data_models import Document

try:
    import yaml
    import json
except ImportError:
    raise ImportError(
        "PyYAML is not installed. Please install it to use the YAMLLoader by running: "
        "'pip install PyYAML'"
    )

class YAMLLoader(DocumentLoader):
    """
    A master-class, intelligent ingestion engine for YAML data.

    This loader is a comprehensive toolkit for handling structured YAML data from
    multiple sources and formats. It is built on four pillars of functionality:

    1.  **Multi-Modal Input:** Can load from a file path, a raw YAML string, or an
        in-memory Python object.
    2.  **Multi-Document Mastery:** Correctly handles YAML files containing multiple
        documents (separated by '---'), treating each as a distinct Document.
    3.  **Intelligent Content Synthesis:** Offers multiple strategies for serializing
        the parsed data into a textual representation for the LLM.
    4.  **Sophisticated Metadata Flattening:** Can automatically flatten the entire
        YAML structure into a single-level metadata dictionary for easy filtering.
    """
    def __init__(
        self,
        config: Optional[YAMLLoaderConfig] = None,
    ):
        """
        Initializes the YAMLLoader.

        Args:
            config: Configuration object for the loader. If None, a default
                    YAMLLoaderConfig will be created.
        """
        if config is None:
            config = YAMLLoaderConfig()
        super().__init__(config)
        
        self.mode = config.content_synthesis_mode
        self.flatten_metadata = config.flatten_metadata
        self.yaml_indent = config.yaml_indent

    def load(self, source: str) -> List[Document]:
        """
        Loads YAML data from a file path or raw string.
        
        Args:
            source: File path or raw YAML string
            
        Returns:
            List of Document objects
        """
        return self._load_with_error_handling(source)

    def _load_with_error_handling(self, source: str) -> List[Document]:
        """Load with error handling based on configuration."""
        start_time = time.time()
        
        try:
            if not self._validate_source(source):
                raise ValueError(f"Invalid source: {source}")
            
            if self.config and self.config.max_file_size:
                if os.path.exists(source):
                    file_size = os.path.getsize(source)
                    if file_size > self.config.max_file_size:
                        raise ValueError(f"File size {file_size} exceeds limit {self.config.max_file_size}")
            
            if os.path.isfile(source):
                documents = self._load_from_file(source)
            else:
                documents = self._process_yaml_content(source, base_metadata={})
            
            documents = self._post_process_documents(documents, source)
            
            processing_time = time.time() - start_time
            self._update_stats(len(documents), processing_time, success=True)
            
            return documents
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(0, processing_time, success=False)
            
            if self.config:
                if self.config.error_handling == "ignore":
                    return []
                elif self.config.error_handling == "warn":
                    print(f"Warning: [YAMLLoader] Failed to load {source}: {e}")
                    return []
                else:
                    raise
            else:
                raise

    def _load_from_file(self, file_path: str) -> List[Document]:
        """Loads and processes a YAML file from the given path."""
        file_path = os.path.abspath(file_path)
        stats = os.stat(file_path)
        base_metadata: Dict[str, Any] = {
            "source": file_path, "file_name": os.path.basename(file_path),
            "file_path": file_path, "file_size": stats.st_size,
            "creation_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "last_modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        }
        
        if self.config and self.config.custom_metadata:
            base_metadata.update(self.config.custom_metadata)
            
        with open(file_path, "r", encoding="utf-8") as f:
            content_string = f.read()
        return self._process_yaml_content(content_string, base_metadata)

    def _process_yaml_content(self, content_string: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """The core processing engine for a raw YAML string."""
        documents = []
        try:
            doc_iterator = yaml.safe_load_all(content_string)
            for i, yaml_doc in enumerate(doc_iterator):
                if yaml_doc is None: continue

                doc_metadata = base_metadata.copy()
                doc_metadata["document_index"] = i
                
                docs = self._process_yaml_object(yaml_doc, base_metadata=doc_metadata)
                documents.extend(docs)
            return documents
        except yaml.YAMLError as e:
            source_info = base_metadata.get("source", "raw content")
            if self.config and self.config.error_handling == "ignore":
                return []
            elif self.config and self.config.error_handling == "warn":
                print(f"Warning: [YAMLLoader] Failed to parse malformed YAML from '{source_info}': {e}")
                return []
            else:
                raise

    def _process_yaml_object(self, yaml_data: Any, base_metadata: Dict[str, Any]) -> List[Document]:
        """Processes a single, parsed YAML document object."""
        
        content = ""
        if self.mode == "canonical_yaml":
            content = yaml.dump(
                yaml_data,
                indent=self.yaml_indent,
                sort_keys=False,
                default_flow_style=False
            )
        elif self.mode == "json":
            content = json.dumps(yaml_data, indent=self.yaml_indent)

        metadata = base_metadata.copy()
        if self.flatten_metadata and isinstance(yaml_data, dict):
            flattened_data = self._flatten_dict(yaml_data)
            metadata.update(flattened_data)
        else:
            metadata["raw_data"] = yaml_data

        return [Document(content=content, metadata=metadata)]

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Recursively flattens a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                         items.extend(self._flatten_dict(item, list_key, sep=sep).items())
                    else:
                        items.append((list_key, item))
            else:
                items.append((new_key, v))
        return dict(items)

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of file extensions supported by this loader."""
        return [".yaml", ".yml"]