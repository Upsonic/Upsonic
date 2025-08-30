from __future__ import annotations
from typing import List, Any, Dict, Union, Literal, Iterator, Optional
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

from .base import DocumentLoader
from .config import XMLLoaderConfig
from ..schemas.data_models import Document

try:
    from lxml import etree
except ImportError:
    raise ImportError(
        "lxml is not installed. It is required for the XMLLoader. "
        "Please run: 'pip install lxml'"
    )


class XMLLoader(DocumentLoader):
    """
    A master-class, ingestion engine for XML data.

    This loader is a toolkit for handling structured XML data,
    built on four pillars of functionality:

    1.  **Multi-Modal Input:** Can load from a file path or a raw XML string.
    2.  **XPath-Powered Splitting:** Can deconstruct a single XML file into multiple
        Documents based on a repeating element specified by an XPath expression.
    3.  **Intelligent Content Synthesis:** Offers multiple strategies for creating the
        Document's content, from clean text extraction to preserving XML structure.
    4.  **Namespace Awareness:** Can gracefully handle or strip XML namespaces and
        flattens elements and attributes into rich, filterable metadata.
    """
    
    def __init__(self, config: Optional[XMLLoaderConfig] = None):
        """
        Initializes the XMLLoader.

        Args:
            config: XMLLoaderConfig object containing all configuration options.
                   If None, uses default configuration.
        """
        super().__init__(config)
        
        if config is None:
            config = XMLLoaderConfig()
        
        self.split_by_xpath = config.split_by_xpath
        self.content_synthesis_mode = config.content_synthesis_mode
        self.strip_namespaces = config.strip_namespaces
        self.include_attributes = config.include_attributes

    def load(self, source: str) -> List[Document]:
        """
        Loads XML data from a file path or a raw string/bytes object.
        
        Args:
            source: File path or XML string content
            
        Returns:
            List of Document objects
        """
        try:
            if os.path.isfile(source):
                return self._load_from_file(source)
            elif isinstance(source, str) and source.strip().startswith("<"):
                return self._process_xml_content(source.encode("utf-8"), base_metadata={})
            else:
                raise ValueError("Source is not a valid file path or a valid XML string.")

        except Exception as e:
            if self.config and self.config.error_handling == "ignore":
                return []
            elif self.config and self.config.error_handling == "warn":
                print(f"Warning: [XMLLoader] Failed to load {source}: {e}")
                return []
            else:
                raise

    def _load_from_file(self, file_path: str) -> List[Document]:
        """Loads and processes an XML file from the given path."""
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
            
        with open(file_path, "rb") as f:
            content_bytes = f.read()
        return self._process_xml_content(content_bytes, base_metadata)

    def _process_xml_content(self, content_bytes: bytes, base_metadata: Dict[str, Any]) -> List[Document]:
        """The core processing engine for a parsed XML object."""
        try:
            parser = etree.XMLParser(recover=True, no_network=True)
            tree = etree.parse(BytesIO(content_bytes), parser)
            
            if self.strip_namespaces:
                for elem in tree.getiterator():
                    if '}' in elem.tag:
                        elem.tag = elem.tag.split('}', 1)[1]
                etree.cleanup_namespaces(tree)
            
            root = tree.getroot()

            if self.split_by_xpath:
                elements_to_process = root.xpath(self.split_by_xpath)
            else:
                elements_to_process = [root]

            documents = []
            for i, element in enumerate(elements_to_process):
                metadata = base_metadata.copy()
                metadata["xpath_index"] = i

                content = ""
                if self.content_synthesis_mode == "xml_snippet":
                    content = etree.tostring(element, pretty_print=True, encoding="unicode")
                elif self.content_synthesis_mode == "smart_text":
                    content = "\n".join(text.strip() for text in element.itertext() if text.strip())

                if self.include_attributes:
                    element_metadata = self._flatten_element_to_dict(element)
                    metadata.update(element_metadata)

                if self.config and self.config.skip_empty_content:
                    if not content or not content.strip():
                        continue

                documents.append(Document(content=content, metadata=metadata))
            
            return documents
        
        except etree.XMLSyntaxError as e:
            source_info = base_metadata.get("source", "raw content")
            if self.config and self.config.error_handling == "ignore":
                return []
            elif self.config and self.config.error_handling == "warn":
                print(f"Warning: [XMLLoader] Failed to parse malformed XML from '{source_info}': {e}")
                return []
            else:
                raise

    def _flatten_element_to_dict(self, element: etree._Element, parent_key: str = '') -> Dict[str, Any]:
        """Recursively flattens an XML element into a single-level dictionary."""
        items = {}
        for key, value in element.attrib.items():
            items[f"{parent_key}{element.tag}.@{key}"] = value

        for child in element:
            child_key = f"{parent_key}{element.tag}.{child.tag}"
            if len(child) == 0 and child.text and child.text.strip():
                items[child_key] = child.text.strip()
            elif len(child) > 0:
                items.update(self._flatten_element_to_dict(child, f"{parent_key}{element.tag}."))
        
        if not list(element) and element.text and element.text.strip():
             items[f"{parent_key}{element.tag}"] = element.text.strip()

        return items

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of file extensions supported by this loader."""
        return [".xml", ".xhtml", ".rss", ".atom"]

    @classmethod
    def can_load(cls, source: str) -> bool:
        """Check if this loader can handle the given source."""
        if not source:
            return False
        
        if os.path.exists(source) and os.path.isfile(source):
            return Path(source).suffix.lower() in cls.get_supported_extensions()
        
        if isinstance(source, str) and source.strip().startswith("<"):
            return True
        
        return False