import asyncio
from pathlib import Path
from typing import List, Optional, Union

try:
    from lxml import etree
    _LXML_AVAILABLE = True
except ImportError:
    etree = None
    _LXML_AVAILABLE = False


from upsonic.schemas.data_models import Document
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import XMLLoaderConfig


class XMLLoader(BaseLoader):
    """
    An advanced, high-performance loader for XML files.

    This loader uses XPath expressions to split a single XML file into multiple
    Documents and to extract specific content and metadata. It is built on top
    of the lxml library for robust and efficient parsing.
    """

    def __init__(self, config: Optional[XMLLoaderConfig] = None):
        """
        Initializes the XMLLoader with its specific configuration.

        Args:
            config: An XMLLoaderConfig object with settings for XML processing.
        """
        if config is None:
            config = XMLLoaderConfig()
        if not _LXML_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="lxml",
                install_command='pip install "upsonic[loaders]"',
                feature_name="XML loader"
            )
        super().__init__(config)
        self.config: XMLLoaderConfig = config


    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Gets a list of file extensions supported by this loader."""
        return [".xml"]

    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads all XML documents from the given source synchronously."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.aload(source))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.aload(source))

    async def aload(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads all XML documents from the given source asynchronously and concurrently."""
        file_paths = self._resolve_sources(source)
        if not file_paths:
            return []

        tasks = [self._process_single_xml_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks)

        return [doc for doc_list in results for doc in doc_list]

    def batch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Loads documents from a list of sources, leveraging the core `load` method."""
        return self.load(sources)

    async def abatch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Loads documents from a list of sources asynchronously, leveraging `aload`."""
        return await self.aload(sources)


    async def _process_single_xml_file(self, path: Path) -> List[Document]:
        """
        Processes a single XML file. Wraps the synchronous parsing logic in a
        separate thread to avoid blocking the asyncio event loop.
        """
        try:
            return await asyncio.to_thread(self._parse_and_extract, path)
        except FileExistsError:
            raise
        except Exception as e:
            return self._handle_loading_error(str(path), e)

    def _parse_and_extract(self, path: Path) -> List[Document]:
        """
        Synchronous helper that performs the actual parsing and document creation.
        """
        document_id = self._generate_document_id(path)
        if document_id in self._processed_document_ids:
            raise FileExistsError(
                f"Source file '{path.resolve()}' has already been processed by this loader instance."
            )
        self._processed_document_ids.add(document_id)

        if not self._check_file_size(path):
            return []

        parser = etree.XMLParser(recover=self.config.recover_mode)
        tree = etree.parse(str(path), parser)

        if self.config.strip_namespaces:
            self._strip_namespaces(tree.getroot())

        elements_to_process = tree.xpath(self.config.split_by_xpath)

        documents = []
        all_element_contents = []
        combined_metadata = self._create_metadata(path)
        
        for element in elements_to_process:
            if self.config.content_xpath:
                content_elements = element.xpath(self.config.content_xpath)
                content_element = content_elements[0] if content_elements else None
            else:
                content_element = element

            if content_element is None:
                continue

            if self.config.content_synthesis_mode == "smart_text":
                content = " ".join(content_element.xpath(".//text()")).strip()
                content = " ".join(content.split())
            else: # "xml_snippet"
                content = etree.tostring(content_element, pretty_print=True).decode("utf-8")

            if self.config.skip_empty_content and not content.strip():
                continue

            all_element_contents.append(content)

            # Collect metadata from all elements
            if self.config.include_attributes:
                combined_metadata.update(dict(element.attrib))
            
            if self.config.metadata_xpaths:
                for key, xpath in self.config.metadata_xpaths.items():
                    result = element.xpath(xpath)
                    if result:
                        value = result[0]
                        if hasattr(value, 'text'):
                            combined_metadata[key] = value.text.strip() if value.text else ""
                        else:
                            combined_metadata[key] = str(value).strip()
        
        if all_element_contents:
            combined_content = "\n\n".join(all_element_contents)
            if self.config.include_metadata:
                combined_metadata["element_count"] = len(all_element_contents)
                documents.append(Document(document_id=document_id, content=combined_content, metadata=combined_metadata))
            else:
                documents.append(Document(document_id=document_id, content=combined_content))
            
        return documents



    @staticmethod
    def _strip_namespaces(root_element: "etree._Element"):
        """
        Recursively removes namespace information from all elements in the tree.
        """
        for elem in root_element.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]
        
        etree.cleanup_namespaces(root_element.getroottree())