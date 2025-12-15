import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    yaml = None
    _YAML_AVAILABLE = False

try:
    import jq
    _JQ_AVAILABLE = True
except ImportError:
    jq = None
    _JQ_AVAILABLE = False


from upsonic.schemas.data_models import Document
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import YAMLLoaderConfig


class YAMLLoader(BaseLoader):
    """
    An advanced loader for YAML files with powerful data extraction capabilities.

    This loader uses jq-style queries to split YAML files into multiple Documents
    and to extract specific content and metadata. It can handle multi-document
    YAML files and offers multiple ways to serialize the output content.
    """

    def __init__(self, config: Optional[YAMLLoaderConfig] = None):
        """
        Initializes the YAMLLoader with its specific configuration.

        Args:
            config: A YAMLLoaderConfig object with settings for YAML processing.
        """
        if config is None:
            config = YAMLLoaderConfig()
        if not _YAML_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pyyaml",
                install_command='pip install "upsonic[loaders]"',
                feature_name="YAML loader"
            )
        if not _JQ_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="jq",
                install_command='pip install "upsonic[loaders]"',
                feature_name="YAML loader"
            )
        super().__init__(config)
        self.config: YAMLLoaderConfig = config


    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Gets a list of file extensions supported by this loader."""
        return [".yaml", ".yml"]

    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads all YAML documents from the given source synchronously."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.aload(source))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.aload(source))

    async def aload(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads all YAML documents from the given source asynchronously and concurrently."""
        file_paths = self._resolve_sources(source)
        if not file_paths:
            return []

        tasks = [self._process_single_yaml_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks)

        return [doc for doc_list in results for doc in doc_list]

    def batch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Loads documents from a list of sources, leveraging the core `load` method."""
        return self.load(sources)

    async def abatch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Loads documents from a list of sources asynchronously, leveraging `aload`."""
        return await self.aload(sources)


    async def _process_single_yaml_file(self, path: Path) -> List[Document]:
        """
        Processes a single YAML file. Wraps the synchronous parsing logic in a
        separate thread to avoid blocking the asyncio event loop.
        """
        try:
            document_id = self._generate_document_id(path)
            if document_id in self._processed_document_ids:
                raise FileExistsError(
                    f"Source file '{path.resolve()}' has already been processed by this loader instance."
                )
            self._processed_document_ids.add(document_id)

            content = await self._read_file_content(path)
            return await asyncio.to_thread(self._parse_and_extract, content, path, document_id)
        except Exception as e:
            return self._handle_loading_error(str(path), e)
            
    async def _read_file_content(self, path: Path) -> str:
        return await asyncio.to_thread(path.read_text, self.config.encoding or 'utf-8')

    def _parse_and_extract(self, content: str, path: Path, document_id: str) -> List[Document]:
        """
        Synchronous helper that performs the actual parsing and document creation.
        """
        documents = []
        
        parsed_docs = yaml.safe_load_all(content) if self.config.handle_multiple_docs else [yaml.safe_load(content)]
        
        for doc_data in parsed_docs:
            if doc_data is None:
                continue

            try:
                data_chunks = jq.all(self.config.split_by_jq_query, doc_data)
            except Exception as e:
                raise ValueError(f"Invalid jq query '{self.config.split_by_jq_query}': {e}") from e

            for chunk in data_chunks:
                if self.config.content_synthesis_mode == "canonical_yaml":
                    doc_content = yaml.dump(chunk, indent=self.config.yaml_indent, sort_keys=False)
                elif self.config.content_synthesis_mode == "json":
                    doc_content = json.dumps(chunk, indent=self.config.json_indent)
                else: # "smart_text"
                    doc_content = " ".join(self._extract_smart_text(chunk))
                
                if self.config.skip_empty_content and not doc_content.strip():
                    continue

                metadata = self._create_metadata(path)
                
                if self.config.flatten_metadata and isinstance(chunk, dict):
                    metadata.update(self._flatten_dict(chunk))

                if self.config.metadata_jq_queries and isinstance(chunk, (dict, list)):
                    for key, query in self.config.metadata_jq_queries.items():
                        try:
                            result = jq.first(query, chunk)
                            if result is not None:
                                metadata[key] = result
                        except Exception:
                            pass
                if self.config.include_metadata:
                    documents.append(Document(document_id=document_id, content=doc_content, metadata=metadata))
                else:
                    documents.append(Document(document_id=document_id, content=doc_content))

        return documents


    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flattens a nested dictionary.
        """
        items = []
        for k, v in data.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _extract_smart_text(self, data: Any) -> Generator[str, None, None]:
        """
        Recursively extracts all string values from a nested data structure.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str):
                    yield key
                yield from self._extract_smart_text(value)
        elif isinstance(data, list):
            for item in data:
                yield from self._extract_smart_text(item)
        elif isinstance(data, str):
            yield data