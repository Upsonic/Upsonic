import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import jq
    _JQ_AVAILABLE = True
except ImportError:
    jq = None
    _JQ_AVAILABLE = False


from upsonic.schemas.data_models import Document
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import JSONLoaderConfig


class JSONLoader(BaseLoader):
    """
    A highly configurable loader for JSON and JSON Lines (.jsonl) files.

    This loader uses JQ queries to extract records, map content, and generate
    metadata from complex JSON structures, allowing it to adapt to any format.
    """

    def __init__(self, config: Optional[JSONLoaderConfig] = None):
        """Initializes the JSONLoader with its specific configuration."""
        if config is None:
            config = JSONLoaderConfig()
        if not _JQ_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="jq",
                install_command='pip install "upsonic[loaders]"',
                feature_name="JSON loader"
            )
        super().__init__(config)
        self.config: JSONLoaderConfig = config
        if self.config.mode == "multi" and not self.config.record_selector:
            raise ValueError("`record_selector` must be provided when using 'multi' mode.")

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Gets the list of supported file extensions."""
        return [".json", ".jsonl"]

    def _execute_jq_query(self, query: str, data: Any) -> Any:
        """Compiles and executes a JQ query on the given data."""
        try:
            return jq.compile(query).input(data).all()
        except Exception as e:
            raise ValueError(f"Error executing JQ query '{query}': {e}")


    def _create_document_from_record(self, record: Any, base_metadata: Dict[str, Any], document_id: str) -> Document:
        """Creates a single Document from a JSON record using config mappers."""
        content_data = self._execute_jq_query(self.config.content_mapper, record)
        if content_data:
            content_data = content_data[0]

        if self.config.content_synthesis_mode == "text":
            content = str(content_data) if not isinstance(content_data, (dict, list)) else json.dumps(content_data)
        else: # "json"
            content = json.dumps(content_data)

        extracted_metadata = {}
        if self.config.metadata_mapper:
            for key, query in self.config.metadata_mapper.items():
                meta_value = self._execute_jq_query(query, record)
                if meta_value:
                    extracted_metadata[key] = meta_value[0] # Take the first result

        final_metadata = {**base_metadata, **extracted_metadata}
        if self.config.include_metadata:
            return Document(document_id=document_id, content=content, metadata=final_metadata)
        else:
            return Document(document_id=document_id, content=content)

    def _process_json_object(self, data: Any, file_path: Path, document_id: str) -> List[Document]:
        """Processes a parsed JSON object based on the configured mode."""
        base_metadata = self._create_metadata(file_path)
        
        if self.config.mode == "single":
            return [self._create_document_from_record(data, base_metadata, document_id)]
        
        records = self._execute_jq_query(self.config.record_selector, data)
        documents = []
        for record in records:
            doc = self._create_document_from_record(record, base_metadata, document_id)
            if not (self.config.skip_empty_content and not doc.content.strip()):
                documents.append(doc)
        return documents

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Loads and processes a single JSON or JSONL file."""
        if not self._check_file_size(file_path):
            return []
        
        try:
            document_id = self._generate_document_id(file_path)
            if document_id in self._processed_document_ids:
                raise FileExistsError(
                    f"Source file '{file_path.resolve()}' has already been processed by this loader instance."
                )
            self._processed_document_ids.add(document_id)

            with open(file_path, "r", encoding=self.config.encoding or "utf-8") as f:
                if self.config.json_lines:
                    all_documents = []
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            data = json.loads(line)
                            line_document_id = f"{document_id}_line_{line_num}"
                            all_documents.extend(self._process_json_object(data, file_path, line_document_id))
                    return all_documents
                else:
                    data = json.load(f)
                    return self._process_json_object(data, file_path, document_id)
        except Exception as e:
            return self._handle_loading_error(str(file_path), e)

    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads documents from the given JSON source(s) synchronously."""
        files_to_process = self._resolve_sources(source)
        all_documents = []
        for file_path in files_to_process:
            all_documents.extend(self._load_single_file(file_path))
        return all_documents

    async def _aload_single_file(self, file_path: Path) -> List[Document]:
        """Async: Loads and processes a single JSON or JSONL file."""
        return await asyncio.to_thread(self._load_single_file, file_path)

    async def aload(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads documents from the given JSON source(s) asynchronously."""
        files_to_process = await asyncio.to_thread(self._resolve_sources, source)
        tasks = [self._aload_single_file(file) for file in files_to_process]
        results = await asyncio.gather(*tasks)
        return [doc for sublist in results for doc in sublist]

    def batch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """A simple synchronous batch load implementation."""
        return self.load(sources)

    async def abatch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """An efficient asynchronous batch load implementation."""
        return await self.aload(sources)