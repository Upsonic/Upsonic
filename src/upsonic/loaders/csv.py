import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import aiofiles
    import aiofiles.os
    _AIOFILES_AVAILABLE = True
except ImportError:
    aiofiles = None
    _AIOFILES_AVAILABLE = False


from upsonic.schemas.data_models import Document
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import CSVLoaderConfig


class CSVLoader(BaseLoader):
    """
    A loader for CSV (Comma-Separated Values) files.

    Each row in the CSV file is treated as a separate Document. The content of the
    document is synthesized from the row's columns based on the configuration.
    """

    def __init__(self, config: Optional[CSVLoaderConfig] = None):
        """Initializes the CSVLoader with its specific configuration."""
        if config is None:
            config = CSVLoaderConfig()
        if not _AIOFILES_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="aiofiles",
                install_command='pip install "upsonic[loaders]"',
                feature_name="CSV loader"
            )
        super().__init__(config)
        self.config: CSVLoaderConfig = config

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Gets the list of supported file extensions."""
        return [".csv"]

    def _filter_row_columns(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Filters columns in a row based on include/exclude rules in the config."""
        if self.config.include_columns:
            return {k: v for k, v in row.items() if k in self.config.include_columns}
        if self.config.exclude_columns:
            return {k: v for k, v in row.items() if k not in self.config.exclude_columns}
        return row

    def _synthesize_content(self, row: Dict[str, Any]) -> str:
        """Creates the document content from a row based on the synthesis mode."""
        filtered_row = self._filter_row_columns(row)
        if self.config.content_synthesis_mode == "json":
            return json.dumps(filtered_row)
        
        content_parts = []
        for k, v in filtered_row.items():
            if v and str(v).strip():
                content_parts.append(f"{k}: {v}")
        
        return "\n".join(content_parts)

    def _create_documents_from_rows(self, all_rows: List[str], file_path: Path, document_id: str) -> List[Document]:
        """Creates documents from processed rows based on split_mode configuration."""
        documents = []
        
        if self.config.split_mode == "per_row":
            # Each row becomes a separate document
            for i, row_content in enumerate(all_rows):
                if self.config.include_metadata:
                    metadata = self._create_metadata(file_path)
                    metadata["row_index"] = i
                    metadata["total_rows"] = len(all_rows)
                row_doc_id = f"{document_id}_row_{i}"
                if metadata:
                    doc = Document(document_id=row_doc_id, content=row_content, metadata=metadata)
                else:
                    doc = Document(document_id=row_doc_id, content=row_content)
                documents.append(doc)
        elif self.config.split_mode == "per_chunk":
            # Group rows into chunks
            for chunk_idx in range(0, len(all_rows), self.config.rows_per_chunk):
                chunk_rows = all_rows[chunk_idx:chunk_idx + self.config.rows_per_chunk]
                chunk_content = "\n\n".join(chunk_rows)
                if self.config.include_metadata:
                    metadata = self._create_metadata(file_path)
                    metadata["chunk_index"] = chunk_idx // self.config.rows_per_chunk
                    metadata["rows_in_chunk"] = len(chunk_rows)
                    metadata["total_rows"] = len(all_rows)
                chunk_doc_id = f"{document_id}_chunk_{chunk_idx // self.config.rows_per_chunk}"
                if metadata:
                    doc = Document(document_id=chunk_doc_id, content=chunk_content, metadata=metadata)
                else:
                    doc = Document(document_id=chunk_doc_id, content=chunk_content)
                documents.append(doc)
        else:
            # Original behavior: single document
            combined_content = "\n\n".join(all_rows)
            if self.config.include_metadata:
                metadata = self._create_metadata(file_path)
                metadata["row_count"] = len(all_rows)
            if metadata:
                doc = Document(document_id=document_id, content=combined_content, metadata=metadata)
            else:
                doc = Document(document_id=document_id, content=combined_content)
            documents.append(doc)
            
        return documents

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Helper method to load documents from a single CSV file."""
        documents = []
        try:
            document_id = self._generate_document_id(file_path)
            if document_id in self._processed_document_ids:
                raise FileExistsError(
                    f"Source file '{file_path.resolve()}' has already been processed."
                )
            self._processed_document_ids.add(document_id)

            with open(file_path, mode="r", encoding=self.config.encoding or "utf-8", newline="") as f:
                if self.config.has_header:
                    reader = csv.DictReader(
                        f,
                        delimiter=self.config.delimiter,
                        quotechar=self.config.quotechar,
                    )
                else:
                    standard_reader = csv.reader(
                        f,
                        delimiter=self.config.delimiter,
                        quotechar=self.config.quotechar,
                    )
                    reader = (
                        {f"column_{i}": val for i, val in enumerate(row)}
                        for row in standard_reader
                    )

                all_rows = []
                for i, row in enumerate(reader):
                    content = self._synthesize_content(row)
                    if self.config.skip_empty_content and not content.strip():
                        continue
                    all_rows.append(content)
                
                if all_rows:
                    documents.extend(self._create_documents_from_rows(all_rows, file_path, document_id))
        except Exception as e:
            docs_from_error = self._handle_loading_error(str(file_path), e)
            documents.extend(docs_from_error)
        
        return documents

    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        Loads documents from the given CSV source(s) synchronously.
        """
        files_to_process = self._resolve_sources(source)
        all_documents = []
        for file_path in files_to_process:
            if not self._check_file_size(file_path):
                continue

            all_documents.extend(self._load_single_file(file_path))
        return all_documents

    async def _aload_single_file(self, file_path: Path) -> List[Document]:
        """Async helper to load documents from a single CSV file."""
        documents = []
        try:
            document_id = self._generate_document_id(file_path)
            if document_id in self._processed_document_ids:
                raise FileExistsError(
                    f"Source file '{file_path.resolve()}' has already been processed."
                )
            self._processed_document_ids.add(document_id)

            if not self._check_file_size(file_path):
                return []

            async with aiofiles.open(file_path, mode="r", encoding=self.config.encoding or "utf-8", newline="") as f:
                content_str = await f.read()
                file_lines = content_str.splitlines()

                if self.config.has_header:
                    reader = csv.DictReader(
                        file_lines,
                        delimiter=self.config.delimiter,
                        quotechar=self.config.quotechar,
                    )
                else:
                    standard_reader = csv.reader(
                        file_lines,
                        delimiter=self.config.delimiter,
                        quotechar=self.config.quotechar,
                    )
                    reader = (
                        {f"column_{i}": val for i, val in enumerate(row)}
                        for row in standard_reader
                    )
                
                all_rows = []
                for i, row in enumerate(reader):
                    content = self._synthesize_content(row)
                    if self.config.skip_empty_content and not content.strip():
                        continue
                    all_rows.append(content)
                
                if all_rows:
                    documents.extend(self._create_documents_from_rows(all_rows, file_path, document_id))
        except Exception as e:
            docs_from_error = self._handle_loading_error(str(file_path), e)
            documents.extend(docs_from_error)

        return documents

    async def aload(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        Loads documents from the given CSV source(s) asynchronously.
        """
        files_to_process = self._resolve_sources(source)
        tasks = [self._aload_single_file(file_path) for file_path in files_to_process]
        results = await asyncio.gather(*tasks)
        return [doc for sublist in results for doc in sublist]

    def batch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """A simple synchronous batch load implementation."""
        return self.load(sources)

    async def abatch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """An efficient asynchronous batch load implementation using asyncio.gather."""
        return await self.aload(sources)