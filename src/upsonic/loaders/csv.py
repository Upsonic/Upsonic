from __future__ import annotations
from typing import List, Optional, Dict, Any
import os
import csv
from datetime import datetime
import json

from .base import DocumentLoader
from .config import CSVLoaderConfig
from ..schemas.data_models import Document

try:
    import chardet
except ImportError:
    chardet = None


class CSVLoader(DocumentLoader):
    """
    A row-aware loader for Comma-Separated Values (`.csv`) files.

    This loader embodies the "row-as-document" philosophy, transforming each row
    in a tabular dataset into a distinct, metadata-rich Document object. It is
    highly configurable, allowing for selective column loading and different
    content synthesis strategies.

    Its core features include:
    - Treating each row as an independent Document.
    - Using column headers as metadata keys for each Document.
    - Configurable content synthesis (human-readable string or JSON).
    - Selective inclusion/exclusion of columns.
    - Rich metadata enrichment, including source file details and row number.
    - Robust error handling for malformed rows and encoding issues.
    - Configuration through CSVLoaderConfig.
    """
    
    def __init__(self, config: Optional[CSVLoaderConfig] = None):
        """
        Initializes the CSVLoader with configuration.

        Args:
            config: CSVLoaderConfig object with all configuration options
        """
        super().__init__(config)
        self.config = config or CSVLoaderConfig()

    def load(self, source: str) -> List[Document]:
        """
        Loads a CSV file, transforming each row into a Document object.
        """
        import time
        start_time = time.time()
        documents: List[Document] = []
        
        try:
            file_path = os.path.abspath(source)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Source path '{source}' is not a valid file.")

            stats = os.stat(file_path)
            base_metadata: Dict[str, Any] = {
                "source": source, "file_name": os.path.basename(file_path),
                "file_path": file_path, "file_size": stats.st_size,
                "creation_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "last_modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            }

            if self.config.custom_metadata:
                base_metadata.update(self.config.custom_metadata)

            encoding = self.config.encoding or "utf-8"
            try:
                with open(file_path, "r", newline="", encoding=encoding) as f:
                    file_content = f.read()
                base_metadata["detected_encoding"] = encoding
            except UnicodeDecodeError:
                if not chardet:
                    error_msg = f"Error: `chardet` not installed. Cannot detect encoding for '{source}'."
                    return self._handle_error(error_msg, ImportError(error_msg))
                
                with open(file_path, "rb") as f_raw:
                    raw_data = f_raw.read()
                detection_result = chardet.detect(raw_data)
                encoding = detection_result.get("encoding", "utf-8")
                base_metadata["detected_encoding"] = encoding
                file_content = raw_data.decode(encoding)
            
            csv_reader = csv.reader(
                file_content.splitlines(),
                delimiter=self.config.delimiter,
                quotechar=self.config.quotechar
            )
            
            if self.config.has_header:
                header = next(csv_reader)
            else:
                first_row = next(csv_reader)
                header = [f"column_{i}" for i in range(len(first_row))]
                csv_reader = csv.reader(
                    file_content.splitlines(),
                    delimiter=self.config.delimiter,
                    quotechar=self.config.quotechar
                )
            
            final_header = header
            if self.config.include_columns:
                final_header = [h for h in header if h in self.config.include_columns]
            elif self.config.exclude_columns:
                final_header = [h for h in header if h not in self.config.exclude_columns]
            
            col_indices = [header.index(h) for h in final_header]

            for i, row in enumerate(csv_reader):
                    
                row_number = i + 1
                if len(row) != len(header):
                    warning_msg = f"Warning: Skipping malformed row {row_number} in '{source}'. Expected {len(header)} columns, found {len(row)}."
                    if self.config.error_handling in ["warn", "raise"]:
                        print(warning_msg)
                    if self.config.error_handling == "raise":
                        raise ValueError(warning_msg)
                    continue

                filtered_row_values = [row[j] for j in col_indices]
                row_data = dict(zip(final_header, filtered_row_values))

                if self.config.skip_empty_content and all(not str(v).strip() for v in row_data.values()):
                    continue

                content = ""
                if self.config.content_synthesis_mode == "concatenated":
                    content = ", ".join([f"{k}: {v}" for k, v in row_data.items()])
                elif self.config.content_synthesis_mode == "json":
                    content = json.dumps(row_data)

                row_metadata = base_metadata.copy()
                if self.config.row_as_document:
                    row_metadata.update(row_data)
                row_metadata["row_number"] = row_number

                documents.append(
                    Document(content=content, metadata=row_metadata)
                )

        except FileNotFoundError as e:
            error_msg = f"Error: [CSVLoader] File not found at path: {e}"
            if self.config and self.config.error_handling == "ignore":
                return []
            elif self.config and self.config.error_handling == "warn":
                print(error_msg)
                return []
            else:
                raise e
        except Exception as e:
            error_msg = f"Error: [CSVLoader] An unexpected error occurred while loading '{source}': {e}"
            return self._handle_error(error_msg, e)
        
        if hasattr(self, '_update_stats'):
            processing_time = time.time() - start_time
            self._update_stats(len(documents), processing_time, success=True)
            
        return documents

    def _handle_error(self, message: str, exception: Exception) -> List[Document]:
        """Handle errors based on configuration."""
        if self.config.error_handling == "ignore":
            return []
        elif self.config.error_handling == "warn":
            print(message)
            return []
        else:
            raise exception

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of file extensions supported by this loader."""
        return ['.csv']