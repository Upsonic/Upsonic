from __future__ import annotations
from typing import List, Any, Dict, Optional, Tuple, Set
import os
import re
import asyncio
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .base import DocumentLoader
from .config import MarkdownLoaderConfig
from ..schemas.data_models import Document

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


class MarkdownLoader(DocumentLoader):
    """
    A structure-aware loader for Markdown (`.md`) files.

    This loader provides markdown parsing capabilities including:
    1. **YAML Front Matter Parsing**: Extracts and processes metadata from YAML front matter
    2. **Table Processing**: Converts markdown tables to structured text or preserves markdown format
    3. **Code Block Intelligence**: Extracts language information and optionally processes code blocks
    4. **Heading Hierarchy**: Extracts and preserves heading structure
    5. **Link and Image Processing**: Handles markdown links and images
    6. **List Processing**: Maintains list structure and formatting
    7. **Blockquote Handling**: Preserves quote formatting
    8. **Async Support**: Full async/await support for high-performance loading
    9. **Configuration-Driven**: Highly configurable through MarkdownLoaderConfig
    10. **Error Resilience**: Robust error handling with configurable strategies
    """

    def __init__(self, config: Optional[MarkdownLoaderConfig] = None):
        """
        Initialize the MarkdownLoader with optional configuration.
        
        Args:
            config: Configuration object for markdown processing options
        """
        super().__init__(config or MarkdownLoaderConfig())
        self.config = self.config

    def load(self, source: str) -> List[Document]:
        """
        Loads a Markdown file, parsing its structure into Document objects.
        
        Args:
            source: Path to the markdown file
            
        Returns:
            List of Document objects (typically one per file)
        """
        return self._load_with_error_handling(source)

    async def load_async(self, source: str) -> List[Document]:
        """
        Asynchronous version of load method.
        
        Args:
            source: Path to the markdown file
            
        Returns:
            List of Document objects
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.load, source)

    def _load_with_error_handling(self, source: str) -> List[Document]:
        """Load with comprehensive error handling and validation."""
        start_time = datetime.now()
        
        try:
            if not self._validate_source(source):
                raise ValueError(f"Invalid source: {source}")
            
            if self.config.max_file_size:
                file_size = os.path.getsize(source)
                if file_size > self.config.max_file_size:
                    raise ValueError(f"File size {file_size} exceeds limit {self.config.max_file_size}")
            
            documents = self._process_markdown_file(source)
            
            documents = self._post_process_documents(documents, source)
            
            return documents
            
        except Exception as e:
            if self.config.error_handling == "ignore":
                return []
            elif self.config.error_handling == "warn":
                print(f"Warning: Failed to load {source}: {e}")
                return []
            else:
                raise

    def _process_markdown_file(self, source: str) -> List[Document]:
        """Process a markdown file and extract structured content."""
        file_path = Path(source).resolve()
        
        with open(file_path, 'r', encoding=self.config.encoding or 'utf-8') as f:
            full_content = f.read()
        
        stats = file_path.stat()
        base_metadata = self._extract_file_metadata(file_path, stats)
        
        if self.config.parse_front_matter and YAML_AVAILABLE:
            front_matter, main_content = self._parse_front_matter(full_content)
            base_metadata.update(front_matter)
        else:
            main_content = full_content
        
        processed_content, extracted_metadata = self._process_markdown_content(main_content)
        base_metadata.update(extracted_metadata)
        
        document = Document(
            content=processed_content,
            metadata=base_metadata
        )
        
        return [document]

    def _extract_file_metadata(self, file_path: Path, stats: os.stat_result) -> Dict[str, Any]:
        """Extract comprehensive file metadata."""
        return {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size": stats.st_size,
            "creation_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "last_modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "file_extension": file_path.suffix.lower(),
            "loader_type": self.__class__.__name__,
        }

    def _parse_front_matter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse YAML front matter from markdown content."""
        front_matter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(front_matter_pattern, content, re.DOTALL)
        
        if match:
            yaml_content = match.group(1)
            try:
                metadata = yaml.safe_load(yaml_content) or {}
                remaining_content = content[match.end():]
                return metadata, remaining_content
            except yaml.YAMLError as e:
                return {}, content
        
        return {}, content

    def _process_markdown_content(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """Process markdown content with parsing."""
        metadata = {
            "code_languages": [],
            "headings": [],
            "links": [],
            "images": [],
            "tables": [],
            "lists": [],
            "blockquotes": []
        }
        
        lines = content.split('\n')
        processed_lines = []
        
        i = 0
        in_code_block = False
        
        while i < len(lines):
            line = lines[i]
            
            if self._is_code_block_start(line):
                if not in_code_block:
                    lang = line.strip()[3:].strip()
                    if lang:
                        metadata["code_languages"].append(lang)
                
                in_code_block = not in_code_block
                processed_lines.append(line)
                i += 1
                continue
            
            if in_code_block:
                processed_lines.append(line)
                i += 1
                continue
            
            if self._is_heading(line):
                processed_line, heading_info = self._process_heading(line)
                processed_lines.append(processed_line)
                metadata["headings"].append(heading_info)
            elif self._is_table_start(lines, i):
                table_lines, table_metadata = self._process_table(lines, i)
                processed_lines.extend(table_lines)
                metadata["tables"].append(table_metadata)
                original_table_length = 0
                j = i
                while j < len(lines) and '|' in lines[j]:
                    original_table_length += 1
                    j += 1
                i += original_table_length - 1
            elif self._is_list_item(line):
                processed_line, list_info = self._process_list_item(line)
                processed_lines.append(processed_line)
                metadata["lists"].append(list_info)
            elif self._is_blockquote(line):
                processed_line, quote_info = self._process_blockquote(line)
                processed_lines.append(processed_line)
                metadata["blockquotes"].append(quote_info)
            elif self._is_link_or_image(line):
                processed_line, link_info = self._process_link_or_image(line)
                processed_lines.append(processed_line)
                if link_info["type"] == "link":
                    metadata["links"].append(link_info)
                else:
                    metadata["images"].append(link_info)
            else:
                processed_lines.append(line)
            
            i += 1
        
        return '\n'.join(processed_lines), metadata

    def _is_heading(self, line: str) -> bool:
        """Check if line is a markdown heading."""
        return bool(re.match(r'^#{1,6}\s+', line))

    def _process_heading(self, line: str) -> Tuple[str, Dict[str, Any]]:
        """Process a markdown heading."""
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            
            heading_info = {
                "level": level,
                "text": text,
                "id": self._generate_heading_id(text)
            }
            
            return line, heading_info
        
        return line, {}

    def _generate_heading_id(self, text: str) -> str:
        """Generate an ID for a heading."""
        id_text = re.sub(r'[^\w\s-]', '', text.lower())
        id_text = re.sub(r'[-\s]+', '-', id_text)
        return id_text.strip('-')

    def _is_code_block_start(self, line: str) -> bool:
        """Check if line starts a code block."""
        return line.strip().startswith('```')

    def _process_code_block(self, lines: List[str], start_index: int) -> Tuple[List[str], Dict[str, Any]]:
        """Process a code block."""
        if not self.config.include_code_blocks:
            return [], {}
        
        start_line = lines[start_index]
        language = start_line.strip()[3:].strip()
        
        code_lines = [start_line]
        metadata = {"languages": [language] if language else []}
        
        i = start_index + 1
        while i < len(lines):
            line = lines[i]
            code_lines.append(line)
            
            if line.strip().startswith('```'):
                break
            
            i += 1
        
        return code_lines, metadata

    def _is_table_start(self, lines: List[str], index: int) -> bool:
        """Check if lines starting at index form a table."""
        if index >= len(lines) - 1:
            return False
        
        current_line = lines[index]
        next_line = lines[index + 1]
        
        if '|' not in current_line:
            return False
        
        separator_pattern = r'^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$'
        return bool(re.match(separator_pattern, next_line))

    def _process_table(self, lines: List[str], start_index: int) -> Tuple[List[str], Dict[str, Any]]:
        """Process a markdown table."""
        table_lines = []
        table_metadata = {"rows": 0, "columns": 0}
        
        i = start_index
        while i < len(lines):
            line = lines[i]
            
            if '|' in line:
                table_lines.append(line)
                table_metadata["rows"] += 1
                
                if table_metadata["rows"] == 1:
                    table_metadata["columns"] = len([cell for cell in line.split('|') if cell.strip()])
            else:
                break
            
            i += 1
        
        if self.config.table_format == "text":
            converted_table = self._convert_table_to_text(table_lines)
            return [converted_table], table_metadata
        elif self.config.table_format == "html":
            converted_table = self._convert_table_to_html(table_lines)
            return [converted_table], table_metadata
        else:
            return table_lines, table_metadata

    def _convert_table_to_text(self, table_lines: List[str]) -> str:
        """Convert markdown table to structured text."""
        if len(table_lines) < 2:
            return '\n'.join(table_lines)
        
        header_cells = table_lines[0].split('|')
        headers = []
        for cell in header_cells:
            cell = cell.strip()
            if cell:
                headers.append(cell)
        
        while headers and not headers[0]:
            headers.pop(0)
        while headers and not headers[-1]:
            headers.pop()
        
        data_rows = []
        for line in table_lines[2:]:
            cells = line.split('|')
            row_cells = []
            for cell in cells:
                cell = cell.strip()
                if cell:
                    row_cells.append(cell)
                else:
                    row_cells.append("")
            
            while row_cells and not row_cells[0]:
                row_cells.pop(0)
            while row_cells and not row_cells[-1]:
                row_cells.pop()
            
            if any(cell for cell in row_cells):
                data_rows.append(row_cells)
        
        text_lines = ["[Table Data]:"]
        for row in data_rows:
            row_items = []
            for i, cell in enumerate(row):
                if i < len(headers):
                    row_items.append(f"{headers[i]}: {cell}")
                else:
                    row_items.append(f"Column_{i+1}: {cell}")
            
            if row_items:
                text_lines.append(f"- {', '.join(row_items)}")
        
        return '\n'.join(text_lines)

    def _convert_table_to_html(self, table_lines: List[str]) -> str:
        """Convert markdown table to HTML."""
        if len(table_lines) < 2:
            return '\n'.join(table_lines)
        
        html_lines = ["<table>"]
        
        for i, line in enumerate(table_lines):
            if i == 1:
                continue
            
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                tag = "th" if i == 0 else "td"
                row_html = f"<tr>{''.join([f'<{tag}>{cell}</{tag}>' for cell in cells])}</tr>"
                html_lines.append(row_html)
        
        html_lines.append("</table>")
        return '\n'.join(html_lines)

    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item."""
        return bool(re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line))

    def _process_list_item(self, line: str) -> Tuple[str, Dict[str, Any]]:
        """Process a list item."""
        list_info = {
            "type": "ordered" if re.match(r'^\s*\d+\.\s+', line) else "unordered",
            "level": len(line) - len(line.lstrip()),
            "text": line.strip()
        }
        
        return line, list_info

    def _is_blockquote(self, line: str) -> bool:
        """Check if line is a blockquote."""
        return line.strip().startswith('>')

    def _process_blockquote(self, line: str) -> Tuple[str, Dict[str, Any]]:
        """Process a blockquote."""
        quote_info = {
            "text": line.strip()[1:].strip(),
            "level": len(line) - len(line.lstrip())
        }
        
        return line, quote_info

    def _is_link_or_image(self, line: str) -> bool:
        """Check if line contains markdown links or images."""
        return bool(re.search(r'\[.*?\]\(.*?\)', line))

    def _process_link_or_image(self, line: str) -> Tuple[str, Dict[str, Any]]:
        """Process links and images in a line."""
        link_pattern = r'!?\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(link_pattern, line)
        
        link_info = {
            "type": "image" if line.strip().startswith('!') else "link",
            "matches": matches
        }
        
        return line, link_info

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of file extensions supported by this loader."""
        return ['.md', '.markdown']

    @classmethod
    def can_load(cls, source: str) -> bool:
        """Check if this loader can handle the given source."""
        if not source:
            return False
        
        source_path = Path(source)
        return source_path.suffix.lower() in cls.get_supported_extensions()