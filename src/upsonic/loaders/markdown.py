import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import frontmatter
    _FRONTMATTER_AVAILABLE = True
except ImportError:
    frontmatter = None
    _FRONTMATTER_AVAILABLE = False


try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
    _MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MarkdownIt = None
    Token = None
    _MARKDOWN_IT_AVAILABLE = False


from upsonic.schemas.data_models import Document
from upsonic.loaders.base import BaseLoader
from upsonic.loaders.config import MarkdownLoaderConfig


class MarkdownLoader(BaseLoader):
    """
    A comprehensive, high-level loader for Markdown (.md) files.

    This loader uses a token-based parser to deeply understand the document's
    structure. It supports semantic chunking by specified heading levels (h1, h2, h3)
    and extracts valuable metadata, such as front matter, heading structures, and
    code block languages, all driven by a detailed configuration.

    It is designed for performance and accuracy, respecting all options provided
    in the MarkdownLoaderConfig to deliver precisely tailored output.
    """

    def __init__(self, config: Optional[MarkdownLoaderConfig] = None):
        """Initializes the MarkdownLoader with its specific configuration."""
        if config is None:
            config = MarkdownLoaderConfig()
        if not _FRONTMATTER_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="python-frontmatter",
                install_command='pip install "upsonic[loaders]"',
                feature_name="Markdown loader"
            )
        if not _MARKDOWN_IT_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="markdown-it-py",
                install_command='pip install "upsonic[loaders]"',
                feature_name="Markdown loader"
            )
        super().__init__(config)
        self.config: MarkdownLoaderConfig = config
        self.md_parser = MarkdownIt()

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Gets the list of supported file extensions."""
        return [".md", ".markdown"]

    def load(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads documents from the given Markdown source(s) synchronously."""
        files_to_process = self._resolve_sources(source)
        all_documents = []
        for file_path in files_to_process:
            all_documents.extend(self._load_single_file(file_path))
        return all_documents

    async def aload(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Loads documents from the given Markdown source(s) asynchronously."""
        files_to_process = await asyncio.to_thread(self._resolve_sources, source)
        tasks = [asyncio.to_thread(self._load_single_file, file) for file in files_to_process]
        results = await asyncio.gather(*tasks)
        return [doc for sublist in results for doc in sublist]

    def batch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """A simple synchronous batch load implementation."""
        return self.load(sources)

    async def abatch(self, sources: List[Union[str, Path]]) -> List[Document]:
        """An efficient asynchronous batch load implementation."""
        return await self.aload(sources)

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Loads, parses, and chunks a single Markdown file based on the config."""
        if not self._check_file_size(file_path):
            return []
        
        try:
            document_id = self._generate_document_id(file_path)
            if document_id in self._processed_document_ids:
                raise FileExistsError(f"Source file '{file_path.resolve()}' has already been processed.")
            self._processed_document_ids.add(document_id)

            post = frontmatter.load(file_path, encoding=self.config.encoding or 'utf-8')
            
            base_metadata = self._create_metadata(file_path)
            if self.config.parse_front_matter:
                base_metadata.update(post.metadata)
            
            tokens = self.md_parser.parse(post.content)
            chunks = self._chunk_tokens(tokens)

            documents = []
            for i, chunk_tokens in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata['chunk_index'] = i
                
                doc = self._process_chunk(chunk_tokens, document_id, chunk_metadata)
                if doc:
                    documents.append(doc)
            
            return documents

        except Exception as e:
            return self._handle_loading_error(str(file_path), e)

    def _chunk_tokens(self, tokens: List[Token]) -> List[List[Token]]:
        """Splits a list of tokens into chunks if split_by_heading is configured."""
        if not self.config.split_by_heading:
            return [tokens]

        chunks: List[List[Token]] = []
        current_chunk: List[Token] = []
        
        for token in tokens:
            is_split_heading = token.type == 'heading_open' and token.tag == self.config.split_by_heading
            
            if is_split_heading and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []

            current_chunk.append(token)

        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def _process_chunk(self, tokens: List[Token], document_id: str, metadata: Dict[str, Any]) -> Optional[Document]:
        """Processes a single chunk of tokens into a Document, extracting content and metadata."""
        content_parts = []
        extracted_headings = {}
        extracted_langs = set()

        for token in tokens:
            if not self.config.include_code_blocks and token.type in ("fence", "code_block"):
                continue
            
            if token.content:
                content_parts.append(token.content)

            if self.config.include_metadata:
                if self.config.heading_metadata and token.type == 'heading_open':
                    inline_token_index = tokens.index(token) + 1
                    if inline_token_index < len(tokens) and tokens[inline_token_index].type == 'inline':
                        heading_text = tokens[inline_token_index].content
                        extracted_headings.setdefault(token.tag, []).append(heading_text)

                if self.config.code_block_language_metadata and token.type == 'fence' and token.info:
                    extracted_langs.add(token.info.strip())

        final_content = "".join(content_parts).strip()
        
        if self.config.skip_empty_content and not final_content:
            return None

        if self.config.include_metadata:
            if extracted_headings:
                metadata['headings'] = extracted_headings
            if extracted_langs:
                metadata['code_languages'] = sorted(list(extracted_langs))
            if self.config.custom_metadata:
                metadata.update(self.config.custom_metadata)
        else:
            metadata = {} # If false, no metadata is included at all.

        return Document(document_id=document_id, content=final_content, metadata=metadata)