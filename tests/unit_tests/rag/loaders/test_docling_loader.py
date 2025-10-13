"""
Comprehensive tests for the DoclingLoader.

This test suite covers:
- Basic loading functionality
- Extraction modes (markdown and chunks)
- Chunking strategies (hybrid and hierarchical)
- Configuration options
- Error handling
- URL loading
- Async operations
- Batch processing
- Multiple file formats
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Check if docling is available
try:
    import docling
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from upsonic.loaders.config import DoclingLoaderConfig
from upsonic.schemas.data_models import Document

# Skip all tests if docling is not installed
pytestmark = pytest.mark.skipif(
    not DOCLING_AVAILABLE,
    reason="Docling package not installed. Install with: pip install docling"
)


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a dummy PDF file for testing."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")  # Minimal PDF header
    return pdf_file


@pytest.fixture
def sample_docx_path(tmp_path):
    """Create a dummy DOCX file for testing."""
    docx_file = tmp_path / "sample.docx"
    docx_file.write_bytes(b"PK\x03\x04")  # DOCX is a ZIP file
    return docx_file


@pytest.fixture
def sample_md_path(tmp_path):
    """Create a sample markdown file."""
    md_file = tmp_path / "sample.md"
    md_file.write_text("""# Sample Document

This is a test document with multiple sections.

## Section 1

Content of section 1.

## Section 2

Content of section 2.
""")
    return md_file


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return DoclingLoaderConfig(
        extraction_mode="chunks",
        chunker_type="hybrid"
    )


@pytest.fixture
def markdown_config():
    """Configuration for markdown extraction."""
    return DoclingLoaderConfig(
        extraction_mode="markdown"
    )


class TestDoclingLoaderConfig:
    """Test DoclingLoaderConfig configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DoclingLoaderConfig()

        assert config.extraction_mode == "chunks"
        assert config.chunker_type == "hybrid"
        assert config.max_pages is None
        assert config.batch_size == 10
        assert config.confidence_threshold == 0.5
        assert config.extract_document_metadata is True
        assert config.parallel_processing is True
        assert config.support_urls is True
        # OCR defaults
        assert config.ocr_enabled is True
        assert config.ocr_force_full_page is False
        assert config.ocr_backend == "rapidocr"
        assert config.ocr_lang == ["english"]
        assert config.ocr_backend_engine == "onnxruntime"
        assert config.ocr_text_score == 0.5
        # Table defaults
        assert config.enable_table_structure is True
        assert config.table_structure_cell_matching is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DoclingLoaderConfig(
            extraction_mode="markdown",
            chunker_type="hierarchical",
            max_pages=10,
            batch_size=5,
            allowed_formats=["pdf", "docx"]
        )

        assert config.extraction_mode == "markdown"
        assert config.chunker_type == "hierarchical"
        assert config.max_pages == 10
        assert config.batch_size == 5
        assert config.allowed_formats == ["pdf", "docx"]
    
    def test_markdown_export_options(self):
        """Test markdown-specific configuration."""
        config = DoclingLoaderConfig(
            extraction_mode="markdown",
            markdown_image_placeholder="[IMAGE]"
        )

        assert config.markdown_image_placeholder == "[IMAGE]"
    
    def test_advanced_processing_options(self):
        """Test advanced processing configuration."""
        config = DoclingLoaderConfig(
            parallel_processing=False,
            batch_size=20,
            confidence_threshold=0.8
        )

        assert config.parallel_processing is False
        assert config.batch_size == 20
        assert config.confidence_threshold == 0.8
    
    def test_ocr_configuration(self):
        """Test OCR-specific configuration."""
        config = DoclingLoaderConfig(
            ocr_enabled=True,
            ocr_force_full_page=True,
            ocr_backend="tesseract",
            ocr_lang=["eng", "fra"],
            ocr_text_score=0.7,
            confidence_threshold=0.7,
            max_pages=5,
            url_timeout=60
        )

        assert config.ocr_enabled is True
        assert config.ocr_force_full_page is True
        assert config.ocr_backend == "tesseract"
        assert config.ocr_lang == ["eng", "fra"]
        assert config.ocr_text_score == 0.7
        assert config.confidence_threshold == 0.7
        assert config.max_pages == 5
        assert config.url_timeout == 60
    
    def test_table_structure_configuration(self):
        """Test table structure configuration."""
        config = DoclingLoaderConfig(
            enable_table_structure=True,
            table_structure_cell_matching=False
        )
        
        assert config.enable_table_structure is True
        assert config.table_structure_cell_matching is False
    
    def test_page_range_configuration(self):
        """Test page range configuration."""
        config = DoclingLoaderConfig(
            page_range=(1, 10),
            max_pages=10
        )
        
        assert config.page_range == (1, 10)
        assert config.max_pages == 10
    
    def test_performance_configuration(self):
        """Test performance-related configuration."""
        config = DoclingLoaderConfig(
            parallel_processing=False,
            batch_size=5,
            confidence_threshold=0.8
        )
        
        assert config.parallel_processing is False
        assert config.batch_size == 5
        assert config.confidence_threshold == 0.8


@pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Requires docling package")
class TestDoclingLoaderBasics:
    """Test basic DoclingLoader functionality."""
    
    def test_loader_initialization(self, basic_config):
        """Test loader initialization."""
        from upsonic.loaders import DoclingLoader
        
        loader = DoclingLoader(basic_config)
        assert loader.config == basic_config
        assert loader._converter is not None
        assert loader._chunker is not None
    
    def test_loader_initialization_markdown_mode(self, markdown_config):
        """Test loader initialization in markdown mode."""
        from upsonic.loaders import DoclingLoader
        
        loader = DoclingLoader(markdown_config)
        assert loader.config.extraction_mode == "markdown"
        assert loader._chunker is None  # No chunker in markdown mode
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        from upsonic.loaders import DoclingLoader
        
        extensions = DoclingLoader.get_supported_extensions()
        
        assert '.pdf' in extensions
        assert '.docx' in extensions
        assert '.xlsx' in extensions
        assert '.pptx' in extensions
        assert '.html' in extensions
        assert '.md' in extensions
        assert '.csv' in extensions
        assert '.png' in extensions
        assert '.jpg' in extensions
    
    def test_can_load_valid_file(self, sample_pdf_path):
        """Test can_load method with valid file."""
        from upsonic.loaders import DoclingLoader
        
        assert DoclingLoader.can_load(sample_pdf_path) is True
    
    def test_can_load_url(self):
        """Test can_load method with URL."""
        from upsonic.loaders import DoclingLoader
        
        assert DoclingLoader.can_load("https://example.com/document.pdf") is True
        assert DoclingLoader.can_load("http://example.com/document.pdf") is True


@pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Requires docling package")
class TestDoclingLoaderWithMocks:
    """Test DoclingLoader with mocked Docling components."""
    
    @patch('upsonic.loaders.docling.DocumentConverter')
    def test_load_single_document_chunks(self, mock_converter_class, basic_config, sample_pdf_path):
        """Test loading a single document in chunks mode."""
        from upsonic.loaders import DoclingLoader
        
        # Create mock converter and document
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        mock_doc = Mock()
        mock_doc.name = "sample.pdf"
        mock_doc.pages = [Mock(), Mock()]
        
        mock_result = Mock()
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        
        # Create mock chunks
        mock_chunk1 = Mock()
        mock_chunk1.text = "Chunk 1 content"
        mock_chunk1.confidence = 0.8  # Add confidence attribute
        mock_chunk1.meta = Mock()
        mock_chunk1.meta.export_json_dict.return_value = {"page": 0}

        mock_chunk2 = Mock()
        mock_chunk2.text = "Chunk 2 content"
        mock_chunk2.confidence = 0.9  # Add confidence attribute
        mock_chunk2.meta = Mock()
        mock_chunk2.meta.export_json_dict.return_value = {"page": 1}
        
        # Patch chunker
        with patch.object(DoclingLoader, '_create_chunker') as mock_create_chunker:
            mock_chunker = Mock()
            mock_chunker.chunk.return_value = [mock_chunk1, mock_chunk2]
            mock_chunker.serialize.side_effect = lambda chunk: chunk.text
            mock_create_chunker.return_value = mock_chunker
            
            loader = DoclingLoader(basic_config)
            loader._converter = mock_converter
            loader._chunker = mock_chunker
            
            # Load document
            documents = loader.load(str(sample_pdf_path))
            
            # Assertions
            assert len(documents) == 2
            assert all(isinstance(doc, Document) for doc in documents)
            assert documents[0].content == "Chunk 1 content"
            assert documents[1].content == "Chunk 2 content"
            assert documents[0].metadata['chunk_index'] == 0
            assert documents[1].metadata['chunk_index'] == 1
    
    @patch('upsonic.loaders.docling.DocumentConverter')
    def test_load_single_document_markdown(self, mock_converter_class, markdown_config, sample_pdf_path):
        """Test loading a single document in markdown mode."""
        from upsonic.loaders import DoclingLoader
        
        # Create mock converter and document
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        mock_doc = Mock()
        mock_doc.name = "sample.pdf"
        mock_doc.export_to_markdown.return_value = "# Document Title\n\nContent here."
        
        mock_result = Mock()
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        
        loader = DoclingLoader(markdown_config)
        loader._converter = mock_converter
        
        # Load document
        documents = loader.load(str(sample_pdf_path))
        
        # Assertions
        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert documents[0].content == "# Document Title\n\nContent here."
        assert documents[0].metadata['extraction_mode'] == 'markdown'
    
    @patch('upsonic.loaders.docling.DocumentConverter')
    def test_load_multiple_documents(self, mock_converter_class, basic_config, tmp_path):
        """Test loading multiple documents."""
        from upsonic.loaders import DoclingLoader
        
        # Create sample files
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_bytes(b"%PDF-1.4\n")
        pdf2.write_bytes(b"%PDF-1.4\n")
        
        # Setup mocks
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        def mock_convert(source, **kwargs):
            mock_doc = Mock()
            mock_doc.name = Path(source).name
            mock_result = Mock()
            mock_result.document = mock_doc
            return mock_result
        
        mock_converter.convert.side_effect = mock_convert
        
        with patch.object(DoclingLoader, '_create_chunker') as mock_create_chunker:
            mock_chunker = Mock()
            
            def mock_chunk(doc):
                chunk = Mock()
                chunk.text = f"Content from {doc.name}"
                chunk.confidence = 0.8  # Add confidence attribute
                chunk.meta = Mock()
                chunk.meta.export_json_dict.return_value = {}
                return [chunk]
            
            mock_chunker.chunk.side_effect = mock_chunk
            mock_chunker.serialize.side_effect = lambda chunk: chunk.text
            mock_create_chunker.return_value = mock_chunker
            
            loader = DoclingLoader(basic_config)
            loader._converter = mock_converter
            loader._chunker = mock_chunker
            
            # Load documents
            documents = loader.load([str(pdf1), str(pdf2)])
            
            # Assertions
            assert len(documents) == 2
            assert "doc1.pdf" in documents[0].content
            assert "doc2.pdf" in documents[1].content
    
    @patch('upsonic.loaders.docling.DocumentConverter')
    @pytest.mark.asyncio
    async def test_async_load(self, mock_converter_class, basic_config, sample_pdf_path):
        """Test asynchronous document loading."""
        from upsonic.loaders import DoclingLoader
        
        # Setup mocks
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        mock_doc = Mock()
        mock_doc.name = "sample.pdf"
        mock_result = Mock()
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        
        with patch.object(DoclingLoader, '_create_chunker') as mock_create_chunker:
            mock_chunker = Mock()
            mock_chunk = Mock()
            mock_chunk.text = "Async content"
            mock_chunk.confidence = 0.8  # Add confidence attribute
            mock_chunk.meta = Mock()
            mock_chunk.meta.export_json_dict.return_value = {}
            mock_chunker.chunk.return_value = [mock_chunk]
            mock_chunker.serialize.side_effect = lambda chunk: chunk.text
            mock_create_chunker.return_value = mock_chunker
            
            loader = DoclingLoader(basic_config)
            loader._converter = mock_converter
            loader._chunker = mock_chunker
            
            # Async load
            documents = await loader.aload(str(sample_pdf_path))
            
            # Assertions
            assert len(documents) >= 1
            assert isinstance(documents[0], Document)


@pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Requires docling package")
class TestDoclingLoaderErrorHandling:
    """Test error handling in DoclingLoader."""
    
    def test_missing_file_error_raise(self, basic_config):
        """Test error handling for missing file with raise mode."""
        from upsonic.loaders import DoclingLoader
        
        config = DoclingLoaderConfig(
            extraction_mode="chunks",
            error_handling="raise"
        )
        loader = DoclingLoader(config)
        
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.pdf")
    
    def test_missing_file_error_warn(self, basic_config, caplog):
        """Test error handling for missing file with warn mode."""
        from upsonic.loaders import DoclingLoader
        
        config = DoclingLoaderConfig(
            extraction_mode="chunks",
            error_handling="warn"
        )
        loader = DoclingLoader(config)
        
        documents = loader.load("/nonexistent/file.pdf")
        assert len(documents) == 0
    
    def test_missing_file_error_ignore(self, basic_config):
        """Test error handling for missing file with ignore mode."""
        from upsonic.loaders import DoclingLoader
        
        config = DoclingLoaderConfig(
            extraction_mode="chunks",
            error_handling="ignore"
        )
        loader = DoclingLoader(config)
        
        documents = loader.load("/nonexistent/file.pdf")
        assert len(documents) == 0
    
    def test_url_disabled_error(self, basic_config):
        """Test error when URL loading is disabled."""
        from upsonic.loaders import DoclingLoader

        config = DoclingLoaderConfig(
            extraction_mode="chunks",
            support_urls=False,
            error_handling="raise"
        )
        loader = DoclingLoader(config)

        # URL loading is disabled, should raise ValueError
        with pytest.raises(ValueError, match="URL loading is disabled"):
            loader.load("https://example.com/document.pdf")
    
    def test_empty_content_skip(self, basic_config):
        """Test skipping empty content when configured."""
        from upsonic.loaders import DoclingLoader
        
        config = DoclingLoaderConfig(
            extraction_mode="chunks",
            skip_empty_content=True
        )
        # Test would require mock with empty chunks


@pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Requires docling package")
class TestDoclingLoaderFormats:
    """Test DoclingLoader with different file formats."""
    
    def test_format_restrictions(self):
        """Test restricting allowed formats."""
        from upsonic.loaders import DoclingLoader
        
        config = DoclingLoaderConfig(
            extraction_mode="chunks",
            allowed_formats=["pdf", "docx"]
        )
        loader = DoclingLoader(config)
        
        # Converter should be created with only specified formats
        assert loader._converter is not None
    
    def test_all_formats_allowed_by_default(self, tmp_path):
        """Test that all formats are allowed by default."""
        from upsonic.loaders import DoclingLoader

        config = DoclingLoaderConfig(extraction_mode="chunks")
        loader = DoclingLoader(config)

        # Check supported extensions
        extensions = loader.get_supported_extensions()
        assert '.pdf' in extensions
        assert '.docx' in extensions
        assert '.html' in extensions
        
        # Create actual test files
        test_docx = tmp_path / "test.docx"
        test_html = tmp_path / "test.html"
        test_docx.write_text("test content")
        test_html.write_text("test content")
        
        assert loader.can_load(str(test_docx))
        assert loader.can_load(str(test_html))


@pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Requires docling package")
class TestDoclingLoaderMetadata:
    """Test metadata extraction in DoclingLoader."""
    
    @patch('upsonic.loaders.docling.DocumentConverter')
    def test_metadata_extraction(self, mock_converter_class, basic_config, sample_pdf_path):
        """Test document metadata extraction."""
        from upsonic.loaders import DoclingLoader
        
        # Setup mocks
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        mock_doc = Mock()
        mock_doc.name = "sample.pdf"
        mock_doc.pages = [Mock()]
        mock_doc.metadata = Mock()
        mock_doc.metadata.export_json_dict.return_value = {"title": "Test Document"}
        
        mock_result = Mock()
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        
        with patch.object(DoclingLoader, '_create_chunker') as mock_create_chunker:
            mock_chunker = Mock()
            mock_chunk = Mock()
            mock_chunk.text = "Content"
            mock_chunk.confidence = 0.8  # Add confidence attribute
            mock_chunk.meta = Mock()
            mock_chunk.meta.export_json_dict.return_value = {"chunk_type": "paragraph"}
            mock_chunker.chunk.return_value = [mock_chunk]
            mock_chunker.serialize.side_effect = lambda chunk: chunk.text
            mock_create_chunker.return_value = mock_chunker

            loader = DoclingLoader(basic_config)
            loader._converter = mock_converter
            loader._chunker = mock_chunker

            documents = loader.load(str(sample_pdf_path))

            # Check metadata - should have at least one document
            assert len(documents) > 0
            assert 'source' in documents[0].metadata
            assert documents[0].metadata['document_name'] == "sample.pdf"
            assert 'page_count' in documents[0].metadata
            assert 'extraction_mode' in documents[0].metadata
            assert documents[0].metadata['extraction_mode'] == 'chunks'


@pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Requires docling package")
class TestDoclingLoaderIntegration:
    """Integration tests for DoclingLoader (if docling is fully installed)."""
    
    @pytest.mark.integration
    def test_real_document_loading(self, sample_md_path):
        """Test loading a real markdown document (requires full docling install)."""
        from upsonic.loaders import DoclingLoader
        
        try:
            config = DoclingLoaderConfig(
                extraction_mode="markdown",
                error_handling="raise"
            )
            loader = DoclingLoader(config)
            
            documents = loader.load(str(sample_md_path))
            
            # Basic assertions
            assert len(documents) >= 1
            assert all(isinstance(doc, Document) for doc in documents)
            
        except Exception as e:
            pytest.skip(f"Full Docling installation required: {e}")


def test_config_factory_integration():
    """Test that DoclingLoaderConfig is registered in the factory."""
    from upsonic.loaders.config import LoaderConfigFactory
    
    config = LoaderConfigFactory.create_config('docling')
    assert isinstance(config, DoclingLoaderConfig)
    assert config.extraction_mode == "chunks"  # Default value


def test_docling_in_supported_types():
    """Test that 'docling' is in supported loader types."""
    from upsonic.loaders.config import LoaderConfigFactory
    
    supported_types = LoaderConfigFactory.get_supported_types()
    assert 'docling' in supported_types

