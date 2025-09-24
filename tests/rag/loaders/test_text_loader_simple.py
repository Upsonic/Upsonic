import unittest
import tempfile
import os
from pathlib import Path
from upsonic.loaders.text import TextLoader
from upsonic.loaders.config import TextLoaderConfig
from upsonic.schemas.data_models import Document


class TestTextLoaderSimple(unittest.TestCase):
    """Simplified tests for TextLoader that match the actual implementation."""

    def setUp(self):
        """Set up test environment with sample text files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple text file
        self.simple_txt = Path(self.temp_dir) / "simple.txt"
        self.simple_txt.write_text("""This is the first paragraph of the text file.
It contains multiple lines and sentences for testing purposes.

This is the second paragraph with different content.
It also has multiple lines to test paragraph detection.

This is the third and final paragraph.
It concludes the document content.""")

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_text_loader_initialization(self):
        """Test text loader initialization."""
        config = TextLoaderConfig()
        loader = TextLoader(config)
        self.assertIsNotNone(loader)

    def test_supported_extensions(self):
        """Test that text loader supports correct file extensions."""
        supported = TextLoader.get_supported_extensions()
        self.assertIn(".txt", supported)
        self.assertGreaterEqual(len(supported), 1)

    def test_simple_text_loading(self):
        """Test loading a simple text file."""
        config = TextLoaderConfig()
        loader = TextLoader(config)
        
        documents = loader.load(str(self.simple_txt))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))
        self.assertTrue(all(hasattr(doc, 'document_id') for doc in documents))

    def test_empty_source_handling(self):
        """Test handling of empty or invalid sources."""
        config = TextLoaderConfig()
        loader = TextLoader(config)
        
        result = loader.load([])
        self.assertEqual(len(result), 0)

    def test_batch_loading(self):
        """Test batch loading multiple text files."""
        config = TextLoaderConfig()
        loader = TextLoader(config)
        
        files = [str(self.simple_txt)]
        documents = loader.batch(files)
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_text_config_options(self):
        """Test text loader configuration options."""
        config = TextLoaderConfig(
            strip_whitespace=True,
            min_chunk_length=10,
            skip_empty_content=True
        )
        loader = TextLoader(config)
        
        self.assertTrue(loader.config.strip_whitespace)
        self.assertEqual(loader.config.min_chunk_length, 10)
        self.assertTrue(loader.config.skip_empty_content)

    def test_metadata_inclusion(self):
        """Test text metadata inclusion."""
        config = TextLoaderConfig(include_metadata=True)
        loader = TextLoader(config)
        
        documents = loader.load(str(self.simple_txt))
        
        self.assertGreater(len(documents), 0)
        for doc in documents:
            self.assertIsInstance(doc.metadata, dict)
            self.assertIn('source', doc.metadata)


if __name__ == "__main__":
    unittest.main()