import unittest
import tempfile
import os
from pathlib import Path
from upsonic.loaders.docx import DOCXLoader
from upsonic.loaders.config import DOCXLoaderConfig
from upsonic.schemas.data_models import Document


class TestDOCXLoaderSimple(unittest.TestCase):
    """Simplified tests for DOCXLoader that match the actual implementation."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_docx_loader_initialization(self):
        """Test DOCX loader initialization."""
        config = DOCXLoaderConfig()
        loader = DOCXLoader(config)
        self.assertIsNotNone(loader)

    def test_supported_extensions(self):
        """Test that DOCX loader supports correct file extensions."""
        supported = DOCXLoader.get_supported_extensions()
        self.assertIn(".docx", supported)
        self.assertGreaterEqual(len(supported), 1)

    def test_empty_source_handling(self):
        """Test handling of empty or invalid sources."""
        config = DOCXLoaderConfig()
        loader = DOCXLoader(config)
        
        result = loader.load([])
        self.assertEqual(len(result), 0)
        
        # Test with non-existent file
        result = loader.load("/path/that/does/not/exist.docx")
        self.assertEqual(len(result), 0)

    def test_batch_loading_interface(self):
        """Test the batch loading interface."""
        config = DOCXLoaderConfig()
        loader = DOCXLoader(config)
        
        # Test batch method with empty list
        result = loader.batch([])
        self.assertEqual(len(result), 0)

    def test_docx_config_options(self):
        """Test DOCX loader configuration options."""
        config = DOCXLoaderConfig(
            include_tables=True,
            include_headers=True,
            include_footers=True,
            table_format="markdown"
        )
        loader = DOCXLoader(config)
        
        self.assertTrue(loader.config.include_tables)
        self.assertTrue(loader.config.include_headers)
        self.assertTrue(loader.config.include_footers)
        self.assertEqual(loader.config.table_format, "markdown")

    def test_table_format_options(self):
        """Test different table format options."""
        formats = ["text", "markdown", "html"]
        for fmt in formats:
            config = DOCXLoaderConfig(table_format=fmt)
            loader = DOCXLoader(config)
            self.assertEqual(loader.config.table_format, fmt)


if __name__ == "__main__":
    unittest.main()