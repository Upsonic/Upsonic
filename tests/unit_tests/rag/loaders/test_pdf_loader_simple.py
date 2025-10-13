import unittest
import tempfile
import os
from pathlib import Path
from upsonic.loaders.pdf import PdfLoader
from upsonic.loaders.config import PdfLoaderConfig
from upsonic.schemas.data_models import Document


class TestPdfLoaderSimple(unittest.TestCase):
    """Simplified tests for PdfLoader that match the actual implementation."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_pdf_loader_initialization(self):
        """Test PDF loader initialization with different configs."""
        # Test with default config
        config = PdfLoaderConfig()
        loader = PdfLoader(config)
        self.assertIsNotNone(loader)
        self.assertEqual(loader.config.extraction_mode, "hybrid")
        
        # Test with custom config
        custom_config = PdfLoaderConfig(
            extraction_mode="hybrid",
            max_file_size=5000000,
            start_page=1,
            end_page=10
        )
        loader_custom = PdfLoader(custom_config)
        self.assertEqual(loader_custom.config.extraction_mode, "hybrid")
        self.assertEqual(loader_custom.config.max_file_size, 5000000)

    def test_supported_extensions(self):
        """Test that PDF loader supports correct file extensions."""
        supported = PdfLoader.get_supported_extensions()
        self.assertIn(".pdf", supported)
        self.assertEqual(len(supported), 1)

    def test_empty_source_handling(self):
        """Test handling of empty or invalid sources."""
        config = PdfLoaderConfig()
        loader = PdfLoader(config)
        
        # Test with empty list
        result = loader.load([])
        self.assertEqual(len(result), 0)
        
        # Test with non-existent file
        result = loader.load("/path/that/does/not/exist.pdf")
        self.assertEqual(len(result), 0)

    def test_pdf_config_validation(self):
        """Test PDF configuration validation."""
        # Test valid extraction modes
        valid_modes = ["text_only", "ocr_only", "hybrid"]
        for mode in valid_modes:
            config = PdfLoaderConfig(extraction_mode=mode)
            self.assertEqual(config.extraction_mode, mode)

    def test_batch_loading_interface(self):
        """Test the batch loading interface."""
        config = PdfLoaderConfig()
        loader = PdfLoader(config)
        
        # Test batch method with empty list
        result = loader.batch([])
        self.assertEqual(len(result), 0)

    def test_error_handling_config(self):
        """Test different error handling configurations."""
        # Test with raise error handling
        config_raise = PdfLoaderConfig(error_handling="raise")
        loader_raise = PdfLoader(config_raise)
        self.assertEqual(loader_raise.config.error_handling, "raise")
        
        # Test with warn error handling
        config_warn = PdfLoaderConfig(error_handling="warn")
        loader_warn = PdfLoader(config_warn)
        self.assertEqual(loader_warn.config.error_handling, "warn")

    def test_ocr_configuration(self):
        """Test OCR-related configuration options."""
        # Test OCR mode requires rapidocr_onnxruntime (will be mocked in real use)
        try:
            config = PdfLoaderConfig(extraction_mode="ocr_only")
            loader = PdfLoader(config)
            # If we get here, OCR engine is available
            self.assertEqual(loader.config.extraction_mode, "ocr_only")
        except ImportError as e:
            # Expected if rapidocr_onnxruntime is not installed
            self.assertIn("rapidocr_onnxruntime", str(e))

    def test_page_range_configuration(self):
        """Test page range configuration options."""
        config = PdfLoaderConfig(
            start_page=2,
            end_page=5,
            extraction_mode="text_only"
        )
        loader = PdfLoader(config)
        
        self.assertEqual(loader.config.start_page, 2)
        self.assertEqual(loader.config.end_page, 5)

    def test_content_cleaning_configuration(self):
        """Test content cleaning configuration options."""
        config = PdfLoaderConfig(
            extra_whitespace_removal=True,
            clean_page_numbers=True,
            skip_empty_content=True
        )
        loader = PdfLoader(config)
        
        self.assertTrue(loader.config.extra_whitespace_removal)
        self.assertTrue(loader.config.clean_page_numbers)
        self.assertTrue(loader.config.skip_empty_content)


if __name__ == "__main__":
    unittest.main()