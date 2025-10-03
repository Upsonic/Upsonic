import unittest
import tempfile
import os
from pathlib import Path
from upsonic.loaders.pymupdf import PyMuPDFLoader
from upsonic.loaders.config import PyMuPDFLoaderConfig
from upsonic.schemas.data_models import Document


class TestPyMuPDFLoaderSimple(unittest.TestCase):
    """Simplified tests for PyMuPDFLoader that match the actual implementation."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_pymupdf_loader_initialization(self):
        """Test PyMuPDF loader initialization with different configs."""
        # Test with default config
        config = PyMuPDFLoaderConfig()
        loader = PyMuPDFLoader(config)
        self.assertIsNotNone(loader)
        self.assertEqual(loader.config.extraction_mode, "hybrid")
        
        # Test with custom config
        custom_config = PyMuPDFLoaderConfig(
            extraction_mode="hybrid",
            max_file_size=5000000,
            start_page=1,
            end_page=10,
            text_extraction_method="dict",
            include_images=True,
            extract_annotations=True
        )
        loader_custom = PyMuPDFLoader(custom_config)
        self.assertEqual(loader_custom.config.extraction_mode, "hybrid")
        self.assertEqual(loader_custom.config.max_file_size, 5000000)
        self.assertEqual(loader_custom.config.text_extraction_method, "dict")
        self.assertTrue(loader_custom.config.include_images)
        self.assertTrue(loader_custom.config.extract_annotations)

    def test_supported_extensions(self):
        """Test that PyMuPDF loader supports correct file extensions."""
        supported = PyMuPDFLoader.get_supported_extensions()
        self.assertIn(".pdf", supported)
        self.assertEqual(len(supported), 1)

    def test_empty_source_handling(self):
        """Test handling of empty or invalid sources."""
        config = PyMuPDFLoaderConfig()
        loader = PyMuPDFLoader(config)
        
        # Test with empty list
        result = loader.load([])
        self.assertEqual(len(result), 0)
        
        # Test with non-existent file
        result = loader.load("/path/that/does/not/exist.pdf")
        self.assertEqual(len(result), 0)

    def test_pymupdf_config_validation(self):
        """Test PyMuPDF configuration validation."""
        # Test valid extraction modes
        valid_modes = ["text_only", "ocr_only", "hybrid"]
        for mode in valid_modes:
            config = PyMuPDFLoaderConfig(extraction_mode=mode)
            self.assertEqual(config.extraction_mode, mode)
        
        # Test valid text extraction methods
        valid_methods = ["text", "dict", "html", "xml"]
        for method in valid_methods:
            config = PyMuPDFLoaderConfig(text_extraction_method=method)
            self.assertEqual(config.text_extraction_method, method)
        
        # Test valid annotation formats
        valid_formats = ["text", "json"]
        for format_type in valid_formats:
            config = PyMuPDFLoaderConfig(annotation_format=format_type)
            self.assertEqual(config.annotation_format, format_type)

    def test_batch_loading_interface(self):
        """Test the batch loading interface."""
        config = PyMuPDFLoaderConfig()
        loader = PyMuPDFLoader(config)
        
        # Test batch method with empty list
        result = loader.batch([])
        self.assertEqual(len(result), 0)

    def test_error_handling_config(self):
        """Test different error handling configurations."""
        # Test with raise error handling
        config_raise = PyMuPDFLoaderConfig(error_handling="raise")
        loader_raise = PyMuPDFLoader(config_raise)
        self.assertEqual(loader_raise.config.error_handling, "raise")
        
        # Test with warn error handling
        config_warn = PyMuPDFLoaderConfig(error_handling="warn")
        loader_warn = PyMuPDFLoader(config_warn)
        self.assertEqual(loader_warn.config.error_handling, "warn")

    def test_ocr_configuration(self):
        """Test OCR-related configuration options."""
        # Test OCR mode requires rapidocr_onnxruntime (will be mocked in real use)
        try:
            config = PyMuPDFLoaderConfig(extraction_mode="ocr_only")
            loader = PyMuPDFLoader(config)
            # If we get here, OCR engine is available
            self.assertEqual(loader.config.extraction_mode, "ocr_only")
        except ImportError as e:
            # Expected if rapidocr_onnxruntime is not installed
            self.assertIn("rapidocr_onnxruntime", str(e))

    def test_page_range_configuration(self):
        """Test page range configuration options."""
        config = PyMuPDFLoaderConfig(
            start_page=2,
            end_page=5,
            extraction_mode="text_only"
        )
        loader = PyMuPDFLoader(config)
        
        self.assertEqual(loader.config.start_page, 2)
        self.assertEqual(loader.config.end_page, 5)

    def test_content_cleaning_configuration(self):
        """Test content cleaning configuration options."""
        config = PyMuPDFLoaderConfig(
            extra_whitespace_removal=True,
            clean_page_numbers=True,
            skip_empty_content=True
        )
        loader = PyMuPDFLoader(config)
        
        self.assertTrue(loader.config.extra_whitespace_removal)
        self.assertTrue(loader.config.clean_page_numbers)
        self.assertTrue(loader.config.skip_empty_content)

    def test_pymupdf_specific_configuration(self):
        """Test PyMuPDF-specific configuration options."""
        config = PyMuPDFLoaderConfig(
            text_extraction_method="dict",
            include_images=True,
            image_dpi=200,
            preserve_layout=True,
            extract_annotations=True,
            annotation_format="json"
        )
        loader = PyMuPDFLoader(config)
        
        self.assertEqual(loader.config.text_extraction_method, "dict")
        self.assertTrue(loader.config.include_images)
        self.assertEqual(loader.config.image_dpi, 200)
        self.assertTrue(loader.config.preserve_layout)
        self.assertTrue(loader.config.extract_annotations)
        self.assertEqual(loader.config.annotation_format, "json")

    def test_image_dpi_validation(self):
        """Test image DPI validation."""
        # Test valid DPI range
        config = PyMuPDFLoaderConfig(image_dpi=150)
        loader = PyMuPDFLoader(config)
        self.assertEqual(loader.config.image_dpi, 150)
        
        # Test minimum DPI
        config = PyMuPDFLoaderConfig(image_dpi=72)
        loader = PyMuPDFLoader(config)
        self.assertEqual(loader.config.image_dpi, 72)
        
        # Test maximum DPI
        config = PyMuPDFLoaderConfig(image_dpi=600)
        loader = PyMuPDFLoader(config)
        self.assertEqual(loader.config.image_dpi, 600)

    def test_page_number_formatting(self):
        """Test page number formatting configuration."""
        config = PyMuPDFLoaderConfig(
            clean_page_numbers=True,
            page_num_start_format="<page {page_nr}>",
            page_num_end_format="</page {page_nr}>"
        )
        loader = PyMuPDFLoader(config)
        
        self.assertTrue(loader.config.clean_page_numbers)
        self.assertEqual(loader.config.page_num_start_format, "<page {page_nr}>")
        self.assertEqual(loader.config.page_num_end_format, "</page {page_nr}>")

    def test_password_configuration(self):
        """Test password configuration for encrypted PDFs."""
        config = PyMuPDFLoaderConfig(
            pdf_password="test_password"
        )
        loader = PyMuPDFLoader(config)
        
        self.assertEqual(loader.config.pdf_password, "test_password")

    def test_metadata_configuration(self):
        """Test metadata configuration options."""
        config = PyMuPDFLoaderConfig(
            include_metadata=True,
            include_images=True,
            extract_annotations=True
        )
        loader = PyMuPDFLoader(config)
        
        self.assertTrue(loader.config.include_metadata)
        self.assertTrue(loader.config.include_images)
        self.assertTrue(loader.config.extract_annotations)

    def test_can_load_method(self):
        """Test the can_load method."""
        config = PyMuPDFLoaderConfig()
        loader = PyMuPDFLoader(config)
        
        # Test with PDF file
        pdf_path = Path(self.temp_dir) / "test.pdf"
        pdf_path.touch()
        self.assertTrue(loader.can_load(pdf_path))
        
        # Test with non-PDF file
        txt_path = Path(self.temp_dir) / "test.txt"
        txt_path.touch()
        self.assertFalse(loader.can_load(txt_path))
        
        # Test with non-existent file
        non_existent = Path(self.temp_dir) / "non_existent.pdf"
        self.assertFalse(loader.can_load(non_existent))


if __name__ == "__main__":
    unittest.main()
