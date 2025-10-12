"""
Safe OCR test suite for PdfPlumberLoader that avoids concurrent processing issues.

This test suite focuses on OCR functionality while avoiding the crashes
that occur with concurrent image processing in the test environment.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Dict
import logging

# Mock dependencies for testing
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from upsonic.loaders.pdfplumber import PdfPlumberLoader
from upsonic.loaders.config import PdfPlumberLoaderConfig
from upsonic.schemas.data_models import Document


class MockPDFCreatorOCRSafe:
    """Creates PDF files specifically designed for safe OCR testing."""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.pdf_files = {}
    
    def create_simple_ocr_pdf(self) -> Path:
        """Create a simple PDF for basic OCR testing."""
        filename = self.temp_dir / "simple_ocr_safe.pdf"
        
        c = canvas.Canvas(str(filename), pagesize=letter)
        _, height = letter
        
        # Simple text content
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 100, "Simple OCR Test")
        
        c.setFont("Helvetica", 12)
        simple_text = [
            "This is a simple test document.",
            "It contains basic text for OCR processing.",
            "The goal is to test OCR functionality.",
            "OCR should be able to read this text.",
            "This text should be extracted correctly."
        ]
        
        y_position = height - 150
        for line in simple_text:
            c.drawString(100, y_position, line)
            y_position -= 25
        
        c.save()
        self.pdf_files["simple_ocr"] = filename
        return filename
    
    def create_text_heavy_pdf(self) -> Path:
        """Create a PDF with lots of text content for OCR testing."""
        filename = self.temp_dir / "text_heavy_safe.pdf"
        
        c = canvas.Canvas(str(filename), pagesize=letter)
        _, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(100, height - 100, "OCR Test Document")
        
        # Add content with various font sizes
        c.setFont("Helvetica", 14)
        c.drawString(100, height - 150, "This document contains text for OCR testing.")
        
        c.setFont("Helvetica", 12)
        text_content = [
            "OCR (Optical Character Recognition) is a technology that enables",
            "computers to read and interpret text from images and scanned documents.",
            "This technology is crucial for digitizing paper documents and",
            "making them searchable and editable.",
            "",
            "Key features of OCR technology include:",
            "• Text recognition from images",
            "• Multi-language support",
            "• Handwriting recognition",
            "• Layout preservation",
            "• Accuracy improvements with AI"
        ]
        
        y_position = height - 200
        for line in text_content:
            c.drawString(100, y_position, line)
            y_position -= 20
        
        c.save()
        self.pdf_files["text_heavy"] = filename
        return filename
    
    def create_table_ocr_pdf(self) -> Path:
        """Create a PDF with tables for OCR testing."""
        filename = self.temp_dir / "table_ocr_safe.pdf"
        
        doc = SimpleDocTemplate(str(filename), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("OCR Table Test Document", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Table for OCR testing
        story.append(Paragraph("Data Table for OCR:", styles['Heading2']))
        data = [
            ['Product', 'Price', 'Quantity', 'Total'],
            ['Widget A', '$10.00', '5', '$50.00'],
            ['Widget B', '$15.00', '3', '$45.00'],
            ['Widget C', '$20.00', '2', '$40.00'],
            ['', '', 'Total:', '$135.00']
        ]
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        
        doc.build(story)
        self.pdf_files["table_ocr"] = filename
        return filename
    
    def create_all_ocr_pdfs(self) -> Dict[str, Path]:
        """Create all OCR test PDF files."""
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("reportlab is required for PDF creation")
        
        self.create_simple_ocr_pdf()
        self.create_text_heavy_pdf()
        self.create_table_ocr_pdf()
        
        return self.pdf_files


class TestPdfPlumberLoaderOCRSafe(unittest.TestCase):
    """Safe OCR test suite for PdfPlumberLoader that avoids concurrent processing issues."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.pdf_creator = MockPDFCreatorOCRSafe(cls.temp_dir)
        
        if PDFPLUMBER_AVAILABLE:
            cls.test_pdfs = cls.pdf_creator.create_all_ocr_pdfs()
        else:
            cls.test_pdfs = {}
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        import shutil
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test."""
        if not PDFPLUMBER_AVAILABLE:
            self.skipTest("reportlab not available")
        if not OCR_AVAILABLE:
            self.skipTest("OCR not available")
    
    def test_ocr_only_extraction(self):
        """Test OCR-only extraction mode."""
        pdf_path = self.test_pdfs["simple_ocr"]
        config = PdfPlumberLoaderConfig(extraction_mode="ocr_only")
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        # OCR should extract some text content
        self.assertGreater(len(doc.content.strip()), 0)
        print(f"OCR extracted content: {doc.content[:100]}...")
    
    def test_hybrid_extraction(self):
        """Test hybrid extraction mode (text + OCR)."""
        pdf_path = self.test_pdfs["text_heavy"]
        config = PdfPlumberLoaderConfig(extraction_mode="hybrid")
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        # Should extract content from both text and OCR
        self.assertGreater(len(doc.content.strip()), 0)
        print(f"Hybrid extracted content: {doc.content[:100]}...")
    
    def test_ocr_with_tables(self):
        """Test OCR extraction with table content."""
        pdf_path = self.test_pdfs["table_ocr"]
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            extract_tables=True,
            table_format="markdown"
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        # Should extract table content via OCR
        self.assertGreater(len(doc.content.strip()), 0)
        print(f"OCR table content: {doc.content[:100]}...")
    
    def test_ocr_with_metadata(self):
        """Test OCR extraction with metadata."""
        pdf_path = self.test_pdfs["simple_ocr"]
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            include_metadata=True,
            extract_page_dimensions=True
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        
        # Check metadata
        metadata = doc.metadata
        self.assertIn("source", metadata)
        self.assertIn("file_name", metadata)
        self.assertIn("file_extension", metadata)
        self.assertIn("loader_type", metadata)
        self.assertEqual(metadata["file_extension"], ".pdf")
        self.assertEqual(metadata["loader_type"], "pdfplumber")
        print(f"OCR metadata: {metadata}")
    
    def test_ocr_async_loading(self):
        """Test async OCR loading."""
        pdf_path = self.test_pdfs["simple_ocr"]
        config = PdfPlumberLoaderConfig(extraction_mode="ocr_only")
        loader = PdfPlumberLoader(config)
        
        async def test_async():
            documents = await loader.aload(pdf_path)
            return documents
        
        documents = asyncio.run(test_async())
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        self.assertGreater(len(doc.content.strip()), 0)
        print(f"Async OCR content: {doc.content[:100]}...")
    
    def test_ocr_with_crop_box(self):
        """Test OCR extraction with crop box."""
        pdf_path = self.test_pdfs["text_heavy"]
        # Crop to a specific region
        crop_box = (50, 50, 400, 600)  # x0, y0, x1, y1
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            crop_box=crop_box
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        # Content should be extracted from the cropped region
        self.assertIsNotNone(doc.content)
        print(f"Cropped OCR content: {doc.content[:100]}...")
    
    def test_ocr_error_handling(self):
        """Test OCR error handling with corrupted PDF."""
        # Create a corrupted PDF
        corrupted_pdf = self.temp_dir / "corrupted_ocr.pdf"
        with open(corrupted_pdf, 'w', encoding='utf-8') as f:
            f.write("This is not a valid PDF file content.")
        
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            error_handling="warn"
        )
        loader = PdfPlumberLoader(config)
        
        # Should handle error gracefully
        documents = loader.load(corrupted_pdf)
        self.assertEqual(len(documents), 0)
        print("OCR error handling: Successfully handled corrupted PDF")
    
    def test_ocr_with_whitespace_removal(self):
        """Test OCR with whitespace removal."""
        pdf_path = self.test_pdfs["text_heavy"]
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            extra_whitespace_removal=True
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        
        # Check that excessive whitespace is removed
        content = doc.content
        self.assertNotIn("\n\n\n", content)
        print(f"OCR with whitespace removal: {doc.content[:100]}...")
    
    def test_ocr_layout_modes(self):
        """Test OCR with different layout modes."""
        pdf_path = self.test_pdfs["text_heavy"]
        
        # Test with layout mode
        config_layout = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            layout_mode="layout"
        )
        loader_layout = PdfPlumberLoader(config_layout)
        documents_layout = loader_layout.load(pdf_path)
        
        # Test with simple mode
        config_simple = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            layout_mode="simple"
        )
        loader_simple = PdfPlumberLoader(config_simple)
        documents_simple = loader_simple.load(pdf_path)
        
        self.assertEqual(len(documents_layout), 1)
        self.assertEqual(len(documents_simple), 1)
        
        # Both should extract content successfully
        self.assertIsNotNone(documents_layout[0].content)
        self.assertIsNotNone(documents_simple[0].content)
        print(f"Layout mode OCR: {documents_layout[0].content[:50]}...")
        print(f"Simple mode OCR: {documents_simple[0].content[:50]}...")
    
    def test_ocr_text_flow(self):
        """Test OCR with text flow analysis."""
        pdf_path = self.test_pdfs["text_heavy"]
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            use_text_flow=True
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        self.assertGreater(len(doc.content.strip()), 0)
        print(f"Text flow OCR: {doc.content[:100]}...")
    
    def test_ocr_character_margins(self):
        """Test OCR with custom character margins."""
        pdf_path = self.test_pdfs["text_heavy"]
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            char_margin=5.0,
            line_margin=1.0
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        self.assertGreater(len(doc.content.strip()), 0)
        print(f"Character margins OCR: {doc.content[:100]}...")
    
    def test_ocr_precision(self):
        """Test OCR with custom precision settings."""
        pdf_path = self.test_pdfs["simple_ocr"]
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            precision=2.0
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        self.assertGreater(len(doc.content.strip()), 0)
        print(f"Precision OCR: {doc.content[:100]}...")
    
    def test_ocr_keep_blank_chars(self):
        """Test OCR with blank character preservation."""
        pdf_path = self.test_pdfs["text_heavy"]
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            keep_blank_chars=True
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        self.assertGreater(len(doc.content.strip()), 0)
        print(f"Keep blank chars OCR: {doc.content[:100]}...")
    
    def test_ocr_table_settings(self):
        """Test OCR with custom table settings."""
        pdf_path = self.test_pdfs["table_ocr"]
        custom_table_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 5,
            "join_tolerance": 5,
            "edge_min_length": 5,
        }
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            table_settings=custom_table_settings
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        self.assertGreater(len(doc.content.strip()), 0)
        print(f"Table settings OCR: {doc.content[:100]}...")
    
    def test_ocr_custom_metadata(self):
        """Test OCR with custom metadata."""
        pdf_path = self.test_pdfs["simple_ocr"]
        custom_metadata = {"ocr_test": "true", "category": "ocr"}
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            include_metadata=True,
            custom_metadata=custom_metadata
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        metadata = doc.metadata
        
        # Check custom metadata is included
        self.assertEqual(metadata["ocr_test"], "true")
        self.assertEqual(metadata["category"], "ocr")
        print(f"Custom metadata OCR: {metadata}")
    
    def test_ocr_file_size_limits(self):
        """Test OCR with file size limits."""
        pdf_path = self.test_pdfs["simple_ocr"]
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            max_file_size=1000  # Very small limit
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(pdf_path)
        
        # Should skip if file is too large
        if pdf_path.stat().st_size > 1000:
            self.assertEqual(len(documents), 0)
            print("OCR file size limit: File skipped due to size")
        else:
            self.assertGreaterEqual(len(documents), 0)
            print("OCR file size limit: File processed")
    
    def test_ocr_skip_empty_content(self):
        """Test OCR with empty content handling."""
        # Create an empty PDF
        empty_pdf = self.temp_dir / "empty_ocr.pdf"
        c = canvas.Canvas(str(empty_pdf), pagesize=letter)
        c.save()
        
        config = PdfPlumberLoaderConfig(
            extraction_mode="ocr_only",
            skip_empty_content=True
        )
        loader = PdfPlumberLoader(config)
        
        documents = loader.load(empty_pdf)
        
        # Should skip empty content
        self.assertEqual(len(documents), 0)
        print("OCR skip empty content: Successfully skipped empty PDF")
    
    def test_ocr_performance_single(self):
        """Test OCR performance with a single document."""
        pdf_path = self.test_pdfs["simple_ocr"]
        config = PdfPlumberLoaderConfig(extraction_mode="ocr_only")
        loader = PdfPlumberLoader(config)
        
        import time
        start_time = time.time()
        documents = loader.load(pdf_path)
        end_time = time.time()
        
        # Should process efficiently
        self.assertEqual(len(documents), 1)
        processing_time = end_time - start_time
        self.assertLess(processing_time, 30)  # Should complete within 30 seconds for OCR
        
        doc = documents[0]
        self.assertIsNotNone(doc.content)
        print(f"OCR performance: {processing_time:.2f} seconds for single document")
        print(f"OCR content: {doc.content[:100]}...")


if __name__ == "__main__":
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
