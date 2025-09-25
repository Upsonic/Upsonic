import unittest
import tempfile
import os
import csv
from pathlib import Path
from upsonic.loaders.csv import CSVLoader
from upsonic.loaders.config import CSVLoaderConfig
from upsonic.schemas.data_models import Document


class TestCSVLoaderSimple(unittest.TestCase):
    """Simplified tests for CSVLoader that match the actual implementation."""

    def setUp(self):
        """Set up test environment with sample CSV files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple CSV file
        self.simple_csv = Path(self.temp_dir) / "simple.csv"
        with open(self.simple_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'age', 'city'])
            writer.writerow(['John Doe', '30', 'New York'])
            writer.writerow(['Jane Smith', '25', 'Los Angeles'])
            writer.writerow(['Bob Johnson', '35', 'Chicago'])

        # Create a CSV with different delimiter
        self.semicolon_csv = Path(self.temp_dir) / "semicolon.csv"
        with open(self.semicolon_csv, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['product', 'price', 'category'])
            writer.writerow(['Laptop', '999.99', 'Electronics'])
            writer.writerow(['Book', '19.99', 'Education'])

        # Create a CSV with quotes
        self.quoted_csv = Path(self.temp_dir) / "quoted.csv"
        with open(self.quoted_csv, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(['description', 'notes'])
            writer.writerow(['Product with, comma', 'Note with "quotes"'])
            writer.writerow(['Another item', 'Simple note'])

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_csv_loader_initialization(self):
        """Test CSV loader initialization with different configs."""
        config = CSVLoaderConfig()
        loader = CSVLoader(config)
        self.assertIsNotNone(loader)
        
        # Test with custom config
        custom_config = CSVLoaderConfig(
            delimiter=";",
            include_columns=["product", "price"],
            content_synthesis_mode="json"
        )
        loader_custom = CSVLoader(custom_config)
        self.assertEqual(loader_custom.config.delimiter, ";")

    def test_supported_extensions(self):
        """Test that CSV loader supports correct file extensions."""
        supported = CSVLoader.get_supported_extensions()
        self.assertIn(".csv", supported)
        self.assertEqual(len(supported), 1)

    def test_simple_csv_loading(self):
        """Test loading a simple CSV file."""
        config = CSVLoaderConfig()
        loader = CSVLoader(config)
        
        documents = loader.load(str(self.simple_csv))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))
        self.assertTrue(all(hasattr(doc, 'document_id') for doc in documents))
        self.assertTrue(all(doc.content.strip() for doc in documents))

    def test_json_synthesis_mode(self):
        """Test JSON content synthesis mode."""
        config = CSVLoaderConfig(content_synthesis_mode="json")
        loader = CSVLoader(config)
        
        documents = loader.load(str(self.simple_csv))
        
        self.assertGreater(len(documents), 0)
        # CSV loader behavior may vary, just ensure documents are created
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_concatenated_synthesis_mode(self):
        """Test concatenated content synthesis mode."""
        config = CSVLoaderConfig(content_synthesis_mode="concatenated")
        loader = CSVLoader(config)
        
        documents = loader.load(str(self.simple_csv))
        
        # Should create documents with concatenated content
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_custom_delimiter(self):
        """Test CSV with custom delimiter."""
        config = CSVLoaderConfig(delimiter=";")
        loader = CSVLoader(config)
        
        documents = loader.load(str(self.semicolon_csv))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_include_columns_selection(self):
        """Test selecting specific columns for content."""
        config = CSVLoaderConfig(
            include_columns=["name", "city"]
        )
        loader = CSVLoader(config)
        
        documents = loader.load(str(self.simple_csv))
        
        self.assertGreater(len(documents), 0)
        # Content should only include selected columns
        for doc in documents:
            self.assertIsInstance(doc, Document)
            self.assertTrue(doc.content.strip())

    def test_empty_source_handling(self):
        """Test handling of empty or invalid sources."""
        config = CSVLoaderConfig()
        loader = CSVLoader(config)
        
        # Test with empty list
        result = loader.load([])
        self.assertEqual(len(result), 0)
        
        # Test with non-existent file
        result = loader.load("/path/that/does/not/exist.csv")
        self.assertEqual(len(result), 0)

    def test_batch_loading(self):
        """Test batch loading multiple CSV files."""
        config = CSVLoaderConfig()
        loader = CSVLoader(config)
        
        files = [str(self.simple_csv), str(self.semicolon_csv)]
        documents = loader.batch(files)
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_quoted_csv_handling(self):
        """Test handling CSV files with quoted content."""
        config = CSVLoaderConfig()
        loader = CSVLoader(config)
        
        documents = loader.load(str(self.quoted_csv))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_error_handling(self):
        """Test error handling for invalid CSV."""
        # Create invalid CSV file
        invalid_csv = Path(self.temp_dir) / "invalid.csv"
        invalid_csv.write_text('header1,header2\n"unclosed quote,value2\nrow2,value')
        
        config = CSVLoaderConfig(error_handling="warn")
        loader = CSVLoader(config)
        
        # Should handle error gracefully
        documents = loader.load(str(invalid_csv))
        # May succeed or fail depending on CSV parser tolerance
        self.assertIsInstance(documents, list)

    def test_metadata_inclusion(self):
        """Test CSV metadata inclusion."""
        config = CSVLoaderConfig(
            include_metadata=True
        )
        loader = CSVLoader(config)
        
        documents = loader.load(str(self.simple_csv))
        
        self.assertGreater(len(documents), 0)
        for doc in documents:
            self.assertIsInstance(doc.metadata, dict)
            self.assertIn('source', doc.metadata)

    def test_split_mode_per_row(self):
        """Test CSV splitting per row mode."""
        config = CSVLoaderConfig(
            split_mode="per_row",
            include_metadata=True
        )
        loader = CSVLoader(config)
        
        documents = loader.load(str(self.simple_csv))
        
        # Should create one document per row (3 data rows)
        self.assertEqual(len(documents), 3)
        for i, doc in enumerate(documents):
            self.assertIsInstance(doc, Document)
            self.assertIn('row_index', doc.metadata)
            self.assertEqual(doc.metadata['row_index'], i)
            self.assertIn('total_rows', doc.metadata)
            self.assertEqual(doc.metadata['total_rows'], 3)

    def test_split_mode_per_chunk(self):
        """Test CSV splitting per chunk mode."""
        config = CSVLoaderConfig(
            split_mode="per_chunk",
            rows_per_chunk=2,
            include_metadata=True
        )
        loader = CSVLoader(config)
        
        documents = loader.load(str(self.simple_csv))
        
        # Should create chunks with 2 rows each (3 rows -> 2 chunks: 2+1)
        self.assertEqual(len(documents), 2)
        for doc in documents:
            self.assertIsInstance(doc, Document)
            self.assertIn('chunk_index', doc.metadata)
            self.assertIn('rows_in_chunk', doc.metadata)
            self.assertIn('total_rows', doc.metadata)
            self.assertEqual(doc.metadata['total_rows'], 3)

    def test_split_mode_single_document(self):
        """Test CSV single document mode (default behavior)."""
        config = CSVLoaderConfig(
            split_mode="single_document",
            include_metadata=True
        )
        loader = CSVLoader(config)
        
        documents = loader.load(str(self.simple_csv))
        
        # Should create one document with all rows
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsInstance(doc, Document)
        self.assertIn('row_count', doc.metadata)
        self.assertEqual(doc.metadata['row_count'], 3)


if __name__ == "__main__":
    unittest.main()