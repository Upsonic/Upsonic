import unittest
import tempfile
import os
from pathlib import Path
from upsonic.loaders.html import HTMLLoader
from upsonic.loaders.config import HTMLLoaderConfig
from upsonic.schemas.data_models import Document


class TestHTMLLoaderSimple(unittest.TestCase):
    """Simplified tests for HTMLLoader that match the actual implementation."""

    def setUp(self):
        """Set up test environment with sample HTML files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple HTML file
        self.simple_html = Path(self.temp_dir) / "simple.html"
        self.simple_html.write_text("""
<!DOCTYPE html>
<html>
<head>
    <title>Simple Page</title>
</head>
<body>
    <h1>Welcome to My Website</h1>
    <p>This is a simple paragraph with some text content.</p>
    <p>This is another paragraph with more information.</p>
    <div>Some content in a div element.</div>
</body>
</html>
""")

        # Create HTML with multiple sections
        self.sectioned_html = Path(self.temp_dir) / "sectioned.html"
        self.sectioned_html.write_text("""
<!DOCTYPE html>
<html>
<head>
    <title>Sectioned Page</title>
</head>
<body>
    <header>
        <h1>Main Header</h1>
    </header>
    <section id="intro">
        <h2>Introduction</h2>
        <p>This is the introduction section.</p>
    </section>
    <section id="content">
        <h2>Main Content</h2>
        <p>This is the main content section.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
        </ul>
    </section>
    <footer>
        <p>Footer content here.</p>
    </footer>
</body>
</html>
""")

        # Create HTML with tables
        self.table_html = Path(self.temp_dir) / "table.html"
        self.table_html.write_text("""
<!DOCTYPE html>
<html>
<head>
    <title>Table Page</title>
</head>
<body>
    <h1>Data Table</h1>
    <table>
        <tr>
            <th>Name</th>
            <th>Age</th>
            <th>City</th>
        </tr>
        <tr>
            <td>John</td>
            <td>30</td>
            <td>New York</td>
        </tr>
        <tr>
            <td>Jane</td>
            <td>25</td>
            <td>Los Angeles</td>
        </tr>
    </table>
</body>
</html>
""")

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_html_loader_initialization(self):
        """Test HTML loader initialization with different configs."""
        config = HTMLLoaderConfig()
        loader = HTMLLoader(config)
        self.assertIsNotNone(loader)
        
        # Test with custom config
        custom_config = HTMLLoaderConfig(
            extract_text=True,
            preserve_structure=False
        )
        loader_custom = HTMLLoader(custom_config)
        self.assertTrue(loader_custom.config.extract_text)

    def test_supported_extensions(self):
        """Test that HTML loader supports correct file extensions."""
        supported = HTMLLoader.get_supported_extensions()
        self.assertIn(".html", supported)
        self.assertIn(".htm", supported)
        # May include .xhtml as well
        self.assertGreaterEqual(len(supported), 2)

    def test_simple_html_loading(self):
        """Test loading a simple HTML file."""
        config = HTMLLoaderConfig()
        loader = HTMLLoader(config)
        
        documents = loader.load(str(self.simple_html))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))
        self.assertTrue(all(hasattr(doc, 'document_id') for doc in documents))
        self.assertTrue(all(doc.content.strip() for doc in documents))

    def test_text_extraction_only(self):
        """Test extracting text content only."""
        config = HTMLLoaderConfig(extract_text=True)
        loader = HTMLLoader(config)
        
        documents = loader.load(str(self.simple_html))
        
        self.assertGreater(len(documents), 0)
        for doc in documents:
            self.assertIsInstance(doc, Document)
            # Text should not contain HTML tags
            self.assertNotIn('<', doc.content)
            self.assertNotIn('>', doc.content)

    def test_preserve_structure(self):
        """Test preserving HTML structure."""
        config = HTMLLoaderConfig(preserve_structure=True)
        loader = HTMLLoader(config)
        
        documents = loader.load(str(self.simple_html))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_element_extraction_options(self):
        """Test different element extraction options."""
        config = HTMLLoaderConfig(
            extract_text=True,
            extract_headers=True,
            extract_paragraphs=True,
            extract_lists=True,
            extract_tables=True
        )
        loader = HTMLLoader(config)
        
        documents = loader.load(str(self.sectioned_html))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_sectioned_html_loading(self):
        """Test loading HTML with multiple sections."""
        config = HTMLLoaderConfig(preserve_structure=True)
        loader = HTMLLoader(config)
        
        documents = loader.load(str(self.sectioned_html))
        
        self.assertGreater(len(documents), 0)
        # May create multiple documents for different sections
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_table_extraction(self):
        """Test extracting content from HTML tables."""
        config = HTMLLoaderConfig(
            extract_text=True,
            extract_tables=True,
            table_format="markdown",
            include_images=False
        )
        loader = HTMLLoader(config)
        
        documents = loader.load(str(self.table_html))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_empty_source_handling(self):
        """Test handling of empty or invalid sources."""
        config = HTMLLoaderConfig()
        loader = HTMLLoader(config)
        
        # Test with empty list
        result = loader.load([])
        self.assertEqual(len(result), 0)
        
        # Test with non-existent file
        result = loader.load("/path/that/does/not/exist.html")
        self.assertEqual(len(result), 0)

    def test_batch_loading(self):
        """Test batch loading multiple HTML files."""
        config = HTMLLoaderConfig()
        loader = HTMLLoader(config)
        
        files = [str(self.simple_html), str(self.sectioned_html)]
        documents = loader.batch(files)
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_error_handling(self):
        """Test error handling for invalid HTML."""
        # Create invalid HTML file
        invalid_html = Path(self.temp_dir) / "invalid.html"
        invalid_html.write_text('<html><body><p>Unclosed paragraph<div>Nested incorrectly</p></div></body>')
        
        config = HTMLLoaderConfig(error_handling="warn")
        loader = HTMLLoader(config)
        
        # Should handle error gracefully
        documents = loader.load(str(invalid_html))
        # HTML parsers are usually tolerant, so this might still succeed
        self.assertIsInstance(documents, list)

    def test_metadata_inclusion(self):
        """Test HTML metadata inclusion."""
        config = HTMLLoaderConfig(include_metadata=True)
        loader = HTMLLoader(config)
        
        documents = loader.load(str(self.simple_html))
        
        self.assertGreater(len(documents), 0)
        for doc in documents:
            self.assertIsInstance(doc.metadata, dict)
            self.assertIn('source', doc.metadata)


if __name__ == "__main__":
    unittest.main()