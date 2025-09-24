import unittest
import tempfile
import os
from pathlib import Path
from upsonic.loaders.xml import XMLLoader
from upsonic.loaders.config import XMLLoaderConfig
from upsonic.schemas.data_models import Document


class TestXMLLoaderSimple(unittest.TestCase):
    """Simplified tests for XMLLoader that match the actual implementation."""

    def setUp(self):
        """Set up test environment with sample XML files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple XML file
        self.simple_xml = Path(self.temp_dir) / "simple.xml"
        self.simple_xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<root>
    <person>
        <name>John Doe</name>
        <age>30</age>
        <city>New York</city>
    </person>
    <person>
        <name>Jane Smith</name>
        <age>25</age>
        <city>Los Angeles</city>
    </person>
</root>""")

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_xml_loader_initialization(self):
        """Test XML loader initialization."""
        config = XMLLoaderConfig()
        loader = XMLLoader(config)
        self.assertIsNotNone(loader)

    def test_supported_extensions(self):
        """Test that XML loader supports correct file extensions."""
        supported = XMLLoader.get_supported_extensions()
        self.assertIn(".xml", supported)
        self.assertGreaterEqual(len(supported), 1)

    def test_simple_xml_loading(self):
        """Test loading a simple XML file."""
        config = XMLLoaderConfig()
        loader = XMLLoader(config)
        
        documents = loader.load(str(self.simple_xml))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))
        self.assertTrue(all(hasattr(doc, 'document_id') for doc in documents))

    def test_empty_source_handling(self):
        """Test handling of empty or invalid sources."""
        config = XMLLoaderConfig()
        loader = XMLLoader(config)
        
        result = loader.load([])
        self.assertEqual(len(result), 0)

    def test_batch_loading(self):
        """Test batch loading multiple XML files."""
        config = XMLLoaderConfig()
        loader = XMLLoader(config)
        
        files = [str(self.simple_xml)]
        documents = loader.batch(files)
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_xml_config_options(self):
        """Test XML loader configuration options."""
        config = XMLLoaderConfig(
            split_by_xpath="//person",
            content_xpath="./name",
            content_synthesis_mode="smart_text",
            include_attributes=True,
            strip_namespaces=True,
            recover_mode=False
        )
        loader = XMLLoader(config)
        
        self.assertEqual(loader.config.split_by_xpath, "//person")
        self.assertEqual(loader.config.content_xpath, "./name")
        self.assertEqual(loader.config.content_synthesis_mode, "smart_text")
        self.assertTrue(loader.config.include_attributes)
        self.assertTrue(loader.config.strip_namespaces)
        self.assertFalse(loader.config.recover_mode)

    def test_content_synthesis_modes(self):
        """Test different content synthesis modes."""
        modes = ["smart_text", "xml_snippet"]
        for mode in modes:
            config = XMLLoaderConfig(content_synthesis_mode=mode)
            loader = XMLLoader(config)
            self.assertEqual(loader.config.content_synthesis_mode, mode)

    def test_metadata_inclusion(self):
        """Test XML metadata inclusion."""
        config = XMLLoaderConfig(include_metadata=True)
        loader = XMLLoader(config)
        
        documents = loader.load(str(self.simple_xml))
        
        self.assertGreater(len(documents), 0)
        for doc in documents:
            self.assertIsInstance(doc.metadata, dict)
            self.assertIn('source', doc.metadata)


if __name__ == "__main__":
    unittest.main()