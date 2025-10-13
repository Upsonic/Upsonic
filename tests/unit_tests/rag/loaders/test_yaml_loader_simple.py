import unittest
import tempfile
import os
import uuid
from pathlib import Path
from upsonic.loaders.yaml import YAMLLoader
from upsonic.loaders.config import YAMLLoaderConfig
from upsonic.schemas.data_models import Document


class TestYAMLLoaderSimple(unittest.TestCase):
    """Simplified tests for YAMLLoader that match the actual implementation."""

    def setUp(self):
        """Set up test environment with sample YAML files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple YAML file
        self.simple_yaml = Path(self.temp_dir) / "simple.yaml"
        self.simple_yaml.write_text("""
name: John Doe
age: 30
city: New York
skills:
  - Python
  - JavaScript
  - SQL
""")

        # Create a multi-document YAML file
        self.multi_yaml = Path(self.temp_dir) / "multi.yaml"
        self.multi_yaml.write_text("""---
name: Alice
department: Engineering
---
name: Bob
department: Marketing
---
name: Charlie
department: Sales
""")

        # Create a nested YAML file
        self.nested_yaml = Path(self.temp_dir) / "nested.yaml"
        self.nested_yaml.write_text("""
company:
  name: Tech Corp
  employees:
    - name: John
      role: Developer
      skills: [Python, Docker]
    - name: Jane
      role: Designer
      skills: [Figma, Photoshop]
  locations:
    - city: New York
      country: USA
    - city: London
      country: UK
""")

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_yaml_loader_initialization(self):
        """Test YAML loader initialization with different configs."""
        # Test with default config
        config = YAMLLoaderConfig()
        loader = YAMLLoader(config)
        self.assertIsNotNone(loader)
        
        # Test with custom config
        custom_config = YAMLLoaderConfig(
            split_by_jq_query=".employees[]",
            content_synthesis_mode="json"
        )
        loader_custom = YAMLLoader(custom_config)
        self.assertEqual(loader_custom.config.split_by_jq_query, ".employees[]")

    def test_supported_extensions(self):
        """Test that YAML loader supports correct file extensions."""
        supported = YAMLLoader.get_supported_extensions()
        self.assertIn(".yaml", supported)
        self.assertIn(".yml", supported)
        self.assertEqual(len(supported), 2)

    def test_simple_yaml_loading(self):
        """Test loading a simple YAML file."""
        config = YAMLLoaderConfig()
        loader = YAMLLoader(config)
        
        documents = loader.load(str(self.simple_yaml))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))
        self.assertTrue(all(hasattr(doc, 'document_id') for doc in documents))
        self.assertTrue(all(doc.content.strip() for doc in documents))

    def test_multi_document_yaml_loading(self):
        """Test loading multi-document YAML files."""
        config = YAMLLoaderConfig()
        loader = YAMLLoader(config)
        
        documents = loader.load(str(self.multi_yaml))
        
        self.assertGreater(len(documents), 0)
        # Should handle multiple YAML documents
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_jq_query_extraction(self):
        """Test JQ query-based content extraction."""
        config = YAMLLoaderConfig(
            split_by_jq_query=".company.employees[]"
        )
        loader = YAMLLoader(config)
        
        documents = loader.load(str(self.nested_yaml))
        
        self.assertGreater(len(documents), 0)
        # Each employee should become a separate document
        for doc in documents:
            self.assertIsInstance(doc, Document)
            self.assertTrue(doc.content.strip())

    def test_serialization_options(self):
        """Test different serialization options."""
        # Test JSON serialization
        config_json = YAMLLoaderConfig(content_synthesis_mode="json")
        loader_json = YAMLLoader(config_json)
        docs_json = loader_json.load(str(self.simple_yaml))
        
        # Test YAML serialization
        config_yaml = YAMLLoaderConfig(content_synthesis_mode="canonical_yaml")
        loader_yaml = YAMLLoader(config_yaml)
        docs_yaml = loader_yaml.load(str(self.simple_yaml))
        
        self.assertGreater(len(docs_json), 0)
        self.assertGreater(len(docs_yaml), 0)

    def test_empty_source_handling(self):
        """Test handling of empty or invalid sources."""
        config = YAMLLoaderConfig()
        loader = YAMLLoader(config)
        
        # Test with empty list
        result = loader.load([])
        self.assertEqual(len(result), 0)
        
        # Test with non-existent file
        result = loader.load("/path/that/does/not/exist.yaml")
        self.assertEqual(len(result), 0)

    def test_batch_loading(self):
        """Test batch loading multiple YAML files."""
        config = YAMLLoaderConfig()
        loader = YAMLLoader(config)
        
        files = [str(self.simple_yaml), str(self.multi_yaml)]
        documents = loader.batch(files)
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_metadata_extraction(self):
        """Test metadata extraction and inclusion."""
        config = YAMLLoaderConfig(
            include_metadata=True
        )
        loader = YAMLLoader(config)
        
        documents = loader.load(str(self.simple_yaml))
        
        self.assertGreater(len(documents), 0)
        for doc in documents:
            self.assertIsInstance(doc.metadata, dict)
            # Should include file metadata
            self.assertIn('source', doc.metadata)

    def test_error_handling(self):
        """Test error handling for invalid YAML."""
        # Create invalid YAML file
        invalid_yaml = Path(self.temp_dir) / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [unclosed")
        
        config = YAMLLoaderConfig(error_handling="warn")
        loader = YAMLLoader(config)
        
        # Should handle error gracefully
        documents = loader.load(str(invalid_yaml))
        self.assertEqual(len(documents), 0)


if __name__ == "__main__":
    unittest.main()