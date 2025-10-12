import unittest
import tempfile
import os
import json
from pathlib import Path
from upsonic.loaders.json import JSONLoader
from upsonic.loaders.config import JSONLoaderConfig
from upsonic.schemas.data_models import Document


class TestJSONLoaderSimple(unittest.TestCase):
    """Simplified tests for JSONLoader that match the actual implementation."""

    def setUp(self):
        """Set up test environment with sample JSON files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple JSON file
        self.simple_json = Path(self.temp_dir) / "simple.json"
        simple_data = {
            "name": "John Doe",
            "age": 30,
            "city": "New York",
            "skills": ["Python", "JavaScript", "SQL"]
        }
        self.simple_json.write_text(json.dumps(simple_data, indent=2))

        # Create a JSON array file
        self.array_json = Path(self.temp_dir) / "array.json"
        array_data = [
            {"name": "Alice", "department": "Engineering"},
            {"name": "Bob", "department": "Marketing"},
            {"name": "Charlie", "department": "Sales"}
        ]
        self.array_json.write_text(json.dumps(array_data, indent=2))

        # Create a nested JSON file
        self.nested_json = Path(self.temp_dir) / "nested.json"
        nested_data = {
            "company": {
                "name": "Tech Corp",
                "employees": [
                    {"name": "John", "role": "Developer"},
                    {"name": "Jane", "role": "Designer"}
                ],
                "locations": ["New York", "London"]
            }
        }
        self.nested_json.write_text(json.dumps(nested_data, indent=2))

        # Create a JSONL file
        self.jsonl_file = Path(self.temp_dir) / "data.jsonl"
        jsonl_data = [
            {"id": 1, "name": "Alice", "score": 95},
            {"id": 2, "name": "Bob", "score": 87},
            {"id": 3, "name": "Charlie", "score": 92}
        ]
        with open(self.jsonl_file, 'w') as f:
            for item in jsonl_data:
                f.write(json.dumps(item) + '\n')

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_json_loader_initialization(self):
        """Test JSON loader initialization with different configs."""
        config = JSONLoaderConfig()
        loader = JSONLoader(config)
        self.assertIsNotNone(loader)
        
        # Test with custom config
        custom_config = JSONLoaderConfig(
            mode="multi",
            record_selector=".employees[]",
            content_synthesis_mode="json"
        )
        loader_custom = JSONLoader(custom_config)
        self.assertEqual(loader_custom.config.record_selector, ".employees[]")

    def test_supported_extensions(self):
        """Test that JSON loader supports correct file extensions."""
        supported = JSONLoader.get_supported_extensions()
        self.assertIn(".json", supported)
        self.assertIn(".jsonl", supported)
        self.assertEqual(len(supported), 2)

    def test_simple_json_loading(self):
        """Test loading a simple JSON file."""
        config = JSONLoaderConfig()
        loader = JSONLoader(config)
        
        documents = loader.load(str(self.simple_json))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))
        self.assertTrue(all(hasattr(doc, 'document_id') for doc in documents))
        self.assertTrue(all(doc.content.strip() for doc in documents))

    def test_json_array_loading(self):
        """Test loading JSON arrays."""
        config = JSONLoaderConfig(mode="multi", record_selector=".[]")
        loader = JSONLoader(config)
        
        documents = loader.load(str(self.array_json))
        
        self.assertGreater(len(documents), 0)
        # Should create separate documents for each array element
        self.assertGreaterEqual(len(documents), 3)

    def test_jq_query_extraction(self):
        """Test JQ query-based content extraction."""
        config = JSONLoaderConfig(mode="multi", record_selector=".company.employees[]")
        loader = JSONLoader(config)
        
        documents = loader.load(str(self.nested_json))
        
        self.assertGreater(len(documents), 0)
        # Each employee should become a separate document
        for doc in documents:
            self.assertIsInstance(doc, Document)
            self.assertTrue(doc.content.strip())

    def test_serialization_options(self):
        """Test different serialization options."""
        # Test JSON serialization
        config_json = JSONLoaderConfig(content_synthesis_mode="json")
        loader_json = JSONLoader(config_json)
        docs_json = loader_json.load(str(self.simple_json))
        
        # Test text serialization
        config_text = JSONLoaderConfig(content_synthesis_mode="text")
        loader_text = JSONLoader(config_text)
        docs_text = loader_text.load(str(self.simple_json))
        
        self.assertGreater(len(docs_json), 0)
        self.assertGreater(len(docs_text), 0)

    def test_empty_source_handling(self):
        """Test handling of empty or invalid sources."""
        config = JSONLoaderConfig()
        loader = JSONLoader(config)
        
        # Test with empty list
        result = loader.load([])
        self.assertEqual(len(result), 0)
        
        # Test with non-existent file
        result = loader.load("/path/that/does/not/exist.json")
        self.assertEqual(len(result), 0)

    def test_batch_loading(self):
        """Test batch loading multiple JSON files."""
        config = JSONLoaderConfig()
        loader = JSONLoader(config)
        
        files = [str(self.simple_json), str(self.array_json)]
        documents = loader.batch(files)
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_error_handling(self):
        """Test error handling for invalid JSON."""
        # Create invalid JSON file
        invalid_json = Path(self.temp_dir) / "invalid.json"
        invalid_json.write_text('{"invalid": json, "missing": quotes}')
        
        config = JSONLoaderConfig(error_handling="warn")
        loader = JSONLoader(config)
        
        # Should handle error gracefully
        documents = loader.load(str(invalid_json))
        self.assertEqual(len(documents), 0)

    def test_jsonl_loading(self):
        """Test loading JSONL files."""
        config = JSONLoaderConfig(json_lines=True)
        loader = JSONLoader(config)
        
        documents = loader.load(str(self.jsonl_file))
        
        self.assertGreater(len(documents), 0)
        # Should create separate documents for each line
        self.assertGreaterEqual(len(documents), 3)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))


if __name__ == "__main__":
    unittest.main()