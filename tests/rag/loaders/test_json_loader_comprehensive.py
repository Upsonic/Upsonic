import os
import sys
import tempfile
import time
import json
import gzip
import bz2
import lzma
import unittest
from pathlib import Path

from upsonic.loaders.json import JSONLoader, filter_by_content_length, filter_by_metadata_key, transform_add_timestamp, transform_content_cleanup
from upsonic.loaders.config import JSONLoaderConfig
from upsonic.schemas.data_models import Document

class TestJSONLoaderComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_files = {}
        cls.create_basic_json(cls)
        cls.create_array_json(cls)
        cls.create_nested_json(cls)
        cls.create_jsonl_file(cls)
        cls.create_compressed_json(cls)
        cls.create_large_json(cls)
        cls.create_malformed_json(cls)
        cls.create_empty_json(cls)
        cls.create_schema_json(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_basic_json(self):
        file_path = os.path.join(self.temp_dir, "basic_test.json")
        data = {
            "title": "Basic Test Document",
            "description": "A simple JSON document for testing",
            "author": "Test Author",
            "version": "1.0",
            "features": ["parsing", "validation", "transformation"],
            "metadata": {
                "created": "2024-01-01",
                "updated": "2024-01-15",
                "tags": ["test", "json", "loader"]
            },
            "content": "This is the main content of the document."
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self.test_files['basic'] = file_path

    @staticmethod
    def create_array_json(self):
        file_path = os.path.join(self.temp_dir, "array_test.json")
        data = [
            {"id": 1, "name": "John Doe", "email": "john@example.com", "age": 30, "department": "Engineering"},
            {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "age": 25, "department": "Marketing"},
            {"id": 3, "name": "Bob Johnson", "email": "bob@example.com", "age": 35, "department": "Sales"}
        ]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self.test_files['array'] = file_path

    @staticmethod
    def create_nested_json(self):
        file_path = os.path.join(self.temp_dir, "nested_test.json")
        data = {
            "company": {
                "name": "Test Company",
                "founded": 2020,
                "departments": {
                    "engineering": {
                        "head": "Alice Brown",
                        "employees": [
                            {"name": "John Doe", "role": "Senior Developer"},
                            {"name": "Jane Smith", "role": "DevOps Engineer"}
                        ],
                        "projects": {
                            "project_a": {"name": "Web Application", "status": "active", "budget": 100000},
                            "project_b": {"name": "Mobile App", "status": "planning", "budget": 75000}
                        }
                    },
                    "marketing": {
                        "head": "Charlie Wilson",
                        "employees": [
                            {"name": "David Lee", "role": "Marketing Manager"},
                            {"name": "Eva Garcia", "role": "Content Writer"}
                        ],
                        "campaigns": [
                            {"name": "Summer Sale", "budget": 50000, "duration": "3 months"},
                            {"name": "Brand Awareness", "budget": 30000, "duration": "6 months"}
                        ]
                    }
                },
                "financials": {
                    "revenue": 1000000,
                    "expenses": 750000,
                    "profit": 250000,
                    "quarterly_data": [
                        {"q1": 200000, "q2": 250000, "q3": 300000, "q4": 250000}
                    ]
                }
            }
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self.test_files['nested'] = file_path

    @staticmethod
    def create_jsonl_file(self):
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        data = [
            {"id": 1, "message": "First log entry", "level": "INFO", "timestamp": "2024-01-01T10:00:00Z"},
            {"id": 2, "message": "Second log entry", "level": "WARNING", "timestamp": "2024-01-01T10:01:00Z"},
            {"id": 3, "message": "Third log entry", "level": "ERROR", "timestamp": "2024-01-01T10:02:00Z"},
            {"id": 4, "message": "Fourth log entry", "level": "DEBUG", "timestamp": "2024-01-01T10:03:00Z"},
            {"id": 5, "message": "Fifth log entry", "level": "INFO", "timestamp": "2024-01-01T10:04:00Z"}
        ]
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        self.test_files['jsonl'] = file_path

    @staticmethod
    def create_compressed_json(self):
        gzip_path = os.path.join(self.temp_dir, "compressed_test.json.gz")
        data = {
            "compressed": True,
            "type": "gzip",
            "content": "This is compressed JSON data",
            "size": 1024,
            "metadata": {"compression_ratio": 0.3, "original_size": 3000}
        }
        with gzip.open(gzip_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self.test_files['gzip'] = gzip_path
        bz2_path = os.path.join(self.temp_dir, "compressed_test.json.bz2")
        with bz2.open(bz2_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self.test_files['bz2'] = bz2_path

    @staticmethod
    def create_large_json(self):
        file_path = os.path.join(self.temp_dir, "large_test.json")
        data = {"title": "Large Test Document", "description": "A large JSON document for performance testing", "items": []}
        for i in range(1000):
            item = {
                "id": i,
                "name": f"Item {i}",
                "category": f"Category {i % 10}",
                "price": 10.99 + (i % 100),
                "stock": 100 - (i % 50),
                "description": f"This is item {i} with detailed description and specifications.",
                "tags": [f"tag{j}" for j in range(i % 5)],
                "metadata": {"created": "2024-01-01", "updated": "2024-01-15", "views": i * 10, "rating": (i % 5) + 1}
            }
            data["items"].append(item)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self.test_files['large'] = file_path

    @staticmethod
    def create_malformed_json(self):
        file_path = os.path.join(self.temp_dir, "malformed_test.json")
        malformed_content = """{
            "title": "Malformed Test Document",
            "description": "This JSON has issues",
            "items": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2",  // Missing closing brace
                {"id": 3, "name": "Item 3"}
            ],
            "metadata": {
                "created": "2024-01-01"
                // Missing closing brace
        }"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(malformed_content)
        self.test_files['malformed'] = file_path

    @staticmethod
    def create_empty_json(self):
        file_path = os.path.join(self.temp_dir, "empty_test.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("{}")
        self.test_files['empty'] = file_path

    @staticmethod
    def create_schema_json(self):
        file_path = os.path.join(self.temp_dir, "schema_test.json")
        data = {
            "name": "Schema Test Document",
            "type": "test",
            "version": "1.0.0",
            "properties": {
                "title": "Test Title",
                "description": "Test Description",
                "required": ["title", "description"],
                "optional": ["tags", "metadata"]
            },
            "items": [
                {"id": 1, "value": "test1"},
                {"id": 2, "value": "test2"}
            ]
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self.test_files['schema'] = file_path

    def test_basic_loading(self):
        """Test basic JSON loading functionality."""
        loader = JSONLoader()
        documents = loader.load(self.test_files['basic'])
        
        self.assertEqual(len(documents), 1, f"Expected 1 document, got {len(documents)}")
        
        doc = documents[0]
        self.assertTrue(doc.content, "Document should have content")
        self.assertIn('Basic Test Document', doc.content, "Should contain the title")
        self.assertIn('Test Author', doc.content, "Should contain the author")
        self.assertIn('parsing', doc.content, "Should contain array elements")
        
        # Check metadata
        self.assertEqual(doc.metadata['title'], 'Basic Test Document', "Should have title in metadata")
        self.assertEqual(doc.metadata['author'], 'Test Author', "Should have author in metadata")
        self.assertIn('loader_type', doc.metadata, "Should have loader type")

    def test_array_processing(self):
        """Test processing of JSON arrays."""
        loader = JSONLoader()
        documents = loader.load(self.test_files['array'])
        
        self.assertEqual(len(documents), 3, f"Expected 3 documents, got {len(documents)}")
        
        # Check first document
        first_doc = documents[0]
        self.assertIn('John Doe', first_doc.content, "Should contain first person's name")
        self.assertEqual(first_doc.metadata['name'], 'John Doe', "Should have name in metadata")
        self.assertEqual(first_doc.metadata['email'], 'john@example.com', "Should have email in metadata")
        
        # Check second document
        second_doc = documents[1]
        self.assertIn('Jane Smith', second_doc.content, "Should contain second person's name")
        self.assertEqual(second_doc.metadata['name'], 'Jane Smith', "Should have name in metadata")

    def test_nested_structure(self):
        """Test processing of nested JSON structures."""
        loader = JSONLoader()
        documents = loader.load(self.test_files['nested'])
        
        self.assertEqual(len(documents), 1, f"Expected 1 document, got {len(documents)}")
        
        doc = documents[0]
        content = doc.content
        
        # Check nested content
        self.assertIn('Test Company', content, "Should contain company name")
        self.assertIn('Alice Brown', content, "Should contain department head")
        self.assertIn('John Doe', content, "Should contain employee name")
        self.assertIn('Web Application', content, "Should contain project name")
        
    def test_jsonl_processing(self):
        """Test JSONL (JSON Lines) processing."""
        config = JSONLoaderConfig(is_jsonl=True)
        loader = JSONLoader(config)
        documents = loader.load(self.test_files['jsonl'])
        
        self.assertEqual(len(documents), 5, f"Expected 5 documents, got {len(documents)}")
        
        # Check first document
        first_doc = documents[0]
        self.assertIn('First log entry', first_doc.content, "Should contain first log message")
        self.assertEqual(first_doc.metadata['level'], 'INFO', "Should have log level in metadata")
        
        # Check error level document
        error_doc = documents[2]  # Third document (index 2)
        self.assertIn('Third log entry', error_doc.content, "Should contain error log message")
        self.assertEqual(error_doc.metadata['level'], 'ERROR', "Should have ERROR level in metadata")
        
    def test_compression_support(self):
        """Test compressed JSON file loading."""
        # Test gzip
        loader = JSONLoader()
        gzip_docs = loader.load(self.test_files['gzip'])
        
        self.assertEqual(len(gzip_docs), 1, f"Expected 1 gzip document, got {len(gzip_docs)}")
        gzip_doc = gzip_docs[0]
        self.assertIn('compressed', gzip_doc.content, "Should contain compressed content")
        
        # Test bzip2
        bz2_docs = loader.load(self.test_files['bz2'])
        
        self.assertEqual(len(bz2_docs), 1, f"Expected 1 bzip2 document, got {len(bz2_docs)}")
        bz2_doc = bz2_docs[0]
        self.assertIn('compressed', bz2_doc.content, "Should contain compressed content")
        
    def test_content_key_extraction(self):
        """Test content key extraction."""
        config = JSONLoaderConfig(content_key="content")
        loader = JSONLoader(config)
        documents = loader.load(self.test_files['basic'])
        
        self.assertEqual(len(documents), 1, f"Expected 1 document, got {len(documents)}")
        
        doc = documents[0]
        # Content should be just the value of the "content" key
        self.assertEqual(doc.content, "This is the main content of the document.", "Should extract content key")
        
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON."""
        # Test with ignore mode
        config = JSONLoaderConfig(error_handling="ignore")
        loader = JSONLoader(config)
        documents = loader.load(self.test_files['malformed'])
        
        # Should return empty list due to malformed JSON
        self.assertEqual(len(documents), 0, f"Expected 0 documents for malformed JSON, got {len(documents)}")
        
    def test_empty_json_handling(self):
        """Test handling of empty JSON."""
        loader = JSONLoader()
        documents = loader.load(self.test_files['empty'])
        
        # Empty JSON should still create a document
        self.assertEqual(len(documents), 1, f"Expected 1 document, got {len(documents)}")
        
        doc = documents[0]
        self.assertEqual(doc.content, "{}", "Should contain empty JSON object")

if __name__ == "__main__":
    unittest.main()
