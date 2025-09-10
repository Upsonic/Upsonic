import os
import tempfile
import json
import unittest
from pathlib import Path
from typing import List, Dict, Any

from upsonic.loaders.yaml import YAMLLoader
from upsonic.loaders.config import YAMLLoaderConfig
from upsonic.schemas.data_models import Document

class TestYAMLLoaderComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_files = []

    @classmethod
    def tearDownClass(cls):
        for file_path in cls.test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    @classmethod
    def create_test_yaml_file(cls, filename: str, content: str) -> str:
        file_path = os.path.join(cls.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        cls.test_files.append(file_path)
        return file_path

    def test_basic_yaml_loading(self):
        yaml_content = """
name: John Doe
age: 30
email: john.doe@example.com
address:
  street: 123 Main St
  city: New York
  zip: 10001
hobbies:
  - reading
  - swimming
  - coding
"""
        file_path = self.create_test_yaml_file("basic.yaml", yaml_content)
        loader = YAMLLoader()
        documents = loader.load(file_path)
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], Document)
        self.assertIn("John Doe", documents[0].content)
        self.assertEqual(documents[0].metadata["file_name"], "basic.yaml")
        self.assertEqual(documents[0].metadata["name"], "John Doe")
        self.assertEqual(documents[0].metadata["age"], 30)

    def test_multi_document_yaml(self):
        yaml_content = """
---
name: Document 1
type: config
version: 1.0
---
name: Document 2
type: data
items:
  - item1
  - item2
---
name: Document 3
type: metadata
description: "Third document"
"""
        file_path = self.create_test_yaml_file("multi_doc.yaml", yaml_content)
        loader = YAMLLoader()
        documents = loader.load(file_path)
        self.assertEqual(len(documents), 3)
        self.assertEqual(documents[0].metadata["document_index"], 0)
        self.assertEqual(documents[1].metadata["document_index"], 1)
        self.assertEqual(documents[2].metadata["document_index"], 2)

    def test_canonical_yaml_mode(self):
        yaml_content = """
database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret
"""
        file_path = self.create_test_yaml_file("canonical.yaml", yaml_content)
        config = YAMLLoaderConfig(content_synthesis_mode="canonical_yaml")
        loader = YAMLLoader(config)
        documents = loader.load(file_path)
        self.assertEqual(len(documents), 1)
        content = documents[0].content
        self.assertIn("database:", content)
        self.assertIn("host: localhost", content)
        self.assertIn("port: 5432", content)

    def test_json_mode(self):
        yaml_content = """
api:
  version: "2.1"
  endpoints:
    - /users
    - /posts
    - /comments
"""
        file_path = self.create_test_yaml_file("json_mode.yaml", yaml_content)
        config = YAMLLoaderConfig(content_synthesis_mode="json")
        loader = YAMLLoader(config)
        documents = loader.load(file_path)
        self.assertEqual(len(documents), 1)
        content = documents[0].content
        parsed_json = json.loads(content)
        self.assertEqual(parsed_json["api"]["version"], "2.1")
        self.assertEqual(len(parsed_json["api"]["endpoints"]), 3)

    def test_metadata_flattening(self):
        yaml_content = """
user:
  profile:
    name: Alice
    age: 25
  settings:
    theme: dark
    notifications: true
  preferences:
    languages:
      - en
      - es
"""
        file_path = self.create_test_yaml_file("flatten.yaml", yaml_content)
        config = YAMLLoaderConfig(flatten_metadata=True)
        loader = YAMLLoader(config)
        documents = loader.load(file_path)
        self.assertEqual(len(documents), 1)
        metadata = documents[0].metadata
        self.assertIn("user.profile.name", metadata)
        self.assertIn("user.settings.theme", metadata)
        self.assertIn("user.preferences.languages[0]", metadata)
        self.assertEqual(metadata["user.profile.name"], "Alice")
        self.assertEqual(metadata["user.settings.theme"], "dark")
        self.assertEqual(metadata["user.preferences.languages[0]"], "en")

    def test_custom_metadata(self):
        yaml_content = """
title: Test Document
content: "This is a test"
"""
        file_path = self.create_test_yaml_file("custom_meta.yaml", yaml_content)
        custom_metadata = {"source_type": "test", "category": "documentation", "priority": "high"}
        config = YAMLLoaderConfig(custom_metadata=custom_metadata)
        loader = YAMLLoader(config)
        documents = loader.load(file_path)
        self.assertEqual(len(documents), 1)
        metadata = documents[0].metadata
        self.assertEqual(metadata["source_type"], "test")
        self.assertEqual(metadata["category"], "documentation")
        self.assertEqual(metadata["priority"], "high")
        self.assertEqual(metadata["title"], "Test Document")

    def test_error_handling_ignore(self):
        invalid_yaml = """
name: John
age: 30
invalid: [unclosed list
"""
        file_path = self.create_test_yaml_file("invalid.yaml", invalid_yaml)
        config = YAMLLoaderConfig(error_handling="ignore")
        loader = YAMLLoader(config)
        documents = loader.load(file_path)
        self.assertEqual(len(documents), 0)

    def test_error_handling_warn(self):
        invalid_yaml = """
name: John
age: 30
invalid: [unclosed list
"""
        file_path = self.create_test_yaml_file("invalid_warn.yaml", invalid_yaml)
        config = YAMLLoaderConfig(error_handling="warn")
        loader = YAMLLoader(config)
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            documents = loader.load(file_path)
        output = f.getvalue()
        self.assertEqual(len(documents), 0)
        self.assertIn("Warning:", output)
        self.assertIn("YAMLLoader", output)

    def test_error_handling_raise(self):
        invalid_yaml = """
name: John
age: 30
invalid: [unclosed list
"""
        file_path = self.create_test_yaml_file("invalid_raise.yaml", invalid_yaml)
        config = YAMLLoaderConfig(error_handling="raise")
        loader = YAMLLoader(config)
        with self.assertRaises(Exception):
            loader.load(file_path)

    def test_file_size_limit(self):
        large_content = "data: " + "x" * 1000
        file_path = self.create_test_yaml_file("large.yaml", large_content)
        config = YAMLLoaderConfig(max_file_size=100, error_handling="raise")
        loader = YAMLLoader(config)
        with self.assertRaises(ValueError) as e:
            loader.load(file_path)
        self.assertIn("File size", str(e.exception))
        self.assertIn("exceeds limit", str(e.exception))

    def test_empty_content_skipping(self):
        yaml_content = """
empty_field: ""
null_field: null
valid_field: "has content"
"""
        file_path = self.create_test_yaml_file("empty.yaml", yaml_content)
        config = YAMLLoaderConfig(skip_empty_content=True)
        loader = YAMLLoader(config)
        documents = loader.load(file_path)
        self.assertEqual(len(documents), 1)
        self.assertIn("valid_field", documents[0].content)

    def test_supported_extensions(self):
        extensions = YAMLLoader.get_supported_extensions()
        self.assertIn(".yaml", extensions)
        self.assertIn(".yml", extensions)
        self.assertTrue(YAMLLoader.can_load("test.yaml"))
        self.assertTrue(YAMLLoader.can_load("test.yml"))
        self.assertFalse(YAMLLoader.can_load("test.txt"))

    def test_raw_yaml_string_loading(self):
        yaml_string = """
config:
  database:
    host: localhost
    port: 5432
  cache:
    enabled: true
    ttl: 3600
"""
        file_path = self.create_test_yaml_file("raw_string.yaml", yaml_string)
        loader = YAMLLoader()
        documents = loader.load(file_path)
        self.assertEqual(len(documents), 1)
        self.assertIn("localhost", documents[0].content)
        self.assertEqual(documents[0].metadata["config.database.host"], "localhost")

    def test_complex_nested_structure(self):
        yaml_content = """
application:
  name: "My App"
  version: "1.0.0"
  environment: production
  services:
    web:
      port: 8080
      replicas: 3
      resources:
        cpu: "500m"
        memory: "1Gi"
    database:
      type: postgresql
      host: db.example.com
      port: 5432
      credentials:
        username: app_user
        password: secret_password
  monitoring:
    enabled: true
    metrics:
      - cpu_usage
      - memory_usage
      - response_time
    alerts:
      cpu_threshold: 80
      memory_threshold: 90
"""
        file_path = self.create_test_yaml_file("complex.yaml", yaml_content)
        config = YAMLLoaderConfig(content_synthesis_mode="canonical_yaml", flatten_metadata=True, custom_metadata={"test_type": "complex_structure"})
        loader = YAMLLoader(config)
        documents = loader.load(file_path)
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIn("My App", doc.content)
        self.assertIn("web:", doc.content)
        self.assertIn("database:", doc.content)
        self.assertEqual(doc.metadata["application.name"], "My App")
        self.assertEqual(doc.metadata["application.services.web.port"], 8080)
        self.assertEqual(doc.metadata["application.services.database.type"], "postgresql")
        self.assertEqual(doc.metadata["application.monitoring.metrics[0]"], "cpu_usage")
        self.assertEqual(doc.metadata["test_type"], "complex_structure")

    def test_loader_statistics(self):
        yaml_content = """
test: data
"""
        file_path = self.create_test_yaml_file("stats.yaml", yaml_content)
        loader = YAMLLoader()
        initial_stats = loader.get_stats()
        self.assertEqual(initial_stats["total_files_processed"], 0)
        documents = loader.load(file_path)
        stats = loader.get_stats()
        self.assertEqual(stats["total_files_processed"], 1)
        self.assertEqual(stats["total_documents_created"], 1)
        self.assertEqual(stats["total_errors"], 0)
        self.assertGreater(stats["avg_processing_time"], 0)

if __name__ == "__main__":
    unittest.main()
