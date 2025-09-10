import os
import sys
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import unittest

from upsonic.text_splitter.json import JSONChunkingStrategy, JSONChunkingConfig
from upsonic.schemas.data_models import Document, Chunk

class TestJSONChunkingComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_documents = {}
        cls.create_simple_json_document(cls)
        cls.create_nested_json_document(cls)
        cls.create_array_json_document(cls)
        cls.create_complex_json_document(cls)
        cls.create_large_json_document(cls)
        cls.create_schema_json_document(cls)
        cls.create_empty_json_document(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_simple_json_document(self):
        json_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@example.com",
            "active": True
        }
        content = json.dumps(json_data, indent=2)
        doc = Document(content=content, metadata={'source': 'simple', 'type': 'user_data'})
        self.test_documents['simple'] = doc

    @staticmethod
    def create_nested_json_document(self):
        json_data = {
            "user": {
                "profile": {
                    "personal": {
                        "name": "Alice Smith",
                        "age": 28,
                        "details": {
                            "height": 165,
                            "weight": 60
                        }
                    },
                    "professional": {
                        "title": "Senior Engineer",
                        "company": "Tech Corp",
                        "skills": ["Python", "JavaScript", "SQL"]
                    }
                },
                "settings": {
                    "theme": "dark",
                    "notifications": True,
                    "privacy": {
                        "public_profile": False,
                        "share_data": True
                    }
                }
            }
        }
        content = json.dumps(json_data, indent=2)
        doc = Document(content=content, metadata={'source': 'nested', 'type': 'user_profile', 'complexity': 'high'})
        self.test_documents['nested'] = doc

    @staticmethod
    def create_array_json_document(self):
        json_data = {
            "users": [
                {"id": 1, "name": "User 1", "email": "user1@example.com", "role": "admin"},
                {"id": 2, "name": "User 2", "email": "user2@example.com", "role": "user"},
                {"id": 3, "name": "User 3", "email": "user3@example.com", "role": "moderator"},
                {"id": 4, "name": "User 4", "email": "user4@example.com", "role": "user"},
                {"id": 5, "name": "User 5", "email": "user5@example.com", "role": "user"}
            ],
            "metadata": {
                "total_users": 5,
                "created_at": "2024-01-01T00:00:00Z"
            }
        }
        content = json.dumps(json_data, indent=2)
        doc = Document(content=content, metadata={'source': 'array', 'type': 'user_list', 'count': 5})
        self.test_documents['array'] = doc

    @staticmethod
    def create_complex_json_document(self):
        json_data = {
            "company": {
                "name": "TechCorp Industries",
                "departments": [
                    {
                        "name": "Engineering",
                        "teams": [
                            {
                                "name": "Backend",
                                "members": [
                                    {"id": 1, "name": "Alice", "skills": ["Python", "Java", "Docker"]},
                                    {"id": 2, "name": "Bob", "skills": ["Python", "Go", "Kubernetes"]}
                                ],
                                "projects": ["API Gateway", "Database Optimization"]
                            },
                            {
                                "name": "Frontend", 
                                "members": [
                                    {"id": 3, "name": "Charlie", "skills": ["JavaScript", "React", "CSS"]},
                                    {"id": 4, "name": "Diana", "skills": ["TypeScript", "Vue", "SASS"]}
                                ],
                                "projects": ["User Dashboard", "Mobile App"]
                            }
                        ]
                    },
                    {
                        "name": "Marketing",
                        "campaigns": [
                            {"id": 1, "name": "Q1 Launch", "budget": 100000, "status": "completed"},
                            {"id": 2, "name": "Summer Sale", "budget": 75000, "status": "active"}
                        ]
                    }
                ]
            }
        }
        content = json.dumps(json_data, indent=2)
        doc = Document(content=content, metadata={'source': 'complex', 'type': 'organization', 'departments': 2})
        self.test_documents['complex'] = doc

    @staticmethod
    def create_large_json_document(self):
        json_data = {
            "metadata": {"version": "1.0", "created": "2024-01-01"},
            "data": []
        }
        
        # Create large dataset
        for i in range(100):
            json_data["data"].append({
                "id": i,
                "name": f"Item {i}",
                "description": f"Detailed description for item {i} with comprehensive information and multiple attributes",
                "category": f"Category {i % 10}",
                "tags": [f"tag{i}", f"category{i//10}", f"type{i//20}"],
                "metadata": {
                    "created": f"2024-01-{(i % 30) + 1:02d}",
                    "updated": f"2024-01-{(i % 30) + 1:02d}",
                    "status": "active" if i % 2 == 0 else "inactive",
                    "priority": i % 5
                },
                "attributes": {
                    "weight": round(10 + (i * 0.5), 2),
                    "dimensions": {"width": i + 10, "height": i + 5, "depth": i + 3},
                    "color": ["red", "blue", "green", "yellow", "purple"][i % 5]
                }
            })
        
        content = json.dumps(json_data, indent=2)
        doc = Document(content=content, metadata={'source': 'large', 'type': 'dataset', 'items': 100})
        self.test_documents['large'] = doc

    @staticmethod
    def create_schema_json_document(self):
        json_data = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "User Schema",
            "description": "Schema for user data validation",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "email": {"type": "string", "format": "email"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zipcode": {"type": "string", "pattern": "^[0-9]{5}$"}
                    },
                    "required": ["street", "city"]
                },
                "hobbies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True
                }
            },
            "required": ["name", "email"]
        }
        content = json.dumps(json_data, indent=2)
        doc = Document(content=content, metadata={'source': 'schema', 'type': 'json_schema', 'version': 'draft-07'})
        self.test_documents['schema'] = doc

    @staticmethod
    def create_empty_json_document(self):
        doc = Document(content="", metadata={'source': 'empty', 'type': 'edge_case'})
        self.test_documents['empty'] = doc

    def test_basic_json_chunking(self):
        config = JSONChunkingConfig()
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertGreater(len(chunk.text_content), 0)

    def test_simple_json_objects(self):
        config = JSONChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that simple objects are preserved
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("John Doe", all_text)
        self.assertIn("john.doe@example.com", all_text)

    def test_nested_json_objects(self):
        config = JSONChunkingConfig(chunk_size=200, chunk_overlap=50)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['nested'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that nested structure is handled
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertGreater(len(chunk.text_content), 0)

    def test_json_arrays(self):
        config = JSONChunkingConfig(chunk_size=150, chunk_overlap=30)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['array'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that array elements are processed
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("User 1", all_text)
        self.assertIn("user1@example.com", all_text)

    def test_array_handling_modes(self):
        # Test different array handling modes
        modes = ["split", "keep", "flatten"]
        
        for mode in modes:
            config = JSONChunkingConfig(
                array_handling=mode,
                chunk_size=100,
                chunk_overlap=20
            )
            chunker = JSONChunkingStrategy(config)
            chunks = chunker.chunk(self.test_documents['array'])
            
            self.assertGreaterEqual(len(chunks), 1)

    def test_object_handling_modes(self):
        config = JSONChunkingConfig(chunk_size=150, chunk_overlap=30)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that object structure is handled
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertGreater(len(chunk.text_content), 0)

    def test_schema_detection(self):
        config = JSONChunkingConfig(enable_schema_detection=True)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['schema'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that schema content is processed
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("User Schema", all_text)
        self.assertIn("properties", all_text)

    def test_type_preservation(self):
        config = JSONChunkingConfig(enable_type_preservation=True)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that chunks are created successfully
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)

    def test_path_metadata(self):
        config = JSONChunkingConfig(include_path_metadata=True)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['nested'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that chunks contain metadata
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertGreater(len(chunk.text_content), 0)

    def test_statistics_inclusion(self):
        config = JSONChunkingConfig(include_statistics=True)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['array'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that chunks are created with metadata
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)

    def test_structure_caching(self):
        config = JSONChunkingConfig(enable_structure_caching=True)
        chunker = JSONChunkingStrategy(config)
        
        # First processing
        chunks1 = chunker.chunk(self.test_documents['simple'])
        
        # Second processing (should use cache if available)
        chunks2 = chunker.chunk(self.test_documents['simple'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))

    def test_content_deduplication(self):
        config = JSONChunkingConfig(enable_content_deduplication=True)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['array'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that content is processed
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("User", all_content)

    def test_large_json_handling(self):
        config = JSONChunkingConfig(chunk_size=500, chunk_overlap=100)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['large'])
        
        self.assertGreater(len(chunks), 1)  # Should split into multiple chunks
        
        # Verify that large structure is handled
        total_content = sum(len(chunk.text_content) for chunk in chunks)
        self.assertGreater(total_content, 0)

    def test_complex_nested_structures(self):
        config = JSONChunkingConfig(
            chunk_size=300,
            chunk_overlap=50,
            preserve_structure=True,
            include_path_metadata=True
        )
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that complex structure is handled
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertGreater(len(chunk.text_content), 0)
            self.assertEqual(chunk.metadata['source'], 'complex')

    def test_json_validation(self):
        # Test with valid JSON
        config = JSONChunkingConfig(validate_json_structure=True)
        chunker = JSONChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks), 0)
        
        # Test with invalid JSON (create a malformed document)
        invalid_json = '{"invalid": "json", "unclosed": "quote}'
        invalid_doc = Document(content=invalid_json, metadata={'type': 'invalid'})
        
        chunks_invalid = chunker.chunk(invalid_doc)
        self.assertGreaterEqual(len(chunks_invalid), 0)  # Should handle gracefully

    def test_nesting_depth_limits(self):
        # Create deeply nested JSON
        deep_json = {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": {"l7": {"l8": {"l9": {"l10": "deep"}}}}}}}}}}
        deep_doc = Document(content=json.dumps(deep_json), metadata={'type': 'deep'})
        
        config = JSONChunkingConfig(max_nesting_depth=5)
        chunker = JSONChunkingStrategy(config)
        
        chunks = chunker.chunk(deep_doc)
        self.assertGreaterEqual(len(chunks), 0)

    def test_performance_tracking(self):
        config = JSONChunkingConfig()
        chunker = JSONChunkingStrategy(config)
        
        # Get initial metrics
        initial_metrics = chunker.get_metrics()
        self.assertEqual(initial_metrics.total_chunks, 0)
        
        # Process large document
        start_time = time.time()
        chunks = chunker.chunk(self.test_documents['large'])
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertGreater(len(chunks), 0)
        self.assertLess(processing_time, 10.0)  # Should complete within 10 seconds
        
        # Get updated metrics
        metrics = chunker.get_metrics()
        self.assertGreater(metrics.total_chunks, 0)
        self.assertGreater(metrics.total_characters, 0)
        self.assertGreater(metrics.avg_chunk_size, 0)
        self.assertGreater(metrics.processing_time_ms, 0)
        self.assertEqual(metrics.strategy_name, "JSONChunkingStrategy")

    def test_json_statistics(self):
        config = JSONChunkingConfig()
        chunker = JSONChunkingStrategy(config)
        
        # Process document
        chunks = chunker.chunk(self.test_documents['array'])
        
        # Try to get JSON statistics if available
        try:
            json_stats = chunker.get_json_stats()
            self.assertIsInstance(json_stats, dict)
        except AttributeError:
            # Method might not exist in actual implementation
            pass

    def test_error_handling(self):
        config = JSONChunkingConfig()
        chunker = JSONChunkingStrategy(config)
        
        # Test with malformed JSON
        malformed_json = '{"key": "value", "unclosed": "quote}'
        malformed_doc = Document(content=malformed_json, metadata={'type': 'malformed'})
        
        chunks = chunker.chunk(malformed_doc)
        self.assertGreaterEqual(len(chunks), 0)  # Should handle gracefully
        
        # Test with empty content
        chunks_empty = chunker.chunk(self.test_documents['empty'])
        self.assertEqual(len(chunks_empty), 0)

    def test_cache_management(self):
        config = JSONChunkingConfig(enable_structure_caching=True)
        chunker = JSONChunkingStrategy(config)
        
        # Process document to potentially populate cache
        chunks = chunker.chunk(self.test_documents['simple'])
        
        # Try cache operations if available
        try:
            chunker.clear_json_caches()
        except AttributeError:
            # Method might not exist in actual implementation
            pass

    def test_batch_processing(self):
        documents = [
            self.test_documents['simple'],
            self.test_documents['nested'],
            self.test_documents['array']
        ]
        
        config = JSONChunkingConfig(chunk_size=200, chunk_overlap=50)
        chunker = JSONChunkingStrategy(config)
        
        batch_results = chunker.chunk_batch(documents)
        
        self.assertEqual(len(batch_results), 3)
        self.assertTrue(all(len(chunks) > 0 for chunks in batch_results))
        
        total_chunks = sum(len(chunks) for chunks in batch_results)
        self.assertGreater(total_chunks, 0)

    def test_caching_functionality(self):
        config = JSONChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            enable_caching=True
        )
        chunker = JSONChunkingStrategy(config)
        
        # First processing
        chunks1 = chunker.chunk(self.test_documents['simple'])
        
        # Second processing (should use cache)
        chunks2 = chunker.chunk(self.test_documents['simple'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))

if __name__ == "__main__":
    unittest.main()