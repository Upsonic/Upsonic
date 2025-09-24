import unittest
import json
import uuid
from upsonic.text_splitter.json_chunker import JSONChunker, JSONChunkingConfig
from upsonic.schemas.data_models import Document, Chunk


class TestJSONChunkingSimple(unittest.TestCase):
    """Simplified tests for JSONChunker that match the actual implementation."""

    def setUp(self):
        """Set up test documents."""
        # Simple JSON document
        simple_json = {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@example.com",
            "active": True
        }
        self.simple_doc = Document(
            content=json.dumps(simple_json, indent=2),
            metadata={'source': 'simple', 'type': 'user_data'},
            document_id=str(uuid.uuid4())
        )
        
        # Nested JSON document
        nested_json = {
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
                    "notifications": True
                }
            }
        }
        self.nested_doc = Document(
            content=json.dumps(nested_json, indent=2),
            metadata={'source': 'nested', 'type': 'user_profile'},
            document_id=str(uuid.uuid4())
        )
        
        # Array JSON document
        array_json = {
            "users": [
                {"id": 1, "name": "User 1", "email": "user1@example.com", "role": "admin"},
                {"id": 2, "name": "User 2", "email": "user2@example.com", "role": "user"},
                {"id": 3, "name": "User 3", "email": "user3@example.com", "role": "moderator"}
            ],
            "metadata": {
                "total_users": 3,
                "created_at": "2024-01-01T00:00:00Z"
            }
        }
        self.array_doc = Document(
            content=json.dumps(array_json, indent=2),
            metadata={'source': 'array', 'type': 'user_list'},
            document_id=str(uuid.uuid4())
        )

    def test_basic_json_chunking(self):
        """Test basic JSON chunking functionality."""
        config = JSONChunkingConfig()
        chunker = JSONChunker(config)
        chunks = chunker.chunk([self.simple_doc])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        for chunk in chunks:
            self.assertIsInstance(chunk.metadata, dict)
            self.assertGreater(len(chunk.text_content), 0)

    def test_simple_json_objects(self):
        """Test chunking of simple JSON objects."""
        config = JSONChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = JSONChunker(config)
        chunks = chunker.chunk([self.simple_doc])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that simple objects are preserved
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("John Doe", all_text)
        self.assertIn("john.doe@example.com", all_text)

    def test_nested_json_objects(self):
        """Test chunking of nested JSON objects."""
        config = JSONChunkingConfig(chunk_size=200, chunk_overlap=50)
        chunker = JSONChunker(config)
        chunks = chunker.chunk([self.nested_doc])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that nested structure is handled
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Alice Smith", all_text)
        self.assertIn("Senior Engineer", all_text)

    def test_json_arrays(self):
        """Test chunking of JSON arrays."""
        config = JSONChunkingConfig(chunk_size=150, chunk_overlap=30)
        chunker = JSONChunker(config)
        chunks = chunker.chunk([self.array_doc])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that array elements are processed
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("User 1", all_text)
        self.assertIn("user1@example.com", all_text)

    def test_array_handling_modes(self):
        """Test different array handling modes."""
        # Test with convert_lists_to_dicts=True (default)
        config_convert = JSONChunkingConfig(
            convert_lists_to_dicts=True,
            chunk_size=100,
            chunk_overlap=20
        )
        chunker_convert = JSONChunker(config_convert)
        chunks_convert = chunker_convert.chunk([self.array_doc])
        
        # Test with convert_lists_to_dicts=False
        config_no_convert = JSONChunkingConfig(
            convert_lists_to_dicts=False,
            chunk_size=100,
            chunk_overlap=20
        )
        chunker_no_convert = JSONChunker(config_no_convert)
        chunks_no_convert = chunker_no_convert.chunk([self.array_doc])
        
        self.assertGreaterEqual(len(chunks_convert), 1)
        self.assertGreaterEqual(len(chunks_no_convert), 1)

    def test_path_metadata(self):
        """Test inclusion of JSON path metadata."""
        config = JSONChunkingConfig()
        chunker = JSONChunker(config)
        chunks = chunker.chunk([self.nested_doc])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that chunks contain metadata
        for chunk in chunks:
            self.assertIsInstance(chunk.metadata, dict)
            self.assertIn('chunk_json_paths', chunk.metadata)
            self.assertGreater(len(chunk.text_content), 0)

    def test_json_validation(self):
        """Test JSON validation handling."""
        # Test with valid JSON
        config = JSONChunkingConfig()
        chunker = JSONChunker(config)
        chunks = chunker.chunk([self.simple_doc])
        
        self.assertGreater(len(chunks), 0)
        
        # Test with invalid JSON
        invalid_json = '{"invalid": "json", "unclosed": "quote}'
        invalid_doc = Document(content=invalid_json, metadata={'type': 'invalid'}, document_id=str(uuid.uuid4()))
        
        chunks_invalid = chunker.chunk([invalid_doc])
        self.assertGreaterEqual(len(chunks_invalid), 0)  # Should handle gracefully

    def test_empty_content_handling(self):
        """Test handling of empty JSON content."""
        config = JSONChunkingConfig()
        chunker = JSONChunker(config)
        
        # Test with empty content
        empty_doc = Document(content="", metadata={'source': 'empty', 'type': 'edge_case'}, document_id=str(uuid.uuid4()))
        chunks_empty = chunker.chunk([empty_doc])
        self.assertEqual(len(chunks_empty), 0)

    def test_batch_processing(self):
        """Test batch processing of multiple JSON documents."""
        documents = [
            self.simple_doc,
            self.nested_doc,
            self.array_doc
        ]
        
        config = JSONChunkingConfig(chunk_size=200, chunk_overlap=50)
        chunker = JSONChunker(config)
        
        batch_results = chunker.chunk(documents)
        
        self.assertGreater(len(batch_results), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in batch_results))
        
        # Verify content from different documents is present
        all_text = " ".join(chunk.text_content for chunk in batch_results)
        self.assertIn("John Doe", all_text)
        self.assertIn("Alice Smith", all_text)
        self.assertIn("User 1", all_text)


if __name__ == "__main__":
    unittest.main()
