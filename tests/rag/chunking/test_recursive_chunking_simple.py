import unittest
import uuid
from upsonic.text_splitter.recursive import RecursiveChunker, RecursiveChunkingConfig
from upsonic.schemas.data_models import Document, Chunk


class TestRecursiveChunkingSimple(unittest.TestCase):
    """Simplified tests for RecursiveChunker that match the actual implementation."""

    def setUp(self):
        """Set up test documents."""
        self.basic_doc = Document(
            content="This is the first paragraph.\n\nThis is the second paragraph.\n\nThis is the third paragraph.",
            metadata={'source': 'test', 'type': 'basic'},
            document_id=str(uuid.uuid4())
        )
        
        self.structured_doc = Document(
            content="# Chapter 1\n\nThis is the first chapter.\n\n## Section 1.1\n\nThis is a section.\n\n## Section 1.2\n\nThis is another section.",
            metadata={'source': 'test', 'type': 'structured'},
            document_id=str(uuid.uuid4())
        )

    def test_basic_recursive_chunking(self):
        """Test basic recursive chunking functionality."""
        config = RecursiveChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = RecursiveChunker(config)
        
        chunks = chunker.chunk([self.basic_doc])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content.strip() for chunk in chunks))

    def test_different_separators(self):
        """Test recursive chunking with different separator configurations."""
        # Test with custom separators
        config = RecursiveChunkingConfig(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=80,
            chunk_overlap=15
        )
        chunker = RecursiveChunker(config)
        chunks = chunker.chunk([self.structured_doc])
        
        self.assertGreater(len(chunks), 0)

    def test_empty_document(self):
        """Test handling of empty documents."""
        empty_doc = Document(
            content="",
            metadata={'source': 'empty'},
            document_id=str(uuid.uuid4())
        )
        
        config = RecursiveChunkingConfig()
        chunker = RecursiveChunker(config)
        chunks = chunker.chunk([empty_doc])
        
        self.assertEqual(len(chunks), 0)

    def test_chunk_size_and_overlap(self):
        """Test different chunk sizes and overlaps."""
        # Small chunks
        config_small = RecursiveChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker_small = RecursiveChunker(config_small)
        chunks_small = chunker_small.chunk([self.basic_doc])
        
        # Large chunks
        config_large = RecursiveChunkingConfig(chunk_size=200, chunk_overlap=0)
        chunker_large = RecursiveChunker(config_large)
        chunks_large = chunker_large.chunk([self.basic_doc])
        
        self.assertGreater(len(chunks_small), 0)
        self.assertGreater(len(chunks_large), 0)
        # Smaller chunks should typically result in more chunks
        self.assertGreaterEqual(len(chunks_small), len(chunks_large))

    def test_regex_separators(self):
        """Test recursive chunking with regex separators."""
        config = RecursiveChunkingConfig(
            separators=[r"\n+", r"\. ", r" "],
            is_separator_regex=True,
            chunk_size=60
        )
        chunker = RecursiveChunker(config)
        chunks = chunker.chunk([self.basic_doc])
        
        self.assertGreater(len(chunks), 0)

    def test_keep_separator(self):
        """Test separator retention functionality."""
        config_keep = RecursiveChunkingConfig(
            separators=["\n\n", "\n"],
            keep_separator=True,
            chunk_size=50
        )
        chunker_keep = RecursiveChunker(config_keep)
        chunks_keep = chunker_keep.chunk([self.basic_doc])
        
        config_no_keep = RecursiveChunkingConfig(
            separators=["\n\n", "\n"],
            keep_separator=False,
            chunk_size=50
        )
        chunker_no_keep = RecursiveChunker(config_no_keep)
        chunks_no_keep = chunker_no_keep.chunk([self.basic_doc])
        
        self.assertGreater(len(chunks_keep), 0)
        self.assertGreater(len(chunks_no_keep), 0)


if __name__ == "__main__":
    unittest.main()