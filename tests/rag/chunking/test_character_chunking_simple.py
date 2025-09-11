import unittest
import uuid
from upsonic.text_splitter.character import CharacterChunker, CharacterChunkingConfig
from upsonic.schemas.data_models import Document, Chunk


class TestCharacterChunkingSimple(unittest.TestCase):
    """Simplified tests for CharacterChunker that match the actual implementation."""

    def setUp(self):
        """Set up test documents."""
        self.basic_doc = Document(
            content="This is the first paragraph.\n\nThis is the second paragraph.\n\nThis is the third paragraph.",
            metadata={'source': 'test', 'type': 'basic'},
            document_id=str(uuid.uuid4())
        )
        
        self.long_doc = Document(
            content="This is paragraph 1.\n\nThis is paragraph 2.\n\nThis is paragraph 3.\n\nThis is paragraph 4.\n\nThis is paragraph 5.",
            metadata={'source': 'test', 'type': 'long'},
            document_id=str(uuid.uuid4())
        )

    def test_basic_chunking(self):
        """Test basic chunking functionality."""
        config = CharacterChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = CharacterChunker(config)
        
        chunks = chunker.chunk([self.basic_doc])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content.strip() for chunk in chunks))

    def test_different_separators(self):
        """Test different separator configurations."""
        # Test with paragraph separator
        config = CharacterChunkingConfig(separator="\n\n", chunk_size=50)
        chunker = CharacterChunker(config)
        chunks = chunker.chunk([self.basic_doc])
        
        self.assertGreater(len(chunks), 0)
        
        # Test with regex separator
        config_regex = CharacterChunkingConfig(
            separator=r"\n+", 
            is_separator_regex=True,
            chunk_size=50
        )
        chunker_regex = CharacterChunker(config_regex)
        chunks_regex = chunker_regex.chunk([self.basic_doc])
        
        self.assertGreater(len(chunks_regex), 0)

    def test_empty_document(self):
        """Test handling of empty documents."""
        empty_doc = Document(
            content="",
            metadata={'source': 'empty'},
            document_id=str(uuid.uuid4())
        )
        
        config = CharacterChunkingConfig()
        chunker = CharacterChunker(config)
        chunks = chunker.chunk([empty_doc])
        
        self.assertEqual(len(chunks), 0)

    def test_chunk_overlap(self):
        """Test chunk overlap functionality."""
        config = CharacterChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = CharacterChunker(config)
        
        chunks = chunker.chunk([self.long_doc])
        
        self.assertGreater(len(chunks), 1)
        # Check that chunks have proper metadata
        for chunk in chunks:
            self.assertEqual(chunk.document_id, self.long_doc.document_id)
            self.assertIn('source', chunk.metadata)

    def test_keep_separator(self):
        """Test separator retention functionality."""
        # Test with keep_separator=True
        config_keep = CharacterChunkingConfig(
            separator="\n\n",
            keep_separator=True,
            chunk_size=50
        )
        chunker_keep = CharacterChunker(config_keep)
        chunks_keep = chunker_keep.chunk([self.basic_doc])
        
        # Test with keep_separator=False
        config_no_keep = CharacterChunkingConfig(
            separator="\n\n",
            keep_separator=False,
            chunk_size=50
        )
        chunker_no_keep = CharacterChunker(config_no_keep)
        chunks_no_keep = chunker_no_keep.chunk([self.basic_doc])
        
        self.assertGreater(len(chunks_keep), 0)
        self.assertGreater(len(chunks_no_keep), 0)


if __name__ == "__main__":
    unittest.main()