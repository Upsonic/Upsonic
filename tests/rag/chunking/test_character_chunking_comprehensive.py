import os
import sys
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import unittest

from upsonic.text_splitter.character import CharacterChunkingStrategy, CharacterChunkingConfig
from upsonic.text_splitter.base import ChunkingMode
from upsonic.schemas.data_models import Document, Chunk

class TestCharacterChunkingComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_documents = {}
        cls.create_basic_document(cls)
        cls.create_paragraph_document(cls)
        cls.create_sentence_document(cls)
        cls.create_complex_structure_document(cls)
        cls.create_large_document(cls)
        cls.create_empty_document(cls)
        cls.create_whitespace_document(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_basic_document(self):
        content = """This is the first paragraph with some basic content.

This is the second paragraph with more detailed information.

This is the third paragraph that wraps up the content nicely."""
        doc = Document(content=content, metadata={'source': 'basic', 'type': 'test'})
        self.test_documents['basic'] = doc

    @staticmethod
    def create_paragraph_document(self):
        content = """First paragraph content here with detailed information about the topic.

Second paragraph with different content that explores another aspect.

Third paragraph concludes the text with final thoughts and summary."""
        doc = Document(content=content, metadata={'source': 'paragraphs', 'type': 'structured'})
        self.test_documents['paragraphs'] = doc

    @staticmethod
    def create_sentence_document(self):
        content = """This is the first sentence. This is the second sentence with more content. This is the third sentence that provides additional context. This is the fourth sentence. This is the fifth sentence with comprehensive details."""
        doc = Document(content=content, metadata={'source': 'sentences', 'type': 'continuous'})
        self.test_documents['sentences'] = doc

    @staticmethod
    def create_complex_structure_document(self):
        content = """# Chapter 1: Introduction

This is the introduction chapter that provides an overview of the topic.

## Section 1.1: Background

The background section contains historical context and previous research findings.

### Subsection 1.1.1: Key Concepts

Key concepts are defined here with detailed explanations.

## Section 1.2: Methodology

The methodology section describes the approach used in this research.

# Chapter 2: Analysis

This chapter presents the analysis of the collected data."""
        doc = Document(content=content, metadata={'source': 'complex', 'type': 'structured', 'chapters': 2})
        self.test_documents['complex'] = doc

    @staticmethod
    def create_large_document(self):
        content_parts = []
        for i in range(100):
            content_parts.append(f"This is paragraph number {i+1} with content that should be chunked properly. " +
                               f"It contains multiple sentences to test the chunking behavior. " +
                               f"The paragraph number is {i+1} for reference purposes.")
        content = "\n\n".join(content_parts)
        doc = Document(content=content, metadata={'source': 'large', 'type': 'performance', 'paragraphs': 100})
        self.test_documents['large'] = doc

    @staticmethod
    def create_empty_document(self):
        doc = Document(content="", metadata={'source': 'empty', 'type': 'edge_case'})
        self.test_documents['empty'] = doc

    @staticmethod
    def create_whitespace_document(self):
        content = "   First part   \n\n   Second part   \n\n   Third part   "
        doc = Document(content=content, metadata={'source': 'whitespace', 'type': 'edge_case'})
        self.test_documents['whitespace'] = doc

    def test_basic_chunking(self):
        config = CharacterChunkingConfig(chunk_size=200, chunk_overlap=50)
        chunker = CharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['basic'])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata['chunk_index'], i)
            self.assertEqual(chunk.metadata['chunking_strategy'], 'CharacterChunkingStrategy')
            self.assertEqual(chunk.metadata['chunk_size_target'], 200)
            self.assertEqual(chunk.metadata['chunk_overlap'], 50)

    def test_chunk_sizes_and_overlap(self):
        # Test small chunks with overlap
        config_small = CharacterChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker_small = CharacterChunkingStrategy(config_small)
        chunks_small = chunker_small.chunk(self.test_documents['basic'])
        
        self.assertGreater(len(chunks_small), 1)
        for chunk in chunks_small:
            self.assertLessEqual(len(chunk.text_content), 70)  # Allow more flexibility for chunking
            self.assertEqual(chunk.metadata['chunk_overlap'], 10)
        
        # Test large chunks with no overlap
        config_large = CharacterChunkingConfig(chunk_size=500, chunk_overlap=0)
        chunker_large = CharacterChunkingStrategy(config_large)
        chunks_large = chunker_large.chunk(self.test_documents['basic'])
        
        self.assertGreaterEqual(len(chunks_large), 1)
        for chunk in chunks_large:
            self.assertEqual(chunk.metadata['chunk_overlap'], 0)

    def test_separator_configurations(self):
        # Test paragraph separators
        config_para = CharacterChunkingConfig(
            chunk_size=100, 
            chunk_overlap=20,
            separator="\n\n"
        )
        chunker_para = CharacterChunkingStrategy(config_para)
        chunks_para = chunker_para.chunk(self.test_documents['paragraphs'])
        
        self.assertGreaterEqual(len(chunks_para), 2)
        
        # Test sentence separators
        config_sent = CharacterChunkingConfig(
            chunk_size=80,
            chunk_overlap=20,
            separator=". "
        )
        chunker_sent = CharacterChunkingStrategy(config_sent)
        chunks_sent = chunker_sent.chunk(self.test_documents['sentences'])
        
        self.assertGreaterEqual(len(chunks_sent), 2)
        
        # Test multiple separators
        config_multi = CharacterChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            separator="\n\n",
            multiple_separators=["\n", ". "]
        )
        chunker_multi = CharacterChunkingStrategy(config_multi)
        chunks_multi = chunker_multi.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks_multi), 0)

    def test_whitespace_handling(self):
        # Test with strip_whitespace=True
        config_strip = CharacterChunkingConfig(
            chunk_size=50,
            chunk_overlap=10,
            strip_whitespace=True,
            skip_empty_splits=True
        )
        chunker_strip = CharacterChunkingStrategy(config_strip)
        chunks_strip = chunker_strip.chunk(self.test_documents['whitespace'])
        
        for chunk in chunks_strip:
            self.assertFalse(chunk.text_content.startswith(' '))
            self.assertFalse(chunk.text_content.endswith(' '))
            self.assertEqual(chunk.text_content.strip(), chunk.text_content)
        
        # Test with strip_whitespace=False
        config_preserve = CharacterChunkingConfig(
            chunk_size=50,
            chunk_overlap=10,
            strip_whitespace=False,
            skip_empty_splits=False
        )
        chunker_preserve = CharacterChunkingStrategy(config_preserve)
        chunks_preserve = chunker_preserve.chunk(self.test_documents['whitespace'])
        
        self.assertGreater(len(chunks_preserve), 0)

    def test_empty_content_handling(self):
        config = CharacterChunkingConfig(skip_empty_splits=True)
        chunker = CharacterChunkingStrategy(config)
        
        # Test completely empty content
        chunks_empty = chunker.chunk(self.test_documents['empty'])
        self.assertEqual(len(chunks_empty), 0)
        
        # Test whitespace-only handling
        whitespace_doc = Document(content="   \n\n   \t   ", metadata={})
        chunks_whitespace = chunker.chunk(whitespace_doc)
        self.assertEqual(len(chunks_whitespace), 0)

    def test_chunking_modes(self):
        # Test standard mode
        config_standard = CharacterChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            mode=ChunkingMode.STANDARD
        )
        chunker_standard = CharacterChunkingStrategy(config_standard)
        chunks_standard = chunker_standard.chunk(self.test_documents['basic'])
        
        self.assertGreater(len(chunks_standard), 0)
        self.assertTrue(all(chunk.metadata['chunking_mode'] == 'standard' for chunk in chunks_standard))
        
        # Test aggressive mode
        config_aggressive = CharacterChunkingConfig(
            chunk_size=50,
            chunk_overlap=10,
            mode=ChunkingMode.AGGRESSIVE
        )
        chunker_aggressive = CharacterChunkingStrategy(config_aggressive)
        chunks_aggressive = chunker_aggressive.chunk(self.test_documents['basic'])
        
        self.assertGreater(len(chunks_aggressive), 0)
        self.assertTrue(all(chunk.metadata['chunking_mode'] == 'aggressive' for chunk in chunks_aggressive))

    def test_fallback_mechanisms(self):
        # Test with text that has no separators
        no_sep_content = "ThisIsOneLongWordWithoutAnySeparatorsThatShouldBeSplitByLength"
        no_sep_doc = Document(content=no_sep_content, metadata={'type': 'no_separators'})
        
        config = CharacterChunkingConfig(
            chunk_size=20,
            chunk_overlap=5,
            separator="\n\n",
            fallback_to_length=True
        )
        chunker = CharacterChunkingStrategy(config)
        chunks = chunker.chunk(no_sep_doc)
        
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text_content), 25)  # Allow flexibility

    def test_metadata_enrichment(self):
        custom_metadata = {
            "test_mode": "metadata_enrichment",
            "custom_field": "custom_value"
        }
        
        config = CharacterChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            add_chunk_index=True,
            add_chunk_count=True,
            add_position_info=True,
            custom_metadata=custom_metadata
        )
        chunker = CharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['basic'])
        
        self.assertGreater(len(chunks), 0)
        
        for i, chunk in enumerate(chunks):
            # Check original document metadata is preserved
            self.assertEqual(chunk.metadata['source'], 'basic')
            self.assertEqual(chunk.metadata['type'], 'test')
            
            # Check chunk-specific metadata
            self.assertEqual(chunk.metadata['chunk_index'], i)
            self.assertEqual(chunk.metadata['total_chunks'], len(chunks))
            self.assertIn('start_position', chunk.metadata)
            self.assertIn('end_position', chunk.metadata)
            self.assertEqual(chunk.metadata['test_mode'], 'metadata_enrichment')
            self.assertEqual(chunk.metadata['custom_field'], 'custom_value')

    def test_separator_optimization(self):
        config = CharacterChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            separator="\n",
            multiple_separators=["\n\n", ". "],
            enable_separator_optimization=True
        )
        chunker = CharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Test separator statistics
        stats = chunker.get_separator_stats()
        self.assertIn("separator_stats", stats)
        self.assertIn("primary_separator", stats)
        self.assertIn("optimization_enabled", stats)
        self.assertTrue(stats["optimization_enabled"])

    def test_performance(self):
        config = CharacterChunkingConfig(chunk_size=200, chunk_overlap=50)
        chunker = CharacterChunkingStrategy(config)
        
        # Get initial metrics
        initial_metrics = chunker.get_metrics()
        self.assertEqual(initial_metrics.total_chunks, 0)
        
        # Process large document
        start_time = time.time()
        chunks = chunker.chunk(self.test_documents['large'])
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertGreater(len(chunks), 0)
        self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds
        
        # Get updated metrics
        metrics = chunker.get_metrics()
        self.assertGreater(metrics.total_chunks, 0)
        self.assertGreater(metrics.total_characters, 0)
        self.assertGreater(metrics.avg_chunk_size, 0)
        self.assertGreater(metrics.processing_time_ms, 0)
        self.assertEqual(metrics.strategy_name, "CharacterChunkingStrategy")

    def test_error_handling(self):
        # Test invalid configuration - overlap >= chunk_size
        with self.assertRaises(ValueError) as context:
            config = CharacterChunkingConfig(chunk_size=100, chunk_overlap=150)
            chunker = CharacterChunkingStrategy(config)
        self.assertIn("overlap", str(context.exception).lower())
        
        # Test very small chunk size
        config_small = CharacterChunkingConfig(chunk_size=5, chunk_overlap=2)
        chunker_small = CharacterChunkingStrategy(config_small)
        chunks_small = chunker_small.chunk(self.test_documents['basic'])
        
        self.assertGreater(len(chunks_small), 0)

    def test_batch_processing(self):
        documents = [
            self.test_documents['basic'],
            self.test_documents['paragraphs'],
            self.test_documents['sentences']
        ]
        
        config = CharacterChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = CharacterChunkingStrategy(config)
        
        batch_results = chunker.chunk_batch(documents)
        
        self.assertEqual(len(batch_results), 3)
        self.assertTrue(all(len(chunks) > 0 for chunks in batch_results))
        
        total_chunks = sum(len(chunks) for chunks in batch_results)
        self.assertGreater(total_chunks, 0)

    def test_caching_functionality(self):
        config = CharacterChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            enable_caching=True
        )
        chunker = CharacterChunkingStrategy(config)
        
        # First processing
        chunks1 = chunker.chunk(self.test_documents['basic'])
        
        # Second processing (should use cache)
        chunks2 = chunker.chunk(self.test_documents['basic'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))
        
        # Check cache info
        cache_info = chunker.get_cache_info()
        self.assertTrue(cache_info["enabled"])
        self.assertGreater(cache_info["size"], 0)

    def test_complex_text_structures(self):
        config = CharacterChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            separator="\n\n",
            multiple_separators=["\n", ". "],
            preserve_paragraphs=True,
            preserve_sentences=True
        )
        chunker = CharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that chunks maintain structure
        for chunk in chunks:
            self.assertGreater(len(chunk.text_content), 0)
            self.assertEqual(chunk.metadata['chunking_strategy'], 'CharacterChunkingStrategy')
            self.assertEqual(chunk.metadata['source'], 'complex')

if __name__ == "__main__":
    unittest.main()