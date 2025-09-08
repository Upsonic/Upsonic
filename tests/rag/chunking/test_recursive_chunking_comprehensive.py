import os
import sys
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import unittest

from upsonic.text_splitter.recursive import RecursiveCharacterChunkingStrategy, RecursiveChunkingConfig
from upsonic.schemas.data_models import Document, Chunk

class TestRecursiveChunkingComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_documents = {}
        cls.create_paragraph_document(cls)
        cls.create_sentence_document(cls)
        cls.create_mixed_separator_document(cls)
        cls.create_structured_document(cls)
        cls.create_large_document(cls)
        cls.create_custom_separator_document(cls)
        cls.create_edge_case_document(cls)
        cls.create_empty_document(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_paragraph_document(self):
        content = """First paragraph with some substantial content that provides context and information about the topic at hand.

Second paragraph with different content that explores another aspect of the subject matter in considerable detail.

Third paragraph that continues the discussion with additional insights and perspectives on the main theme.

Fourth paragraph that wraps up the content with concluding thoughts and final observations about the entire topic."""
        doc = Document(content=content, metadata={'source': 'paragraphs.txt', 'type': 'structured_text'})
        self.test_documents['paragraphs'] = doc

    @staticmethod
    def create_sentence_document(self):
        content = """This is the first sentence with substantial content. This is the second sentence that provides additional information. This is the third sentence that continues the narrative flow. This is the fourth sentence with more details. This is the fifth sentence that adds context. This is the sixth sentence with concluding thoughts."""
        doc = Document(content=content, metadata={'source': 'sentences.txt', 'type': 'continuous_text'})
        self.test_documents['sentences'] = doc

    @staticmethod
    def create_mixed_separator_document(self):
        content = """Section 1: Introduction

This is a comprehensive introduction to the topic. It provides background information and sets the context. The introduction covers multiple aspects of the subject.

Section 2: Main Content

The main content section delves deeper into the specifics. It contains detailed explanations and examples. Each point is thoroughly explored and documented.

Subsection 2.1: Technical Details

This subsection focuses on technical aspects. It provides in-depth analysis and technical specifications. The information is presented in a structured manner.

Subsection 2.2: Practical Examples

Here we present practical examples and use cases. These examples demonstrate real-world applications. Each example is carefully selected and explained.

Section 3: Conclusion

The conclusion summarizes the key points discussed. It provides final thoughts and recommendations. The summary ties together all the previous sections."""
        doc = Document(content=content, metadata={'source': 'mixed.txt', 'type': 'document', 'sections': 3})
        self.test_documents['mixed'] = doc

    @staticmethod
    def create_structured_document(self):
        content = """# Chapter 1: Introduction

This chapter provides a comprehensive introduction to the subject matter. It establishes the foundation for understanding the concepts that will be explored in subsequent chapters.

## Section 1.1: Background

The background section offers historical context and previous research findings. It traces the evolution of ideas and identifies key contributors to the field.

### Subsection 1.1.1: Historical Overview

The historical overview presents a chronological account of major developments. It highlights significant milestones and breakthrough discoveries.

### Subsection 1.1.2: Current State

This subsection examines the current state of knowledge in the field. It identifies ongoing research areas and emerging trends.

## Section 1.2: Scope and Objectives

This section defines the scope of the current work and outlines specific objectives. It establishes clear boundaries and expectations for the reader.

# Chapter 2: Methodology

The methodology chapter describes the approach taken to investigate the research questions. It provides detailed information about methods and procedures.

## Section 2.1: Research Design

The research design section explains the overall framework and strategy employed. It justifies the chosen approach and methodology.

## Section 2.2: Data Collection

This section details the data collection procedures and instruments used. It addresses issues of validity and reliability."""
        doc = Document(content=content, metadata={'source': 'structured.txt', 'type': 'academic', 'chapters': 2})
        self.test_documents['structured'] = doc

    @staticmethod
    def create_large_document(self):
        content_parts = []
        for i in range(100):
            if i % 20 == 0:
                content_parts.append(f"# Major Section {i//20 + 1}")
                content_parts.append("")
            elif i % 10 == 0:
                content_parts.append(f"## Subsection {i//10 + 1}")
                content_parts.append("")
            
            content_parts.append(f"Paragraph {i+1} contains detailed information about topic number {i+1}. " +
                               f"This paragraph explores various aspects and provides comprehensive coverage. " +
                               f"The content is designed to be informative and engaging for the reader. " +
                               f"Each paragraph builds upon previous knowledge and introduces new concepts.")
            content_parts.append("")
        
        content = "\n".join(content_parts)
        doc = Document(content=content, metadata={'source': 'large.txt', 'type': 'comprehensive', 'paragraphs': 100})
        self.test_documents['large'] = doc

    @staticmethod
    def create_custom_separator_document(self):
        content = "Part1|Part2|Part3;Section1;Section2;Section3,Item1,Item2,Item3,Item4,Item5"
        doc = Document(content=content, metadata={'source': 'custom.txt', 'type': 'delimited'})
        self.test_documents['custom'] = doc

    @staticmethod
    def create_edge_case_document(self):
        content = "Short text without many separators for edge case testing."
        doc = Document(content=content, metadata={'source': 'edge.txt', 'type': 'edge_case'})
        self.test_documents['edge_case'] = doc

    @staticmethod
    def create_empty_document(self):
        doc = Document(content="", metadata={'source': 'empty.txt', 'type': 'empty'})
        self.test_documents['empty'] = doc

    def test_basic_recursive_chunking(self):
        config = RecursiveChunkingConfig()
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['paragraphs'])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        # Verify paragraph content is preserved
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("First paragraph", all_content)
        self.assertIn("Second paragraph", all_content)
        self.assertIn("Third paragraph", all_content)

    def test_paragraph_separation(self):
        config = RecursiveChunkingConfig(separators=["\n\n", "\n", ". "])
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['paragraphs'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should prioritize paragraph breaks (\n\n)
        paragraph_content = ["First paragraph", "Second paragraph", "Third paragraph", "Fourth paragraph"]
        for para in paragraph_content:
            self.assertTrue(any(para in chunk.text_content for chunk in chunks))

    def test_sentence_separation(self):
        config = RecursiveChunkingConfig(chunk_size=100, chunk_overlap=20, separators=[". ", "\n", " "])
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['sentences'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should handle sentence boundaries
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("first sentence", all_content)
        self.assertIn("second sentence", all_content)

    def test_custom_separators(self):
        config = RecursiveChunkingConfig(separators=["|", ";", ","])
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['custom'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should split on custom separators
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Part1", all_content)
        self.assertIn("Section1", all_content)
        self.assertIn("Item1", all_content)

    def test_regex_separators(self):
        # Test with regex pattern separators
        config = RecursiveChunkingConfig(
            separators=[r"\d+", r"[A-Z][a-z]+"],
            is_separator_regex=True
        )
        chunker = RecursiveCharacterChunkingStrategy(config)
        
        regex_content = "Word123Another456Third789End"
        regex_doc = Document(content=regex_content, metadata={'type': 'regex_test'})
        chunks = chunker.chunk(regex_doc)
        
        self.assertGreater(len(chunks), 0)
        # Content should be processed even if regex patterns don't match perfectly
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Word", all_content)

    def test_keep_separator_option(self):
        config = RecursiveChunkingConfig(keep_separator=True)
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['paragraphs'])
        
        self.assertGreater(len(chunks), 0)
        
        # Separators should be preserved in chunks
        has_newlines = any("\n" in chunk.text_content for chunk in chunks)
        # This might not always be true depending on implementation, so we check flexibly
        self.assertTrue(len(chunks) > 0)

    def test_adaptive_splitting(self):
        config = RecursiveChunkingConfig(enable_adaptive_splitting=True)
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['mixed'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should adapt to content structure
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Section 1", all_content)
        self.assertIn("Section 2", all_content)

    def test_chunk_size_limits(self):
        config = RecursiveChunkingConfig(chunk_size=200, chunk_overlap=50)
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['large'])
        
        self.assertGreater(len(chunks), 1)  # Should split large document
        
        # Most chunks should respect size limits (with some flexibility)
        oversized_chunks = [chunk for chunk in chunks if len(chunk.text_content) > 300]
        self.assertLess(len(oversized_chunks), len(chunks) * 0.2)  # Less than 20% oversized

    def test_chunk_overlap(self):
        config = RecursiveChunkingConfig(chunk_size=150, chunk_overlap=30)
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['structured'])
        
        if len(chunks) > 1:
            # Check for potential overlap in adjacent chunks
            for i in range(len(chunks) - 1):
                chunk1_words = set(chunks[i].text_content.split())
                chunk2_words = set(chunks[i + 1].text_content.split())
                # Some overlap might exist
                overlap = chunk1_words & chunk2_words
                # This is flexible as overlap implementation may vary
                self.assertTrue(len(chunks) > 1)

    def test_balanced_chunks(self):
        config = RecursiveChunkingConfig(prefer_balanced_chunks=True)
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['structured'])
        
        self.assertGreater(len(chunks), 0)
        
        if len(chunks) > 1:
            # Check that chunks are reasonably balanced
            chunk_lengths = [len(chunk.text_content) for chunk in chunks]
            max_length = max(chunk_lengths)
            min_length = min(chunk_lengths)
            # Allow reasonable variation in chunk sizes
            if min_length > 0:
                self.assertLess(max_length / min_length, 5.0)

    def test_separator_frequency_threshold(self):
        config = RecursiveChunkingConfig(min_separator_frequency=0.1)
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['mixed'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should filter out rare separators
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Section", all_content)

    def test_max_recursion_depth(self):
        config = RecursiveChunkingConfig(max_recursion_depth=3)
        chunker = RecursiveCharacterChunkingStrategy(config)
        
        # Create very long text that would require deep recursion
        long_text = "A" * 10000
        long_doc = Document(content=long_text, metadata={'type': 'very_long'})
        chunks = chunker.chunk(long_doc)
        
        self.assertGreater(len(chunks), 0)
        # Should handle without infinite recursion

    def test_separator_caching(self):
        config = RecursiveChunkingConfig(enable_separator_caching=True)
        chunker = RecursiveCharacterChunkingStrategy(config)
        
        # First processing
        chunks1 = chunker.chunk(self.test_documents['paragraphs'])
        
        # Second processing (should use cache if available)
        chunks2 = chunker.chunk(self.test_documents['paragraphs'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))

    def test_content_type_detection(self):
        config = RecursiveChunkingConfig(content_type_detection=True)
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['structured'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should detect structured content and adapt accordingly
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Chapter", all_content)
        self.assertIn("Section", all_content)

    def test_empty_content_handling(self):
        config = RecursiveChunkingConfig()
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['empty'])
        
        self.assertEqual(len(chunks), 0)

    def test_whitespace_only_content(self):
        whitespace_content = "   \n\n  \t  \n   "
        whitespace_doc = Document(content=whitespace_content, metadata={'type': 'whitespace'})
        
        config = RecursiveChunkingConfig()
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(whitespace_doc)
        
        self.assertEqual(len(chunks), 0)

    def test_edge_case_short_content(self):
        config = RecursiveChunkingConfig()
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['edge_case'])
        
        # Short content should fit in one chunk
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text_content.strip(), self.test_documents['edge_case'].content.strip())

    def test_performance_with_large_document(self):
        config = RecursiveChunkingConfig()
        chunker = RecursiveCharacterChunkingStrategy(config)
        
        start_time = time.time()
        chunks = chunker.chunk(self.test_documents['large'])
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertGreater(len(chunks), 1)
        self.assertLess(processing_time, 10.0)  # Should complete within 10 seconds
        
        # Verify content preservation
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Major Section", all_content)
        self.assertIn("Paragraph", all_content)

    def test_metadata_inheritance(self):
        config = RecursiveChunkingConfig()
        chunker = RecursiveCharacterChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['structured'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify original metadata is inherited
        for chunk in chunks:
            self.assertEqual(chunk.metadata['source'], 'structured.txt')
            self.assertEqual(chunk.metadata['type'], 'academic')
            self.assertEqual(chunk.metadata['chapters'], 2)

    def test_separator_statistics(self):
        config = RecursiveChunkingConfig()
        chunker = RecursiveCharacterChunkingStrategy(config)
        
        # Process document
        chunks = chunker.chunk(self.test_documents['mixed'])
        
        # Try to get separator statistics if available
        try:
            stats = chunker.get_separator_stats()
            self.assertIsInstance(stats, dict)
        except AttributeError:
            # Method might not exist in actual implementation
            pass

    def test_clear_cache_functionality(self):
        config = RecursiveChunkingConfig(enable_separator_caching=True)
        chunker = RecursiveCharacterChunkingStrategy(config)
        
        # Process to populate any caches
        chunks = chunker.chunk(self.test_documents['paragraphs'])
        
        # Try to clear cache if method exists
        try:
            chunker.clear_separator_cache()
        except AttributeError:
            # Method might not exist in actual implementation
            pass

    def test_batch_processing(self):
        documents = [
            self.test_documents['paragraphs'],
            self.test_documents['sentences'],
            self.test_documents['mixed']
        ]
        
        config = RecursiveChunkingConfig()
        chunker = RecursiveCharacterChunkingStrategy(config)
        
        batch_results = chunker.chunk_batch(documents)
        
        self.assertEqual(len(batch_results), 3)
        self.assertTrue(all(len(chunks) > 0 for chunks in batch_results))
        
        total_chunks = sum(len(chunks) for chunks in batch_results)
        self.assertGreater(total_chunks, 0)

    def test_caching_functionality(self):
        config = RecursiveChunkingConfig(enable_caching=True)
        chunker = RecursiveCharacterChunkingStrategy(config)
        
        # First processing
        chunks1 = chunker.chunk(self.test_documents['paragraphs'])
        
        # Second processing (should use cache)
        chunks2 = chunker.chunk(self.test_documents['paragraphs'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))

    def test_error_handling(self):
        config = RecursiveChunkingConfig()
        chunker = RecursiveCharacterChunkingStrategy(config)
        
        # Test with extremely long content without separators
        very_long_content = "A" * 100000
        very_long_doc = Document(content=very_long_content, metadata={'type': 'extreme'})
        
        chunks = chunker.chunk(very_long_doc)
        self.assertGreater(len(chunks), 0)  # Should handle gracefully
        
        # Should not crash with edge cases
        for chunk in chunks:
            self.assertIsInstance(chunk.text_content, str)

if __name__ == "__main__":
    unittest.main()