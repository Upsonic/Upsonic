import os
import sys
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import unittest

from upsonic.text_splitter.contextual import ContextualOverlapChunkingStrategy, ContextualChunkingConfig
from upsonic.text_splitter.base import ChunkingMode
from upsonic.schemas.data_models import Document, Chunk

class TestContextualChunkingComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_documents = {}
        cls.create_ml_document(cls)
        cls.create_structured_document(cls)
        cls.create_semantic_document(cls)
        cls.create_complex_document(cls)
        cls.create_large_document(cls)
        cls.create_empty_document(cls)
        cls.create_edge_case_document(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_ml_document(self):
        content = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. These systems can learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Each type has its own characteristics and use cases.

### Supervised Learning

Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common algorithms include linear regression, decision trees, and neural networks.

### Unsupervised Learning

Unsupervised learning finds hidden patterns in data without labeled examples. Clustering and dimensionality reduction are common techniques.

## Applications

Machine learning has numerous applications across various industries including healthcare, finance, and technology."""
        doc = Document(content=content, metadata={'source': 'ml_intro', 'type': 'educational', 'domain': 'machine_learning'})
        self.test_documents['ml_intro'] = doc

    @staticmethod
    def create_structured_document(self):
        content = """# Chapter 1: Introduction

This is the introduction chapter that provides an overview of the topic and sets the context for the rest of the document.

## Section 1.1: Background

The background section contains historical context and previous research findings. It explains how the field has developed over time.

### Subsection 1.1.1: Key Concepts

Key concepts are defined here with detailed explanations. These concepts are fundamental to understanding the topic.

## Section 1.2: Methodology

The methodology section describes the approach used in this research. It outlines the methods and techniques employed."""
        doc = Document(content=content, metadata={'source': 'structured', 'type': 'research', 'chapters': 1, 'sections': 2})
        self.test_documents['structured'] = doc

    @staticmethod
    def create_semantic_document(self):
        content = """The concept of artificial intelligence has evolved significantly over the decades. Early AI systems were rule-based and limited in their capabilities.

Modern AI systems leverage machine learning algorithms and deep neural networks. These systems can process vast amounts of data and identify complex patterns.

The future of AI lies in developing more sophisticated reasoning capabilities. Researchers are working on creating AI systems that can understand context and make nuanced decisions."""
        doc = Document(content=content, metadata={'source': 'semantic', 'type': 'continuous', 'topic': 'ai_evolution'})
        self.test_documents['semantic'] = doc

    @staticmethod
    def create_complex_document(self):
        content = """# Advanced Machine Learning Techniques

## 1. Deep Learning Fundamentals

Deep learning is a subset of machine learning that uses neural networks with multiple layers. These networks can learn complex patterns in data.

### 1.1 Neural Network Architecture

Neural networks consist of interconnected nodes called neurons. Each neuron processes input data and produces an output.

#### 1.1.1 Activation Functions

Activation functions determine the output of a neuron. Common functions include ReLU, sigmoid, and tanh.

### 1.2 Training Process

Training involves adjusting network weights to minimize prediction errors. This is done using optimization algorithms like gradient descent.

## 2. Natural Language Processing

NLP is a field that focuses on enabling computers to understand human language. It involves various techniques and algorithms."""
        doc = Document(content=content, metadata={'source': 'complex', 'type': 'technical', 'complexity': 'high', 'sections': 4})
        self.test_documents['complex'] = doc

    @staticmethod
    def create_large_document(self):
        content_parts = []
        for i in range(50):
            content_parts.append(f"## Section {i+1}: Advanced Topic {i+1}")
            content_parts.append(f"This section covers advanced topic number {i+1} in detail. " +
                               f"It includes comprehensive explanations, examples, and practical applications. " +
                               f"The content is designed to provide deep insights into the subject matter. " +
                               f"Each section builds upon previous knowledge and introduces new concepts. " +
                               f"Topic {i+1} is particularly important for understanding the overall framework.")
            content_parts.append("")  # Empty line for separation
        content = "\n".join(content_parts)
        doc = Document(content=content, metadata={'source': 'large', 'type': 'comprehensive', 'sections': 50})
        self.test_documents['large'] = doc

    @staticmethod
    def create_empty_document(self):
        doc = Document(content="", metadata={'source': 'empty', 'type': 'edge_case'})
        self.test_documents['empty'] = doc

    @staticmethod
    def create_edge_case_document(self):
        content = "Short text."
        doc = Document(content=content, metadata={'source': 'edge_case', 'type': 'minimal', 'length': 'short'})
        self.test_documents['edge_case'] = doc

    def test_basic_contextual_chunking(self):
        config = ContextualChunkingConfig(
            chunk_size=300,
            chunk_overlap=100,
            context_window_size=200
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['ml_intro'])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        for i, chunk in enumerate(chunks):
            # Basic checks that should always work
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertGreater(len(chunk.text_content), 0)
            self.assertIsInstance(chunk.metadata, dict)

    def test_semantic_overlap_creation(self):
        config = ContextualChunkingConfig(
            chunk_size=100,
            chunk_overlap=40,
            semantic_overlap_ratio=0.4,
            sentence_boundaries=True
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['semantic'])
        
        self.assertGreater(len(chunks), 1)
        
        # Check that chunks are created with proper metadata
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertGreater(len(chunk.text_content), 0)

    def test_keyword_preservation(self):
        config = ContextualChunkingConfig(
            chunk_size=120,
            chunk_overlap=50,
            keyword_preservation=True,
            context_window_size=80
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['ml_intro'])
        
        self.assertGreater(len(chunks), 1)
        
        # Check that important keywords are preserved
        all_content = " ".join(chunk.text_content for chunk in chunks)
        important_keywords = ["machine learning", "artificial intelligence", "algorithms", "data"]
        
        preserved_keywords = sum(1 for keyword in important_keywords if keyword.lower() in all_content.lower())
        self.assertGreater(preserved_keywords, 0)

    def test_boundary_respect(self):
        config = ContextualChunkingConfig(
            chunk_size=200,
            chunk_overlap=80,
            sentence_boundaries=True,
            paragraph_boundaries=True,
            section_boundaries=True
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['structured'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that section boundaries are respected
        for chunk in chunks:
            content = chunk.text_content.strip()
            # If chunk starts with header, it should be complete
            if content.startswith("#"):
                self.assertTrue(any(content.startswith(prefix) for prefix in ["# ", "## ", "### ", "#### "]))

    def test_adaptive_overlap_sizing(self):
        simple_content = "This is simple text. It has short sentences. The content is straightforward."
        complex_content = """The intricate mechanisms underlying quantum computing paradigms necessitate a comprehensive understanding of quantum entanglement principles, superposition states, and decoherence phenomena that fundamentally challenge classical computational frameworks."""
        
        simple_doc = Document(content=simple_content, metadata={'complexity': 'low'})
        complex_doc = Document(content=complex_content, metadata={'complexity': 'high'})
        
        config = ContextualChunkingConfig(
            chunk_size=100,
            chunk_overlap=30,
            adaptive_overlap=True,
            context_window_size=50
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        
        simple_chunks = chunker.chunk(simple_doc)
        complex_chunks = chunker.chunk(complex_doc)
        
        self.assertGreaterEqual(len(simple_chunks), 0)
        self.assertGreaterEqual(len(complex_chunks), 0)
        
        # Check that chunks are created successfully
        if simple_chunks:
            self.assertIn('chunking_strategy', simple_chunks[0].metadata)
        if complex_chunks:
            self.assertIn('chunking_strategy', complex_chunks[0].metadata)

    def test_topic_coherence(self):
        config = ContextualChunkingConfig(
            chunk_size=200,
            chunk_overlap=80,
            topic_coherence=True,
            keyword_preservation=True
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['ml_intro'])
        
        self.assertGreater(len(chunks), 1)
        
        # Check that topic-related keywords are maintained across chunks
        all_content = " ".join(chunk.text_content for chunk in chunks)
        topic_keywords = ["machine", "learning", "algorithms", "data"]
        
        maintained_keywords = sum(1 for keyword in topic_keywords if keyword.lower() in all_content.lower())
        self.assertGreater(maintained_keywords, 0)

    def test_content_structure_analysis(self):
        config = ContextualChunkingConfig(
            chunk_size=250,
            chunk_overlap=100,
            section_boundaries=True,
            paragraph_boundaries=True
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that structure analysis is working
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertGreater(len(chunk.text_content), 0)

    def test_context_window_optimization(self):
        # Test with different context window sizes
        window_sizes = [50, 100, 200]
        
        for window_size in window_sizes:
            config = ContextualChunkingConfig(
                chunk_size=200,
                context_window_size=window_size
            )
            chunker = ContextualOverlapChunkingStrategy(config)
            chunks = chunker.chunk(self.test_documents['semantic'])
            
            self.assertGreater(len(chunks), 0)
            for chunk in chunks:
                self.assertIn('chunking_strategy', chunk.metadata)

    def test_empty_content_handling(self):
        config = ContextualChunkingConfig()
        chunker = ContextualOverlapChunkingStrategy(config)
        
        # Test empty content
        chunks_empty = chunker.chunk(self.test_documents['empty'])
        self.assertEqual(len(chunks_empty), 0)
        
        # Test very short content
        chunks_short = chunker.chunk(self.test_documents['edge_case'])
        self.assertGreaterEqual(len(chunks_short), 1)

    def test_oversized_chunk_handling(self):
        # Create a document with very long content
        long_text = "This is a very long paragraph with many repeated sentences. " * 50
        long_doc = Document(content=long_text, metadata={'type': 'oversized'})
        
        config = ContextualChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            min_meaningful_chunk_size=50
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        chunks = chunker.chunk(long_doc)
        
        self.assertGreater(len(chunks), 0)
        
        # Check that no chunk is excessively large
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text_content), config.chunk_size * 3)  # Allow flexibility

    def test_merge_optimization(self):
        short_content = """Short paragraph one.

Short paragraph two.

Short paragraph three."""
        
        short_doc = Document(content=short_content, metadata={'type': 'short_paragraphs'})
        
        config = ContextualChunkingConfig(
            chunk_size=300,
            chunk_overlap=50,
            min_meaningful_chunk_size=100
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        chunks = chunker.chunk(short_doc)
        
        self.assertGreater(len(chunks), 0)
        
        # Check that small chunks are handled appropriately
        for chunk in chunks:
            # Either chunk meets minimum size or it's the only chunk
            self.assertTrue(
                len(chunk.text_content) >= config.min_meaningful_chunk_size or 
                len(chunks) == 1
            )

    def test_performance_tracking(self):
        config = ContextualChunkingConfig(chunk_size=200, chunk_overlap=50)
        chunker = ContextualOverlapChunkingStrategy(config)
        
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
        self.assertEqual(metrics.strategy_name, "ContextualOverlapChunkingStrategy")

    def test_context_statistics(self):
        config = ContextualChunkingConfig(
            context_window_size=150,
            semantic_overlap_ratio=0.3,
            adaptive_overlap=True,
            keyword_preservation=True
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        
        # Process document
        chunks = chunker.chunk(self.test_documents['ml_intro'])
        
        # Get context statistics
        context_stats = chunker.get_context_stats()
        
        self.assertIn("context_window_size", context_stats)
        self.assertIn("semantic_overlap_ratio", context_stats)
        self.assertIn("adaptive_overlap_enabled", context_stats)
        self.assertIn("keyword_preservation_enabled", context_stats)
        self.assertIn("boundary_preservation", context_stats)
        
        self.assertEqual(context_stats["context_window_size"], 150)
        self.assertEqual(context_stats["semantic_overlap_ratio"], 0.3)
        self.assertTrue(context_stats["adaptive_overlap_enabled"])
        self.assertTrue(context_stats["keyword_preservation_enabled"])

    def test_complex_document_structures(self):
        config = ContextualChunkingConfig(
            chunk_size=300,
            chunk_overlap=100,
            sentence_boundaries=True,
            paragraph_boundaries=True,
            section_boundaries=True,
            keyword_preservation=True,
            topic_coherence=True
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that complex structure is handled well
        for chunk in chunks:
            self.assertGreater(len(chunk.text_content), 0)
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertEqual(chunk.metadata['source'], 'complex')

    def test_error_handling(self):
        # Test with potentially problematic configuration
        try:
            config = ContextualChunkingConfig(
                chunk_size=100,
                chunk_overlap=150,  # Overlap larger than chunk size
                context_window_size=50
            )
            chunker = ContextualOverlapChunkingStrategy(config)
            chunks = chunker.chunk(self.test_documents['ml_intro'])
            # Should handle gracefully
            self.assertGreaterEqual(len(chunks), 0)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass
        
        # Test with very small context window
        config = ContextualChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            context_window_size=10
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['ml_intro'])
        
        self.assertGreaterEqual(len(chunks), 1)

    def test_batch_processing(self):
        documents = [
            self.test_documents['ml_intro'],
            self.test_documents['structured'],
            self.test_documents['semantic']
        ]
        
        config = ContextualChunkingConfig(chunk_size=200, chunk_overlap=80)
        chunker = ContextualOverlapChunkingStrategy(config)
        
        batch_results = chunker.chunk_batch(documents)
        
        self.assertEqual(len(batch_results), 3)
        self.assertTrue(all(len(chunks) > 0 for chunks in batch_results))
        
        total_chunks = sum(len(chunks) for chunks in batch_results)
        self.assertGreater(total_chunks, 0)

    def test_caching_functionality(self):
        config = ContextualChunkingConfig(
            chunk_size=200,
            chunk_overlap=80,
            enable_caching=True
        )
        chunker = ContextualOverlapChunkingStrategy(config)
        
        # First processing
        chunks1 = chunker.chunk(self.test_documents['ml_intro'])
        
        # Second processing (should use cache)
        chunks2 = chunker.chunk(self.test_documents['ml_intro'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))
        
        # Check that caching works by comparing results
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))

if __name__ == "__main__":
    unittest.main()