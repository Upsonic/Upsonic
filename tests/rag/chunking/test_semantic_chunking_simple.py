import unittest
import uuid
import os
import pytest
from upsonic.text_splitter.semantic import SemanticChunker, SemanticChunkingConfig
from upsonic.schemas.data_models import Document, Chunk
from upsonic.embeddings.fastembed_provider import FastEmbedProvider


class TestSemanticChunkingSimple(unittest.TestCase):
    """Simplified tests for SemanticChunker that match the actual implementation."""

    def setUp(self):
        """Set up test documents and embedding provider."""
        
        self.embedding_provider = FastEmbedProvider()
        
        self.simple_doc = Document(
            content="""Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is a rapidly evolving field of computer science. The field encompasses several key areas and methodologies that are transforming various industries.

Machine Learning Fundamentals
Machine learning is a subset of AI that focuses on algorithms and statistical models. These systems learn patterns from data without explicit programming instructions.

Climate Change and Environmental Science
Climate change represents one of the most significant challenges facing humanity. The scientific consensus is clear about human activities being the primary driver.

Greenhouse Gas Emissions
The burning of fossil fuels releases large quantities of greenhouse gases. Carbon dioxide is the most abundant greenhouse gas in our atmosphere.""",
            metadata={'source': 'mixed_topics.txt', 'type': 'educational'},
            document_id=str(uuid.uuid4())
        )
        
        # Document with single topic
        self.single_topic_doc = Document(
            content="""Machine Learning Algorithms

Supervised learning algorithms learn from labeled training data. Examples include linear regression, decision trees, and neural networks.

Unsupervised learning algorithms find patterns in data without labels. Clustering and dimensionality reduction are common unsupervised techniques.

Reinforcement learning algorithms learn through interaction with an environment. They receive rewards or penalties based on their actions.""",
            metadata={'source': 'ml_algorithms.txt', 'type': 'technical'},
            document_id=str(uuid.uuid4())
        )

    def test_basic_semantic_chunking(self):
        """Test basic semantic chunking functionality."""
        config = SemanticChunkingConfig(
            embedding_provider=self.embedding_provider,
            chunk_size=200,
            chunk_overlap=50
        )
        chunker = SemanticChunker(config)
        
        chunks = chunker.chunk([self.simple_doc])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))

    def test_single_topic_document(self):
        """Test semantic chunking with single topic document."""
        config = SemanticChunkingConfig(
            embedding_provider=self.embedding_provider,
            chunk_size=150,
            chunk_overlap=30
        )
        chunker = SemanticChunker(config)
        
        chunks = chunker.chunk([self.single_topic_doc])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify content preservation
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Machine Learning", all_content)
        self.assertIn("Supervised learning", all_content)

    def test_different_threshold_types(self):
        """Test different breakpoint threshold types."""
        threshold_types = ["percentile", "standard_deviation", "interquartile"]
        
        for threshold_type in threshold_types:
            config = SemanticChunkingConfig(
                embedding_provider=self.embedding_provider,
                breakpoint_threshold_type=threshold_type,
                chunk_size=100
            )
            chunker = SemanticChunker(config)
            chunks = chunker.chunk([self.simple_doc])
            
            self.assertGreaterEqual(len(chunks), 1)

    def test_empty_document(self):
        """Test handling of empty documents."""
        empty_doc = Document(
            content="",
            metadata={'source': 'empty'},
            document_id=str(uuid.uuid4())
        )
        
        config = SemanticChunkingConfig(embedding_provider=self.embedding_provider)
        chunker = SemanticChunker(config)
        chunks = chunker.chunk([empty_doc])
        
        self.assertEqual(len(chunks), 0)

    def test_short_document(self):
        """Test handling of very short documents."""
        short_doc = Document(
            content="This is a very short document with only one sentence.",
            metadata={'source': 'short'},
            document_id=str(uuid.uuid4())
        )
        
        config = SemanticChunkingConfig(embedding_provider=self.embedding_provider)
        chunker = SemanticChunker(config)
        chunks = chunker.chunk([short_doc])
        
        # Should create at least one chunk
        self.assertGreaterEqual(len(chunks), 1)

    def test_batch_processing(self):
        """Test batch processing of multiple documents."""
        documents = [self.simple_doc, self.single_topic_doc]
        
        config = SemanticChunkingConfig(
            embedding_provider=self.embedding_provider,
            chunk_size=150
        )
        chunker = SemanticChunker(config)
        
        batch_results = chunker.chunk(documents)
        
        self.assertGreater(len(batch_results), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in batch_results))
        
        # Verify content from different documents is present
        all_text = " ".join(chunk.text_content for chunk in batch_results)
        self.assertIn("Artificial Intelligence", all_text)
        self.assertIn("Machine Learning", all_text)

    def test_async_chunking(self):
        """Test async chunking functionality."""
        async def _test():
            config = SemanticChunkingConfig(
                embedding_provider=self.embedding_provider,
                chunk_size=200
            )
            chunker = SemanticChunker(config)
            
            chunks = await chunker._achunk_document(self.simple_doc)
            
            self.assertGreater(len(chunks), 0)
            self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        
        import asyncio
        asyncio.run(_test())

    def test_async_batch_processing(self):
        """Test async batch processing."""
        async def _test():
            documents = [self.simple_doc, self.single_topic_doc]
            
            config = SemanticChunkingConfig(
                embedding_provider=self.embedding_provider,
                chunk_size=150
            )
            chunker = SemanticChunker(config)
            
            batch_results = await chunker.abatch(documents)
            
            self.assertGreater(len(batch_results), 0)
            self.assertTrue(all(isinstance(chunk, Chunk) for chunk in batch_results))
        
        import asyncio
        asyncio.run(_test())


if __name__ == "__main__":
    unittest.main()
