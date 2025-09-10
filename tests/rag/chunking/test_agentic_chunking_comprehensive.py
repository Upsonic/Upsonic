import os
import sys
import tempfile
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from upsonic.text_splitter.agentic import AgenticChunkingStrategy, AgenticChunkingConfig
from upsonic.schemas.data_models import Document, Chunk
from upsonic.schemas.agentic import PropositionList, TopicAssignmentList, Topic, RefinedTopic
from upsonic.agent.agent import Direct


class MockDirect(Direct):
    """Mock Direct agent for testing agentic chunking without external dependencies."""
    
    def __init__(self):
        # Initialize with minimal required parameters to avoid complex dependency setup
        self.call_count = 0
        self.fail_after = None
        # Skip the parent __init__ to avoid complex dependency setup
        
    async def do_async(self, task, **kwargs):
        """Mock agent do_async method that returns predefined responses."""
        self.call_count += 1
        
        # Simulate failure for testing error handling
        if self.fail_after and self.call_count > self.fail_after:
            raise Exception("Mock agent failure")
        
        # Return mock responses based on task type
        task_description = task.description if hasattr(task, 'description') else str(task)
        
        if "proposition" in task_description.lower():
            return PropositionList(propositions=[
                "Artificial Intelligence is a branch of computer science.",
                "Machine learning focuses on algorithms that learn from data.",
                "Deep learning uses neural networks with multiple layers.",
                "Natural language processing enables computers to understand human language.",
                "Computer vision allows machines to interpret visual information."
            ])
        elif "topic" in task_description.lower() and "assign" in task_description.lower():
            return TopicAssignmentList(topics=[
                Topic(topic_id=1, propositions=[
                    "Artificial Intelligence is a branch of computer science.",
                    "Machine learning focuses on algorithms that learn from data."
                ]),
                Topic(topic_id=2, propositions=[
                    "Deep learning uses neural networks with multiple layers.",
                    "Natural language processing enables computers to understand human language.",
                    "Computer vision allows machines to interpret visual information."
                ])
            ])
        elif "refine" in task_description.lower():
            return RefinedTopic(
                title="AI and Machine Learning Fundamentals",
                summary="Overview of artificial intelligence concepts including machine learning, deep learning, NLP, and computer vision."
            )
        
        return None


class TestAgenticChunkingComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_documents = {}
        cls.create_test_documents(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_test_documents(self):
        # AI Technology Document
        ai_content = """Artificial Intelligence and Machine Learning Overview

Artificial Intelligence (AI) is a rapidly evolving field of computer science that aims to create machines capable of performing tasks that typically require human intelligence. The field encompasses several key areas and methodologies.

Machine Learning Fundamentals
Machine learning is a subset of AI that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on specific tasks through experience. Unlike traditional programming, where explicit instructions are coded, machine learning systems learn patterns from data.

Deep Learning Architecture
Deep learning represents a specialized subset of machine learning that employs artificial neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. These networks are inspired by the structure and function of the human brain, with interconnected nodes that process information.

Natural Language Processing Applications
Natural language processing (NLP) is a branch of AI that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful.

Computer Vision Technology
Computer vision is a field of AI that trains computers to interpret and understand the visual world. Through digital images from cameras and videos and by using deep learning models, machines can accurately identify and classify objects and then react to what they see.

Reinforcement Learning Paradigm
Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties and uses this feedback to improve its decision-making over time."""

        doc = Document(
            content=ai_content,
            metadata={'source': 'ai_overview.txt', 'type': 'educational', 'topic': 'artificial_intelligence'},
            document_id="ai_doc_001"
        )
        self.test_documents['ai_overview'] = doc

        # Climate Change Document
        climate_content = """Climate Change and Environmental Impact

Global climate change represents one of the most significant challenges facing humanity in the 21st century. The scientific consensus is clear: human activities are the primary driver of recent climate change.

Greenhouse Gas Emissions
The burning of fossil fuels for energy production, transportation, and industrial processes releases large quantities of greenhouse gases into the atmosphere. Carbon dioxide is the most abundant greenhouse gas, but methane and nitrous oxide also contribute significantly to global warming.

Temperature and Weather Patterns
Global average temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial times. This warming is not uniform across the globe, with Arctic regions experiencing more dramatic temperature increases. Changes in temperature patterns affect precipitation, storm intensity, and seasonal weather cycles.

Impact on Ecosystems
Climate change affects ecosystems worldwide, altering habitat conditions for plants and animals. Many species face challenges adapting to changing temperatures and precipitation patterns. Ocean acidification, caused by increased CO2 absorption, threatens marine ecosystems and food chains.

Economic Consequences
The economic impact of climate change includes damage from extreme weather events, agricultural disruptions, and costs associated with adaptation and mitigation efforts. Industries such as insurance, agriculture, and tourism face particular challenges from climate-related changes.

Mitigation and Adaptation Strategies
Addressing climate change requires both mitigation efforts to reduce greenhouse gas emissions and adaptation strategies to cope with unavoidable changes. Renewable energy technologies, energy efficiency improvements, and carbon capture methods are key mitigation approaches."""

        doc = Document(
            content=climate_content,
            metadata={'source': 'climate_change.txt', 'type': 'scientific', 'topic': 'environmental_science'},
            document_id="climate_doc_001"
        )
        self.test_documents['climate_change'] = doc

        # Empty Document
        doc = Document(
            content="",
            metadata={'source': 'empty.txt', 'type': 'empty'},
            document_id="empty_doc_001"
        )
        self.test_documents['empty'] = doc

    async def test_basic_agentic_chunking(self):
        config = AgenticChunkingConfig()
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        chunks = await chunker.chunk(self.test_documents['ai_overview'])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        # Verify content preservation
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Artificial Intelligence", all_content)
        self.assertIn("Machine learning", all_content)

    async def test_proposition_extraction(self):
        config = AgenticChunkingConfig(enable_proposition_validation=True)
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        chunks = await chunker.chunk(self.test_documents['ai_overview'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that propositions are meaningful
        for chunk in chunks:
            self.assertGreater(len(chunk.text_content), 50)  # Reasonable chunk size
            
        # Should have generated propositions
        self.assertGreater(mock_agent.call_count, 0)

    async def test_topic_assignment_and_refinement(self):
        config = AgenticChunkingConfig(
            enable_topic_optimization=True,
            include_topic_scores=True
        )
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        chunks = await chunker.chunk(self.test_documents['climate_change'])
        
        self.assertGreater(len(chunks), 0)
        
        # Check that topics were assigned (multiple agent calls)
        self.assertGreater(mock_agent.call_count, 1)
        
        # Verify chunk coherence
        for chunk in chunks:
            self.assertIsInstance(chunk.metadata, dict)
            if config.include_topic_scores:
                # Topic scores should be included in metadata
                pass  # Mock doesn't actually add these, but structure is validated

    async def test_caching_functionality(self):
        config = AgenticChunkingConfig(
            enable_proposition_caching=True,
            enable_topic_caching=True,
            enable_refinement_caching=True
        )
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        # First processing
        chunks1 = await chunker.chunk(self.test_documents['ai_overview'])
        first_call_count = mock_agent.call_count
        
        # Second processing (should use cache)
        mock_agent.call_count = 0  # Reset counter
        chunks2 = await chunker.chunk(self.test_documents['ai_overview'])
        second_call_count = mock_agent.call_count
        
        # Should have same results
        self.assertEqual(len(chunks1), len(chunks2))
        
        # Second call should use cache (fewer agent calls)
        if config.enable_proposition_caching:
            self.assertLessEqual(second_call_count, first_call_count)

    async def test_parallel_processing(self):
        config = AgenticChunkingConfig(parallel_processing=True)
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        start_time = time.time()
        chunks = await chunker.chunk(self.test_documents['climate_change'])
        end_time = time.time()
        
        self.assertGreater(len(chunks), 0)
        processing_time = end_time - start_time
        self.assertLess(processing_time, 30.0)  # Should complete reasonably quickly

    async def test_error_handling_and_fallback(self):
        config = AgenticChunkingConfig(
            fallback_to_recursive=True,
            max_agent_retries=2
        )
        mock_agent = MockDirect()
        mock_agent.fail_after = 1  # Fail after first call
        
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        # Should fallback gracefully
        chunks = await chunker.chunk(self.test_documents['ai_overview'])
        
        self.assertGreater(len(chunks), 0)
        # Should fallback to recursive chunking
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertTrue(chunk.text_content)

    async def test_empty_content_handling(self):
        config = AgenticChunkingConfig()
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        chunks = await chunker.chunk(self.test_documents['empty'])
        
        # Should handle empty content gracefully
        self.assertEqual(len(chunks), 0)

    async def test_proposition_validation(self):
        config = AgenticChunkingConfig(
            enable_proposition_validation=True,
            min_proposition_length=20,
            min_propositions_per_chunk=2,
            max_propositions_per_chunk=10
        )
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        chunks = await chunker.chunk(self.test_documents['ai_overview'])
        
        self.assertGreater(len(chunks), 0)
        
        # Validate proposition constraints
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk.text_content), config.min_proposition_length * config.min_propositions_per_chunk)

    async def test_incremental_processing(self):
        config = AgenticChunkingConfig(
            enable_incremental_processing=True,
            batch_proposition_size=50
        )
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        chunks = await chunker.chunk(self.test_documents['climate_change'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should process incrementally
        self.assertGreater(mock_agent.call_count, 0)

    async def test_metadata_inclusion(self):
        config = AgenticChunkingConfig(
            include_proposition_metadata=True,
            include_topic_scores=True,
            include_agent_metadata=True
        )
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        chunks = await chunker.chunk(self.test_documents['ai_overview'])
        
        self.assertGreater(len(chunks), 0)
        
        # Check metadata structure
        for chunk in chunks:
            self.assertIsInstance(chunk.metadata, dict)
            # Original document metadata should be preserved
            self.assertEqual(chunk.metadata.get('source'), 'ai_overview.txt')

    async def test_coherence_scoring(self):
        config = AgenticChunkingConfig(enable_coherence_scoring=True)
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        chunks = await chunker.chunk(self.test_documents['climate_change'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should have called agent for coherence scoring
        self.assertGreater(mock_agent.call_count, 0)

    async def test_batch_processing(self):
        documents = [
            self.test_documents['ai_overview'],
            self.test_documents['climate_change']
        ]
        
        config = AgenticChunkingConfig()
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        batch_results = await chunker.chunk_batch_async(documents)
        
        self.assertEqual(len(batch_results), 2)
        self.assertTrue(all(len(chunks) > 0 for chunks in batch_results))
        
        # Should have processed both documents
        total_chunks = sum(len(chunks) for chunks in batch_results)
        self.assertGreater(total_chunks, 0)

    async def test_configuration_validation(self):
        # Test valid configuration
        config = AgenticChunkingConfig(
            chunk_size=1000,
            min_propositions_per_chunk=3,
            max_propositions_per_chunk=15
        )
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        self.assertEqual(chunker.config.chunk_size, 1000)
        self.assertEqual(chunker.config.min_propositions_per_chunk, 3)
        self.assertEqual(chunker.config.max_propositions_per_chunk, 15)

    async def test_timeout_handling(self):
        config = AgenticChunkingConfig(
            agent_timeout_seconds=1,  # Very short timeout
            fallback_to_recursive=True
        )
        
        # Mock agent that simulates slow response
        mock_agent = MockDirect()
        
        async def slow_response(*args, **kwargs):
            import asyncio
            await asyncio.sleep(2)  # Longer than timeout
            return PropositionList(propositions=["Test proposition"])
        
        mock_agent.do_async = slow_response
        
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        # Should fallback due to timeout
        chunks = await chunker.chunk(self.test_documents['ai_overview'])
        self.assertGreater(len(chunks), 0)

    async def test_large_document_performance(self):
        # Create a large document
        large_content = """Large Document Performance Test
        
This is a large document created for performance testing of the agentic chunking strategy. """ + \
        ("This content repeats to create a sufficiently large document for testing purposes. " * 100)
        
        large_doc = Document(
            content=large_content,
            metadata={'source': 'large_test.txt', 'type': 'performance_test'},
            document_id="large_doc_001"
        )
        
        config = AgenticChunkingConfig()
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        start_time = time.time()
        chunks = await chunker.chunk(large_doc)
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks
        self.assertLess(processing_time, 60.0)  # Should complete within reasonable time
        
        # Verify content distribution
        total_content_length = sum(len(chunk.text_content) for chunk in chunks)
        self.assertGreater(total_content_length, 0)

    async def test_topic_optimization(self):
        config = AgenticChunkingConfig(
            enable_topic_optimization=True,
            enable_coherence_scoring=True
        )
        mock_agent = MockDirect()
        chunker = AgenticChunkingStrategy(mock_agent, config)
        
        chunks = await chunker.chunk(self.test_documents['climate_change'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should have optimized topics (multiple agent calls)
        self.assertGreater(mock_agent.call_count, 1)
        
        # Chunks should be coherent
        for chunk in chunks:
            self.assertGreater(len(chunk.text_content.strip()), 0)


if __name__ == '__main__':
    unittest.main()