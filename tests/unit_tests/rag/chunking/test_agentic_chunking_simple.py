import unittest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import pytest
from upsonic.text_splitter.agentic import AgenticChunker, AgenticChunkingConfig
from upsonic.schemas.data_models import Document, Chunk
from upsonic.schemas.agentic import PropositionList, TopicAssignmentList, Topic, RefinedTopic
from upsonic.agent.agent import Agent


class MockDirect(Agent):
    """Mock agent for testing agentic chunking without external dependencies."""
    
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


class TestAgenticChunkingSimple(unittest.TestCase):
    """Simplified tests for AgenticChunker that match the actual implementation."""

    def setUp(self):
        """Set up test documents."""
        # AI Technology Document
        self.ai_doc = Document(
            content="""Artificial Intelligence and Machine Learning Overview

Artificial Intelligence (AI) is a rapidly evolving field of computer science that aims to create machines capable of performing tasks that typically require human intelligence. The field encompasses several key areas and methodologies.

Machine Learning Fundamentals
Machine learning is a subset of AI that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on specific tasks through experience. Unlike traditional programming, where explicit instructions are coded, machine learning systems learn patterns from data.

Deep Learning Architecture
Deep learning represents a specialized subset of machine learning that employs artificial neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. These networks are inspired by the structure and function of the human brain, with interconnected nodes that process information.

Natural Language Processing Applications
Natural language processing (NLP) is a branch of AI that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful.

Computer Vision Technology
Computer vision is a field of AI that trains computers to interpret and understand the visual world. Through digital images from cameras and videos and by using deep learning models, machines can accurately identify and classify objects and then react to what they see.""",
            metadata={'source': 'ai_overview.txt', 'type': 'educational', 'topic': 'artificial_intelligence'},
            document_id="ai_doc_001"
        )
        
        # Climate Change Document
        self.climate_doc = Document(
            content="""Climate Change and Environmental Impact

Global climate change represents one of the most significant challenges facing humanity in the 21st century. The scientific consensus is clear: human activities are the primary driver of recent climate change.

Greenhouse Gas Emissions
The burning of fossil fuels for energy production, transportation, and industrial processes releases large quantities of greenhouse gases into the atmosphere. Carbon dioxide is the most abundant greenhouse gas, but methane and nitrous oxide also contribute significantly to global warming.

Temperature and Weather Patterns
Global average temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial times. This warming is not uniform across the globe, with Arctic regions experiencing more dramatic temperature increases. Changes in temperature patterns affect precipitation, storm intensity, and seasonal weather cycles.""",
            metadata={'source': 'climate_change.txt', 'type': 'scientific', 'topic': 'environmental_science'},
            document_id="climate_doc_001"
        )

    def test_basic_agentic_chunking(self):
        """Test basic agentic chunking functionality."""
        async def _test():
            config = AgenticChunkingConfig()
            mock_agent = MockDirect()
            chunker = AgenticChunker(mock_agent, config)
            
            chunks = await chunker._achunk_document(self.ai_doc)
            
            self.assertGreater(len(chunks), 0)
            self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
            self.assertTrue(all(chunk.text_content for chunk in chunks))
            
            # Verify content preservation
            all_content = " ".join(chunk.text_content for chunk in chunks)
            self.assertIn("Artificial Intelligence", all_content)
            self.assertIn("Machine learning", all_content)
        
        asyncio.run(_test())

    def test_proposition_extraction(self):
        """Test proposition extraction functionality."""
        async def _test():
            config = AgenticChunkingConfig(enable_proposition_validation=True)
            mock_agent = MockDirect()
            chunker = AgenticChunker(mock_agent, config)
            
            chunks = await chunker._achunk_document(self.ai_doc)
            
            self.assertGreater(len(chunks), 0)
            
            # Verify that propositions are meaningful
            for chunk in chunks:
                self.assertGreater(len(chunk.text_content), 0)
                
            # Should have generated propositions
            self.assertGreater(mock_agent.call_count, 0)
        
        asyncio.run(_test())

    def test_topic_assignment_and_refinement(self):
        """Test topic assignment and refinement functionality."""
        async def _test():
            config = AgenticChunkingConfig(
                enable_topic_optimization=True,
                include_topic_scores=True
            )
            mock_agent = MockDirect()
            chunker = AgenticChunker(mock_agent, config)
            
            chunks = await chunker._achunk_document(self.climate_doc)
            
            self.assertGreater(len(chunks), 0)
            
            # Check that topics were assigned (multiple agent calls)
            self.assertGreater(mock_agent.call_count, 1)
            
            # Verify chunk coherence
            for chunk in chunks:
                self.assertIsInstance(chunk.metadata, dict)
        
        asyncio.run(_test())

    def test_error_handling_and_fallback(self):
        """Test error handling and fallback functionality."""
        async def _test():
            config = AgenticChunkingConfig(
                fallback_to_recursive=True,
                max_agent_retries=2
            )
            mock_agent = MockDirect()
            mock_agent.fail_after = 1  # Fail after first call
            
            chunker = AgenticChunker(mock_agent, config)
            
            # Should fallback gracefully
            chunks = await chunker._achunk_document(self.ai_doc)
            
            self.assertGreater(len(chunks), 0)
            # Should fallback to recursive chunking
            for chunk in chunks:
                self.assertIsInstance(chunk, Chunk)
                self.assertTrue(chunk.text_content)
        
        asyncio.run(_test())

    def test_empty_content_handling(self):
        """Test handling of empty content."""
        async def _test():
            config = AgenticChunkingConfig()
            mock_agent = MockDirect()
            chunker = AgenticChunker(mock_agent, config)
            
            empty_doc = Document(
                content="",
                metadata={'source': 'empty.txt', 'type': 'empty'},
                document_id="empty_doc_001"
            )
            
            chunks = await chunker._achunk_document(empty_doc)
            
            # Should handle empty content gracefully
            self.assertEqual(len(chunks), 0)
        
        asyncio.run(_test())

    def test_batch_processing(self):
        """Test batch processing of multiple documents."""
        async def _test():
            documents = [
                self.ai_doc,
                self.climate_doc
            ]
            
            config = AgenticChunkingConfig()
            mock_agent = MockDirect()
            chunker = AgenticChunker(mock_agent, config)
            
            batch_results = await chunker.abatch(documents)
            
            # abatch returns a flat list of all chunks from all documents
            self.assertGreater(len(batch_results), 0)
            self.assertTrue(all(isinstance(chunk, Chunk) for chunk in batch_results))
            
            # Should have processed both documents
            all_text = " ".join(chunk.text_content for chunk in batch_results)
            self.assertIn("Artificial Intelligence", all_text)
            self.assertIn("Climate Change", all_text)
        
        asyncio.run(_test())


if __name__ == "__main__":
    unittest.main()
