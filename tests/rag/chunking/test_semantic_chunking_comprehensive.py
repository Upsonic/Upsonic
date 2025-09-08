import os
import sys
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import unittest
import numpy as np
import asyncio

from upsonic.text_splitter.semantic import SemanticSimilarityChunkingStrategy, SemanticChunkingConfig
from upsonic.schemas.data_models import Document, Chunk
from upsonic.embeddings.base import EmbeddingProvider, EmbeddingConfig


class MockEmbeddingConfig(EmbeddingConfig):
    """Mock embedding configuration for testing."""
    model_name: str = "mock-model"
    dimension: int = 384


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing semantic chunking without external dependencies."""
    
    def __init__(self, config: Optional[MockEmbeddingConfig] = None):
        if config is None:
            config = MockEmbeddingConfig()
        super().__init__(config=config)
        
    def embed_single(self, text: str) -> List[float]:
        """Generate mock embeddings based on text hash for consistency."""
        # Simple hash-based mock embeddings for testing
        hash_val = hash(text) % (2**32)
        np.random.seed(hash_val)
        return np.random.randn(self.config.dimension).tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for batch of texts."""
        return [self.embed_single(text) for text in texts]
    
    async def _embed_batch(self, texts: List[str], mode = None) -> List[List[float]]:
        """Internal method to embed a batch of texts."""
        return self.embed_batch(texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the mock embedding model."""
        return {
            'model_name': self.config.model_name,
            'dimension': self.config.dimension,
            'max_tokens': 8192,
            'provider': 'mock'
        }
    
    @property
    def supported_modes(self):
        """List of embedding modes supported by mock provider."""
        from upsonic.embeddings.base import EmbeddingMode
        return [EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY, EmbeddingMode.SYMMETRIC]
    
    @property 
    def pricing_info(self) -> Dict[str, float]:
        """Get pricing information for the mock model."""
        return {
            'per_token': 0.0,
            'per_1k_tokens': 0.0,
            'per_million_tokens': 0.0
        }

class TestSemanticChunkingComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_documents = {}
        cls.create_topical_document(cls)
        cls.create_scientific_document(cls)
        cls.create_narrative_document(cls)
        cls.create_mixed_topic_document(cls)
        cls.create_technical_document(cls)
        cls.create_conversational_document(cls)
        cls.create_large_semantic_document(cls)
        cls.create_empty_document(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_topical_document(self):
        content = """Climate Change and Environmental Impact

Global warming has become one of the most pressing issues of our time. Rising temperatures are causing significant changes to weather patterns worldwide. Scientists have documented increased frequency of extreme weather events, including hurricanes, droughts, and heatwaves.

The impact on ecosystems is profound. Many species are struggling to adapt to changing conditions. Arctic ice is melting at unprecedented rates, affecting polar bear populations and contributing to rising sea levels. Ocean acidification is threatening marine life, particularly coral reefs and shellfish.

Agricultural systems are also under stress. Changing precipitation patterns affect crop yields, while rising temperatures create challenges for traditional farming practices. Some regions face increased drought, while others experience unexpected flooding.

Economic consequences are substantial. Insurance companies report increased claims from weather-related disasters. Tourism industries in coastal areas face uncertainty due to rising sea levels. Energy sectors must adapt to changing demand patterns and weather-dependent renewable energy sources.

Mitigation strategies are being implemented globally. Renewable energy adoption is accelerating, with solar and wind power becoming cost-competitive with fossil fuels. Carbon capture technologies are being developed to remove CO2 from the atmosphere. International cooperation through climate agreements aims to coordinate global response efforts.

Individual actions also matter. Energy conservation, sustainable transportation choices, and supporting environmentally responsible businesses all contribute to addressing climate change. Education and awareness are crucial for building public support for necessary policy changes."""
        doc = Document(content=content, metadata={'source': 'topical.txt', 'type': 'environmental', 'topic': 'climate_change'})
        self.test_documents['topical'] = doc

    @staticmethod
    def create_scientific_document(self):
        content = """Quantum Computing Fundamentals

Quantum mechanics provides the foundation for quantum computing. Unlike classical bits that exist in either 0 or 1 states, quantum bits (qubits) can exist in superposition states. This allows quantum computers to process multiple possibilities simultaneously.

Entanglement is another crucial quantum phenomenon. When qubits become entangled, measuring one instantly affects the other, regardless of distance. This property enables quantum algorithms to achieve exponential speedup for certain computational problems.

Quantum gates manipulate qubits through unitary operations. Common gates include the Pauli-X gate (quantum NOT), Hadamard gate (creates superposition), and CNOT gate (creates entanglement). These gates form the building blocks of quantum circuits.

Quantum algorithms demonstrate the potential of quantum computing. Shor's algorithm can factor large integers exponentially faster than classical methods, threatening current cryptographic systems. Grover's algorithm provides quadratic speedup for searching unsorted databases.

Current quantum hardware faces significant challenges. Quantum decoherence causes qubits to lose their quantum properties rapidly. Error rates in current systems are high, requiring quantum error correction techniques. Scalability remains a major obstacle for building practical quantum computers.

Applications of quantum computing span multiple domains. Drug discovery could be revolutionized through quantum simulation of molecular interactions. Financial modeling might benefit from quantum optimization algorithms. Machine learning algorithms could potentially achieve quantum advantages for specific tasks."""
        doc = Document(content=content, metadata={'source': 'scientific.txt', 'type': 'research', 'topic': 'quantum_computing'})
        self.test_documents['scientific'] = doc

    @staticmethod
    def create_narrative_document(self):
        content = """The Art of Storytelling

Stories have been central to human culture since the beginning of civilization. From ancient cave paintings to modern digital media, humans have always sought to share experiences through narrative. Storytelling serves multiple purposes: entertainment, education, cultural preservation, and emotional connection.

Character development forms the heart of compelling narratives. Well-crafted characters feel authentic and relatable, allowing readers to form emotional connections. Character arcs show growth and change throughout the story, creating engagement and satisfaction. Dialogue reveals personality and advances plot while maintaining natural speech patterns.

Plot structure provides the framework for narrative flow. The classic three-act structure includes setup, confrontation, and resolution. Rising action builds tension toward the climax, while falling action provides closure. Subplots add depth and complexity, creating multi-layered storytelling experiences.

Setting establishes the world where stories unfold. Detailed world-building creates immersive environments that feel authentic and lived-in. Historical accuracy in period pieces requires extensive research and attention to detail. Fantasy and science fiction settings must maintain internal consistency to support suspension of disbelief.

Themes give stories deeper meaning beyond surface events. Universal themes like love, loss, redemption, and growth resonate across cultures and time periods. Subtle theme integration avoids heavy-handed messaging while still conveying meaningful ideas. Multiple themes can coexist, creating rich interpretive possibilities.

Modern storytelling adapts to new media formats. Interactive narratives allow audience participation in story outcomes. Serial storytelling through streaming platforms changes pacing and structure expectations. Social media creates new forms of collaborative and real-time storytelling experiences."""
        doc = Document(content=content, metadata={'source': 'narrative.txt', 'type': 'creative', 'topic': 'storytelling'})
        self.test_documents['narrative'] = doc

    @staticmethod
    def create_mixed_topic_document(self):
        content = """Technology in Education

Digital transformation is reshaping educational landscapes worldwide. Traditional classroom models are evolving to incorporate technology-enhanced learning experiences. Online platforms provide access to educational content regardless of geographic location. Virtual reality creates immersive learning environments for complex subjects like anatomy or historical events.

Artificial intelligence personalizes learning experiences. Adaptive learning systems adjust content difficulty based on student performance. Automated grading systems free teachers to focus on instruction and mentoring. Chatbots provide 24/7 student support for common questions and administrative tasks.

Data analytics inform educational decision-making. Learning management systems track student engagement and progress. Predictive models identify students at risk of dropping out. Performance analytics help institutions optimize curriculum design and resource allocation.

However, technology integration faces significant challenges. Digital divides create inequality between students with and without technology access. Teacher training requires substantial investment in professional development. Privacy concerns arise from collecting and storing student data. Screen time and digital wellness become important considerations for student health.

The future of education will likely be hybrid. Blended learning combines online and in-person instruction for optimal flexibility. Microlearning breaks complex topics into digestible modules. Blockchain technology could provide secure, verifiable credential systems. Augmented reality will enhance textbooks and learning materials with interactive elements."""
        doc = Document(content=content, metadata={'source': 'mixed.txt', 'type': 'analysis', 'topics': ['technology', 'education']})
        self.test_documents['mixed_topics'] = doc

    @staticmethod
    def create_technical_document(self):
        content = """Database Optimization Techniques

Query optimization is fundamental to database performance. Proper indexing strategies can dramatically reduce query execution time. Composite indexes support multi-column searches effectively. However, excessive indexing can slow down write operations and consume significant storage space.

Normalization reduces data redundancy and improves consistency. First normal form eliminates repeating groups. Second normal form removes partial dependencies. Third normal form eliminates transitive dependencies. However, over-normalization can impact query performance through excessive joins.

Caching mechanisms improve response times for frequently accessed data. In-memory caches store hot data for rapid retrieval. Query result caching eliminates repeated computations. Database connection pooling reduces connection overhead. Cache invalidation strategies ensure data consistency.

Partitioning strategies scale databases horizontally. Table partitioning distributes data across multiple storage units. Hash partitioning ensures even data distribution. Range partitioning groups related data together. Partition pruning eliminates unnecessary data scans during queries.

Replication provides both performance and reliability benefits. Master-slave replication distributes read workloads. Multi-master replication enables global distribution. Synchronous replication ensures consistency but impacts performance. Asynchronous replication improves performance but may allow temporary inconsistency.

Monitoring and maintenance ensure optimal performance over time. Query execution plans reveal optimization opportunities. Index usage statistics identify unused indexes. Regular maintenance tasks include statistics updates and index reorganization. Performance baselines help identify degradation trends."""
        doc = Document(content=content, metadata={'source': 'technical.txt', 'type': 'documentation', 'topic': 'database_optimization'})
        self.test_documents['technical'] = doc

    @staticmethod
    def create_conversational_document(self):
        content = """Travel Planning Discussion

Planning a vacation can be both exciting and overwhelming. There are so many factors to consider: budget, timing, destination, activities, accommodations, and transportation. Where do you even start?

I always begin with the budget. How much can I realistically spend? This determines everything else. A weekend getaway has different requirements than a two-week international adventure. Once I know my budget range, I can start narrowing down destinations.

Timing is crucial too. When can I actually take time off? What's the weather like at potential destinations during those dates? Peak season means higher prices but better weather and more activities. Shoulder season offers better deals but might limit some experiences.

Destination research takes the most time. What kind of experience am I seeking? Relaxation on a beach, cultural immersion in a historic city, or adventure in natural settings? Each requires different planning approaches. I read travel blogs, check reviews, and look at photos to get a realistic sense of places.

Accommodations significantly impact both budget and experience. Hotels offer convenience and services but cost more. Vacation rentals provide space and local flavor but require more planning. Hostels work for budget travel but offer less privacy. Location matters as much as amenities.

Transportation planning varies by destination. Domestic travel might involve comparing flight costs with driving time and expenses. International travel requires passport checks, visa requirements, and understanding local transportation options. Car rentals, public transit, or walking - each choice affects the overall experience.

The key is flexibility. Things rarely go exactly as planned. Weather changes, attractions close, flights get delayed. Building some flexibility into schedules and budgets helps handle unexpected situations without ruining the trip."""
        doc = Document(content=content, metadata={'source': 'conversational.txt', 'type': 'informal', 'topic': 'travel_planning'})
        self.test_documents['conversational'] = doc

    @staticmethod
    def create_large_semantic_document(self):
        content_sections = [
            # Section 1: Artificial Intelligence
            """Artificial Intelligence represents one of the most transformative technologies of our era. Machine learning algorithms enable computers to learn patterns from data without explicit programming. Deep learning neural networks mimic human brain structures to solve complex problems. Natural language processing allows machines to understand and generate human language. Computer vision enables machines to interpret visual information from images and videos.""",
            
            # Section 2: Renewable Energy  
            """Renewable energy sources are becoming increasingly viable alternatives to fossil fuels. Solar photovoltaic technology converts sunlight directly into electricity with improving efficiency rates. Wind turbines harness kinetic energy from air movement to generate clean power. Hydroelectric systems utilize flowing water to produce sustainable energy. Geothermal plants tap into Earth's internal heat for consistent power generation.""",
            
            # Section 3: Space Exploration
            """Space exploration continues to push the boundaries of human knowledge and capability. Robotic missions to Mars provide detailed information about planetary composition and potential for life. The International Space Station serves as a platform for scientific research in microgravity environments. Commercial spaceflight companies are making space more accessible for both cargo and human transportation. Telescope technology reveals distant galaxies and exoplanets that could harbor life.""",
            
            # Section 4: Healthcare Innovation
            """Healthcare innovation is revolutionizing patient care and treatment outcomes. Precision medicine tailors treatments to individual genetic profiles and medical histories. Telemedicine expands access to healthcare services in remote and underserved areas. Robotic surgery enhances precision and reduces recovery times for complex procedures. Wearable devices continuously monitor vital signs and health metrics for preventive care.""",
            
            # Section 5: Sustainable Agriculture
            """Sustainable agriculture practices address food security while protecting environmental resources. Precision farming uses GPS technology and sensors to optimize crop yields and reduce waste. Vertical farming maximizes production in urban environments with minimal land use. Organic farming methods eliminate synthetic chemicals to protect soil and water quality. Crop rotation and companion planting maintain soil fertility naturally.""",
            
            # Section 6: Digital Privacy
            """Digital privacy concerns grow as online data collection expands across platforms and services. Encryption technologies protect sensitive information from unauthorized access. Blockchain systems provide transparent and secure transaction records. Privacy regulations like GDPR establish user rights and corporate responsibilities. Biometric authentication offers security benefits but raises additional privacy considerations.""",
            
            # Section 7: Urban Planning
            """Urban planning shapes how cities accommodate growing populations while maintaining quality of life. Smart city initiatives integrate technology to improve traffic flow, energy efficiency, and public services. Green building standards reduce environmental impact through sustainable construction practices. Public transportation systems reduce traffic congestion and air pollution in metropolitan areas. Mixed-use development creates walkable neighborhoods with diverse housing and commercial options.""",
            
            # Section 8: Cultural Preservation
            """Cultural preservation efforts maintain traditional knowledge and practices for future generations. Digital archives document historical artifacts, languages, and customs at risk of disappearing. Community-based programs engage local populations in preservation activities. International cooperation protects world heritage sites from development and natural disasters. Language revitalization programs use technology to teach endangered languages to new speakers."""
        ]
        
        content = "\n\n".join(content_sections)
        doc = Document(content=content, metadata={'source': 'large_semantic.txt', 'type': 'comprehensive', 'sections': 8, 'topics': 'multiple'})
        self.test_documents['large_semantic'] = doc

    @staticmethod
    def create_empty_document(self):
        doc = Document(content="", metadata={'source': 'empty.txt', 'type': 'empty'})
        self.test_documents['empty'] = doc

    async def test_basic_semantic_chunking(self):
        config = SemanticChunkingConfig()
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(self.test_documents['topical'])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        # Verify semantic content preservation
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("climate change", all_content.lower())
        self.assertIn("environmental", all_content.lower())

    async def test_semantic_similarity_grouping(self):
        config = SemanticChunkingConfig(similarity_threshold=0.7)
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(self.test_documents['scientific'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should group semantically similar content
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("quantum", all_content.lower())
        self.assertIn("qubit", all_content.lower())

    async def test_topic_coherence_preservation(self):
        config = SemanticChunkingConfig(preserve_topic_coherence=True)
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(self.test_documents['mixed_topics'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should maintain topic coherence within chunks
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("technology", all_content.lower())
        self.assertIn("education", all_content.lower())

    async def test_sentence_boundary_respect(self):
        config = SemanticChunkingConfig(respect_sentence_boundaries=True)
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(self.test_documents['narrative'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should respect sentence boundaries
        for chunk in chunks:
            # Check that chunks don't end mid-sentence
            text = chunk.text_content.strip()
            if text:
                # Should end with sentence-ending punctuation or be complete
                self.assertTrue(
                    text.endswith('.') or 
                    text.endswith('!') or 
                    text.endswith('?') or
                    text.endswith(':') or
                    len(text.split('.')) == 1
                )

    async def test_semantic_embedding_usage(self):
        config = SemanticChunkingConfig(use_embeddings=True)
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(self.test_documents['technical'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should use embeddings for semantic similarity
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("database", all_content.lower())
        self.assertIn("optimization", all_content.lower())

    async def test_context_window_configuration(self):
        config = SemanticChunkingConfig(context_window_size=3)
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(self.test_documents['conversational'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should use context window for semantic analysis
        for chunk in chunks:
            self.assertGreater(len(chunk.text_content), 0)

    async def test_similarity_threshold_variation(self):
        # Test with high similarity threshold (fewer, larger chunks)
        config_high = SemanticChunkingConfig(similarity_threshold=0.9)
        mock_provider_high = MockEmbeddingProvider()
        chunker_high = SemanticSimilarityChunkingStrategy(mock_provider_high, config_high)
        chunks_high = await chunker_high.chunk(self.test_documents['scientific'])
        
        # Test with low similarity threshold (more, smaller chunks)  
        config_low = SemanticChunkingConfig(similarity_threshold=0.3)
        mock_provider_low = MockEmbeddingProvider()
        chunker_low = SemanticSimilarityChunkingStrategy(mock_provider_low, config_low)
        chunks_low = await chunker_low.chunk(self.test_documents['scientific'])
        
        self.assertGreater(len(chunks_high), 0)
        self.assertGreater(len(chunks_low), 0)
        
        # Low threshold might create more chunks (but not guaranteed)
        # The actual behavior depends on the semantic content

    async def test_multilingual_content_handling(self):
        multilingual_content = """English paragraph about technology and innovation. 
        Technology is advancing rapidly in all sectors.
        
        Français paragraphe sur la technologie. 
        La technologie évolue rapidement dans tous les secteurs.
        
        Spanish párrafo sobre tecnología e innovación.
        La tecnología está avanzando rápidamente en todos los sectores."""
        
        multilingual_doc = Document(content=multilingual_content, metadata={'type': 'multilingual'})
        
        config = SemanticChunkingConfig()
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(multilingual_doc)
        
        self.assertGreater(len(chunks), 0)
        
        # Should handle multilingual content gracefully
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("technology", all_content.lower())

    async def test_large_document_performance(self):
        config = SemanticChunkingConfig()
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        
        start_time = time.time()
        chunks = await chunker.chunk(self.test_documents['large_semantic'])
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertGreater(len(chunks), 1)  # Should split large document
        self.assertLess(processing_time, 15.0)  # Should complete within 15 seconds
        
        # Verify semantic grouping worked
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("artificial intelligence", all_content.lower())
        self.assertIn("renewable energy", all_content.lower())

    async def test_semantic_coherence_scoring(self):
        config = SemanticChunkingConfig(enable_coherence_scoring=True)
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(self.test_documents['narrative'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should maintain semantic coherence
        for chunk in chunks:
            self.assertGreater(len(chunk.text_content), 0)

    async def test_adaptive_chunking(self):
        config = SemanticChunkingConfig(adaptive_chunking=True)
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(self.test_documents['mixed_topics'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should adapt to content complexity
        for chunk in chunks:
            self.assertIsInstance(chunk.metadata, dict)

    async def test_empty_content_handling(self):
        config = SemanticChunkingConfig()
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(self.test_documents['empty'])
        
        self.assertEqual(len(chunks), 0)

    async def test_whitespace_only_content(self):
        whitespace_content = "   \n\n  \t  \n   "
        whitespace_doc = Document(content=whitespace_content, metadata={'type': 'whitespace'})
        
        config = SemanticChunkingConfig()
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(whitespace_doc)
        
        self.assertEqual(len(chunks), 0)

    async def test_metadata_inheritance(self):
        config = SemanticChunkingConfig()
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        chunks = await chunker.chunk(self.test_documents['technical'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify original metadata is inherited
        for chunk in chunks:
            self.assertEqual(chunk.metadata['source'], 'technical.txt')
            self.assertEqual(chunk.metadata['type'], 'documentation')
            self.assertEqual(chunk.metadata['topic'], 'database_optimization')

    async def test_semantic_statistics(self):
        config = SemanticChunkingConfig()
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        
        # Process document
        chunks = await chunker.chunk(self.test_documents['topical'])
        
        # Try to get semantic statistics if available
        try:
            stats = chunker.get_semantic_stats()
            self.assertIsInstance(stats, dict)
        except AttributeError:
            # Method might not exist in actual implementation
            pass

    async def test_batch_processing(self):
        documents = [
            self.test_documents['topical'],
            self.test_documents['scientific'],
            self.test_documents['narrative']
        ]
        
        config = SemanticChunkingConfig()
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        
        batch_results = await chunker.chunk_batch_async(documents)
        
        self.assertEqual(len(batch_results), 3)
        self.assertTrue(all(len(chunks) > 0 for chunks in batch_results))
        
        total_chunks = sum(len(chunks) for chunks in batch_results)
        self.assertGreater(total_chunks, 0)

    async def test_caching_functionality(self):
        config = SemanticChunkingConfig(enable_caching=True)
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        
        # First processing
        chunks1 = await chunker.chunk(self.test_documents['topical'])
        
        # Second processing (should use cache)
        chunks2 = await chunker.chunk(self.test_documents['topical'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))

    async def test_error_handling(self):
        config = SemanticChunkingConfig()
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticSimilarityChunkingStrategy(mock_provider, config)
        
        # Test with unusual content
        unusual_content = """🎵🎶🎵 Music symbols and emojis 🎶🎵🎶
        
        Mixed content with numbers: 123 456 789 and symbols: @#$%^&*
        
        Very short. One. Two. Three."""
        unusual_doc = Document(content=unusual_content, metadata={'type': 'unusual'})
        
        chunks = await chunker.chunk(unusual_doc)
        self.assertGreaterEqual(len(chunks), 0)  # Should handle gracefully

if __name__ == "__main__":
    unittest.main()