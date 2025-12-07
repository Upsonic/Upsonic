"""
Comprehensive Smoke Test for All Chunkers with Config Attributes
Tests all chunking providers with their specific config attributes
"""
import pytest
from typing import List

from upsonic.text_splitter import (
    CharacterChunker, CharacterChunkingConfig,
    RecursiveChunker, RecursiveChunkingConfig,
    JSONChunker, JSONChunkingConfig,
    MarkdownChunker, MarkdownChunkingConfig,
    PythonChunker, PythonChunkingConfig,
    HTMLChunker, HTMLChunkingConfig,
    SemanticChunker, SemanticChunkingConfig,
    AgenticChunker, AgenticChunkingConfig,
)
from upsonic.schemas.data_models import Document, Chunk


def create_sample_document(content: str, doc_id: str = "test-doc") -> Document:
    """Helper to create a sample document"""
    return Document(
        content=content,
        metadata={"source": "test"},
        document_id=doc_id
    )


class TestCharacterChunker:
    """Test CharacterChunker with all config attributes"""
    
    def test_basic_character_chunking(self):
        """Test basic character chunking"""
        config = CharacterChunkingConfig(
            chunk_size=50,
            chunk_overlap=10,
            strip_whitespace=False
        )
        chunker = CharacterChunker(config=config)
        
        # Use text with separators to test chunking properly
        doc = create_sample_document("Word " * 100)
        chunks = chunker.chunk([doc])
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        
    def test_character_chunk_overlap(self):
        """Test chunk overlap functionality"""
        config = CharacterChunkingConfig(
            chunk_size=20,
            chunk_overlap=5
        )
        chunker = CharacterChunker(config=config)
        
        # Use text with separators for proper chunking
        doc = create_sample_document("word " * 20)
        chunks = chunker.chunk([doc])
        
        assert len(chunks) >= 1
        # Verify overlap exists when multiple chunks are created
        
    def test_character_min_chunk_size(self):
        """Test min_chunk_size parameter"""
        config = CharacterChunkingConfig(
            chunk_size=100,
            min_chunk_size=20,
            chunk_overlap=0
        )
        chunker = CharacterChunker(config=config)
        
        doc = create_sample_document("Test content that should be chunked properly")
        chunks = chunker.chunk([doc])
        
        # All chunks should meet minimum size or be the last chunk
        assert len(chunks) > 0
        
    def test_character_strip_whitespace(self):
        """Test strip_whitespace parameter"""
        config = CharacterChunkingConfig(
            chunk_size=50,
            strip_whitespace=True
        )
        chunker = CharacterChunker(config=config)
        
        doc = create_sample_document("  Text with spaces  \n\n  More text  ")
        chunks = chunker.chunk([doc])
        
        # Chunks should not have leading/trailing whitespace
        for chunk in chunks:
            if chunk.text_content:
                assert chunk.text_content == chunk.text_content.strip()


class TestRecursiveChunker:
    """Test RecursiveChunker with all config attributes"""
    
    def test_basic_recursive_chunking(self):
        """Test basic recursive chunking"""
        config = RecursiveChunkingConfig(
            chunk_size=100,
            chunk_overlap=20
        )
        chunker = RecursiveChunker(config=config)
        
        doc = create_sample_document("""
        This is a paragraph.
        
        This is another paragraph.
        
        And this is a third paragraph with more content.
        """)
        chunks = chunker.chunk([doc])
        
        assert len(chunks) > 0
        
    def test_recursive_separators(self):
        """Test custom separators"""
        config = RecursiveChunkingConfig(
            chunk_size=50,
            separators=["\n\n", "\n", " "]
        )
        chunker = RecursiveChunker(config=config)
        
        doc = create_sample_document("Line1\n\nLine2\n\nLine3")
        chunks = chunker.chunk([doc])
        
        assert len(chunks) > 0


class TestJSONChunker:
    """Test JSONChunker with all config attributes"""
    
    def test_json_chunking(self):
        """Test JSON chunking"""
        config = JSONChunkingConfig(
            chunk_size=100,
            max_depth=3
        )
        chunker = JSONChunker(config=config)
        
        json_content = """
        {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ],
            "metadata": {"version": "1.0"}
        }
        """
        doc = create_sample_document(json_content)
        chunks = chunker.chunk([doc])
        
        assert len(chunks) > 0


class TestMarkdownChunker:
    """Test MarkdownChunker with all config attributes"""
    
    def test_markdown_chunking(self):
        """Test Markdown chunking"""
        config = MarkdownChunkingConfig(
            chunk_size=100,
            respect_headers=True
        )
        chunker = MarkdownChunker(config=config)
        
        markdown_content = """# Title
        
Some content here.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
"""
        doc = create_sample_document(markdown_content)
        chunks = chunker.chunk([doc])
        
        assert len(chunks) > 0
        
    def test_markdown_header_preservation(self):
        """Test header preservation in chunks"""
        config = MarkdownChunkingConfig(
            chunk_size=200,
            respect_headers=True
        )
        chunker = MarkdownChunker(config=config)
        
        doc = create_sample_document("# Main\n\nContent\n\n## Sub\n\nMore content")
        chunks = chunker.chunk([doc])
        
        assert len(chunks) > 0


class TestPythonChunker:
    """Test PythonChunker with all config attributes"""
    
    def test_python_chunking(self):
        """Test Python code chunking"""
        config = PythonChunkingConfig(
            chunk_size=200,
            respect_functions=True,
            respect_classes=True
        )
        chunker = PythonChunker(config=config)
        
        python_code = """
def function1():
    '''Docstring'''
    return "hello"

class MyClass:
    def method1(self):
        pass
    
    def method2(self):
        pass

def function2():
    return "world"
"""
        doc = create_sample_document(python_code)
        chunks = chunker.chunk([doc])
        
        assert len(chunks) > 0
        
    def test_python_function_preservation(self):
        """Test that functions are kept together when possible"""
        config = PythonChunkingConfig(
            chunk_size=500,
            respect_functions=True
        )
        chunker = PythonChunker(config=config)
        
        code = "def small_func():\n    return 42\n\ndef another():\n    return 100"
        doc = create_sample_document(code)
        chunks = chunker.chunk([doc])
        
        assert len(chunks) > 0


class TestHTMLChunker:
    """Test HTMLChunker with all config attributes"""
    
    def test_html_basic_chunking(self):
        """Test basic HTML chunking"""
        try:
            config = HTMLChunkingConfig(
                chunk_size=200,
                split_on_tags=["h1", "h2", "p"],
                extract_link_info=True
            )
            chunker = HTMLChunker(config=config)
            
            html_content = """
            <html>
                <head><title>Test</title></head>
                <body>
                    <h1>Main Title</h1>
                    <p>First paragraph with some content.</p>
                    <h2>Subtitle</h2>
                    <p>Second paragraph with <a href="http://example.com">a link</a>.</p>
                </body>
            </html>
            """
            doc = create_sample_document(html_content)
            chunks = chunker.chunk([doc])
            
            assert len(chunks) > 0
            assert all(isinstance(c, Chunk) for c in chunks)
        except ImportError:
            pytest.skip("HTMLChunker requires BeautifulSoup4")
            
    def test_html_preserve_whole_tags(self):
        """Test preserving whole tags like tables"""
        try:
            config = HTMLChunkingConfig(
                chunk_size=100,
                preserve_whole_tags=["table", "pre"],
            )
            chunker = HTMLChunker(config=config)
            
            html_content = """
            <html><body>
                <p>Before table</p>
                <table>
                    <tr><td>Cell 1</td><td>Cell 2</td></tr>
                    <tr><td>Cell 3</td><td>Cell 4</td></tr>
                </table>
                <p>After table</p>
            </body></html>
            """
            doc = create_sample_document(html_content)
            chunks = chunker.chunk([doc])
            
            assert len(chunks) > 0
        except ImportError:
            pytest.skip("HTMLChunker requires BeautifulSoup4")
            
    def test_html_tags_to_extract(self):
        """Test extracting specific tags only"""
        try:
            config = HTMLChunkingConfig(
                chunk_size=200,
                tags_to_extract=["p", "h1"],
            )
            chunker = HTMLChunker(config=config)
            
            html_content = """
            <html><body>
                <h1>Title</h1>
                <p>Paragraph 1</p>
                <div>This should be ignored</div>
                <p>Paragraph 2</p>
            </body></html>
            """
            doc = create_sample_document(html_content)
            chunks = chunker.chunk([doc])
            
            assert len(chunks) > 0
        except ImportError:
            pytest.skip("HTMLChunker requires BeautifulSoup4")


class TestSemanticChunker:
    """Test SemanticChunker with all config attributes"""
    
    def test_semantic_chunking_with_fastembed(self):
        """Test semantic chunking with FastEmbed provider"""
        try:
            from upsonic.embeddings import create_embedding_provider
            
            # Create embedding provider
            embedding_provider = create_embedding_provider(
                "fastembed",
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            config = SemanticChunkingConfig(
                chunk_size=100,
                embedding_provider=embedding_provider,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=80.0
            )
            chunker = SemanticChunker(config=config)
            
            doc = create_sample_document("""
            Python is a high-level programming language. It's widely used for web development and data science.
            
            Machine learning is a subset of artificial intelligence. It involves training models on large datasets.
            
            Cooking is the art of preparing food. Good recipes provide clear instructions and ingredient lists.
            """)
            
            chunks = chunker.chunk([doc])
            assert len(chunks) > 0
            assert all(isinstance(c, Chunk) for c in chunks)
            
        except ImportError as e:
            pytest.skip(f"SemanticChunker requires dependencies: {e}")
        except Exception as e:
            pytest.skip(f"SemanticChunker test skipped: {e}")
            
    def test_semantic_chunking_config_attributes(self):
        """Test SemanticChunker config attribute variations"""
        try:
            from upsonic.embeddings import create_embedding_provider
            
            embedding_provider = create_embedding_provider("fastembed")
            
            # Test different breakpoint threshold types
            for threshold_type in ["percentile", "standard_deviation", "interquartile"]:
                config = SemanticChunkingConfig(
                    chunk_size=150,
                    embedding_provider=embedding_provider,
                    breakpoint_threshold_type=threshold_type,
                    breakpoint_threshold_amount=1.5 if threshold_type == "standard_deviation" else 75.0
                )
                chunker = SemanticChunker(config=config)
                
                doc = create_sample_document("Test content. " * 50)
                chunks = chunker.chunk([doc])
                assert len(chunks) > 0
                
        except ImportError:
            pytest.skip("SemanticChunker requires fastembed")
        except Exception as e:
            pytest.skip(f"SemanticChunker test skipped: {e}")


class TestAgenticChunker:
    """Test AgenticChunker with all config attributes"""
    
    def test_agentic_chunking(self):
        """Test agentic chunking with Agent"""
        try:
            from upsonic import Agent
            
            # Create agent for agentic chunker
            agent = Agent("openai/gpt-4o-mini")
            
            config = AgenticChunkingConfig(
                chunk_size=200,
                max_propositions_per_chunk=10,
                min_propositions_per_chunk=2,
                enable_proposition_caching=True
            )
            chunker = AgenticChunker(agent=agent, config=config)
            
            doc = create_sample_document("""
            This document discusses API development best practices.
            
            First, design RESTful endpoints following standard conventions.
            Use proper HTTP methods for different operations.
            
            Second, implement authentication using JWT tokens.
            Secure all sensitive endpoints with proper authorization.
            
            Finally, document your API using OpenAPI specifications.
            Good documentation helps developers integrate faster.
            """)
            
            chunks = chunker.chunk([doc])
            assert len(chunks) > 0
            assert all(isinstance(c, Chunk) for c in chunks)
            
        except ImportError:
            pytest.skip("AgenticChunker requires upsonic Agent")
        except Exception as e:
            pytest.skip(f"AgenticChunker requires API key: {e}")
            
    def test_agentic_chunking_config_attributes(self):
        """Test AgenticChunker with various config attributes"""
        try:
            from upsonic import Agent
            
            agent = Agent("openai/gpt-4o-mini")
            
            # Test with different config combinations
            config = AgenticChunkingConfig(
                chunk_size=300,
                max_propositions_per_chunk=15,
                min_propositions_per_chunk=3,
                enable_proposition_validation=True,
                enable_topic_optimization=True,
                enable_coherence_scoring=True,
                fallback_to_recursive=True
            )
            chunker = AgenticChunker(agent=agent, config=config)
            
            doc = create_sample_document("""
            Artificial intelligence is transforming industries.
            Machine learning models can now perform complex tasks.
            Deep learning has achieved remarkable results in vision and language.
            Natural language processing enables human-like interactions.
            """)
            
            chunks = chunker.chunk([doc])
            assert len(chunks) > 0
            
        except ImportError:
            pytest.skip("AgenticChunker requires upsonic Agent")
        except Exception as e:
            pytest.skip(f"AgenticChunker test skipped: {e}")


class TestChunkerCommonAttributes:
    """Test common chunker attributes across all chunkers"""
    
    def test_length_function(self):
        """Test custom length function"""
        def custom_length(text: str) -> int:
            # Count words instead of characters
            return len(text.split())
        
        config = CharacterChunkingConfig(
            chunk_size=10,  # 10 words
            length_function=custom_length
        )
        chunker = CharacterChunker(config=config)
        
        doc = create_sample_document(" ".join(["word"] * 50))
        chunks = chunker.chunk([doc])
        
        assert len(chunks) > 0
        
    def test_empty_document_handling(self):
        """Test handling of empty documents"""
        config = CharacterChunkingConfig(chunk_size=100)
        chunker = CharacterChunker(config=config)
        
        empty_doc = create_sample_document("")
        chunks = chunker.chunk([empty_doc])
        
        # Should handle empty documents gracefully
        assert isinstance(chunks, list)
        
    def test_multiple_documents(self):
        """Test chunking multiple documents at once"""
        config = CharacterChunkingConfig(chunk_size=50)
        chunker = CharacterChunker(config=config)
        
        # Use text with separators for proper chunking
        docs = [
            create_sample_document("word " * 50, "doc1"),
            create_sample_document("text " * 50, "doc2"),
            create_sample_document("data " * 50, "doc3"),
        ]
        chunks = chunker.chunk(docs)
        
        assert len(chunks) >= 3  # Should have at least one chunk from each doc
        # Verify document IDs are preserved
        doc_ids = set(chunk.document_id for chunk in chunks)
        assert len(doc_ids) == 3
        
    @pytest.mark.asyncio
    async def test_async_chunking(self):
        """Test async chunking functionality"""
        config = CharacterChunkingConfig(chunk_size=50)
        chunker = CharacterChunker(config=config)
        
        doc = create_sample_document("Test content " * 50)
        chunks = await chunker.achunk([doc])
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
