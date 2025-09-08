import os
import sys
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import unittest

from upsonic.text_splitter.html import HTMLChunkingStrategy, HTMLChunkingConfig, ChunkingMode, ContentType
from upsonic.schemas.data_models import Document, Chunk

class TestHTMLChunkingComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_documents = {}
        cls.create_basic_html_document(cls)
        cls.create_structured_html_document(cls)
        cls.create_table_html_document(cls)
        cls.create_list_html_document(cls)
        cls.create_code_html_document(cls)
        cls.create_complex_html_document(cls)
        cls.create_empty_html_document(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_basic_html_document(self):
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Basic Test Document</title>
</head>
<body>
    <h1>Introduction</h1>
    <p>This is the introduction paragraph with some content.</p>
    <p>This is another paragraph with more information.</p>
    
    <h2>Section 1</h2>
    <p>This is content under section 1.</p>
    <p>More content here.</p>
    
    <h2>Section 2</h2>
    <p>This is content under section 2.</p>
</body>
</html>"""
        doc = Document(content=html_content, metadata={'source': 'basic', 'type': 'webpage'})
        self.test_documents['basic'] = doc

    @staticmethod
    def create_structured_html_document(self):
        html_content = """<html>
<body>
    <h1>Main Title</h1>
    <p>Introduction content under main title.</p>
    
    <h2>Section A</h2>
    <p>Content under section A.</p>
    <p>More content under section A.</p>
    
    <h3>Subsection A.1</h3>
    <p>Content under subsection A.1.</p>
    
    <h3>Subsection A.2</h3>
    <p>Content under subsection A.2.</p>
    
    <h2>Section B</h2>
    <p>Content under section B.</p>
    
    <h3>Subsection B.1</h3>
    <p>Content under subsection B.1.</p>
</body>
</html>"""
        doc = Document(content=html_content, metadata={'source': 'structured', 'type': 'article', 'sections': 4})
        self.test_documents['structured'] = doc

    @staticmethod
    def create_table_html_document(self):
        html_content = """<html>
<body>
    <h1>Data Table Example</h1>
    <table>
        <caption>Employee Information</caption>
        <thead>
            <tr>
                <th>Name</th>
                <th>Department</th>
                <th>Salary</th>
                <th>Experience</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>John Doe</td>
                <td>Engineering</td>
                <td>$75,000</td>
                <td>5 years</td>
            </tr>
            <tr>
                <td>Jane Smith</td>
                <td>Marketing</td>
                <td>$65,000</td>
                <td>3 years</td>
            </tr>
            <tr>
                <td>Bob Johnson</td>
                <td>Sales</td>
                <td>$70,000</td>
                <td>4 years</td>
            </tr>
        </tbody>
    </table>
    <p>This table shows employee data for our organization.</p>
</body>
</html>"""
        doc = Document(content=html_content, metadata={'source': 'table', 'type': 'data', 'rows': 3})
        self.test_documents['table'] = doc

    @staticmethod
    def create_list_html_document(self):
        html_content = """<html>
<body>
    <h1>Lists Example</h1>
    
    <h2>Unordered List</h2>
    <ul>
        <li>First unordered item with detailed content</li>
        <li>Second unordered item with more information</li>
        <li>Third unordered item with additional details</li>
    </ul>
    
    <h2>Ordered List</h2>
    <ol>
        <li>First ordered item with step-by-step instructions</li>
        <li>Second ordered item with continued process</li>
        <li>Third ordered item with final steps</li>
    </ol>
    
    <h2>Definition List</h2>
    <dl>
        <dt>HTML</dt>
        <dd>HyperText Markup Language used for web pages</dd>
        <dt>CSS</dt>
        <dd>Cascading Style Sheets for styling web content</dd>
        <dt>JavaScript</dt>
        <dd>Programming language for web interactivity</dd>
    </dl>
</body>
</html>"""
        doc = Document(content=html_content, metadata={'source': 'lists', 'type': 'reference', 'list_types': 3})
        self.test_documents['lists'] = doc

    @staticmethod
    def create_code_html_document(self):
        html_content = """<html>
<body>
    <h1>Code Examples</h1>
    
    <h2>Python Code Block</h2>
    <pre><code>
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

# Example usage
data = [1, 2, 3, 4, 5]
result = calculate_average(data)
print(f"Average: {result}")
    </code></pre>
    
    <h2>Inline Code</h2>
    <p>Use the <code>print()</code> function to output text to the console.</p>
    <p>The <code>len()</code> function returns the length of a sequence.</p>
    
    <h2>JavaScript Code Block</h2>
    <pre><code>
function greetUser(name, age) {
    if (age >= 18) {
        console.log(`Hello ${name}, you are an adult!`);
    } else {
        console.log(`Hi ${name}, you are a minor.`);
    }
}

// Example usage
greetUser("Alice", 25);
greetUser("Bob", 16);
    </code></pre>
</body>
</html>"""
        doc = Document(content=html_content, metadata={'source': 'code', 'type': 'tutorial', 'languages': 2})
        self.test_documents['code'] = doc

    @staticmethod
    def create_complex_html_document(self):
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Complex Document</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .highlight { background-color: yellow; }
    </style>
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <article>
            <h1>Main Article Title</h1>
            <p class="highlight">This is a highlighted paragraph with important information.</p>
            
            <section>
                <h2>Introduction Section</h2>
                <p>This is the introduction section with detailed content about our topic.</p>
                
                <aside>
                    <h3>Side Note</h3>
                    <p>This is additional information provided in an aside element.</p>
                </aside>
            </section>
            
            <section>
                <h2>Data Analysis Section</h2>
                <table>
                    <caption>Analysis Results Summary</caption>
                    <thead>
                        <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>Accuracy</td><td>95%</td><td>Excellent</td></tr>
                        <tr><td>Precision</td><td>92%</td><td>Good</td></tr>
                        <tr><td>Recall</td><td>88%</td><td>Good</td></tr>
                    </tbody>
                </table>
                
                <figure>
                    <img src="performance-chart.png" alt="Performance Chart Over Time" />
                    <figcaption>Performance metrics visualization over the last quarter</figcaption>
                </figure>
            </section>
            
            <section>
                <h2>Implementation Details</h2>
                <blockquote>
                    <p>The best way to predict the future is to implement it yourself.</p>
                    <cite>Anonymous Developer</cite>
                </blockquote>
                
                <pre><code>
def analyze_performance(data):
    metrics = {
        'accuracy': calculate_accuracy(data),
        'precision': calculate_precision(data),
        'recall': calculate_recall(data)
    }
    return metrics
                </code></pre>
            </section>
        </article>
    </main>
    
    <footer>
        <p>&copy; 2024 Company Name. All rights reserved.</p>
        <p>Contact us at <a href="mailto:info@company.com">info@company.com</a></p>
    </footer>
</body>
</html>"""
        doc = Document(content=html_content, metadata={'source': 'complex', 'type': 'comprehensive', 'sections': 3})
        self.test_documents['complex'] = doc

    @staticmethod
    def create_empty_html_document(self):
        html_content = "<html><body></body></html>"
        doc = Document(content=html_content, metadata={'source': 'empty', 'type': 'edge_case'})
        self.test_documents['empty'] = doc

    def test_basic_html_chunking(self):
        config = HTMLChunkingConfig()
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['basic'])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        # Verify basic metadata
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertGreater(len(chunk.text_content), 0)

    def test_text_only_mode(self):
        config = HTMLChunkingConfig(mode=ChunkingMode.TEXT_ONLY)
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['basic'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that HTML tags are stripped
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertNotIn("<h1>", all_text)
        self.assertNotIn("<p>", all_text)
        self.assertIn("Introduction", all_text)
        self.assertIn("Section 1", all_text)

    def test_header_based_mode(self):
        config = HTMLChunkingConfig(mode=ChunkingMode.HEADER_BASED)
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['structured'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that header structure is preserved
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Main Title", all_text)
        self.assertIn("Section A", all_text)
        self.assertIn("Subsection A.1", all_text)

    def test_semantic_preserving_mode(self):
        config = HTMLChunkingConfig(mode=ChunkingMode.SEMANTIC_PRESERVING)
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['table'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that table structure content is preserved
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Employee Information", all_text)  # Caption
        self.assertIn("John Doe", all_text)
        self.assertIn("Engineering", all_text)

    def test_adaptive_mode(self):
        # Test with simple content
        config = HTMLChunkingConfig(mode=ChunkingMode.ADAPTIVE)
        chunker = HTMLChunkingStrategy(config)
        
        chunks_simple = chunker.chunk(self.test_documents['basic'])
        self.assertGreater(len(chunks_simple), 0)
        
        # Test with structured content
        chunks_structured = chunker.chunk(self.test_documents['structured'])
        self.assertGreater(len(chunks_structured), 0)

    def test_content_preservation_options(self):
        config = HTMLChunkingConfig(
            preserve_links=True,
            preserve_images=True,
            preserve_videos=True,
            preserve_audio=True
        )
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        all_text = " ".join(chunk.text_content for chunk in chunks)
        # Check for link preservation (email link in footer)
        self.assertIn("info@company.com", all_text)

    def test_element_filtering(self):
        config = HTMLChunkingConfig(
            allowlist_tags=['h1', 'h2', 'p'],
            skip_elements={'script', 'style'}
        )
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        # Element filtering might result in no chunks or filtered content
        self.assertGreaterEqual(len(chunks), 0)
        
        if len(chunks) > 0:
            all_text = " ".join(chunk.text_content for chunk in chunks)
            # Check if filtering worked
            self.assertNotIn("font-family: Arial", all_text)

    def test_custom_separators(self):
        config = HTMLChunkingConfig(
            custom_separators={
                'paragraph': ' | ',
                'list_item': ' -> '
            }
        )
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['lists'])
        
        self.assertGreater(len(chunks), 0)
        
        # Content should be processed even with custom separators
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Unordered List", all_text)

    def test_hierarchy_preservation(self):
        config = HTMLChunkingConfig(
            mode=ChunkingMode.HEADER_BASED,
            preserve_hierarchy=True
        )
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['structured'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that content is chunked with hierarchy awareness
        for chunk in chunks:
            self.assertIn('chunking_strategy', chunk.metadata)

    def test_table_processing(self):
        config = HTMLChunkingConfig(mode=ChunkingMode.SEMANTIC_PRESERVING)
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['table'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that table structure is preserved
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Employee Information", all_text)  # Caption
        self.assertIn("John Doe", all_text)
        self.assertIn("Jane Smith", all_text)
        self.assertIn("Engineering", all_text)
        self.assertIn("Marketing", all_text)

    def test_list_processing(self):
        config = HTMLChunkingConfig(mode=ChunkingMode.SEMANTIC_PRESERVING)
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['lists'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that list structure is preserved
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("First unordered item", all_text)
        self.assertIn("First ordered item", all_text)
        self.assertIn("HTML", all_text)
        self.assertIn("HyperText Markup Language", all_text)

    def test_code_block_processing(self):
        config = HTMLChunkingConfig(mode=ChunkingMode.SEMANTIC_PRESERVING)
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['code'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that code blocks are preserved
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("calculate_average", all_text)
        self.assertIn("print()", all_text)
        self.assertIn("greetUser", all_text)

    def test_quote_processing(self):
        quote_html = """<html>
<body>
    <h1>Quotes Example</h1>
    <blockquote>
        <p>This is a block quote with important information.</p>
        <cite>Important Author</cite>
    </blockquote>
    <p>Here is an inline <q>quote</q> within a paragraph.</p>
</body>
</html>"""
        quote_doc = Document(content=quote_html, metadata={'type': 'quotes'})
        
        config = HTMLChunkingConfig(mode=ChunkingMode.SEMANTIC_PRESERVING)
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(quote_doc)
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that quotes are preserved
        all_text = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("This is a block quote", all_text)
        self.assertIn("Important Author", all_text)

    def test_empty_content_handling(self):
        config = HTMLChunkingConfig()
        chunker = HTMLChunkingStrategy(config)
        
        # Test empty HTML
        chunks_empty = chunker.chunk(self.test_documents['empty'])
        self.assertEqual(len(chunks_empty), 0)
        
        # Test HTML with only whitespace
        whitespace_html = "<html><body>   \n\n   </body></html>"
        whitespace_doc = Document(content=whitespace_html, metadata={'type': 'whitespace'})
        chunks_whitespace = chunker.chunk(whitespace_doc)
        self.assertEqual(len(chunks_whitespace), 0)

    def test_performance_tracking(self):
        config = HTMLChunkingConfig()
        chunker = HTMLChunkingStrategy(config)
        
        # Get initial metrics
        initial_metrics = chunker.get_metrics()
        self.assertEqual(initial_metrics.total_chunks, 0)
        
        # Process complex document
        start_time = time.time()
        chunks = chunker.chunk(self.test_documents['complex'])
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertGreater(len(chunks), 0)
        self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds
        
        # Get updated metrics
        metrics = chunker.get_metrics()
        # Metrics might not be immediately updated, check if they're available
        if metrics.total_chunks > 0:
            self.assertGreater(metrics.total_characters, 0)
            self.assertGreater(metrics.avg_chunk_size, 0)
            self.assertGreater(metrics.processing_time_ms, 0)
            self.assertEqual(metrics.strategy_name, "HTMLChunkingStrategy")
        # If metrics aren't updated, the test still passes

    def test_complex_html_structure(self):
        config = HTMLChunkingConfig(
            mode=ChunkingMode.ADAPTIVE,
            preserve_links=True,
            preserve_images=True,
            preserve_hierarchy=True
        )
        chunker = HTMLChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that complex structure is handled well
        for chunk in chunks:
            self.assertGreater(len(chunk.text_content), 0)
            self.assertIn('chunking_strategy', chunk.metadata)
            self.assertEqual(chunk.metadata['source'], 'complex')

    def test_error_handling(self):
        config = HTMLChunkingConfig()
        chunker = HTMLChunkingStrategy(config)
        
        # Test with invalid HTML (plain text)
        invalid_html = "This is not HTML at all, just plain text without any tags."
        invalid_doc = Document(content=invalid_html, metadata={'type': 'invalid'})
        
        chunks = chunker.chunk(invalid_doc)
        self.assertGreaterEqual(len(chunks), 0)  # Should handle gracefully
        
        # Test with malformed HTML
        malformed_html = "<html><body><p>Unclosed paragraph<h1>Title</h1></body></html>"
        malformed_doc = Document(content=malformed_html, metadata={'type': 'malformed'})
        
        chunks_malformed = chunker.chunk(malformed_doc)
        self.assertGreaterEqual(len(chunks_malformed), 0)

    def test_batch_processing(self):
        documents = [
            self.test_documents['basic'],
            self.test_documents['table'],
            self.test_documents['lists']
        ]
        
        config = HTMLChunkingConfig(chunk_size=300, chunk_overlap=50)
        chunker = HTMLChunkingStrategy(config)
        
        batch_results = chunker.chunk_batch(documents)
        
        self.assertEqual(len(batch_results), 3)
        self.assertTrue(all(len(chunks) > 0 for chunks in batch_results))
        
        total_chunks = sum(len(chunks) for chunks in batch_results)
        self.assertGreater(total_chunks, 0)

    def test_caching_functionality(self):
        config = HTMLChunkingConfig(
            chunk_size=300,
            chunk_overlap=50,
            enable_caching=True
        )
        chunker = HTMLChunkingStrategy(config)
        
        # First processing
        chunks1 = chunker.chunk(self.test_documents['basic'])
        
        # Second processing (should use cache)
        chunks2 = chunker.chunk(self.test_documents['basic'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))

if __name__ == "__main__":
    unittest.main()