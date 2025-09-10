import os
import sys
import tempfile
import time
from pathlib import Path
import unittest

from upsonic.loaders.html import HTMLLoader
from upsonic.loaders.config import HTMLLoaderConfig
from upsonic.schemas.data_models import Document

try:
    from bs4 import BeautifulSoup
    import requests
except ImportError:
    raise ImportError("beautifulsoup4 and requests are not installed. Please install them with: pip install beautifulsoup4 requests")

class TestHTMLLoaderComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_files = {}
        cls.create_basic_html(cls)
        cls.create_table_html(cls)
        cls.create_complex_html(cls)
        cls.create_metadata_html(cls)
        cls.create_empty_html(cls)
        cls.create_special_formatting_html(cls)
        cls.create_large_html(cls)
        cls.create_malformed_html(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_basic_html(self):
        file_path = os.path.join(self.temp_dir, "basic_test.html")
        html_content = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Basic Test Document</title>
    <meta name=\"description\" content=\"A basic test document for HTML loader\">
    <meta name=\"author\" content=\"Test Author\">
</head>
<body>
    <h1>Welcome to Our Website</h1>
    <p>This is a basic test document for the HTML loader.</p>
    <p>It contains multiple paragraphs with different content.</p>
    <h2>Section 1: Introduction</h2>
    <p>This is the introduction section with some important information.</p>
    <h3>Subsection 1.1</h3>
    <p>Here are the detailed information and specifications.</p>
    <h2>Section 2: Features</h2>
    <ul>
        <li>Feature 1: Text extraction</li>
        <li>Feature 2: Structure preservation</li>
        <li>Feature 3: Metadata extraction</li>
    </ul>
    <h2>Section 3: Conclusion</h2>
    <p>This concludes our basic test document.</p>
</body>
</html>"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.test_files['basic'] = file_path

    @staticmethod
    def create_table_html(self):
        file_path = os.path.join(self.temp_dir, "table_test.html")
        html_content = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Table Test Document</title>
    <meta name=\"description\" content=\"HTML document with tables for testing\">
</head>
<body>
    <h1>Table Test Document</h1>
    <p>This document contains various tables for testing.</p>
    <h2>Employee Data</h2>
    <table border=\"1\">
        <thead>
            <tr>
                <th>Name</th>
                <th>Age</th>
                <th>Department</th>
                <th>Salary</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>John Doe</td>
                <td>30</td>
                <td>Engineering</td>
                <td>$75,000</td>
            </tr>
            <tr>
                <td>Jane Smith</td>
                <td>25</td>
                <td>Marketing</td>
                <td>$65,000</td>
            </tr>
            <tr>
                <td>Bob Johnson</td>
                <td>35</td>
                <td>Sales</td>
                <td>$70,000</td>
            </tr>
        </tbody>
    </table>
    <h2>Product Information</h2>
    <table border=\"1\">
        <tr>
            <th>Product</th>
            <th>Price</th>
            <th>Stock</th>
            <th>Category</th>
        </tr>
        <tr>
            <td>Laptop</td>
            <td>$999</td>
            <td>50</td>
            <td>Electronics</td>
        </tr>
        <tr>
            <td>Book</td>
            <td>$15</td>
            <td>200</td>
            <td>Education</td>
        </tr>
    </table>
</body>
</html>"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.test_files['table'] = file_path

    @staticmethod
    def create_complex_html(self):
        file_path = os.path.join(self.temp_dir, "complex_test.html")
        html_content = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Complex Document Structure</title>
    <meta name=\"description\" content=\"Complex HTML document with mixed content\">
    <meta name=\"keywords\" content=\"html, test, complex, structure\">
    <meta name=\"author\" content=\"Complex Test Author\">
    <style>
        .highlight { background-color: yellow; }
        .important { font-weight: bold; }
    </style>
    <script>
        function testFunction() {
            console.log(\"This is a test script\");
        }
    </script>
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href=\"#home\">Home</a></li>
                <li><a href=\"#about\">About</a></li>
                <li><a href=\"#contact\">Contact</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <article>
            <h1>Complex Document Structure</h1>
            <p>This is a complex document that tests various HTML features.</p>
            <section>
                <h2>Executive Summary</h2>
                <p>This section contains the executive summary of our findings.</p>
                <p class=\"important\">Key points include:</p>
                <ol>
                    <li>First important point</li>
                    <li>Second important point</li>
                    <li>Third important point</li>
                </ol>
            </section>
            <section>
                <h2>Data Analysis</h2>
                <p>The following table shows our analysis results:</p>
                <table border=\"1\">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Revenue</td>
                            <td>$1,000,000</td>
                            <td class=\"highlight\">Good</td>
                        </tr>
                        <tr>
                            <td>Costs</td>
                            <td>$750,000</td>
                            <td>Acceptable</td>
                        </tr>
                        <tr>
                            <td>Profit</td>
                            <td>$250,000</td>
                            <td class=\"highlight\">Excellent</td>
                        </tr>
                    </tbody>
                </table>
            </section>
            <section>
                <h2>Images and Links</h2>
                <p>Here are some sample links and images:</p>
                <p>
                    <a href=\"https://example.com\">External Link</a> |
                    <a href=\"#internal\">Internal Link</a> |
                    <a href=\"mailto:test@example.com\">Email Link</a>
                </p>
                <img src=\"https://via.placeholder.com/150\" alt=\"Sample Image\" title=\"Placeholder Image\">
                <img src=\"/local/image.jpg\" alt=\"Local Image\">
            </section>
            <section>
                <h2>Conclusion</h2>
                <p>Based on our analysis, we recommend the following actions.</p>
                <p>This concludes our comprehensive report.</p>
            </section>
        </article>
    </main>
    <footer>
        <p>&copy; 2024 Test Company. All rights reserved.</p>
    </footer>
</body>
</html>"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.test_files['complex'] = file_path

    @staticmethod
    def create_metadata_html(self):
        file_path = os.path.join(self.temp_dir, "metadata_test.html")
        html_content = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Metadata Test Document</title>
    <meta name=\"description\" content=\"HTML document with rich metadata for testing\">
    <meta name=\"keywords\" content=\"html, metadata, test, loader, extraction\">
    <meta name=\"author\" content=\"Metadata Test Author\">
    <meta name=\"robots\" content=\"index, follow\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <meta property=\"og:title\" content=\"Metadata Test Document\">
    <meta property=\"og:description\" content=\"Testing metadata extraction\">
    <meta property=\"og:type\" content=\"website\">
    <meta http-equiv=\"X-UA-Compatible\" content=\"IE=edge\">
    <link rel=\"canonical\" href=\"https://example.com/metadata-test\">
    <link rel=\"stylesheet\" href=\"styles.css\">
    <script src=\"script.js\"></script>
</head>
<body>
    <h1>Metadata Test Document</h1>
    <p>This document contains rich metadata for testing.</p>
    <p>The loader should extract all document properties.</p>
    <h2>Document Information</h2>
    <ul>
        <li>Title: Metadata Test Document</li>
        <li>Author: Metadata Test Author</li>
        <li>Language: English</li>
        <li>Description: HTML document with rich metadata for testing</li>
    </ul>
</body>
</html>"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.test_files['metadata'] = file_path

    @staticmethod
    def create_empty_html(self):
        file_path = os.path.join(self.temp_dir, "empty_test.html")
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Empty Document</title>
</head>
<body>
</body>
</html>"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.test_files['empty'] = file_path

    @staticmethod
    def create_special_formatting_html(self):
        file_path = os.path.join(self.temp_dir, "special_formatting_test.html")
        html_content = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Special Formatting Test</title>
</head>
<body>
    <h1>Special Formatting Test</h1>
    <p>This document tests special formatting and edge cases.</p>
    <h2>Special Characters</h2>
    <p>Special characters: é, ñ, ü, ç, €, £, ¥, ©, ®, ™</p>
    <p>Numbers: 123, 45.67, 1,000,000, 1.23e-4</p>
    <p>Symbols: @#$%^&*()_+-=[]{}|;:,.<>?</p>
    <h2>Whitespace and Formatting</h2>
    <p>   Multiple   spaces   between   words   </p>
    <p>
        Line breaks
        and
        formatting
    </p>
    <p>&nbsp;&nbsp;&nbsp;Non-breaking spaces</p>
    <h2>Empty Elements</h2>
    <p></p>
    <div></div>
    <span></span>
    <h2>Table with Empty Cells</h2>
    <table border=\"1\">
        <tr>
            <th>Column 1</th>
            <th>Column 2</th>
            <th>Column 3</th>
        </tr>
        <tr>
            <td>Data 1</td>
            <td></td>
            <td>Data 3</td>
        </tr>
        <tr>
            <td></td>
            <td>Data 2</td>
            <td></td>
        </tr>
    </table>
    <h2>Lists</h2>
    <ul>
        <li>Unordered item 1</li>
        <li>Unordered item 2</li>
        <li></li>
        <li>Unordered item 4</li>
    </ul>
    <ol>
        <li>Ordered item 1</li>
        <li>Ordered item 2</li>
        <li>Ordered item 3</li>
    </ol>
</body>
</html>"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.test_files['special_formatting'] = file_path

    @staticmethod
    def create_large_html(self):
        file_path = os.path.join(self.temp_dir, "large_test.html")
        html_content = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Large Document for Performance Testing</title>
</head>
<body>
    <h1>Large Document for Performance Testing</h1>
    <p>This document contains many sections for performance testing.</p>
"""
        for i in range(100):
            html_content += f"""
    <h2>Section {i+1}</h2>
    <p>This is paragraph {i+1} of the large document.</p>
    <p>It contains detailed information about topic {i+1}.</p>
    <p>Additional details and specifications for section {i+1}.</p>
"""
        for i in range(10):
            html_content += f"""
    <h3>Data Table {i+1}</h3>
    <table border=\"1\">
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Value</th>
            <th>Status</th>
        </tr>
        <tr>
            <td>1</td>
            <td>Item 1</td>
            <td>100</td>
            <td>Active</td>
        </tr>
        <tr>
            <td>2</td>
            <td>Item 2</td>
            <td>200</td>
            <td>Inactive</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Item 3</td>
            <td>300</td>
            <td>Active</td>
        </tr>
    </table>
"""
        html_content += """
</body>
</html>"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.test_files['large'] = file_path

    @staticmethod
    def create_malformed_html(self):
        file_path = os.path.join(self.temp_dir, "malformed_test.html")
        html_content = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <title>Malformed Test Document</title>
</head>
<body>
    <h1>Malformed Test Document</h1>
    <p>This document has some malformed HTML for testing.</p>
    <h2>Unclosed Tags</h2>
    <p>This paragraph has an unclosed <strong>tag
    <h2>Nested Lists</h2>
    <ul>
        <li>Item 1
            <ul>
                <li>Subitem 1
                <li>Subitem 2
            </ul>
        </li>
        <li>Item 2</li>
    </ul>
    <h2>Table Issues</h2>
    <table>
        <tr>
            <td>Cell 1
            <td>Cell 2
        </tr>
        <tr>
            <td>Cell 3</td>
        </tr>
    </table>
    <h2>Special Characters in Attributes</h2>
    <img src=\"image with spaces.jpg\" alt=\"Image with spaces\">
    <a href=\"link with spaces.html\">Link with spaces</a>
</body>
</html>"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.test_files['malformed'] = file_path

    def test_basic_loading(self):
        loader = HTMLLoader()
        documents = loader.load(self.test_files['basic'])
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertTrue(doc.content)
        self.assertIn('Basic Test Document', doc.content)
        self.assertIn('Welcome to Our Website', doc.content)
        self.assertIn('Section 1: Introduction', doc.content)
        self.assertIn('Feature 1: Text extraction', doc.content)
        self.assertEqual(doc.metadata['file_name'], 'basic_test.html')
        self.assertIn('file_size', doc.metadata)
        self.assertIn('creation_time', doc.metadata)

    def test_table_parsing(self):
        loader = HTMLLoader()
        documents = loader.load(self.test_files['table'])
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        content = doc.content
        self.assertIn('[Table]', content)
        self.assertIn('John Doe', content)
        self.assertIn('Engineering', content)
        self.assertIn('Laptop', content)
        self.assertIn('$999', content)
        table_count = content.count('[Table]')
        self.assertEqual(table_count, 2)

    def test_complex_structure(self):
        loader = HTMLLoader()
        documents = loader.load(self.test_files['complex'])
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        content = doc.content
        self.assertIn('Complex Document Structure', content)
        self.assertIn('Executive Summary', content)
        self.assertIn('Data Analysis', content)
        self.assertIn('First important point', content)
        self.assertIn('Revenue', content)
        self.assertIn('External Link', content)
        self.assertNotIn('testFunction', content)
        self.assertNotIn('background-color: yellow', content)

    def test_metadata_extraction(self):
        loader = HTMLLoader()
        documents = loader.load(self.test_files['metadata'])
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        metadata = doc.metadata
        self.assertEqual(metadata.get('title'), 'Metadata Test Document')
        self.assertEqual(metadata.get('description'), 'HTML document with rich metadata for testing')
        self.assertEqual(metadata.get('keywords'), 'html, metadata, test, loader, extraction')
        self.assertEqual(metadata.get('author'), 'Metadata Test Author')
        self.assertEqual(metadata.get('language'), 'en')
        self.assertIn('meta_robots', metadata)
        self.assertIn('meta_viewport', metadata)
        self.assertIn('meta_og:title', metadata)
        self.assertIn('element_counts', metadata)
        counts = metadata['element_counts']
        self.assertGreater(counts['paragraphs'], 0)
        self.assertGreater(counts['headings'], 0)
        self.assertIn('links', counts)

    def test_special_formatting(self):
        loader = HTMLLoader()
        documents = loader.load(self.test_files['special_formatting'])
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        content = doc.content
        self.assertIn('é, ñ, ü, ç', content)
        self.assertIn('€, £, ¥', content)
        self.assertIn('©, ®, ™', content)
        self.assertIn('123, 45.67', content)
        self.assertIn('@#$%^&*()', content)
        self.assertIn('[Table]', content)
        self.assertIn('Data 1', content)
        self.assertIn('Data 2', content)
        self.assertIn('Unordered item 1', content)
        self.assertIn('Ordered item 1', content)

    def test_empty_document(self):
        loader = HTMLLoader()
        documents = loader.load(self.test_files['empty'])
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsInstance(doc.content, str)
        self.assertEqual(doc.metadata['file_name'], 'empty_test.html')

    def test_malformed_html(self):
        loader = HTMLLoader()
        documents = loader.load(self.test_files['malformed'])
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        content = doc.content
        self.assertIn('Malformed Test Document', content)
        self.assertIn('Unclosed Tags', content)
        self.assertIn('Item 1', content)
        self.assertIn('Cell 1', content)

    def test_performance(self):
        loader = HTMLLoader()
        start_time = time.time()
        documents = loader.load(self.test_files['large'])
        end_time = time.time()
        processing_time = end_time - start_time
        self.assertEqual(len(documents), 1)
        self.assertLess(processing_time, 5.0)
        stats = loader.get_stats()
        self.assertEqual(stats['total_files_processed'], 1)
        self.assertEqual(stats['total_documents_created'], 1)
        doc = documents[0]
        content = doc.content
        self.assertIn('Large Document for Performance Testing', content)
        self.assertIn('Section 1', content)
        self.assertIn('Section 100', content)
        self.assertIn('Data Table 1', content)
        self.assertIn('Data Table 10', content)

    def test_file_not_found(self):
        config_ignore = HTMLLoaderConfig(error_handling="ignore")
        loader_ignore = HTMLLoader(config_ignore)
        docs = loader_ignore.load("non_existent_file.html")
        self.assertEqual(len(docs), 0)
        config_warn = HTMLLoaderConfig(error_handling="warn")
        loader_warn = HTMLLoader(config_warn)
        docs_warn = loader_warn.load("non_existent_file.html")
        self.assertEqual(len(docs_warn), 0)
        config_raise = HTMLLoaderConfig(error_handling="raise")
        loader_raise = HTMLLoader(config_raise)
        try:
            docs_raise = loader_raise.load("non_existent_file.html")
            self.assertEqual(len(docs_raise), 0)
        except Exception as e:
            self.assertTrue("Invalid source" in str(e) or "not a valid file" in str(e))

    def test_configuration_options(self):
        custom_metadata = {"source_type": "test", "version": "1.0"}
        config_custom = HTMLLoaderConfig(custom_metadata=custom_metadata)
        loader_custom = HTMLLoader(config_custom)
        docs_custom = loader_custom.load(self.test_files['basic'])
        self.assertEqual(len(docs_custom), 1)
        doc = docs_custom[0]
        self.assertEqual(doc.metadata['source_type'], 'test')
        self.assertEqual(doc.metadata['version'], '1.0')
        config_size = HTMLLoaderConfig(max_file_size=1000)
        loader_size = HTMLLoader(config_size)
        docs_size = loader_size.load(self.test_files['large'])
        self.assertEqual(len(docs_size), 0)
        config_markdown = HTMLLoaderConfig(table_format="markdown")
        loader_markdown = HTMLLoader(config_markdown)
        docs_markdown = loader_markdown.load(self.test_files['table'])
        self.assertEqual(len(docs_markdown), 1)
        content_markdown = docs_markdown[0].content
        self.assertIn('|', content_markdown)

    def test_batch_loading(self):
        loader = HTMLLoader()
        sources = [self.test_files['basic'], self.test_files['table'], self.test_files['metadata']]
        results = loader.load_batch(sources)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.success for result in results))
        self.assertTrue(all(len(result.documents) == 1 for result in results))

    def test_directory_loading(self):
        loader = HTMLLoader()
        results = loader.load_directory(self.temp_dir, file_patterns=["*.html"])
        self.assertGreaterEqual(len(results), 8)
        successful_results = [result for result in results if result.success]
        total_docs = sum(len(result.documents) for result in successful_results)
        self.assertGreaterEqual(len(successful_results), 7)
        self.assertGreaterEqual(total_docs, 7)

if __name__ == "__main__":
    unittest.main()
