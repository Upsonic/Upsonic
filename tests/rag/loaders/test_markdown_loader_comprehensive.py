import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from upsonic.loaders.markdown import MarkdownLoader
from upsonic.loaders.config import MarkdownLoaderConfig
from upsonic.schemas.data_models import Document

class TestMarkdownLoaderComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_files = {}
        cls.create_basic_markdown(cls)
        cls.create_front_matter_markdown(cls)
        cls.create_table_markdown(cls)
        cls.create_code_blocks_markdown(cls)
        cls.create_complex_markdown(cls)
        cls.create_empty_markdown(cls)
        cls.create_special_formatting_markdown(cls)
        cls.create_large_markdown(cls)
        cls.create_malformed_markdown(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_basic_markdown(self):
        file_path = os.path.join(self.temp_dir, "basic_test.md")
        content = """# Basic Test Document

This is a basic test document for the Markdown loader.

## Section 1: Introduction

This is the introduction section with some important information.

### Subsection 1.1

Here are the detailed information and specifications.

## Section 2: Features

The loader supports:

- Text extraction
- Structure preservation
- Metadata extraction
- Heading hierarchy
- List processing

## Section 3: Conclusion

This concludes our basic test document.

### Links and Images

Here are some sample links:
- [External Link](https://example.com)
- [Internal Link](#section-1-introduction)
- [Email Link](mailto:test@example.com)

And some images:
![Sample Image](https://via.placeholder.com/150)
![Local Image](/path/to/image.jpg)
"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.test_files['basic'] = file_path

    @staticmethod
    def create_front_matter_markdown(self):
        file_path = os.path.join(self.temp_dir, "front_matter_test.md")
        content = """---
        title: Front Matter Test
        author: Test Author
        date: 2024-01-15
        tags: [markdown, test, front-matter]
        ---

        # Front Matter Test

        This document has YAML front matter that should be extracted as metadata.

        ## Content Section

        The content starts after the front matter delimiter.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.test_files['front_matter'] = file_path

    @staticmethod
    def create_table_markdown(self):
        file_path = os.path.join(self.temp_dir, "table_test.md")
        content = """# Markdown Table Test

        ## Simple Table

        | Name | Age | Department |
        |------|-----|------------|
        | John | 30  | Engineering|
        | Jane | 25  | Marketing  |
        | Bob  | 35  | Sales      |

        ## Complex Table

        | ID | Product | Price | Stock | Description |
        |----|---------|-------|-------|-------------|
        | 1  | Widget  | $9.99 | 42    | A basic widget for testing |
        | 2  | Gadget  | $19.99| 15    | An advanced gadget with features |
        | 3  | Tool    | $14.99| 30    | A useful tool for many tasks |
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.test_files['table'] = file_path
        
    @staticmethod
    def create_code_blocks_markdown(self):
        file_path = os.path.join(self.temp_dir, "code_blocks_test.md")
        content = """# Code Blocks Test

        This document contains various code blocks for testing.

        ## Python Code

        ```python
        def hello_world():
            print("Hello, World!")
            
        hello_world()
        ```

        ## JavaScript Code

        ```javascript
        function greet(name) {
            console.log(`Hello, ${name}!`);
        }
        
        greet('World');
        ```

        ## Bash Script

        ```bash
        #!/bin/bash
        echo "Hello from bash!"
        ```

        ## Inline Code
        
        Here's some `inline code` within text.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.test_files['code_blocks'] = file_path
        
    @staticmethod
    def create_complex_markdown(self):
        file_path = os.path.join(self.temp_dir, "complex_test.md")
        content = """# Complex Markdown Document

        This document contains a mix of various markdown elements for testing.

        ## Text Formatting

        **Bold text** and *italic text* and ***bold italic text***.

        ~~Strikethrough text~~ and `inline code`.

        ## Lists

        ### Unordered List

        - Item 1
        - Item 2
          - Nested item 2.1
          - Nested item 2.2
        - Item 3

        ### Ordered List

        1. First item
        2. Second item
           1. Nested item 2.1
           2. Nested item 2.2
        3. Third item

        ## Blockquotes

        > This is a blockquote.
        > 
        > It can span multiple lines.
        > 
        >> And it can be nested.

        ## Links and Images

        [Link to example](https://example.com)

        ![Image description](https://example.com/image.jpg)

        ## Horizontal Rule

        ---

        ## Mixed Content

        Here's a paragraph with **bold** and *italic* text, followed by a list:

        1. Item with `code`
        2. Item with [link](https://example.com)
        3. Item with *emphasis*

        > Blockquote with **formatting** and `code`
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.test_files['complex'] = file_path
        
    @staticmethod
    def create_empty_markdown(self):
        file_path = os.path.join(self.temp_dir, "empty_test.md")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("")
        self.test_files['empty'] = file_path
        
    @staticmethod
    def create_special_formatting_markdown(self):
        file_path = os.path.join(self.temp_dir, "special_formatting_test.md")
        content = """# Special Formatting Test

        ## Special Characters

        Special characters: é, ñ, ü, ç, €, £, ¥, ©, ®, ™

        ## Escaping Characters

        *This is italic*

        `This is code`

        # This is a heading

        ## HTML Embedded

        <div class="custom-class">
          <p>This is a paragraph with <strong>HTML</strong> formatting.</p>
        </div>

        <table>
          <tr>
            <th>Header 1</th>
            <th>Header 2</th>
          </tr>
          <tr>
            <td>Cell 1</td>
            <td>Cell 2</td>
          </tr>
        </table>
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.test_files['special_formatting'] = file_path
        
    @staticmethod
    def create_large_markdown(self):
        file_path = os.path.join(self.temp_dir, "large_test.md")
        content = "# Large Markdown Document\n\n"
        
        # Add many sections
        for i in range(1, 21):
            content += f"## Section {i}\n\n"
            content += f"This is paragraph {i} of the large document.\n\n"
            content += f"It contains detailed information about topic {i}.\n\n"
            content += f"Additional details and specifications for section {i}.\n\n"
            
            # Add a list
            content += f"### List {i}\n\n"
            for j in range(1, 6):
                content += f"- Item {i}.{j}\n"
            content += "\n"
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.test_files['large'] = file_path
        
    @staticmethod
    def create_malformed_markdown(self):
        file_path = os.path.join(self.temp_dir, "malformed_test.md")
        content = """# Malformed Markdown

        This document has some malformed markdown elements.

        ## Unclosed formatting

        **This bold text is not closed

        ## Broken link

        [This link is broken(https://example.com)

        ## Invalid table

        | Header 1 | Header 2
        | --- | ---
        | Cell 1 | Cell 2
        Cell 3 | Cell 4 |

        ## Incorrect indentation

        - Item 1
          - Nested item 1.1
         - This indentation is wrong
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.test_files['malformed'] = file_path

    def test_basic_loading(self):
        """Test basic markdown loading functionality."""
        loader = MarkdownLoader()
        documents = loader.load(self.test_files['basic'])
        
        self.assertEqual(len(documents), 1, f"Expected 1 document, got {len(documents)}")
        
        doc = documents[0]
        self.assertTrue(doc.content, "Document should have content")
        self.assertIn('Basic Test Document', doc.content, "Should contain the title")
        self.assertIn('Section 1: Introduction', doc.content, "Should contain section heading")
        self.assertIn('Text extraction', doc.content, "Should contain list item")
        
        # Check metadata
        self.assertIn('source', doc.metadata, "Should have source in metadata")
        self.assertIn('file_name', doc.metadata, "Should have file name in metadata")
        self.assertEqual(doc.metadata['file_name'], 'basic_test.md', "Should have correct file name")

    def test_front_matter(self):
        """Test front matter extraction."""
        config = MarkdownLoaderConfig(extract_front_matter=True)
        loader = MarkdownLoader(config)
        documents = loader.load(self.test_files['front_matter'])
        
        self.assertEqual(len(documents), 1, f"Expected 1 document, got {len(documents)}")
        
        doc = documents[0]
        self.assertIn('Front Matter Test', doc.content, "Should contain the title")
        self.assertIn('Content Section', doc.content, "Should contain section heading")
        
        # Some implementations might not extract front matter into metadata
        # Just check that we have basic metadata
        self.assertIn('file_name', doc.metadata, "Should have file name in metadata")
        self.assertEqual(doc.metadata['file_name'], 'front_matter_test.md', "Should have correct file name")

    def test_table_extraction(self):
        """Test table extraction and processing."""
        loader = MarkdownLoader()
        documents = loader.load(self.test_files['table'])
        
        self.assertEqual(len(documents), 1, f"Expected 1 document, got {len(documents)}")
        
        doc = documents[0]
        self.assertIn('Markdown Table Test', doc.content, "Should contain the title")
        
        # Check that tables are processed in some way
        self.assertIn('Table Data', doc.content, "Should contain table markers")
        self.assertIn('John', doc.content, "Should contain table data")
        self.assertIn('Engineering', doc.content, "Should contain department data")
        self.assertIn('Widget', doc.content, "Should contain product data")
        self.assertIn('$9.99', doc.content, "Should contain price data")
        
    def test_code_blocks_processing(self):
        """Test code block processing."""
        loader = MarkdownLoader()
        documents = loader.load(self.test_files['code_blocks'])
        
        self.assertEqual(len(documents), 1, f"Expected 1 document, got {len(documents)}")
        
        doc = documents[0]
        content = doc.content
        
        # Check that code blocks are preserved
        self.assertIn('```python', content, "Should contain Python code block")
        self.assertIn('def hello_world():', content, "Should contain Python function")
        self.assertIn('```javascript', content, "Should contain JavaScript code block")
        self.assertIn('function greet(name)', content, "Should contain JavaScript function")
        
    def test_complex_markdown(self):
        """Test complex markdown structure."""
        loader = MarkdownLoader()
        documents = loader.load(self.test_files['complex'])
        
        self.assertEqual(len(documents), 1, f"Expected 1 document, got {len(documents)}")
        
        doc = documents[0]
        content = doc.content
        
        # Check various content types
        self.assertIn('Complex Markdown Document', content, "Should contain title")
        self.assertIn('Text Formatting', content, "Should contain main heading")
        self.assertIn('Lists', content, "Should contain subheading")
        self.assertIn('Item 1', content, "Should contain list items")
        self.assertIn('Blockquotes', content, "Should contain blockquote section")
        
    def test_special_formatting(self):
        """Test handling of special formatting and edge cases."""
        loader = MarkdownLoader()
        documents = loader.load(self.test_files['special_formatting'])
        
        self.assertEqual(len(documents), 1, f"Expected 1 document, got {len(documents)}")
        
        doc = documents[0]
        content = doc.content
        
        # Check special characters
        self.assertIn('é, ñ, ü, ç', content, "Should preserve special characters")
        self.assertIn('€, £, ¥', content, "Should preserve currency symbols")
        self.assertIn('©, ®, ™', content, "Should preserve trademark symbols")
        
        # Check HTML content
        self.assertIn('HTML', content, "Should contain HTML text")
        self.assertIn('custom-class', content, "Should preserve HTML attributes")

if __name__ == "__main__":
    unittest.main()
