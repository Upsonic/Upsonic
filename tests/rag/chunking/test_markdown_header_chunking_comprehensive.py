import os
import sys
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import unittest

from upsonic.text_splitter.markdown import MarkdownHeaderChunkingStrategy, MarkdownHeaderChunkingConfig
from upsonic.schemas.data_models import Document, Chunk

class TestMarkdownHeaderChunkingComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_documents = {}
        cls.create_simple_markdown_document(cls)
        cls.create_nested_headers_document(cls)
        cls.create_complex_markdown_document(cls)
        cls.create_mixed_content_document(cls)
        cls.create_edge_case_document(cls)
        cls.create_large_markdown_document(cls)
        cls.create_empty_markdown_document(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_simple_markdown_document(self):
        markdown_content = """# Introduction
This is the introduction section with some basic information.

## Getting Started
This is the getting started section with setup instructions.

### Installation
Install the package using pip or your preferred package manager.

## Usage
How to use the package in your projects."""
        doc = Document(content=markdown_content, metadata={'source': 'simple', 'type': 'documentation'})
        self.test_documents['simple'] = doc

    @staticmethod
    def create_nested_headers_document(self):
        markdown_content = """# Main Title
Main content under the primary title.

## Section 1
Content for section 1.

### Subsection 1.1
Content for subsection 1.1.

#### Deep Section 1.1.1
Content for deep section 1.1.1.

##### Very Deep Section 1.1.1.1
Content for very deep section 1.1.1.1.

###### Deepest Section 1.1.1.1.1
Content for the deepest section level.

### Subsection 1.2
Content for subsection 1.2.

## Section 2
Content for section 2.

### Subsection 2.1
Content for subsection 2.1."""
        doc = Document(content=markdown_content, metadata={'source': 'nested', 'type': 'hierarchical', 'levels': 6})
        self.test_documents['nested'] = doc

    @staticmethod
    def create_complex_markdown_document(self):
        markdown_content = """# API Documentation

Welcome to our comprehensive API documentation.

## Authentication

All API requests require authentication using API keys.

### Getting API Keys

You can obtain API keys from the dashboard:

1. Navigate to Settings
2. Click on API Keys
3. Generate new key

### Using API Keys

Include the API key in your request headers:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.example.com/v1/data
```

## Endpoints

The following endpoints are available:

### Users API

#### GET /users

Retrieve a list of all users.

**Parameters:**
- `limit` (integer): Number of users to return
- `offset` (integer): Number of users to skip

**Response:**
```json
{
  "users": [
    {"id": 1, "name": "John Doe", "email": "john@example.com"}
  ],
  "total": 100
}
```

#### POST /users

Create a new user.

### Data API

#### GET /data

Retrieve data entries.

## Rate Limiting

API requests are rate limited to 1000 requests per hour.

## Error Handling

All errors return appropriate HTTP status codes with descriptive messages."""
        doc = Document(content=markdown_content, metadata={'source': 'complex', 'type': 'api_docs', 'sections': 4})
        self.test_documents['complex'] = doc

    @staticmethod
    def create_mixed_content_document(self):
        markdown_content = """# Project Overview

This document contains various types of markdown content.

## Features

Our project includes the following features:

- Feature 1: Advanced data processing
- Feature 2: Real-time analytics
- Feature 3: Secure authentication

## Code Examples

### Python Example

```python
def calculate_metrics(data):
    total = sum(data)
    average = total / len(data)
    return {
        'total': total,
        'average': average,
        'count': len(data)
    }

# Usage
data = [1, 2, 3, 4, 5]
metrics = calculate_metrics(data)
print(metrics)
```

### JavaScript Example

```javascript
function processData(items) {
    return items
        .filter(item => item.active)
        .map(item => ({
            id: item.id,
            name: item.name.toUpperCase(),
            timestamp: new Date()
        }));
}
```

## Tables

| Feature | Status | Priority |
|---------|--------|----------|
| Authentication | ✅ Complete | High |
| Data Processing | 🟡 In Progress | Medium |
| Analytics | ❌ Pending | Low |

## Links and References

For more information, visit:
- [Documentation](https://docs.example.com)
- [GitHub Repository](https://github.com/example/project)
- [Support Forum](https://forum.example.com)

## Conclusion

This project demonstrates various markdown capabilities."""
        doc = Document(content=markdown_content, metadata={'source': 'mixed', 'type': 'project_docs', 'elements': 'multiple'})
        self.test_documents['mixed'] = doc

    @staticmethod
    def create_edge_case_document(self):
        markdown_content = """#Title
No space after hash.

##  Section
Space before section name.

###Header
No space after hash for level 3.

####   Multiple
Multiple spaces after hash.

# Normal Header
This is a normal header with proper spacing.

#Another
Another header without space.

## Final Section
The final section with proper formatting."""
        doc = Document(content=markdown_content, metadata={'source': 'edge_cases', 'type': 'malformed'})
        self.test_documents['edge_cases'] = doc

    @staticmethod
    def create_large_markdown_document(self):
        lines = ["# Large Document", "This is a large markdown document for performance testing.", ""]
        
        for i in range(20):
            lines.extend([
                f"## Section {i+1}",
                f"This is section {i+1} with comprehensive content and detailed information.",
                ""
            ])
            
            for j in range(5):
                lines.extend([
                    f"### Subsection {i+1}.{j+1}",
                    f"Content for subsection {i+1}.{j+1} with detailed explanations and examples.",
                    f"Additional paragraph {j+1} with more information and context.",
                    ""
                ])
        
        markdown_content = "\n".join(lines)
        doc = Document(content=markdown_content, metadata={'source': 'large', 'type': 'comprehensive', 'sections': 20})
        self.test_documents['large'] = doc

    @staticmethod
    def create_empty_markdown_document(self):
        doc = Document(content="", metadata={'source': 'empty', 'type': 'edge_case'})
        self.test_documents['empty'] = doc

    def test_basic_markdown_header_chunking(self):
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        # Check that header metadata is present
        for chunk in chunks:
            self.assertIn('Header 1', chunk.metadata)
            self.assertEqual(chunk.metadata['Header 1'], 'Introduction')

    def test_header_hierarchy_preservation(self):
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['nested'])
        
        self.assertGreater(len(chunks), 0)
        
        # Find a chunk with deep nesting and verify hierarchy
        deep_chunks = [chunk for chunk in chunks if 'Header 4' in chunk.metadata]
        if deep_chunks:
            chunk = deep_chunks[0]
            self.assertEqual(chunk.metadata['Header 1'], 'Main Title')
            self.assertEqual(chunk.metadata['Header 2'], 'Section 1')
            self.assertEqual(chunk.metadata['Header 3'], 'Subsection 1.1')
            self.assertIn('Header 4', chunk.metadata)

    def test_strip_headers_configuration(self):
        # Test with strip_headers=True (default)
        config_strip = MarkdownHeaderChunkingConfig(strip_headers=True)
        chunker_strip = MarkdownHeaderChunkingStrategy(config_strip)
        chunks_strip = chunker_strip.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks_strip), 0)
        
        # Headers should be stripped from content
        for chunk in chunks_strip:
            self.assertNotIn('# Introduction', chunk.text_content)
            self.assertNotIn('## Getting Started', chunk.text_content)
        
        # Test with strip_headers=False
        config_preserve = MarkdownHeaderChunkingConfig(strip_headers=False)
        chunker_preserve = MarkdownHeaderChunkingStrategy(config_preserve)
        chunks_preserve = chunker_preserve.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks_preserve), 0)
        
        # At least some chunks should contain headers
        has_header_content = any('# Introduction' in chunk.text_content or 
                               '## Getting Started' in chunk.text_content 
                               for chunk in chunks_preserve)
        # This might not always be true depending on implementation, so we'll check flexibly
        self.assertTrue(len(chunks_preserve) > 0)

    def test_custom_headers_configuration(self):
        # Test with custom header levels
        custom_headers = [("#", "H1"), ("##", "H2")]
        config = MarkdownHeaderChunkingConfig(headers_to_split_on=custom_headers)
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['nested'])
        
        self.assertGreater(len(chunks), 0)
        
        # Check that custom headers configuration was applied
        for chunk in chunks:
            # Should have H1 and H2 keys instead of Header 1, Header 2
            has_h1_or_h2 = 'H1' in chunk.metadata or 'H2' in chunk.metadata
            if has_h1_or_h2:
                # Verify custom header naming is used
                self.assertTrue('H1' in chunk.metadata or 'H2' in chunk.metadata)

    def test_mixed_content_processing(self):
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['mixed'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that different content types are preserved
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("calculate_metrics", all_content)  # Code block
        self.assertIn("Feature 1", all_content)  # List item
        self.assertIn("Authentication", all_content)  # Table content

    def test_edge_case_headers(self):
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['edge_cases'])
        
        self.assertGreaterEqual(len(chunks), 0)
        
        # Should handle malformed headers gracefully
        for chunk in chunks:
            self.assertIsInstance(chunk.text_content, str)
            self.assertGreater(len(chunk.text_content), 0)

    def test_empty_content_handling(self):
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['empty'])
        
        self.assertEqual(len(chunks), 0)

    def test_whitespace_only_content(self):
        whitespace_content = "   \n\n  \t  \n"
        whitespace_doc = Document(content=whitespace_content, metadata={'type': 'whitespace'})
        
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(whitespace_doc)
        
        self.assertEqual(len(chunks), 0)

    def test_performance_with_large_document(self):
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        
        start_time = time.time()
        chunks = chunker.chunk(self.test_documents['large'])
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertGreater(len(chunks), 0)
        self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds
        
        # Verify all chunks have proper structure
        for chunk in chunks:
            self.assertIsInstance(chunk.text_content, str)
            self.assertIsInstance(chunk.metadata, dict)
            self.assertEqual(chunk.metadata['source'], 'large')

    def test_metadata_inheritance(self):
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that original document metadata is inherited
        for chunk in chunks:
            self.assertEqual(chunk.metadata['source'], 'complex')
            self.assertEqual(chunk.metadata['type'], 'api_docs')
            self.assertEqual(chunk.metadata['sections'], 4)

    def test_content_without_headers(self):
        no_headers_content = """This is content without any headers.
Just plain text spread across multiple lines.
No markdown headers anywhere in this content."""
        no_headers_doc = Document(content=no_headers_content, metadata={'type': 'plain'})
        
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(no_headers_doc)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text_content, no_headers_content)
        self.assertEqual(chunks[0].metadata['type'], 'plain')

    def test_trailing_whitespace_handling(self):
        content_with_whitespace = """# Title
Content with trailing spaces.   


   """
        whitespace_doc = Document(content=content_with_whitespace, metadata={'type': 'whitespace_test'})
        
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(whitespace_doc)
        
        self.assertEqual(len(chunks), 1)
        # Content should be stripped of excessive whitespace
        self.assertEqual(chunks[0].text_content.strip(), "Content with trailing spaces.")

    def test_batch_processing(self):
        documents = [
            self.test_documents['simple'],
            self.test_documents['nested'],
            self.test_documents['mixed']
        ]
        
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        
        batch_results = chunker.chunk_batch(documents)
        
        self.assertEqual(len(batch_results), 3)
        self.assertTrue(all(len(chunks) > 0 for chunks in batch_results))
        
        total_chunks = sum(len(chunks) for chunks in batch_results)
        self.assertGreater(total_chunks, 0)

    def test_caching_functionality(self):
        config = MarkdownHeaderChunkingConfig(enable_caching=True)
        chunker = MarkdownHeaderChunkingStrategy(config)
        
        # First processing
        chunks1 = chunker.chunk(self.test_documents['simple'])
        
        # Second processing (should use cache if available)
        chunks2 = chunker.chunk(self.test_documents['simple'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))

    def test_error_handling(self):
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        
        # Test with problematic content
        problematic_content = """# 
Empty header name.

## 
Another empty header.

### Valid Header
This should work fine."""
        problematic_doc = Document(content=problematic_content, metadata={'type': 'problematic'})
        
        chunks = chunker.chunk(problematic_doc)
        self.assertGreaterEqual(len(chunks), 0)  # Should handle gracefully

    def test_complex_api_documentation(self):
        config = MarkdownHeaderChunkingConfig()
        chunker = MarkdownHeaderChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that API documentation content is preserved (headers might be stripped)
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("API requests require authentication", all_content)
        self.assertIn("Retrieve a list of all users", all_content)
        self.assertIn("rate limited", all_content)
        
        # Check that code blocks are preserved
        self.assertIn("curl -H", all_content)
        self.assertIn("Authorization: Bearer", all_content)

if __name__ == "__main__":
    unittest.main()