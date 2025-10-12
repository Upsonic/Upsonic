import unittest
import tempfile
import os
from pathlib import Path
from upsonic.loaders.markdown import MarkdownLoader
from upsonic.loaders.config import MarkdownLoaderConfig
from upsonic.schemas.data_models import Document


class TestMarkdownLoaderSimple(unittest.TestCase):
    """Simplified tests for MarkdownLoader that match the actual implementation."""

    def setUp(self):
        """Set up test environment with sample Markdown files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple Markdown file
        self.simple_md = Path(self.temp_dir) / "simple.md"
        self.simple_md.write_text("""# Main Title

This is the introduction paragraph with some **bold** and *italic* text.

## Section 1

This is the first section with some content.

- Item 1
- Item 2
- Item 3

## Section 2

This is the second section with more content.

```python
def hello_world():
    print("Hello, World!")
```

### Subsection 2.1

This is a subsection with additional details.""")

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_markdown_loader_initialization(self):
        """Test Markdown loader initialization."""
        config = MarkdownLoaderConfig()
        loader = MarkdownLoader(config)
        self.assertIsNotNone(loader)

    def test_supported_extensions(self):
        """Test that Markdown loader supports correct file extensions."""
        supported = MarkdownLoader.get_supported_extensions()
        self.assertIn(".md", supported)
        self.assertGreaterEqual(len(supported), 1)

    def test_simple_markdown_loading(self):
        """Test loading a simple Markdown file."""
        config = MarkdownLoaderConfig()
        loader = MarkdownLoader(config)
        
        documents = loader.load(str(self.simple_md))
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))
        self.assertTrue(all(hasattr(doc, 'document_id') for doc in documents))

    def test_empty_source_handling(self):
        """Test handling of empty or invalid sources."""
        config = MarkdownLoaderConfig()
        loader = MarkdownLoader(config)
        
        result = loader.load([])
        self.assertEqual(len(result), 0)

    def test_batch_loading(self):
        """Test batch loading multiple Markdown files."""
        config = MarkdownLoaderConfig()
        loader = MarkdownLoader(config)
        
        files = [str(self.simple_md)]
        documents = loader.batch(files)
        
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_markdown_config_options(self):
        """Test Markdown loader configuration options."""
        config = MarkdownLoaderConfig(
            parse_front_matter=True,
            include_code_blocks=True,
            code_block_language_metadata=True,
            heading_metadata=True,
            split_by_heading="h2"
        )
        loader = MarkdownLoader(config)
        
        self.assertTrue(loader.config.parse_front_matter)
        self.assertTrue(loader.config.include_code_blocks)
        self.assertTrue(loader.config.code_block_language_metadata)
        self.assertTrue(loader.config.heading_metadata)
        self.assertEqual(loader.config.split_by_heading, "h2")

    def test_heading_split_options(self):
        """Test different heading split options."""
        heading_levels = ["h1", "h2", "h3"]
        for level in heading_levels:
            config = MarkdownLoaderConfig(split_by_heading=level)
            loader = MarkdownLoader(config)
            self.assertEqual(loader.config.split_by_heading, level)

    def test_metadata_inclusion(self):
        """Test Markdown metadata inclusion."""
        config = MarkdownLoaderConfig(include_metadata=True)
        loader = MarkdownLoader(config)
        
        documents = loader.load(str(self.simple_md))
        
        self.assertGreater(len(documents), 0)
        for doc in documents:
            self.assertIsInstance(doc.metadata, dict)
            self.assertIn('source', doc.metadata)


if __name__ == "__main__":
    unittest.main()