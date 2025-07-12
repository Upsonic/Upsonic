import unittest
from unittest.mock import Mock, patch
from upsonic.context import context_proceess, ContextProcessor, get_context_processor
from upsonic.context.default_prompt import DefaultPrompt


class TestContext(unittest.TestCase):
    """Test suite for context processing functionality"""

    def test_context_proceess_basic(self):
        """Test basic functionality of context_proceess with None input"""
        result = context_proceess(None)

        # Check that result is a string
        self.assertIsInstance(result, str)

        # Check that it contains expected XML structure
        self.assertIn("<Context>", result)
        self.assertIn("</Context>", result)
        self.assertIn("<Default Prompt>", result)
        self.assertIn("</Default Prompt>", result)

        # Check that the result is not empty
        self.assertGreater(len(result), 100)

    def test_context_proceess_with_empty_list(self):
        """Test context_proceess with empty list input"""
        result = context_proceess([])

        # Should behave the same as None input
        self.assertIsInstance(result, str)
        self.assertIn("<Context>", result)
        self.assertIn("</Context>", result)
        self.assertIn("<Default Prompt>", result)
        self.assertIn("</Default Prompt>", result)

    def test_context_proceess_with_default_prompt(self):
        """Test context_proceess with DefaultPrompt object"""
        custom_prompt = DefaultPrompt(prompt="Custom test prompt")
        result = context_proceess([custom_prompt])

        # Should contain the custom prompt text
        self.assertIn("Custom test prompt", result)
        self.assertIn("<Default Prompt>", result)
        self.assertIn("</Default Prompt>", result)

    def test_context_processor_initialization(self):
        """Test ContextProcessor can be initialized and has strategies"""
        processor = ContextProcessor()

        # Should have default strategies
        self.assertGreater(len(processor.strategies), 0)
        self.assertEqual(len(processor.strategies), 4)

        # Should be able to get strategy names
        strategy_names = processor.get_available_strategies()
        self.assertIn("TaskContextStrategy", strategy_names)
        self.assertIn("AgentContextStrategy", strategy_names)
        self.assertIn("DefaultPromptContextStrategy", strategy_names)
        self.assertIn("KnowledgeBaseContextStrategy", strategy_names)

    def test_get_context_processor(self):
        """Test getting global context processor instance"""
        processor = get_context_processor()

        # Should return a ContextProcessor instance
        self.assertIsInstance(processor, ContextProcessor)

        # Should have strategies
        self.assertGreater(len(processor.strategies), 0)

    def test_context_proceess_backward_compatibility(self):
        """Test that context_proceess maintains backward compatibility"""
        # Test the function signature hasn't changed
        result1 = context_proceess(None)
        result2 = context_proceess([])
        result3 = context_proceess([DefaultPrompt(prompt="test")])

        # All should return strings
        self.assertIsInstance(result1, str)
        self.assertIsInstance(result2, str)
        self.assertIsInstance(result3, str)

        # All should have basic XML structure
        for result in [result1, result2, result3]:
            self.assertIn("<Context>", result)
            self.assertIn("</Context>", result)

    def test_context_processor_add_strategy(self):
        """Test adding a custom strategy to processor"""
        processor = ContextProcessor()
        initial_count = len(processor.strategies)

        # Create a mock strategy
        mock_strategy = Mock()
        mock_strategy.can_process.return_value = False
        mock_strategy.get_section_name.return_value = "Test"

        processor.add_strategy(mock_strategy)

        # Should have one more strategy
        self.assertEqual(len(processor.strategies), initial_count + 1)
        self.assertIn(mock_strategy, processor.strategies)

    def test_context_processor_with_error_handling(self):
        """Test context processor handles unknown items gracefully"""
        processor = ContextProcessor()

        # Process an unknown object type
        result = processor.process_context(["unknown_string_item"])

        # Should still return valid XML structure
        self.assertIsInstance(result, str)
        self.assertIn("<Context>", result)
        self.assertIn("</Context>", result)

        # Should contain error information in comments
        self.assertIn("Processing Errors:", result)


if __name__ == "__main__":
    unittest.main()
