import unittest
from upsonic.agent.agent import Direct as Agent
from upsonic.models.providers import OpenAI, Anthropic, Gemini
from upsonic.models.factory import ModelFactory

class TestModelSpecification(unittest.TestCase):
    def test_traditional_provider_instances(self):
        openai_provider = OpenAI(model_name="gpt-4o")
        agent1 = Agent(name="Traditional Agent", model=openai_provider)
        self.assertEqual(agent1.model_provider.model_name, "gpt-4o")
        self.assertIsInstance(agent1.model_provider, OpenAI)

        agent2 = Agent(
            name="Inline Agent",
            model=Anthropic(model_name="claude-3-5-sonnet-latest")
        )
        self.assertEqual(agent2.model_provider.model_name, "claude-3-5-sonnet-latest")
        self.assertIsInstance(agent2.model_provider, Anthropic)

    def test_string_based_specifications(self):
        agent1 = Agent(name="String Agent 1", model="openai/gpt-4o")
        self.assertEqual(agent1.model_provider.model_name, "gpt-4o")
        self.assertIsInstance(agent1.model_provider, OpenAI)

        agent2 = Agent(name="String Agent 2", model="anthropic/claude-3-5-sonnet-latest")
        self.assertEqual(agent2.model_provider.model_name, "claude-3-5-sonnet-latest")
        self.assertIsInstance(agent2.model_provider, Anthropic)

        agent3 = Agent(name="String Agent 3", model="gemini/gemini-2.5-pro")
        self.assertEqual(agent3.model_provider.model_name, "gemini-2.5-pro")
        self.assertIsInstance(agent3.model_provider, Gemini)

        agent4 = Agent(name="String Agent 4", model="claude/claude-3-5-sonnet-latest")
        self.assertEqual(agent4.model_provider.model_name, "claude-3-5-sonnet-latest")
        self.assertIsInstance(agent4.model_provider, Anthropic)

    def test_model_factory_direct(self):
        openai_provider = ModelFactory.create("openai/gpt-4o")
        anthropic_provider = ModelFactory.create("anthropic/claude-3-5-sonnet-latest")
        self.assertEqual(openai_provider.model_name, "gpt-4o")
        self.assertIsInstance(openai_provider, OpenAI)
        self.assertEqual(anthropic_provider.model_name, "claude-3-5-sonnet-latest")
        self.assertIsInstance(anthropic_provider, Anthropic)

        providers = ModelFactory.list_supported_providers()
        self.assertIn("openai", providers)
        self.assertIn("anthropic", providers)

        openai_models = ModelFactory.list_supported_models("openai")
        self.assertIn("gpt-4o", openai_models)

    def test_error_handling(self):
        # Test cases that should raise exceptions
        error_cases = [
            ("invalid/gpt-4o", "provider"),
            ("just-a-model-name", "format"),
        ]
        for model_spec, expected_error in error_cases:
            with self.assertRaises(Exception) as excinfo:
                Agent(name="Invalid Agent", model=model_spec)
            self.assertIn(expected_error, str(excinfo.exception).lower())
        
        # Test case that should only show a warning (not raise exception)
        # This tests that invalid model names are handled gracefully
        try:
            agent = Agent(name="Warning Agent", model="openai/invalid-model")
            # Should not raise exception, just show warning
            self.assertIsNotNone(agent)
            self.assertEqual(agent.model_provider.model_name, "invalid-model")
        except Exception as e:
            self.fail(f"openai/invalid-model should not raise exception, but got: {e}")

if __name__ == "__main__":
    unittest.main()
