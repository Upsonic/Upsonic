import unittest
from upsonic import Task, Agent
from upsonic.context.context import context_proceess
from upsonic.context.default_prompt import DefaultPrompt
from upsonic.knowledge_base.knowledge_base import KnowledgeBase


class TestContextBuilder(unittest.TestCase):
    def test_context_builder_output(self):
        task = Task("Who developed you?")
        agent = Agent(name="Coder")

        prompt = DefaultPrompt(prompt="Test prompt")
        kb = KnowledgeBase(content="Sample knowledge base content")

        context = [task, agent, prompt, kb]

        result = context_proceess(context)
        
        self.assertIn("<Tasks>", result)
        self.assertIn("<Agents>", result)
        self.assertIn("<Default Prompt>", result)
        self.assertIn("<Knowledge Base>", result)
        self.assertIn("</Context>", result)


if __name__ == '__main__':
    unittest.main()