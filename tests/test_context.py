import unittest
from upsonic.context.context import context_proceess
from upsonic.context.default_prompt import default_prompt
from upsonic.tasks.tasks import Task
from upsonic.direct.direct_llm_cal import Direct
from upsonic.knowledge_base.knowledge_base import KnowledgeBase


class TestContextProcess(unittest.TestCase):

    def test_none_context(self):
        """Test context_proceess with None input - should return basic structure with default prompt"""
        result = context_proceess(None)

        self.assertIn("<Context>", result)
        self.assertIn("</Context>", result)
        self.assertIn("<Default Prompt>", result)
        self.assertIn("</Default Prompt>", result)
        self.assertIn(default_prompt().prompt, result)

    def test_empty_list_context(self):
        """Test context_proceess with empty list - should behave same as None"""
        result = context_proceess([])

        self.assertIn("<Context>", result)
        self.assertIn("</Context>", result)
        self.assertIn("<Default Prompt>", result)
        self.assertIn("</Default Prompt>", result)

    def test_with_task_context(self):
        """Test context_proceess with real Task object"""
        task = Task("test task description")

        result = context_proceess([task])

        self.assertIn("<Context>", result)
        self.assertIn("<Tasks>", result)
        self.assertIn("</Tasks>", result)
        self.assertIn("test task description", result)
        self.assertIn("Task ID", result)

    def test_with_agent_context(self):
        """Test context_proceess with real Agent object"""
        agent = Direct(name="TestAgent")

        result = context_proceess([agent])

        self.assertIn("<Context>", result)
        self.assertIn("<Agents>", result)
        self.assertIn("</Agents>", result)
        self.assertIn("TestAgent", result)
        self.assertIn("Agent ID", result)

    def test_with_knowledge_base_context(self):
        """Test context_proceess with real KnowledgeBase object"""
        kb = KnowledgeBase()

        result = context_proceess([kb])

        self.assertIn("<Context>", result)
        self.assertIn("<Knowledge Base>", result)
        self.assertIn("</Knowledge Base>", result)

    def test_mixed_context_types(self):
        """Test context_proceess with multiple real context types"""
        task = Task("test task")
        agent = Direct(name="TestAgent")
        kb = KnowledgeBase()

        result = context_proceess([task, agent, kb])

        # Should contain all sections
        self.assertIn("<Context>", result)
        self.assertIn("<Tasks>", result)
        self.assertIn("<Agents>", result)
        self.assertIn("<Knowledge Base>", result)
        self.assertIn("<Default Prompt>", result)

        # Should contain content from each object
        self.assertIn("test task", result)
        self.assertIn("TestAgent", result)

    def test_output_structure(self):
        """Test that output has correct XML-like structure and ordering"""
        result = context_proceess([])

        # Check overall structure
        self.assertTrue(result.startswith("<Context>"))
        self.assertTrue(result.endswith("</Context>"))

        # Check section ordering (Agents, Tasks, Default Prompt, Knowledge Base)
        context_pos = result.find("<Context>")
        agents_pos = result.find("<Agents>")
        tasks_pos = result.find("<Tasks>")
        default_pos = result.find("<Default Prompt>")
        kb_pos = result.find("<Knowledge Base>")

        self.assertTrue(context_pos < agents_pos < tasks_pos < default_pos < kb_pos)


if __name__ == "__main__":
    unittest.main()
