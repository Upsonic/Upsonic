import unittest
from unittest.mock import patch, MagicMock
from upsonic import Task, Agent


class TestDo(unittest.TestCase):
    """Test suite for Task, Agent, and do functionality"""
    
    def test_agent_print_do_basic(self):
        """Test basic functionality of Agent.print_do with a Task"""
        # Create a task
        task = Task("Who developed you?")
        
        # Create an agent
        agent = Agent(name="Coder")
        
        
        result = agent.do(task)

        self.assertNotEqual(task.response, None)
        self.assertNotEqual(task.response, "")
        self.assertIsInstance(task.response, str)


        self.assertNotEqual(result, None)
        self.assertNotEqual(result, "")
        self.assertIsInstance(result, str)

        

