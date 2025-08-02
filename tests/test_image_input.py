import pytest
from upsonic import Task, Agent
from pydantic import BaseModel

class Names(BaseModel):
    names: list[str]

class TestTaskImageContextHandling:
    
    def test_agent_with_multiple_images_returns_combined_names(self):
        images = ["paper1.png", "paper2.png"]
        
        task = Task(
            "Extract the names in the paper",
            images=images,
            response_format=Names
        )
        
        agent = Agent(name="OCR Agent")
        
        result = agent.print_do(task)
        
        assert isinstance(result, Names)
        assert isinstance(result.names, list)
        assert all(isinstance(name, str) for name in result.names)
