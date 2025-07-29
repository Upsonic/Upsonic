import pytest
from unittest.mock import patch
from upsonic import Task, Agent
from pydantic import BaseModel

class Names(BaseModel):
    names: list[str]

class TestTaskImageContextHandling:
    
    def test_agent_with_multiple_images_returns_combined_names(self):
        images = ["paper1.png", "paper2.png"]
        expected_names = ["Mumtaz", "Bartu", "Onur"]
        
        task = Task(
            "Extract the names in the paper",
            images=images,
            response_format=Names
        )
        
        agent = Agent(name="OCR Agent")

        # Patch agent.print_do to simulate name extraction from all images
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect(task_obj):
                fake_names = []
                for img in task_obj.images:
                    if img == "paper1.png":
                        fake_names += ["Mumtaz", "Bartu"]
                    elif img == "paper2.png":
                        fake_names += ["Onur"]
                task_obj._response = Names(names=fake_names)
                return task_obj.response
            
            mock_print_do.side_effect = side_effect
            
            result = agent.print_do(task)
            
            assert isinstance(result, Names)
            assert result.names == expected_names
            assert len(result.names) == 3
