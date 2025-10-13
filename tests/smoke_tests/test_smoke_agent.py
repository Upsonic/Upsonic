import pytest
import random
import string
from upsonic import Task, Agent

#t
def test_agent_company_attributes():
    """Test that checks if company attributes are effecting the result"""
    company_name = ''.join(random.choices(string.ascii_lowercase, k=4))
    
    task = Task(
        description="What is my companys name",
    )
    
    agent = Agent(
        name="Company Agent",
        company_url=f"https://{company_name}.com/",
        company_objective=f"{company_name} AI Platform for FinTech Operations",
        company_description="Secure and efficient onboarding and landing steps for FinTech clients"
    )
    
    result = agent.do(task)

    assert result is not None
 

    assert company_name.lower() in result.lower()


def test_agent_roles_attributes():
    """Test that an checks if agent roles and attributes are effecting the result"""
    task = Task("shorten this text: Hello, how are you")
    
    agent = Agent(
        role="Shortner Agent",
        goal="Shorten the given text to their first letters",
		instructions="Shorten the given text to their first letters",
        education="English degree",
        work_experience="5 years teaching english"
    )
    
    result = agent.do(task)
    
    assert result is not None
	
    original_text = "Hello, how are you?"
    assert len(result) < len(original_text)
    assert "h, h a y" in result.lower() 

