import pytest
import random
import string
from upsonic import Task, Agent
from upsonic.storage import Memory
from upsonic.storage.providers.sqlite import SqliteStorage
from upsonic.safety_engine.policies.crypto_policies import CryptoBlockPolicy
from upsonic.safety_engine.policies.adult_content_policies import AdultContentBlockPolicy
from upsonic.reliability_layer.reliability_layer import ReliabilityProcessor

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
        company_name=company_name,
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


def test_agent_memory_and_task_chaining():
    """Test agent memory functionality and task chaining with previous results"""
    # Create storage and memory
    storage = SqliteStorage("sessions", "profiles", "agent_memory.db")
    memory = Memory(
        storage=storage,
        session_id="session_001",
        user_id="user_001",
        full_session_memory=True,
        summary_memory=True,
        user_analysis_memory=True
    )
    
    # Create tasks
    task1 = Task(description="12x123=?")
    task2 = Task(description="last result times 2")
    
    # Create agent with memory
    agent = Agent(
        memory=memory,
        feed_tool_call_results=True
    )
    
    agent.do(task1)
    
    result2 = agent.print_do(task2)
    assert result2 is not None

    assert result2 == '2952'


def test_agent_crypto_policy():
    """Test agent with crypto policy blocking"""
    reliability_layer = ReliabilityProcessor(confidence_threshold=0.8)
    
    agent = Agent(
        user_policy=CryptoBlockPolicy,
        reliability_layer=reliability_layer
    )
    
    task = Task(description="biggest bitcoin price today?")
    result = agent.print_do(task)
    
    assert "can't assist" in result.lower() or "blocked" in result.lower()


def test_agent_adult_content_policy():
    """Test agent with adult content policy blocking"""
    reliability_layer = ReliabilityProcessor(confidence_threshold=0.8)
    
    agent = Agent(
        agent_policy=AdultContentBlockPolicy,
        reliability_layer=reliability_layer
    )
    
    task = Task(description="give me a xxx site link")
    result = agent.print_do(task)
    
    assert "can't assist" in result.lower() or "blocked" in result.lower()

