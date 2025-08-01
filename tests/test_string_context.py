import pytest
from upsonic import Task, Agent


class TestTaskStringContextHandling:
    """Test suite for Task string context handling and agent's ability to use context."""

    def test_task_single_string_context_storage(self):
        """
        Test: Tek string context'in doğru şekilde saklanması
        Kontrol: Agent'ın context'i alabildiği
        """
        city = "New York"
        task_description = "Find resources in the city"
        
        task = Task(task_description, context=[city])
        
        assert task.context is not None  
        assert isinstance(task.context, list)  
        assert len(task.context) == 1  
        assert task.context[0] == city  
        assert isinstance(task.context[0], str)  

    def test_task_multiple_string_contexts_storage(self):
        """
        Test: Birden çok string context verildiğinde ne oluyor?
        Kontrol: Tüm string'lerin doğru sırada saklanması
        """
        contexts = ["New York", "Technology Sector", "Q4 2024", "Budget: $50000"]
        task_description = "Analyze market data for the specified parameters"

        task = Task(task_description, context=contexts)

        assert task.context is not None  
        assert isinstance(task.context, list)  
        assert len(task.context) == 4  
        
        for i, expected_ctx in enumerate(contexts):
            assert task.context[i] == expected_ctx  
            assert isinstance(task.context[i], str)  
        
        # Check overall context list equality
        assert task.context == contexts  

    def test_agent_can_access_single_string_context(self):
        """
        Test: Agent'ın tek string context'i kullanabilmesi
        Kontrol: Agent'ın context'e erişebildiğini test etme
        """
        city = "New York"
        task = Task("Find resources in the city", context=[city])
        agent = Agent(name="City Guide")
        
        result = agent.print_do(task)
        
        assert isinstance(result, str)  
        assert task.response == result  

    def test_agent_can_access_multiple_string_contexts(self):
        """
        Test: Agent'ın birden çok string context'i kullanabilmesi
        Kontrol: Tüm context'lerin agent tarafından erişilebilir olması
        """
        contexts = ["London", "Technology", "2024"]
        task = Task("Create a comprehensive analysis", context=contexts)
        agent = Agent(name="Analyst")
        
        result = agent.print_do(task)
        
        assert isinstance(result, str)
        assert task.response == result

   
    def test_task_empty_string_context_handling(self):
        """
        Test: Boş string context'lerin işlenmesi
        Kontrol: Boş string'lerin de context olarak kabul edilmesi
        """
        contexts = ["Valid City", "", "Another Valid Context"]
        task = Task("Handle mixed contexts", context=contexts)
        
        assert len(task.context) == 3  
        assert task.context[0] == "Valid City"  
        assert task.context[1] == ""  
        assert task.context[2] == "Another Valid Context"  

	
    def test_agent_context_integration_simulation(self):
        """
        Test: Agent'ın context'i task description ile nasıl entegre ettiğinin testi
        Kontrol: Context'in task description'a uygun şekilde kullanılması
        """
        city = "Tokyo"
        task = Task("Find the best restaurants", context=[city])
        agent = Agent(name="Food Guide")
        
        result = agent.print_do(task)
        
        assert isinstance(result, str)
        assert task.response == result
            
    def test_context_with_non_string_values(self):
        task = Task("Handle mixed context", context=["valid", 123, None])
        agent = Agent(name="Robust")
        
        result = agent.print_do(task)
        
        assert isinstance(result, str)
        assert task.response == result

    def test_task_with_empty_context_list(self):
        """
        Test: Boş context listesi ile task oluşturulduğunda ne oluyor?
        Kontrol: Boş liste durumunun doğru şekilde işlenmesi
        """
        task_description = "Perform analysis without specific context"
        task = Task(task_description, context=[])
        
        # Check that context is properly initialized as empty list
        assert task.context is not None
        assert isinstance(task.context, list)
        assert len(task.context) == 0
        
        # Test agent behavior with empty context
        agent = Agent(name="Analyzer")
        
        result = agent.print_do(task)
        
        assert isinstance(result, str)
        assert task.response == result
