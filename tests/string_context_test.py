import pytest
from unittest.mock import patch, Mock
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
        Simulasyon: Agent'ın context'e erişebildiğini mock ile test etme
        """
        city = "New York"
        task = Task("Find resources in the city", context=[city])
        agent = Agent(name="City Guide")
        
        # Mock agent.print_do 
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect(task_obj):
                if task_obj.context and len(task_obj.context) > 0:
                    used_city = task_obj.context[0]
                    response = f"Found resources in {used_city}: Museums, Parks, Restaurants"
                    task_obj._response = response
                    return response
                else:
                    response = "No context provided"
                    task_obj._response = response
                    return response
            
            mock_print_do.side_effect = side_effect
            
            result = agent.print_do(task)
            
            assert "New York" in result  
            assert task.response == result  
            assert isinstance(result, str)  

    def test_agent_can_access_multiple_string_contexts(self):
        """
        Test: Agent'ın birden çok string context'i kullanabilmesi
        Kontrol: Tüm context'lerin agent tarafından erişilebilir olması
        """
        contexts = ["London", "Technology", "2024"]
        task = Task("Create a comprehensive analysis", context=contexts)
        agent = Agent(name="Analyst")
        
        # Mock agent.print_do 
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect(task_obj):
                if task_obj.context and len(task_obj.context) > 0:
                    all_contexts = ", ".join(task_obj.context)
                    response = f"Analysis completed using contexts: {all_contexts}"
                    task_obj._response = response
                    return response
                else:
                    response = "No contexts available"
                    task_obj._response = response
                    return response
            
            mock_print_do.side_effect = side_effect
            
            result = agent.print_do(task)
            
            assert "London" in result  
            assert "Technology" in result  
            assert "2024" in result  
            assert "London, Technology, 2024" in result  
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
        Test: Agent'ın context'i task description ile nasıl entegre ettiğinin simülasyonu
        Kontrol: Context'in task description'a uygun şekilde kullanılması
        """
        city = "Tokyo"
        task = Task("Find the best restaurants", context=[city])
        agent = Agent(name="Food Guide")
        
        # Mock 
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect(task_obj):
                description = task_obj.description
                context_city = task_obj.context[0] if task_obj.context else "unknown location"
                
                integrated_response = f"Task: {description} | Location: {context_city} | Result: Top sushi restaurants in {context_city}"
                task_obj._response = integrated_response
                return integrated_response
            
            mock_print_do.side_effect = side_effect
            
            result = agent.print_do(task)
            
            assert "Find the best restaurants" in result  
            assert "Tokyo" in result  
            assert "Location: Tokyo" in result  
            assert task.response == result  
            
    def test_context_with_non_string_values(self):
        task = Task("Handle mixed context", context=["valid", 123, None])
        agent = Agent(name="Robust")
        
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect(task_obj):
                return f"Context Types: {[type(c).__name__ for c in task_obj.context]}"
            mock_print_do.side_effect = side_effect
            result = agent.print_do(task)
            
            assert "int" in result
            assert "NoneType" in result

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
        
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect(task_obj):
                if not task_obj.context or len(task_obj.context) == 0:
                    response = "No context provided - performing general analysis"
                    task_obj._response = response
                    return response
                else:
                    response = f"Analysis with context: {task_obj.context}"
                    task_obj._response = response
                    return response
            
            mock_print_do.side_effect = side_effect
            
            result = agent.print_do(task)
            
            assert "No context provided" in result
            assert task.response == result
            assert isinstance(result, str)
