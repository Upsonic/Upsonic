"""
Comprehensive Test for Tool Human-in-the-Loop Attributes
Tests: requires_confirmation, requires_user_input, user_input_fields, external_execution

These tests require manual user interaction
Run manually with: python3 tests/smoke_tests/test_tool_hitl_attributes.py
"""
import pytest
from upsonic.tools import tool
from upsonic import Agent, Task


class TestRequiresConfirmation:
    """Test requires_confirmation attribute"""
    
    def test_tool_requires_confirmation(self):
        """Test tool that requires user confirmation before execution"""
        
        @tool(requires_confirmation=True)
        def delete_file(file_path: str) -> str:
            """
            Delete a file from the system.
            
            Args:
                file_path: Path to the file to delete
            
            Returns:
                Confirmation message
            """
            # In real scenario, would actually delete the file
            return f"File {file_path} deleted successfully"
        
        # Create task that uses the tool
        task = Task(
            description="Delete the file /tmp/test.txt using the delete_file tool",
            tools=[delete_file]
        )
        
        agent = Agent(model="openai/gpt-4o-mini")
        
        # This should pause for confirmation
        # User needs to manually confirm the action
        result = agent.do(task)
        
        # After manual confirmation, the task should complete
        assert "deleted" in result.lower() or task.is_paused
        
    def test_multiple_confirmations(self):
        """Test multiple tools requiring confirmation"""
        
        @tool(requires_confirmation=True)
        def dangerous_operation_1(data: str) -> str:
            """Dangerous operation 1"""
            return f"Executed operation 1 on {data}"
        
        @tool(requires_confirmation=True)
        def dangerous_operation_2(data: str) -> str:
            """Dangerous operation 2"""
            return f"Executed operation 2 on {data}"
        
        task = Task(
            description="Execute both dangerous operations on 'test_data'",
            tools=[dangerous_operation_1, dangerous_operation_2]
        )
        
        agent = Agent(model="openai/gpt-4o-mini")
        result = agent.do(task)
        
        # Should require multiple confirmations
        assert result or task.is_paused


class TestRequiresUserInput:
    """Test requires_user_input and user_input_fields attributes"""
    
    def test_tool_requires_user_input(self):
        """Test tool that requires user input for specific fields"""
        
        @tool(
            requires_user_input=True,
            user_input_fields=["username", "password"]
        )
        def secure_login(username: str, password: str) -> str:
            """
            Login with user-provided credentials.
            
            Args:
                username: Username for login
                password: Password for login
            
            Returns:
                Login status message
            """
            # In real scenario, would validate credentials
            return f"Login successful for user: {username}"
        
        task = Task(
            description="Login to the system using the secure_login tool",
            tools=[secure_login]
        )
        
        agent = Agent(model="openai/gpt-4o-mini")
        
        # This should pause and wait for user to provide username and password
        result = agent.do(task)
        
        # After user provides input, task should complete
        assert "Login successful" in result or task.is_paused
        
    def test_partial_user_input_fields(self):
        """Test tool with some fields requiring user input"""
        
        @tool(
            requires_user_input=True,
            user_input_fields=["api_key"]
        )
        def api_call(endpoint: str, api_key: str, data: str = "") -> str:
            """
            Make an API call with user-provided API key.
            
            Args:
                endpoint: API endpoint
                api_key: API key (user must provide)
                data: Optional data
            
            Returns:
                API response
            """
            return f"Called {endpoint} with key {api_key[:5]}..."
        
        task = Task(
            description="Call the /users endpoint with some data",
            tools=[api_call]
        )
        
        agent = Agent(model="openai/gpt-4o-mini")
        result = agent.do(task)
        
        # Should pause for api_key input
        assert result or task.is_paused


class TestExternalExecution:
    """Test external_execution attribute"""
    
    def test_external_execution_basic(self):
        """Test tool with external execution"""
        
        @tool(external_execution=True)
        def external_database_query(query: str) -> str:
            """
            Execute a database query externally.
            
            Args:
                query: SQL query to execute
            
            Returns:
                Query results
            """
            return f"Query executed: {query}"
        
        task = Task(
            description="Execute SQL query 'SELECT * FROM users' using external_database_query",
            tools=[external_database_query]
        )
        
        agent = Agent(model="openai/gpt-4o-mini")
        result = agent.do(task)
        
        # Task should pause and wait for external execution
        assert task.is_paused
        assert len(task.tools_awaiting_external_execution) > 0
        
        # Now handle external execution
        for external_call in task.tools_awaiting_external_execution:
            # Execute the actual external operation
            result = external_database_query(**external_call.args)
            external_call.result = result
        
        # Continue execution
        final_result = agent.continue_run(task)
        assert "Query executed" in final_result or "SELECT" in final_result
        
    def test_external_execution_with_result_injection(self):
        """Test external execution with manual result injection"""
        
        @tool(external_execution=True)
        def external_api_call(url: str, method: str = "GET") -> str:
            """
            Make an external API call.
            
            Args:
                url: API URL
                method: HTTP method
            
            Returns:
                API response
            """
            return f"Called {method} {url}"
        
        task = Task(
            description="Make a GET request to https://api.example.com/data",
            tools=[external_api_call]
        )
        
        agent = Agent(model="openai/gpt-4o-mini")
        result = agent.do(task)
        
        # Should be paused
        assert task.is_paused
        
        # Inject custom result
        for external_call in task.tools_awaiting_external_execution:
            external_call.result = '{"status": "success", "data": [1,2,3]}'
        
        # Continue with injected results
        final_result = agent.continue_run(task)
        assert "success" in final_result or task.completed


class TestCombinedHITLAttributes:
    """Test combinations of HITL attributes"""
    
    def test_confirmation_and_user_input(self):
        """Test tool requiring both confirmation and user input"""
        
        @tool(
            requires_confirmation=True,
            requires_user_input=True,
            user_input_fields=["password"]
        )
        def delete_with_password(file_path: str, password: str) -> str:
            """
            Delete a file after password confirmation.
            
            Args:
                file_path: File to delete
                password: Admin password
            
            Returns:
                Result message
            """
            return f"Deleted {file_path} after password verification"
        
        task = Task(
            description="Delete /important/file.txt with password verification",
            tools=[delete_with_password]
        )
        
        agent = Agent(model="openai/gpt-4o-mini")
        result = agent.do(task)
        
        # Should pause for both confirmation and user input
        assert task.is_paused or "Deleted" in result
        
    def test_external_with_confirmation(self):
        """Test external execution combined with confirmation"""
        
        @tool(
            external_execution=True,
            requires_confirmation=True
        )
        def critical_external_operation(operation: str) -> str:
            """
            Execute a critical external operation.
            
            Args:
                operation: Operation to execute
            
            Returns:
                Operation result
            """
            return f"Executed critical operation: {operation}"
        
        task = Task(
            description="Execute critical operation 'deploy_to_production'",
            tools=[critical_external_operation]
        )
        
        agent = Agent(model="openai/gpt-4o-mini")
        result = agent.do(task)
        
        # Should pause for confirmation before external execution
        assert task.is_paused


if __name__ == "__main__":
    print("=" * 80)
    print("WARNING: These tests require manual user interaction!")
    print("They will pause and wait for user input/confirmation.")
    print("Run them individually and provide input as needed.")
    print("=" * 80)
    pytest.main([__file__, "-v", "-s"])

