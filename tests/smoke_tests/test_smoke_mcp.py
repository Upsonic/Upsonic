import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from upsonic import Agent, Task
from upsonic.tools import WebSearch, WebRead
from pydantic import BaseModel


# MCP 1: SQLite Database Operations
class DatabaseMCP:
    """
    MCP server for SQLite database operations.
    Provides comprehensive database management capabilities.
    """
    command = "uvx"
    args = ["mcp-server-sqlite", "--db-path", "/tmp/library.db"]


# Response Format for Structured Output
class DatabaseReport(BaseModel):
    tables_created: int
    records_inserted: int
    sample_data: str
    summary: str


# Fixture for database agent
@pytest.fixture
def database_agent():
    """Create database management agent"""
    return Agent(
        name="Database Management Expert",
        role="Database operations and data management specialist",
        goal="Provide comprehensive database management with structured reporting"
    )


# Fixture for research agent
@pytest.fixture
def research_agent():
    """Create research and database agent"""
    return Agent(
        name="Research and Database Agent",
        role="Combined research and data management specialist",
        goal="Research topics and store findings in a structured database"
    )


# Fixture for workflow agent
@pytest.fixture
def workflow_agent():
    """Create multi-MCP workflow agent"""
    return Agent(
        name="Multi-MCP Workflow Agent",
        role="Multi-tool specialist",
        goal="Use multiple MCP tools to accomplish complex tasks"
    )


class TestMCPOutputMessages:
    """Test suite to verify specific output messages appear in the execution"""
    
    def test_basic_database_operations(self, database_agent, capsys):
        """
        Test basic database operations and check for required output lines
        """
        # Example 1: Basic Database Operations
        basic_task = Task(
            description="Create a comprehensive library management database. Create a 'books' table with columns: id (integer primary key), title (text), author (text), isbn (text), published_year (integer), and available (boolean). Insert 5 sample books with different authors and years. Then create a 'borrowers' table with columns: id (integer primary key), name (text), email (text), and phone (text). Insert 3 sample borrowers. Finally, show all books and all borrowers in separate queries.",
            tools=[DatabaseMCP]
        )
        
        # Execute the basic database operations
        result = database_agent.do(basic_task)
        
        # Capture output
        captured = capsys.readouterr()
        output = captured.out + captured.err
        
        # Assert that specific lines are present in the output
        assert "✅ MCP tools discovered via thread" in output or "✅ MCP tools discovered" in output, \
            "Expected MCP tools discovery message not found in output"
            
        assert "Found 6 tools" in output, \
            "Expected 'Found 6 tools from DatabaseMCP' message not found in output"
        
        # Check for agent started message (with flexible formatting)
        assert "Agent Started" in output, \
            "Expected 'Agent Started' message not found in output"
        
        # Check for tool usage summary
        assert "Tool Usage Summary" in output, \
            "Expected 'Tool Usage Summary' message not found in output"
    
    
  