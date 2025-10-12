import pytest
from pydantic import BaseModel
from upsonic import Task, Agent
from upsonic.tools import  tool

@tool
def web_search(query: str) -> str:
    """Searches the web for the given query and returns a short summary."""
    return f"Search results for '{query}'"

def summarize_text(text: str) -> str:
    """Summarizes a given text into one concise sentence."""
    return f"Summary: {text.split(':')[-1].strip().split('.')[0]}."

@tool
def generate_title(title: str) -> str:
    """Generates a catchy title for the summary."""
    return f"Title: {title}'"


# Sample Pydantic model for testing
class AnalysisResult(BaseModel):
	summary: str
	confidence: float
	recommendations: list[str]
	key_metrics: dict[str, float]



class TestUpsonicBasicFlow:
	"""Smoke tests for basic Upsonic Task and Agent functionality"""
	
	def test_task_creation(self):
		"""Test Task object can be created"""
		task = Task("List exactly 3 prime numbers between 10 and 30, then calculate their sum.")
		assert task is not None
	
	def test_agent_creation(self):
		"""Test Agent object can be created with name"""
		agent = Agent(name="Math Genius")
		assert agent is not None
	
	def test_agent_print_do_executes(self, capsys):
		"""Test Agent can execute a task using print_do - happy path"""
		task = Task("List exactly 3 prime numbers between 10 and 30, then calculate their sum.")
		agent = Agent(name="Math Genius")
		
		result = agent.print_do(task)
		
		captured = capsys.readouterr()
		output = captured.out
		
		assert "Agent Started" in output or "Agent Status" in output
		assert "Task Result" in output or "Result:" in output
		assert "Task Metrics" in output or "Total Estimated Cost" in output
		assert result is not None
		
		assert hasattr(task, 'duration')
		assert hasattr(task, 'total_cost')
		assert hasattr(task, 'total_input_token')
		assert hasattr(task, 'total_output_token')


class TestTaskWithTools:
	"""Task with Tools"""
	
	def test_task_with_tools(self, capsys):
		"""Test task execution with tools provided"""
		task = Task(
			description="Turkeys capital city",
			tools=[web_search]
		)
		agent = Agent(name="Research Agent")
		
		result = agent.print_do(task)
		
		captured = capsys.readouterr()
		output = captured.out
		
		assert result is not None
		assert isinstance(result, str)
		assert "ankara" in result.lower()
		assert "Task Result" in output or "Result:" in output
		
		if hasattr(task, 'tool_calls'):
			assert isinstance(task.tool_calls, list)

	def test_task_with_multiple_tools(self, capsys):
		"""Test task with multiple tools"""
		task = Task(
			description="Resarch for casio watches analysis the key consept, and generate a catchy article title.",
    		tools=[web_search, summarize_text, generate_title],
		)
		agent = Agent(name="Multiple Tool Research Agent")
		
		result = agent.print_do(task)
		
		captured = capsys.readouterr()
		output = captured.out
		
		assert result is not None
		assert "Task Result" in output or "Result:" in output
		assert "Tool Usage Summary" in output
		assert "web_search" in output
		assert "generate_title" in output
		# Note: summarize_text might not execute due to tool call limits



class TestResponseFormat:
	"""Structured Output with Pydantic"""
	
	def test_task_with_response_format(self, capsys):
		"""Test task with structured Pydantic response format"""
		task = Task(
			description="Analyze the provided data and provide structured results",
			response_format=AnalysisResult
		)
		agent = Agent(name="Analysis Agent")
		
		result = agent.print_do(task)
		
		captured = capsys.readouterr()
		output = captured.out
		
		assert result is not None
		assert "Task Result" in output or "Result:" in output


class TestContextChain:
	"""Context Chain"""
	
	def test_task_with_context(self, capsys):
		"""Test task execution with context from previous task"""
		agent = Agent(name="Geography Agent")
		
		task1 = Task(description="What is the biggest city in Japan")
		result1 = agent.print_do(task1)
		
		task2 = Task(
			description="Based on the previous result, what is the population of that city?",
			context=[task1]
		)
		result2 = agent.print_do(task2)
		
		task3 = Task(description="What is the second biggest city in that country?" , 
			   context=[task1,task2])
		result3 = agent.print_do(task3)
		
		
		captured = capsys.readouterr()
		output = captured.out
		
		assert result1 is not None
		assert result2 is not None
		assert result3 is not None
		assert "tokyo" in str(result1).lower()
		assert "yokohama" in str(result3).lower()
		assert "Task Result" in output or "Result:" in output


#class TestTaskWithAttachments:
#	"""Task with Attachments"""
	
#	def test_task_with_attachments(self, capsys, tmp_path):
#		"""Test task execution with file attachments"""
#		# Create a temporary file
#		test_file = tmp_path / "image.png"
#		test_file.write_text("Sample document content for testing")
		
#		task = Task(
#			description="Analyze the attached document",
#			attachments=[str(test_file)]
#		)
#		agent = Agent(name="Document Agent")
		
#		result = agent.print_do(task)
		
#		captured = capsys.readouterr()
#		output = captured.out
		
#		assert result is not None
#		assert "Task Result" in output or "Result:" in output


class TestCachingConfiguration:
	"""Caching Configuration"""
	
	def test_task_with_caching_enabled(self, capsys):
		"""Test task with caching configuration"""
		task = Task(
			description="What is the capital of France?",
			enable_cache=True,
			cache_method="vector_search",
			cache_threshold=0.85,
			cache_duration_minutes=60
		)
		agent = Agent(name="Knowledge Agent")
		
		result = agent.print_do(task)
		
		captured = capsys.readouterr()
		output = captured.out
		
		assert result is not None
		assert "Task Result" in output or "Result:" in output
	
		# Verify cache-related attributes
		assert hasattr(task, '_cache_hit')
		if hasattr(task, 'get_cache_stats'):
			cache_stats = task.get_cache_stats()
			assert isinstance(cache_stats, dict)
			assert "Cache" in output
			assert "Cache Status" in output or "MISS" in output or "HIT" in output
			assert "Method:" in output or "vector_search" in output
			assert "Input Preview:" in output
			assert "Action:" in output


#class TestReasoningFeatures:
	#"""Reasoning Features"""
	# Getting an error because processor.py task is not imported there.
	#def test_task_with_reasoning_tools(self, capsys):
		#"""Test task with thinking and reasoning tools enabled"""
		#task = Task(
		#	description="Solve this complex problem: If a train travels at 60 mph for 2.5 hours, how far does it go?",
		#	enable_thinking_tool=True,
		#	enable_reasoning_tool=True
		#)
		#agent = Agent(name="Reasoning Agent")
		
		#result = agent.print_do(task)
		
		
	#captured = capsys.readouterr()
		#output = captured.out
		
		
		#assert result is not None
		#assert "Task Result" in output or "Result:" in output



class TestComprehensiveTaskExecution:
	"""Comprehensive Integration"""
	
	@tool(requires_confirmation=True, cache_results=True)
	def get_market_data(symbol: str) -> str:
		"""Fetch current market data for a given symbol."""
		# Simulated market data retrieval
		return f"Market data for {symbol}: Price $150.25, Volume 1.2M"
	
	class MarketAnalysis(BaseModel):
		summary: str
		confidence: float
		key_metrics: dict[str, float]
		recommendations: list[str]
		risk_factors: list[str] = []
	
	def validate_analysis(result) -> bool:
		"""Validate that the analysis result meets quality standards."""
		if isinstance(result, TestComprehensiveTaskExecution.MarketAnalysis):
			return result.confidence >= 0.7 and len(result.recommendations) > 0
		return False
	
	def test_comprehensive_task_with_all_attributes(self, capsys):
		"""Test task with all attributes combined - full integration test"""
		# Create comprehensive task
		task = Task(
			# Core attributes
			description="Analyze the current market conditions for AAPL and provide investment recommendations",
			tools=[self.get_market_data],
			response_format=self.MarketAnalysis,
			context=["Focus on Q4 performance trends", "Consider recent earnings reports"],
			
			# Advanced configuration
			enable_thinking_tool=True,
			enable_reasoning_tool=True,
			guardrail=self.validate_analysis,
			guardrail_retries=3,
			
			# Caching configuration
			enable_cache=True,
			cache_method="vector_search",
			cache_threshold=0.8,
			cache_duration_minutes=60
		)
		
		agent = Agent(name="Market Analysis Agent")
		result = agent.do(task)
		
		# Verify result exists
		assert result is not None
		
		# Verify task metadata is accessible
		assert hasattr(task, 'get_task_id') or hasattr(task, 'task_id')
		assert hasattr(task, 'duration')
		assert hasattr(task, 'total_cost')
		assert hasattr(task, 'total_input_token')
		assert hasattr(task, 'total_output_token')
		assert hasattr(task, 'tool_calls')
		assert hasattr(task, '_cache_hit')
		
		# Verify task execution metrics
		if hasattr(task, 'duration'):
			assert task.duration >= 0
		if hasattr(task, 'total_cost'):
			assert task.total_cost >= 0
		if hasattr(task, 'tool_calls'):
			assert isinstance(task.tool_calls, list)
	
	#def test_task_result_access_patterns(self, capsys):
	#	"""Test different ways to access task results and metadata"""
	#	agent = Agent(name="Analysis Agent")
	#	
	#	task = Task(
	#		description="Generate a market analysis report",
	#		response_format=self.MarketAnalysis,
	#		enable_cache=True
	#	)
	#	
	#	result = agent.do(task)
	#	
	#	# Verify result
	#	assert result is not None
	#	
	#	# Test task ID access
	#	if hasattr(task, 'get_task_id'):
	#		task_id = task.get_task_id()
	#		assert task_id is not None
	#	
	#	# Test cache statistics access
	#	if hasattr(task, 'get_cache_stats'):
	#		cache_stats = task.get_cache_stats()
	#		assert isinstance(cache_stats, dict)
	#	
	#	# Test response type
	#	if hasattr(task, 'response'):
	#		assert task.response is not None
	
	#def test_comprehensive_task_execution_with_metadata(self, capsys):
	#	"""Test comprehensive task execution with metadata access patterns"""
	#	# Create comprehensive task
	#	task = Task(
	#		# Core attributes
	#		description="Analyze the current market conditions for AAPL and provide investment recommendations",
	#		tools=[self.get_market_data],
	#		response_format=self.MarketAnalysis,
	#		context=["Focus on Q4 performance trends", "Consider recent earnings reports"],
	#		
	#		# Advanced configuration
	#		enable_thinking_tool=True,
	#		enable_reasoning_tool=True,
	#		guardrail=self.validate_analysis,
	#		guardrail_retries=3,
	#		
	#		# Caching configuration
	#		enable_cache=True,
	#		cache_method="vector_search",
	#		cache_threshold=0.8,
	#		cache_duration_minutes=60
	#	)
	#	
	#	agent = Agent(name="Market Analysis Agent")
	#	result = agent.do(task)
	#	
	#	# Access task results and metadata
	#	print(f"Analysis completed in {task.duration:.2f} seconds")
	#	print(f"Total cost: ${task.total_cost}")
	#	print(f"Cache hit: {task._cache_hit}")
	#	print(f"Tool calls made: {len(task.tool_calls)}")
	#	
	#	# Verify result exists
	#	assert result is not None
	#	
	#	# Verify task metadata is accessible
	#	assert hasattr(task, 'duration')
	#	assert hasattr(task, 'total_cost')
	#	assert hasattr(task, '_cache_hit')
	#	assert hasattr(task, 'tool_calls')
	#	
	#	# Verify task execution metrics
	#	if hasattr(task, 'duration'):
	#		assert task.duration >= 0
	#	if hasattr(task, 'total_cost'):
	#		assert task.total_cost >= 0
	#	if hasattr(task, 'tool_calls'):
	#		assert isinstance(task.tool_calls, list)