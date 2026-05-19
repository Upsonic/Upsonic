from upsonic import Agent, Task
from pydantic import BaseModel

class ReportResult(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    confidence: float

# Create and execute task
agent = Agent(model="openai/gpt-4o", name="Analysis Agent")
task = Task(
    description="Generate a market analysis report",
    response_format=ReportResult,
    enable_cache=True
)

result = agent.do(task)

# Access all available information
print("=== TASK EXECUTION SUMMARY ===")
print(f"Task ID: {task.get_task_id()}")
print(f"Duration: {task.usage.duration:.2f} seconds")
print(f"Cost: ${task.usage.cost if task.usage else None}")
print(f"Tokens: {task.usage.input_tokens if task.usage else 0} in, "
      f"{task.usage.output_tokens if task.usage else 0} out")
print(f"Tool calls made: {len(task.tool_calls)}")
print(f"Cache hit: {task._cache_hit}")

print("\n=== TASK RESULT ===")
print(f"Result: {result}")
print(f"Response type: {type(task.response)}")

print("\n=== CACHE STATISTICS ===")
cache_stats = task.get_cache_stats()
for key, value in cache_stats.items():
    print(f"{key}: {value}")