from upsonic import Task, Agent
from upsonic.tools import tool
from pydantic import BaseModel
from upsonic.tools import YFinanceTools





task = Task(
	description="Solve this complex problem: If a train travels at 60 mph for 2.5 hours, how far does it go?",
	enable_thinking_tool=True,
	enable_reasoning_tool=True
)
agent = Agent(name="Reasoning Agent")
agent.print_do(task)	
	
