import os

import time
import shutil
from pathlib import Path


from upsonic.agent.agent import Direct
from upsonic.tasks.tasks import Task
from upsonic.tools.tool import tool, ToolHooks
from upsonic.tools.processor import ToolValidationError


os.environ["LLM_MODEL_KEY"] = "openai/gpt-3.5-turbo"
CACHE_DIR = Path.home() / '.upsonic' / 'cache'
if CACHE_DIR.exists():
    shutil.rmtree(CACHE_DIR)



def get_current_time_as_string(timezone: str) -> str:
    """A simple, valid tool that can be used for multiple tests."""
    return f"The time in {timezone} is {time.time()}"

class MyToolbox:
    """A class to test instance methods as tools."""
    def __init__(self, prefix: str):
        self.prefix = prefix

    def greet(self, name: str) -> str:
        """Greets the given name with a prefix."""
        return f"{self.prefix}, {name}!"

hook_events = []

def before_hook_func(*args, **kwargs):
    hook_events.append("before_hook_called")

def after_hook_func(result: any):
    hook_events.append("after_hook_called")

def print_header(title):
    print("\n" + "="*80)
    print(f"  🧪  TESTING: {title}")
    print("="*80)

def print_result(description, result):
    print(f"\n  - Description: {description}")
    print(f"  - Agent Output: \n\n    {result}\n")
    print("-" * 80)


def run_tests():
    agent = Direct(name="Test", debug=True, model="openai/gpt-4o")

    print_header("Tool Definition Styles")


    """
    task = Task("Use the tool to get the current time for the 'UTC' timezone", tools=[get_current_time_as_string])
    result = agent.do(task)
    print_result("Undecorated function as a tool", result)

    # Test undecorated class instance methods
    my_toolbox = MyToolbox(prefix="Welcome")
    task = Task("Use the toolbox to greet 'Onur'", tools=[my_toolbox])
    result = agent.do(task)
    print_result("Undecorated class instance methods as tools", result)
    """

    print_header("Tool Behaviors")

    @tool(requires_confirmation=True)
    def sensitive_action(action: str) -> str:
        """Performs a sensitive action that requires confirmation."""
        return f"Action '{action}' has been confirmed and executed."
    
    task = Task("Perform the sensitive action 'delete_database'", tools=[sensitive_action])
    print("\n--- Testing requires_confirmation ---")
    print(">>> When prompted, please type 'y' and press Enter <<<")
    result = agent.do(task)
    print_result("Tool with requires_confirmation (user says 'yes')", result)
    
    print("\n>>> When prompted, please type 'n' and press Enter <<<")
    result = agent.do(task)
    print_result("Tool with requires_confirmation (user says 'no')", result)

    @tool(requires_user_input=True, user_input_fields=['destination'])
    def book_flight(destination: str, passengers: int = 1) -> str:
        """Books a flight, getting the destination interactively."""
        return f"Booked a flight for {passengers} to '{destination}'."

    task = Task("Book a flight for one person to a destination you will ask me for.", tools=[book_flight])
    print("\n--- Testing requires_user_input ---")
    print(">>> When prompted for 'destination', please type 'Paris' and press Enter <<<")
    result = agent.do(task)
    print_result("Tool with requires_user_input", result)
    
    @tool(stop_after_tool_call=True, show_result=True)
    def get_raw_data() -> dict:
        """Fetches raw data and the agent should stop immediately."""
        return {"id": "user-123", "data": [1, 2, 3], "status": "ok"}
    
    task = Task("Get the raw data and show it to me.", tools=[get_raw_data])
    print("\n--- Testing stop_after_tool_call and show_result ---")
    result = agent.do(task)
    print_result("Tool with stop_after_tool_call and show_result", f"Type of result: {type(result)}\n    Value: {result}")
    

    @tool(cache_results=True)
    def get_cached_time(timezone: str) -> float:
        """Returns a precise time that should be cached."""
        return time.time()
        
    task_cache = Task("Get the cached time for 'PST'", tools=[get_cached_time])
    print("\n--- Testing cache_results ---")
    print(">>> First call (should execute normally)...")
    result1 = agent.do(task_cache)
    print_result("Cache Test - First Call", result1)

    print(">>> Second call (should hit the cache and return the exact same value)...")
    time.sleep(0.1) 
    result2 = agent.do(task_cache)
    print_result("Cache Test - Second Call (Cache Hit)", f"{result2} (Same as first call: {result1 == result2})")
    
    hook_events.clear()
    hooks = ToolHooks(before=before_hook_func, after=after_hook_func)
    @tool(tool_hooks=hooks)
    def tool_with_hooks() -> str:
        """A tool that triggers hooks."""
        hook_events.append("tool_executed")
        return "Tool has finished execution."

    task_hooks = Task("Run the tool that has hooks", tools=[tool_with_hooks])
    print("\n--- Testing tool_hooks ---")
    agent.do(task_hooks)
    print_result("Tool with before/after hooks", f"Execution order: {hook_events}")


if __name__ == "__main__":
    run_tests()