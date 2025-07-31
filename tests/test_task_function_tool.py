from upsonic import Agent, Task

class CallTracker:
    """
    This class wraps a function and tracks if it was called and with which arguments.
    """
    def __init__(self):
        self.called_with = None
        self.call_count = 0

    def sum(self, a: int, b: int):
        """
        Custom sum function that also logs its call parameters.
        """
        self.called_with = (a, b)
        self.call_count += 1
        print(f"[DEBUG] sum() called with: {a}, {b}")
        return a + b


def test_agent_tool_function_call():
    # Test parameters
    num_a = 12
    num_b = 51
    expected_result = num_a + num_b
    
    tracker = CallTracker()
    task = Task(f"What is the sum of {num_a} and {num_b}? Use Tool", tools=[tracker.sum])
    agent = Agent(name="Sum Agent", model="openai/gpt-4o")

    result = agent.do(task)

    print(f"[DEBUG] Agent result: {result}")
    print(f"[DEBUG] Function call count: {tracker.call_count}")
    print(f"[DEBUG] Called with: {tracker.called_with}")

    # Assert
    assert tracker.call_count == 1, "The tool function was not called exactly once."
    assert tracker.called_with == (num_a, num_b), f"Function was called with wrong arguments: {tracker.called_with}"
    assert str(expected_result) in str(result), f"Expected result '{expected_result}' not found in agent output: {result}"
