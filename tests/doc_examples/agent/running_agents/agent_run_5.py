from upsonic import Agent, Task
from upsonic.run.events.events import TextDeltaEvent

agent = Agent("openai/gpt-4o")
task = Task("Explain AI briefly")

# Stream events
for event in agent.stream(task, events=True):
    if isinstance(event, TextDeltaEvent):
        print(event.content, end='', flush=True)

# After streaming completes, access the final output
# Option 1: From the task (most direct)
final_output = task.response
print(f"\nFinal (from task.response): {final_output}")

# Option 2: From the agent's run output (includes additional metadata)
run_output = agent.get_run_output()
if run_output:
    print(f"Final (from agent.get_run_output().content): {run_output.content}")
    # You can also access other metadata:
    # - run_output.usage (token usage)
    # - run_output.messages (all messages)
    # - run_output.tools (tool executions)
    # - run_output.status (run status)