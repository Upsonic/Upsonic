from upsonic import Task, Agent

task = Task("What is the capital of France?")

agent = Agent()

# Use do() to get RunResult, or just inspect get_run_result() after print_do()
agent.print_do(task)

print("\n=== RESULT OBJECT ===")
result = agent.do(task)  # This returns the output (string), not RunResult
print(f"Result type: {type(result)}")
print(f"Result value: {result}")

print("\n=== RUN RESULT ===")
run_result = agent.get_run_result()
print(f"RunResult type: {type(run_result)}")
print(f"RunResult.output: {run_result.output}")
print(f"RunResult.output type: {type(run_result.output)}")
print(f"RunResult._all_messages count: {len(run_result._all_messages)}")
print(f"RunResult._run_boundaries: {run_result._run_boundaries}")

print("\n=== ALL MESSAGES ===")
all_msgs = run_result.all_messages()
print(f"all_messages() count: {len(all_msgs)}")
for i, msg in enumerate(all_msgs):
    print(f"  Message {i}: {type(msg).__name__}")
    if hasattr(msg, "parts"):
        print(f"    Parts: {len(msg.parts)}")
        for j, part in enumerate(msg.parts):
            print(f"      Part {j}: {type(part).__name__}")
            if hasattr(part, "content"):
                print(f"        Content preview: {str(part.content)[:50]}...")

print("\n=== NEW MESSAGES ===")
new_msgs = run_result.new_messages()
print(f"new_messages() count: {len(new_msgs)}")

print("\n=== REPR ===")
print(repr(run_result))
