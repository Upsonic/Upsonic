from upsonic import Task, Agent

list_of_models = [
    "openai/gpt-4o",
    "anthropic/claude-3-5-sonnet-latest",
    "gemini/gemini-2.5-pro",
]

task = Task("What is the capital of France?")

for model in list_of_models:
    agent = Agent(model=model)
    result = agent.print_do(task)
