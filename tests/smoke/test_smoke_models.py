import pytest
from upsonic import Task, Agent
#t

def test_models(capsys):
    list_of_models = [
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet-latest",
        "gemini/gemini-2.5-pro",
    ]
    task = Task("What is the capital of Turkey?")
    for model in list_of_models:
        agent = Agent(model=model)
        agent.print_do(task)

    captured = capsys.readouterr()
    out = captured.out

    agent_started_count = out.count("Agent Started")
    task_result_count = out.count("Task Result")

    assert agent_started_count == len(list_of_models), (
        f"Expected {len(list_of_models)} 'Agent Started', got {agent_started_count}"
    )
    assert task_result_count == len(list_of_models), (
        f"Expected {len(list_of_models)} 'Task Result', got {task_result_count}"
    )
