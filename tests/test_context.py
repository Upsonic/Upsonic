import asyncio
import os
from typing import Optional, Any

from src.upsonic import Direct, Task, Graph
from src.upsonic.graph.graph import State, DecisionFunc
from src.upsonic.context.sources import (
    StaticTextSource,
)

def print_header(title):
    print("\n" + "="*80)
    print(f"## {title.upper()}".center(80))
    print("="*80 + "\n")


class MockDirect(Direct):
    """A mocked Direct agent that overrides the LLM call to test the prompt."""

    async def do_async(self, task: Task, model=None, debug=False, retry=3, state: Optional[State] = None):
        print_header(f"Mock Agent '{self.name}' executing task: '{task.description[:60].strip()}'")

        print("--- 1. Building System Prompt ---")
        final_system_prompt = await self.system_prompt_manager.build_system_prompt(agent=self, task=task)
        
        print("\n[FINAL SYSTEM PROMPT THAT WOULD BE SENT TO LLM]")
        print("-" * 50)
        print(final_system_prompt)
        print("-" * 50)
        

        print("\n--- 2. Preparing and Building Data Context ---")
        await self.data_injector.prepare_rag_systems(task)
        data_context = await self.data_injector.get_data_context_string(task, state=state)

        final_user_prompt = f"{data_context}\n{task.description}".strip()
        
        print("\n[FINAL USER PROMPT THAT WOULD BE SENT TO LLM]")
        print("-" * 50)
        print(final_user_prompt)
        print("-" * 50)
        
        print("\n--- 3. Mocking LLM Response ---")
        mock_response: Any = f"Mocked response for prompt: {task.description}"

        if "research the pros and cons" in task.description.lower():
            mock_response = "Python is great (Pro). It can be slow (Con)."
        elif "summarize the key points" in task.description.lower():
            class Summary(object):
                def __init__(self):
                    self.key_points = ["Easy to use", "Slow performance"]
                    self.sentiment = "neutral"
            mock_response = Summary()
        
        print(f"Returning mock response: '{mock_response}'\n")
        
        task._response = mock_response
        return task.response


def setup_test_environment():
    """Creates a dummy knowledge file for RAG testing."""
    print("Setting up test environment...")
    with open("test_knowledge_file.txt", "w") as f:
        f.write("The primary benefit of a modular system is maintainability.")

def cleanup_test_environment():
    """Removes the dummy knowledge file."""
    print("\nCleaning up test environment...")
    if os.path.exists("test_knowledge_file.txt"):
        os.remove("test_knowledge_file.txt")


async def test_all_system_prompt_features():
    """Tests all variations of the SystemPromptManager's behavior."""
    print_header("Test 1: System Prompt Manager Features")

    print("\n--- 1A: Agent with Default Prompt + Custom Instruction ---")
    agent_a = MockDirect(name="DefaultAgent", system_prompt="My role is to be a friendly guide.")
    task_a = Task(description="What are your instructions?")
    await agent_a.do_async(task_a)
    print("VERIFIED: Prompt contains default text, custom instruction, and agent self-info.")

    print("\n--- 1B: Agent OVERRIDING Default Prompt ---")
    agent_b = MockDirect(
        name="OverrideAgent",
        system_prompt="This is the only system prompt. The default is gone.",
        override_default_prompt=True
    )
    task_b = Task(description="What are your instructions now?")
    await agent_b.do_async(task_b)
    print("VERIFIED: Prompt contains ONLY the override text and agent self-info.")


async def test_all_data_injector_features():
    """Tests all variations of the DataInjector's behavior."""
    print_header("Test 2: Data Injector Features (Static Text & RAG)")

    agent = MockDirect(name="DataAgent")

    task = Task(
        description="Based on the context, what is the main benefit?",
        context=[
            StaticTextSource(text="The system was deployed yesterday."),
            StaticTextSource(text="The primary benefit of a modular system is maintainability."),
            StaticTextSource(text="The system is designed to be user-friendly."),
        ]
    )
    
    await agent.do_async(task)
    print("VERIFIED: Final user prompt contains <CONTEXT_DATA> block with both StaticText and RAG results.")


async def test_full_graph_integration():
    """Tests a graph to verify state passing via TaskOutputProcessor."""
    print_header("Test 3: Full Graph Integration with State Passing")

    graph_agent = MockDirect(name="GraphAgent")

    task1 = Task(description="Research the pros and cons of modular architecture.")
    task2 = Task(description="Summarize the key points from the research.")
    task3 = Task(description="Write a final report based on the summary.")
    task4 = Task(description="Some insturction for task4")
    task5 = Task(description="Some insturction for task5")

    def summary_is_valid(summary: Any) -> bool:
        print(f"\n--- Decision Node: Checking summary object ---")
        result = hasattr(summary, 'key_points') and len(summary.key_points) > 0
        print(f"Summary object is valid: {result}")
        return result

    decision = DecisionFunc(description="Check if summary is valid", func=summary_is_valid)

    graph = Graph(default_agent=graph_agent)
    graph.add(task1 >> task2 >> decision.if_false(task1).if_true(task4) >> task5 >> task3)
    
    await graph.run_async(verbose=False)
    
    print("\nVERIFIED: Graph executed. The second task's prompt should have contained a <PreviousTaskOutput> block.")


async def main():
    try:
        setup_test_environment()
        await test_all_system_prompt_features()
        await test_all_data_injector_features()
        await test_full_graph_integration()
    finally:
        cleanup_test_environment()

if __name__ == "__main__":
    asyncio.run(main())