"""
Comprehensive test file for all debug levels across all classes

This file demonstrates debug level 1 and 2 for:
- Agent
- Chat
- Team
- DeepAgent
- Graph

Run with: python test_debug_all.py
"""

import asyncio
from upsonic import Agent, Chat, Team, Task
from upsonic.agent.deepagent import DeepAgent
from upsonic import Graph


async def test_all_debug_levels():
    """Test all classes with both debug levels."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DEBUG LEVEL TESTING")
    print("=" * 80)
    
    # ============================================
    # 1. AGENT TESTS
    # ============================================
    print("\n" + "=" * 80)
    print("1. AGENT - Debug Level 1")
    print("=" * 80)
    agent1 = Agent(
        model="openai/gpt-4o-mini",
        name="Test Agent",
        debug=True,
        debug_level=1
    )
    result1 = await agent1.do_async("What is 2 + 2?")
    print(f"Result: {result1}")
    
    print("\n" + "=" * 80)
    print("2. AGENT - Debug Level 2")
    print("=" * 80)
    agent2 = Agent(
        model="openai/gpt-4o-mini",
        name="Test Agent",
        debug=True,
        debug_level=2
    )
    result2 = await agent2.do_async("What is 2 + 2?")
    print(f"Result: {result2}")
    
    # ============================================
    # 3. CHAT TESTS
    # ============================================
    print("\n" + "=" * 80)
    print("3. CHAT - Debug Level 1")
    print("=" * 80)
    chat_agent1 = Agent(model="openai/gpt-4o-mini")
    chat1 = Chat(
        session_id="test_1",
        user_id="user_1",
        agent=chat_agent1,
        debug=True,
        debug_level=1
    )
    chat_result1 = await chat1.invoke("Hello! What is 3 + 3?")
    print(f"Result: {chat_result1}")
    
    print("\n" + "=" * 80)
    print("4. CHAT - Debug Level 2")
    print("=" * 80)
    chat_agent2 = Agent(model="openai/gpt-4o-mini")
    chat2 = Chat(
        session_id="test_2",
        user_id="user_2",
        agent=chat_agent2,
        debug=True,
        debug_level=2
    )
    chat_result2 = await chat2.invoke("Hello! What is 3 + 3?")
    print(f"Result: {chat_result2}")
    
    # ============================================
    # 5. TEAM TESTS
    # ============================================
    print("\n" + "=" * 80)
    print("5. TEAM - Debug Level 1")
    print("=" * 80)
    team_agent1 = Agent(
        model="openai/gpt-4o-mini",
        name="Team Agent 1",
        debug=True,
        debug_level=1
    )
    team1 = Team(
        agents=[team_agent1],
        mode="sequential",
        debug=True,
        debug_level=1
    )
    team_result1 = await team1.multi_agent_async([team_agent1], Task("What is 4 + 4?"))
    print(f"Result: {team_result1}")
    
    print("\n" + "=" * 80)
    print("6. TEAM - Debug Level 2")
    print("=" * 80)
    team_agent2 = Agent(
        model="openai/gpt-4o-mini",
        name="Team Agent 2",
        debug=True,
        debug_level=2
    )
    team2 = Team(
        agents=[team_agent2],
        mode="sequential",
        debug=True,
        debug_level=2
    )
    team_result2 = await team2.multi_agent_async([team_agent2], Task("What is 4 + 4?"))
    print(f"Result: {team_result2}")
    
    # ============================================
    # 7. DEEPAGENT TESTS
    # ============================================
    print("\n" + "=" * 80)
    print("7. DEEPAGENT - Debug Level 1")
    print("=" * 80)
    deepagent1 = DeepAgent(
        model="openai/gpt-4o-mini",
        name="DeepAgent 1",
        debug=True,
        debug_level=1
    )
    deep_result1 = await deepagent1.do_async(Task("What is 5 + 5?"))
    print(f"Result: {deep_result1}")
    
    print("\n" + "=" * 80)
    print("8. DEEPAGENT - Debug Level 2")
    print("=" * 80)
    deepagent2 = DeepAgent(
        model="openai/gpt-4o-mini",
        name="DeepAgent 2",
        debug=True,
        debug_level=2
    )
    deep_result2 = await deepagent2.do_async(Task("What is 5 + 5?"))
    print(f"Result: {deep_result2}")
    
    # ============================================
    # 9. GRAPH TESTS
    # ============================================
    print("\n" + "=" * 80)
    print("9. GRAPH - Debug Level 1")
    print("=" * 80)
    graph_agent1 = Agent(
        model="openai/gpt-4o-mini",
        name="Graph Agent 1",
        debug=True,
        debug_level=1
    )
    graph1 = Graph(
        default_agent=graph_agent1,
        debug=True,
        debug_level=1
    )
    graph1.add(Task("What is 6 + 6?"))
    graph_result1 = await graph1.run()
    print(f"Result: {graph_result1}")
    
    print("\n" + "=" * 80)
    print("10. GRAPH - Debug Level 2")
    print("=" * 80)
    graph_agent2 = Agent(
        model="openai/gpt-4o-mini",
        name="Graph Agent 2",
        debug=True,
        debug_level=2
    )
    graph2 = Graph(
        default_agent=graph_agent2,
        debug=True,
        debug_level=2
    )
    graph2.add(Task("What is 6 + 6?"))
    graph_result2 = await graph2.run()
    print(f"Result: {graph_result2}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_all_debug_levels())
