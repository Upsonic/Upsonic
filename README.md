

<img src="https://github.com/user-attachments/assets/cb6539cc-cea2-4a1c-8c26-762868828ac9" >
<br>
<br>
<a name="readme-top"></a>

<div align="center">


</div>


  <p>
    <a href="https://discord.gg/dNKGm4dfnR">
    <img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white">
    </a>
    <a href="https://twitter.com/upsonicai">
    <img src="https://img.shields.io/twitter/follow/upsonicai?style=social">
    </a>
    <a href="https://trendshift.io/repositories/10584" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10584" alt="unclecode%2Fcrawl4ai | Trendshift" style="width: 100px; height: 20px;"     
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg" alt="Made_with_python">
    </a>
  </p>


# Introduction
Upsonic is an AI agent development framework used by fintech leaders and tested at their scale against attacks and reasoning puzzles.


```bash
pip install upsonic

```

```python
from upsonic import Task, Agent

task = Task("Who developed you?")

agent = Agent(name="Coder")

agent.print_do(task)
```

<br>
<br>

# Guides | 5 Step
See our guides to jumpstart your AI agent within minutes. We design them to onboard the new users to the framework.


1. [Create an Agent](https://docs.upsonic.ai/guides/1_create_a_task)
2. [Create a Task](https://docs.upsonic.ai/guides/2_create_an_agent)
3. [Add a Safety Engine](https://docs.upsonic.ai/guides/3_add_a_safety_engine)
4. [Add a Tool](https://docs.upsonic.ai/guides/4_add_a_tool)
5. [Add an MCP](https://docs.upsonic.ai/guides/5_add_an_mcp)
6. [Integrate a Memory](https://docs.upsonic.ai/guides/6_integrate_a_memory)
7. [Creating a Team of Agents](https://docs.upsonic.ai/guides/7_creating_a_team_of_agents)

<br>

# Why Upsonic?

Upsonic provides a feature set to build safety-first, high-performance AI Agents. It helps you go to production without spending hours on research and boilerplate. These are the main parts:

- **Safety First**: Upsonic provides its own **Safety Engine** that manages User and Agent messages and checks their status for your policies. You can customize it by designing new **rule** and **action** sets.
- **Direct LLM Calls**: In Upsonic we support the same interface for your whole AI operations. You don't need to go with another framework to complete your **small jobs**.
- **Structured Outputs**: Upsonic sets agent outputs to make them **Python objects**. So you can integrate your application without struggling with **LLM outputs**.
- **Built-in RAG and Memory**: In Upsonic you can create world class . We support the Agentic RAG, Memory Logics and providers of them.
- **Customizable Memory Logics**: You are able to create **memories** that focus on **user**, **event** and **chat**. Also you are free to use **Local** and **Cloud databases**.
- **Agent Teams**: Upsonic provides the most **reliable** agent team architecture with memory, context management and leader agent.
- **FastAPI Compatible Agents**: You can turn your agents into production-ready APIs
- **Tracking the Executions**: You can use <u>Upsonic AgentOS</u> to get the execution history, monthly costs andresponse times  of your agents.
- **Deploy at scale**: Upsonic agents work in the greatest and fastest-growing fintech companies and scaling is available on <u>Upsonic AgentOS</u>.



# 📙 Documentation

You can access our documentation at [docs.upsonic.ai](https://docs.upsonic.ai/) All concepts and examples are available there.

<br>






## Telemetry

We use anonymous telemetry to collect usage data. We do this to focus our developments on more accurate points. You can disable it by setting the UPSONIC_TELEMETRY environment variable to false.

```python
import os
os.environ["UPSONIC_TELEMETRY"] = "False"
```
<br>
<br>



