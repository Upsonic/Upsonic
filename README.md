

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

# Guides | 7 Step
See our guides to jumpstart your AI agent within minutes. We design them to onboard the new users to the framework.


1. [Create an Agent](https://docs.upsonic.ai/guides/1-create-a-task)
2. [Create a Task](https://docs.upsonic.ai/guides/2-create-an-agent)
3. [Add a Safety Engine](https://docs.upsonic.ai/guides/3-add-a-safety-engine)
4. [Add a Tool](https://docs.upsonic.ai/guides/4-add-a-tool)
5. [Add an MCP](https://docs.upsonic.ai/guides/5-add-an-mcp)
6. [Integrate a Memory](https://docs.upsonic.ai/guides/6-integrate-a-memory)
7. [Creating a Team of Agents](https://docs.upsonic.ai/guides/7-creating-a-team-of-agents)

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






## 📊 Telemetry & Privacy

Upsonic uses **anonymous telemetry** to help us understand how the framework is being used and improve our development focus. We are committed to transparency and user privacy.

All telemetry is **anonymous** - we only track a randomly generated system ID to distinguish unique installations.

### Why Collect Telemetry?

Telemetry helps us:
- 🎯 Focus development on frequently-used features
- 🐛 Identify and fix common errors and edge cases
- 📈 Understand performance characteristics at scale
- 🔧 Improve framework reliability

### How to Disable Telemetry

You can **completely disable** telemetry in multiple ways:

**Option 1: Environment Variable (Recommended)**
```bash
export UPSONIC_TELEMETRY=false
```

**Option 2: In Python Code**
```python
import os
os.environ["UPSONIC_TELEMETRY"] = "false"

from upsonic import Agent  # Import after setting env var
```

**Option 3: .env File**
```bash
# .env
UPSONIC_TELEMETRY=false
```

Once disabled, **no data** will be sent to our telemetry service.


<br>
<br>



