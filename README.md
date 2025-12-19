
# What is Upsonic
Upsonic is the AI Agent Development Framework and AgentOS that used by the fintech and banks.


<img src="https://github.com/user-attachments/assets/fbe7219f-55bc-4748-ac4a-dd2fb2b8d9e5" />

<br/>
<br/>

## Upsonic Framework

You can use the Upsonic Framework to build safety-first AI Agents or teams with Memory, KnowledgeBase, OCR, Human in the Loop, tools and MCP Support. The Upsonic framework orchestrates all of the operations with its pipeline architecture.

You are able to create complex and basic agents in one unified system. Our development process is based on what our community wants. Currently we are doubling down on Safety Engine and OCR capabilities.

```bash
pip install upsonic

```

```python
from upsonic import Task, Agent

task = Task("Who developed you?")

agent = Agent(name="Coder", model="openai/gpt-5-mini")

agent.print_do(task)
```

[Docs](https://docs.upsonic.ai/get-started/introduction), [Guides](https://docs.upsonic.ai/guides/1-create-a-task)

<br/>
<br/>


## Why Upsonic?

At Upsonic, we don't just build features in isolation. We listen to our community and prioritize what matters most to you. Right now, that means doubling down on Safety and OCR capabilities: two areas our users have made clear are critical for production workloads.

And of course, we've got you covered on the fundamentals. Upsonic ships with all the core features you'd expect from a modern framework, so you're never trading off functionality for innovation.

TL;DR: We're focused on what you need (Safety + OCR), while delivering everything you expect.

### Safety Engine

**It's our most differentiating feature in the competition.** In the current development cycle of agents, the main problem is being sure about safety. There are lots of wrong ways and potential problems that go against your company policy. So we made a feature where you can create policies, put them on your agents, and track them. This way you'll see your safety policies enforced on your agents. And it's an LLM-agnostic feature, so you can use your policies on any agent once you create them.

```python
from upsonic import Agent, Task
from upsonic.safety_engine.policies.pii_policies import PIIBlockPolicy

agent = Agent(
    model="openai/gpt-4o-mini",
    agent_policy=PIIBlockPolicy,
)

task = Task(
    description="Create a realistic customer profile with name Alice, email alice@example.com, phone number 1234567890, and address 123 Main St, Anytown, USA"
)

result = agent.do(task)
print(result)
```

[Concept Docs](https://docs.upsonic.ai/concepts/safety-engine/overview)


## OCR

In our framework, we directly support many local and cloud OCR providers to speed up this process. This way, developers don't need to struggle with the OCR step anymore. You can directly use all OCRs from one unified interface.

[Concept Docs](https://docs.upsonic.ai/concepts/ocr/overview)

<br/>
<br/>

# Upsonic AgentOS

AgentOS is a deployment and management platform for your AI Agents. You can click on the buttons to deploy production-ready and stable agent projects. The most important points are:

- **K8s-based FastAPI runtime**: Upsonic AgentOS turns your agents into microservices by design. So you can integrate your agents into any of your systems easily, scalably, isolated and securely.
- **Metric Dashboard**: We have an integrated metric system. Every agent transaction and LLM costs are saved. So you have great visibility of your daily, monthly and yearly agent costs, tokens and other metrics.
- **Available for On-premise**: You can deploy the entire AgentOS platform on your local infrastructure.

<img width="3024" height="1590" alt="image" src="https://github.com/user-attachments/assets/42fceaca-2dec-4496-ab67-4b9067caca42" />



<br/>
<br/>



## Your Complete AI Agent Infrastructure

Together, the Upsonic Framework and AgentOS provide everything a financial institution needs to build, deploy, and manage production-grade AI agents. From development to deployment, from local testing to enterprise-scale operations, from single agents to complex multi-agent systems. Upsonic delivers the complete infrastructure for your AI agent initiatives.

Whether you're a fintech startup building your first intelligent automation or an established bank deploying agents across multiple business units, Upsonic provides the end-to-end tooling to bring your AI agent vision to life safely, efficiently, and at scale.

[Website](https://upsonic.ai/)
