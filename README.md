<div align="center">

<img src="https://github.com/user-attachments/assets/fbe7219f-55bc-4748-ac4a-dd2fb2b8d9e5" width="600" />

# Upsonic

**Production-Ready AI Agent Framework with Safety First**

[![PyPI version](https://badge.fury.io/py/upsonic.svg)](https://badge.fury.io/py/upsonic)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENCE)
[![Python Version](https://img.shields.io/pypi/pyversions/upsonic.svg)](https://pypi.org/project/upsonic/)
[![GitHub stars](https://img.shields.io/github/stars/Upsonic/Upsonic.svg?style=social&label=Star)](https://github.com/Upsonic/Upsonic)
[![GitHub issues](https://img.shields.io/github/issues/Upsonic/Upsonic.svg)](https://github.com/Upsonic/Upsonic/issues)
[![Documentation](https://img.shields.io/badge/docs-upsonic.ai-brightgreen.svg)](https://docs.upsonic.ai)

[Documentation](https://docs.upsonic.ai) • [Quickstart](https://docs.upsonic.ai/get-started/quickstart) • [Examples](https://docs.upsonic.ai/examples)

</div>

---

## Overview

Upsonic is an open-source AI agent development framework that makes building production-ready agents simple, safe, and scalable. Whether you're building your first agent or orchestrating complex multi-agent systems, Upsonic provides everything you need in one unified framework.

Built by the community, for the community. We listen to what you need and prioritize features based on real-world use cases. Currently, we're focused on **Safety Engine** and **OCR capabilities**, two critical features for production workloads.

## What Can You Build?

Upsonic is used by fintech companies, banks, and developers worldwide to build production-grade AI agents for:

- **Document Analysis**: Extract, process, and understand documents with advanced OCR and NLP
- **Customer Service Automation**: Build intelligent chatbots with memory and context awareness
- **Financial Analysis**: Create agents that analyze market data, generate reports, and provide insights
- **Compliance Monitoring**: Ensure all AI operations follow safety policies and regulatory requirements
- **Research & Data Gathering**: Automate research workflows with multi-agent collaboration
- **Multi-Agent Workflows**: Orchestrate complex tasks across specialized agent teams

## Quick Start

### Installation

Install Upsonic using uv:

```bash
uv pip install upsonic
# pip install upsonic
```

### Basic Agent

Create your first agent in just a few lines of code:

```python
from upsonic import Agent, Task

agent = Agent(model="openai/gpt-4o", name="Stock Analyst Agent")

task = Task(description="Analyze the current market trends")

agent.print_do(task)
```

### Agent with Tools

Enhance your agent with tools for real-world tasks:

```python
from upsonic import Agent, Task
from upsonic.tools.common_tools import YFinanceTools

agent = Agent(model="openai/gpt-4o", name="Stock Analyst Agent")

task = Task(
    description="Give me a summary about tesla stock with tesla car models",
    tools=[YFinanceTools()]
)

agent.print_do(task)
```

### Agent with Memory

Add memory to make your agent remember past conversations:

```python
from upsonic import Agent, Task
from upsonic.storage import Memory, InMemoryStorage

memory = Memory(
    storage=InMemoryStorage(),
    session_id="session_001",
    full_session_memory=True
)

agent = Agent(model="openai/gpt-4o", memory=memory)

task1 = Task(description="My name is John")
agent.print_do(task1)

task2 = Task(description="What is my name?")
agent.print_do(task2)  # Agent remembers: "Your name is John"
```

**Ready for more?** Check out the [Quickstart Guide](https://docs.upsonic.ai/get-started/quickstart) for additional examples including Knowledge Base and Team workflows.

## Key Features

- **Safety Engine**: Built-in policy engine to ensure your agents follow company guidelines and compliance requirements
- **OCR Support**: Unified interface for local and cloud OCR providers with document processing capabilities
- **Memory Management**: Give your agents context and long-term memory with flexible storage backends
- **Multi-Agent Teams**: Build collaborative agent systems with sequential and parallel execution modes
- **Tool Integration**: Extensive tool support including MCP, custom tools, and human-in-the-loop workflows
- **Production Ready**: Designed for enterprise deployment with comprehensive monitoring and metrics

## Core Capabilities

### Safety Engine

Safety isn't an afterthought in Upsonic. It's built into the core. Create reusable policies, attach them to any agent, and ensure compliance across your entire system. The Safety Engine is LLM-agnostic and production-ready from day one.

Key capabilities include:
- Pre-built policies for common safety requirements (PII blocking, content filtering, etc.)
- Custom policy creation for your specific compliance needs
- Real-time monitoring and enforcement
- Detailed audit logs for compliance reporting

**Example:**

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

Learn more: [Safety Engine Documentation](https://docs.upsonic.ai/concepts/safety-engine/overview)

### OCR and Document Processing

Upsonic provides a unified interface for working with multiple OCR providers, both local and cloud-based. This eliminates the complexity of integrating different OCR services and allows you to switch providers without changing your code.

Supported providers include:
- Cloud providers (Google Vision, AWS Textract, Azure Computer Vision)
- Local providers (Tesseract, EasyOCR, PaddleOCR)
- Specialized document processors (DocTR, Surya)

Learn more: [OCR Documentation](https://docs.upsonic.ai/concepts/ocr/overview)

## Upsonic AgentOS

AgentOS is an optional deployment and management platform that takes your agents from development to production. It provides enterprise-grade infrastructure for deploying, monitoring, and scaling your AI agents.

**Key Features:**

- **Kubernetes-based FastAPI Runtime**: Deploy your agents as isolated, scalable microservices with enterprise-grade reliability
- **Comprehensive Metrics Dashboard**: Track every agent transaction, LLM costs, token usage, and performance metrics for complete visibility
- **Self-Hosted Deployment**: Deploy the entire AgentOS platform on your own infrastructure with full control over your data and operations
- **One-Click Deployment**: Go from code to production with automated deployment pipelines

<img width="3024" height="1590" alt="AgentOS Dashboard" src="https://github.com/user-attachments/assets/42fceaca-2dec-4496-ab67-4b9067caca42" />

## Your Complete AI Agent Infrastructure

Together, the Upsonic Framework and AgentOS provide everything a financial institution needs to build, deploy, and manage production-grade AI agents. From development to deployment, from local testing to enterprise-scale operations, from single agents to complex multi-agent systems, Upsonic delivers the complete infrastructure for your AI agent initiatives.

Whether you're a fintech startup building your first intelligent automation or an established bank deploying agents across multiple business units, Upsonic provides the end-to-end tooling to bring your AI agent vision to life safely, efficiently, and at scale.

## Documentation and Resources

- **[Documentation](https://docs.upsonic.ai)** - Complete guides and API reference
- **[Quickstart Guide](https://docs.upsonic.ai/get-started/quickstart)** - Get started in 5 minutes
- **[Examples](https://docs.upsonic.ai/examples)** - Real-world examples and use cases
- **[API Reference](https://docs.upsonic.ai/reference)** - Detailed API documentation

## Community and Support

- **[Issue Tracker](https://github.com/Upsonic/Upsonic/issues)** - Report bugs and request features
- **[Changelog](https://docs.upsonic.ai/changelog)** - See what's new in each release

## License

Upsonic is released under the MIT License. See [LICENCE](LICENCE) for details.

## Contributing

We welcome contributions from the community! Please read our contributing guidelines and code of conduct before submitting pull requests.

---

**Learn more at [upsonic.ai](https://upsonic.ai)**
