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
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/pmYDMSQHqY)

[Documentation](https://docs.upsonic.ai) • [Quickstart](https://docs.upsonic.ai/get-started/quickstart) • [Examples](https://docs.upsonic.ai/examples) • [Discord](https://discord.gg/pmYDMSQHqY)

</div>

---

## Overview

Upsonic is an open-source AI agent framework for building production-ready agents. It supports multiple AI providers (OpenAI, Anthropic, Azure, Bedrock) and includes built-in safety policies, OCR, memory, multi-agent coordination, and MCP tool integration.

## What Can You Build?

- **Document Analysis**: Extract and process text from images and PDFs
- **Customer Service Automation**: Agents with memory and session context
- **Financial Analysis**: Agents that analyze data, generate reports, and provide insights
- **Compliance Monitoring**: Enforce safety policies across all agent interactions
- **Research & Data Gathering**: Automate research workflows with multi-agent collaboration
- **Multi-Agent Workflows**: Orchestrate tasks across specialized agent teams

## Quick Start

### Installation

```bash
uv pip install upsonic
# pip install upsonic
```

### Basic Agent

```python
from upsonic import Agent, Task

agent = Agent(model="anthropic/claude-sonnet-4-5", name="Stock Analyst Agent")

task = Task(description="Analyze the current market trends")

agent.print_do(task)
```

### Agent with Tools

```python
from upsonic import Agent, Task
from upsonic.tools.common_tools import YFinanceTools

agent = Agent(model="anthropic/claude-sonnet-4-5", name="Stock Analyst Agent")

task = Task(
    description="Give me a summary about tesla stock with tesla car models",
    tools=[YFinanceTools()]
)

agent.print_do(task)
```

### Agent with Memory

```python
from upsonic import Agent, Task
from upsonic.storage import Memory, InMemoryStorage

memory = Memory(
    storage=InMemoryStorage(),
    session_id="session_001",
    full_session_memory=True
)

agent = Agent(model="anthropic/claude-sonnet-4-5", memory=memory)

task1 = Task(description="My name is John")
agent.print_do(task1)

task2 = Task(description="What is my name?")
agent.print_do(task2)  # Agent remembers: "Your name is John"
```

**Ready for more?** Check out the [Quickstart Guide](https://docs.upsonic.ai/get-started/quickstart) for additional examples including Knowledge Base and Team workflows.

## Key Features

- **Autonomous Agent**: An agent that can read, write, and execute code inside a sandboxed workspace, no tool setup required
- **Safety Engine**: Policy-based content filtering applied to user inputs, agent outputs, and tool interactions
- **OCR Support**: Unified interface for multiple OCR engines with PDF and image support
- **Memory Management**: Session memory and long-term storage with multiple backend options
- **Multi-Agent Teams**: Sequential and parallel agent coordination
- **Tool Integration**: MCP tools, custom tools, and human-in-the-loop workflows
- **Production Ready**: Monitoring, metrics, and enterprise deployment support

## Core Capabilities

### Autonomous Agent

`AutonomousAgent` extends `Agent` with built-in filesystem and shell tools, automatic session memory, and workspace sandboxing. Useful for coding assistants, DevOps automation, and any task that needs direct file or terminal access.

```python
from upsonic import AutonomousAgent, Task

agent = AutonomousAgent(
    model="anthropic/claude-sonnet-4-5",
    workspace="/path/to/project"
)

task = Task("Read the main.py file and add error handling to every function")
agent.print_do(task)
```

All file and shell operations are restricted to `workspace`. Path traversal and dangerous commands are blocked.

---

### Safety Engine

The Safety Engine applies policies at three points: user inputs, agent outputs, and tool interactions. Policies can block, anonymize, replace, or raise exceptions on matched content.

```python
from upsonic import Agent, Task
from upsonic.safety_engine.policies.pii_policies import PIIAnonymizePolicy

agent = Agent(
    model="anthropic/claude-sonnet-4-5",
    user_policy=PIIAnonymizePolicy,  # anonymizes PII before sending to the LLM
)

task = Task(
    description="My email is john.doe@example.com and phone is 555-1234. What are my email and phone?"
)

# PII is anonymized before reaching the LLM, then de-anonymized in the response
result = agent.do(task)
print(result)  # "Your email is john.doe@example.com and phone is 555-1234"
```

Pre-built policies cover PII, adult content, profanity, financial data, and more. Custom policies are also supported.

Learn more: [Safety Engine Documentation](https://docs.upsonic.ai/concepts/safety-engine/overview)

---

### OCR and Document Processing

Upsonic provides a unified OCR interface with a layered pipeline: Layer 0 handles document preparation (PDF to image conversion, preprocessing), Layer 1 runs the OCR engine.

```bash
uv pip install "upsonic[ocr]"
```

```python
from upsonic.ocr import OCR
from upsonic.ocr.layer_1.engines import EasyOCREngine

engine = EasyOCREngine(languages=["en"])
ocr = OCR(layer_1_ocr_engine=engine)

text = ocr.get_text("invoice.pdf")
print(text)
```

Supported engines: EasyOCR, RapidOCR, Tesseract, PaddleOCR, DeepSeek OCR, DeepSeek via Ollama.

Learn more: [OCR Documentation](https://docs.upsonic.ai/concepts/ocr/overview)

## Upsonic AgentOS

AgentOS is an optional deployment platform for running agents in production. It provides a Kubernetes-based runtime, metrics dashboard, and self-hosted deployment.

- **Kubernetes-based FastAPI Runtime**: Deploy agents as isolated, scalable microservices
- **Metrics Dashboard**: Track LLM costs, token usage, and performance per transaction
- **Self-Hosted**: Full control over your data and infrastructure
- **One-Click Deployment**: Automated deployment pipelines

<img width="3024" height="1590" alt="AgentOS Dashboard" src="https://github.com/user-attachments/assets/42fceaca-2dec-4496-ab67-4b9067caca42" />

## Documentation and Resources

- **[Documentation](https://docs.upsonic.ai)** - Complete guides and API reference
- **[Quickstart Guide](https://docs.upsonic.ai/get-started/quickstart)** - Get started in 5 minutes
- **[Examples](https://docs.upsonic.ai/examples)** - Real-world examples and use cases
- **[API Reference](https://docs.upsonic.ai/reference)** - Detailed API documentation

## Community and Support

> **💬 [Join our Discord community!](https://discord.gg/pmYDMSQHqY)** — Ask questions, share what you're building, get help from the team, and connect with other developers using Upsonic.

- **[Discord](https://discord.gg/pmYDMSQHqY)** - Chat with the community and get real-time support
- **[Issue Tracker](https://github.com/Upsonic/Upsonic/issues)** - Report bugs and request features
- **[Changelog](https://docs.upsonic.ai/changelog)** - See what's new in each release

## License

Upsonic is released under the MIT License. See [LICENCE](LICENCE) for details.

## Contributing

We welcome contributions from the community! Please read our contributing guidelines and code of conduct before submitting pull requests.

---

**Learn more at [upsonic.ai](https://upsonic.ai)**
