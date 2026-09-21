<div align="center">

<img width="947" alt="Upsonic_README" src="https://github.com/user-attachments/assets/acb3f413-e4fe-44a6-9aff-40d4e9031188" />

# Upsonic

**Build Autonomous AI Agents in Python**

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

Upsonic is a Python framework for building autonomous agents like OpenClaw and Claude Cowork, as well as more traditional agent systems.

## Quick Start

### Installation

```bash
uv pip install upsonic
# pip install upsonic
```

### IDE Integration

Add Upsonic docs as a source in your coding tools:

**Cursor:** Settings → Indexing & Docs → Add `https://docs.upsonic.ai/llms-full.txt`

Also works with VSCode, Windsurf, and similar tools.

---

## Create Autonomous Agent

### Build Your Own

```python
from upsonic import AutonomousAgent, Task

agent = AutonomousAgent(
    model="anthropic/claude-sonnet-4-5",
    workspace="/path/to/logs"
)

task = Task("Analyze server logs and detect anomaly patterns")

agent.print_do(task)
```

All file and shell operations are restricted to `workspace`. Path traversal and dangerous commands are blocked.

### Use Our Prebuilt Ones

Prebuilt autonomous agents are ready-to-run agents built by the Upsonic community, each packaging a skill, system prompt, and first message so you can go from install to running in seconds. The collection is [open to contributions](https://github.com/Upsonic/Upsonic/tree/master/src/upsonic/prebuilt), bring your agent and open a PR.

Learn more: [Prebuilt Autonomous Agents](https://docs.upsonic.ai/concepts/prebuilt-autonomous-agents/overview)

> **Next steps:** Connect a [Sandbox Provider (E2B)](https://docs.upsonic.ai/concepts/autonomous-agent/overview) for isolated cloud execution environments.

---

## Create Traditional Agent

```python
from upsonic import Agent, Task

agent = Agent(model="anthropic/claude-sonnet-4-5", name="Stock Analyst Agent")

task = Task(description="Analyze the current market trends")

agent.print_do(task)
```

### Add Custom Tools

```python
from upsonic import Agent, Task
from upsonic.tools import tool

@tool
def sum_tool(a: float, b: float) -> float:
    """
    Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b

task = Task(
    description="Calculate 15 + 27",
    tools=[sum_tool]
)

agent = Agent(model="anthropic/claude-sonnet-4-5", name="Calculator Agent")

result = agent.print_do(task)
```

### Choose a web search backend

`WebSearch` uses DuckDuckGo by default. To select Parallel's anonymous Search
MCP instead, install the MCP extra and pass `provider="parallel"`:

```bash
pip install "upsonic[mcp]"
```

```python
from upsonic.tools import WebSearch, aWebSearch

results = WebSearch("Upsonic framework documentation", provider="parallel")
# In an async application:
# results = await aWebSearch("Upsonic framework documentation", provider="parallel")
```

Both return titles, source URLs and excerpts. `max_results` limits the sources
returned to your caller locally; it doesn't set a server search limit. This
option uses `https://search.parallel.ai/mcp` over Streamable HTTP and needs no
Parallel account or API key. Anonymous access is rate limited. It doesn't replace
provider-side `WebSearchTool` or change other search integrations.

Selecting Parallel sends the supplied query (also used as the search objective)
to Parallel. If you register `WebSearch` as an agent tool, the agent can invoke
it with that provider. Requests include `upsonic/<package version>` in the
User-Agent to measure aggregate Upsonic usage; this identifier contains no user
or installation ID. This adapter uses the anonymous path and doesn't read
Parallel API keys from the environment. See the [Search MCP documentation](https://docs.parallel.ai/integrations/mcp/search-mcp)
for service details.

> **Next steps:** Integrate [MCP Tools](https://docs.upsonic.ai/concepts/tools/mcp-tools/overview) to connect your agents to thousands of external data sources and services.

---

## OCR and Document Processing

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

---

## Check Our Videos

<table>
  <tr>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=GOYko0KfBtg">
        <img src="https://img.youtube.com/vi/GOYko0KfBtg/maxresdefault.jpg" width="400" alt="Upsonic Demo Video 1"/>
      </a>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=ulUEFIolesQ">
        <img src="https://img.youtube.com/vi/ulUEFIolesQ/maxresdefault.jpg" width="400" alt="Upsonic Demo Video 2"/>
      </a>
    </td>
  </tr>
</table>

---

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

We welcome contributions from the community! Please read our [Contributing Guide](CONTRIBUTING.md) and code of conduct before submitting pull requests.