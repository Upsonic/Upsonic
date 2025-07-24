# Investment Report Generator – Upsonic AI Agent Framework Example

> **Status**: Example / Educational &nbsp;&nbsp;|&nbsp;&nbsp;Python ⟶ 3.9+  |   Fast prototyping with [Upsonic](https://github.com/upsonic-ai/upsonic)

---

## 📖 Overview

This repository showcases how to build a multi-stage **AI investment research workflow** with
[**Upsonic**](https://github.com/upsonic-ai/upsonic) – an open-source framework for orchestrating
LLM-powered **Agents**, **Tasks** and **Graphs**.

The example answers a common buy-side use-case:
1. **Stock Analysis** – Pull real-time fundamentals & qualitative insights.
2. **Comparative Research** – Score and rank securities on a risk-adjusted basis.
3. **Portfolio Construction** – Allocate capital across the universe given risk limits.

Two progressively richer scripts are provided:

| Script | Style | Key Concepts | Output Folder |
| ------ | ----- | ------------ | ------------- |
| `basic.py` | 🥾 *Bootstrapped* | single-file, sequential `Agent.print_do()` calls | `output_basic/` |
| `advanced.py` | 🚀 *Graph driven* | explicit `Graph` DSL, CLI flags, cleaner separation of concerns | `output_advanced/` |

---

## 📂 Directory layout

``` markdown
.
├── basic.py               # minimal working example
├── advanced.py            # production-style example with Graph & CLI
├── output_basic/          # auto-generated reports from basic run
├── output_advanced/       # auto-generated reports from advanced run
└── venv/                  # (optional) virtual-env
```

---

## 🔧 Installation

1. **Clone & enter** the project directory:

   ```bash
   git clone https://github.com/Upsonic/Upsonic.git
   cd Upsonic/src/examples/investment_report_generator
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

   *(Or)*

   ```bash
   virtualenv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -U upsonic yfinance
   ```

   *(Or)*

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your OpenAI key** (required by Upsonic):

   ```bash
   # macOS / Linux
   export OPENAI_API_KEY=sk-••••••••••••••••••

   # Windows
   set OPENAI_API_KEY=sk-••••••••••••••••••
   ```

---

## 🚀 Quick-start

### 1. Basic mode

Runs with a hard-coded universe (`AAPL`, `MSFT`, `GOOG`, `AMZN`, `TSLA`) and $1 billion capital.

```bash
python basic.py
# ▶ progress logs …
# ✅ Reports written to: ./output_basic
```

Generated reports:

- `1_stock_analyst_report.md`
- `2_research_analyst_report.md`
- `3_investment_report.md`

### 2. Advanced mode (CLI-driven)

```bash
python advanced.py "AAPL,MSFT,NVDA" --capital 5_000_000
# ▶ verbose agent-graph execution …
# ✅ Reports written to: ./output_advanced
```

Command-line flags:

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `symbols` | `str` | *required* | Comma-separated tickers (no spaces) |
| `--capital` | `float` | `1_000_000` | Notional USD to allocate in Stage 3 |

---

## 🛠️ Under the hood
### Upsonic primitives used
* **Agent** – wraps an LLM (GPT-4o here) with a role / objective.
* **Task** – natural-language instruction with optional tool access & context.
* **Graph** – declarative DAG that wires tasks together (advanced example).

### Data flow

``` markdown
 ┌─────────────────────────────┐
 │ 1️⃣  User / CLI Invocation   │
 └──────────────┬──────────────┘
                │ "basic.py" OR "advanced.py"
                ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                UNIVERSAL PRE-FLIGHT (both scripts)                         │
 │  • Load `OPENAI_API_KEY`                                                   │
 │  • Import yfinance, Upsonic                                                │
 │  • Define `yfinance_metrics_tool()` / `fetch_key_metrics()`                │
 └───────────────────┬────────────────────────────────────────────────────────┘
                     │
    ┌────────────────┴────────────────┐
    │                                 │
    │                                 │
╔══════════════════╗        ╔════════════════════════╗
║     BASIC.PY     ║        ║     ADVANCED.PY        ║
║  (Bootstrap)     ║        ║  (Graph-Driven)        ║
╚══════════════════╝        ╚════════════════════════╝
    │                                 │
    ▼                                 ▼
┌────────────┐                 ┌─────────────────┐
│Hard-coded  │                 │Parse CLI args   │
│universe +  │                 │`symbols`,       │
│capital     │                 │`--capital`      │
└─────┬──────┘                 └────────┬────────┘
      │                                 │
      ▼                                 ▼
┌──────────────┐              ┌──────────────────────┐
│Task 1 object │              │Task 1 via builder    │
│(StockAnalysis│              │`build_stock_task()`  │
└─────┬────────┘              └────────┬─────────────┘
      │ Agent: Senior Investment       │ Agent: Senior Investment
      │ Analyst (`print_do`)           │ Analyst  (Graph node)
      ▼                                ▼
┌────────────────┐           ┌────────────────────────┐
│Markdown report │           │Graph stores Stage-1    │
│1_stock_analyst │           │output                  │
└─────┬──────────┘           └────────┬───────────────┘
      │                               │
      ▼                               ▼
┌──────────────┐              ┌────────────────────────┐
│Task 2 object │              │Task 2 via builder      │
│(Research)    │              │`build_research_task()` │
└─────┬────────┘              └───────┬────────────────┘
      │ Agent: Senior Research        │ Agent: Senior Research
      │ Analyst (`print_do`)          │ Analyst  (Graph node)
      ▼                               ▼
┌────────────────┐           ┌─────────────────────────┐
│Markdown report │           │Graph stores Stage-2     │
│2_research_anal │           │output                   │
└─────┬──────────┘           └────────┬────────────────┘
      │                               │
      ▼                               ▼
┌──────────────┐              ┌────────────────────────┐
│Task 3 object │              │Task 3 via builder      │
│(Portfolio)   │              │`build_portfolio_task()`│
└─────┬────────┘              └───────┬────────────────┘
      │ Agent: Senior Investment      │ Agent: Senior Investment
      │ Lead (`print_do`)             │ Lead  (Graph node)
      ▼                               ▼
┌────────────────┐           ┌─────────────────────────┐
│Markdown report │           │Graph `.get_output()`    │
│3_investment.md │           │returns Stage-3 output   │
└─────┬──────────┘           └────────┬────────────────┘
      │                               │
      ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Reports written to disk (per mode)                      │
│  • `output_basic/` or `output_advanced/`                                    │
│  • Filenames: 1_stock_analyst_report.md, 2_research_analyst_report.md,      │
│    3_investment_report.md                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Domain-specific tool

`fetch_key_metrics()` wraps **yfinance** to provide headline valuation & quality metrics that
the LLM can call via Upsonic's built-in tool calling interface.

---

## 📝 Extending the example

1. **Add alternative data** – sentiment scores, ESG metrics, etc. via additional tools.
2. **Parallel execution** – branch graphs for industry specialists.
3. **Persist state** – plug in a database backend for long-running research processes.
4. **Add a new task** – add a new task to the graph.
5. **Create your own tool** – create your own tool to be used in the Tasks like markdown to pdf converter.

---

## 📜 License

This example code is released under the MIT License – see `LICENSE` for details.
