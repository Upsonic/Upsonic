# investment_report_generator_upsonic.py
from pathlib import Path
from textwrap import dedent
from typing import List, Dict
from shutil import rmtree

import yfinance as yf
from upsonic import Agent, Task, Graph


# ──────────────────────────────────────────
# 1. ─── Domain-specific TOOL DEFINITIONS ──
# ──────────────────────────────────────────
def fetch_key_metrics(ticker: str) -> Dict[str, str]:
    """
    Pull headline valuation and quality metrics via yfinance.

    Args:
        ticker: e.g. 'AAPL'

    Returns:
        Dictionary with price, market-cap, PE, EPS, revenue, net-income
    """
    info = yf.Ticker(ticker).info
    return {
        "price": info.get("currentPrice", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe": info.get("trailingPE", "N/A"),
        "eps": info.get("trailingEps", "N/A"),
        "revenue": info.get("totalRevenue", "N/A"),
        "net_income": info.get("netIncomeToCommon", "N/A")
    }


# Upsonic accepts plain Python callables as tools; the type hints are enough
YFMetricsTool = fetch_key_metrics     # alias for readability

# ──────────────────────────────────────────
# 2. ─── AGENT DEFINITION (roles & rigor) ─
# ──────────────────────────────────────────
stock_analyst = Agent(
    name="Senior Investment Analyst",
    company_objective="Deliver exhaustive bottom-up security research",
    model="openai/gpt-4o",
)

research_analyst = Agent(
    name="Senior Research Analyst",
    company_objective="Rank securities by risk-adjusted potential",
    model="openai/gpt-4o"
)

investment_lead = Agent(
    name="Senior Investment Lead",
    company_objective="Translate ideas into balanced portfolio allocations",
    model="openai/gpt-4o"
)

# ──────────────────────────────────────────
# 3. ─── TASK BLUEPRINTS (markdown output) ─
# ──────────────────────────────────────────


def build_stock_task(symbols: List[str]) -> Task:
    return Task(
        dedent(f"""\
        ### Stage 1 – Comprehensive Stock Analysis
        Analyse fundamentals, relative valuation, recent news flow and macro drivers
        for the following tickers: {', '.join(symbols)}.

        **Deliverable**: Markdown report with sections:
        1. Company snapshot
        2. Valuation multiples (with peer comparison)
        3. Growth catalysts
        4. Risk factors
        5. News sentiment summary
        """),
        tools=[YFMetricsTool],
        response_lang="en"
    )


def build_research_task(previous: Task) -> Task:
    return Task(
        dedent("""\
        ### Stage 2 – Investment Potential & Ranking
        Using the supplied Stage 1 research, rank each company from strongest to
        weakest on a risk-adjusted basis. For each name provide:

        - Numerical score (0-100)
        - Qualitative rationale (≤80 words)
        - Key upside catalyst
        - Principal downside risk
        """),
        context=[previous]
    )


def build_portfolio_task(previous: Task, capital: float) -> Task:
    return Task(
        dedent(f"""\
        ### Stage 3 – Strategic Portfolio Construction
        Allocate a hypothetical **${capital:,.0f}** across the ranked securities.
        Constraints:
        - Max single-name weight 20 %
        - Sector weight ≤40 %
        - Target volatility ≤12 % (use qualitative proxy)

        Output a markdown table with columns:
        | Rank | Ticker | Allocation $ | Weight % | Rationale | Expected 12-m return |
        """),
        context=[previous]
    )

# ──────────────────────────────────────────
# 4. ─── GRAPH ASSEMBLY (sequential flow) ─
# ──────────────────────────────────────────


def build_graph(tickers: List[str], capital: float = 1_000_000) -> Graph:
    # explicit agents for each task
    g = Graph()

    t1 = build_stock_task(tickers)
    t1.agent = stock_analyst            # assign specialised agent

    t2 = build_research_task(t1)
    t2.agent = research_analyst

    t3 = build_portfolio_task(t2, capital)
    t3.agent = investment_lead

    # chain tasks: Stage 1 → Stage 2 → Stage 3
    g.add(t1 >> t2 >> t3)
    return g


# ──────────────────────────────────────────
# 5. ─── CLI ENTRY POINT ─
# ──────────────────────────────────────────
if __name__ == "__main__":
    import os
    import argparse

    # ensure API key is exported:  export OPENAI_API_KEY=sk-…
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY first."

    parser = argparse.ArgumentParser(
        description="Upsonic Investment Report Generator - Advanced"
    )
    parser.add_argument(
        "symbols",
        help="Comma-separated tickers, e.g. AAPL,MSFT,GOOGL"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000,
        help="Portfolio size in USD (default 1,000,000)"
    )
    args = parser.parse_args()

    tickers = [s.strip().upper() for s in args.symbols.split(",")]
    graph = build_graph(tickers, capital=args.capital)

    # run the full workflow
    graph.run(verbose=True)

    # persist each stage to disk
    out_dir = Path("output_advanced")
    if out_dir.exists():
        rmtree(path=out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "1_stock_analyst_report.md").write_text(
        graph.state.get_task_output(graph.nodes[0].id)
    )
    (out_dir / "2_research_analyst_report.md").write_text(
        graph.state.get_task_output(graph.nodes[1].id)
    )
    (out_dir / "3_investment_report.md").write_text(
        graph.get_output()
    )

    print(f"\n✅  Reports written to: {out_dir.resolve()}")
