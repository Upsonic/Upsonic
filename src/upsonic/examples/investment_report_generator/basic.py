#  Built-in tools import
import os
from typing import Dict
from pathlib import Path
from shutil import rmtree
import logging

#  Importing the Agent and Task classes from the upsonic package
from upsonic import Agent, Task

#  Importing Third Party Libraries
import yfinance as yf
from textwrap import dedent

logging.basicConfig(
    filename="basic.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logging.info("Upsonic Investment Report Generator - Basic")

# ensure API key is exported:  export OPENAI_API_KEY=sk-…
assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY first."

#  Set the project root
PROJECT_ROOT = Path(__file__).parent

#  Set the output directory
OUTPUT_DIR = PROJECT_ROOT / "output_basic"

#  Clear the output directory if it exists and create a new one
if OUTPUT_DIR.exists():
    rmtree(path=OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_STOCK_ANALYSIS = OUTPUT_DIR.joinpath("1_stock_analyst_report.md")
REPORT_RESEARCH_ANALYSIS = OUTPUT_DIR.joinpath("2_research_analyst_report.md")
REPORT_INVESTMENT = OUTPUT_DIR.joinpath("3_investment_report.md")


#  Set Global Variables
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
TICKERS_STR = ", ".join(TICKERS)
CAPITAL = 1000000000

logging.info(f"Tickers: {TICKERS_STR}")
logging.info(f"Capital: ${CAPITAL:,.0f}")


#  Function to fetch key metrics from yfinance
def yfinance_metrics_tool(ticker: str) -> Dict[str, str]:
    """
    Pull headline valuation and quality metrics via yfinance.

    Args:
        ticker: e.g. 'AAPL'

    Returns:
        Dictionary with price, market-cap, PE, EPS, revenue, net-income
    """
    logging.info(f"Fetching metrics for {ticker}...")
    info = yf.Ticker(ticker).info
    return {
        "ticker": ticker,
        "price": info.get("currentPrice", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe": info.get("trailingPE", "N/A"),
        "eps": info.get("trailingEps", "N/A"),
        "revenue": info.get("totalRevenue", "N/A"),
        "net_income": info.get("netIncomeToCommon", "N/A"),
    }


#  Define the tasks
task_1_stock_analysis = Task(
    description=dedent(
        f"""
        # ROLE
        You are an elite Senior Investment Analyst with expertise in:
        - Comprehensive market analysis
        - Financial statement evaluation
        - Industry trend identification
        - News impact assessment
        - Risk factor analysis
        - Growth potential evaluation

        # TASK 1 - Stock Analysis
        Do an in-depth stock analysis about the following tickers: **{TICKERS_STR}**
        using given **yfinance_metrics_tool(ticker: str) -> Dict[str, str]** tool.

        **Deliverable**: Markdown report with sections:
        - **Snapshot Table** of key metrics (market-cap, EV/EBITDA, P/E, dividend yield, 1-yr price change).
        - **Narrative Analysis** covering business model, latest earnings surprises, and management guidance.
        - **News Pulse** – three most material headlines (≤ 45 words each) plus a one-sentence relevance note.
        - **Initial Analyst Sentiment** – consensus rating, target-price dispersion, and short-interest trend.
        - **Recommendations**: 1-2 sentences on why the stock is a good or bad investment.
        - **Conclusion**: 1-2 sentences on why the stock is a good or bad investment.
        """
    ),
    tools=[yfinance_metrics_tool],
    # tools=[Search, yfinance_metrics_tool],
    response_format=str,
    response_lang="en",
    context=None,
)

task_2_research_analysis = Task(
    description=dedent(
        """
        # ROLE
        You are an elite Senior Research Analyst specializing in:
        - Investment opportunity evaluation
        - Comparative analysis
        - Risk-reward assessment
        - Growth potential ranking
        - Strategic recommendations

        # TASK 2 - Investment Potential & Ranking
        Using the **TASK 1** research, rank each company from strongest to weakest on a risk-adjusted basis.

        **Deliverable**: Markdown report with sections for every name provide:

        - **Scoring Grid** columns: Valuation, Growth, Profitability, Balance-Sheet, Moat, Macro-Sensitivity (1-5 scale).
        - **Weighted Score** (weights: 25 % Valuation, 20 % Growth, 15 % Profitability, 15 % Balance-Sheet, 15 % Moat, 10 % Macro).
        - **Ranked List** from highest to lowest composite score, each with a two-sentence rationale.
        - **Key Risks** at least one idiosyncratic and one sector-level risk per company.
        """
    ),
    tools=[],
    response_format=str,
    response_lang="en",
    context=[task_1_stock_analysis],
)

task_3_investment = Task(
    description=dedent(
        """
        # ROLE
        You are a distinguished Senior Investment Lead with expertise in:
        - Portfolio strategy development
        - Asset allocation optimization
        - Investment rationale articulation
        - Client recommendation delivery

        # TASK 3 - Strategic Portfolio Construction
        Using the **TASK 2** research, allocate a hypothetical ${CAPITAL:,.0f} across the ranked securities.

        **Deliverable**: Markdown report with sections:
        - **Target Weights** (sum = 100 %) by ticker in a clean table.
        - **Position Thesis** a short paragraph per position explaining weight choice, catalyst, and time-horizon.
        - **Risk Controls** stop-loss levels, correlation considerations, and position-sizing logic.
        - **Action Checklist** bullet list of next steps (data to monitor, entry timing, review cadence).
        """
    ),
    tools=[],
    response_format=str,
    response_lang="en",
    context=[task_2_research_analysis],
)

#  Define the agents
agent_1_stock_analyst = Agent(
    name="Senior Investment Analyst",
    model="openai/gpt-4o",
    debug=False,
    memory=False,
    agent_id_="StockAnalyst-1",
    canvas=None,
)

agent_2_research_analyst = Agent(
    name="Senior Research Analyst",
    model="openai/gpt-4o",
    debug=False,
    memory=False,
    agent_id_="ResearchAnalyst-1",
    canvas=None,
)

agent_3_investment_lead = Agent(
    name="Senior Investment Lead",
    model="openai/gpt-4o",
    debug=False,
    memory=False,
    agent_id_="InvestmentLead-1",
    canvas=None,
)

#  Run the agents
agent_1_stock_analysis_response = agent_1_stock_analyst.print_do(task_1_stock_analysis)
REPORT_STOCK_ANALYSIS.write_text(agent_1_stock_analysis_response)

agent_2_research_analysis_response = agent_2_research_analyst.print_do(task_2_research_analysis)
REPORT_RESEARCH_ANALYSIS.write_text(agent_2_research_analysis_response)

agent_3_investment_response = agent_3_investment_lead.print_do(task_3_investment)
REPORT_INVESTMENT.write_text(agent_3_investment_response)
