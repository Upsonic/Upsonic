# Example of how to use the YFinanceTools class

from textwrap import dedent

import rich

from upsonic import Agent, Task
from upsonic.tools.finance.yfinance import YFinanceTools

yfinance_tools = YFinanceTools()
companies = "AAPL, MSFT"

stock_analyst = Agent(name="Stock Analyst")
research_analyst = Agent(name="Research Analyst")
investment_lead = Agent(name="Investment Lead")


def stock_analyst_task(companies: str):
    stock_analyst_task = Task(
        description=dedent(
            f"""\
            You are MarketMaster-X, an elite Senior Investment Analyst at Goldman Sachs with expertise in:
            - Comprehensive market analysis
            - Financial statement evaluation
            - Industry trend identification
            - News impact assessment
            - Risk factor analysis
            - Growth potential evaluation

            Instructions:
            1. Market Research 📊
            - Analyze company fundamentals and metrics
            - Review recent market performance
            - Evaluate competitive positioning
            - Assess industry trends and dynamics
            2. Financial Analysis 💹
            - Examine key financial ratios
            - Review analyst recommendations
            - Analyze recent news impact
            - Identify growth catalysts
            3. Risk Assessment 🎯
            - Evaluate market risks
            - Assess company-specific challenges
            - Consider macroeconomic factors
            - Identify potential red flags
            Note: This analysis is for educational purposes only.

            Analyze the following companies and produce a comprehensive market analysis report in markdown format: {companies}
        """
        ),
        tools=[yfinance_tools],
    )

    return stock_analyst.do(stock_analyst_task)


def researcher_task(stock_analyst_result: str):
    researcher_task = Task(
        description=dedent(
            f"""\
        You are ValuePro-X, an elite Senior Research Analyst at Goldman Sachs specializing in:
        - Investment opportunity evaluation
        - Comparative analysis
        - Risk-reward assessment
        - Growth potential ranking
        - Strategic recommendations

        Instructions:
        1. Investment Analysis 🔍
           - Evaluate each company's potential
           - Compare relative valuations
           - Assess competitive advantages
           - Consider market positioning
        2. Risk Evaluation 📈
           - Analyze risk factors
           - Consider market conditions
           - Evaluate growth sustainability
           - Assess management capability
        3. Company Ranking 🏆
           - Rank based on investment potential
           - Provide detailed rationale
           - Consider risk-adjusted returns
           - Explain competitive advantages

        Based on the following market analysis, rank the companies by investment potential and provide a detailed investment analysis and ranking report in markdown format:

        {stock_analyst_result}
    """
        ),
        tools=[],
    )
    return research_analyst.do(researcher_task)


def investment_lead_task(researcher_result: str) -> str:
    investment_lead_task = Task(
        description=dedent(
            f"""\
        You are PortfolioSage-X, a distinguished Senior Investment Lead at Goldman Sachs expert in:
        - Portfolio strategy development
        - Asset allocation optimization
        - Risk management
        - Investment rationale articulation
        - Client recommendation delivery

        Instructions:
        1. Portfolio Strategy 💼
           - Develop allocation strategy
           - Optimize risk-reward balance
           - Consider diversification
           - Set investment timeframes
        2. Investment Rationale 📝
           - Explain allocation decisions
           - Support with analysis
           - Address potential concerns
           - Highlight growth catalysts
        3. Recommendation Delivery 📊
           - Present clear allocations
           - Explain investment thesis
           - Provide actionable insights
           - Include risk considerations

        Based on the following investment analysis and ranking, develop a portfolio allocation strategy and provide a final investment report in markdown format:

        {researcher_result}
    """
        ),
        tools=[],
    )
    return investment_lead.do(investment_lead_task)


def save_as_markdown(result: str):
    with open("result.md", "w") as f:
        f.write(result)


def run_all_tasks(
    companies: str, print_result: bool = True, save_to_file: bool = False
):
    stock_analyst_result = stock_analyst_task(companies)
    rich.print(stock_analyst_result)
    researcher_result = researcher_task(stock_analyst_result)
    rich.print(researcher_result)
    investment_lead_result = investment_lead_task(researcher_result)
    rich.print(investment_lead_result)
    if save_to_file:
        save_as_markdown(investment_lead_result)
    return investment_lead_result


if __name__ == "__main__":
    run_all_tasks(companies, print_result=True, save_to_file=True)
