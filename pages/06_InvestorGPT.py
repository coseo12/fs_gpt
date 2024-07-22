import streamlit as st
import yfinance as yf
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import create_openai_functions_agent, AgentExecutor

# LLM 모델 초기화
llm = ChatOpenAI(
    model="gpt-4-turbo", temperature=0.1
)  # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)


class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = "Use this tool to find the stock market symbol of a company."
    args_description = {
        "query": "The query you will search for",
    }

    def _run(self, query: str):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query=query)


class CompanyHistoryTool(BaseTool):
    name = "CompanyHistory"
    description = """
    Use this to get an history of the financials of the company.
    You should enter a stock symbol.
    """
    args_description = {"symbol": "Stock symbol of the company.\nExample:APPL,TSLA"}

    def _run(self, symbol: str):
        c = yf.Ticker(symbol)
        return c.history(period="1mo")


class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
    Use this to get the income statement of a company.
    You should enter a stock symbol.
    """
    args_description = {"symbol": "Stock symbol of the company.\nExample:APPL,TSLA"}

    def _run(self, symbol: str):
        c = yf.Ticker(symbol)
        return c.income_stmt


class CompanyBalanceSheetTool(BaseTool):
    name = "CompanyBalanceSheet"
    description = """
    Use this to get the balance sheet of a company stock.
    You should enter a stock symbol.
    """
    args_description = {"symbol": "Stock symbol of the company.\nExample:APPL,TSLA"}

    def _run(self, symbol: str):
        c = yf.Ticker(symbol)
        return c.balance_sheet


class CompanyCashflowTool(BaseTool):
    name = "CompanyCashflow"
    description = """
    Use this to get the cashflow of a company stock.
    You should enter a stock symbol.
    """
    args_description = {"symbol": "Stock symbol of the company.\nExample:APPL,TSLA"}

    def _run(self, symbol: str):
        c = yf.Ticker(symbol)
        return c.cashflow


tools = [
    StockMarketSymbolSearchTool(),
    CompanyHistoryTool(),
    CompanyIncomeStatementTool(),
    CompanyBalanceSheetTool(),
    CompanyCashflowTool(),
]

# System prompt
instructions = """
You are a hedge fund manager.

You evaluate a company and provide your opinion and reasons why the stock is a buy or not.

Consider the performance of a stock, the company history, income statement and cashflow.

Be assertive in your judgement and recommand the stock or advise the user against it.
"""

# prompt
prompt = """
Give me financial information on Unity software stock,
considering it's history, financials, income statements, balance sheet and cashflow help me analyze if it's a potential good investment.
"""

# Base prompt
base_prompt = hub.pull("langchain-ai/openai-functions-template")

# Agent
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=base_prompt)

# Agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


st.set_page_config(
    page_title="InvestorGPT",
    page_icon="📈",
)

st.title("Investor GPT")

st.markdown(
    """
Welcome to InvestorGPT.
            
Write down the name of a company and our Agent will do the research for you.
"""
)

company = st.text_input("Write the name of the company you are interested on.")

if company:
    result = agent_executor.invoke({"input": company, "instructions": instructions})

    st.write(result["output"])
