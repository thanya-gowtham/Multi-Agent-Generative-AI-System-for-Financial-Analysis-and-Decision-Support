import openai 
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq
from fastapi import FastAPI

import os
import phi
from phi.playground import Playground, serve_playground_app

# Load environment variables from .env file
load_dotenv()
phi.api = os.getenv("PHI_API_KEY")

### --- AGENTS CONFIGURATION --- ###

# Web Search Agent Definition
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

# Financial Agent Definition (Automated Stock Analysis)
financial_agent = Agent(
    name="Financial Agent",
    role="Financial Analysis",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True),
    ],
    instructions=[
        "Use tables to display the data.",
        "Perform automated stock analysis and display results.",
        "Provide insights based on technical indicators like RSI and MACD.",
        "Send alerts if RSI indicates overbought or oversold conditions."
    ],
    show_tools_calls=True,
    markdown=True,
)

# Budget Tracking and Planning Agent Definition
budget_tracking_agent = Agent(
    name="Budget Tracking Agent",
    role="Help users track their income, expenses, and savings goals.",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[],
    instructions=[
        "Analyze user-provided financial data (income, expenses).",
        "Suggest personalized budgets based on spending patterns.",
        "Provide tips to save money and achieve savings goals.",
        "Use charts to visualize spending trends."
    ],
    show_tools_calls=True,
    markdown=True,
)

# Debt Management Agent Definition
debt_management_agent = Agent(
    name="Debt Management Agent",
    role="Assist users in managing debt effectively.",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[],
    instructions=[
        "Analyze user debt data (loan amounts, interest rates, repayment schedules).",
        "Suggest optimized repayment plans to minimize interest.",
        "Send reminders for payment due dates.",
        "Provide insights into improving creditworthiness."
    ],
    show_tools_calls=True,
    markdown=True,
)

### --- PLAYGROUND APP CONFIGURATION --- ###

# Create a Playground app with all agents
app = Playground(agents=[financial_agent, web_search_agent, budget_tracking_agent, debt_management_agent]).get_app()

### --- MAIN EXECUTION BLOCK --- ###
if __name__ == "__main__":
    
    # Example: Demonstrating Budget Tracking and Debt Management Agents
    
    # Budget Tracking Example Query
    budget_query = {
        "income": 50000,
        "expenses": {
            "rent": 15000,
            "groceries": 5000,
            "entertainment": 3000,
            "utilities": 2000,
            "miscellaneous": 5000
        }
    }
    
    print("Budget Tracking Results:")
    
    # Assuming the agent processes the query (mock example)
    budget_results = budget_tracking_agent.run(budget_query)
    
    print(budget_results)
    
    # Debt Management Example Query
    debt_query = {
        "loans": [
            {"amount": 200000, "interest_rate": 5.5, "monthly_payment": 5000},
            {"amount": 100000, "interest_rate": 4.0, "monthly_payment": 3000}
        ],
        "credit_score": 650
    }
    
    print("Debt Management Results:")
    
    # Assuming the agent processes the query (mock example)
    debt_results = debt_management_agent.run(debt_query)
    
    print(debt_results)
    
    # Serve the Playground app (web interface for agents)
    serve_playground_app("playground:app", reload=False)