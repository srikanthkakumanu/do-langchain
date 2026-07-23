from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient()


@tool
def web_search(query: str) -> dict[str, Any]:
    """Search the web for information"""

    return tavily_client.search(query)


def without_web_search_tool():
    agent = create_agent("google_genai:gemini-2.5-flash")
    question = HumanMessage(content="How up to date is your training knowledge?")

    response = agent.invoke({"messages": [question]})
    print(response["messages"])


def with_web_search_tool():
    agent = create_agent(model="google_genai:gemini-2.5-flash", tools=[web_search])
    question = HumanMessage(content="Who is the current prime minister of Japan?")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    print("Without web search tool:")
    without_web_search_tool()
    print("\nWith web search tool:")
    with_web_search_tool()
