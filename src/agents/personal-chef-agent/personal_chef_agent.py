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


SYSTEM_PROMPT = """You are a personal chef. The user will give you a list of ingredients
they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the
ingredients they have. Return recipe suggestions, and the full recipe instructions
if the user asks for them.
"""


def build_agent():
    return create_agent(
        model="google_genai:gemini-2.5-flash",
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT,
    )


# Module-level binding required by langgraph.json ("./personal_chef_agent.py:agent")
# so `langgraph dev` can find and serve this agent.
agent = build_agent()


def main():
    question = HumanMessage(content="I have eggs, spinach, and feta cheese. What can I make?")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
