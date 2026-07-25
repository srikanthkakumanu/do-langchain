"""
ReAct (Reasoning + Acting) pattern.

The agent alternates between:
  Thought      - reasons about what to do next
  Action       - calls a tool
  Observation  - reads the tool's result
until it has enough information to produce a Final Answer.

`create_agent` already runs this Thought -> Action -> Observation loop under
the hood (it keeps calling the model and executing tool calls until the model
stops requesting tools). This example just makes that loop visible by walking
the returned message history and labeling each step.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient()


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date facts."""
    results = tavily_client.search(query)
    return "\n".join(r["content"] for r in results.get("results", [])[:3])


@tool
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression, e.g. '1341000000 * 0.01'."""
    return eval(expression, {"__builtins__": {}}, {})


system_prompt = """
You are a research assistant that solves problems step by step using the
ReAct pattern: Thought, Action, Observation, repeated until you can give a
Final Answer.

Before every tool call, briefly state your Thought (what you need to find
out next). Use the web_search tool for facts you don't know and the
calculate tool for arithmetic. Do not guess numbers you can look up or
compute.
"""

# provider:model strings understood by create_agent/init_chat_model.
# Each provider needs its own API key in .env (GOOGLE_API_KEY, OPENAI_API_KEY,
# ANTHROPIC_API_KEY, GROQ_API_KEY).
models = {
    "gemini": "google_genai:gemini-2.5-flash",
    "openai": "openai:gpt-5-nano",
    "claude": "anthropic:claude-opus-4-8",
    "groq": "groq:llama-3.3-70b-versatile",
}


def build_agent(model: str):
    return create_agent(
        model=model,
        tools=[web_search, calculate],
        system_prompt=system_prompt,
    )


def as_text(content) -> str:
    """Flatten a message's content into plain text (Gemini/Claude return content blocks)."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def run(question: str, model_key: str = "gemini"):
    agent = build_agent(models[model_key])
    response = agent.invoke({"messages": [HumanMessage(content=question)]})

    for message in response["messages"]:
        kind = message.__class__.__name__

        if kind == "HumanMessage":
            print(f"Question: {as_text(message.content)}\n")
        elif kind == "AIMessage":
            text = as_text(message.content)
            if text:
                print(f"Thought: {text}")
            for call in getattr(message, "tool_calls", None) or []:
                print(f"Action: {call['name']}({call['args']})")
        elif kind == "ToolMessage":
            print(f"Observation: {message.content}\n")

    print(f"Final Answer: {as_text(response['messages'][-1].content)}")


if __name__ == "__main__":
    question = "What is the current population of Japan, and what is 1% of that number?"

    for model_key in models:
        print(f"\n=== {model_key} ({models[model_key]}) ===\n")
        run(question, model_key)
