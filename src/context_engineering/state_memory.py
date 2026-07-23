"""
Conversational memory via a checkpointer.

`create_agent(..., checkpointer=...)` persists message history per `thread_id`,
so a second `invoke()` call with the same thread_id remembers earlier turns
without the caller re-sending history manually.

For a comparison of memory *strategies* (buffer vs. window vs. summary), see
patterns/Memory_Pattern.py -- this file only covers the checkpointer basics.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# provider:model strings understood by create_agent.
# Each provider needs its own API key in .env (GOOGLE_API_KEY, OPENAI_API_KEY,
# ANTHROPIC_API_KEY, GROQ_API_KEY).
models = {
    "gemini": "google_genai:gemini-2.5-flash",
    "openai": "openai:gpt-5-nano",
    "claude": "anthropic:claude-opus-4-8",
    "groq": "groq:llama-3.1-8b-instant",
}


def run(model_key: str):
    agent = create_agent(models[model_key], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"state-memory-{model_key}"}}

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Hello, my name is John. What is the oldest galaxy in the universe?"
                )
            ]
        },
        config=config,
    )
    print(f"First call: {response['messages'][-1].content}")

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the oldest star in the universe?")]},
        config=config,
    )
    print(f"Second call: {response['messages'][-1].content}")


def main():
    for model_key in models:
        print(f"\n=== {model_key} ({models[model_key]}) ===")
        run(model_key)


if __name__ == "__main__":
    main()
