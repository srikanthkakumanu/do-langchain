"""
Conversation Memory patterns: Buffer, Window, and Summary.

Buffer persists the full message history and resends all of it every turn --
simple, but unbounded. Window trims what's sent to the model (via
middleware) without touching the persisted history -- bounded, but it
silently forgets anything that scrolls out. Summary compresses the messages
that fall out of the window into a running summary instead of dropping them,
so old context survives in compressed form.

Each demo runs the same three-turn conversation (the first turn states a
name and task, the last turn asks for both back) on a fresh thread_id, and
prints how many messages are actually sent to/persisted by the model each
turn, so the difference between the three strategies is observable.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, wrap_model_call
from langchain.messages import HumanMessage
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# provider:model strings understood by create_agent/init_chat_model.
# Each provider needs its own API key in .env (GOOGLE_API_KEY, OPENAI_API_KEY,
# ANTHROPIC_API_KEY, GROQ_API_KEY).
models = {
    "gemini": "google_genai:gemini-2.5-flash",
    "openai": "openai:gpt-5-nano",
    "claude": "anthropic:claude-opus-4-8",
    "groq": "groq:llama-3.3-70b-versatile",
}

turns = [
    "My name is Priya and I'm debugging a langchain agent.",
    "It keeps forgetting earlier context after a few turns.",
    "What's my name, and what am I working on?",
]


@wrap_model_call
def windowed_history(request, handler):
    trimmed = trim_messages(
        request.messages,
        strategy="last",
        token_counter=len,  # counts messages, not tokens
        max_tokens=4,  # keep only the last few messages sent to the model
        start_on="human",
        include_system=True,
    )
    print(
        f"  [window] sending {len(trimmed)}/{len(request.messages)} messages to the model"
    )
    return handler(request.override(messages=trimmed))


def run_buffer(model_key: str):
    print(f"\n--- Buffer memory: {model_key} ---")
    agent = create_agent(models[model_key], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"buffer-{model_key}"}}

    response = None
    for turn in turns:
        response = agent.invoke({"messages": [HumanMessage(content=turn)]}, config=config)
        persisted = agent.get_state(config).values["messages"]
        print(f"  persisted history = {len(persisted)} messages")

    print(f"Final Answer: {response['messages'][-1].content}")


def run_window(model_key: str):
    print(f"\n--- Window memory: {model_key} ---")
    agent = create_agent(
        models[model_key],
        checkpointer=InMemorySaver(),
        middleware=[windowed_history],
    )
    config = {"configurable": {"thread_id": f"window-{model_key}"}}

    response = None
    for turn in turns:
        response = agent.invoke({"messages": [HumanMessage(content=turn)]}, config=config)

    print(f"Final Answer: {response['messages'][-1].content}")


def run_summary(model_key: str):
    print(f"\n--- Summary memory: {model_key} ---")
    summarize = SummarizationMiddleware(
        model=models[model_key],
        trigger=("messages", 4),  # summarize once history exceeds 4 messages
        keep=("messages", 2),  # always keep the most recent 2 messages verbatim
    )
    agent = create_agent(
        models[model_key],
        checkpointer=InMemorySaver(),
        middleware=[summarize],
    )
    config = {"configurable": {"thread_id": f"summary-{model_key}"}}

    response = None
    for turn in turns:
        response = agent.invoke({"messages": [HumanMessage(content=turn)]}, config=config)
        persisted = agent.get_state(config).values["messages"]
        print(f"  persisted history = {len(persisted)} messages")

    print(f"Final Answer: {response['messages'][-1].content}")


if __name__ == "__main__":
    for model_key in models:
        print(f"\n=== {model_key} ({models[model_key]}) ===")
        run_buffer(model_key)
        run_window(model_key)
        run_summary(model_key)
