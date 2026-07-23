from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

load_dotenv()


def build_agent(model: str):
    return create_agent(model=model)


def stream_tokens(messages):
    """Prints each token as the model produces it, instead of waiting for the full response."""

    agent = build_agent("groq:llama-3.1-8b-instant")

    # stream_mode="messages" yields (token, metadata) pairs: token is a message
    # chunk, metadata says which node produced it.
    for token, _metadata in agent.stream(messages, stream_mode="messages"):
        if token.content:
            print(token.content, end="", flush=True)
    print()


def stream_simple():
    stream_tokens({"messages": [HumanMessage(content="What is the capital of France?")]})


def stream_history():
    stream_tokens(
        {
            "messages": [
                HumanMessage(content="What is the capital of France?"),
                AIMessage(content="The capital of France is Paris."),
                HumanMessage(content="What is the population of Paris?"),
            ]
        }
    )


def main():
    print("=== No history ===")
    stream_simple()

    print("\n=== With history ===")
    stream_history()


if __name__ == "__main__":
    main()
