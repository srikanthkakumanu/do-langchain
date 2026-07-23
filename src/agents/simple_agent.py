from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

load_dotenv()


def build_agent(model: str):
    return create_agent(model=model)


def simple():
    """A single question, no prior history."""

    agent = build_agent("groq:llama-3.1-8b-instant")
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]}
    )
    print(response["messages"][-1].content)


def send_history_to_agent():
    """A follow-up question that only makes sense given the earlier turns."""

    agent = build_agent("groq:llama-3.1-8b-instant")
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="What is the capital of France?"),
                AIMessage(content="The capital of France is Paris."),
                HumanMessage(content="What is the population of Paris?"),
            ]
        }
    )
    print(response["messages"][-1].content)


def main():
    print("=== No history ===")
    simple()

    print("\n=== With history ===")
    send_history_to_agent()


if __name__ == "__main__":
    main()
