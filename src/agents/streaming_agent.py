from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

load_dotenv()


def createAgent(model: str):
    return create_agent(model=model)


def stream_simple():
    agent = createAgent("groq:llama-3.1-8b-instant")

    for token, metadata in agent.stream(
        {"messages": [HumanMessage(content="What is the capital of France?")]},
        stream_mode="messages",
    ):
        # token is a message chunk with token content
        # metadata contains which node produced the token

        if token.content:  # Check if there's actual content
            print(token.content, end="", flush=True)  # Print token


def stream_history(messages):
    agent = createAgent("groq:llama-3.1-8b-instant")

    for token, metadata in agent.stream(
        messages,
        stream_mode="messages",
    ):
        # token is a message chunk with token content
        # metadata contains which node produced the token
        if token.content:  # Check if there's actual content
            print(token.content, end="", flush=True)  # Print token


def main():
    # stream_simple()
    stream_history(
        {
            "messages": [
                HumanMessage(content="What is the capital of France?"),
                AIMessage(content="The capital of France is Paris."),
                HumanMessage(content="What is the population of Paris?"),
            ]
        }
    )


if __name__ == "__main__":
    main()
