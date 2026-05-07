from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

load_dotenv()


def createAgent(model: str):

    system_prompt = """You are a expert in Cosmology and you are answering questions at the users request 
    about the universe.

    Please keep to the below structure.

    User:  What is the Nearest star to the Earth?
    Cosmologist: The nearest start to Earth is Sun.

    User:  How old is this Universe?
    Cosmologist: This Universe is approximately 13.4 billion years old
    """

    return create_agent(model=model, system_prompt=system_prompt)


def stream_simple(message: str):
    agent = createAgent("groq:llama-3.1-8b-instant")

    for token, metadata in agent.stream(
        {"messages": [HumanMessage(content=message)]},
        stream_mode="messages",
    ):
        # token is a message chunk with token content
        # metadata contains which node produced the token

        if token.content:  # Check if there's actual content
            print(token.content, end="", flush=True)  # Print token


def stream_history(messages: dict):
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
    stream_simple("What is oldest galaxy in the universe?")
    stream_history(
        {
            "messages": [
                HumanMessage(content="What is oldest galaxy in the universe?"),
                AIMessage(
                    content="The oldest galaxy with spectroscopic confirmation is GN-z11, at redshift z ≈ 11.09, seen as it was about 400 million years after the Big Bang (roughly 13.4 billion years ago). JWST has found several high-redshift galaxy candidates (z ~ 12–16) based on photometric data, which could be older if spectroscopic confirmation confirms their redshifts, but GN-z11 remains the oldest confirmed galaxy so far."
                ),
                HumanMessage(content="What is the oldest blackhole in the universe?"),
            ]
        }
    )


if __name__ == "__main__":
    main()
