from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

load_dotenv()


def createAgent(model: str):
    return create_agent(model=model)


def simple():
    agent = createAgent("groq:llama-3.1-8b-instant")

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]}
    )
    print(response)
    print(response["messages"][-1].content)


def send_history_to_agent():
    agent = createAgent("groq:llama-3.1-8b-instant")

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="What is the capital of France?"),
                AIMessage(content="The capital of France is Paris."),
                HumanMessage(content="What is the population of Paris?"),
            ]
        }
    )
    print(response)
    print(response["messages"][-1].content)


def main():
    # simple()
    send_history_to_agent()


if __name__ == "__main__":
    main()
