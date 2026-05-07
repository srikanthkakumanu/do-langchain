from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

load_dotenv()


class CosmologyInfo(BaseModel):
    """Structured information about a cosmology question."""

    answer: str = Field(description="The answer to the user's question.")
    confidence: float = Field(
        description="A confidence score for the answer, between 0.0 and 1.0."
    )
    sources: list[str] = Field(
        description="A list of sources used to find the answer. Can be empty."
    )


def createAgent(model: str):
    system_prompt = """You are an expert in Cosmology. Your goal is to answer the user's question accurately.
    You must provide your response in the requested structured format, including the answer, a confidence score, and a list of sources.
    - The answer should be a clear and concise response to the question.
    - The confidence score should be a float between 0.0 and 1.0, representing how sure you are of your answer.
    - The sources should be a list of URLs or references. If you cannot find any, provide an empty list.
    """

    return create_agent(
        model=model, system_prompt=system_prompt, response_format=CosmologyInfo
    )


def simple(message: str):
    agent = createAgent("groq:llama-3.1-8b-instant")
    response = agent.invoke({"messages": [HumanMessage(content=message)]})
    structured_response = response.get("structured_response")
    print(response["structured_response"])
    if structured_response:
        print(structured_response)
        print(f"Answer: {structured_response.answer}")
        print(f"Confidence: {structured_response.confidence}")
        print(f"Sources: {structured_response.sources}")
    else:
        print("Failed to get structured response.")
        print("Full response:")
        print(response)


def main():
    simple("What is oldest galaxy in the universe?")


if __name__ == "__main__":
    main()
