"""
Streaming and batching the same chain
"""

from utils.llm_utils import load_environment_variables, get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_environment_variables()


def build_chain():
    llm_model = get_model("llama70b")
    prompt = ChatPromptTemplate.from_template("Give one fun fact about {topic}.")
    return prompt | llm_model | StrOutputParser()


def streaming_example():
    """Streams the same chain's output chunk by chunk as it's produced."""

    chain = build_chain()

    print("Streaming: ", end="")
    for chunk in chain.stream({"topic": "octopuses"}):
        print(chunk, end="", flush=True)
    print()


def batching_example():
    """Runs the same chain across multiple inputs in one call."""

    chain = build_chain()

    results = chain.batch(
        [
            {"topic": "LCEL"},
            {"topic": "Runnables"},
            {"topic": "LangSmith"},
        ]
    )
    for topic, result in zip(["LCEL", "Runnables", "LangSmith"], results):
        print(f"{topic}: {result}")


def main():
    streaming_example()
    batching_example()


if __name__ == "__main__":
    main()
