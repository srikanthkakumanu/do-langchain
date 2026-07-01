"""
Parallel and branching chains: RunnableParallel and RunnableBranch
"""

from utils.llm_utils import load_environment_variables, get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch


load_environment_variables()


def parallel_chain():
    """Runs multiple branches concurrently on the same input and merges results."""

    llm_model = get_model("llama70b")
    parser = StrOutputParser()

    summary_chain = (
        ChatPromptTemplate.from_template("Summarize in one sentence: {text}")
        | llm_model
        | parser
    )
    keyword_chain = (
        ChatPromptTemplate.from_template(
            "List 3 comma-separated keywords for: {text}"
        )
        | llm_model
        | parser
    )

    chain = RunnableParallel(summary=summary_chain, keywords=keyword_chain)

    result = chain.invoke({"text": "LangChain lets you compose LLM calls into chains."})
    print(f"Summary: {result['summary']}")
    print(f"Keywords: {result['keywords']}")


def branching_chain():
    """Routes input to one of several chains based on a condition."""

    llm_model = get_model("llama70b")
    parser = StrOutputParser()

    math_chain = (
        ChatPromptTemplate.from_template("Solve this math problem: {question}")
        | llm_model
        | parser
    )
    general_chain = (
        ChatPromptTemplate.from_template("Answer this question: {question}")
        | llm_model
        | parser
    )

    chain = RunnableBranch(
        (lambda x: x["type"] == "math", math_chain),
        general_chain,
    )

    math_result = chain.invoke({"type": "math", "question": "What is 12 * 7?"})
    print(f"Math result: {math_result}")

    general_result = chain.invoke(
        {"type": "general", "question": "What is the capital of Japan?"}
    )
    print(f"General result: {general_result}")


def main():
    parallel_chain()
    branching_chain()


if __name__ == "__main__":
    main()
