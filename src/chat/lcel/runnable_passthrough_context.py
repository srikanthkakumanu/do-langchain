"""
Passing extra context through with RunnablePassthrough
"""

from utils.llm_utils import load_environment_variables, get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_environment_variables()


def passthrough_chain():
    """Keeps the original input while adding a computed 'context' key for the prompt."""

    def fake_retriever(inputs: dict) -> str:
        # Stand-in for a real retriever/database lookup.
        facts = {
            "LCEL": "LCEL is LangChain's declarative chain-composition language.",
        }
        return facts.get(inputs["question"], "No context found.")

    llm_model = get_model("llama70b")
    prompt = ChatPromptTemplate.from_template(
        "Using this context: {context}\nAnswer the question: {question}"
    )

    chain = (
        RunnablePassthrough.assign(context=fake_retriever)
        | prompt
        | llm_model
        | StrOutputParser()
    )

    result = chain.invoke({"question": "LCEL"})
    print(f"Result: {result}")


def main():
    passthrough_chain()


if __name__ == "__main__":
    main()
