"""
Custom steps with RunnableLambda
"""

from utils.llm_utils import load_environment_variables, get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


load_environment_variables()


def custom_step_chain():
    """Wraps plain Python logic as a Runnable step inside an LCEL chain."""

    def clean_text(text: str) -> str:
        return text.strip().lower()

    def shout(text: str) -> str:
        return text.upper() + "!!!"

    clean_input = RunnableLambda(clean_text)
    excited_output = RunnableLambda(shout)

    llm_model = get_model("llama70b")
    prompt = ChatPromptTemplate.from_template("Explain briefly: {topic}")

    chain = clean_input | (lambda topic: {"topic": topic}) | prompt | llm_model | StrOutputParser() | excited_output

    result = chain.invoke("  Explain LCEL  ")
    print(f"Result: {result}")


def main():
    custom_step_chain()


if __name__ == "__main__":
    main()
