"""
Basic sequential chain: prompt | model | parser
"""

from utils.llm_utils import load_environment_variables, get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_environment_variables()


def sequential_chain():
    """Runs prompt -> model -> parser, one step feeding the next."""

    llm_model = get_model("llama70b")
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in exactly one short sentence."
    )
    parser = StrOutputParser()

    chain = prompt | llm_model | parser

    result = chain.invoke({"topic": "LCEL"})
    print(f"Result: {result}")


def main():
    sequential_chain()


if __name__ == "__main__":
    main()
