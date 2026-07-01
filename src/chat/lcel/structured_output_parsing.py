"""
Structured output parsing: prompt | model | parser into JSON/Pydantic
"""

from utils.llm_utils import load_environment_variables, get_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


load_environment_variables()


class MovieReview(BaseModel):
    title: str = Field(description="Title of the movie")
    rating: int = Field(description="Rating out of 10")
    verdict: str = Field(description="One-sentence verdict")


def structured_output_chain():
    """Parses model output into a validated Pydantic object instead of free text."""

    parser = JsonOutputParser(pydantic_object=MovieReview)

    prompt = ChatPromptTemplate.from_template(
        "Write a short review for the movie: {movie}\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    llm_model = get_model("llama70b")
    chain = prompt | llm_model | parser

    result = chain.invoke({"movie": "The Matrix"})
    print(f"Result: {result}")
    print(f"Title: {result['title']}, Rating: {result['rating']}/10")


def main():
    structured_output_chain()


if __name__ == "__main__":
    main()
