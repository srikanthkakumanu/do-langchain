"""
Output parsers and Structured output using LangChain V.1
"""

from email import parser
from pyexpat import model

from typer import prompt

from utils.llm_utils import load_environment_variables, get_model
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional


load_environment_variables()


def string_output_parser():
    """Basic string output parser."""

    llm_model = get_model("llama70b")
    prompt = ChatPromptTemplate.from_template(
        "Give me one-word answer: What color is the sky?"
    )

    parser = StrOutputParser()
    chain = prompt | llm_model | parser

    result = chain.invoke({})

    print(f"Result: '{result}' (type: {type(result).__name__})")


def json_output_parser():
    """Basic JSON output parser."""

    llm_model = get_model("llama70b")
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'city' and 'country' for: {place}\n"
        "Return ONLY valid JSON, no explanation."
    )

    parser = JsonOutputParser()
    chain = prompt | llm_model | parser

    result = chain.invoke({"place": "The Eiffel Tower"})

    print(f"Result: {result}")
    print(f"City: {result['city']}, Country: {result['country']}")


def pydantic_structured_output_parser():
    """Pydantic output parser for type-safe structured data."""

    # Define schema
    class Recipe(BaseModel):
        name: str = Field(description="Name of the recipe")
        ingredients: List[str] = Field(description="List of ingredients")
        prep_time_minutes: int = Field(description="Preparation time in minutes")
        difficulty: str = Field(description="easy, medium, or hard")

    parser = PydanticOutputParser(pydantic_object=Recipe)

    prompt = ChatPromptTemplate.from_template(
        "Create a simple recipe for: {dish}\n\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    llm_model = get_model("llama70b")

    chain = prompt | llm_model | parser

    result = chain.invoke({"dish": "scrambled eggs"})
    print(f"Recipe: {result.name}")
    print(f"Ingredients: {result.ingredients}")
    print(f"Prep time: {result.prep_time_minutes} mins")
    print(f"Difficulty: {result.difficulty}")

    # Type-safe access
    print(
        f"\nType check - prep_time is int: {isinstance(result.prep_time_minutes, int)}"
    )


def main():
    # string_output_parser()
    # json_output_parser()
    pydantic_structured_output_parser()


if __name__ == "__main__":
    main()
