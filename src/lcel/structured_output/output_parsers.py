"""
Output parsers and Structured output using LangChain V.1
"""

from utils.llm_utils import load_environment_variables, get_model
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    SimpleJsonOutputParser,
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
    XMLOutputParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


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
        ingredients: list[str] = Field(description="List of ingredients")
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


def simple_json_output_parser():
    """SimpleJsonOutputParser streams partial JSON as it is generated."""

    llm_model = get_model("llama70b")
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'name' and 'age' for a fictional character.\n"
        "Return ONLY valid JSON, no explanation."
    )

    parser = SimpleJsonOutputParser()
    chain = prompt | llm_model | parser

    for chunk in chain.stream({}):
        print(f"Chunk: {chunk}")


def comma_separated_list_output_parser():
    """CommaSeparatedListOutputParser turns a comma-separated response into a list."""

    parser = CommaSeparatedListOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "List 5 primary colors.\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    llm_model = get_model("llama70b")
    chain = prompt | llm_model | parser

    result = chain.invoke({})
    print(f"Result: {result}")
    print(f"Type: {type(result).__name__}, Count: {len(result)}")


def markdown_list_output_parser():
    """MarkdownListOutputParser turns a markdown '- item' response into a list."""

    parser = MarkdownListOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "List 3 benefits of regular exercise.\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    llm_model = get_model("llama70b")
    chain = prompt | llm_model | parser

    result = chain.invoke({})
    print(f"Result: {result}")


def numbered_list_output_parser():
    """NumberedListOutputParser turns a numbered-list response into a list."""

    parser = NumberedListOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "List 3 steps to make a cup of tea.\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    llm_model = get_model("llama70b")
    chain = prompt | llm_model | parser

    result = chain.invoke({})
    print(f"Result: {result}")


def xml_output_parser():
    """XMLOutputParser parses a tagged XML response into a nested dict."""

    parser = XMLOutputParser(tags=["person", "name", "age"])

    prompt = ChatPromptTemplate.from_template(
        "Generate a fictional person's name and age.\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    llm_model = get_model("llama70b")
    chain = prompt | llm_model | parser

    result = chain.invoke({})
    print(f"Result: {result}")


def pydantic_tools_parser():
    """PydanticToolsParser converts tool-call responses into Pydantic objects."""

    class GetWeather(BaseModel):
        """Get the current weather for a location."""

        location: str = Field(description="City and state, e.g. San Francisco, CA")

    llm_model = get_model("llama70b")
    llm_with_tools = llm_model.bind_tools([GetWeather])

    prompt = ChatPromptTemplate.from_template("What's the weather like in {city}?")
    parser = PydanticToolsParser(tools=[GetWeather])

    chain = prompt | llm_with_tools | parser

    result = chain.invoke({"city": "Boston"})
    print(f"Result: {result}")
    if result:
        print(f"Location: {result[0].location}")


DEMOS = [
    string_output_parser,
    json_output_parser,
    pydantic_structured_output_parser,
    simple_json_output_parser,
    comma_separated_list_output_parser,
    markdown_list_output_parser,
    numbered_list_output_parser,
    xml_output_parser,
    pydantic_tools_parser,
]


def main():
    for demo in DEMOS:
        print(f"\n=== {demo.__name__} ===")
        demo()


if __name__ == "__main__":
    main()
