# LangChain Output Parsers

Output parsers turn raw model output into a shape that application code can use. In LCEL, they usually sit at the end of a chain:

```python
chain = prompt | model | parser
result = chain.invoke(input_data)
```

Use a parser when the model does not provide native structured output, or when you want an extra parsing or validation step after the model responds.

## Common Parser Types

| Parser                             | Output                             | Use when                                                     |
| ---------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| `StrOutputParser`                | `str`                            | You want plain text from a chat/model response.              |
| `JsonOutputParser`               | `dict` / `list`                | You want valid JSON, optionally with streaming partial JSON. |
| `SimpleJsonOutputParser`         | `dict` / `list`                | You want the simple JSON parser name used by some examples.  |
| `PydanticOutputParser`           | Pydantic model                     | You want typed, validated structured data.                   |
| `CommaSeparatedListOutputParser` | `list[str]`                      | You want a simple comma-separated list.                      |
| `MarkdownListOutputParser`       | `list[str]`                      | You ask the model for a Markdown bullet list.                |
| `NumberedListOutputParser`       | `list[str]`                      | You ask the model for a numbered list.                       |
| `XMLOutputParser`                | nested`dict`                     | You want XML-like structured output.                         |
| Tool call parsers                  | tool call data or Pydantic objects | You are parsing model tool/function-call outputs.            |
| Base parser classes                | custom output                      | You need to build your own parser.                           |

## `StrOutputParser`

Parses the model response into a simple string. This is the most common parser for normal text generation.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"topic": "LCEL"})

print(result)
```

## `JsonOutputParser`

Parses model output into a Python JSON value, usually a `dict`. It is useful when later code needs reliable keys instead of free-form prose.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template(
    "Return only JSON with keys name and difficulty for the dish: {dish}"
)

chain = prompt | model | parser
result = chain.invoke({"dish": "scrambled eggs"})

print(result["name"])
print(result["difficulty"])
```

## `SimpleJsonOutputParser`

An alternate exported name for simple JSON parsing. In most new code, prefer `JsonOutputParser`; use this when following examples or codebases that already use the simple name.

```python
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = SimpleJsonOutputParser()
prompt = ChatPromptTemplate.from_template(
    "Return only JSON with keys title and author for the book: {book}"
)

chain = prompt | model | parser
result = chain.invoke({"book": "Dune"})

print(result["title"])
```

## `PydanticOutputParser`

Parses and validates model output as a Pydantic object. Use it when fields should have specific names and types.

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class Recipe(BaseModel):
    name: str = Field(description="Recipe name")
    ingredients: list[str] = Field(description="Main ingredients")
    prep_time_minutes: int = Field(description="Preparation time in minutes")


parser = PydanticOutputParser(pydantic_object=Recipe)
prompt = ChatPromptTemplate.from_template(
    "Create a recipe for {dish}.\n\n{format_instructions}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser
result = chain.invoke({"dish": "pancakes"})

print(result.name)
print(result.prep_time_minutes)
```

## `CommaSeparatedListOutputParser`

Parses comma-separated text into a list of strings. Use it for lightweight lists where JSON would be overkill.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = CommaSeparatedListOutputParser()
prompt = ChatPromptTemplate.from_template(
    "List 5 programming languages.\n\n{format_instructions}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser
result = chain.invoke({})

print(result)
```

## `MarkdownListOutputParser`

Parses a Markdown bullet list into a Python list. Use it when Markdown is natural for the prompt or final answer format.

```python
from langchain_core.output_parsers import MarkdownListOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = MarkdownListOutputParser()
prompt = ChatPromptTemplate.from_template(
    "Give three benefits of testing as a Markdown bullet list."
)

chain = prompt | model | parser
result = chain.invoke({})

print(result)
```

## `NumberedListOutputParser`

Parses a numbered list into a Python list. Use it when order matters or the prompt asks for ranked steps.

```python
from langchain_core.output_parsers import NumberedListOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = NumberedListOutputParser()
prompt = ChatPromptTemplate.from_template(
    "Give the top 3 steps to debug a failing test as a numbered list."
)

chain = prompt | model | parser
result = chain.invoke({})

print(result)
```

## `XMLOutputParser`

Parses XML-like model output into a nested dictionary. Use it when tag-based structure is easier for the model or the downstream system expects XML-like data.

```python
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = XMLOutputParser(tags=["movie", "title", "genre"])
prompt = ChatPromptTemplate.from_template(
    "Return one movie using XML tags: <movie><title>...</title><genre>...</genre></movie>"
)

chain = prompt | model | parser
result = chain.invoke({})

print(result)
```

## Tool Call Parsers

Tool call parsers read structured tool/function-call data from chat model responses. They are useful when the model is calling tools and you want just the parsed call arguments instead of the whole message.

Common tool parsers include:

- `JsonOutputToolsParser` - returns tool calls as JSON-like dictionaries.
- `JsonOutputKeyToolsParser` - returns tool calls for a specific tool name.
- `PydanticToolsParser` - returns tool calls parsed into Pydantic objects.

```python
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(description="Search query to run")


model_with_tools = model.bind_tools([SearchQuery])
parser = PydanticToolsParser(tools=[SearchQuery])

chain = model_with_tools | parser
result = chain.invoke("Search for recent LangChain parser examples")

print(result)
```

## Base Parser Classes

Base classes are for custom parsers. Use them when the built-in parsers do not match the format you need.

Common base classes include:

- `BaseOutputParser` - base class for text-like parser inputs.
- `BaseLLMOutputParser` - base class for parsers that work directly with LLM generation results.
- `BaseTransformOutputParser` - base class for parsers that transform streamed output.
- `BaseCumulativeTransformOutputParser` - base class for parsers that accumulate streamed chunks.

```python
from langchain_core.output_parsers import BaseOutputParser


class UppercaseParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        return text.strip().upper()


parser = UppercaseParser()
chain = prompt | model | parser

result = chain.invoke({"topic": "langchain"})
print(result)
```

## Choosing a Parser

- Use `StrOutputParser` for normal answers.
- Use `JsonOutputParser` for flexible structured data.
- Use `PydanticOutputParser` when schema validation matters.
- Use list parsers for quick list-shaped responses.
- Use `XMLOutputParser` when XML tags are the easiest format to prompt or integrate with.
- Use tool call parsers when working with model tool/function calling.
