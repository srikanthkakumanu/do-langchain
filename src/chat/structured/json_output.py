from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

query = "What is the capital of France?"


# 1. Define Pydantic Schema
class CapitalCity(BaseModel):
    """Information about a country's capital city."""

    country: str = Field(description="The name of the country being asked about.")
    capital: str = Field(description="The name of the capital city.")


# 2. Create the Parser
parser = PydanticOutputParser(pydantic_object=CapitalCity)

# 3. Build the Prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that provides information about capital cities.",
        ),
        ("user", "{query}\n\n{format_instructions}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 4. Initialize LLMs
llms = {
    "llama-3.1-8b-instant": init_chat_model(
        model="llama-3.1-8b-instant", model_provider="groq", temperature=0
    ),
    "gemini-2.5-flash": ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0),
    # "OpenAI (gpt-5-nano)": init_chat_model(model="gpt-5-nano", temperature=0),
}

# 5. Create Chain prompt -> model -> parser and Invoke LLMs
for model_name, llm in llms.items():
    print(f"--- Calling {model_name} ---")
    try:
        # Chain prompt -> model -> parser
        chain = prompt_template | llm | parser
        response = chain.invoke({"query": query})
        print(f"Country: {response.country}")
        print(f"Capital: {response.capital}\n")
    except Exception as e:
        print(f"Error calling {model_name}: {e}\n")
        continue
