from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()


# 1. Define the desired Pydantic output schema
class AtomicParticle(BaseModel):
    """Information about the smallest particle in an atomic structure."""

    particle_name: str = Field(
        description="The name of the smallest particle or concept."
    )
    description: str = Field(
        description="A brief, one to two-sentence description of the particle and its role or nature."
    )


# 2. Set up a parser
parser = JsonOutputParser(pydantic_object=AtomicParticle)

# 3. Create a prompt template with format instructions
prompt_template = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. Initialize a dictionary of models
llms = {
    # "OpenAI (gpt-5-nano)": ChatOpenAI(model="gpt-5-nano", temperature=0),
    "Groq (llama-3.1-8b-instant)": ChatGroq(
        model="llama-3.1-8b-instant", temperature=0
    ),
    "Google (gemini-2.5-flash)": ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0
    ),
}

# The user query
query = "What is the smallest particle in the atomic structure?"

# 5. Create chains and invoke them
for name, llm in llms.items():
    print(f"--- Calling {name} ---")
    try:
        # Create the chain by piping the components together
        chain = prompt_template | llm | parser
        # The response is now a dictionary parsed from the JSON output
        response = chain.invoke({"query": query})
        print(f"Particle Name: {response['particle_name']}")
        print(f"Description: {response['description']}\n")
    except Exception as e:
        # Add error handling in case an API key is missing or another issue occurs
        print(f"Error calling {name}: {e}\n")
