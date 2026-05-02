from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

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


# 2. Define the prompt for the LLMs
prompt = "What is the smallest particle in the atomic structure?"

# 3. Initialize a dictionary of models to compare
#    Each model is wrapped with .with_structured_output to parse the response
#    into our AtomicParticle schema.
llms = {
    # "OpenAI (gpt-5-nano)": ChatOpenAI(model="gpt-5-nano", temperature=0).with_structured_output(AtomicParticle),
    "Groq (llama-3.1-8b-instant)": ChatGroq(
        model="llama-3.1-8b-instant", temperature=0
    ).with_structured_output(AtomicParticle),
    "Google (gemini-2.5-flash)": ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0
    ).with_structured_output(AtomicParticle),
}

# 4. Invoke each model and print its structured response
for name, llm in llms.items():
    print(f"--- Calling {name} ---")
    try:
        # The response is now a Pydantic object
        response = llm.invoke(prompt)
        print(f"Particle Name: {response.particle_name}")
        print(f"Description: {response.description}\n")
    except Exception as e:
        # Add error handling in case an API key is missing or another issue occurs
        print(f"Error calling {name}: {e}\n")
