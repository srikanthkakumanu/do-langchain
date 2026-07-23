"""
Structured output via native tool-calling: model.with_structured_output(Schema)

Each provider's model returns a validated Pydantic object directly, instead of
parsing free text or JSON manually. This is the recommended approach for
structured output in LangChain v1.
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

QUERY = "What is the smallest particle in the atomic structure?"


class AtomicParticle(BaseModel):
    """Information about the smallest particle in an atomic structure."""

    particle_name: str = Field(
        description="The name of the smallest particle or concept."
    )
    description: str = Field(
        description="A brief, one to two-sentence description of the particle and its role or nature."
    )


def _ask(model: str, model_provider: str) -> AtomicParticle:
    llm = init_chat_model(model=model, model_provider=model_provider, temperature=0)
    structured_llm = llm.with_structured_output(AtomicParticle)
    return structured_llm.invoke(QUERY)


def groq() -> AtomicParticle:
    return _ask("llama-3.1-8b-instant", "groq")


def gemini() -> AtomicParticle:
    return _ask("gemini-2.5-flash", "google_genai")


def openai() -> AtomicParticle:
    return _ask("gpt-5-nano", "openai")


def claude() -> AtomicParticle:
    return _ask("claude-opus-4-8", "anthropic")


def main():
    for name, fn in [("Groq", groq), ("Gemini", gemini), ("OpenAI", openai), ("Claude", claude)]:
        print(f"--- Calling {name} ---")
        try:
            response = fn()
            print(f"Particle Name: {response.particle_name}")
            print(f"Description: {response.description}\n")
        except Exception as e:
            print(f"Error calling {name}: {e}\n")


if __name__ == "__main__":
    main()
