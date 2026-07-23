"""
Common utilities for working with LLMs using LangChain v1.

Provides a shared, lazily-initialized chat model factory (Groq, Google Gemini,
OpenAI, Anthropic Claude) so example scripts don't each redefine their own
model dictionary, and only need the API key(s) for the models they actually use.
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

load_dotenv()


def load_environment_variables():
    """Loads environment variables from a .env file."""
    load_dotenv()


# provider/model pairs understood by init_chat_model. Each provider needs its
# own API key in .env (GROQ_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY).
_MODEL_CONFIG = {
    "llama": {"model": "llama-3.1-8b-instant", "model_provider": "groq"},
    "llama70b": {"model": "llama-3.3-70b-versatile", "model_provider": "groq"},
    "gemini": {"model": "gemini-2.5-flash", "model_provider": "google_genai"},
    "openai": {"model": "gpt-5-nano", "model_provider": "openai"},
    "claude": {"model": "claude-opus-4-8", "model_provider": "anthropic"},
}

_model_cache: dict[str, BaseChatModel] = {}


def get_model(llm_name: str = "llama") -> BaseChatModel:
    """Returns a cached, initialized chat model for the given name.

    Available names: llama, llama70b, gemini, openai, claude.
    """
    if llm_name not in _MODEL_CONFIG:
        raise ValueError(f"Unknown model name: {llm_name}")

    if llm_name not in _model_cache:
        _model_cache[llm_name] = init_chat_model(
            temperature=0.7,
            streaming=True,
            max_tokens=200,
            max_retries=3,
            **_MODEL_CONFIG[llm_name],
        )
    return _model_cache[llm_name]


def invoke_model(llm: BaseChatModel, prompts: list[BaseMessage] | str) -> AIMessage:
    """Invokes the provided model with the given prompt and returns the response."""

    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown model")
    print(f"\n--- Calling {model_name} ---")

    return llm.invoke(prompts)
