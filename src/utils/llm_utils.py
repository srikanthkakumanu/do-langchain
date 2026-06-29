"""
Common utilities for working with LLMs using LangChain V.1
This module provides utility functions for initializing and invoking
different chat models (LLaMA, Gemini, OpenAI) and handling their responses.
It also includes functions for loading environment variables and managing
multi-shot conversations.
The code is structured to allow easy addition of new models and response formats
in the future.
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    message_to_dict,
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

load_dotenv()


def load_environment_variables():
    """Loads environment variables from a .env file."""
    load_dotenv()


llms = {
    "llama": init_chat_model(
        model="llama-3.1-8b-instant",
        model_provider="groq",
        temperature=0.7,
        streaming=True,
        max_tokens=200,
        max_retries=3,
    ),
    "llama70b": init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq",
        temperature=0.7,
        streaming=True,
        max_tokens=200,
        max_retries=3,
    ),
    "gemini": init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        temperature=0.7,
        streaming=True,
        max_tokens=200,
        max_retries=3,
    ),
    # ,
    # "openai": init_chat_model(
    #     model="gpt-5-nano",
    #     model_provider="openai",
    #     temperature=0.7,
    #     streaming=True,
    #     max_tokens=200,
    #     max_retries=3,
    # ),
}


def get_model(llm_name: str = "llama"):
    """Returns the initialized model based on the provided name."""

    match llm_name:
        case "llama":
            return llms["llama"]
        case "llama70b":
            return llms["llama70b"]
        case "gemini":
            return llms["gemini"]
        case _:
            raise ValueError(f"Unknown model name: {llm_name}")


def invoke_model(llm: BaseChatModel, prompts: list[BaseMessage]):
    """Invokes the provided model with the given prompt and returns the response."""

    model_name = getattr(llm, "model_name", None) or getattr(
        llm, "model", "unknown model"
    )
    print(f"\n--- Calling {model_name} ---")

    response: AIMessage = llm.invoke(prompts)
    return response


def main():
    load_environment_variables()


if __name__ == "__main__":
    main()
