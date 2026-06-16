from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain.messages import AIMessage
from langchain_core.messages import message_to_dict

"""
Simple chat application using LangChain using new chat model interface. 
This example demonstrates how to initialize and invoke multiple 
chat models (LLaMA, Gemini, OpenAI) and print their responses in 
various formats (content, JSON, dict). 
The code is structured to allow easy addition of new models and 
response formats in the future.
"""

load_dotenv()

prompt = "What is the capital of France? Answer in one word."

llms = {
    "llama": init_chat_model(
        model="llama-3.1-8b-instant",
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

    if llm_name == "llama":
        return llms["llama"]
    elif llm_name == "gemini":
        return llms["gemini"]
    # elif llm_name == "openai":
    #     return llms["openai"]
    else:
        raise ValueError(f"Unknown model name: {llm_name}")


def invoke_model(llm: BaseChatModel, prompt: str):
    """Invokes the provided model with the given prompt and returns the response."""

    model_name = getattr(llm, "model_name", None) or llm.model
    print(f"\n--- Calling {model_name} ---")
    response: AIMessage = llm.invoke(prompt)
    return response


def print_model_response(response: AIMessage):
    """Prints the response from the model for the given prompt."""

    # response, content, to_json, to_dict
    print(f"\n\n--- Response from {response.response_metadata['model_name']}: ---")
    print(f"Metadata: {response.response_metadata}")
    print(f"Content: {response.content}")
    print(f"JSON: {response.to_json()}")
    print(f"Dict: {message_to_dict(response)}")


def main():
    """Main function to invoke models and print their responses."""
    for llm_name in llms.keys():
        llm = get_model(llm_name)
        response = invoke_model(llm, prompt)
        print_model_response(response)


if __name__ == "__main__":
    main()
