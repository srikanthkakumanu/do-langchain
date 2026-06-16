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


"""
Simple chat application using LangChain using new chat model interface. 
This example demonstrates how to initialize multi-shot conversations
and invoke multiple chat models (LLaMA, Gemini, OpenAI) and print 
their responses in various formats (content, JSON, dict). 
The code is structured to allow easy addition of new models and 
response formats in the future.
"""

load_dotenv()

sys_prompt = "You are a helpful assistant that provides concise and accurate answers to user queries."
human_prompt = "What is the capital of France?"
follow_up_prompt = "What country is it in?"

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
    else:
        raise ValueError(f"Unknown model name: {llm_name}")


def invoke_model(llm: BaseChatModel, prompts: list[BaseMessage]):
    """Invokes the provided model with the given prompt and returns the response."""

    model_name = getattr(llm, "model_name", None) or getattr(
        llm, "model", "unknown model"
    )
    print(f"\n--- Calling {model_name} ---")

    response: AIMessage = llm.invoke(prompts)
    return response


def build_prompts(messages: list[BaseMessage]):
    """Builds the prompts for the model invocation."""

    return [SystemMessage(content=sys_prompt), *messages]


def print_model_response(response: AIMessage):
    """Prints the response from the model for the given prompt."""

    # response, content, to_json, to_dict
    model_name = response.response_metadata.get("model_name", "unknown model")
    print(f"\n\n--- Response from {model_name}: ---")
    print(f"Content: {response.content}")
    # print(f"Metadata: {response.response_metadata}")
    # print(f"JSON: {response.to_json()}")
    print(f"Dict: {message_to_dict(response)}")


def main():
    """Main function to invoke models, multi-shot conversations and print their responses."""
    for llm_name in llms.keys():
        llm = get_model(llm_name)
        messages: list[BaseMessage] = [HumanMessage(content=human_prompt)]
        response = invoke_model(llm, build_prompts(messages))
        print_model_response(response)

        messages.extend(
            [
                response,
                HumanMessage(content=follow_up_prompt),
            ]
        )
        response = invoke_model(llm, build_prompts(messages))
        print_model_response(response)


if __name__ == "__main__":
    main()
