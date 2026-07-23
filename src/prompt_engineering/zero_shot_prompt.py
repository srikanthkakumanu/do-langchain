"""
Zero-shot chat using LangChain v1's unified chat model interface (init_chat_model).

A single instruction, no examples, no history: invokes the same prompt
against multiple providers and prints each response in a couple of
different formats (content, dict).
"""

from langchain_core.messages import AIMessage, message_to_dict

from utils.llm_utils import get_model, invoke_model, load_environment_variables

PROMPT = "What is the capital of France? Answer in one word."


def print_model_response(response: AIMessage):
    """Prints the response from the model for the given prompt."""

    model_name = response.response_metadata.get("model_name", "unknown model")
    print(f"\n\n--- Response from {model_name}: ---")
    print(f"Content: {response.content}")
    print(f"Dict: {message_to_dict(response)}")


def llama() -> AIMessage:
    return invoke_model(get_model("llama"), PROMPT)


def gemini() -> AIMessage:
    return invoke_model(get_model("gemini"), PROMPT)


def openai() -> AIMessage:
    return invoke_model(get_model("openai"), PROMPT)


def claude() -> AIMessage:
    return invoke_model(get_model("claude"), PROMPT)


def main():
    """Main function to invoke models and print their responses."""

    load_environment_variables()

    for fn in (llama, gemini, openai, claude):
        print_model_response(fn())


if __name__ == "__main__":
    main()
