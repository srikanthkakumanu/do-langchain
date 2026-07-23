"""
Multi-turn chat using LangChain v1's unified chat model interface (init_chat_model).

Builds a growing message history (system prompt, user turn, follow-up turn
that depends on the first answer) and replays it against multiple providers.
"""

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    message_to_dict,
)

from utils.llm_utils import get_model, invoke_model, load_environment_variables

SYSTEM_PROMPT = "You are a helpful assistant that provides concise and accurate answers to user queries."
HUMAN_PROMPT = "What is the capital of France?"
FOLLOW_UP_PROMPT = "What country is it in?"


def build_prompts(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Prepends the shared system prompt to a message history."""

    return [SystemMessage(content=SYSTEM_PROMPT), *messages]


def print_model_response(response: AIMessage):
    """Prints the response from the model for the given prompt."""

    model_name = response.response_metadata.get("model_name", "unknown model")
    print(f"\n\n--- Response from {model_name}: ---")
    print(f"Content: {response.content}")
    print(f"Dict: {message_to_dict(response)}")


def _converse(model_name: str):
    llm = get_model(model_name)
    messages: list[BaseMessage] = [HumanMessage(content=HUMAN_PROMPT)]

    response = invoke_model(llm, build_prompts(messages))
    print_model_response(response)

    messages.extend([response, HumanMessage(content=FOLLOW_UP_PROMPT)])
    response = invoke_model(llm, build_prompts(messages))
    print_model_response(response)


def llama():
    return _converse("llama")


def gemini():
    return _converse("gemini")


def openai():
    return _converse("openai")


def claude():
    return _converse("claude")


def main():
    """Main function to invoke models, run multi-shot conversations, and print their responses."""

    load_environment_variables()

    for fn in (llama, gemini, openai, claude):
        fn()


if __name__ == "__main__":
    main()
