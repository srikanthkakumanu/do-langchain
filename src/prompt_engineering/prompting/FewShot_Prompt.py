"""
Few-shot prompting (in-context learning) using LangChain v1's unified chat
model interface (init_chat_model).

A handful of solved examples are given as Human/AI message pairs before the
real question, so the model infers the task pattern (format, reasoning style)
purely from context -- no fine-tuning, no explicit instructions needed.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, message_to_dict

from utils.llm_utils import get_model, invoke_model, load_environment_variables

SYSTEM_PROMPT = "You classify the sentiment of a sentence as Positive, Negative, or Neutral."

# Few-shot examples: each pair teaches the model the expected input/output
# format by demonstration rather than by description.
EXAMPLES = [
    (HumanMessage(content="The movie was a thrilling ride from start to finish."), "Positive"),
    (HumanMessage(content="The service at the restaurant was painfully slow."), "Negative"),
    (HumanMessage(content="The package arrived on Tuesday as scheduled."), "Neutral"),
]

QUESTION = "Honestly, I expected so much more from this phone for the price."


def build_messages() -> list:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for human_msg, label in EXAMPLES:
        messages.append(human_msg)
        messages.append(AIMessage(content=label))
    messages.append(HumanMessage(content=QUESTION))
    return messages


def print_model_response(response: AIMessage):
    """Prints the response from the model for the given prompt."""

    model_name = response.response_metadata.get("model_name", "unknown model")
    print(f"\n\n--- Response from {model_name}: ---")
    print(f"Content: {response.content}")
    print(f"Dict: {message_to_dict(response)}")


def _ask(model_name: str):
    llm = get_model(model_name)
    response = invoke_model(llm, build_messages())
    print_model_response(response)


def llama():
    return _ask("llama")


def gemini():
    return _ask("gemini")


def openai():
    return _ask("openai")


def claude():
    return _ask("claude")


def main():
    """Main function to invoke models with few-shot examples and print their responses."""

    load_environment_variables()

    for fn in (llama, gemini, openai):
        fn()


if __name__ == "__main__":
    main()
