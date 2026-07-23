"""
Role / persona prompting using LangChain v1's unified chat model interface (init_chat_model).

Assigns the model a persona and a couple of few-shot example turns via the
system prompt, then asks it a new question in that voice, across multiple
providers.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, message_to_dict

from utils.llm_utils import get_model, invoke_model, load_environment_variables

SYSTEM_PROMPT = """You are an expert in Cosmology, answering questions at the user's request
about the universe.

Please keep to the below structure.

User: What is the nearest star to Earth?
Cosmologist: The nearest star to Earth is the Sun.

User: How old is this Universe?
Cosmologist: This Universe is approximately 13.4 billion years old.
"""

QUESTION = "What is the oldest galaxy in the universe?"


def print_model_response(response: AIMessage):
    """Prints the response from the model for the given prompt."""

    model_name = response.response_metadata.get("model_name", "unknown model")
    print(f"\n\n--- Response from {model_name}: ---")
    print(f"Content: {response.content}")
    print(f"Dict: {message_to_dict(response)}")


def _ask(model_name: str):
    llm = get_model(model_name)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=QUESTION)]

    response = invoke_model(llm, messages)
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
    """Main function to invoke models with a persona prompt and print their responses."""

    load_environment_variables()

    for fn in (llama, gemini, openai, claude):
        fn()


if __name__ == "__main__":
    main()
