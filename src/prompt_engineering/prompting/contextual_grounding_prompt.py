"""
Contextual grounding (retrieval-augmented prompting) using LangChain v1's
unified chat model interface (init_chat_model).

Supplies retrieved reference text directly in the prompt and instructs the
model to answer strictly from it, with an explicit fallback for when the
context doesn't contain the answer. Runs the same question against a context
that actually contains the answer and one that doesn't, to show that the
fallback instruction is what keeps the model from guessing on the miss case.
"""

from langchain_core.messages import AIMessage, message_to_dict

from utils.llm_utils import get_model, invoke_model, load_environment_variables

QUESTION = "What is the maximum number of retries configured for the chat client?"

# Contains the answer.
RELEVANT_CHUNK = """
The internal chat client wraps every provider call with automatic retries.
By default, max_retries is set to 3, and requests use exponential backoff
starting at 1 second between attempts.
"""

# Doesn't contain the answer -- a different, unrelated excerpt from the
# same knowledge base.
UNRELATED_CHUNK = """
Streaming responses are enabled by default. When streaming is on, partial
tokens are yielded to the caller as they arrive instead of waiting for the
full response to complete.
"""


def build_prompt(context: str) -> str:
    return f"""
Answer the question using ONLY the context below. If the answer isn't in the
context, say "I don't have enough information to answer that."

<context>
{context}
</context>

Question: {QUESTION}
"""


def print_model_response(response: AIMessage):
    """Prints the response from the model for the given prompt."""

    model_name = response.response_metadata.get("model_name", "unknown model")
    print(f"\n\n--- Response from {model_name}: ---")
    print(f"Content: {response.content}")
    print(f"Dict: {message_to_dict(response)}")


def _ask(model_name: str, user_prompt: str):
    llm = get_model(model_name)
    response = invoke_model(llm, user_prompt)
    print_model_response(response)


def _compare(model_name: str):
    print(f"\n=== {model_name}: context contains the answer ===")
    _ask(model_name, build_prompt(RELEVANT_CHUNK))

    print(f"\n=== {model_name}: context doesn't contain the answer ===")
    _ask(model_name, build_prompt(UNRELATED_CHUNK))


def llama():
    return _compare("llama")


def gemini():
    return _compare("gemini")


def openai():
    return _compare("openai")


def claude():
    return _compare("claude")


def main():
    """Main function to compare grounded answers against grounded refusals across providers."""

    load_environment_variables()

    for fn in (llama, gemini, openai):
        fn()


if __name__ == "__main__":
    main()
