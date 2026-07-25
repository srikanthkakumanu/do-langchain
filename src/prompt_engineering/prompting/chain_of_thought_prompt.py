"""
Chain-of-thought prompting (zero-shot CoT) using LangChain v1's unified chat
model interface (init_chat_model).

Contrasts a direct-answer prompt against the same question with "Let's think
step by step" appended, at the single-call level -- no parsing of the
reasoning, no looping. This is the smallest version of the technique; the
full runnable pattern (LCEL chain, answer extraction, few-shot exemplars,
and CoT's more expensive siblings Self-Consistency and Tree-of-Thought) lives
under src/prompt_engineering/patterns/CoT_Pattern.py.
"""

from langchain_core.messages import AIMessage, message_to_dict

from utils.llm_utils import get_model, invoke_model, load_environment_variables

QUESTION = "A store has 12 apples, sells 5, then receives 8 more. How many apples does it have?"

# Direct -- asks for the answer with no room for intermediate reasoning,
# which is exactly where a multi-step arithmetic problem tends to go wrong.
DIRECT_PROMPT = f"{QUESTION} Answer with just the number."

# Zero-shot CoT -- one extra clause invites the model to work through the
# steps before landing on a final answer.
COT_PROMPT = f"{QUESTION} Let's think step by step."


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
    print(f"\n=== {model_name}: direct prompt ===")
    _ask(model_name, DIRECT_PROMPT)

    print(f"\n=== {model_name}: chain-of-thought prompt ===")
    _ask(model_name, COT_PROMPT)


def llama():
    return _compare("llama")


def gemini():
    return _compare("gemini")


def openai():
    return _compare("openai")


def claude():
    return _compare("claude")


def main():
    """Main function to compare direct vs. chain-of-thought prompts across providers."""

    load_environment_variables()

    for fn in (llama, gemini, openai):
        fn()


if __name__ == "__main__":
    main()
