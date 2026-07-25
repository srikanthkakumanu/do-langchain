"""
Self-critique / reflection prompting using LangChain v1's unified chat model
interface (init_chat_model).

After the model answers, a follow-up turn asks it to review its own draft
against a specific critique lens -- factual errors, unsupported claims,
contradictions with the source context -- before finalizing. Needs real
conversation history: the follow-up is appended to a growing message list,
the same list shape used in multi_turn_prompt.py, rather than repeated as a
fresh call shape per provider.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, message_to_dict

from utils.llm_utils import get_model, invoke_model, load_environment_variables

CONTEXT = """
The Apollo 11 mission landed on the Moon on July 20, 1969. Neil Armstrong
was the first person to step onto the lunar surface, followed by Buzz
Aldrin about 19 minutes later. Michael Collins remained in lunar orbit
aboard the command module and did not walk on the Moon.
"""

USER_PROMPT = f"""
Using the context below, write two sentences about who walked on the Moon
during the Apollo 11 mission.

<context>
{CONTEXT}
</context>
"""

# Specific about what to check -- a generic "double-check your work" catches
# less than naming the exact failure modes to look for.
FOLLOWUP_PROMPT = """
Review your previous answer for factual errors, unsupported claims, and
anything that contradicts the source context above. If you find issues,
provide a corrected answer. If it's already correct, restate it unchanged.
"""


def print_model_response(response: AIMessage, label: str):
    """Prints the response from the model for the given prompt."""

    model_name = response.response_metadata.get("model_name", "unknown model")
    print(f"\n\n--- {label} from {model_name}: ---")
    print(f"Content: {response.content}")
    print(f"Dict: {message_to_dict(response)}")


def _critique(model_name: str):
    llm = get_model(model_name)
    history: list[BaseMessage] = [HumanMessage(content=USER_PROMPT)]

    draft = invoke_model(llm, history)
    print_model_response(draft, "Draft answer")

    history.extend([draft, HumanMessage(content=FOLLOWUP_PROMPT)])
    critique = invoke_model(llm, history)
    print_model_response(critique, "Critique / final answer")


def llama():
    return _critique("llama")


def gemini():
    return _critique("gemini")


def openai():
    return _critique("openai")


def claude():
    return _critique("claude")


def main():
    """Main function to run the draft-then-critique conversation across providers."""

    load_environment_variables()

    for fn in (llama, gemini, openai):
        fn()


if __name__ == "__main__":
    main()
