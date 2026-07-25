"""
Instruction clarity & task decomposition using LangChain v1's unified chat
model interface (init_chat_model).

Contrasts a vague, single-sentence ask with a specific one broken into
explicit, ordered sub-tasks, against the same source text. The technique
lives entirely in how the prompt is worded -- the call shape is identical
to zero-shot prompting.
"""

from langchain_core.messages import AIMessage, message_to_dict

from utils.llm_utils import get_model, invoke_model, load_environment_variables

CONTRACT_TEXT = """
This Consulting Agreement ("Agreement") is entered into as of March 1, 2026,
between Acme Robotics, Inc. ("Client") and Nova Systems LLC ("Consultant").
The initial term of this Agreement is twelve (12) months from the effective
date, and shall automatically renew for successive six (6) month periods
unless either party provides sixty (60) days' written notice of
non-renewal. Client may terminate this Agreement for convenience at any
time upon thirty (30) days' written notice. Consultant may terminate only
for Client's uncured material breach after fifteen (15) days' written
notice. Consultant shall not perform services for any direct competitor of
Client during the term of this Agreement.
"""

# Vague -- invites a different shape of answer every time it's run: a
# paragraph, a bullet list, a summary, whatever the model feels fits.
VAGUE_PROMPT = f"Tell me about this contract.\n\nContract:\n{CONTRACT_TEXT}"

# Specific and decomposed -- an explicit, ordered checklist forces the same
# answer shape on every run, regardless of provider.
DECOMPOSED_PROMPT = f"""
Review the contract below and answer, in this exact order:
1. Who are the parties to the agreement?
2. What is the effective date and term length?
3. List every termination clause, quoted verbatim.
4. Flag any clause that imposes an obligation on only one party.

Contract:
{CONTRACT_TEXT}
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
    print(f"\n=== {model_name}: vague prompt ===")
    _ask(model_name, VAGUE_PROMPT)

    print(f"\n=== {model_name}: decomposed prompt ===")
    _ask(model_name, DECOMPOSED_PROMPT)


def llama():
    return _compare("llama")


def gemini():
    return _compare("gemini")


def openai():
    return _compare("openai")


def claude():
    return _compare("claude")


def main():
    """Main function to compare vague vs. decomposed prompts across providers."""

    load_environment_variables()

    for fn in (llama, gemini, openai):
        fn()


if __name__ == "__main__":
    main()
