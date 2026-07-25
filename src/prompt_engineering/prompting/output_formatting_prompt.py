"""
Output formatting constraints using LangChain v1's unified chat model
interface (init_chat_model).

Asks the model, in plain prompt text, to return only valid JSON matching a
described shape -- no provider-enforced schema, just an explicit instruction
about the output format. Contrasts that against an open-ended version of the
same ask to show how much more the response format wanders without a
constraint. This is the "ask nicely" version of the technique; where the
provider supports it, prefer the schema-enforced version in
structured_output_prompt.py instead.
"""

from langchain_core.messages import AIMessage, message_to_dict

from utils.llm_utils import get_model, invoke_model, load_environment_variables

EMAIL_TEXT = """
From: priya.nair@vendor-example.com
Subject: Invoice #4521 -- payment overdue

Hi team, our records show invoice #4521 (due June 30) is still unpaid.
Please process payment or let us know of any discrepancy by end of week
to avoid a late fee.
"""

# Open-ended -- the model can answer in a paragraph, a bulleted list, or
# anything else it judges appropriate; nothing constrains the shape.
UNCONSTRAINED_PROMPT = f"What are the sender, subject, and whether action is required in this email?\n\nEmail:\n{EMAIL_TEXT}"

# Constrained -- an explicit schema and an "ONLY valid JSON" instruction push
# the model toward one parseable shape instead of free-form prose.
CONSTRAINED_PROMPT = f"""
Extract the following fields from the email below and return ONLY valid
JSON, no other text: {{"sender": str, "subject": str, "action_required": bool}}

Email:
{EMAIL_TEXT}
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
    print(f"\n=== {model_name}: unconstrained prompt ===")
    _ask(model_name, UNCONSTRAINED_PROMPT)

    print(f"\n=== {model_name}: constrained (JSON) prompt ===")
    _ask(model_name, CONSTRAINED_PROMPT)


def llama():
    return _compare("llama")


def gemini():
    return _compare("gemini")


def openai():
    return _compare("openai")


def claude():
    return _compare("claude")


def main():
    """Main function to compare unconstrained vs. JSON-constrained prompts across providers."""

    load_environment_variables()

    for fn in (llama, gemini, openai):
        fn()


if __name__ == "__main__":
    main()
