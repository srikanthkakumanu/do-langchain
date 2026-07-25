"""
Prompt chaining using LangChain v1's unified chat model interface
(init_chat_model).

Breaks one sprawling ask (research this, analyze it, format it) into a
sequence of narrow calls, each with a single job, where each call's output
becomes the next call's input. Compares that chained result against a single
prompt asked to do all three steps at once, on the same source document.
"""

from langchain_core.messages import AIMessage, message_to_dict

from utils.llm_utils import get_model, invoke_model, load_environment_variables

DOCUMENT = """
Our spring lineup includes the Trailblazer Jacket at $129, the Summit Boot
at $89, and the Alpine Pack at $149. All three ship free on orders over
$75. The Trailblazer Jacket is available in three colors; the Summit Boot
and Alpine Pack are available in two.
"""

# Single sprawling prompt -- one call asked to extract, transform, and
# summarize at once, with no narrow focus at any step.
MONOLITHIC_PROMPT = f"""
From the document below, extract all product names and prices, convert them
into a markdown table, and then write a one-paragraph summary of that table.

Document:
{DOCUMENT}
"""


def print_model_response(response: AIMessage, label: str):
    """Prints the response from the model for the given prompt."""

    model_name = response.response_metadata.get("model_name", "unknown model")
    print(f"\n\n--- {label} from {model_name}: ---")
    print(f"Content: {response.content}")
    print(f"Dict: {message_to_dict(response)}")


def _llm_call(model_name: str, prompt: str) -> str:
    llm = get_model(model_name)
    response = invoke_model(llm, prompt)
    return response.content


def _chain(model_name: str):
    print(f"\n=== {model_name}: chained (extract -> transform -> summarize) ===")

    extracted = _llm_call(model_name, f"Extract all product names and prices from:\n{DOCUMENT}")
    print(f"\n--- Step 1 (extract) ---\n{extracted}")

    structured = _llm_call(model_name, f"Convert this into a markdown table:\n{extracted}")
    print(f"\n--- Step 2 (transform) ---\n{structured}")

    summary = _llm_call(model_name, f"Write a one-paragraph summary of this table:\n{structured}")
    print(f"\n--- Step 3 (summarize) ---\n{summary}")

    print(f"\n=== {model_name}: monolithic (single call, all three steps) ===")
    llm = get_model(model_name)
    response = invoke_model(llm, MONOLITHIC_PROMPT)
    print_model_response(response, "Monolithic result")


def llama():
    return _chain("llama")


def gemini():
    return _chain("gemini")


def openai():
    return _chain("openai")


def claude():
    return _chain("claude")


def main():
    """Main function to compare a chained pipeline vs. a single monolithic prompt across providers."""

    load_environment_variables()

    for fn in (llama, gemini, openai):
        fn()


if __name__ == "__main__":
    main()
