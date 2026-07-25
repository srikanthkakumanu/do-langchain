"""
Iterative refinement (prompt testing & versioning) using LangChain v1's
unified chat model interface (init_chat_model).

This is a process, not an API shape: run the same set of representative test
inputs -- including known-tricky edge cases -- through a prompt, change one
variable at a time, and diff the outputs to see whether the wording change
actually helped. Below, two versions of a "summarize" prompt (vague vs. one
that pins down length and audience) are run against the same small test set
so the difference in output is directly comparable.
"""

from utils.llm_utils import get_model, invoke_model, load_environment_variables

# A small, representative test set: an ordinary case, a very short input
# (edge case: little to compress), and a technical input (edge case: jargon
# that a "for a general audience" instruction should actually change).
TEST_INPUTS = {
    "ordinary": """
        The city council voted 6-2 Tuesday to approve funding for a new
        public library branch in the Riverside neighborhood, with
        construction expected to begin next spring and take about 18 months.
    """,
    "very_short": "The meeting was postponed to Friday.",
    "technical": """
        The service exhibited elevated p99 latency after the deploy due to a
        connection pool exhaustion under load; a rollback restored nominal
        throughput within four minutes.
    """,
}

# v1 -- vague about length and audience, so the model's judgment fills in
# both, and that judgment can vary run to run.
PROMPT_V1 = "Summarize this: {text}"

# v2 -- one variable changed at a time from v1: an explicit sentence count
# and an explicit audience, so the output shape stops depending on the
# model's guess.
PROMPT_V2 = "Summarize this in exactly two sentences for a general audience with no prior context: {text}"


def _run_test_set(model_name: str, prompt_template: str, label: str):
    llm = get_model(model_name)
    print(f"\n=== {model_name}: {label} ===")

    for case_name, text in TEST_INPUTS.items():
        response = invoke_model(llm, prompt_template.format(text=text))
        print(f"\n--- Case: {case_name} ---\n{response.content}")


def _compare(model_name: str):
    _run_test_set(model_name, PROMPT_V1, "prompt v1 (vague)")
    _run_test_set(model_name, PROMPT_V2, "prompt v2 (length + audience pinned down)")


def llama():
    return _compare("llama")


def gemini():
    return _compare("gemini")


def openai():
    return _compare("openai")


def claude():
    return _compare("claude")


def main():
    """Main function to run both prompt versions against the same test set across providers."""

    load_environment_variables()

    for fn in (llama, gemini, openai):
        fn()


if __name__ == "__main__":
    main()
