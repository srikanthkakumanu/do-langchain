"""
Cost estimation.

Cost = tokens x price-per-token, and the tokens come from the same place as
token_counting.py: `AIMessage.usage_metadata`. This file adds two things on
top of that:

  1. A per-model $/1M-token pricing table (see docs/LLM.md for the full,
     regularly-updated comparison across providers).
  2. `get_usage_metadata_callback()` -- a LangChain callback, provider
     agnostic, that aggregates usage_metadata across *any number* of calls
     to *any number* of models within a `with` block. That's what makes it
     possible to answer "what did this whole run cost, across all four
     providers?" with one accumulator instead of four separate ones.

This is an ESTIMATE from list pricing, not a bill. It ignores prompt-cache
discounts, provider-side rounding, and any negotiated/enterprise pricing --
always reconcile against the provider's billing dashboard for real spend.
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks.usage import get_usage_metadata_callback

load_dotenv()

# provider:model strings understood by init_chat_model.
# Each provider needs its own API key in .env (GOOGLE_API_KEY, OPENAI_API_KEY,
# ANTHROPIC_API_KEY, GROQ_API_KEY).
models = {
    "gemini": "google_genai:gemini-2.5-flash",
    # "openai": "openai:gpt-5-nano",
    # "claude": "anthropic:claude-opus-4-8",
    "groq": "groq:llama-3.1-8b-instant",
}

# $ per 1M tokens. Figures for gemini-2.5-flash and gpt-5-nano are the paid
# tier rates from docs/LLM.md; claude-opus-4-8 pricing is Anthropic's
# published rate. Groq's llama models are free-tier only for this project
# (see docs/LLM.md) so they cost $0 here -- swap in Groq's paid per-token
# rate if you're on a paid Groq plan.
PRICING_PER_MILLION_TOKENS = {
    "gemini-2.5-flash": {"input": 0.10, "output": 0.40},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "llama-3.1-8b-instant": {"input": 0.00, "output": 0.00},
    "llama-3.3-70b-versatile": {"input": 0.00, "output": 0.00},
}


def estimate_cost(model_name: str, usage: dict) -> float:
    """usage is an AIMessage.usage_metadata dict:
    {"input_tokens": int, "output_tokens": int, "total_tokens": int}.
    """
    prices = PRICING_PER_MILLION_TOKENS.get(model_name)
    if prices is None:
        raise ValueError(f"No pricing entry for {model_name!r} -- add one above.")

    input_cost = usage.get("input_tokens", 0) / 1_000_000 * prices["input"]
    output_cost = usage.get("output_tokens", 0) / 1_000_000 * prices["output"]
    return input_cost + output_cost


def run(question: str = "Explain rate limiting in two sentences."):
    # One callback tracks usage across every model invoked inside the
    # `with` block -- keyed by model name, summed across repeated calls to
    # the same model.
    with get_usage_metadata_callback() as cb:
        for name, model in models.items():
            try:
                llm = init_chat_model(model, max_tokens=150)
                response = llm.invoke(question, config={"callbacks": [cb]})
                print(f"\n=== {name} ({model}) ===")
                print(response.content)
            except Exception as exc:  # noqa: BLE001 -- a rate-limited/quota-exhausted
                # provider shouldn't stop the other providers' demo from running.
                print(f"\nSkipping {name}: {exc}")

        print("\n=== Aggregated usage across the whole run ===")
        total_cost = 0.0
        for model_name, usage in cb.usage_metadata.items():
            cost = estimate_cost(model_name, usage)
            total_cost += cost
            print(f"{model_name}: {usage} -> ${cost:.6f}")

        print(f"\nEstimated total cost: ${total_cost:.6f}")


if __name__ == "__main__":
    run()
