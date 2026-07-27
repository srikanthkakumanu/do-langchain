"""
Token counting.

There are two different reasons to count tokens, and they need different
tools:

  1. BEFORE a call -- estimate cost or check you're under a context-window
     limit. `model.get_num_tokens_from_messages()` answers this without
     spending money on a real request, but *how* each provider's LangChain
     integration answers it differs, which matters if you rely on the exact
     mechanism:
       - OpenAI: counts locally with `tiktoken` -- no network call.
       - Gemini: calls Google's real `count_tokens` API -- an extra network
         call, but exact (Gemini's tokenizer isn't tiktoken-compatible).
       - Anthropic: calls Claude's official `messages.count_tokens` API --
         same tradeoff, exact rather than estimated.
       - Groq: has no tokenizer of its own wired into LangChain. The base
         class falls back to a GPT-2 tokenizer that requires the
         `transformers` package (not installed here), so it raises
         `ImportError`. `estimate_before_call` below catches that and
         falls back to a local `tiktoken` estimate instead -- close enough
         for budgeting, but not exact for Llama's own tokenizer.

  2. AFTER a call -- get the *actual* billed token counts. Every LangChain
     chat model normalizes this onto `AIMessage.usage_metadata`
     (`input_tokens`, `output_tokens`, `total_tokens`), regardless of
     provider. This is the number that determines your bill, not the
     pre-call estimate -- see cost_estimation.py.
"""

import tiktoken
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Fallback encoding for providers LangChain has no built-in tokenizer for
# (currently: Groq). cl100k_base is OpenAI's encoding, not Llama's -- this
# is an approximation, not an exact count.
_FALLBACK_ENCODING = tiktoken.get_encoding("cl100k_base")

# provider:model strings understood by init_chat_model.
# Each provider needs its own API key in .env (GOOGLE_API_KEY, OPENAI_API_KEY,
# ANTHROPIC_API_KEY, GROQ_API_KEY).
models = {
    "gemini": "google_genai:gemini-2.5-flash",
    # "openai": "openai:gpt-5-nano",
    # "claude": "anthropic:claude-opus-4-8",
    "groq": "groq:llama-3.1-8b-instant",
}

messages = [
    SystemMessage(content="You are a concise assistant."),
    HumanMessage(content="Name three uses of a token bucket rate limiter."),
]


def estimate_before_call(model: str) -> int:
    """Pre-call estimate, using each provider's most accurate mechanism --
    see the module docstring for what actually happens per provider.
    """
    llm = init_chat_model(model)
    try:
        return llm.get_num_tokens_from_messages(messages)
    except ImportError:
        # Groq today: no LangChain-side tokenizer, and no `transformers`
        # package installed for the generic fallback. Approximate instead
        # of failing the whole estimate.
        text = "\n".join(str(m.content) for m in messages)
        return len(_FALLBACK_ENCODING.encode(text))


def actual_after_call(model: str) -> dict:
    """The provider-reported truth, read off the response after invoking."""
    llm = init_chat_model(model, max_tokens=100)
    response = llm.invoke(messages)
    return response.usage_metadata


def run():
    for name, model in models.items():
        print(f"\n=== {name} ({model}) ===")

        try:
            estimate = estimate_before_call(model)
            print(f"Pre-call estimate (input only): {estimate} tokens")

            usage = actual_after_call(model)
            print(f"Actual usage after the call: {usage}")

            if usage and usage.get("input_tokens") is not None:
                drift = usage["input_tokens"] - estimate
                print(f"Estimate vs. actual input tokens: {drift:+d}")
        except Exception as exc:  # noqa: BLE001 -- a rate-limited/quota-exhausted
            # provider shouldn't stop the other providers' demo from running.
            print(f"Skipping {name}: {exc}")


if __name__ == "__main__":
    run()
