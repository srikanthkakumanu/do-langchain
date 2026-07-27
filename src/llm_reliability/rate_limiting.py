"""
Client-side rate limiting.

Retries (see retries_and_backoff.py) react *after* a provider has already
rejected a request with 429. Rate limiting is the proactive counterpart: cap
how fast requests leave your process so you rarely hit 429 in the first
place. This matters most on free/low tiers with tight per-minute limits --
see docs/LLM.md for each provider's current limits.

LangChain ships a token-bucket limiter, `InMemoryRateLimiter`: it refills a
bucket of "request tokens" at `requests_per_second`, and `.invoke()` blocks
until a token is available. Attach it to any chat model via the
`rate_limiter=` constructor argument -- the model calls it automatically
before every request, so nothing about your call sites changes.
"""

import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

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

# requests_per_second is the steady-state rate; max_bucket_size lets a few
# requests burst above that rate before throttling kicks in.
# check_every_n_seconds controls how often the bucket is polled while
# waiting for a token -- lower is more precise timing, more CPU wakeups.
#
# Gemini's free tier caps gemini-2.5-flash at 15 requests/minute (see
# docs/LLM.md), so 15/60 = 0.25 req/s keeps a long-running loop under that
# ceiling instead of bursting 15 requests in the first second and stalling
# for the rest of the minute.
RATE_LIMITS = {
    "gemini": InMemoryRateLimiter(requests_per_second=0.25, max_bucket_size=1),
    "openai": InMemoryRateLimiter(requests_per_second=1, max_bucket_size=2),
    "claude": InMemoryRateLimiter(requests_per_second=1, max_bucket_size=2),
    # Groq's free tier is generous on RPM but caps requests/day -- a light
    # per-second cap here just avoids bursty client-side hammering.
    "groq": InMemoryRateLimiter(requests_per_second=2, max_bucket_size=4),
}


def build_rate_limited_model(name: str, model: str):
    return init_chat_model(model, rate_limiter=RATE_LIMITS[name], max_retries=2)


def run(question: str = "Reply with a single word.", calls_per_provider: int = 3):
    """Fire several calls per provider and print the wall-clock gap between
    them -- with the limiter attached, consecutive calls are visibly spaced
    out instead of firing back-to-back.
    """
    for name, model in models.items():
        print(f"\n=== {name} ({model}) -- rate limited ===")
        llm = build_rate_limited_model(name, model)

        try:
            last = time.monotonic()
            for i in range(calls_per_provider):
                llm.invoke(question)
                now = time.monotonic()
                print(f"call {i + 1}: {now - last:.2f}s since previous call")
                last = now
        except Exception as exc:  # noqa: BLE001 -- a rate-limited/quota-exhausted
            # provider shouldn't stop the other providers' demo from running.
            print(f"Skipping {name}: {exc}")


if __name__ == "__main__":
    run()
