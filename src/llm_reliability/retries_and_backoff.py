"""
Retries and exponential backoff.

LLM calls fail transiently -- rate limits (429), momentary server errors
(5xx), and dropped connections are all *expected* in production, not bugs.
There are two layers to handle this:

  1. Built-in retries: every LangChain chat model wraps the provider's SDK,
     and each SDK already retries 429/5xx/connection errors with exponential
     backoff internally. You opt in with `max_retries=`. This is enough for
     almost all cases.

  2. Custom retries (tenacity): when you need policy the built-in retry
     doesn't give you -- e.g. logging every attempt, retrying only specific
     exception types, or backing off around a *chain* rather than a single
     call -- wrap the call yourself with `tenacity`.

Each provider's SDK raises its own exception classes (openai.RateLimitError,
anthropic.RateLimitError, groq.RateLimitError), so a cross-provider retry
decorator needs a tuple of all of them -- there is no single shared "LLM
rate limited" exception in LangChain. Gemini is the exception worth calling
out: `langchain-google-genai` catches the underlying `google.genai.errors`
(ClientError for 4xx, ServerError for 5xx) and re-raises everything --
429s, 400s, auth failures alike -- as one generic
`ChatGoogleGenerativeAIError`. You lose the ability to tell "retry this"
from "don't bother" by exception type alone, which is exactly why relying
on Gemini's own built-in `max_retries` (which decides retryability from the
real status code *before* it gets collapsed) matters more there than for
the other three providers.
"""

import anthropic
import groq
import openai
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

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

# Exceptions worth retrying: rate limits, transient server errors, and
# dropped connections. NOT included: authentication/validation errors
# (401, 400) -- retrying those just fails the same way N more times.
#
# ChatGoogleGenerativeAIError is the odd one out -- as explained above, it
# covers 429/5xx *and* permanent errors like bad requests or invalid keys
# in a single class, so retrying on it can waste attempts on a request that
# will never succeed. It's included here for a working example, but in
# production prefer Gemini's own `max_retries` over type-based retry logic.
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.InternalServerError,
    ChatGoogleGenerativeAIError,
)


def with_builtin_retries(model: str, max_retries: int = 3):
    """The recommended default: let the SDK's own retry loop handle it.

    max_retries controls how many times the underlying client re-sends a
    request that failed with a retryable status. Backoff timing (delay,
    jitter) is chosen by each provider's SDK, not by LangChain.
    """
    return init_chat_model(model, max_retries=max_retries, timeout=30)


@retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    wait=wait_random_exponential(multiplier=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def call_with_custom_retry(model: str, prompt: str) -> str:
    """A hand-rolled retry policy for when max_retries isn't enough control.

    Useful when you want to: log/alert on every retry, apply a shared
    backoff budget across a multi-step chain, or intentionally NOT retry
    at the SDK level (max_retries=0) so tenacity is the single source of
    retry behavior instead of two overlapping retry loops.
    """
    llm = init_chat_model(model, max_retries=0, timeout=30)
    return llm.invoke(prompt).content


def run(question: str = "In one sentence, why do LLM calls need retries?"):
    for name, model in models.items():
        print(f"\n=== {name} ({model}) -- built-in max_retries ===")
        try:
            llm = with_builtin_retries(model)
            print(llm.invoke(question).content)
        except RETRYABLE_ERRORS as exc:
            # Every attempt was exhausted -- surface the failure instead of
            # crashing the whole script, since a single provider being
            # rate-limited shouldn't stop the others from running.
            print(f"Gave up after retries: {exc}")

        print(f"\n=== {name} ({model}) -- tenacity custom retry ===")
        try:
            print(call_with_custom_retry(model, question))
        except RETRYABLE_ERRORS as exc:
            print(f"Gave up after retries: {exc}")


if __name__ == "__main__":
    run()
