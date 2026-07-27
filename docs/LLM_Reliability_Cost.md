# LLM Reliability & Cost

A hands-on tutorial on running LLM calls in a way that survives contact with production: **retries** (what to do when a call fails), **rate limiting** (how to avoid failing in the first place), **token counting** (knowing how big a request/response is), and **cost estimation** (turning tokens into dollars). Every example is cross-provider — the same four calls made against OpenAI, Groq, Google Gemini, and Anthropic Claude through **LangChain 1.3** (`langchain-core` 1.5, `langchain-openai` 1.2, `langchain-anthropic` 1.4, `langchain-groq` 1.1, `langchain-google-genai` 4.2, current as of this writing), so you can see where the four providers behave the same way and — more usefully — where they don't.

Examples are runnable from [src/llm_reliability/retries_and_backoff.py](../src/llm_reliability/retries_and_backoff.py), [src/llm_reliability/rate_limiting.py](../src/llm_reliability/rate_limiting.py), [src/llm_reliability/token_counting.py](../src/llm_reliability/token_counting.py), and [src/llm_reliability/cost_estimation.py](../src/llm_reliability/cost_estimation.py).

## Table of Contents

- [Why This Matters](#why-this-matters)
- [Retries & Backoff](#retries--backoff)
  - [Built-in retries](#built-in-retries)
  - [Custom retries with `tenacity`](#custom-retries-with-tenacity)
  - [Gotcha: Gemini collapses every error into one exception class](#gotcha-gemini-collapses-every-error-into-one-exception-class)
- [Rate Limiting](#rate-limiting)
  - [`InMemoryRateLimiter`](#inmemoryratelimiter)
  - [Picking a rate](#picking-a-rate)
- [Token Counting](#token-counting)
  - [Before a call: estimates](#before-a-call-estimates)
  - [Gotcha: Groq has no built-in tokenizer](#gotcha-groq-has-no-built-in-tokenizer)
  - [After a call: `usage_metadata`](#after-a-call-usage_metadata)
- [Cost Estimation](#cost-estimation)
  - [A pricing table](#a-pricing-table)
  - [Aggregating cost across providers with `get_usage_metadata_callback`](#aggregating-cost-across-providers-with-get_usage_metadata_callback)
- [Summary](#summary)

## Why This Matters

A single `llm.invoke(prompt)` call in a notebook always "just works" — until it's running in a loop, in production, against a free tier, or across a fleet of requests. At that point four questions become unavoidable:

1. **Retries** — the call failed. Was that transient (rate limit, momentary server error) or permanent (bad request, bad key)? Should this exact call be retried, and how many times?
2. **Rate limiting** — instead of reacting to failures, can the client avoid triggering them by pacing its own request rate?
3. **Token counting** — how large is this request, in the unit the provider actually bills and enforces context-window limits in?
4. **Cost estimation** — given the tokens actually used, what did this call (or this whole run) cost?

LangChain gives each of these a provider-agnostic primitive — `max_retries`, `InMemoryRateLimiter`, `get_num_tokens_from_messages`, `usage_metadata` — but "provider-agnostic API" does not mean "identical behavior underneath." The sections below cover both the shared API and the per-provider differences worth knowing about, several of which only show up once you actually run the calls (see the callouts below — they're all things this repo hit while writing these examples, not hypotheticals).

## Retries & Backoff

LLM calls fail transiently in normal operation: rate limits (HTTP 429), momentary server errors (5xx), and dropped connections are expected, not bugs. There are two layers to handle it.

### Built-in retries

Every LangChain chat model wraps a provider SDK, and each SDK already retries retryable errors with exponential backoff internally. Opt in with `max_retries` on the model constructor (or `init_chat_model(..., max_retries=N)`):

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("groq:llama-3.1-8b-instant", max_retries=3, timeout=30)
llm.invoke("...")  # retried internally on 429/5xx/connection errors
```

This is enough for almost all cases — the SDK already knows, from the real HTTP status code, whether a given failure is worth retrying, and picks its own backoff/jitter timing.

### Custom retries with `tenacity`

Reach for a hand-rolled retry loop when you need policy the built-in retry doesn't give you: logging every attempt, a shared backoff budget across a multi-step chain, or intentionally disabling the SDK's own retry (`max_retries=0`) so `tenacity` is the single source of retry behavior instead of two overlapping loops.

```python
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

@retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    wait=wait_random_exponential(multiplier=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def call_with_custom_retry(model: str, prompt: str) -> str:
    llm = init_chat_model(model, max_retries=0, timeout=30)
    return llm.invoke(prompt).content
```

The catch: each provider SDK raises its **own** exception classes — there is no single shared "LLM was rate limited" exception in LangChain. A cross-provider retry decorator needs a tuple of all of them:

| Provider  | Rate limit           | Connection error         | Server error            |
| --------- | --------------------- | -------------------------- | -------------------------- |
| OpenAI    | `openai.RateLimitError` | `openai.APIConnectionError` | `openai.InternalServerError` |
| Anthropic | `anthropic.RateLimitError` | `anthropic.APIConnectionError` | `anthropic.InternalServerError` |
| Groq      | `groq.RateLimitError` | `groq.APIConnectionError` | `groq.InternalServerError` |
| Gemini    | *(see below)* | *(see below)* | *(see below)* |

Deliberately **not** included: authentication/validation errors (401, 400). Retrying those just fails the same way N more times, slower.

### Gotcha: Gemini collapses every error into one exception class

The other three providers preserve the distinction between "retry this" and "don't bother" in the exception type. Gemini's LangChain integration (`langchain-google-genai`) does not: it catches the underlying `google.genai.errors.ClientError` (4xx) / `ServerError` (5xx) and re-raises **everything** — a 429 rate limit, a 400 bad request, an invalid API key — as one generic `langchain_google_genai.chat_models.ChatGoogleGenerativeAIError`.

```python
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

try:
    llm.invoke(prompt)
except ChatGoogleGenerativeAIError as e:
    ...  # could be a 429 worth retrying, or a 400 that never will be
```

That means a `retry_if_exception_type(ChatGoogleGenerativeAIError)` policy can waste retry attempts on a request that will never succeed — you've lost the information needed to discriminate by type. In practice this makes Gemini's own `max_retries` (which decides retryability from the real status code *before* it gets collapsed into one class) more load-bearing than it is for the other three providers, where a custom `tenacity` policy can safely be just as selective as the SDK's own.

This is not a hypothetical: `src/llm_reliability/retries_and_backoff.py` hits Gemini's free-tier daily quota (`RESOURCE_EXHAUSTED`, 20 requests/day for `gemini-2.5-flash`) routinely while testing, and both the built-in and `tenacity` paths need to catch `ChatGoogleGenerativeAIError` — not `google.genai.errors.ClientError` — to actually observe it.

## Rate Limiting

Retries react *after* a provider has already rejected a request. Rate limiting is the proactive counterpart: cap how fast requests leave your process so you rarely hit 429 in the first place. This matters most on free/low tiers with tight per-minute limits — see [docs/LLM.md](LLM.md) for each provider's current limits.

### `InMemoryRateLimiter`

LangChain ships a token-bucket limiter. It refills a bucket of "request tokens" at `requests_per_second`, and `.invoke()` blocks until a token is available. Attach it via the `rate_limiter=` constructor argument — every call through that model instance is throttled automatically, with no change to call sites:

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

limiter = InMemoryRateLimiter(
    requests_per_second=0.25,   # steady-state rate
    max_bucket_size=1,          # how many requests can burst above that rate
    check_every_n_seconds=0.1,  # how often to poll while waiting for a token
)
llm = init_chat_model("google_genai:gemini-2.5-flash", rate_limiter=limiter)
```

### Picking a rate

Convert the provider's published limit (requests per minute) to requests per second, and go slightly under it — `RPM / 60`, not `RPM / 60` exactly, if the limit is a hard cliff rather than a rolling average:

| Provider | Example free-tier limit (see [docs/LLM.md](LLM.md)) | `requests_per_second` |
| -------- | ---------------------------------------------------- | ---------------------- |
| Gemini (`gemini-2.5-flash`) | 15 RPM | `15 / 60 = 0.25` |
| Groq | generous RPM, capped requests/day | a light per-second cap avoids bursty client-side hammering |
| OpenAI / Anthropic | tier-dependent (check your account dashboard) | set to your tier's RPM / 60 |

`max_bucket_size` controls burst tolerance: `1` means strictly steady-state (no burst above the rate), a higher value lets a handful of requests fire back-to-back before throttling kicks in — useful if your traffic is naturally bursty rather than steady.

## Token Counting

There are two different reasons to count tokens, and they need different tools.

### Before a call: estimates

`model.get_num_tokens_from_messages(messages)` estimates size before spending money on a real request — useful for a context-window budget check or a rough cost estimate. *How* each provider's LangChain integration answers this differs:

| Provider | Mechanism | Network call? |
| -------- | --------- | -------------- |
| OpenAI | `tiktoken`, locally | No |
| Anthropic | Claude's official `messages.count_tokens` API | Yes (exact, not tiktoken-based — Claude's tokenizer isn't tiktoken-compatible) |
| Gemini | Google's real `count_tokens` API | Yes (exact — Gemini's tokenizer isn't tiktoken-compatible either) |
| Groq | *(see below)* | *(see below)* |

```python
llm = init_chat_model("anthropic:claude-opus-4-8")
llm.get_num_tokens_from_messages(messages)  # calls Claude's count_tokens API
```

### Gotcha: Groq has no built-in tokenizer

`ChatGroq` doesn't override `get_num_tokens_from_messages` at all, so it falls back to LangChain's generic default — a GPT-2 tokenizer that requires the `transformers` package. Without `transformers` installed (it isn't, by default, in this project — it's a heavy dependency for a token-count estimate), this raises `ImportError`:

```
ImportError: Could not import transformers python package. This is needed in
order to calculate get_token_ids. Please install it with `pip install transformers`.
```

`token_counting.py` handles this by catching the `ImportError` and falling back to a local `tiktoken` (`cl100k_base`) estimate instead:

```python
try:
    return llm.get_num_tokens_from_messages(messages)
except ImportError:
    text = "\n".join(str(m.content) for m in messages)
    return len(_FALLBACK_ENCODING.encode(text))  # cl100k_base, an approximation
```

`cl100k_base` is OpenAI's tokenizer, not Llama's — this narrows the estimate close enough for budgeting, but it is *not* an exact count. In one test run this produced a pre-call estimate of 17 tokens against an actual (provider-reported) input of 52 — a meaningful gap. This is exactly why the next section matters more than the estimate.

### After a call: `usage_metadata`

Every LangChain chat model normalizes actual usage onto `AIMessage.usage_metadata`, regardless of provider — this is the number that determines your bill, not the pre-call estimate:

```python
response = llm.invoke(messages)
response.usage_metadata
# {'input_tokens': 52, 'output_tokens': 100, 'total_tokens': 152}
```

Because the shape (`input_tokens` / `output_tokens` / `total_tokens`) is identical across OpenAI, Anthropic, Groq, and Gemini, this is the one part of this tutorial with no per-provider gotcha — read it the same way regardless of which model produced it.

## Cost Estimation

Cost = tokens × price-per-token, using the same `usage_metadata` from above.

### A pricing table

Keep a small `$/1M tokens` table per model (see [docs/LLM.md](LLM.md) for the full, regularly-updated comparison across providers):

```python
PRICING_PER_MILLION_TOKENS = {
    "gemini-2.5-flash": {"input": 0.10, "output": 0.40},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "llama-3.1-8b-instant": {"input": 0.00, "output": 0.00},  # free tier
}

def estimate_cost(model_name: str, usage: dict) -> float:
    prices = PRICING_PER_MILLION_TOKENS[model_name]
    return (
        usage.get("input_tokens", 0) / 1_000_000 * prices["input"]
        + usage.get("output_tokens", 0) / 1_000_000 * prices["output"]
    )
```

This is an **estimate from list pricing, not a bill.** It ignores prompt-cache discounts, provider-side rounding, and any negotiated/enterprise pricing — always reconcile against the provider's billing dashboard for real spend.

### Aggregating cost across providers with `get_usage_metadata_callback`

`langchain_core.callbacks.usage.get_usage_metadata_callback()` is a context manager, provider-agnostic, that sums `usage_metadata` across *any number* of calls to *any number* of models within a `with` block — keyed by model name. That turns "what did this whole run cost, across all four providers?" into one accumulator instead of four separate ones:

```python
from langchain_core.callbacks.usage import get_usage_metadata_callback

with get_usage_metadata_callback() as cb:
    for name, model in models.items():
        llm = init_chat_model(model, max_tokens=150)
        llm.invoke(question, config={"callbacks": [cb]})

    for model_name, usage in cb.usage_metadata.items():
        print(model_name, usage, "->", estimate_cost(model_name, usage))
```

`cb.usage_metadata` ends up as `{"gemini-2.5-flash": {...}, "llama-3.1-8b-instant": {...}, ...}` — one entry per distinct model actually invoked, summed if a model was called more than once.

## Summary

| Concern | LangChain primitive | Reacts or prevents? | Per-provider gotcha |
| ------- | -------------------- | --------------------- | ---------------------- |
| Retries | `max_retries=` / `tenacity` | Reacts, after a failed call | Gemini errors all collapse to `ChatGoogleGenerativeAIError` — can't discriminate retryable vs. not by type |
| Rate limiting | `InMemoryRateLimiter` (`rate_limiter=`) | Prevents, before a call is sent | None — works uniformly, but the *right rate* is provider- and tier-specific |
| Token counting | `get_num_tokens_from_messages()` (pre-call), `usage_metadata` (post-call) | N/A | Groq has no built-in tokenizer; falls back to an approximation |
| Cost estimation | pricing table + `get_usage_metadata_callback()` | N/A | None — `usage_metadata`'s shape is identical across all four providers |

Retries and rate limiting are two sides of the same problem (call failure), applied at different points in time; token counting and cost estimation are two sides of the same measurement (how much did this cost), taken before and after the fact. Together they're what turns "an LLM call that works in a notebook" into "an LLM call that survives a loop, a free tier, and a billing review."
