# LCEL Chains and Patterns

**LCEL (LangChain Expression Language)** is a declarative way to compose chains using the `|` (pipe) operator, similar to Unix pipes.

## Core ideas

- **Composability** — chain components together: `prompt | model | output_parser`. Each piece implements the common `Runnable` interface.
- **Uniform interface** — prompts, models, retrievers, and parsers all expose the same methods (`invoke`, `batch`, `stream`, `ainvoke`, `abatch`, `astream`), so they snap together regardless of type.
- **Built-in execution modes** — any chain built with LCEL automatically supports sync/async invocation, batching, and streaming with no extra code.
- **Parallelism** — `RunnableParallel` (or a plain dict literal) runs multiple branches concurrently and merges their results.
- **Observability** — the explicit graph structure plugs directly into LangSmith tracing.

## Composition primitives

- `RunnableLambda` — wrap a plain Python function as a `Runnable`.
- `RunnablePassthrough` — pass input through unchanged, optionally adding keys.
- `RunnableParallel` — run multiple runnables concurrently, merge results.
- `RunnableBranch` — conditional routing between runnables.

## Example

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "bears"})
```

## Composition Patterns

- Basic sequential chains (`prompt | model | parser`)
- Parallel/branching chains (`RunnableParallel`, `RunnableBranch`)
- Custom steps with `RunnableLambda`
- Passing extra context through with `RunnablePassthrough`
- Streaming and batching the same chain
- Structured output parsing

**Basic sequential chain** — runs each step one after another, passing the output of one step into the next. Use it for simple linear flows like prompt, model, then parser.

  ```python
  chain = prompt | model | StrOutputParser()
  result = chain.invoke({"topic": "LangChain"})
  ```

**Parallel chain** — runs multiple independent branches with the same input and returns their outputs together. Use it when one input should produce several results.

  ```python
  chain = {
      "summary": summary_chain,
      "keywords": keyword_chain,
  }
  result = chain.invoke({"text": document})
  ```

**Branching chain** — chooses one chain from multiple options based on a condition. Use it when different inputs need different handling.

  ```python
  from langchain_core.runnables import RunnableBranch

  chain = RunnableBranch(
      (lambda x: x["type"] == "math", math_chain),
      general_chain,
  )
  result = chain.invoke({"type": "math", "question": "2 + 2"})
  ```

**Custom step chain** — adds normal Python logic into an LCEL flow with `RunnableLambda`. Use it for small transformations, cleanup, or formatting.

  ```python
  from langchain_core.runnables import RunnableLambda

  clean_input = RunnableLambda(lambda text: text.strip().lower())
  chain = clean_input | prompt | model
  result = chain.invoke("  Explain LCEL  ")
  ```

**Passthrough chain** — keeps the original input while adding extra computed values. Use it when a prompt needs both the user's input and additional context.

  ```python
  from langchain_core.runnables import RunnablePassthrough

  chain = RunnablePassthrough.assign(context=retriever) | prompt | model
  result = chain.invoke({"question": "What is LCEL?"})
  ```

**Streaming chain** — emits output chunks as they are produced. Use it for chat interfaces or long responses where the user should see progress immediately.

  ```python
  for chunk in chain.stream({"topic": "LCEL"}):
      print(chunk, end="")
  ```

**Batching chain** — runs the same chain across many inputs at once. Use it for processing multiple prompts, documents, or records efficiently.

  ```python
  results = chain.batch([
      {"topic": "LCEL"},
      {"topic": "Runnables"},
  ])
  ```

**Structured output chain** — parses model output into a predictable format, such as plain text, JSON, or a Pydantic object. Use it when later code needs reliable fields instead of free-form text.

  ```python
  from langchain_core.output_parsers import JsonOutputParser

  chain = prompt | model | JsonOutputParser()
  result = chain.invoke({"topic": "LCEL"})
  ```
