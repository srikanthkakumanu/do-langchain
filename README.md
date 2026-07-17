# Overview

Explore the LangChain, LangGraph, and LangChain Protocol Python packages.

## Docs

- [docs/Loaders.md](docs/Loaders.md) — LangChain document loaders (`TextLoader`, `WebBaseLoader`, `DirectoryLoader`, `PyPDFLoader`) that read raw sources into `Document` objects.
- [docs/Splitters.md](docs/Splitters.md) — Text splitters (`CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `TokenTextSplitter`, `MarkdownHeaderTextSplitter`, language-aware splitting) that break documents into chunks.
- [docs/Chunking.md](docs/Chunking.md) — Comparing splitter strategies and tuning `chunk_size`/`chunk_overlap` to produce good retrieval chunks.
- [docs/LCEL.md](docs/LCEL.md) — LangChain Expression Language: composing chains with the `|` operator, `Runnable` primitives (`RunnableLambda`, `RunnableParallel`, `RunnableBranch`, `RunnablePassthrough`), and common composition patterns.
- [docs/Parsers.md](docs/Parsers.md) — Output parsers (`StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`, list/XML/tool-call parsers) that turn raw model output into usable shapes at the end of a chain.
- [docs/Patterns.md](docs/Patterns.md) — Reasoning and agent patterns (ReAct, Chain-of-Thought, Self-Consistency, Tree-of-Thought), injection-resistant system prompt design, and conversation memory strategies (Buffer/Window/Summary), with a side-by-side comparison table.
- [docs/LLM.md](docs/LLM.md) — Reference list of cheap/free LLM providers and models (Google, Groq, Mistral, OpenAI) with rate limits and pricing.
