<!-- markdownlint-disable MD029 -- Learning Path uses one continuous 1-28 sequence across sections, by design. -->

# Overview

Explore the LangChain, LangGraph, and LangChain Protocol Python packages.

## Table of Contents

- [Docs](#docs)
- [Learning Path](#learning-path)
  1. [Setup](#1-setup)
  2. [Prompting Basics](#2-prompting-basics)
  3. [LCEL: Composing Chains](#3-lcel-composing-chains)
  4. [Structured Output](#4-structured-output)
  5. [Reasoning Patterns](#5-reasoning-patterns)
  6. [Context Engineering](#6-context-engineering)
  7. [Agents & Tools](#7-agents--tools)
  8. [Conversation Memory](#8-conversation-memory)
  9. [Securing Prompts](#9-securing-prompts)
- [Bonus: Notebooks](#bonus-notebooks)

## Docs

- [docs/LLM.md](docs/LLM.md) — Reference list of cheap/free LLM providers and models (Google, Groq, Mistral, OpenAI) with rate limits and pricing.
- [docs/Prompt_Engineering.md](docs/Prompt_Engineering.md) — Prompt anatomy, core prompting techniques (zero-shot, few-shot, role, structured output, etc.), and the agentic **Patterns** built from them (ReAct, Chain-of-Thought, Self-Consistency, Tree-of-Thought, injection-resistant system prompts, conversation memory), with a side-by-side comparison table.
- [docs/LCEL.md](docs/LCEL.md) — LangChain Expression Language: composing chains with the `|` operator, `Runnable` primitives (`RunnableLambda`, `RunnableParallel`, `RunnableBranch`, `RunnablePassthrough`), and common composition patterns.
- [docs/Context_Engineering.md](docs/Context_Engineering.md) — Getting external data into a model's context: document loaders (`TextLoader`, `WebBaseLoader`, `DirectoryLoader`, `PyPDFLoader`), text splitters (`CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `TokenTextSplitter`, `MarkdownHeaderTextSplitter`, language-aware splitting), and chunking strategy (`chunk_size`/`chunk_overlap` tuning) to produce good retrieval chunks.

## Learning Path

A suggested order to work through this repo, doc and code side by side, from raw model calls to full agents with tools, memory, and hardened prompts. Each step links the concept (doc section) to a runnable example (`src/...`).

### 1. Setup

Before writing any example, know which models are available and how the repo talks to them:

- **[docs/LLM.md](docs/LLM.md)** — pick a cheap/free provider and model to run the examples against.
- **[src/utils/llm_utils.py](src/utils/llm_utils.py)** — the shared, cached chat-model factory (`get_model`, `invoke_model`) that the prompt-engineering examples below reuse instead of each redefining their own client setup.

### 2. Prompting Basics

Start with single-call prompting — no chains, no agents, just shaping the input. See [docs/Prompt_Engineering.md](docs/Prompt_Engineering.md#core-prompt-engineering-techniques).

1. **Zero-shot prompting** — [src/prompt_engineering/zero_shot_prompt.py](src/prompt_engineering/zero_shot_prompt.py)
2. **Role / persona prompting** — [src/prompt_engineering/persona_prompt.py](src/prompt_engineering/persona_prompt.py)
3. **Multi-turn conversations** — [src/prompt_engineering/multi_turn_prompt.py](src/prompt_engineering/multi_turn_prompt.py)

### 3. LCEL: Composing Chains

Move from a single call to composed chains with the `|` operator. See [docs/LCEL.md](docs/LCEL.md).

4. **Basic sequential chain** (`prompt | model | parser`) — [src/lcel/sequential_chain.py](src/lcel/sequential_chain.py)
5. **Parallel & branching chains** (`RunnableParallel`, `RunnableBranch`) — [src/lcel/parallel_branching_chain.py](src/lcel/parallel_branching_chain.py)
6. **Custom steps with `RunnableLambda`** — [src/lcel/runnable_lambda_step.py](src/lcel/runnable_lambda_step.py)
7. **Passing extra context with `RunnablePassthrough`** — [src/context_engineering/runnable_passthrough_context.py](src/context_engineering/runnable_passthrough_context.py)
8. **Streaming and batching the same chain** — [src/lcel/streaming_and_batching.py](src/lcel/streaming_and_batching.py)

### 4. Structured Output

Turn free-form model output into typed, reliable data. See [docs/Prompt_Engineering.md § Structured Output & Parsing](docs/Prompt_Engineering.md#12-structured-output--parsing).

9. **Native structured output** (`with_structured_output`) — [src/lcel/structured_output/structured_output.py](src/lcel/structured_output/structured_output.py)
10. **Output parsers** (string, JSON, Pydantic, lists, XML, tool calls) — [src/lcel/structured_output/output_parsers.py](src/lcel/structured_output/output_parsers.py)

### 5. Reasoning Patterns

Single chains that get better answers out of one model call (or a few), before introducing agents/tools. See [docs/Prompt_Engineering.md § Patterns](docs/Prompt_Engineering.md#patterns).

11. **Chain-of-Thought** — [src/prompt_engineering/CoT_Pattern.py](src/prompt_engineering/CoT_Pattern.py)
12. **Self-Consistency** (vote across sampled CoT paths) — [src/prompt_engineering/SelfConsistency_Pattern.py](src/prompt_engineering/SelfConsistency_Pattern.py)
13. **Tree-of-Thought** (search over candidate reasoning paths) — [src/prompt_engineering/ToT_Pattern.py](src/prompt_engineering/ToT_Pattern.py)

### 6. Context Engineering

Get external data (files, web pages, PDFs) into a model's context before you need agents to act on it. See [docs/Context_Engineering.md](docs/Context_Engineering.md).

14. **Loading a local text file** — [src/context_engineering/loaders_chunking/TextLoader.py](src/context_engineering/loaders_chunking/TextLoader.py)
15. **Loading a web page** — [src/context_engineering/loaders_chunking/WebLoader.py](src/context_engineering/loaders_chunking/WebLoader.py)
16. **The `Document` object, `DirectoryLoader`, and `PyPDFLoader`** — [src/context_engineering/loaders_chunking/Document_Loaders.py](src/context_engineering/loaders_chunking/Document_Loaders.py)
17. **Splitters and chunking strategy** (`chunk_size`/`chunk_overlap`) — [src/context_engineering/loaders_chunking/chunking_splitters.py](src/context_engineering/loaders_chunking/chunking_splitters.py)

### 7. Agents & Tools

Move from chains to agents: a model that decides what to do next, optionally calling tools. See [docs/Prompt_Engineering.md § ReAct Pattern](docs/Prompt_Engineering.md#react-reasoning--acting-pattern).

18. **Your first agent** (no tools) — [src/agents/simple_agent.py](src/agents/simple_agent.py)
19. **Streaming an agent's output** — [src/agents/streaming_agent.py](src/agents/streaming_agent.py)
20. **Agent with a calculator tool** — [src/agent-tools/calc_tool_agent.py](src/agent-tools/calc_tool_agent.py)
21. **Agent with a web-search tool** — [src/agent-tools/web_search_tool_agent.py](src/agent-tools/web_search_tool_agent.py)
22. **ReAct pattern** (making the Thought → Action → Observation loop visible) — [src/prompt_engineering/ReAct_Pattern.py](src/prompt_engineering/ReAct_Pattern.py)
23. **Agent returning structured output** — [src/agents/structured_output_agent.py](src/agents/structured_output_agent.py)
24. **Multimodal input** (text, image, audio) — [src/agents/multimodal_messages_agent.py](src/agents/multimodal_messages_agent.py)
25. **Project: personal chef agent** (tool-using agent end to end) — [src/agents/personal-chef-agent/personal_chef_agent.py](src/agents/personal-chef-agent/personal_chef_agent.py)

### 8. Conversation Memory

Now that agents can act, give them memory across turns. See [docs/Prompt_Engineering.md § Conversation Memory Patterns](docs/Prompt_Engineering.md#conversation-memory-patterns).

26. **Conversational memory via a checkpointer** — [src/context_engineering/state_memory.py](src/context_engineering/state_memory.py)
27. **Buffer vs. Window vs. Summary memory** — [src/context_engineering/Memory_Pattern.py](src/context_engineering/Memory_Pattern.py)

### 9. Securing Prompts

The capstone: harden everything above against a user or a retrieved document trying to hijack the agent's instructions. See [docs/Prompt_Engineering.md § System Prompts & Injection-Resistant Design](docs/Prompt_Engineering.md#system-prompts--injection-resistant-design).

28. **Injection-resistant system prompt design** — [src/prompt_engineering/SystemPrompt_Pattern.py](src/prompt_engineering/SystemPrompt_Pattern.py)

## Bonus: Notebooks

Exploratory Jupyter notebooks, useful for interactive tinkering once you've been through the scripted examples above:

- [src/notebooks/foundational_models.ipynb](src/notebooks/foundational_models.ipynb) — initializing a model and generating structured output, interactively.
- [src/notebooks/multi_model_messages.ipynb](src/notebooks/multi_model_messages.ipynb) — multimodal messages (text/image/audio) with an interactive file upload widget.
