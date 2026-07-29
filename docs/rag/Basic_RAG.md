# Basic RAG (Naive RAG)

Basic RAG is the smallest useful Retrieval-Augmented Generation system:

```text
load -> split -> embed -> store -> retrieve -> augment -> generate
```

![Basic RAG workflow](images/basic-rag-workflow.png)

It is called "naive" because it does the direct version of RAG: retrieve the top matching chunks once, put them into the prompt, and generate one answer. There is no query rewriting, reranking, routing, graph traversal, or agent loop yet.

The runnable companion example is [basic_rag.py](../../src/context_engineering/rags/basic_rag.py).

## What You Will Build

The example answers a question from a local text file:

- Source document: [sample.txt](../../src/context_engineering/resources/sample.txt)
- Python example: [basic_rag.py](../../src/context_engineering/rags/basic_rag.py)
- Vector store: `InMemoryVectorStore` by default, with an optional Chroma mode
- Retriever: `vector_store.as_retriever(search_kwargs={"k": 3})`
- Generator: configured repo LLM using API keys from `.env`, with a deterministic local fallback if the call fails

The example intentionally uses a tiny local `HashingEmbeddings` class. That keeps the retrieval workflow runnable without downloading embedding models or needing an embeddings API key. It is good for learning the pipeline, not for production semantic search quality.

## Basic RAG Workflow

| Stage | Function in `basic_rag.py` | What it teaches |
| --- | --- | --- |
| Load | `load_documents()` | RAG starts with source text wrapped in `Document` objects |
| Split | `split_documents()` | Long documents become smaller retrievable chunks |
| Embed | `create_embeddings()` | Chunks and questions must use the same embedding model |
| Store | `create_vector_store()` | Vectors, text, and metadata are indexed together |
| Retrieve | `retrieve_documents()` | The question pulls back the top matching chunks |
| Augment | `format_documents()` + `build_prompt()` | Retrieved text becomes prompt context |
| Generate | `generate_answer()` | The model answers from the retrieved context |

## Run It

From the repo root:

```bash
uv run python src/context_engineering/rags/basic_rag.py
```

Expected shape of the output:

```text
Loaded documents: 1
Created chunks: 4
Retrieved chunks: 3

Retrieved chunk 1:
Metadata: {...}
Preview: ...

Final answer:
LangChain provides building blocks for prompt templates, chat models, output parsers,
document loaders, retrievers, and agents.
```

You can ask a different question without editing the file:

```bash
BASIC_RAG_QUESTION="What are document loaders?" uv run python src/context_engineering/rags/basic_rag.py
```

You can choose any model alias supported by [llm_utils.py](../../src/utils/llm_utils.py):

```bash
BASIC_RAG_MODEL=gemini uv run python src/context_engineering/rags/basic_rag.py
BASIC_RAG_MODEL=openai uv run python src/context_engineering/rags/basic_rag.py
BASIC_RAG_MODEL=llama70b uv run python src/context_engineering/rags/basic_rag.py
```

You can also switch the vector store implementation with an environment variable:

```bash
BASIC_RAG_VECTOR_STORE=memory uv run python src/context_engineering/rags/basic_rag.py
```

`memory` is the default because it works with the existing project dependencies. Use Chroma only when you want to demonstrate a persistent local vector database.

## Stage 1: Load

Loading converts a raw source into one or more LangChain `Document` objects.

```python
def load_documents(source_path: Path = DEFAULT_SOURCE_PATH) -> list[Document]:
    text = source_path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={"source": str(source_path), "loader": "Path.read_text"},
        )
    ]
```

A `Document` has two important parts:

| Field | Meaning |
| --- | --- |
| `page_content` | The text that can later be embedded and retrieved |
| `metadata` | Source details such as file path, page, URL, title, or section |

Keep metadata early. It lets you inspect answers later and build citations or filters.

## Stage 2: Split

LLMs and retrievers work better with chunks than with huge documents. The example uses `RecursiveCharacterTextSplitter`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(documents)
```

Why this matters:

- Small chunks are easier to retrieve precisely.
- Some overlap prevents important context from being cut at boundaries.
- Metadata from the original document is copied onto each chunk.

Start with a chunk size that fits the shape of your source. For short prose examples, `250` characters is easy to inspect. For real documents, you will usually tune larger chunks and evaluate retrieval quality.

## Stage 3: Embed

Embedding turns text into vectors.

The example creates one embedding model and uses it for both indexing and querying:

```python
embeddings = create_embeddings()
```

That rule is non-negotiable: **the query and stored chunks must be embedded by the same model**. If you embed documents with one model and questions with another, the vectors no longer live in the same space.

The tutorial example uses:

```python
class HashingEmbeddings(Embeddings):
    ...
```

This class is only a local learning aid. In a real RAG app, replace it with a semantic embedding model such as OpenAI, Gemini, Voyage, or a local HuggingFace sentence-transformer.

## Stage 4: Store

The vector store saves chunks and their vectors:

```python
vector_store = create_vector_store(
    chunks,
    embeddings,
    store_type="memory",
)
```

`InMemoryVectorStore` is perfect for learning because there is no database to install. It disappears when the process exits, so it is not the right choice for data that must persist.

![Basic RAG vector store options](images/basic-rag-vector-store-options.png)

Use a persistent vector store when:

- your corpus is larger than a toy example
- indexing takes noticeable time
- the app must survive restarts
- multiple users or processes need the same index

### Optional: Use Chroma

Chroma is a local vector database. It is useful when you want the example to persist vectors on disk instead of rebuilding an in-memory index every run.

Install the optional packages:

```bash
uv add langchain-chroma chromadb
```

Then run:

```bash
BASIC_RAG_VECTOR_STORE=chroma uv run python src/context_engineering/rags/basic_rag.py
```

The example writes the local Chroma collection under:

```text
.chroma/basic_rag
```

The code path is:

```python
def create_chroma_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    *,
    persist_directory: Path = DEFAULT_CHROMA_PATH,
    collection_name: str = "basic_rag_demo",
):
    from langchain_chroma import Chroma

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    vector_store.add_documents(chunks, ids=ids)
    return vector_store
```

The same retriever code works for both stores:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

## Stage 5: Retrieve

A retriever wraps the vector store in a chain-friendly interface:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retrieved_documents = retriever.invoke(question)
```

`k` controls how many chunks come back.

| `k` value | Effect |
| --- | --- |
| Too small | The answer chunk may be missed |
| Reasonable | Enough evidence without clutter |
| Too large | Irrelevant chunks can distract the model |

The most useful debugging habit in RAG is to print retrieved chunks before generation. If the answer is not present in those chunks, you have a retrieval problem, not a prompt problem.

![Basic RAG debug loop](images/basic-rag-debug-loop.png)

## Stage 6: Augment

Augmentation means inserting retrieved context into the prompt:

```python
def format_documents(documents: list[Document]) -> str:
    formatted = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown source")
        formatted.append(f"[Chunk {index} | source={source}]\n{document.page_content}")
    return "\n\n".join(formatted)
```

The prompt then receives both `context` and `question`:

```python
prompt = ChatPromptTemplate.from_template(
    "Use only the context below. If the context does not contain the answer, "
    "say you do not know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
```

The key instruction is "use only the context." Without that, the model may blend retrieved evidence with training-time memory.

## Stage 7: Generate

The example calls the repo's configured chat model. The default is `llama70b`, and you can override it with `BASIC_RAG_MODEL`:

```python
model = get_model(model_name)
chain = prompt | model | StrOutputParser()
answer = chain.invoke({"context": context, "question": question})
```

Because this repo loads API keys from `.env`, generation will use the real provider as long as the selected model's key is present. If the key is missing or the model call fails, the script falls back to `generate_extractive_fallback()`. That fallback selects relevant sentences from the retrieved chunks. It is not a replacement for an LLM, but it keeps the learning example runnable while still demonstrating the full workflow.

## Read the Whole Flow

The `basic_rag()` method ties every stage together:

```python
def basic_rag(question: str, *, source_path: Path = DEFAULT_SOURCE_PATH, k: int = 3):
    documents = load_documents(source_path)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    retriever = create_retriever(vector_store, k=k)
    retrieved_documents = retrieve_documents(retriever, question)
    answer = generate_answer(question, retrieved_documents)
    ...
```

This is the simplest mental model for Basic RAG:

```text
documents become chunks
chunks become vectors
vectors become a searchable index
question becomes a query vector
retriever returns matching chunks
prompt combines chunks + question
model generates a grounded answer
```

## What to Change First

Tune one thing at a time:

| Change | Why try it |
| --- | --- |
| `BASIC_RAG_QUESTION` | Test different retrieval behavior |
| `chunk_size` | Control how much context each chunk carries |
| `chunk_overlap` | Preserve context across chunk boundaries |
| `k` | Return more or fewer chunks |
| `BASIC_RAG_MODEL` | Switch the chat model used for final generation |
| `BASIC_RAG_VECTOR_STORE` | Switch between in-memory and Chroma-backed retrieval |
| Embedding model | Improve semantic retrieval quality |
| Prompt wording | Improve answer faithfulness once retrieval is good |

Do not start by changing the LLM. First verify whether the right chunks are retrieved.

## Common Beginner Mistakes

| Mistake | Symptom | Fix |
| --- | --- | --- |
| Not printing retrieved chunks | You cannot tell why the answer is wrong | Inspect `retrieved_documents` before generation |
| Chunks are too small | Answers lack context | Increase `chunk_size` or overlap |
| Chunks are too large | Retrieval returns broad, noisy context | Decrease `chunk_size` |
| Using a different embedding model for queries | Retrieval quality collapses | Re-index documents with the same model used for queries |
| Too many chunks in the prompt | The model gets distracted | Reduce `k` or use MMR/reranking later |
| No grounding instruction | The model hallucinates outside the source | Tell it to answer only from context |

## When Basic RAG Is Enough

Basic RAG is often enough when:

- the corpus is small or medium-sized
- documents are mostly plain text
- questions are direct factual questions
- one retrieval pass usually finds the answer
- you can tolerate simple top-`k` retrieval

Move beyond Basic RAG when:

- top-`k` returns repeated chunks
- queries need rewriting
- results need reranking
- multiple data sources must be routed
- the system needs multi-step tool use or agentic planning

For those next steps, return to [RAG.md](RAG.md#types-of-rag).
