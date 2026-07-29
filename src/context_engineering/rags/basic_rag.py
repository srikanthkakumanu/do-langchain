"""
Basic / Naive RAG, end to end.

This example intentionally keeps the retrieval side local and dependency-light:

1. Load a local text file into LangChain Document objects.
2. Split the document into smaller chunks.
3. Embed chunks with a tiny deterministic HashingEmbeddings class.
4. Store chunks in a vector store.
   - Default: LangChain's InMemoryVectorStore, which needs no extra setup.
   - Optional: Chroma, if langchain-chroma is installed.
5. Retrieve the most relevant chunks for a question.
6. Augment a prompt with retrieved context.
7. Generate an answer.

The generator tries to use the repo's configured chat model. If no provider API
key is available, it falls back to a deterministic extractive answer so the
workflow is still runnable while learning the mechanics.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from utils.llm_utils import get_model, load_environment_variables  # noqa: E402

DEFAULT_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "sample.txt"
)
DEFAULT_CHROMA_PATH = Path(__file__).resolve().parents[3] / ".chroma" / "basic_rag"


class HashingEmbeddings(Embeddings):
    """Small local embedding model for demos.

    This is not a production-quality semantic embedding model. It maps tokens
    into a fixed-size vector with a hashing trick, then normalizes the vector so
    cosine/dot-product retrieval has useful behavior for simple examples.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many chunk texts for indexing."""

        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one user question for retrieval."""

        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            bucket, sign = _hash_token(token, self.dimensions)
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def _tokenize(text: str) -> list[str]:
    """Tokenize enough for the demo embedding model."""

    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _hash_token(token: str, dimensions: int) -> tuple[int, float]:
    """Map a token to one embedding-vector bucket and sign."""

    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "big") % dimensions
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return bucket, sign


def load_documents(source_path: Path = DEFAULT_SOURCE_PATH) -> list[Document]:
    """Load a local text file into Document objects."""

    text = source_path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={"source": str(source_path), "loader": "Path.read_text"},
        )
    ]


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = 250,
    chunk_overlap: int = 40,
) -> list[Document]:
    """Split loaded documents into retrieval-sized chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_embeddings() -> Embeddings:
    """Create the embedding model used for both chunks and queries."""

    return HashingEmbeddings(dimensions=384)


def create_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    *,
    store_type: str = "memory",
    verbose: bool = True,
):
    """Embed chunks and store them in a vector store.

    Use store_type="memory" for the dependency-free learning path. Use
    store_type="chroma" when langchain-chroma is installed and you want a
    persistent local Chroma collection.
    """

    if store_type == "memory":
        return create_in_memory_vector_store(chunks, embeddings, verbose=verbose)
    if store_type == "chroma":
        return create_chroma_vector_store(chunks, embeddings, verbose=verbose)

    raise ValueError(f"Unknown vector store type: {store_type}")


def create_in_memory_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    *,
    verbose: bool = True,
) -> InMemoryVectorStore:
    """Embed chunks and store them in an in-memory vector store."""

    vector_store = InMemoryVectorStore(embedding=embeddings)
    ids = [_document_id(chunk, index) for index, chunk in enumerate(chunks)]
    vector_store.add_documents(chunks, ids=ids)

    if verbose:
        print_store_write_debug(
            "memory",
            chunks,
            embeddings,
            ids=ids,
            vector_store=vector_store,
        )

    return vector_store


def create_chroma_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    *,
    persist_directory: Path = DEFAULT_CHROMA_PATH,
    collection_name: str = "basic_rag_demo",
    verbose: bool = True,
):
    """Embed chunks and store them in a persistent local Chroma collection."""

    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise ImportError(
            "Chroma support requires the optional langchain-chroma package. "
            "Install it with: uv add langchain-chroma chromadb"
        ) from exc

    persist_directory.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    ids = [_document_id(chunk, index) for index, chunk in enumerate(chunks)]

    # Re-running the demo should update the same logical chunks instead of
    # accumulating duplicates in the persistent Chroma collection.
    try:
        vector_store.delete(ids=ids)
    except Exception:
        pass

    vector_store.add_documents(chunks, ids=ids)

    if verbose:
        print_store_write_debug(
            "chroma",
            chunks,
            embeddings,
            ids=ids,
            vector_store=vector_store,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    return vector_store


def _document_id(document: Document, index: int) -> str:
    """Create a stable enough id for demo Chroma upserts."""

    source = document.metadata.get("source", "unknown")
    digest = hashlib.blake2b(
        f"{source}:{index}:{document.page_content}".encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"basic-rag-{index}-{digest}"


def create_retriever(vector_store, *, k: int = 3):
    """Turn the vector store into a retriever Runnable."""

    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_documents(retriever, question: str) -> list[Document]:
    """Retrieve the most relevant chunks for a question."""

    return retriever.invoke(question)


def retrieve_documents_with_scores(vector_store, question: str, *, k: int = 3):
    """Retrieve documents with similarity scores when the store supports it."""

    if hasattr(vector_store, "similarity_search_with_score"):
        return vector_store.similarity_search_with_score(question, k=k)
    return [(document, None) for document in vector_store.similarity_search(question, k=k)]


def format_documents(documents: list[Document]) -> str:
    """Format retrieved chunks into the context string inserted into the prompt."""

    formatted = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown source")
        formatted.append(f"[Chunk {index} | source={source}]\n{document.page_content}")
    return "\n\n".join(formatted)


def build_prompt() -> ChatPromptTemplate:
    """Create the grounded-answer prompt template."""

    return ChatPromptTemplate.from_template(
        "You are answering with Basic RAG.\n"
        "Use only the context below. If the context does not contain the answer, "
        "say you do not know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )


def generate_answer(
    question: str,
    retrieved_documents: list[Document],
    *,
    model_name: str = "llama70b",
    verbose: bool = True,
) -> str:
    """Generate the final answer from retrieved context.

    Uses the configured chat model when possible. Falls back to an extractive
    response if API credentials are missing or the model call fails.
    """

    context = format_documents(retrieved_documents)
    prompt = build_prompt()

    try:
        load_environment_variables()
        model = get_model(model_name)
        chain = prompt | model | StrOutputParser()
        if verbose:
            print_generate_debug(model_name, model, using_fallback=False)
        answer = chain.invoke({"context": context, "question": question})
        if verbose:
            print_generated_answer_debug(answer)
        return answer
    except Exception as exc:
        if verbose:
            print_generate_debug(model_name, None, using_fallback=True, error=exc)
        answer = generate_extractive_fallback(question, retrieved_documents, exc)
        if verbose:
            print_generated_answer_debug(answer)
        return answer


def generate_extractive_fallback(
    question: str,
    retrieved_documents: list[Document],
    error: Exception | None = None,
) -> str:
    """Return a deterministic answer when no LLM call is available."""

    question_terms = set(_tokenize(question))
    scored_sentences: list[tuple[int, str]] = []

    for document in retrieved_documents:
        sentences = re.split(r"(?<=[.!?])\s+", document.page_content)
        for sentence in sentences:
            score = len(question_terms & set(_tokenize(sentence)))
            if score:
                scored_sentences.append((score, sentence.strip()))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    best_sentences = [sentence for _, sentence in scored_sentences[:2]]

    if not best_sentences:
        best_sentences = [retrieved_documents[0].page_content.strip()]

    note = ""
    if error is not None:
        note = f"\n\n(Note: LLM generation was skipped: {type(error).__name__}: {error})"

    return " ".join(best_sentences) + note


def basic_rag(
    question: str,
    *,
    source_path: Path = DEFAULT_SOURCE_PATH,
    k: int = 3,
    model_name: str = "llama70b",
    store_type: str = "memory",
    verbose: bool = True,
) -> dict[str, object]:
    """Run the complete Basic RAG workflow from load to generate."""

    documents = load_documents(source_path)
    chunks = split_documents(documents)
    embeddings = create_embeddings()

    if verbose:
        print_pipeline_debug(question, documents, chunks, embeddings)

    vector_store = create_vector_store(
        chunks,
        embeddings,
        store_type=store_type,
        verbose=verbose,
    )
    retriever = create_retriever(vector_store, k=k)
    retrieved_documents = retrieve_documents(retriever, question)

    if verbose:
        scored_documents = retrieve_documents_with_scores(vector_store, question, k=k)
        print_retrieval_debug(store_type, question, embeddings, scored_documents)

    if verbose:
        print_augmentation_debug(question, retrieved_documents)

    answer = generate_answer(
        question,
        retrieved_documents,
        model_name=model_name,
        verbose=verbose,
    )

    return {
        "documents": documents,
        "chunks": chunks,
        "store_type": store_type,
        "retrieved_documents": retrieved_documents,
        "answer": answer,
    }


def print_run_summary(result: dict[str, object]) -> None:
    """Print the important artifacts produced by the RAG workflow."""

    documents = result["documents"]
    chunks = result["chunks"]
    retrieved_documents = result["retrieved_documents"]
    answer = result["answer"]

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Vector store: {result['store_type']}")
    print(f"Retrieved chunks: {len(retrieved_documents)}")

    for index, document in enumerate(retrieved_documents, start=1):
        preview = " ".join(document.page_content.split())[:160]
        print(f"\nRetrieved chunk {index}:")
        print(f"Metadata: {document.metadata}")
        print(f"Preview: {preview}...")

    print("\nFinal answer:")
    print(answer)


def print_section(title: str) -> None:
    """Print a readable section divider for tutorial output."""

    print(f"\n{'=' * 78}")
    print(title)
    print("=" * 78)


def preview_text(text: str, *, limit: int = 140) -> str:
    """Return a compact one-line text preview."""

    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def vector_summary(vector: list[float], *, limit: int = 8) -> str:
    """Summarize a sparse-ish vector without printing all dimensions."""

    non_zero = [
        (index, round(value, 4))
        for index, value in enumerate(vector)
        if abs(value) > 1e-12
    ]
    return (
        f"dimensions={len(vector)}, non_zero={len(non_zero)}, "
        f"first_non_zero={non_zero[:limit]}"
    )


def print_tokenization_debug(label: str, text: str, *, token_limit: int = 16) -> None:
    """Show how the demo tokenizer breaks text into tokens."""

    tokens = _tokenize(text)
    print(f"{label} token count: {len(tokens)}")
    print(f"{label} first tokens: {tokens[:token_limit]}")


def print_hashing_debug(
    label: str,
    text: str,
    embeddings: Embeddings,
    *,
    token_limit: int = 8,
) -> None:
    """Show how sample tokens become vector bucket updates."""

    if not isinstance(embeddings, HashingEmbeddings):
        print(f"{label}: embedding internals are provider-managed for this model.")
        return

    tokens = _tokenize(text)[:token_limit]
    print(f"{label} token -> vector bucket updates:")
    for token in tokens:
        bucket, sign = _hash_token(token, embeddings.dimensions)
        print(f"  {token!r:>16} -> bucket={bucket:<3} sign={sign:+.0f}")

    vector = embeddings.embed_query(text)
    print(f"{label} vector summary: {vector_summary(vector)}")


def print_pipeline_debug(
    question: str,
    documents: list[Document],
    chunks: list[Document],
    embeddings: Embeddings,
) -> None:
    """Print load, split, tokenization, and embedding details."""

    print_section("1. Load")
    for index, document in enumerate(documents, start=1):
        print(f"Document {index}: chars={len(document.page_content)}")
        print(f"Metadata: {document.metadata}")
        print(f"Preview: {preview_text(document.page_content)}")

    print_section("2. Split")
    print(f"Created {len(chunks)} chunks")
    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}: chars={len(chunk.page_content)}")
        print(f"  Metadata: {chunk.metadata}")
        print(f"  Preview: {preview_text(chunk.page_content, limit=110)}")

    print_section("3. Tokenize and Embed")
    print_tokenization_debug("Question", question)
    print_hashing_debug("Question", question, embeddings)

    sample_chunk = chunks[0]
    print_tokenization_debug("Chunk 1", sample_chunk.page_content)
    print_hashing_debug("Chunk 1", sample_chunk.page_content, embeddings)


def print_store_write_debug(
    store_type: str,
    chunks: list[Document],
    embeddings: Embeddings,
    *,
    ids: list[str],
    vector_store: Any,
    persist_directory: Path | None = None,
    collection_name: str | None = None,
) -> None:
    """Print how chunks are written into the selected vector store."""

    store_label = "InMemoryVectorStore" if store_type == "memory" else "Chroma"
    print_section(f"4. Store in {store_label}")

    if store_type == "memory":
        print("Storage location: Python process memory only")
        print("Persistence: lost when this script exits")
    else:
        print(f"Storage location: {persist_directory}")
        print(f"Collection: {collection_name}")
        print("Persistence: saved on disk by Chroma")

    print(f"Stored chunk count: {len(chunks)}")
    for index, (chunk_id, chunk) in enumerate(zip(ids, chunks), start=1):
        vector = embeddings.embed_query(chunk.page_content)
        print(f"\nStored item {index}:")
        print(f"  id: {chunk_id}")
        print(f"  text: {preview_text(chunk.page_content, limit=100)}")
        print(f"  metadata: {chunk.metadata}")
        print(f"  embedding: {vector_summary(vector, limit=5)}")

    if store_type == "memory":
        memory_store = getattr(vector_store, "store", None)
        if isinstance(memory_store, dict):
            print(f"\nIn-memory backing dict keys: {list(memory_store.keys())[:5]}")
    else:
        collection = getattr(vector_store, "_collection", None)
        if collection is not None:
            print(f"\nChroma collection count: {collection.count()}")
            sample = collection.get(limit=1, include=["documents", "metadatas"])
            print(f"Chroma sample ids: {sample.get('ids', [])}")


def print_retrieval_debug(
    store_type: str,
    question: str,
    embeddings: Embeddings,
    scored_documents,
) -> None:
    """Print how retrieval is performed and what came back."""

    store_label = "InMemoryVectorStore" if store_type == "memory" else "Chroma"
    print_section(f"5. Retrieve from {store_label}")
    print(f"Question: {question}")
    print_hashing_debug("Query", question, embeddings)
    print(f"Returned documents: {len(scored_documents)}")

    for index, (document, score) in enumerate(scored_documents, start=1):
        print(f"\nRetrieved result {index}:")
        if score is not None:
            print(f"  similarity/distance score: {score}")
        print(f"  metadata: {document.metadata}")
        print(f"  text: {preview_text(document.page_content, limit=130)}")


def print_augmentation_debug(
    question: str,
    retrieved_documents: list[Document],
) -> None:
    """Print how retrieved documents become prompt context."""

    print_section("6. Augment")
    context = format_documents(retrieved_documents)
    prompt = build_prompt()
    prompt_value = prompt.invoke({"context": context, "question": question})
    messages = prompt_value.to_messages()

    print(f"Question inserted into prompt: {question}")
    print(f"Retrieved documents inserted: {len(retrieved_documents)}")
    print(f"Formatted context characters: {len(context)}")
    print("\nFormatted context preview:")
    print(preview_text(context, limit=900))

    print("\nPrompt message preview:")
    for index, message in enumerate(messages, start=1):
        content = getattr(message, "content", "")
        print(f"Message {index} ({message.type}):")
        print(preview_text(str(content), limit=900))


def print_generate_debug(
    model_name: str,
    model,
    *,
    using_fallback: bool,
    error: Exception | None = None,
) -> None:
    """Print how the generation stage is executed."""

    print_section("7. Generate")
    print(f"Requested model alias: {model_name}")

    if using_fallback:
        print("Generation path: extractive fallback")
        if error is not None:
            print(f"Fallback reason: {type(error).__name__}: {error}")
        print("Fallback behavior: rank retrieved sentences by token overlap with the question")
        return

    provider_model = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or model.__class__.__name__
    )
    print("Generation path: chat model")
    print(f"Resolved model: {provider_model}")
    print("Input: augmented prompt messages from step 6")


def print_generated_answer_debug(answer: str) -> None:
    """Print the generated answer inside the step 7 trace."""

    print("\nGenerated answer:")
    print(answer)


def main() -> None:
    question = os.getenv(
        "BASIC_RAG_QUESTION",
        "What building blocks does LangChain provide?",
    )
    store_type = os.getenv("BASIC_RAG_VECTOR_STORE", "memory")
    model_name = os.getenv("BASIC_RAG_MODEL", "llama70b")
    verbose = os.getenv("BASIC_RAG_VERBOSE", "1") != "0"
    try:
        result = basic_rag(
            question,
            store_type=store_type,
            model_name=model_name,
            verbose=verbose,
        )
    except ImportError as exc:
        print(f"Could not start Basic RAG: {exc}")
        raise SystemExit(1) from exc

    print_run_summary(result)


if __name__ == "__main__":
    main()
