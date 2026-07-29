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
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def _tokenize(text: str) -> list[str]:
    """Tokenize enough for the demo embedding model."""

    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


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
):
    """Embed chunks and store them in a vector store.

    Use store_type="memory" for the dependency-free learning path. Use
    store_type="chroma" when langchain-chroma is installed and you want a
    persistent local Chroma collection.
    """

    if store_type == "memory":
        return create_in_memory_vector_store(chunks, embeddings)
    if store_type == "chroma":
        return create_chroma_vector_store(chunks, embeddings)

    raise ValueError(f"Unknown vector store type: {store_type}")


def create_in_memory_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
) -> InMemoryVectorStore:
    """Embed chunks and store them in an in-memory vector store."""

    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    return vector_store


def create_chroma_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    *,
    persist_directory: Path = DEFAULT_CHROMA_PATH,
    collection_name: str = "basic_rag_demo",
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
        return chain.invoke({"context": context, "question": question})
    except Exception as exc:
        return generate_extractive_fallback(question, retrieved_documents, exc)


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
) -> dict[str, object]:
    """Run the complete Basic RAG workflow from load to generate."""

    documents = load_documents(source_path)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vector_store = create_vector_store(chunks, embeddings, store_type=store_type)
    retriever = create_retriever(vector_store, k=k)
    retrieved_documents = retrieve_documents(retriever, question)
    answer = generate_answer(question, retrieved_documents, model_name=model_name)

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


def main() -> None:
    question = os.getenv(
        "BASIC_RAG_QUESTION",
        "What building blocks does LangChain provide?",
    )
    store_type = os.getenv("BASIC_RAG_VECTOR_STORE", "memory")
    model_name = os.getenv("BASIC_RAG_MODEL", "llama70b")
    try:
        result = basic_rag(question, store_type=store_type, model_name=model_name)
    except ImportError as exc:
        print(f"Could not start Basic RAG: {exc}")
        raise SystemExit(1) from exc

    print_run_summary(result)


if __name__ == "__main__":
    main()
