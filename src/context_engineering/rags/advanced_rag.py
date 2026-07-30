"""
Advanced RAG with Pinecone, end to end.

This example demonstrates the same seven RAG stages as basic_rag.py, but adds
production-style retrieval improvements:

1. Load a local text file into LangChain Document objects.
2. Split the document into retrieval chunks.
3. Embed chunks and the query with the same embedding model.
4. Store vectors in Pinecone.
5. Retrieve with an improved query, metadata filtering, scores, and reranking.
6. Augment a prompt with only the best evidence.
7. Generate an answer with OpenAI or Llama70b.

Required environment variables:
    PINECONE_API_KEY

Optional environment variables:
    ADVANCED_RAG_MODEL=llama70b | openai
    ADVANCED_RAG_QUESTION="What building blocks does LangChain provide?"
    ADVANCED_RAG_INDEX=do-langchain-advanced-rag
    ADVANCED_RAG_NAMESPACE=advanced-rag-demo
    ADVANCED_RAG_PINECONE_CLOUD=aws
    ADVANCED_RAG_PINECONE_REGION=us-east-1
    ADVANCED_RAG_VERBOSE=1 | 0
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from utils.llm_utils import get_model, load_environment_variables  # noqa: E402

DEFAULT_SOURCE_PATH = Path(__file__).resolve().parents[1] / "resources" / "sample.txt"
DEFAULT_INDEX_NAME = "do-langchain-advanced-rag"
DEFAULT_NAMESPACE = "advanced-rag-demo"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
EMBEDDING_DIMENSIONS = 384


class HashingEmbeddings(Embeddings):
    """Small local embedding model for transparent tutorial output.

    Pinecone can store vectors from any embedding model as long as the index
    dimension matches. This toy embedding keeps the example cheap and makes it
    possible to print exactly how tokens become vector bucket updates.
    """

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            bucket, sign = hash_token(token, self.dimensions)
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def tokenize(text: str) -> list[str]:
    """Tokenize enough for the transparent demo embedding model."""

    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def hash_token(token: str, dimensions: int) -> tuple[int, float]:
    """Map a token into one vector bucket and a positive/negative sign."""

    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "big") % dimensions
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return bucket, sign


def load_documents(source_path: Path = DEFAULT_SOURCE_PATH) -> list[Document]:
    """Load a local source file into LangChain Document objects."""

    text = source_path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={
                "source": str(source_path),
                "loader": "Path.read_text",
                "rag_type": "advanced",
            },
        )
    ]


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = 250,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split documents into chunks and attach stable chunk metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = document_id(chunk, index)
    return chunks


def create_embeddings() -> HashingEmbeddings:
    """Create the embedding model used for chunks and queries."""

    return HashingEmbeddings(dimensions=EMBEDDING_DIMENSIONS)


def create_or_get_pinecone_index(
    *,
    index_name: str,
    dimensions: int,
    cloud: str,
    region: str,
    verbose: bool,
):
    """Create a Pinecone serverless index if it does not already exist."""

    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError as exc:
        raise ImportError(
            "Pinecone support requires: uv add pinecone langchain-pinecone"
        ) from exc

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is required to run Advanced RAG.")

    pc = Pinecone(api_key=api_key)

    if verbose:
        print_section("4a. Pinecone Index Setup")
        print(f"Index name: {index_name}")
        print(f"Expected dimension: {dimensions}")
        print(f"Metric: cosine")
        print(f"Serverless cloud/region: {cloud}/{region}")

    if not pc.has_index(index_name):
        if verbose:
            print("Index does not exist. Creating Pinecone serverless index...")
        pc.create_index(
            name=index_name,
            dimension=dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    elif verbose:
        print("Index already exists. Reusing it.")

    description = pc.describe_index(index_name)
    if verbose:
        print(f"Index host: {getattr(description, 'host', 'unknown')}")
        print(f"Index status: {getattr(description, 'status', 'unknown')}")

    return pc.Index(index_name)


def create_pinecone_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    *,
    index_name: str,
    namespace: str,
    cloud: str,
    region: str,
    verbose: bool = True,
):
    """Store embedded chunks in Pinecone through LangChain's vector store."""

    try:
        from langchain_pinecone import PineconeVectorStore
    except ImportError as exc:
        raise ImportError(
            "PineconeVectorStore requires: uv add pinecone langchain-pinecone"
        ) from exc

    pinecone_index = create_or_get_pinecone_index(
        index_name=index_name,
        dimensions=EMBEDDING_DIMENSIONS,
        cloud=cloud,
        region=region,
        verbose=verbose,
    )

    vector_store = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        namespace=namespace,
    )
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    if verbose:
        print_store_debug(chunks, embeddings, ids, index_name, namespace)

    try:
        vector_store.delete(ids=ids)
    except Exception as exc:
        if verbose:
            print(f"Delete-before-upsert skipped: {type(exc).__name__}: {exc}")

    vector_store.add_documents(chunks, ids=ids)

    if verbose:
        stats = pinecone_index.describe_index_stats()
        print("\nPinecone write complete.")
        print(f"Index stats: {stats}")

    return vector_store


def rewrite_query(question: str, *, model_name: str, verbose: bool = True) -> str:
    """Advanced RAG pre-retrieval step: rewrite a short query for retrieval."""

    prompt = ChatPromptTemplate.from_template(
        "Rewrite this user question as a concise semantic search query. "
        "Keep important nouns and technical terms. Return only the rewritten query.\n\n"
        "Question: {question}"
    )

    try:
        model = get_model(model_name)
        chain = prompt | model | StrOutputParser()
        rewritten = chain.invoke({"question": question}).strip()
    except Exception:
        rewritten = deterministic_query_rewrite(question)

    if verbose:
        print_section("5a. Query Rewrite")
        print(f"Original question: {question}")
        print(f"Rewritten retrieval query: {rewritten}")

    return rewritten


def deterministic_query_rewrite(question: str) -> str:
    """Fallback query rewrite when an LLM call is unavailable."""

    terms = [token for token in tokenize(question) if token not in STOPWORDS]
    return " ".join(terms) or question


def retrieve_documents(
    vector_store,
    query: str,
    *,
    k: int = 3,
    metadata_filter: dict[str, Any] | None = None,
):
    """Retrieve documents from Pinecone with scores."""

    return vector_store.similarity_search_with_score(
        query,
        k=k,
        filter=metadata_filter,
    )


def rerank_documents(
    question: str,
    scored_documents: list[tuple[Document, float]],
    *,
    keep: int = 3,
    verbose: bool = True,
) -> list[Document]:
    """Advanced RAG post-retrieval step: simple lexical rerank."""

    question_terms = set(tokenize(question)) - STOPWORDS
    reranked = []
    for document, vector_score in scored_documents:
        doc_terms = set(tokenize(document.page_content))
        lexical_overlap = len(question_terms & doc_terms)
        combined_score = lexical_overlap - float(vector_score)
        reranked.append((combined_score, lexical_overlap, vector_score, document))

    reranked.sort(key=lambda item: item[0], reverse=True)

    if verbose:
        print_section("5c. Rerank Retrieved Chunks")
        for rank, (combined, overlap, vector_score, document) in enumerate(
            reranked, start=1
        ):
            print(
                f"Rerank {rank}: combined={combined:.4f}, lexical_overlap={overlap}, vector_score={vector_score}"
            )
            print(f"  chunk_id: {document.metadata.get('chunk_id')}")
            print(f"  text: {preview_text(document.page_content, limit=130)}")

    return [document for *_scores, document in reranked[:keep]]


def retrieve(
    vector_store,
    question: str,
    *,
    model_name: str,
    k: int = 4,
    keep: int = 3,
    metadata_filter: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the Retrieve stage of Advanced RAG (the "R" in the acronym).

    Advanced RAG expands plain retrieval with a pre-retrieval query rewrite
    and a post-retrieval rerank, so this wraps rewrite_query,
    retrieve_documents, and rerank_documents under one entry point instead
    of leaving all three steps inline in advanced_rag(). Callers still get
    Retrieve -> Augment -> Generate as the top-level shape.
    """

    rewritten_query = rewrite_query(question, model_name=model_name, verbose=verbose)
    scored_documents = retrieve_documents(
        vector_store,
        rewritten_query,
        k=k,
        metadata_filter=metadata_filter,
    )

    if verbose:
        print_retrieval_debug(rewritten_query, scored_documents, metadata_filter)

    reranked_documents = rerank_documents(
        question,
        scored_documents,
        keep=keep,
        verbose=verbose,
    )

    return {
        "rewritten_query": rewritten_query,
        "scored_documents": scored_documents,
        "reranked_documents": reranked_documents,
    }


def format_documents(documents: list[Document]) -> str:
    """Format reranked documents as compact prompt context."""

    formatted = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        chunk_index = document.metadata.get("chunk_index", "unknown")
        formatted.append(
            f"[Evidence {index} | source={source} | chunk={chunk_index}]\n"
            f"{document.page_content}"
        )
    return "\n\n".join(formatted)


def build_prompt() -> ChatPromptTemplate:
    """Build the final grounded-answer prompt."""

    return ChatPromptTemplate.from_template(
        "You are answering with Advanced RAG.\n"
        "Use only the evidence below. If the evidence is insufficient, say you do not know.\n"
        "Cite evidence labels like [Evidence 1] when useful.\n\n"
        "Evidence:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )


def generate_answer(
    question: str,
    reranked_documents: list[Document],
    *,
    model_name: str,
    verbose: bool = True,
) -> str:
    """Generate a final answer from augmented context."""

    context = format_documents(reranked_documents)
    prompt = build_prompt()
    model = get_model(model_name)
    chain = prompt | model | StrOutputParser()

    if verbose:
        print_augment_debug(question, reranked_documents, context, prompt)
        print_generate_debug(model_name, model)

    answer = chain.invoke({"context": context, "question": question})

    if verbose:
        print("\nGenerated answer:")
        print(answer)

    return answer


def advanced_rag(
    question: str,
    *,
    source_path: Path = DEFAULT_SOURCE_PATH,
    index_name: str = DEFAULT_INDEX_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    cloud: str = DEFAULT_CLOUD,
    region: str = DEFAULT_REGION,
    model_name: str = "llama70b",
    k: int = 4,
    keep: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run Advanced RAG from load to generate."""

    load_environment_variables()
    load_dotenv()

    documents = load_documents(source_path)
    chunks = split_documents(documents)
    embeddings = create_embeddings()

    if verbose:
        print_load_split_embed_debug(question, documents, chunks, embeddings)

    vector_store = create_pinecone_vector_store(
        chunks,
        embeddings,
        index_name=index_name,
        namespace=namespace,
        cloud=cloud,
        region=region,
        verbose=verbose,
    )

    metadata_filter = {"rag_type": "advanced"}
    retrieval = retrieve(
        vector_store,
        question,
        model_name=model_name,
        k=k,
        keep=keep,
        metadata_filter=metadata_filter,
        verbose=verbose,
    )

    answer = generate_answer(
        question,
        retrieval["reranked_documents"],
        model_name=model_name,
        verbose=verbose,
    )

    return {
        "documents": documents,
        "chunks": chunks,
        "rewritten_query": retrieval["rewritten_query"],
        "retrieved_documents": [doc for doc, _score in retrieval["scored_documents"]],
        "reranked_documents": retrieval["reranked_documents"],
        "answer": answer,
    }


def document_id(document: Document, index: int) -> str:
    source = document.metadata.get("source", "unknown")
    digest = hashlib.blake2b(
        f"advanced:{source}:{index}:{document.page_content}".encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"advanced-rag-{index}-{digest}"


def print_section(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print("=" * 78)


def preview_text(text: str, *, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def vector_summary(vector: list[float], *, limit: int = 8) -> str:
    non_zero = [
        (index, round(value, 4))
        for index, value in enumerate(vector)
        if abs(value) > 1e-12
    ]
    return (
        f"dimensions={len(vector)}, non_zero={len(non_zero)}, "
        f"first_non_zero={non_zero[:limit]}"
    )


def print_hashing_debug(label: str, text: str, embeddings: HashingEmbeddings) -> None:
    tokens = tokenize(text)
    print(f"{label} token count: {len(tokens)}")
    print(f"{label} first tokens: {tokens[:12]}")
    print(f"{label} token -> vector bucket updates:")
    for token in tokens[:8]:
        bucket, sign = hash_token(token, embeddings.dimensions)
        print(f"  {token!r:>16} -> bucket={bucket:<3} sign={sign:+.0f}")
    print(f"{label} vector summary: {vector_summary(embeddings.embed_query(text))}")


def print_load_split_embed_debug(
    question: str,
    documents: list[Document],
    chunks: list[Document],
    embeddings: HashingEmbeddings,
) -> None:
    print_section("1. Load")
    for index, document in enumerate(documents, start=1):
        print(f"Document {index}: chars={len(document.page_content)}")
        print(f"Metadata: {document.metadata}")
        print(f"Preview: {preview_text(document.page_content)}")

    print_section("2. Split")
    print(f"Created chunks: {len(chunks)}")
    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}: id={chunk.metadata['chunk_id']}")
        print(f"  chars={len(chunk.page_content)}")
        print(f"  metadata={chunk.metadata}")
        print(f"  preview={preview_text(chunk.page_content, limit=120)}")

    print_section("3. Embed")
    print("Embedding model: transparent local HashingEmbeddings")
    print_hashing_debug("Question", question, embeddings)
    print_hashing_debug("Chunk 1", chunks[0].page_content, embeddings)


def print_store_debug(
    chunks: list[Document],
    embeddings: Embeddings,
    ids: list[str],
    index_name: str,
    namespace: str,
) -> None:
    print_section("4b. Store in Pinecone")
    print(f"VectorDB: Pinecone")
    print(f"Index: {index_name}")
    print(f"Namespace: {namespace}")
    print(f"Upsert records: {len(chunks)}")
    for index, (chunk_id, chunk) in enumerate(zip(ids, chunks), start=1):
        vector = embeddings.embed_query(chunk.page_content)
        print(f"\nPinecone record {index}:")
        print(f"  id: {chunk_id}")
        print(f"  text: {preview_text(chunk.page_content, limit=110)}")
        print(f"  metadata: {chunk.metadata}")
        print(f"  embedding: {vector_summary(vector, limit=5)}")


def print_retrieval_debug(
    rewritten_query: str,
    scored_documents: list[tuple[Document, float]],
    metadata_filter: dict[str, Any],
) -> None:
    print_section("5b. Retrieve from Pinecone")
    print(f"Query sent to Pinecone: {rewritten_query}")
    print(f"Metadata filter: {metadata_filter}")
    print(f"Returned matches: {len(scored_documents)}")
    for index, (document, score) in enumerate(scored_documents, start=1):
        print(f"\nPinecone match {index}:")
        print(f"  score/distance: {score}")
        print(f"  chunk_id: {document.metadata.get('chunk_id')}")
        print(f"  metadata: {document.metadata}")
        print(f"  text: {preview_text(document.page_content, limit=130)}")


def print_augment_debug(
    question: str,
    documents: list[Document],
    context: str,
    prompt: ChatPromptTemplate,
) -> None:
    print_section("6. Augment")
    prompt_value = prompt.invoke({"context": context, "question": question})
    messages = prompt_value.to_messages()
    print(f"Question inserted into prompt: {question}")
    print(f"Reranked evidence documents inserted: {len(documents)}")
    print(f"Formatted context characters: {len(context)}")
    print("\nFormatted context preview:")
    print(preview_text(context, limit=900))
    print("\nPrompt message preview:")
    for index, message in enumerate(messages, start=1):
        print(f"Message {index} ({message.type}):")
        print(preview_text(str(message.content), limit=900))


def print_generate_debug(model_name: str, model) -> None:
    print_section("7. Generate")
    resolved = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or model.__class__.__name__
    )
    print(f"Requested model alias: {model_name}")
    print(f"Resolved model: {resolved}")
    print("Input: augmented evidence prompt from step 6")


def print_run_summary(result: dict[str, Any]) -> None:
    print("\nRun summary:")
    print(f"Loaded documents: {len(result['documents'])}")
    print(f"Chunks: {len(result['chunks'])}")
    print(f"Rewritten query: {result['rewritten_query']}")
    print(f"Retrieved chunks: {len(result['retrieved_documents'])}")
    print(f"Reranked chunks used: {len(result['reranked_documents'])}")
    print("\nFinal answer:")
    print(result["answer"])


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def main() -> None:
    question = os.getenv(
        "ADVANCED_RAG_QUESTION",
        "What ancient history does Tenali Town has?",
    )
    model_name = os.getenv("ADVANCED_RAG_MODEL", "llama70b")
    index_name = os.getenv("ADVANCED_RAG_INDEX", DEFAULT_INDEX_NAME)
    namespace = os.getenv("ADVANCED_RAG_NAMESPACE", DEFAULT_NAMESPACE)
    cloud = os.getenv("ADVANCED_RAG_PINECONE_CLOUD", DEFAULT_CLOUD)
    region = os.getenv("ADVANCED_RAG_PINECONE_REGION", DEFAULT_REGION)
    verbose = os.getenv("ADVANCED_RAG_VERBOSE", "1") != "0"

    try:
        result = advanced_rag(
            question,
            index_name=index_name,
            namespace=namespace,
            cloud=cloud,
            region=region,
            model_name=model_name,
            verbose=verbose,
        )
    except Exception as exc:
        print(f"Could not run Advanced RAG: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    print_run_summary(result)


if __name__ == "__main__":
    main()
