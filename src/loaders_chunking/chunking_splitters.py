"""
Text splitting and chunking using LangChain V.1
This example demonstrates the different text splitter strategies available in
langchain_text_splitters, compares how they chunk the same source document, and
shows why chunk_overlap matters when splitting text for retrieval.
"""

import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

SAMPLE_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "resources", "sample.txt"
)
PDF_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "resources", "langchain_demo.pdf"
)

MARKDOWN_TEXT = """# LangChain
LangChain is a framework for developing applications powered by large language models.

## Document Loaders
Document loaders read data from a source and convert it into Document objects.

## Text Splitters
Text splitters break large documents into smaller chunks that fit within a model's context window.
"""

PYTHON_CODE = '''\
def greet(name: str) -> str:
    """Return a friendly greeting for name."""
    return f"Hello, {name}!"


class Greeter:
    """Greets people, remembering a default name to fall back on."""

    def __init__(self, default_name: str = "World"):
        self.default_name = default_name

    def greet(self, name: str | None = None) -> str:
        return greet(name or self.default_name)
'''


def _load_sample_text() -> str:
    """Loads the sample text file and returns its raw content."""

    loader = TextLoader(SAMPLE_FILE_PATH, encoding="utf-8")
    documents = loader.load()
    return documents[0].page_content


def character_splitter():
    """Splits text on a single fixed separator (CharacterTextSplitter)."""

    text = _load_sample_text()
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=150, chunk_overlap=0)
    chunks = splitter.split_text(text)

    print(f"CharacterTextSplitter produced {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} ({len(chunk)} chars):\n{chunk}")


def recursive_splitter():
    """Splits text by trying separators in order (paragraph, line, sentence,
    word) until each piece fits chunk_size (RecursiveCharacterTextSplitter)."""

    text = _load_sample_text()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    print(f"RecursiveCharacterTextSplitter produced {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} ({len(chunk)} chars):\n{chunk}")


def token_splitter():
    """Splits text by token count instead of character count, which keeps
    chunks aligned with a model's actual context window (TokenTextSplitter)."""

    text = _load_sample_text()
    splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=40, chunk_overlap=5)
    chunks = splitter.split_text(text)

    print(f"TokenTextSplitter produced {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} ({len(chunk)} chars):\n{chunk}")


def markdown_header_splitter():
    """Splits markdown by header level and attaches each header's text to
    chunk metadata instead of duplicating it in every chunk's content
    (MarkdownHeaderTextSplitter)."""

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(MARKDOWN_TEXT)

    print(f"MarkdownHeaderTextSplitter produced {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} metadata: {chunk.metadata}")
        print(f"Content: {chunk.page_content}")


def code_splitter():
    """Splits source code using language-aware separators (class/def
    boundaries first, then blank lines, then lines) so chunks break between
    functions and classes instead of mid-statement (RecursiveCharacterTextSplitter
    .from_language)."""

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=120, chunk_overlap=0
    )
    chunks = splitter.split_text(PYTHON_CODE)

    print(f"Language.PYTHON splitter produced {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} ({len(chunk)} chars):\n{chunk}")


def pdf_splitter():
    """Loads a multi-page PDF and splits its pages into chunks, showing that
    split_documents() carries each page's metadata (e.g. page number) forward
    onto every chunk derived from it."""

    documents = PyPDFLoader(PDF_FILE_PATH).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print(f"Loaded {len(documents)} page(s) from PDF")
    print(f"Split into {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\nChunk {i + 1} metadata: {chunk.metadata}")
        print(f"Content preview: {chunk.page_content[:150]}...")


def compare_splitters():
    """Runs the same text through each splitter strategy with a comparable
    chunk_size and compares how many chunks, and what sizes, each produces."""

    text = _load_sample_text()
    chunk_size = 150

    splitters = {
        "CharacterTextSplitter": CharacterTextSplitter(
            separator="\n\n", chunk_size=chunk_size, chunk_overlap=0
        ),
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0
        ),
        "TokenTextSplitter": TokenTextSplitter(chunk_size=40, chunk_overlap=0),
    }

    print(f"Comparing splitters on {len(text)} characters of source text\n")
    for name, splitter in splitters.items():
        chunks = splitter.split_text(text)
        sizes = [len(chunk) for chunk in chunks]
        print(f"{name}: {len(chunks)} chunk(s), sizes={sizes}")


def compare_chunk_sizes():
    """Splits the same text at several chunk_size values to show the
    size/count trade-off: smaller chunks mean more chunks with less context
    per chunk, larger chunks mean fewer chunks with more context each."""

    text = _load_sample_text()
    print(f"Source text length: {len(text)} characters\n")

    for chunk_size in (50, 100, 200, 400):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        chunks = splitter.split_text(text)
        sizes = [len(chunk) for chunk in chunks]
        print(f"chunk_size={chunk_size}: {len(chunks)} chunk(s), sizes={sizes}")


def overlap_importance():
    """Demonstrates why chunk_overlap matters: without overlap, a sentence
    that straddles a chunk boundary gets cut in half and neither chunk has
    the full context. With overlap, the boundary region is duplicated across
    both chunks so retrieving either one still keeps that context intact."""

    # Collapse line breaks so the splitter has to break mid-sentence on word
    # boundaries instead of along the file's existing (short) hard-wrapped lines.
    text = " ".join(_load_sample_text().split())
    chunk_size = 120

    no_overlap = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separators=[" ", ""]
    )
    with_overlap = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=40, separators=[" ", ""]
    )

    no_overlap_chunks = no_overlap.split_text(text)
    with_overlap_chunks = with_overlap.split_text(text)

    print("--- Without chunk_overlap ---")
    for i, chunk in enumerate(no_overlap_chunks):
        print(f"\nChunk {i + 1}:\n{chunk!r}")

    print("\n--- With chunk_overlap=40 ---")
    for i, chunk in enumerate(with_overlap_chunks):
        print(f"\nChunk {i + 1}:\n{chunk!r}")

    if len(no_overlap_chunks) > 1:
        print(f"\nEnd of chunk 1 (no overlap):   {no_overlap_chunks[0][-30:]!r}")
        print(f"Start of chunk 2 (no overlap): {no_overlap_chunks[1][:30]!r}")
        print("There is no shared text between the two chunks above: a retriever")
        print("that only returns chunk 2 has lost whatever led up to it.")

    if len(with_overlap_chunks) > 1:
        shared = set(with_overlap_chunks[0][-40:].split()) & set(
            with_overlap_chunks[1][:40].split()
        )
        print(f"\nWords shared between consecutive overlapping chunks: {shared}")


def main():
    character_splitter()
    recursive_splitter()
    token_splitter()
    markdown_header_splitter()
    code_splitter()
    pdf_splitter()
    compare_splitters()
    compare_chunk_sizes()
    overlap_importance()


if __name__ == "__main__":
    main()
