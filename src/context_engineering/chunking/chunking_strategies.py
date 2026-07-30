"""
Chunking strategies using LangChain V.1

Reusable wrappers around each `langchain_text_splitters` strategy: one
function per strategy, each taking the text (or `Document`s) to split plus
the strategy's key knobs, and returning the resulting chunks. See
chunking_examples.py for runnable demonstrations of each of these.
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


def split_by_character(
    text: str, separator: str = "\n\n", chunk_size: int = 150, chunk_overlap: int = 0
) -> list[str]:
    """Splits text on a single fixed separator (CharacterTextSplitter)."""

    splitter = CharacterTextSplitter(
        separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def split_recursive(
    text: str,
    chunk_size: int = 150,
    chunk_overlap: int = 20,
    separators: list[str] | None = None,
) -> list[str]:
    """Splits text by trying separators in order (paragraph, line, sentence,
    word) until each piece fits chunk_size (RecursiveCharacterTextSplitter)."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return splitter.split_text(text)


def split_by_tokens(
    text: str,
    encoding_name: str = "cl100k_base",
    chunk_size: int = 40,
    chunk_overlap: int = 5,
) -> list[str]:
    """Splits text by token count instead of character count, which keeps
    chunks aligned with a model's actual context window (TokenTextSplitter)."""

    splitter = TokenTextSplitter(
        encoding_name=encoding_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def split_markdown_by_headers(
    markdown_text: str, headers_to_split_on: list[tuple[str, str]] | None = None
) -> list[Document]:
    """Splits markdown by header level and attaches each header's text to
    chunk metadata instead of duplicating it in every chunk's content
    (MarkdownHeaderTextSplitter)."""

    if headers_to_split_on is None:
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return splitter.split_text(markdown_text)


def split_code(
    source_code: str,
    language: Language = Language.PYTHON,
    chunk_size: int = 120,
    chunk_overlap: int = 0,
) -> list[str]:
    """Splits source code using language-aware separators (class/def
    boundaries first, then blank lines, then lines) so chunks break between
    functions and classes instead of mid-statement (RecursiveCharacterTextSplitter
    .from_language)."""

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(source_code)


def split_documents(
    documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Document]:
    """Splits a list of Documents into chunks, carrying each source
    document's metadata (e.g. page number) forward onto every chunk derived
    from it (RecursiveCharacterTextSplitter.split_documents)."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
