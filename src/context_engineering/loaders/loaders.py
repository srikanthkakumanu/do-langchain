"""
Minimal document loaders.

langchain-community (home of TextLoader, DirectoryLoader, PyPDFLoader, and
WebBaseLoader) was sunset and archived in June 2026: no dedicated partner
package replaces these thin, single-purpose loaders, so this module
reimplements them directly on top of the libraries they always wrapped
internally (the standard library, pypdf, and requests/BeautifulSoup) and
returns plain LangChain `Document` objects.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader


def load_text_file(path: str, encoding: str = "utf-8") -> Document:
    """Reads a local text file into one Document (mirrors TextLoader)."""

    with open(path, encoding=encoding) as f:
        text = f.read()
    return Document(page_content=text, metadata={"source": path})


def iter_directory(directory: str, glob: str = "*.txt", encoding: str = "utf-8") -> Iterator[Document]:
    """Lazily yields one Document per matching file in a directory (mirrors DirectoryLoader)."""

    for path in sorted(Path(directory).glob(glob)):
        yield load_text_file(str(path), encoding=encoding)


def load_pdf(path: str) -> list[Document]:
    """Loads a PDF into one Document per page (mirrors PyPDFLoader)."""

    reader = PdfReader(path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": path, "page": i, "total_pages": len(reader.pages)},
        )
        for i, page in enumerate(reader.pages)
    ]


def load_web_page(url: str) -> Document:
    """Fetches a URL and extracts its visible text into one Document (mirrors WebBaseLoader)."""

    headers = {"User-Agent": os.environ.get("USER_AGENT", "do-langchain-example/1.0")}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    return Document(page_content=text, metadata={"source": url, "title": title})
