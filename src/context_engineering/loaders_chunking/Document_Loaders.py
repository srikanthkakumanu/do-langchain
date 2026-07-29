"""
Document loaders using LangChain V.1

Covers what TextLoader.py and WebLoader.py don't: the Document object's
structure, lazily loading a whole directory of files, and loading a
multi-page PDF. langchain-community's DirectoryLoader and PyPDFLoader are
sunset, so loaders.iter_directory() and loaders.load_pdf() reimplement them
directly.
"""

import os
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from loaders import iter_directory, load_pdf

PDF_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "resources", "langchain_demo.pdf"
)


def doc_structure():
    """Shows the two fields every Document carries: page_content and metadata."""

    doc = Document(
        page_content="This is a sample document.",
        metadata={
            "source": "manual_creation.txt",
            "author": "Paulo",
            "length": 30,
            "tags": ["sample", "test"],
            "created_at": "2024-06-01",
        },
    )

    print("Document Structure:")
    print(f"  page_content (type): {type(doc.page_content)}")
    print(f"  page_content: {doc.page_content}")
    print(f"  metadata: {doc.metadata}")


def lazy_directory_loader():
    """Lazily loads every .txt file in a directory, yielding Documents one at a time."""

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            path = Path(tmpdir) / f"doc_{i}.txt"
            path.write_text(f"This is document {i}. It contains sample content.")

        print(f"Initialized lazy loader for directory: {tmpdir}")
        for doc in iter_directory(tmpdir, glob="*.txt"):
            print(f"Content preview: {doc.page_content[:50]}...")
            print(f"Metadata: {doc.metadata}")


def pdf_loader():
    """Loads a multi-page PDF, one Document per page."""

    documents = load_pdf(PDF_FILE_PATH)

    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i + 1} content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")


def main():
    doc_structure()
    lazy_directory_loader()
    pdf_loader()


if __name__ == "__main__":
    main()
