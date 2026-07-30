"""
Document loaders using LangChain V.1
This example demonstrates how to read one or more web pages into LangChain
Document objects and inspect their page_content and metadata.
langchain-community's WebBaseLoader is sunset, so loaders.load_web_page()
reimplements it directly with requests + BeautifulSoup, exactly what
WebBaseLoader did internally.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("USER_AGENT", "do-langchain-example/1.0")

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from context_engineering.loaders import load_web_page as _load_web_page  # noqa: E402

WEB_PATHS = [
    "https://python.langchain.com/docs/concepts/document_loaders/",
]


def load_web_page():
    """Loads web page(s) into a list of Document objects."""

    documents = [_load_web_page(url) for url in WEB_PATHS]

    print(f"Loaded {len(documents)} document(s)")
    for document in documents:
        print(f"\nMetadata: {document.metadata}")
        print(f"Content Preview: {document.page_content[:100]}...")  # Print first 100 characters


def lazy_load_web_page():
    """Loads web page(s) one at a time, printing each as it arrives."""

    for url in WEB_PATHS:
        document = _load_web_page(url)
        print(f"\nMetadata: {document.metadata}")
        print(f"Content length: {len(document.page_content)} characters")


def main():
    load_web_page()
    # lazy_load_web_page()


if __name__ == "__main__":
    main()
