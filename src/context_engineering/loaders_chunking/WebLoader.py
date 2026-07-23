"""
Document loaders using LangChain V.1
This example demonstrates how to use WebBaseLoader to read one or more web
pages into LangChain Document objects, and inspect their page_content and metadata.
"""

import os

os.environ.setdefault("USER_AGENT", "do-langchain-example/1.0")

from langchain_community.document_loaders import WebBaseLoader

WEB_PATHS = [
    "https://python.langchain.com/docs/concepts/document_loaders/",
]


def load_web_page():
    """Loads web page(s) into a list of Document objects."""

    loader = WebBaseLoader(web_path=WEB_PATHS)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s)")
    for document in documents:
        print(f"\nMetadata: {document.metadata}")
        print(f"Content Preview: {document.page_content[:100]}...")  # Print first 100 characters


def lazy_load_web_page():
    """Lazily loads web page(s), yielding Documents one at a time."""

    loader = WebBaseLoader(web_path=WEB_PATHS)

    for document in loader.lazy_load():
        print(f"\nMetadata: {document.metadata}")
        print(f"Content length: {len(document.page_content)} characters")


def main():
    load_web_page()
    # lazy_load_web_page()


if __name__ == "__main__":
    main()
