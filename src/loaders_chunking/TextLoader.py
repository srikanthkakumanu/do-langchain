"""
Document loaders using LangChain V.1
This example demonstrates how to use TextLoader to read a local .txt file
into LangChain Document objects, and inspect their page_content and metadata.
"""

import os

from langchain_community.document_loaders import TextLoader

SAMPLE_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "resources", "sample.txt"
)


def load_text_file():
    """Loads a local .txt file into a list of Document objects."""

    loader = TextLoader(SAMPLE_FILE_PATH, encoding="utf-8")
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s)")
    for document in documents:
        print(f"\nMetadata: {document.metadata}")
        print(f"Content Preview: {document.page_content[:100]}...")  # Print first 100 characters
        print(f"Content:\n{document.page_content}")


def lazy_load_text_file():
    """Lazily loads a local .txt file, yielding Documents one at a time."""

    loader = TextLoader(SAMPLE_FILE_PATH, encoding="utf-8")

    for document in loader.lazy_load():
        print(f"\nMetadata: {document.metadata}")
        print(f"Content length: {len(document.page_content)} characters")


def main():
    load_text_file()
    # lazy_load_text_file()


if __name__ == "__main__":
    main()
