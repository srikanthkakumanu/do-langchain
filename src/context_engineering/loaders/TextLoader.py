"""
Document loaders using LangChain V.1
This example demonstrates how to read a local .txt file into a LangChain
Document object and inspect its page_content and metadata. langchain-community's
TextLoader is sunset, so loaders.load_text_file() reimplements it directly.
"""

import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from context_engineering.loaders import load_text_file as _load_text_file  # noqa: E402

SAMPLE_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "resources", "sample.txt"
)


def load_text_file():
    """Loads a local .txt file into a Document."""

    document = _load_text_file(SAMPLE_FILE_PATH, encoding="utf-8")

    print("Loaded 1 document(s)")
    print(f"\nMetadata: {document.metadata}")
    print(f"Content Preview: {document.page_content[:100]}...")  # Print first 100 characters
    print(f"Content:\n{document.page_content}")


def lazy_load_text_file():
    """Loads a local .txt file and reports its content length."""

    document = _load_text_file(SAMPLE_FILE_PATH, encoding="utf-8")

    print(f"\nMetadata: {document.metadata}")
    print(f"Content length: {len(document.page_content)} characters")


def main():
    load_text_file()
    # lazy_load_text_file()


if __name__ == "__main__":
    main()
