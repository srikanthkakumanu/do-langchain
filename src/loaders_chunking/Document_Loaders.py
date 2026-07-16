import os
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
)

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("USER_AGENT", "do-langchain/0.1")

SAMPLE_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "resources", "sample.txt"
)
PDF_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "resources", "langchain_demo.pdf"
)


def load_text_file():
    # Load the sample text file using TextLoader
    loader = TextLoader(SAMPLE_FILE_PATH, encoding="utf-8")
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s)")
    print(f"Content preview: {documents[0].page_content[:100]}...")
    print(f"Metadata: {documents[0].metadata}")


def web_loader():
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Web_scraping")
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from web")
    print(f"Source: {documents[0].metadata.get('source', 'N/A')}")
    print(f"Content length: {len(documents[0].page_content)} characters")
    print(f"Preview: {documents[0].page_content[:200]}...")


def lazy_loader():

    # Create temp directory with sample files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample files
        for i in range(5):
            path = Path(tmpdir) / f"doc_{i}.txt"
            path.write_text(f"This is document {i}. It contains sample content.")

        loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)

        print("Initialized lazy loader for directory:", tmpdir)
        for doc in loader.lazy_load():
            print("Document Content Preview:", doc.page_content[:50], "...")
            print("Metadata:", doc.metadata["source"])


def doc_structure():
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


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i + 1} Content Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    load_text_file()
    web_loader()
    lazy_loader()
    doc_structure()
    pdf_loader(PDF_FILE_PATH)
