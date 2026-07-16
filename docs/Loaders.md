# LangChain Document Loaders

Document loaders read data from a source (a file, a directory, a web page, a PDF, ...) and convert it into LangChain `Document` objects — the common unit that prompts, splitters, and vector stores all understand.

```python
loader = TextLoader(path)
documents = loader.load()
```

Every loader exposes the same two entry points:

- `load()` — reads the whole source and returns a `list[Document]`.
- `lazy_load()` — returns an iterator/generator of `Document`, yielding one at a time instead of holding everything in memory.

## Common Loader Types

| Loader           | Reads                        | Output                                                | Use when                                                  |
| ---------------- | ----------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| `TextLoader`      | A single local text file      | One `Document`                                        | You have plain text/markdown on disk.                     |
| `WebBaseLoader`   | One or more URLs               | One `Document` per URL                                | You need the text content of a web page.                  |
| `DirectoryLoader`  | A folder of files              | One `Document` per matching file                      | You want to load many files with the same loader class.   |
| `PyPDFLoader`     | A local PDF file                | One `Document` per page                               | You need per-page text and metadata from a PDF.            |
| `Document`         | (not a loader) in-memory data | A single `Document`                                    | You are constructing documents manually, e.g. in tests.    |

## `TextLoader`

Reads a single local text file into one `Document`. `page_content` holds the raw file text and `metadata["source"]` holds the file path.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader(SAMPLE_FILE_PATH, encoding="utf-8")
documents = loader.load()

print(documents[0].page_content[:100])
print(documents[0].metadata)  # {'source': '.../sample.txt'}
```

Use `loader.lazy_load()` instead of `load()` when you only need to stream through the content rather than hold the whole `list[Document]` in memory.

## `WebBaseLoader`

Fetches one or more URLs with `requests` and parses the HTML with BeautifulSoup, returning the extracted page text as one `Document` per URL.

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Web_scraping")
# or: WebBaseLoader(web_path=["https://url-1", "https://url-2"])
documents = loader.load()

print(documents[0].metadata["source"])
print(documents[0].page_content[:200])
```

Set the `USER_AGENT` environment variable before creating the loader (e.g. `os.environ.setdefault("USER_AGENT", "my-app/1.0")`) — some sites are picky about requests with no user agent, and LangChain otherwise prints a warning.

## `DirectoryLoader`

Walks a directory, matches files against a glob pattern, and loads each one with a given loader class (`loader_cls`), combining every file's `Document`(s) into a single collection.

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)

for doc in loader.lazy_load():
    print(doc.page_content[:50])
    print(doc.metadata["source"])
```

`lazy_load()` is especially useful here since it loads and yields one file at a time instead of reading the entire directory into memory up front.

## `PyPDFLoader`

Loads a PDF and returns one `Document` per page, with page-level metadata (`page`, `page_label`, `total_pages`, plus PDF metadata like `producer`/`creationdate`).

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(pdf_path)
documents = loader.load()

for doc in documents:
    print(doc.metadata["page"], doc.page_content[:100])
```

Per-page `Document`s are what make PDFs easy to chunk later — a text splitter's `split_documents()` carries each page's metadata forward onto every chunk derived from it.

## `Document`

Not a loader itself, but the object every loader produces. It's a simple container with `page_content: str` and `metadata: dict`, and can be constructed directly when you need to build documents by hand (tests, fixtures, manual ingestion).

```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is a sample document.",
    metadata={"source": "manual_creation.txt", "author": "Paulo"},
)

print(doc.page_content)
print(doc.metadata)
```

## `load()` vs `lazy_load()`

- `load()` reads everything up front and returns a `list[Document]`. Simple, but holds all content in memory at once.
- `lazy_load()` returns an iterator that reads and yields one `Document` at a time. Prefer it for large directories, large PDFs, or any source where you don't need every document in memory simultaneously.

## Choosing a Loader

- Use `TextLoader` for a single local text/markdown file.
- Use `WebBaseLoader` for HTML pages you want as plain text.
- Use `DirectoryLoader` to apply one loader class across many files in a folder.
- Use `PyPDFLoader` for PDFs, especially when you need per-page metadata.
- Construct `Document` directly when content isn't coming from an external source at all.
