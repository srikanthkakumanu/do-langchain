# Context Engineering

A hands-on tutorial on **context engineering**: getting the right external data — a file, a web page, a PDF — into an LLM's limited context window as clean, well-sized, retrievable pieces. It walks the full pipeline in order: **loading** raw sources into LangChain `Document` objects, **splitting** those documents into chunks with a text splitter, and **chunking** strategy — comparing splitters, tuning `chunk_size`, and using `chunk_overlap` — to produce chunks that are actually good retrieval units. Examples are runnable from [Document_Loaders.py](../src/context_engineering/loaders_chunking/Document_Loaders.py), [TextLoader.py](../src/context_engineering/loaders_chunking/TextLoader.py), [WebLoader.py](../src/context_engineering/loaders_chunking/WebLoader.py), and [chunking_splitters.py](../src/context_engineering/loaders_chunking/chunking_splitters.py), all using **LangChain 1.3** (`langchain-text-splitters` 1.1, current as of this writing).

> `langchain-community` — the package that used to provide `TextLoader`, `WebBaseLoader`, `DirectoryLoader`, and `PyPDFLoader` — was sunset and archived in June 2026, with no dedicated partner package replacing these thin, single-purpose loaders. The loading examples below reimplement them directly on top of the libraries they always wrapped internally (the standard library, `pypdf`, `requests`, and `BeautifulSoup`), in [loaders.py](../src/context_engineering/loaders_chunking/loaders.py). Behavior and `Document` output are unchanged.

## Table of Contents

- [What Is Context Engineering](#what-is-context-engineering)
- [Document Loaders](#document-loaders)
  - [Loading a text file](#loading-a-text-file)
  - [Loading a web page](#loading-a-web-page)
  - [Loading a directory](#loading-a-directory)
  - [Loading a PDF](#loading-a-pdf)
  - [`Document`](#document)
  - [Eager vs. lazy loading](#eager-vs-lazy-loading)
  - [Choosing a Loader](#choosing-a-loader)
- [Text Splitters](#text-splitters)
  - [`CharacterTextSplitter`](#charactertextsplitter)
  - [`RecursiveCharacterTextSplitter`](#recursivecharactertextsplitter)
  - [`TokenTextSplitter`](#tokentextsplitter)
  - [`MarkdownHeaderTextSplitter`](#markdownheadertextsplitter)
  - [Code splitting with `Language`](#code-splitting-with-language)
  - [Splitting PDFs and other `Document`s](#splitting-pdfs-and-other-documents)
  - [Choosing a Splitter](#choosing-a-splitter)
- [Chunking: Strategy, Size, and Overlap](#chunking-strategy-size-and-overlap)
  - [Comparing splitter strategies](#comparing-splitter-strategies)
  - [Comparing chunk sizes](#comparing-chunk-sizes)
  - [Why `chunk_overlap` matters](#why-chunk_overlap-matters)
- [Next: Retrieval-Augmented Generation (RAG)](#next-retrieval-augmented-generation-rag)

## What Is Context Engineering

**Context engineering** is the practice of getting the right external information into a model's context window, in a shape it can actually use — as opposed to [prompt engineering](Prompt_Engineering.md), which shapes *how* the model is instructed once that information is already there. The two compose: a well-engineered prompt with the wrong (or missing) context still produces an ungrounded or hallucinated answer, since the model is reasoning from what it happens to remember from training rather than from real, current data (see "Why LLMs Hallucinate" in `LLM_Engineering.md` and [Contextual Grounding](Prompt_Engineering.md#7-contextual-grounding-retrieval-augmented-prompting)).

Getting there is a three-stage pipeline:

```text
raw source  ->  load()  ->  Document(s)  ->  split  ->  chunks  ->  embed / retrieve
(file, URL,     Loader                       Splitter
 PDF, ...)
```

1. **Load** — a **document loader** reads a source (file, directory, web page, PDF) and converts it into one or more LangChain `Document` objects, the common unit every downstream component understands.
2. **Split** — a **text splitter** breaks each `Document`'s text into smaller pieces sized to fit a model's context window (or an embedding model's input limit).
3. **Chunk** — the *outcome* of splitting. The same source text can chunk very differently depending on which splitter and settings are chosen; getting this outcome right — not too large, not too small, with enough overlap to preserve context across a boundary — is what makes a retrieved chunk actually useful on its own.

The rest of this tutorial covers each stage in turn.

## Document Loaders

Document loaders read data from a source (a file, a directory, a web page, a PDF, ...) and convert it into LangChain `Document` objects — the common unit that prompts, splitters, and vector stores all understand.

```python
document = load_text_file(path)
```

[loaders.py](../src/context_engineering/loaders_chunking/loaders.py) mirrors the two entry points the old `langchain-community` loader classes exposed:

- an eager function that reads the whole source and returns `Document`(s).
- `iter_directory()` — a generator that yields one `Document` at a time instead of holding everything in memory (the `lazy_load()` equivalent).

| Function                        | Reads                          | Output                                 | Use when                                                 |
| -------------------------------- | ------------------------------- | ---------------------------------------- | ---------------------------------------------------------- |
| `load_text_file()`              | A single local text file        | One `Document`                         | You have plain text/markdown on disk.                    |
| `load_web_page()`               | One URL                        | One `Document`                         | You need the text content of a web page.                 |
| `iter_directory()`              | A folder of files               | One `Document` per matching file (generator) | You want to load many files the same way.           |
| `load_pdf()`                    | A local PDF file                | One `Document` per page                | You need per-page text and metadata from a PDF.          |
| `Document`                      | (not a loader) in-memory data   | A single `Document`                    | You are constructing documents manually, e.g. in tests.  |

### Loading a text file

Reads a single local text file into one `Document`. `page_content` holds the raw file text and `metadata["source"]` holds the file path.

```python
from loaders import load_text_file

document = load_text_file(SAMPLE_FILE_PATH, encoding="utf-8")

print(document.page_content[:100])
print(document.metadata)  # {'source': '.../sample.txt'}
```

Internally it's `open(path, encoding=encoding).read()` wrapped in a `Document` — exactly what `langchain-community`'s `TextLoader` did.

### Loading a web page

Fetches a URL with `requests` and parses the HTML with BeautifulSoup, returning the extracted page text as one `Document`.

```python
from loaders import load_web_page

document = load_web_page("https://python.langchain.com/docs/concepts/document_loaders/")

print(document.metadata["source"])
print(document.page_content[:200])
```

Set the `USER_AGENT` environment variable before fetching (e.g. `os.environ.setdefault("USER_AGENT", "my-app/1.0")`) — some sites are picky about requests with no user agent.

### Loading a directory

Walks a directory, matches files against a glob pattern, and lazily loads each one, yielding each file's `Document` in turn.

```python
from loaders import iter_directory

for doc in iter_directory(tmpdir, glob="*.txt"):
    print(doc.page_content[:50])
    print(doc.metadata["source"])
```

Generator-based, so it loads and yields one file at a time instead of reading the entire directory into memory up front — the same behavior `DirectoryLoader.lazy_load()` gave you.

### Loading a PDF

Loads a PDF and returns one `Document` per page, with page-level metadata (`page`, `total_pages`, `source`). It's backed directly by `pypdf` (v6 as of this writing) — the same library `langchain-community`'s `PyPDFLoader` used under the hood.

```python
from loaders import load_pdf

documents = load_pdf(pdf_path)

for doc in documents:
    print(doc.metadata["page"], doc.page_content[:100])
```

Per-page `Document`s are what make PDFs easy to chunk later — a text splitter's `split_documents()` carries each page's metadata forward onto every chunk derived from it (see [Splitting PDFs and other `Document`s](#splitting-pdfs-and-other-documents)).

### `Document`

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

### Eager vs. lazy loading

- Eager functions (`load_text_file()`, `load_pdf()`, `load_web_page()`) read everything up front and return `Document`(s) directly. Simple, but holds all content in memory at once.
- `iter_directory()` is a generator that reads and yields one `Document` at a time. Prefer this pattern for large directories, large PDFs, or any source where you don't need every document in memory simultaneously.

### Choosing a Loader

- Use `load_text_file()` for a single local text/markdown file.
- Use `load_web_page()` for HTML pages you want as plain text.
- Use `iter_directory()` to apply the same loading logic across many files in a folder.
- Use `load_pdf()` for PDFs, especially when you need per-page metadata.
- Construct `Document` directly when content isn't coming from an external source at all.

## Text Splitters

Text splitters break large text (or `Document`s) into smaller chunks that fit within a model's context window and make good retrieval units. They live in `langchain_text_splitters` and share two entry points:

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = splitter.split_text(text)          # str -> list[str]
chunks = splitter.split_documents(documents)  # list[Document] -> list[Document], metadata preserved
```

Every splitter is configured with:

- `chunk_size` — the target maximum size of each chunk (in characters, unless the splitter measures otherwise).
- `chunk_overlap` — how much of the end of one chunk is repeated at the start of the next, so context isn't lost across a boundary. See [Why `chunk_overlap` matters](#why-chunk_overlap-matters) below for what this buys you in practice.

| Splitter                                       | Splits on                                                       | Use when                                                                                |
| ------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `CharacterTextSplitter`                        | A single fixed separator (e.g. `"\n\n"`)                        | Text has one consistent delimiter and you want simple, predictable splits.              |
| `RecursiveCharacterTextSplitter`                | A list of separators, falling back in order                     | General-purpose default — keeps paragraphs/sentences/words intact as long as possible.  |
| `TokenTextSplitter`                             | Token count (via `tiktoken`)                                     | You need chunks sized by what the model actually counts against its context window, not raw characters. |
| `MarkdownHeaderTextSplitter`                    | Markdown headers (`#`, `##`, ...)                                | You want each section under its own metadata instead of duplicating headers into every chunk's content. |
| `RecursiveCharacterTextSplitter.from_language`  | Language-aware separators (`class`/`def`, blank lines, lines)  | Splitting source code, so chunks break between functions/classes instead of mid-statement. |

### `CharacterTextSplitter`

Splits on one separator and greedily packs pieces up to `chunk_size`. If a single piece between separators is already bigger than `chunk_size`, it is kept whole (with a warning) rather than broken further — this splitter never falls back to finer-grained separators.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(separator="\n\n", chunk_size=150, chunk_overlap=0)
chunks = splitter.split_text(text)
```

### `RecursiveCharacterTextSplitter`

The recommended default for plain text. It tries separators in order — paragraphs (`"\n\n"`), then lines (`"\n"`), then sentences (`". "`), then words (`" "`), then characters (`""`) — recursively splitting only as finely as needed to fit `chunk_size`. This keeps semantically related text (a paragraph, a sentence) together whenever it can.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(text)
```

### `TokenTextSplitter`

Measures chunk size in tokens (using a `tiktoken` encoding like `cl100k_base`) instead of characters. Useful because character count is a rough proxy for what actually fills a model's context window — a chunk that "looks small" in characters can still be token-heavy.

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=40, chunk_overlap=5)
chunks = splitter.split_text(text)
```

Because it splits purely by token boundaries, chunks can start or end mid-word — it trades readability for precise context-window sizing.

### `MarkdownHeaderTextSplitter`

Splits markdown on header syntax and, instead of repeating each header inside every chunk's text, attaches the header hierarchy to `metadata`. This keeps chunk content focused while still letting you filter or display which section a chunk came from.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

print(chunks[0].metadata)  # {'Header 1': '...', 'Header 2': '...'}
```

### Code splitting with `Language`

`RecursiveCharacterTextSplitter.from_language()` swaps in a separator list tuned for a given programming language (e.g. `class `/`def ` boundaries before blank lines before plain lines for Python), so chunks break between top-level definitions instead of inside them.

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=120, chunk_overlap=0
)
chunks = splitter.split_text(source_code)
```

`Language` supports many languages beyond Python (JS/TS, Java, Go, Rust, Markdown, HTML, and more).

### Splitting PDFs and other `Document`s

`load_pdf()` returns one `Document` per page. Passing those documents through `split_documents()` (instead of `split_text()`) chunks the text while carrying each source document's `metadata` (page number, source path, ...) forward onto every chunk derived from it.

```python
documents = load_pdf(pdf_path)
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)

print(chunks[0].metadata)  # {'page': 0, 'source': '...', ...}
```

### Choosing a Splitter

- Use `RecursiveCharacterTextSplitter` as the default for plain text.
- Use `CharacterTextSplitter` when a single, reliable separator already exists and you want the simplest possible behavior.
- Use `TokenTextSplitter` when chunk size needs to match the model's token-based context window, not character count.
- Use `MarkdownHeaderTextSplitter` when section/header structure should live in metadata rather than be repeated in every chunk.
- Use `RecursiveCharacterTextSplitter.from_language` for source code, so chunks respect function/class boundaries.
- Use `split_documents()` instead of `split_text()` whenever you're chunking `Document`s (e.g. from `load_pdf()`) and want to keep their metadata.
- Always set a non-zero `chunk_overlap` for retrieval use cases — it's cheap insurance against losing context at chunk boundaries.

## Chunking: Strategy, Size, and Overlap

Chunking is the *outcome* of running a splitter over a document — the pieces that actually get embedded, stored, and retrieved. The splitter (above) is the tool; chunking is the result, and the same source text can chunk very differently depending on which splitter and settings are picked. This section compares that outcome: how splitter strategies differ, how `chunk_size` trades off against chunk count, and why `chunk_overlap` is worth the extra stored text.

Models and embeddings have a limited context window, so a whole document usually can't be handed over (or embedded) as one unit. Chunking splits it into pieces that:

- fit within that window,
- are small enough that a retriever can return just the relevant part instead of an entire document, and
- still make sense on their own, since a chunk is what gets read in isolation at retrieval time.

Good chunking is a balance: chunks too large waste context on irrelevant text and blur embeddings across unrelated ideas; chunks too small lose the surrounding context needed to make sense of them.

### Comparing splitter strategies

Running the same source text through each splitter, at a comparable `chunk_size`, shows how differently they carve it up:

```python
splitters = {
    "CharacterTextSplitter": CharacterTextSplitter(separator="\n\n", chunk_size=150, chunk_overlap=0),
    "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0),
    "TokenTextSplitter": TokenTextSplitter(chunk_size=40, chunk_overlap=0),
}
for name, splitter in splitters.items():
    chunks = splitter.split_text(text)
    print(name, len(chunks), [len(c) for c in chunks])
```

On the 598-character sample text, this produces:

| Splitter                        | Chunks | Chunk sizes (chars)     |
| -------------------------------- | ------ | ------------------------ |
| `CharacterTextSplitter`          | 3      | 93, 192, 308             |
| `RecursiveCharacterTextSplitter` | 6      | 93, 96, 95, 90, 96, 120   |
| `TokenTextSplitter`              | 4      | 192, 176, 155, 75         |

`CharacterTextSplitter` only splits on `"\n\n"`, so it produces fewer, uneven chunks — some well over `chunk_size` — because it won't break a paragraph down further. `RecursiveCharacterTextSplitter` falls back to line/sentence/word separators and stays close to the target size. `TokenTextSplitter` measures in tokens rather than characters, so its character counts don't line up with the others at all — a reminder that `chunk_size` means different things for different splitters.

### Comparing chunk sizes

For a fixed splitter, `chunk_size` alone controls the count/context trade-off:

```python
for chunk_size in (50, 100, 200, 400):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = splitter.split_text(text)
    print(chunk_size, len(chunks), [len(c) for c in chunks])
```

| `chunk_size` | Chunks | Observation                                                  |
| ------------ | ------ | -------------------------------------------------------------- |
| 50           | 15     | Many small, choppy chunks — some as short as 7-8 characters.  |
| 100          | 7      | Chunks roughly track natural sentence/line boundaries.        |
| 200          | 4      | Chunks span multiple sentences each.                          |
| 400          | 2      | Nearly the whole document in each chunk.                      |

Smaller `chunk_size` means more, narrower chunks — better retrieval precision, but each chunk carries less surrounding context. Larger `chunk_size` means fewer, broader chunks — more context per chunk, but retrieval is coarser and irrelevant text is more likely to ride along with the relevant part. There's no universally correct value; it depends on how much context a single retrieved chunk needs to be useful on its own.

### Why `chunk_overlap` matters

`chunk_overlap` duplicates the tail of one chunk at the head of the next, so text isn't lost at the boundary between chunks. The effect only shows up when a chunk boundary actually falls mid-sentence — collapsing the sample text's line breaks and splitting on words demonstrates it clearly:

```python
no_overlap = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0, separators=[" ", ""])
with_overlap = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=40, separators=[" ", ""])
```

Without overlap, consecutive chunks share nothing:

```
End of chunk 1:   '...powered by large language models (LLMs). It provides building'
Start of chunk 2: 'blocks for prompt templates, ...'
```

A retriever that returns only chunk 2 has no idea it followed a sentence about "large language models" — that context is gone.

With `chunk_overlap=40`, the boundary region is duplicated across both chunks:

```
Chunk 1: '...powered by large language models (LLMs). It provides building'
Chunk 2: 'models (LLMs). It provides building blocks for prompt templates, ...'
```

Shared words between the two: `{'models', '(LLMs).', 'It', 'provides', 'building'}`. Either chunk retrieved on its own still carries that connecting phrase.

The cost is duplicated (and re-embedded) text, so overlap shouldn't be excessive — a common starting point is 10-20% of `chunk_size`. But for anything used in retrieval, some non-zero overlap is nearly always worth that cost.

## Next: Retrieval-Augmented Generation (RAG)

Loading, splitting, and chunking exist to produce good retrieval units — but a chunk sitting in memory isn't useful on its own. The next step is embedding those chunks, indexing them in a vector store, and retrieving the most relevant ones at query time to ground an LLM's answer. See **[docs/RAG.md](RAG.md)** for the embedding, vector store, retriever, and LCEL chain-building stages that pick up where this document leaves off.
