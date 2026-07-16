# LangChain Text Splitters

Text splitters break large text (or `Document`s) into smaller chunks that fit within a model's context window and make good retrieval units. They live in `langchain_text_splitters` and share two entry points:

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = splitter.split_text(text)          # str -> list[str]
chunks = splitter.split_documents(documents)  # list[Document] -> list[Document], metadata preserved
```

Every splitter is configured with:

- `chunk_size` — the target maximum size of each chunk (in characters, unless the splitter measures otherwise).
- `chunk_overlap` — how much of the end of one chunk is repeated at the start of the next, so context isn't lost across a boundary.

## Common Splitter Types

| Splitter                        | Splits on                                   | Use when                                                          |
| -------------------------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| `CharacterTextSplitter`          | A single fixed separator (e.g. `"\n\n"`)     | Text has one consistent delimiter and you want simple, predictable splits. |
| `RecursiveCharacterTextSplitter` | A list of separators, falling back in order  | General-purpose default — keeps paragraphs/sentences/words intact as long as possible. |
| `TokenTextSplitter`              | Token count (via `tiktoken`)                 | You need chunks sized by what the model actually counts against its context window, not raw characters. |
| `MarkdownHeaderTextSplitter`     | Markdown headers (`#`, `##`, ...)             | You want each section under its own metadata instead of duplicating headers into every chunk's content. |
| `RecursiveCharacterTextSplitter.from_language` | Language-aware separators (`class`/`def`, blank lines, lines) | Splitting source code, so chunks break between functions/classes instead of mid-statement. |

## `CharacterTextSplitter`

Splits on one separator and greedily packs pieces up to `chunk_size`. If a single piece between separators is already bigger than `chunk_size`, it is kept whole (with a warning) rather than broken further — this splitter never falls back to finer-grained separators.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(separator="\n\n", chunk_size=150, chunk_overlap=0)
chunks = splitter.split_text(text)
```

## `RecursiveCharacterTextSplitter`

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

## `TokenTextSplitter`

Measures chunk size in tokens (using a `tiktoken` encoding like `cl100k_base`) instead of characters. Useful because character count is a rough proxy for what actually fills a model's context window — a chunk that "looks small" in characters can still be token-heavy.

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=40, chunk_overlap=5)
chunks = splitter.split_text(text)
```

Because it splits purely by token boundaries, chunks can start or end mid-word — it trades readability for precise context-window sizing.

## `MarkdownHeaderTextSplitter`

Splits markdown on header syntax and, instead of repeating each header inside every chunk's text, attaches the header hierarchy to `metadata`. This keeps chunk content focused while still letting you filter or display which section a chunk came from.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

print(chunks[0].metadata)  # {'Header 1': '...', 'Header 2': '...'}
```

## Code splitting with `Language`

`RecursiveCharacterTextSplitter.from_language()` swaps in a separator list tuned for a given programming language (e.g. `class `/`def ` boundaries before blank lines before plain lines for Python), so chunks break between top-level definitions instead of inside them.

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=120, chunk_overlap=0
)
chunks = splitter.split_text(source_code)
```

`Language` supports many languages beyond Python (JS/TS, Java, Go, Rust, Markdown, HTML, and more).

## Splitting PDFs and other `Document`s

`PyPDFLoader` returns one `Document` per page. Passing those documents through `split_documents()` (instead of `split_text()`) chunks the text while carrying each source document's `metadata` (page number, source path, ...) forward onto every chunk derived from it.

```python
documents = PyPDFLoader(pdf_path).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)

print(chunks[0].metadata)  # {'page': 0, 'source': '...', ...}
```

## Why `chunk_overlap` matters

Without overlap, a chunk boundary can land in the middle of a sentence or idea — the sentence gets cut in half, and each resulting chunk is missing the context the other one has. If a retriever only returns one of those chunks, whatever led up to it (or followed it) is gone.

With `chunk_overlap`, the tail of one chunk is duplicated at the head of the next, so the boundary region appears in both chunks. Retrieving either chunk on its own still preserves the context that spans the split point — at the cost of some duplicated (and reprocessed) text.

```python
no_overlap = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0)
with_overlap = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=40)
```

As a rule of thumb, `chunk_overlap` is usually set to 10-20% of `chunk_size` — enough to preserve continuity without duplicating so much text that retrieval becomes redundant or expensive.

## Choosing a Splitter

- Use `RecursiveCharacterTextSplitter` as the default for plain text.
- Use `CharacterTextSplitter` when a single, reliable separator already exists and you want the simplest possible behavior.
- Use `TokenTextSplitter` when chunk size needs to match the model's token-based context window, not character count.
- Use `MarkdownHeaderTextSplitter` when section/header structure should live in metadata rather than be repeated in every chunk.
- Use `RecursiveCharacterTextSplitter.from_language` for source code, so chunks respect function/class boundaries.
- Use `split_documents()` instead of `split_text()` whenever you're chunking `Document`s (e.g. from `PyPDFLoader`) and want to keep their metadata.
- Always set a non-zero `chunk_overlap` for retrieval use cases — it's cheap insurance against losing context at chunk boundaries.
