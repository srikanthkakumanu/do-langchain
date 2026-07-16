# Chunking

Chunking is the act of running a splitter over a document to produce the pieces that actually get embedded, stored, and retrieved. The splitter (see [Splitters.md](Splitters.md)) is the tool; chunking is the outcome — and the same source text can chunk very differently depending on which splitter and settings you pick. This doc looks at that outcome: how to compare strategies, how `chunk_size` trades off against chunk count, and why `chunk_overlap` is worth the extra stored text. Examples are runnable from [chunking_splitters.py](../src/loaders_chunking/chunking_splitters.py).

## Why chunking matters

Models and embeddings have a limited context window, so a whole document usually can't be handed over (or embedded) as one unit. Chunking splits it into pieces that:

- fit within that window,
- are small enough that a retriever can return just the relevant part instead of an entire document, and
- still make sense on their own, since a chunk is what gets read in isolation at retrieval time.

Good chunking is a balance: chunks too large waste context on irrelevant text and blur embeddings across unrelated ideas; chunks too small lose the surrounding context needed to make sense of them.

## Comparing splitter strategies

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

| Splitter                        | Chunks | Chunk sizes (chars)          |
| -------------------------------- | ------ | ------------------------------ |
| `CharacterTextSplitter`          | 3      | 93, 192, 308                  |
| `RecursiveCharacterTextSplitter` | 6      | 93, 96, 95, 90, 96, 120        |
| `TokenTextSplitter`              | 4      | 192, 176, 155, 75              |

`CharacterTextSplitter` only splits on `"\n\n"`, so it produces fewer, uneven chunks — some well over `chunk_size` — because it won't break a paragraph down further. `RecursiveCharacterTextSplitter` falls back to line/sentence/word separators and stays close to the target size. `TokenTextSplitter` measures in tokens rather than characters, so its character counts don't line up with the others at all — a reminder that `chunk_size` means different things for different splitters.

## Comparing chunk sizes

For a fixed splitter, `chunk_size` alone controls the count/context trade-off:

```python
for chunk_size in (50, 100, 200, 400):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = splitter.split_text(text)
    print(chunk_size, len(chunks), [len(c) for c in chunks])
```

| `chunk_size` | Chunks | Observation                                              |
| ------------ | ------ | ---------------------------------------------------------- |
| 50           | 15     | Many small, choppy chunks — some as short as 7-8 characters. |
| 100          | 7      | Chunks roughly track natural sentence/line boundaries.     |
| 200          | 4      | Chunks span multiple sentences each.                       |
| 400          | 2      | Nearly the whole document in each chunk.                   |

Smaller `chunk_size` means more, narrower chunks — better retrieval precision, but each chunk carries less surrounding context. Larger `chunk_size` means fewer, broader chunks — more context per chunk, but retrieval is coarser and irrelevant text is more likely to ride along with the relevant part. There's no universally correct value; it depends on how much context a single retrieved chunk needs to be useful on its own.

## Overlap importance

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
