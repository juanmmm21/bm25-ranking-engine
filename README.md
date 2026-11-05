# bm25-ranking-engine

**Project 5/10 of the [`beacon-search-engine`](https://github.com/juanmmm21/beacon-search-engine) ecosystem** — *Ranking and Relevance* category.
Repository: [`github.com/juanmmm21/bm25-ranking-engine`](https://github.com/juanmmm21/bm25-ranking-engine)

The industry-standard lexical relevance scorer — the same algorithm Elasticsearch
and Lucene use internally — implemented from scratch: term frequency saturation,
inverse document frequency, and document-length normalization, combined into a
single score for every candidate document against a query. No `rank_bm25`, no
`gensim`, no third-party ranking library: the formula, the IDF variant, and the
NumPy-vectorized batch scoring are all implemented here.

## What problem it solves

An inverted index tells you *which* documents contain a query term. It says
nothing about *how relevant* each of those documents actually is. A document
that mentions a term once in ten thousand words is not as relevant as one where
the term is central to a short, focused page — and a term that appears in every
document in the corpus carries far less signal than one that appears in only a
handful. BM25 is the classical answer to "how do I turn raw term occurrences
into a relevance score that reflects both of these things at once," refined
over decades of information retrieval research (it descends directly from the
probabilistic ranking work behind the Okapi retrieval system) and still the
lexical baseline every modern search engine either uses directly or measures
itself against. This project is exactly that scoring layer: given a query and
an inverted index, rank every candidate document by lexical relevance, with
nothing borrowed from a ranking library.

## Role in `beacon-search-engine`

```text
inverted-index-builder ── index-compression-codec
                                  │ documents.jsonl, stats.json,
                                  │ postings (plain or compressed)
                                  ▼
                    ┌───────────────────────────┐
                    │   bm25-ranking-engine      │   (this project)
                    │   query terms → BM25 score │
                    └─────────────┬──────────────┘
                                  │ ranked (doc_id, score) list
                                  ▼
                       learning-to-rank-reranker
                                  │        ▲
                                  │        │ pagerank-link-analysis
                                  ▼
                       (converges in beacon-search-console)
```

`bm25-ranking-engine` sits directly on top of the indexing stage: it reads
whichever index it is pointed at — the uncompressed output of
[`inverted-index-builder`](https://github.com/juanmmm21/inverted-index-builder)
or the compressed, randomly-addressable one from
[`index-compression-codec`](https://github.com/juanmmm21/index-compression-codec)
— and produces one of the input signals
[`learning-to-rank-reranker`](https://github.com/juanmmm21/learning-to-rank-reranker)
combines with page authority from
[`pagerank-link-analysis`](https://github.com/juanmmm21/pagerank-link-analysis).
It never decides the final result order on its own: it is a lexical relevance
signal, not the whole ranking pipeline.

## Goal and skills demonstrated

- **The BM25 formula implemented directly from its probabilistic derivation**,
  not copied from a library: term frequency saturation via `k1`, document-length
  normalization via `b`, and an IDF term with Lucene/Elasticsearch's +1 smoothing
  to keep it non-negative for very common terms — each piece parametrized and
  justified on its own, not hidden behind a single opaque `.score()` call.
- **NumPy-vectorized batch scoring**: for every query term, the contribution to
  every candidate document is computed as array operations over that term's
  posting list — `doc_id`, `term_frequency`, and document length all resolved
  and combined as vectors — never a Python `for` loop over documents, which is
  what actually makes scoring thousands of candidates per query practical.
  Combining scores *across* the handful of terms in a query uses a plain
  dictionary — the honest scope of "batch operations over documents," not
  "vectorize everything at any cost."
- **Independent re-implementation of a sibling repo's binary contract, not an
  import of it**: to read `index-compression-codec`'s compressed index, this
  project re-implements its documented varint and bitpacking decode logic from
  the README's own byte-layout description, entirely independently of that
  repo's Python package — the polyrepo integration contract of
  `beacon-search-engine` (data formats on disk, never cross-repo imports),
  applied to a real binary format, not just JSON.
- **Deterministic ranking with an explicit tie-break**: equal scores are
  ordered by ascending `doc_id`, not by whatever order a term happened to be
  processed in — the same determinism guarantee the rest of the ecosystem's
  indexing and ranking modules hold themselves to.

## How it works

1. **Opening the index** (`index_reader.py`): `manifest.json` is read first;
   if it declares a `terms_file`, the index is a compressed
   `index-compression-codec` index and `CompressedIndexReader` is used —
   random-access lookup, decoding only the bytes of the term requested. If it
   doesn't, it's a plain `inverted-index-builder` index and `PlainIndexReader`
   loads all posting lists into memory once (that format has no offset
   directory to seek into, so there is no random access to preserve). A
   directory with a `segments.json` is a segmented index (`inverted-index-builder`
   ≥ 0.2.0) and `SegmentedIndexReader` merges its live documents, skipping
   every `doc_id` listed in a segment's tombstones and recomputing `N`, the
   average length and each term's document frequency over live documents
   only. All readers precompute a `document_lengths` NumPy array indexed by
   `doc_id`, built once at open time.
2. **Tokenizing the query** (`tokenizer.py`, `search_text` only): Unicode NFC
   normalization plus locale-aware `casefold()`, matching the normalization
   `inverted-index-builder` applies at indexing time, so free-text query terms
   land on the same vocabulary. This is a CLI convenience, not the ecosystem's
   query parser — `query-parser-autocomplete` owns that role, including
   operators this project doesn't handle.
3. **Scoring** (`scoring.py`): for each distinct query term, its posting list
   is resolved, converted to `doc_id`/`term_frequency` NumPy arrays, and BM25's
   per-document contribution is computed as vector arithmetic — including
   resolving every matched document's length via the precomputed
   `document_lengths` array. Those per-term arrays are kept in a bounded LRU
   cache (4096 terms by default), so a frequent term is converted once, not on
   every query. Contributions from different query terms are summed with a
   single `np.bincount` (same addition order as a per-document accumulator, so
   the same float bit for bit), the top-K is preselected with `np.partition`
   and ties are broken by ascending `doc_id`.
4. **Top-K strategy** (`dynamic_pruning.py`): `exhaustive` (the default, step
   3), `wand` or `block_max_wand` — see
   [Dynamic pruning](#dynamic-pruning-wand-and-block-max-wand).
5. **Orchestration** (`pipeline.py`): `BM25RankingPipeline` ties the reader,
   the scorer, and the tokenizer together behind two methods —
   `search_tokens` (pre-tokenized terms, the ecosystem contract) and
   `search_text` (raw text, for standalone use) — and owns closing the
   underlying reader's resources.

## Architecture

```text
src/bm25_ranking_engine/
├── models.py                     # Posting, PostingsList, DocumentRecord, BM25Params, ScoredDocument
├── protocols.py                   # IndexReader interface
├── compressed_postings_codec.py    # decode-only re-implementation of index-compression-codec's binary format
├── index_reader.py                  # Plain, Compressed (v1/v2) and Segmented readers, open_index_reader()
├── tokenizer.py                       # minimal NFC + casefold tokenizer, for CLI convenience only
├── scoring.py                          # BM25Scorer: NumPy-vectorized scoring, strategy selection
├── dynamic_pruning.py                   # exact WAND and Block-Max WAND top-k
├── pipeline.py                          # BM25RankingPipeline: ties reader + scorer + tokenizer together
└── __main__.py                            # CLI (`search`, `benchmark`, `term-idf`)
```

## Requirements and installation

- Python `>=3.11`
- Runtime dependency: `numpy` — the one exception to this ecosystem's usual
  "no runtime dependencies" default, required explicitly by
  `beacon-search-engine`'s engineering standards for vectorized scoring
  (`../CLAUDE.md`, section 1).

```bash
git clone https://github.com/juanmmm21/bm25-ranking-engine.git
cd bm25-ranking-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage (CLI)

```bash
# Rank documents against a free-text query (tokenized internally)
bm25-ranking-engine search ./index "search engine ranking" --top-k 5

# Tune k1 (term frequency saturation) and b (length normalization)
bm25-ranking-engine search ./index "search engine" --k1 1.2 --b 0.6

# Same ranking with dynamic pruning (identical results, fewer documents scored)
bm25-ranking-engine search ./index "search engine" --strategy block_max_wand

# p50/p99 latency of every strategy over a file with one query per line;
# fails if any strategy returns a different top-k than the exhaustive one
bm25-ranking-engine benchmark ./index queries.txt --top-k 10 --repeat 3

# Inspect document_frequency and IDF for a single term
bm25-ranking-engine term-idf ./index search
```

Try it end-to-end against the tiny fixture checked into `samples/tiny_example/`
(a 3-document index already built by `inverted-index-builder`):

```bash
bm25-ranking-engine search samples/tiny_example/index "python search" --top-k 5
bm25-ranking-engine term-idf samples/tiny_example/index search
```

`./index` can be either an uncompressed index directory produced by
`inverted-index-builder` or a compressed one produced by
`index-compression-codec` — `search`/`term-idf` detect which from
`manifest.json` automatically. It can also be invoked as a module:
`python -m bm25_ranking_engine search ...`.

## Data formats

This project only reads existing index directories; see
[`inverted-index-builder`](https://github.com/juanmmm21/inverted-index-builder#data-formats)
and
[`index-compression-codec`](https://github.com/juanmmm21/index-compression-codec#data-formats)
for the exact schemas of `manifest.json`, `documents.jsonl`, `stats.json`,
`postings.jsonl` / `terms.jsonl` + `postings.bin`. Supported: the plain
format v1, compressed formats v1 and v2 (block-max, `index-compression-codec`
≥ 0.2.0) and the segmented format v1 (`segments.json`,
`inverted-index-builder` ≥ 0.2.0); any other version is rejected with an
explicit error instead of being misread. It produces no on-disk
format of its own — its output is an in-memory ranked list of
`ScoredDocument(doc_id, score)`.

## Dynamic pruning: WAND and Block-Max WAND

For a top-10 almost every candidate document is wasted work: what matters is
discarding early the ones that cannot enter the top-k. Both algorithms walk
the posting lists in `doc_id` order with one cursor per term and keep the
threshold `θ` (score of the current k-th best document):

- **WAND** (Broder et al., 2003) uses one upper bound per term. With cursors
  sorted by current document, the *pivot* is the first cursor where the sum of
  bounds exceeds `θ`; every earlier cursor jumps straight to the pivot's
  document without scoring anything in between.
- **Block-Max WAND** (Ding & Suel, 2011) adds one bound per block of postings.
  Before scoring the pivot it sums the bounds of the blocks that contain it;
  if they don't beat `θ`, it skips to the end of the shortest block. With a
  compressed v2 index the bounds come from the block directory on disk (no
  postings decoded); otherwise they are computed from the real scores.

**Same result as exhaustive evaluation, proven by tests.** Every strategy
reads the scores from the same per-term arrays and sums a document's terms in
the same order from the same `0.0`, so a pruned score is bit-for-bit the
exhaustive one. Bounds are inflated by a tiny relative margin so rounding can
never push one below a real score, and a document is only skipped when its
bound is `<= θ` (with documents visited in ascending `doc_id`, a later
document tying `θ` would lose the tie-break anyway).
`tests/test_dynamic_pruning.py` compares the three strategies on seeded random
corpora, on other `k1`/`b`, on a compressed v2 index and on a segmented index
with deletes; the `benchmark` command re-checks it on every query it times.

**Measured latency.** FiQA-2018 from BEIR (57,638 documents, 4.85 M postings,
648 test queries tokenized by this project's tokenizer), top-10, plain
in-memory index, 3 passes, AMD Ryzen 7 7700X under WSL2, Python 3.12.3,
NumPy 2.5.4, measured 2026-11-05:

- `0.1.0` scorer (per-posting Python dict accumulator): p50 41.32 ms,
  p99 77.77 ms.
- `0.2.0` exhaustive (cached arrays + `bincount` + `partition`): p50 0.67 ms,
  p99 1.15 ms, 49,759 documents scored per query on average. First pass with a
  cold term cache: p50 0.85 ms, p99 5.43 ms.
- `0.2.0` WAND: p50 51.99 ms, p99 215.51 ms, 1,720 documents scored per query.
- `0.2.0` Block-Max WAND: p50 92.90 ms, p99 410.66 ms, 1,115 documents scored
  per query (97.33 / 423.45 ms reading the bounds from a v2 compressed index).

The honest reading: pruning cuts the documents scored by 29× (WAND) and 45×
(Block-Max WAND), but in Python the per-document cursor loop costs far more
than NumPy scoring every candidate in one vectorized pass, so `exhaustive`
stays the default and the fastest strategy here. The pruning strategies earn
their keep where the per-document work is cheap (a compiled inner loop) or
where posting lists no longer fit a vectorized pass; they are kept exact and
tested so that switch is a strategy flag, not a rewrite.

## Programmatic usage

```python
from pathlib import Path

from bm25_ranking_engine.models import BM25Params
from bm25_ranking_engine.pipeline import BM25RankingPipeline

with BM25RankingPipeline(Path("./index"), BM25Params(k1=1.5, b=0.75)) as pipeline:
    # Primary contract: pre-tokenized terms (e.g. from query-parser-autocomplete)
    results = pipeline.search_tokens(["search", "engine"], top_k=10)
    for scored in results:
        document = pipeline.document(scored.doc_id)
        print(scored.doc_id, scored.score, document.title if document else None)

    # Convenience for standalone / CLI use: raw text, tokenized internally
    results = pipeline.search_text("search engine ranking", top_k=10)
```

## Development

```bash
pytest
ruff check .
mypy --strict src/
```

Tests cover the domain models (`BM25Params` validation), the compressed
postings decoder (hand-built VARINT and PACKED-mode blobs, truncation and
consistency errors), both index readers (plain and compressed, against small
on-disk fixtures), the tokenizer (NFC equivalence between composed and
decomposed accented forms, locale-aware case folding), the BM25 scorer (a
small fixed corpus where every expected score is precomputed independently
from the formula, covering single- and multi-term queries, candidate
restriction, and the ascending-`doc_id` tie-break), WAND and Block-Max WAND
against exhaustive evaluation (random corpora, compressed v2 bounds,
segmented indexes with tombstones), the end-to-end pipeline, and the CLI.

## Troubleshooting

- **`ValueError: average_document_length debe ser positiva ...`**: the index
  being queried has zero documents, or its `stats.json` reports a zero average
  length — BM25's length normalization is undefined in that case; rebuild the
  index from a non-empty corpus.
- **A term I expect to match returns no results**: this project's own
  `tokenize()` is a minimal NFC + casefold splitter meant for CLI convenience,
  not the ecosystem's query parser — it doesn't support phrase queries,
  exclusion, or any operator. If you're integrating programmatically, prefer
  `search_tokens` with terms already tokenized the same way
  `inverted-index-builder` tokenized the corpus.
- **`ValueError: inconsistencia de índice para '<term>'` while querying a
  compressed index**: the `document_frequency` recorded in that index's
  `terms.jsonl` doesn't match the `num_postings` encoded inside the
  corresponding `postings.bin` blob — the compressed index directory is
  corrupted or was assembled from mismatched files. Recompress it with
  `index-compression-codec`.
- **`FileNotFoundError` on an index directory that looks correct**: every
  reader requires `manifest.json` to exist and declare a `format_version` this
  project understands (via the file names it references) — verify the
  directory was produced by a compatible version of `inverted-index-builder`
  or `index-compression-codec`, not partially copied.
- **`mypy` fails under `tests/`**: expected — only `src/` is type-checked in
  `--strict` mode; test fixtures use deliberately looser typing.

## License

MIT — see [`LICENSE`](./LICENSE).
