"""WAND y Block-Max WAND frente a la evaluación exhaustiva: mismo top-k, mismo
orden y los mismos floats bit a bit, sobre corpus aleatorios con semilla fija
(incluidos documentos duplicados que empatan en score), sobre un índice
comprimido v2 con cotas precalculadas y sobre un índice segmentado."""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import numpy as np
import pytest

from bm25_ranking_engine.dynamic_pruning import RetrievalStrategy, blocks_from_scores
from bm25_ranking_engine.index_reader import (
    CompressedIndexReader,
    PlainIndexReader,
    SegmentedIndexReader,
    open_index_reader,
)
from bm25_ranking_engine.models import BM25Params
from bm25_ranking_engine.scoring import BM25Scorer

_VOCABULARY = [f"w{i}" for i in range(40)]


def _random_corpus(seed: int, num_docs: int) -> list[list[str]]:
    rng = random.Random(seed)
    # Zipf aproximado: unos pocos términos muy frecuentes y una cola larga.
    weights = [1.0 / (rank + 1) for rank in range(len(_VOCABULARY))]
    documents = [rng.choices(_VOCABULARY, weights, k=rng.randint(3, 60)) for _ in range(num_docs)]
    # Duplicados exactos: empates de score que solo el desempate por doc_id ordena.
    for index in range(0, num_docs, 17):
        documents[(index + 5) % num_docs] = list(documents[index])
    return documents


def _write_plain_index(index_dir: Path, documents: list[list[str]], first_doc_id: int = 0) -> None:
    index_dir.mkdir(parents=True)
    postings: dict[str, dict[int, list[int]]] = {}
    with (index_dir / "documents.jsonl").open("w", encoding="utf-8") as documents_file:
        for offset, tokens in enumerate(documents):
            doc_id = first_doc_id + offset
            record = {"doc_id": doc_id, "url": f"u{doc_id}", "title": "", "length": len(tokens)}
            documents_file.write(json.dumps(record) + "\n")
            for position, token in enumerate(tokens):
                postings.setdefault(token, {}).setdefault(doc_id, []).append(position)
    with (index_dir / "postings.jsonl").open("w", encoding="utf-8") as postings_file:
        for term in sorted(postings):
            by_doc = postings[term]
            entry = {
                "term": term,
                "document_frequency": len(by_doc),
                "postings": [
                    {"doc_id": d, "term_frequency": len(by_doc[d]), "positions": by_doc[d]}
                    for d in sorted(by_doc)
                ],
            }
            postings_file.write(json.dumps(entry) + "\n")
    total = sum(len(tokens) for tokens in documents)
    stats = {
        "total_documents": len(documents),
        "vocabulary_size": len(postings),
        "total_postings": sum(len(v) for v in postings.values()),
        "average_document_length": total / len(documents),
    }
    (index_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    manifest = {
        "format_version": 1,
        "documents_file": "documents.jsonl",
        "postings_file": "postings.jsonl",
        "stats_file": "stats.json",
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _queries(seed: int, count: int) -> list[list[str]]:
    rng = random.Random(seed)
    return [rng.choices(_VOCABULARY, k=rng.randint(1, 6)) for _ in range(count)]


def _assert_identical(scorer: BM25Scorer, query: list[str], top_k: int) -> None:
    exhaustive = scorer.score(query, top_k=top_k)
    for strategy in (RetrievalStrategy.WAND, RetrievalStrategy.BLOCK_MAX_WAND):
        pruned = scorer.score(query, top_k=top_k, strategy=strategy)
        assert [(d.doc_id, d.score) for d in pruned] == [(d.doc_id, d.score) for d in exhaustive], (
            f"{strategy} difiere en query={query} top_k={top_k}"
        )


@pytest.mark.parametrize("seed", range(6))
def test_pruning_matches_exhaustive_on_random_corpora(tmp_path: Path, seed: int) -> None:
    _write_plain_index(tmp_path / "index", _random_corpus(seed, 400))
    scorer = BM25Scorer(PlainIndexReader(tmp_path / "index"))
    for query in _queries(seed + 100, 40):
        for top_k in (1, 3, 10, 50):
            _assert_identical(scorer, query, top_k)


def test_pruning_matches_exhaustive_with_other_bm25_params(tmp_path: Path) -> None:
    _write_plain_index(tmp_path / "index", _random_corpus(9, 300))
    scorer = BM25Scorer(PlainIndexReader(tmp_path / "index"), BM25Params(k1=0.9, b=0.4))
    for query in _queries(9, 30):
        _assert_identical(scorer, query, 10)


def test_block_max_wand_scores_fewer_documents(tmp_path: Path) -> None:
    _write_plain_index(tmp_path / "index", _random_corpus(3, 3000))
    scorer = BM25Scorer(PlainIndexReader(tmp_path / "index"))
    query = ["w0", "w1", "w2", "w30"]
    _, exhaustive = scorer.score_with_stats(query, top_k=10)
    _, wand = scorer.score_with_stats(query, top_k=10, strategy=RetrievalStrategy.WAND)
    _, bmw = scorer.score_with_stats(query, top_k=10, strategy=RetrievalStrategy.BLOCK_MAX_WAND)
    assert exhaustive.scored_documents == exhaustive.candidate_documents
    assert bmw.scored_documents <= wand.scored_documents <= exhaustive.scored_documents
    assert bmw.scored_documents < exhaustive.scored_documents


def test_without_top_k_or_with_candidates_falls_back_to_exhaustive(tmp_path: Path) -> None:
    _write_plain_index(tmp_path / "index", _random_corpus(1, 50))
    scorer = BM25Scorer(PlainIndexReader(tmp_path / "index"))
    query = ["w0", "w3"]
    assert scorer.score(query, strategy=RetrievalStrategy.WAND) == scorer.score(query)
    candidates = [1, 2, 3, 40]
    assert scorer.score(
        query, candidate_doc_ids=candidates, top_k=2, strategy=RetrievalStrategy.BLOCK_MAX_WAND
    ) == scorer.score(query, candidate_doc_ids=candidates, top_k=2)
    assert scorer.score([], top_k=3, strategy=RetrievalStrategy.WAND) == []
    assert scorer.score(["missing"], top_k=3, strategy=RetrievalStrategy.WAND) == []


def test_blocks_from_scores() -> None:
    doc_ids = np.array([1, 4, 9, 12, 20], dtype=np.int64)
    scores = np.array([0.5, 2.0, 1.0, 0.1, 3.0])
    last, maxima = blocks_from_scores(doc_ids, scores, block_size=2)
    assert last.tolist() == [4, 12, 20]
    assert maxima.tolist() == [2.0, 1.0, 3.0]


# --- índice comprimido v2 (cotas leídas de disco) ------------------------------


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _float32_ceil(value: float) -> float:
    (rounded,) = struct.unpack("<f", struct.pack("<f", value))
    if rounded >= value:
        return float(rounded)
    (bits,) = struct.unpack("<I", struct.pack("<f", rounded))
    return float(struct.unpack("<f", struct.pack("<I", bits + 1))[0])


def _write_v2_index(plain_dir: Path, out_dir: Path, block_size: int, params: BM25Params) -> None:
    """Escritor mínimo del formato v2 de index-compression-codec, solo para
    el test: este repo nunca escribe índices comprimidos."""
    out_dir.mkdir()
    lengths = {}
    for line in (plain_dir / "documents.jsonl").read_text("utf-8").splitlines():
        record = json.loads(line)
        lengths[record["doc_id"]] = record["length"]
    stats = json.loads((plain_dir / "stats.json").read_text("utf-8"))
    avgdl = stats["average_document_length"]
    terms = []
    blob = bytearray()
    for line in (plain_dir / "postings.jsonl").read_text("utf-8").splitlines():
        entry = json.loads(line)
        postings = entry["postings"]
        directory = bytearray()
        blocks = bytearray()
        previous_last = -1
        previous_dir_last = 0
        term_max = 0.0
        chunks = [postings[i : i + block_size] for i in range(0, len(postings), block_size)]
        for chunk in chunks:
            block = bytearray([0])
            block += _varint(chunk[0]["doc_id"] - previous_last)
            for a, b in zip(chunk, chunk[1:], strict=False):
                block += _varint(b["doc_id"] - a["doc_id"])
            for posting in chunk:
                block += _varint(posting["term_frequency"])
            for posting in chunk:
                positions = posting["positions"]
                block += _varint(len(positions)) + _varint(positions[0])
                for a, b in zip(positions, positions[1:], strict=False):
                    block += _varint(b - a)
            tf_parts = [
                p["term_frequency"]
                * (params.k1 + 1.0)
                / (
                    p["term_frequency"]
                    + params.k1 * (1.0 - params.b + params.b * (lengths[p["doc_id"]] / avgdl))
                )
                for p in chunk
            ]
            max_score = _float32_ceil(max(tf_parts))
            term_max = max(term_max, max_score)
            last = chunk[-1]["doc_id"]
            directory += (
                _varint(last - previous_dir_last) + _varint(len(chunk)) + _varint(len(block))
            )
            directory += _varint(max(p["term_frequency"] for p in chunk))
            directory += _varint(min(lengths[p["doc_id"]] for p in chunk))
            directory += struct.pack("<f", max_score)
            blocks += block
            previous_last = last
            previous_dir_last = last
        data = _varint(len(postings)) + _varint(len(chunks)) + bytes(directory) + bytes(blocks)
        terms.append(
            {
                "term": entry["term"],
                "offset": len(blob),
                "length": len(data),
                "document_frequency": len(postings),
                "max_score": term_max,
            }
        )
        blob += data
    (out_dir / "postings.bin").write_bytes(bytes(blob))
    (out_dir / "terms.jsonl").write_text("".join(json.dumps(t) + "\n" for t in terms), "utf-8")
    for name in ("documents.jsonl", "stats.json"):
        (out_dir / name).write_text((plain_dir / name).read_text("utf-8"), "utf-8")
    manifest = {
        "format_version": 2,
        "documents_file": "documents.jsonl",
        "stats_file": "stats.json",
        "terms_file": "terms.jsonl",
        "postings_file": "postings.bin",
        "compression_stats_file": "compression_stats.json",
        "block_size": block_size,
        "bitpack_threshold": 1_000_000,
        "block_max": {
            "scoring": "bm25",
            "k1": params.k1,
            "b": params.b,
            "average_document_length": avgdl,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest), "utf-8")


@pytest.mark.parametrize("params", [BM25Params(), BM25Params(k1=1.2, b=0.5)])
def test_compressed_v2_index_reads_and_prunes_identically(
    tmp_path: Path, params: BM25Params
) -> None:
    _write_plain_index(tmp_path / "plain", _random_corpus(4, 500))
    _write_v2_index(tmp_path / "plain", tmp_path / "v2", block_size=16, params=BM25Params())
    plain = PlainIndexReader(tmp_path / "plain")
    compressed = CompressedIndexReader(tmp_path / "v2")
    try:
        for term in _VOCABULARY:
            assert compressed.lookup(term) == plain.lookup(term)
        bounds = compressed.block_bounds("w0", BM25Params())
        assert bounds is not None and bounds.last_doc_ids.size > 1
        scorer = BM25Scorer(compressed, params)
        reference = BM25Scorer(plain, params)
        for query in _queries(4, 30):
            for top_k in (1, 10):
                _assert_identical(scorer, query, top_k)
                assert scorer.score(query, top_k=top_k) == reference.score(query, top_k=top_k)
    finally:
        compressed.close()


# --- índice segmentado de inverted-index-builder ------------------------------


def test_segmented_index_scores_like_the_equivalent_single_index(tmp_path: Path) -> None:
    corpus = _random_corpus(5, 120)
    root = tmp_path / "seg"
    root.mkdir()
    _write_plain_index(root / "seg-000001", corpus[:60])
    _write_plain_index(root / "seg-000002", corpus[60:], first_doc_id=60)
    (root / "deletes").mkdir()
    deleted = [3, 61, 90]
    (root / "deletes" / "seg-000001-g000003.json").write_text(
        json.dumps({"segment": "seg-000001", "deleted_doc_ids": [3]}), "utf-8"
    )
    (root / "deletes" / "seg-000002-g000003.json").write_text(
        json.dumps({"segment": "seg-000002", "deleted_doc_ids": [61, 90]}), "utf-8"
    )
    manifest = {
        "format": "beacon-segmented-index",
        "format_version": 1,
        "generation": 3,
        "next_doc_id": 120,
        "next_segment_number": 3,
        "segments": [
            {
                "name": "seg-000001",
                "doc_count": 60,
                "deleted_count": 1,
                "deletes_file": "deletes/seg-000001-g000003.json",
            },
            {
                "name": "seg-000002",
                "doc_count": 60,
                "deleted_count": 2,
                "deletes_file": "deletes/seg-000002-g000003.json",
            },
        ],
    }
    (root / "segments.json").write_text(json.dumps(manifest), "utf-8")

    segmented = open_index_reader(root)
    assert isinstance(segmented, SegmentedIndexReader)
    assert segmented.total_documents == 117
    assert segmented.document(3) is None

    # Índice único equivalente con los mismos doc_id: el segmentado no tiene
    # huecos en sus segmentos, así que se escribe y se borra lo mismo a mano.
    _write_plain_index(tmp_path / "single", corpus)
    single = PlainIndexReader(tmp_path / "single")
    live_terms = {t for i, doc in enumerate(corpus) if i not in deleted for t in doc}
    for term in live_terms:
        reference = single.lookup(term)
        assert reference is not None
        expected = [p for p in reference.postings if p.doc_id not in deleted]
        actual = segmented.lookup(term)
        assert actual is not None and list(actual.postings) == expected
    scorer = BM25Scorer(segmented)
    for query in _queries(5, 20):
        _assert_identical(scorer, query, 10)
        assert all(d.doc_id not in deleted for d in scorer.score(query, top_k=50))


def test_segmented_reader_rejects_unknown_version(tmp_path: Path) -> None:
    (tmp_path / "segments.json").write_text(
        json.dumps({"format": "beacon-segmented-index", "format_version": 7, "segments": []}),
        "utf-8",
    )
    with pytest.raises(ValueError):
        SegmentedIndexReader(tmp_path)
