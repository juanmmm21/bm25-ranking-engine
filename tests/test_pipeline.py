"""Tests end-to-end de `BM25RankingPipeline` sobre un índice plano pequeño en
disco, reutilizando el mismo corpus (N=4, avgdl=4.0) y los mismos valores
esperados que `test_scoring.py`, para verificar que abrir el índice de verdad
desde disco produce el mismo ranking que puntuar directamente contra el doble
de test."""

from __future__ import annotations

import json
import math
from pathlib import Path

from bm25_ranking_engine.pipeline import BM25RankingPipeline

_DOC1_CAT_PLUS_DOG_SCORE = 2.074208446974167


def _write_index(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    documents = [
        {"doc_id": 0, "url": "http://example.com/0", "title": "About Cats", "length": 4},
        {"doc_id": 1, "url": "http://example.com/1", "title": "Cats and Dogs", "length": 2},
        {"doc_id": 2, "url": "http://example.com/2", "title": "About Dogs", "length": 6},
        {"doc_id": 3, "url": "http://example.com/3", "title": "Unrelated", "length": 4},
    ]
    (index_dir / "documents.jsonl").write_text(
        "\n".join(json.dumps(doc) for doc in documents) + "\n", encoding="utf-8"
    )
    (index_dir / "stats.json").write_text(
        json.dumps(
            {
                "total_documents": 4,
                "vocabulary_size": 2,
                "total_postings": 4,
                "average_document_length": 4.0,
            }
        ),
        encoding="utf-8",
    )

    postings = [
        {
            "term": "cat",
            "document_frequency": 2,
            "postings": [
                {"doc_id": 0, "term_frequency": 1, "positions": [0]},
                {"doc_id": 1, "term_frequency": 2, "positions": [0, 1]},
            ],
        },
        {
            "term": "dog",
            "document_frequency": 2,
            "postings": [
                {"doc_id": 1, "term_frequency": 1, "positions": [1]},
                {"doc_id": 2, "term_frequency": 3, "positions": [0, 2, 4]},
            ],
        },
    ]
    (index_dir / "postings.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in postings) + "\n", encoding="utf-8"
    )

    manifest = {
        "format_version": 1,
        "documents_file": "documents.jsonl",
        "postings_file": "postings.jsonl",
        "stats_file": "stats.json",
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_search_tokens_end_to_end(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    _write_index(index_dir)

    with BM25RankingPipeline(index_dir) as pipeline:
        results = pipeline.search_tokens(["cat", "dog"])
        assert [r.doc_id for r in results] == [1, 2, 0]
        assert math.isclose(results[0].score, _DOC1_CAT_PLUS_DOG_SCORE, rel_tol=1e-9)

        top_document = pipeline.document(results[0].doc_id)
        assert top_document is not None
        assert top_document.title == "Cats and Dogs"


def test_search_text_tokenizes_free_text_query(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    _write_index(index_dir)

    with BM25RankingPipeline(index_dir) as pipeline:
        results_text = pipeline.search_text("Cat, Dog!")
        results_tokens = pipeline.search_tokens(["cat", "dog"])
        assert results_text == results_tokens


def test_term_document_frequency_and_idf(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    _write_index(index_dir)

    with BM25RankingPipeline(index_dir) as pipeline:
        result = pipeline.term_document_frequency_and_idf("cat")
        assert result is not None
        document_frequency, idf = result
        assert document_frequency == 2
        assert math.isclose(idf, math.log(2.0), rel_tol=1e-12)

        assert pipeline.term_document_frequency_and_idf("missing") is None


def test_top_k_is_respected_end_to_end(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    _write_index(index_dir)

    with BM25RankingPipeline(index_dir) as pipeline:
        results = pipeline.search_tokens(["cat", "dog"], top_k=1)
        assert len(results) == 1
        assert results[0].doc_id == 1
