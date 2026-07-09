"""Tests de los tipos de dominio: sobre todo la validación de `BM25Params`,
que es la única con lógica no trivial (el resto son contenedores planos)."""

from __future__ import annotations

import pytest

from bm25_ranking_engine.models import (
    BM25Params,
    DocumentRecord,
    Posting,
    PostingsList,
    ScoredDocument,
)


def test_bm25_params_defaults_are_industry_standard() -> None:
    params = BM25Params()
    assert params.k1 == 1.5
    assert params.b == 0.75


def test_bm25_params_accepts_custom_values() -> None:
    params = BM25Params(k1=1.2, b=0.5)
    assert params.k1 == 1.2
    assert params.b == 0.5


def test_bm25_params_rejects_negative_k1() -> None:
    with pytest.raises(ValueError, match="k1"):
        BM25Params(k1=-0.1)


@pytest.mark.parametrize("b", [-0.01, 1.01])
def test_bm25_params_rejects_b_out_of_range(b: float) -> None:
    with pytest.raises(ValueError, match="b"):
        BM25Params(b=b)


def test_bm25_params_accepts_b_boundaries() -> None:
    assert BM25Params(b=0.0).b == 0.0
    assert BM25Params(b=1.0).b == 1.0


def test_posting_and_postings_list_are_immutable() -> None:
    posting = Posting(doc_id=0, term_frequency=2, positions=(0, 3))
    postings_list = PostingsList(term="search", document_frequency=1, postings=(posting,))
    assert postings_list.postings[0].positions == (0, 3)
    with pytest.raises(AttributeError):
        posting.doc_id = 1  # type: ignore[misc]


def test_document_record_and_scored_document_are_plain_containers() -> None:
    record = DocumentRecord(doc_id=1, url="http://example.com", title="Example", length=42)
    scored = ScoredDocument(doc_id=1, score=3.14)
    assert record.doc_id == scored.doc_id
    assert record.length == 42
    assert scored.score == 3.14
