"""Tests del núcleo de scoring BM25 contra un corpus pequeño y determinista
donde cada score esperado se ha precalculado de forma independiente a partir
de la fórmula (ver el módulo `_expected` más abajo, ejecutado aparte para
generar los literales, no invocado por el propio test).

Corpus (N=4 documentos, average_document_length=4.0):
    doc0: length=4  — contiene "cat" (tf=1)
    doc1: length=2  — contiene "cat" (tf=2) y "dog" (tf=1)
    doc2: length=6  — contiene "dog" (tf=3)
    doc3: length=4  — no contiene ni "cat" ni "dog"

Con k1=1.5, b=0.75 (valores por defecto) y document_frequency=2 para ambos
términos, el IDF con suavizado +1 da exactamente ln(2) — de ahí que este
corpus se eligiera así, para que los números de referencia sean verificables
a mano sin arrastrar demasiados decimales.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from bm25_ranking_engine.models import BM25Params, DocumentRecord, Posting, PostingsList
from bm25_ranking_engine.scoring import BM25Scorer

_DOC0_CAT_SCORE = 0.6931471805599453
_DOC1_CAT_PLUS_DOG_SCORE = 2.074208446974167
_DOC2_DOG_SCORE = 1.0268847119406597


class _FakeIndexReader:
    """Doble de test de `protocols.IndexReader`: expone el corpus descrito
    arriba sin pasar por ningún formato en disco, para aislar la aritmética de
    `BM25Scorer` de la lectura de índices (ya cubierta en `test_index_reader.py`)."""

    def __init__(self) -> None:
        self._documents = {
            0: DocumentRecord(doc_id=0, url="http://example.com/0", title="Doc 0", length=4),
            1: DocumentRecord(doc_id=1, url="http://example.com/1", title="Doc 1", length=2),
            2: DocumentRecord(doc_id=2, url="http://example.com/2", title="Doc 2", length=6),
            3: DocumentRecord(doc_id=3, url="http://example.com/3", title="Doc 3", length=4),
        }
        self._postings = {
            "cat": PostingsList(
                term="cat",
                document_frequency=2,
                postings=(
                    Posting(doc_id=0, term_frequency=1, positions=(0,)),
                    Posting(doc_id=1, term_frequency=2, positions=(0, 1)),
                ),
            ),
            "dog": PostingsList(
                term="dog",
                document_frequency=2,
                postings=(
                    Posting(doc_id=1, term_frequency=1, positions=(1,)),
                    Posting(doc_id=2, term_frequency=3, positions=(0, 2, 4)),
                ),
            ),
        }

    @property
    def total_documents(self) -> int:
        return 4

    @property
    def average_document_length(self) -> float:
        return 4.0

    @property
    def document_lengths(self) -> NDArray[np.int64]:
        return np.array([4, 2, 6, 4], dtype=np.int64)

    def lookup(self, term: str) -> PostingsList | None:
        return self._postings.get(term)

    def document(self, doc_id: int) -> DocumentRecord | None:
        return self._documents.get(doc_id)

    def close(self) -> None:
        return None


@pytest.fixture
def scorer() -> BM25Scorer:
    return BM25Scorer(_FakeIndexReader(), BM25Params())


def test_idf_with_plus_one_smoothing() -> None:
    scorer = BM25Scorer(_FakeIndexReader(), BM25Params())
    assert math.isclose(scorer.idf(2), math.log(2.0), rel_tol=1e-12)


def test_idf_rejects_non_positive_document_frequency() -> None:
    scorer = BM25Scorer(_FakeIndexReader(), BM25Params())
    with pytest.raises(ValueError, match="document_frequency"):
        scorer.idf(0)


def test_single_term_query_scores_and_ranks(scorer: BM25Scorer) -> None:
    results = scorer.score(["cat"])
    scores_by_doc = {r.doc_id: r.score for r in results}

    assert set(scores_by_doc) == {0, 1}
    assert math.isclose(scores_by_doc[0], _DOC0_CAT_SCORE, rel_tol=1e-9)
    # doc1 tiene tf=2 y menor longitud relativa: debe puntuar más alto que doc0.
    assert scores_by_doc[1] > scores_by_doc[0]
    # Orden descendente por score.
    assert [r.doc_id for r in results] == [1, 0]


def test_multi_term_query_sums_contributions_per_document(scorer: BM25Scorer) -> None:
    results = scorer.score(["cat", "dog"])
    scores_by_doc = {r.doc_id: r.score for r in results}

    assert set(scores_by_doc) == {0, 1, 2}
    assert math.isclose(scores_by_doc[0], _DOC0_CAT_SCORE, rel_tol=1e-9)
    assert math.isclose(scores_by_doc[1], _DOC1_CAT_PLUS_DOG_SCORE, rel_tol=1e-9)
    assert math.isclose(scores_by_doc[2], _DOC2_DOG_SCORE, rel_tol=1e-9)
    # doc1 coincide con ambos términos: debe ser el primero del ranking.
    assert [r.doc_id for r in results] == [1, 2, 0]


def test_term_not_in_vocabulary_is_ignored(scorer: BM25Scorer) -> None:
    results = scorer.score(["bird"])
    assert results == []


def test_empty_query_returns_empty_list(scorer: BM25Scorer) -> None:
    assert scorer.score([]) == []


def test_top_k_truncates_results(scorer: BM25Scorer) -> None:
    results = scorer.score(["cat", "dog"], top_k=1)
    assert len(results) == 1
    assert results[0].doc_id == 1


def test_candidate_doc_ids_restricts_scoring(scorer: BM25Scorer) -> None:
    results = scorer.score(["cat", "dog"], candidate_doc_ids=[0, 2])
    scores_by_doc = {r.doc_id: r.score for r in results}

    assert set(scores_by_doc) == {0, 2}
    assert math.isclose(scores_by_doc[0], _DOC0_CAT_SCORE, rel_tol=1e-9)
    assert math.isclose(scores_by_doc[2], _DOC2_DOG_SCORE, rel_tol=1e-9)


def test_score_is_deterministic_across_repeated_calls(scorer: BM25Scorer) -> None:
    first = scorer.score(["cat", "dog"])
    second = scorer.score(["cat", "dog"])
    assert first == second
