"""Top-k con poda dinámica: WAND (Broder et al., 2003) y Block-Max WAND
(Ding y Suel, 2011).

La evaluación exhaustiva (`scoring.BM25Scorer`, estrategia `exhaustive`)
puntúa todos los documentos que contienen algún término de la query. Para un
top-k casi todos sobran: lo que importa es descartar pronto los que no pueden
entrar. Ambos algoritmos recorren las posting lists en orden de `doc_id` con
un cursor por término y mantienen el umbral `θ` = score del k-ésimo mejor
documento visto hasta ahora:

- **WAND** usa una cota por término (el mayor score que el término aporta a
  cualquier documento). Con los cursores ordenados por documento actual, el
  *pivote* es el primer cursor en el que la suma de cotas supera `θ`: ningún
  documento anterior al del pivote puede superar el umbral, así que los
  cursores previos saltan directamente hasta él sin puntuar nada en medio.
- **Block-Max WAND** añade cotas por bloque de postings. Antes de puntuar el
  pivote suma las cotas de los bloques que lo contienen, mucho más ajustadas
  que las del término entero; si no superan `θ`, salta hasta el final del
  bloque más corto sin puntuar ningún documento intermedio.

## Por qué el resultado es idéntico al exhaustivo

- Los scores por posting son los mismos arrays que calcula
  `BM25Scorer.term_score_arrays`, y el score final de un documento se suma en
  el mismo orden de términos y desde el mismo `0.0` que la acumulación
  exhaustiva: mismo valor en coma flotante, bit a bit.
- Las cotas se inflan con un margen relativo minúsculo (`_inflate`), de modo
  que un redondeo al sumarlas nunca las deja por debajo de un score real.
- Un documento solo se descarta si su cota es `<= θ`. Como se recorre en
  orden creciente de `doc_id` y el desempate del ranking es `doc_id`
  ascendente, un documento posterior con score igual a `θ` tampoco habría
  entrado: descartarlo no cambia el top-k.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from bm25_ranking_engine.models import ScoredDocument

DEFAULT_BLOCK_SIZE = 64
"""Mismo tamaño de bloque por defecto que el formato v2 de
`index-compression-codec`, para que las cotas calculadas al vuelo y las
leídas del índice describan bloques equivalentes."""

_EXHAUSTED = sys.maxsize
_RELATIVE_MARGIN = 1e-9
_ABSOLUTE_MARGIN = 1e-12


class RetrievalStrategy(StrEnum):
    EXHAUSTIVE = "exhaustive"
    WAND = "wand"
    BLOCK_MAX_WAND = "block_max_wand"


@dataclass(frozen=True, slots=True)
class TermPostings:
    """Lo que la poda necesita de un término de la query."""

    order: int
    """Posición del término en el orden de acumulación del scorer."""
    doc_ids: NDArray[np.int64]
    scores: NDArray[np.float64]
    """Contribución exacta del término a cada documento de `doc_ids`."""
    block_last_doc_ids: NDArray[np.int64]
    block_max_scores: NDArray[np.float64]
    """Cota (en unidades de score, ya con IDF) de cada bloque."""


@dataclass(frozen=True, slots=True)
class PruningStats:
    scored_documents: int
    """Documentos puntuados por completo."""
    candidate_documents: int
    """Documentos con al menos un término: los que puntúa el exhaustivo."""


def _inflate(value: float) -> float:
    return value * (1.0 + _RELATIVE_MARGIN) + _ABSOLUTE_MARGIN


def blocks_from_scores(
    doc_ids: NDArray[np.int64], scores: NDArray[np.float64], block_size: int = DEFAULT_BLOCK_SIZE
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Cotas por bloque exactas calculadas a partir de los scores reales,
    para lectores que no traen cotas precalculadas en disco."""
    if block_size < 1:
        raise ValueError(f"block_size debe ser >= 1, recibido {block_size}")
    starts = np.arange(0, doc_ids.size, block_size, dtype=np.int64)
    ends = np.minimum(starts + block_size, doc_ids.size) - 1
    return doc_ids[ends], np.maximum.reduceat(scores, starts)


class _Cursor:
    __slots__ = (
        "block_last",
        "block_max",
        "doc",
        "doc_ids",
        "max_score",
        "order",
        "position",
        "scores",
        "size",
    )

    def __init__(self, term: TermPostings) -> None:
        self.order = term.order
        self.doc_ids = term.doc_ids
        self.scores = term.scores
        self.size = int(term.doc_ids.size)
        self.block_last = term.block_last_doc_ids
        self.block_max = term.block_max_scores
        self.max_score = _inflate(float(term.scores.max())) if self.size else 0.0
        self.position = 0
        self.doc = int(term.doc_ids[0]) if self.size else _EXHAUSTED

    def advance_to(self, target: int) -> None:
        if self.doc >= target:
            return
        offset = int(np.searchsorted(self.doc_ids[self.position :], target, side="left"))
        self.position += offset
        self.doc = int(self.doc_ids[self.position]) if self.position < self.size else _EXHAUSTED

    def next(self) -> None:
        self.position += 1
        self.doc = int(self.doc_ids[self.position]) if self.position < self.size else _EXHAUSTED

    def block_bound(self, doc_id: int) -> tuple[float, int]:
        """`(cota del bloque que contiene doc_id, primer doc_id después de ese
        bloque)`. Si la lista termina antes de `doc_id`, el término ya no
        aporta nada: cota 0 y sin límite."""
        block = int(np.searchsorted(self.block_last, doc_id, side="left"))
        if block >= self.block_last.size:
            return 0.0, _EXHAUSTED
        return _inflate(float(self.block_max[block])), int(self.block_last[block]) + 1


def pruned_top_k(
    terms: Sequence[TermPostings], top_k: int, *, block_max: bool
) -> tuple[list[ScoredDocument], PruningStats]:
    """Top-`top_k` exacto con WAND (`block_max=False`) o Block-Max WAND."""
    if top_k < 1:
        raise ValueError(f"top_k debe ser >= 1, recibido {top_k}")
    cursors = [_Cursor(term) for term in terms if term.doc_ids.size]
    candidate_documents = (
        int(np.unique(np.concatenate([term.doc_ids for term in terms])).size) if cursors else 0
    )
    heap: list[tuple[float, int]] = []
    scored = 0

    while True:
        cursors = [cursor for cursor in cursors if cursor.doc != _EXHAUSTED]
        if not cursors:
            break
        cursors.sort(key=lambda cursor: cursor.doc)
        threshold = heap[0][0] if len(heap) == top_k else -math.inf

        upper_bound = 0.0
        pivot = -1
        for index, cursor in enumerate(cursors):
            upper_bound += cursor.max_score
            if upper_bound > threshold:
                pivot = index
                break
        if pivot < 0:
            break
        pivot_doc = cursors[pivot].doc
        while pivot + 1 < len(cursors) and cursors[pivot + 1].doc == pivot_doc:
            pivot += 1

        if block_max and threshold > -math.inf:
            block_upper_bound = 0.0
            next_boundary = _EXHAUSTED
            for cursor in cursors[: pivot + 1]:
                bound, boundary = cursor.block_bound(pivot_doc)
                block_upper_bound += bound
                next_boundary = min(next_boundary, boundary)
            if block_upper_bound <= threshold:
                target = next_boundary
                if pivot + 1 < len(cursors):
                    target = min(target, cursors[pivot + 1].doc)
                target = max(target, pivot_doc + 1)
                for cursor in cursors[: pivot + 1]:
                    cursor.advance_to(target)
                continue

        if cursors[0].doc == pivot_doc:
            matching = sorted(cursors[: pivot + 1], key=lambda cursor: cursor.order)
            score = 0.0
            for cursor in matching:
                score += float(cursor.scores[cursor.position])
            scored += 1
            entry = (score, -pivot_doc)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif score > threshold:
                heapq.heapreplace(heap, entry)
            for cursor in matching:
                cursor.next()
        else:
            for cursor in cursors[:pivot]:
                cursor.advance_to(pivot_doc)

    ranked = sorted(
        ((score, -negated) for score, negated in heap), key=lambda item: (-item[0], item[1])
    )
    return (
        [ScoredDocument(doc_id=doc_id, score=score) for score, doc_id in ranked],
        PruningStats(scored_documents=scored, candidate_documents=candidate_documents),
    )
