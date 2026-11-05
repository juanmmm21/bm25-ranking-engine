"""Núcleo de scoring BM25.

Para cada término de la query, `BM25Scorer` resuelve su posting list en el
índice y calcula la contribución de ese término a cada documento donde aparece
con operaciones vectorizadas de NumPy sobre arrays de `doc_id`/
`term_frequency`/longitud de documento — nunca un bucle Python por documento
(`../CLAUDE.md`, sección 1: "cálculo vectorizado"). Combinar las contribuciones
de varios términos de una misma query sí se hace con un diccionario Python
normal: una query real tiene un puñado de términos, no miles, así que iterarlos
uno a uno no es el cuello de botella — el que sí lo sería es, por término,
recorrer con un bucle interpretado los (potencialmente miles o millones de)
documentos donde aparece.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bm25_ranking_engine.dynamic_pruning import (
    PruningStats,
    RetrievalStrategy,
    TermPostings,
    blocks_from_scores,
    pruned_top_k,
)
from bm25_ranking_engine.models import BM25Params, ScoredDocument
from bm25_ranking_engine.protocols import IndexReader


class BM25Scorer:
    """Calcula scores BM25 de los documentos candidatos frente a una query.

    El IDF usa la variante con suavizado +1 de Lucene/Elasticsearch —
    `ln(1 + (N - df + 0.5) / (df + 0.5))` — en vez de la fórmula clásica de
    Robertson-Sparck Jones (`ln((N - df + 0.5) / (df + 0.5))`, sin el +1): la
    clásica se vuelve negativa para términos que aparecen en más de la mitad
    de los documentos del corpus, lo que penalizaría a un documento por
    contener un término común en vez de simplemente no premiarlo por él — el
    +1 mantiene el IDF siempre no negativo sin alterar el orden relativo entre
    términos de distinta frecuencia.
    """

    def __init__(
        self,
        reader: IndexReader,
        params: BM25Params | None = None,
        *,
        max_cached_terms: int = 4096,
    ) -> None:
        """`max_cached_terms` acota la caché LRU de arrays por término (ver
        `_term_arrays`): sin tope crecería hasta duplicar el índice en memoria."""
        if max_cached_terms < 0:
            raise ValueError(f"max_cached_terms debe ser >= 0, recibido {max_cached_terms}")
        self._reader = reader
        self._params = params if params is not None else BM25Params()
        self._max_cached_terms = max_cached_terms
        self._cache: OrderedDict[str, _TermArrays | None] = OrderedDict()

    @property
    def params(self) -> BM25Params:
        return self._params

    def idf(self, document_frequency: int) -> float:
        """IDF con suavizado +1 para `document_frequency` documentos que contienen el término."""
        if document_frequency <= 0:
            raise ValueError(f"document_frequency debe ser positivo, recibido {document_frequency}")
        n = self._reader.total_documents
        return math.log(1.0 + (n - document_frequency + 0.5) / (document_frequency + 0.5))

    def _term_arrays(self, term: str) -> _TermArrays | None:
        """`doc_id` y contribución BM25 (sin la frecuencia en la query) de
        `term`, calculadas una vez y reutilizadas entre queries: convertir una
        posting list de objetos `Posting` a arrays es O(df) en Python puro y
        dominaba el coste de las queries con términos frecuentes."""
        if term in self._cache:
            self._cache.move_to_end(term)
            return self._cache[term]
        arrays = self._compute_term_arrays(term)
        if self._max_cached_terms:
            self._cache[term] = arrays
            if len(self._cache) > self._max_cached_terms:
                self._cache.popitem(last=False)
        return arrays

    def _compute_term_arrays(self, term: str) -> _TermArrays | None:
        postings_list = self._reader.lookup(term)
        if postings_list is None or postings_list.document_frequency == 0:
            return None
        avgdl = self._reader.average_document_length
        lengths = self._reader.document_lengths
        k1, b = self._params.k1, self._params.b
        doc_ids = np.fromiter(
            (posting.doc_id for posting in postings_list.postings),
            dtype=np.int64,
            count=len(postings_list.postings),
        )
        term_frequencies = np.fromiter(
            (posting.term_frequency for posting in postings_list.postings),
            dtype=np.float64,
            count=len(postings_list.postings),
        )
        document_lengths = lengths[doc_ids].astype(np.float64)
        length_norm = 1.0 - b + b * (document_lengths / avgdl)
        numerator = term_frequencies * (k1 + 1.0)
        denominator = term_frequencies + k1 * length_norm
        idf = self.idf(postings_list.document_frequency)
        scores = idf * (numerator / denominator)
        return _TermArrays(term=term, idf=idf, doc_ids=doc_ids, scores=scores)

    def term_score_arrays(
        self, query_terms: Sequence[str]
    ) -> list[tuple[str, NDArray[np.int64], NDArray[np.float64]]]:
        """Por término distinto de la query (en orden de primera aparición),
        sus `doc_id` y la contribución BM25 exacta a cada uno, ya multiplicada
        por la frecuencia del término en la query. Es la única fuente de
        scores de las tres estrategias: así WAND y Block-Max WAND suman
        exactamente los mismos valores que la evaluación exhaustiva."""
        if not query_terms or self._reader.total_documents == 0:
            return []
        avgdl = self._reader.average_document_length
        if avgdl <= 0:
            raise ValueError(
                "average_document_length debe ser positiva para normalizar por longitud "
                f"(recibida {avgdl}); ¿es un índice vacío?"
            )
        result: list[tuple[str, NDArray[np.int64], NDArray[np.float64]]] = []
        for term, query_term_frequency in Counter(query_terms).items():
            arrays = self._term_arrays(term)
            if arrays is None:
                continue
            scores = (
                arrays.scores * query_term_frequency if query_term_frequency > 1 else arrays.scores
            )
            result.append((term, arrays.doc_ids, scores))
        return result

    def score(
        self,
        query_terms: Sequence[str],
        *,
        candidate_doc_ids: Iterable[int] | None = None,
        top_k: int | None = None,
        strategy: RetrievalStrategy = RetrievalStrategy.EXHAUSTIVE,
    ) -> list[ScoredDocument]:
        """Puntúa `query_terms` frente al índice y devuelve el ranking descendente.

        `candidate_doc_ids`, si se pasa, restringe el scoring a ese subconjunto
        de documentos (útil cuando otro módulo aguas abajo, como
        `distributed-index-sharding`, ya ha preseleccionado candidatos).
        `top_k`, si se pasa, trunca el resultado a los `top_k` mejores. El
        desempate entre scores iguales es por `doc_id` ascendente, no el orden
        de inserción del diccionario interno, para que el resultado sea
        determinista sin depender de en qué orden llegaron los términos
        (`../CLAUDE.md`, sección 2.B).

        `strategy` elige cómo se calcula el top-k (`dynamic_pruning.py`): las
        tres devuelven exactamente el mismo resultado. WAND y Block-Max WAND
        solo podan cuando hay un `top_k` y ningún `candidate_doc_ids`; en
        cualquier otro caso no hay umbral que aprovechar y se evalúa de forma
        exhaustiva.
        """
        return self.score_with_stats(
            query_terms, candidate_doc_ids=candidate_doc_ids, top_k=top_k, strategy=strategy
        )[0]

    def score_with_stats(
        self,
        query_terms: Sequence[str],
        *,
        candidate_doc_ids: Iterable[int] | None = None,
        top_k: int | None = None,
        strategy: RetrievalStrategy = RetrievalStrategy.EXHAUSTIVE,
    ) -> tuple[list[ScoredDocument], PruningStats]:
        """Como `score`, más cuántos documentos se puntuaron por completo."""
        arrays = self.term_score_arrays(query_terms)
        prunable = (
            strategy is not RetrievalStrategy.EXHAUSTIVE
            and top_k is not None
            and candidate_doc_ids is None
        )
        if prunable and top_k is not None:
            terms = [
                TermPostings(order, doc_ids, scores, *self._block_bounds(term, doc_ids, scores))
                for order, (term, doc_ids, scores) in enumerate(arrays)
            ]
            return pruned_top_k(
                terms, top_k, block_max=strategy is RetrievalStrategy.BLOCK_MAX_WAND
            )

        candidate_array = (
            np.fromiter(sorted(set(candidate_doc_ids)), dtype=np.int64)
            if candidate_doc_ids is not None
            else None
        )
        id_parts: list[NDArray[np.int64]] = []
        score_parts: list[NDArray[np.float64]] = []
        for _term, doc_ids, term_scores in arrays:
            if candidate_array is not None:
                keep_mask = np.isin(doc_ids, candidate_array)
                doc_ids = doc_ids[keep_mask]
                term_scores = term_scores[keep_mask]
            if doc_ids.size:
                id_parts.append(doc_ids)
                score_parts.append(term_scores)
        if not id_parts:
            return [], PruningStats(scored_documents=0, candidate_documents=0)

        # `bincount` suma los pesos en el orden en que aparecen, empezando en
        # 0.0: con los términos concatenados en el orden de la query, cada
        # documento acumula exactamente la misma secuencia de sumas que el
        # antiguo diccionario por documento — mismo float bit a bit, pero sin
        # un bucle Python por posting.
        all_ids = np.concatenate(id_parts)
        totals = np.bincount(all_ids, weights=np.concatenate(score_parts))
        touched = np.zeros(totals.size, dtype=bool)
        touched[all_ids] = True
        candidates = np.flatnonzero(touched)
        candidate_scores = totals[candidates]
        evaluated = int(candidates.size)
        if top_k is not None and top_k < candidates.size:
            # Preselección O(n) del top-k; se conservan todos los empatados con
            # el k-ésimo para que el desempate por doc_id lo resuelva lexsort.
            kth = np.partition(candidate_scores, candidates.size - top_k)[candidates.size - top_k]
            keep = candidate_scores >= kth
            candidates, candidate_scores = candidates[keep], candidate_scores[keep]
        order = np.lexsort((candidates, -candidate_scores))
        if top_k is not None:
            order = order[:top_k]
        ranked = [
            ScoredDocument(doc_id=int(candidates[index]), score=float(candidate_scores[index]))
            for index in order
        ]
        return ranked, PruningStats(scored_documents=evaluated, candidate_documents=evaluated)

    def _block_bounds(
        self, term: str, doc_ids: NDArray[np.int64], scores: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Cotas por bloque en unidades de score. Si el lector trae cotas
        precalculadas (formato v2 de `index-compression-codec`) se usan esas,
        multiplicadas por el IDF y la frecuencia en la query de este scorer;
        si no, se calculan exactas a partir de `scores`."""
        provider = getattr(self._reader, "block_bounds", None)
        arrays = self._term_arrays(term)
        if provider is not None and arrays is not None and scores.size:
            bounds = provider(term, self._params)
            if bounds is not None:
                # scores = idf * tf_part * qtf: el cociente con los scores base
                # recupera la frecuencia en la query sin volver a contarla.
                query_factor = float(scores[0] / arrays.scores[0])
                return bounds.last_doc_ids, bounds.max_tf_parts * arrays.idf * query_factor
        return blocks_from_scores(doc_ids, scores)


@dataclass(frozen=True, slots=True)
class _TermArrays:
    term: str
    idf: float
    doc_ids: NDArray[np.int64]
    scores: NDArray[np.float64]
