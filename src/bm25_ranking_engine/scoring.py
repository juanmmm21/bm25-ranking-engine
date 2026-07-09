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
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

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

    def __init__(self, reader: IndexReader, params: BM25Params | None = None) -> None:
        self._reader = reader
        self._params = params if params is not None else BM25Params()

    @property
    def params(self) -> BM25Params:
        return self._params

    def idf(self, document_frequency: int) -> float:
        """IDF con suavizado +1 para `document_frequency` documentos que contienen el término."""
        if document_frequency <= 0:
            raise ValueError(f"document_frequency debe ser positivo, recibido {document_frequency}")
        n = self._reader.total_documents
        return math.log(1.0 + (n - document_frequency + 0.5) / (document_frequency + 0.5))

    def score(
        self,
        query_terms: Sequence[str],
        *,
        candidate_doc_ids: Iterable[int] | None = None,
        top_k: int | None = None,
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
        """
        if not query_terms or self._reader.total_documents == 0:
            return []

        avgdl = self._reader.average_document_length
        if avgdl <= 0:
            raise ValueError(
                "average_document_length debe ser positiva para normalizar por longitud "
                f"(recibida {avgdl}); ¿es un índice vacío?"
            )

        candidate_array = (
            np.fromiter(sorted(set(candidate_doc_ids)), dtype=np.int64)
            if candidate_doc_ids is not None
            else None
        )

        lengths = self._reader.document_lengths
        k1, b = self._params.k1, self._params.b
        accumulator: dict[int, float] = {}

        for term, query_term_frequency in Counter(query_terms).items():
            postings_list = self._reader.lookup(term)
            if postings_list is None or postings_list.document_frequency == 0:
                continue

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

            if candidate_array is not None:
                keep_mask = np.isin(doc_ids, candidate_array)
                doc_ids = doc_ids[keep_mask]
                term_frequencies = term_frequencies[keep_mask]
                if doc_ids.size == 0:
                    continue

            document_lengths = lengths[doc_ids].astype(np.float64)
            length_norm = 1.0 - b + b * (document_lengths / avgdl)
            numerator = term_frequencies * (k1 + 1.0)
            denominator = term_frequencies + k1 * length_norm
            term_scores = self.idf(postings_list.document_frequency) * (numerator / denominator)
            if query_term_frequency > 1:
                term_scores = term_scores * query_term_frequency

            for doc_id, term_score in zip(doc_ids.tolist(), term_scores.tolist(), strict=True):
                accumulator[doc_id] = accumulator.get(doc_id, 0.0) + term_score

        ranked = sorted(accumulator.items(), key=lambda item: (-item[1], item[0]))
        if top_k is not None:
            ranked = ranked[:top_k]
        return [ScoredDocument(doc_id=doc_id, score=score) for doc_id, score in ranked]
