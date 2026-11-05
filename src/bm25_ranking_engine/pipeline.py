"""Orquesta lector de índice + tokenizador + scorer para responder una query de
principio a fin: abre el índice (delegando en `index_reader.open_index_reader`
qué formato es, comprimido o no), puntúa con `BM25Scorer.score` y devuelve el
top-K ya ordenado.

El contrato principal de este proyecto, documentado en `../AGENTS.md`
("Consume ... una query dada, ya tokenizada por `query-parser-autocomplete`"),
es recibir términos ya tokenizados: `search_tokens` es el método pensado para
integrarse con ese proyecto cuando exista. `search_text` existe solo para que
la CLI de este repo sea usable de forma standalone hoy, apoyándose en el
tokenizador mínimo propio de `tokenizer.py`, que no pretende sustituir al de
`query-parser-autocomplete`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

from bm25_ranking_engine.dynamic_pruning import RetrievalStrategy
from bm25_ranking_engine.index_reader import open_index_reader
from bm25_ranking_engine.models import BM25Params, DocumentRecord, ScoredDocument
from bm25_ranking_engine.protocols import IndexReader
from bm25_ranking_engine.scoring import BM25Scorer
from bm25_ranking_engine.tokenizer import tokenize


class BM25RankingPipeline:
    """Punto de entrada programático único: abre el índice, expone búsqueda y
    cierra el lector explícitamente al terminar (gestor de contexto o
    `.close()` manual — ver `../CLAUDE.md`, sección 2.E)."""

    def __init__(
        self,
        index_dir: Path,
        params: BM25Params | None = None,
        strategy: RetrievalStrategy = RetrievalStrategy.EXHAUSTIVE,
    ) -> None:
        """`strategy` elige cómo se calcula el top-k (`dynamic_pruning.py`);
        las tres dan exactamente el mismo resultado."""
        self._reader: IndexReader = open_index_reader(index_dir)
        self._scorer = BM25Scorer(self._reader, params)
        self._strategy = strategy

    def __enter__(self) -> BM25RankingPipeline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._reader.close()

    @property
    def params(self) -> BM25Params:
        return self._scorer.params

    def search_tokens(self, query_terms: Sequence[str], *, top_k: int = 10) -> list[ScoredDocument]:
        """Contrato principal: puntúa una query ya tokenizada (por
        `query-parser-autocomplete` en producción) y devuelve el top-K."""
        return self._scorer.score(query_terms, top_k=top_k, strategy=self._strategy)

    def search_text(self, query_text: str, *, top_k: int = 10) -> list[ScoredDocument]:
        """Conveniencia para uso standalone: tokeniza `query_text` con el
        tokenizador mínimo propio de este repo antes de puntuar."""
        return self.search_tokens(tokenize(query_text), top_k=top_k)

    def document(self, doc_id: int) -> DocumentRecord | None:
        return self._reader.document(doc_id)

    def term_document_frequency_and_idf(self, term: str) -> tuple[int, float] | None:
        """Metadatos de depuración de un término: cuántos documentos lo contienen
        y su IDF resultante, para explicar por qué un documento puntúa como puntúa."""
        postings_list = self._reader.lookup(term)
        if postings_list is None:
            return None
        return postings_list.document_frequency, self._scorer.idf(postings_list.document_frequency)
