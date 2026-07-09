"""Interfaz que desacopla `scoring.BM25Scorer` de cómo se lee físicamente el
índice invertido, para poder puntuar exactamente igual sobre el índice sin
comprimir de `inverted-index-builder` o sobre el índice comprimido de
`index-compression-codec` (ver `../AGENTS.md`, integración) sin que el propio
scorer conozca ninguno de los dos formatos en disco — esa responsabilidad es de
`index_reader.py`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from bm25_ranking_engine.models import DocumentRecord, PostingsList


@runtime_checkable
class IndexReader(Protocol):
    """Acceso de solo lectura a un índice invertido ya construido."""

    @property
    def total_documents(self) -> int:
        """Número total de documentos del corpus (`N` en la fórmula BM25)."""
        ...

    @property
    def average_document_length(self) -> float:
        """Longitud media de documento del corpus, en tokens indexados."""
        ...

    @property
    def document_lengths(self) -> NDArray[np.int64]:
        """Longitudes de documento indexadas por `doc_id` (`0..total_documents-1`).

        Precalculado una vez al abrir el índice para que `BM25Scorer` pueda
        resolver la longitud de un lote entero de `doc_id`s con indexado
        vectorizado de NumPy (`document_lengths[doc_ids]`), en vez de resolver
        la longitud documento a documento con un bucle Python.
        """
        ...

    def lookup(self, term: str) -> PostingsList | None:
        """Postings de `term`, o `None` si no aparece en el vocabulario del índice."""
        ...

    def document(self, doc_id: int) -> DocumentRecord | None:
        """Metadatos de `doc_id`, o `None` si no existe en el índice."""
        ...

    def close(self) -> None:
        """Libera los recursos del lector (descriptores de archivo abiertos).

        No todas las implementaciones mantienen un recurso que cerrar
        (`PlainIndexReader` carga todo en memoria de una vez), pero el método
        es parte del contrato para que el llamador pueda cerrar cualquier
        `IndexReader` de forma uniforme sin conocer la implementación
        concreta (`../CLAUDE.md`, sección 2.E).
        """
        ...
