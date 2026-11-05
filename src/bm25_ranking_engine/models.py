"""Tipos de datos del dominio para bm25-ranking-engine.

`Posting`, `PostingsList` y `DocumentRecord` son copias propias, mínimas, de los
mismos conceptos expuestos por `inverted-index-builder` e `index-compression-codec`:
no se importa el paquete Python de ninguno de los dos (la integración entre
subproyectos de `beacon-search-engine` es siempre vía formatos serializados en
disco, nunca imports cruzados — ver `../AGENTS.md`), así que este repo define su
propia forma mínima de esos datos a partir de lo que necesita para puntuar, no de
lo que esos repos podrían exponer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Posting:
    """Aparición de un término en un documento concreto: frecuencia y posiciones.

    Las posiciones se conservan aunque BM25 no las use directamente (solo
    necesita `term_frequency`), porque ambos formatos de índice de origen las
    incluyen en cada posting y descartarlas al leer no ahorraría nada: dejarlas
    disponibles es lo que permitiría añadir proximidad de términos más adelante
    sin cambiar el formato leído.
    """

    doc_id: int
    term_frequency: int
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PostingsList:
    """Lista de postings de un término, ordenada por `doc_id` ascendente."""

    term: str
    document_frequency: int
    postings: tuple[Posting, ...]


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Metadatos mínimos de un documento indexado: longitud para BM25, resto para mostrar."""

    doc_id: int
    url: str
    title: str
    length: int


@dataclass(frozen=True, slots=True)
class BM25Params:
    """Parámetros libres de la fórmula BM25.

    `k1` controla la saturación de la frecuencia de término: cuanto mayor, más
    tarda en saturar el aporte de repetir un término dentro del mismo
    documento. `b` controla cuánto penaliza la normalización por longitud de
    documento (`0.0` = sin normalizar en absoluto, `1.0` = normalización
    completa por longitud relativa a la media del corpus). `k1=1.5, b=0.75`
    son los valores por defecto de facto de la industria (Lucene,
    Elasticsearch), validados empíricamente sobre las colecciones TREC durante
    más de dos décadas — no un punto de partida arbitrario, aunque siguen
    siendo parametrizables porque el ajuste óptimo depende del corpus real.
    """

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError(f"k1 debe ser no negativo, recibido {self.k1}")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError(f"b debe estar en el rango [0, 1], recibido {self.b}")


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    """Un documento candidato junto a su score BM25 frente a una query."""

    doc_id: int
    score: float


@dataclass(frozen=True, slots=True)
class TermBlockBounds:
    """Cotas por bloque de la posting list de un término: el bloque `i`
    termina en `last_doc_ids[i]` y ningún documento suyo supera
    `max_tf_parts[i]` en la parte de BM25 que no depende del IDF
    (`tf * (k1 + 1) / (tf + k1 * norm(len))`)."""

    last_doc_ids: NDArray[np.int64]
    max_tf_parts: NDArray[np.float64]
