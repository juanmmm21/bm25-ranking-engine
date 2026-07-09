"""Lectores de índice para bm25-ranking-engine: dos implementaciones del mismo
`protocols.IndexReader`, para las dos formas en que puede llegar el índice
(`../AGENTS.md`, integración: "el índice invertido (comprimido, vía
index-compression-codec, o sin comprimir directamente desde
inverted-index-builder en desarrollo)"):

- `PlainIndexReader` lee el índice sin comprimir de `inverted-index-builder`
  (`documents.jsonl`, `postings.jsonl`, `stats.json`), cargando todas las
  posting lists en memoria de una vez: ese formato no tiene un directorio de
  offsets del que hacer acceso aleatorio, así que no hay nada que ganar
  leyendo bajo demanda.
- `CompressedIndexReader` lee el índice comprimido de `index-compression-codec`
  (`terms.jsonl` + `postings.bin`), decodificando bajo demanda solo los bytes
  del término consultado (ver `compressed_postings_codec.py`) — reimplementado
  aquí sin importar el paquete Python de ese repo (`../AGENTS.md`).

`open_index_reader` decide cuál instanciar inspeccionando `manifest.json`: el
manifiesto comprimido declara `terms_file` (el directorio de términos que hace
posible el acceso aleatorio), el manifiesto sin comprimir no.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from bm25_ranking_engine.compressed_postings_codec import decode_postings_list
from bm25_ranking_engine.models import DocumentRecord, Posting, PostingsList
from bm25_ranking_engine.protocols import IndexReader

_MANIFEST_FILENAME: Final[str] = "manifest.json"


def _read_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no existe {manifest_path}: ¿es un directorio de índice válido?")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest


def _load_documents(
    index_dir: Path, documents_filename: str
) -> tuple[dict[int, DocumentRecord], NDArray[np.int64]]:
    documents: dict[int, DocumentRecord] = {}
    documents_path = index_dir / documents_filename
    for line in documents_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        record = DocumentRecord(
            doc_id=int(data["doc_id"]),
            url=str(data["url"]),
            title=str(data["title"]),
            length=int(data["length"]),
        )
        documents[record.doc_id] = record

    if not documents:
        return documents, np.zeros(0, dtype=np.int64)

    # Indexado directo por `doc_id` (no por posición de inserción): los IDs son
    # la posición de línea original en `inverted-index-builder`, así que este
    # array debe tener ese mismo tamaño aunque, en teoría, pudiera haber huecos.
    lengths = np.zeros(max(documents) + 1, dtype=np.int64)
    for doc_id, record in documents.items():
        lengths[doc_id] = record.length
    return documents, lengths


def _read_stats(index_dir: Path, stats_filename: str) -> tuple[int, float]:
    stats_path = index_dir / stats_filename
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return int(stats["total_documents"]), float(stats["average_document_length"])


class PlainIndexReader:
    """Lee un índice sin comprimir de `inverted-index-builder`.

    Carga documentos, estadísticas y todas las posting lists en memoria en el
    constructor: ese formato serializa `postings.jsonl` como texto plano línea
    a línea, sin ningún directorio de offsets, así que no existe un camino de
    acceso aleatorio que aprovechar — a diferencia de `CompressedIndexReader`.
    """

    def __init__(self, index_dir: Path) -> None:
        manifest = _read_manifest(index_dir)
        self._documents, self._document_lengths = _load_documents(
            index_dir, str(manifest["documents_file"])
        )
        self._total_documents, self._average_document_length = _read_stats(
            index_dir, str(manifest["stats_file"])
        )
        self._postings_lists = self._load_postings(index_dir / str(manifest["postings_file"]))

    @staticmethod
    def _load_postings(postings_path: Path) -> dict[str, PostingsList]:
        postings_lists: dict[str, PostingsList] = {}
        for line in postings_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            postings = tuple(
                Posting(
                    doc_id=int(entry["doc_id"]),
                    term_frequency=int(entry["term_frequency"]),
                    positions=tuple(int(position) for position in entry["positions"]),
                )
                for entry in data["postings"]
            )
            postings_list = PostingsList(
                term=str(data["term"]),
                document_frequency=int(data["document_frequency"]),
                postings=postings,
            )
            postings_lists[postings_list.term] = postings_list
        return postings_lists

    @property
    def total_documents(self) -> int:
        return self._total_documents

    @property
    def average_document_length(self) -> float:
        return self._average_document_length

    @property
    def document_lengths(self) -> NDArray[np.int64]:
        return self._document_lengths

    def lookup(self, term: str) -> PostingsList | None:
        return self._postings_lists.get(term)

    def document(self, doc_id: int) -> DocumentRecord | None:
        return self._documents.get(doc_id)

    def close(self) -> None:
        """No-op: este lector no mantiene ningún descriptor de archivo abierto."""
        return None


class CompressedIndexReader:
    """Lee un índice comprimido de `index-compression-codec` con acceso aleatorio.

    `lookup` decodifica únicamente los bytes del término consultado —
    localizados vía el directorio `terms.jsonl`, cargado en memoria en el
    constructor (proporcional al vocabulario, no a los postings) — sin
    descomprimir el resto del índice, igual que el lector original de ese
    repo (ver `compressed_postings_codec.py`).
    """

    def __init__(self, index_dir: Path) -> None:
        manifest = _read_manifest(index_dir)
        self._block_size = int(manifest["block_size"])
        self._documents, self._document_lengths = _load_documents(
            index_dir, str(manifest["documents_file"])
        )
        self._total_documents, self._average_document_length = _read_stats(
            index_dir, str(manifest["stats_file"])
        )
        self._terms = self._load_terms_directory(index_dir / str(manifest["terms_file"]))
        self._file = (index_dir / str(manifest["postings_file"])).open("rb")

    @staticmethod
    def _load_terms_directory(terms_path: Path) -> dict[str, tuple[int, int, int]]:
        entries: dict[str, tuple[int, int, int]] = {}
        for line in terms_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            entries[str(data["term"])] = (
                int(data["offset"]),
                int(data["length"]),
                int(data["document_frequency"]),
            )
        return entries

    @property
    def total_documents(self) -> int:
        return self._total_documents

    @property
    def average_document_length(self) -> float:
        return self._average_document_length

    @property
    def document_lengths(self) -> NDArray[np.int64]:
        return self._document_lengths

    def lookup(self, term: str) -> PostingsList | None:
        entry = self._terms.get(term)
        if entry is None:
            return None

        offset, length, document_frequency = entry
        self._file.seek(offset)
        data = self._file.read(length)
        if len(data) != length:
            raise ValueError(
                f"lectura truncada para '{term}': esperados {length} bytes, leídos {len(data)}"
            )
        return decode_postings_list(term, document_frequency, data, block_size=self._block_size)

    def document(self, doc_id: int) -> DocumentRecord | None:
        return self._documents.get(doc_id)

    def close(self) -> None:
        self._file.close()


def open_index_reader(index_dir: Path) -> IndexReader:
    """Instancia el lector adecuado inspeccionando `manifest.json`.

    Si `index_dir` resulta ser un índice comprimido, el `CompressedIndexReader`
    devuelto mantiene abierto un descriptor de archivo de `postings.bin` — el
    llamador es responsable de invocar `.close()` cuando termine (`pipeline.py`
    lo hace vía `contextlib.closing`), igual que exige `../CLAUDE.md`, sección 2.E.
    """
    manifest = _read_manifest(index_dir)
    if "terms_file" in manifest:
        return CompressedIndexReader(index_dir)
    return PlainIndexReader(index_dir)
