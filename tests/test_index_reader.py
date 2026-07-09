"""Tests de los dos lectores de índice contra fixtures pequeñas en disco,
escritas a mano con el formato exacto documentado por `inverted-index-builder`
(plano) e `index-compression-codec` (comprimido) — sin importar ninguno de los
dos paquetes, igual que hace el propio código de producción."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bm25_ranking_engine.index_reader import (
    CompressedIndexReader,
    PlainIndexReader,
    open_index_reader,
)

_DOCUMENTS_JSONL = (
    '{"doc_id": 0, "url": "http://example.com/a", "title": "A", "length": 3}\n'
    '{"doc_id": 1, "url": "http://example.com/b", "title": "B", "length": 4}\n'
)
_STATS_JSON = {
    "total_documents": 2,
    "vocabulary_size": 1,
    "total_postings": 1,
    "average_document_length": 3.5,
}


def _varint(value: int) -> bytes:
    result = bytearray()
    remaining = value
    while True:
        byte = remaining & 0x7F
        remaining >>= 7
        if remaining:
            result.append(byte | 0x80)
        else:
            result.append(byte)
            break
    return bytes(result)


def _write_plain_index(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "documents.jsonl").write_text(_DOCUMENTS_JSONL, encoding="utf-8")
    (index_dir / "stats.json").write_text(json.dumps(_STATS_JSON), encoding="utf-8")

    postings_line = json.dumps(
        {
            "term": "search",
            "document_frequency": 2,
            "postings": [
                {"doc_id": 0, "term_frequency": 1, "positions": [2]},
                {"doc_id": 1, "term_frequency": 2, "positions": [0, 5]},
            ],
        }
    )
    (index_dir / "postings.jsonl").write_text(postings_line + "\n", encoding="utf-8")

    manifest = {
        "format_version": 1,
        "documents_file": "documents.jsonl",
        "postings_file": "postings.jsonl",
        "stats_file": "stats.json",
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_compressed_index(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "documents.jsonl").write_text(_DOCUMENTS_JSONL, encoding="utf-8")
    (index_dir / "stats.json").write_text(json.dumps(_STATS_JSON), encoding="utf-8")

    # doc_ids = [0, 1], tf = [1, 2], posiciones = [(2,), (0, 5)] — modo VARINT.
    blob = bytearray()
    blob.append(0)  # encoding_mode = VARINT
    blob.extend(_varint(2))  # num_postings
    blob.extend(_varint(0))  # first_doc_id
    blob.extend(_varint(1))  # gap doc_id 0 -> 1
    blob.extend(_varint(1))  # tf posting 0
    blob.extend(_varint(2))  # tf posting 1
    blob.extend(_varint(1))  # posting 0: 1 posición
    blob.extend(_varint(2))  # posting 0: posición 2
    blob.extend(_varint(2))  # posting 1: 2 posiciones
    blob.extend(_varint(0))  # posting 1: primera posición 0
    blob.extend(_varint(5))  # posting 1: gap 5

    (index_dir / "postings.bin").write_bytes(bytes(blob))

    terms_line = json.dumps(
        {"term": "search", "offset": 0, "length": len(blob), "document_frequency": 2}
    )
    (index_dir / "terms.jsonl").write_text(terms_line + "\n", encoding="utf-8")

    manifest = {
        "format_version": 1,
        "documents_file": "documents.jsonl",
        "stats_file": "stats.json",
        "terms_file": "terms.jsonl",
        "postings_file": "postings.bin",
        "compression_stats_file": "compression_stats.json",
        "block_size": 128,
        "bitpack_threshold": 64,
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_plain_index_reader_reads_documents_stats_and_postings(tmp_path: Path) -> None:
    index_dir = tmp_path / "plain-index"
    _write_plain_index(index_dir)

    reader = PlainIndexReader(index_dir)
    try:
        assert reader.total_documents == 2
        assert reader.average_document_length == 3.5
        assert np.array_equal(reader.document_lengths, np.array([3, 4]))

        postings_list = reader.lookup("search")
        assert postings_list is not None
        assert postings_list.document_frequency == 2
        assert [p.doc_id for p in postings_list.postings] == [0, 1]
        assert reader.lookup("missing") is None

        document = reader.document(0)
        assert document is not None
        assert document.title == "A"
        assert reader.document(99) is None
    finally:
        reader.close()  # no-op, pero debe ser seguro llamarlo


def test_compressed_index_reader_decodes_only_looked_up_term(tmp_path: Path) -> None:
    index_dir = tmp_path / "compressed-index"
    _write_compressed_index(index_dir)

    reader = CompressedIndexReader(index_dir)
    try:
        assert reader.total_documents == 2
        assert reader.average_document_length == 3.5
        assert np.array_equal(reader.document_lengths, np.array([3, 4]))

        postings_list = reader.lookup("search")
        assert postings_list is not None
        assert postings_list.document_frequency == 2
        assert [p.doc_id for p in postings_list.postings] == [0, 1]
        assert [p.term_frequency for p in postings_list.postings] == [1, 2]
        assert postings_list.postings[0].positions == (2,)
        assert postings_list.postings[1].positions == (0, 5)
        assert reader.lookup("missing") is None

        document = reader.document(1)
        assert document is not None
        assert document.title == "B"
    finally:
        reader.close()


def test_open_index_reader_dispatches_to_plain_reader(tmp_path: Path) -> None:
    index_dir = tmp_path / "plain-index"
    _write_plain_index(index_dir)

    reader = open_index_reader(index_dir)
    try:
        assert isinstance(reader, PlainIndexReader)
    finally:
        reader.close()


def test_open_index_reader_dispatches_to_compressed_reader(tmp_path: Path) -> None:
    index_dir = tmp_path / "compressed-index"
    _write_compressed_index(index_dir)

    reader = open_index_reader(index_dir)
    try:
        assert isinstance(reader, CompressedIndexReader)
    finally:
        reader.close()


def test_missing_manifest_raises_file_not_found(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        open_index_reader(empty_dir)
