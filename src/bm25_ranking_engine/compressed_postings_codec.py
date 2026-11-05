"""Reimplementación propia — solo de lectura — del formato binario de postings
comprimidas que documenta `index-compression-codec` (ver su README, sección
"Data formats", y su propio `postings_codec.py`). No se importa el paquete
Python de ese repo: la integración entre subproyectos de `beacon-search-engine`
es siempre vía formatos serializados en disco, nunca imports cruzados
(`../AGENTS.md`), así que este módulo reproduce, a partir de ese contrato ya
documentado y estable, únicamente el camino de decodificación que
`bm25-ranking-engine` necesita — este repo nunca escribe índices comprimidos,
solo los consulta.

Recordatorio del layout de un blob (uno por término, ver el README de
`index-compression-codec` para el detalle completo):

    byte    encoding_mode        0 = VARINT, 1 = PACKED
    varint  num_postings
    -- solo si num_postings > 0 --
    varint  first_doc_id
    ...     doc_id_gaps          num_postings - 1 valores, varint o bitpacked
    ...     term_frequencies     num_postings valores, siempre varint
    -- por posting, mismo orden que doc_id_gaps --
        varint  num_positions
        -- solo si num_positions > 0 --
        varint  first_position
        ...     position_gaps    num_positions - 1 valores, siempre varint
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from bm25_ranking_engine.models import Posting, PostingsList

_CONTINUATION_BIT = 0x80
_PAYLOAD_MASK = 0x7F
_PAYLOAD_BITS = 7

_MODE_VARINT = 0
_MODE_PACKED = 1


def decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decodifica un varint LEB128 a partir de `offset`. Devuelve `(valor, nuevo_offset)`."""
    result = 0
    shift = 0
    pos = offset
    while True:
        if pos >= len(data):
            raise ValueError("varint truncado: fin del buffer antes del byte de terminación")
        byte = data[pos]
        pos += 1
        result |= (byte & _PAYLOAD_MASK) << shift
        if not (byte & _CONTINUATION_BIT):
            break
        shift += _PAYLOAD_BITS
    return result, pos


def _decode_varint_block(data: bytes, offset: int, count: int) -> tuple[list[int], int]:
    values: list[int] = []
    pos = offset
    for _ in range(count):
        value, pos = decode_varint(data, pos)
        values.append(value)
    return values, pos


def _decode_packed_block(
    data: bytes, offset: int, count: int, block_size: int
) -> tuple[list[int], int]:
    """Decodifica `count` valores empaquetados en bloques de `block_size`.

    Cada bloque empieza con un byte de ancho de bits, seguido de los valores
    del bloque empaquetados a ese ancho exacto — mismo formato que
    `PackedBlockCodec` en `index-compression-codec`, reimplementado aquí en
    solitario porque este repo nunca necesita el lado de codificación.
    """
    values: list[int] = []
    pos = offset
    remaining = count
    while remaining > 0:
        chunk_len = min(block_size, remaining)
        if pos >= len(data):
            raise ValueError("bloque de bitpacking truncado: falta el byte de ancho de bits")
        bit_width = data[pos]
        pos += 1

        if bit_width == 0:
            values.extend([0] * chunk_len)
            remaining -= chunk_len
            continue

        total_bits = chunk_len * bit_width
        total_bytes = (total_bits + 7) // 8
        chunk = data[pos : pos + total_bytes]
        if len(chunk) < total_bytes:
            raise ValueError("bloque de bitpacking truncado: faltan bytes empaquetados")
        pos += total_bytes

        bit_buffer = int.from_bytes(chunk, byteorder="little")
        mask = (1 << bit_width) - 1
        for _ in range(chunk_len):
            values.append(bit_buffer & mask)
            bit_buffer >>= bit_width

        remaining -= chunk_len
    return values, pos


def _cumulative_sum(first: int, gaps: list[int]) -> list[int]:
    values = [first]
    for gap in gaps:
        values.append(values[-1] + gap)
    return values


def _decode_positions(data: bytes, offset: int) -> tuple[tuple[int, ...], int]:
    num_positions, pos = decode_varint(data, offset)
    if num_positions == 0:
        return (), pos
    first_position, pos = decode_varint(data, pos)
    gaps, pos = _decode_varint_block(data, pos, num_positions - 1)
    return tuple(_cumulative_sum(first_position, gaps)), pos


def decode_postings_list(
    term: str, document_frequency: int, data: bytes, *, block_size: int
) -> PostingsList:
    """Reconstruye la `PostingsList` de `term` a partir de su blob binario.

    `document_frequency` viene del directorio `terms.jsonl` de
    `index-compression-codec` y se contrasta contra el `num_postings`
    codificado dentro del propio blob: una discrepancia indica un índice
    corrupto o un directorio desincronizado con `postings.bin`, y se rechaza
    explícitamente en vez de devolver un resultado parcialmente incorrecto
    (mismo contrato que el `decode_postings_list` original).
    """
    if not data:
        raise ValueError(f"blob de postings vacío para '{term}': índice corrupto")

    mode = data[0]
    if mode not in (_MODE_VARINT, _MODE_PACKED):
        raise ValueError(f"modo de codificación desconocido para '{term}': {mode}")

    num_postings, pos = decode_varint(data, 1)
    if num_postings != document_frequency:
        raise ValueError(
            f"inconsistencia de índice para '{term}': el directorio declara "
            f"document_frequency={document_frequency}, el blob codifica {num_postings}"
        )

    if num_postings == 0:
        return PostingsList(term=term, document_frequency=0, postings=())

    first_doc_id, pos = decode_varint(data, pos)
    if mode == _MODE_PACKED:
        doc_id_gaps, pos = _decode_packed_block(data, pos, num_postings - 1, block_size)
    else:
        doc_id_gaps, pos = _decode_varint_block(data, pos, num_postings - 1)
    doc_ids = _cumulative_sum(first_doc_id, doc_id_gaps)

    frequencies, pos = _decode_varint_block(data, pos, num_postings)

    postings = []
    for doc_id, term_frequency in zip(doc_ids, frequencies, strict=True):
        positions, pos = _decode_positions(data, pos)
        postings.append(Posting(doc_id=doc_id, term_frequency=term_frequency, positions=positions))

    return PostingsList(term=term, document_frequency=num_postings, postings=tuple(postings))


# --- Formato v2 por bloques (block-max) --------------------------------------
#
# Reimplementación de solo lectura del layout v2 que documenta
# `index-compression-codec` desde 0.2.0 (README, "Block-max layout"):
#
#     varint  num_postings
#     varint  num_blocks
#     -- directorio, una entrada por bloque --
#     varint  last_doc_id_delta    respecto al último doc_id del bloque anterior
#                                  (el primero, desde 0)
#     varint  block_postings
#     varint  block_byte_length
#     varint  max_term_frequency
#     varint  min_document_length
#     4 bytes max_score            float32 little-endian, redondeado hacia arriba
#     -- bloques --
#     byte    encoding_mode
#     varint  first_doc_id_delta   primer doc_id menos el último del bloque anterior
#                                  (el primero, doc_id + 1)
#     ...     doc_id_gaps, term_frequencies, posiciones como en v1

_FLOAT32 = struct.Struct("<f")


@dataclass(frozen=True, slots=True)
class BlockDirectoryEntry:
    last_doc_id: int
    num_postings: int
    byte_length: int
    max_term_frequency: int
    min_document_length: int
    max_score: float


def decode_block_directory(data: bytes) -> tuple[int, list[BlockDirectoryEntry], int]:
    """`(num_postings, entradas, offset del primer bloque)`."""
    num_postings, pos = decode_varint(data, 0)
    num_blocks, pos = decode_varint(data, pos)
    entries: list[BlockDirectoryEntry] = []
    previous_last = 0
    for _ in range(num_blocks):
        delta, pos = decode_varint(data, pos)
        block_postings, pos = decode_varint(data, pos)
        byte_length, pos = decode_varint(data, pos)
        max_tf, pos = decode_varint(data, pos)
        min_length, pos = decode_varint(data, pos)
        if pos + _FLOAT32.size > len(data):
            raise ValueError("directorio de bloques truncado: falta max_score")
        (max_score,) = _FLOAT32.unpack_from(data, pos)
        pos += _FLOAT32.size
        previous_last += delta
        entries.append(
            BlockDirectoryEntry(
                last_doc_id=previous_last,
                num_postings=block_postings,
                byte_length=byte_length,
                max_term_frequency=max_tf,
                min_document_length=min_length,
                max_score=float(max_score),
            )
        )
    if sum(entry.num_postings for entry in entries) != num_postings:
        raise ValueError("directorio de bloques inconsistente con num_postings")
    return num_postings, entries, pos


def decode_block_postings_list(
    term: str, document_frequency: int, data: bytes, *, block_size: int
) -> PostingsList:
    num_postings, entries, pos = decode_block_directory(data)
    if num_postings != document_frequency:
        raise ValueError(
            f"inconsistencia de índice para '{term}': el directorio declara "
            f"document_frequency={document_frequency}, el blob codifica {num_postings}"
        )
    postings: list[Posting] = []
    previous_last = -1
    for entry in entries:
        block_start = pos
        mode = data[pos]
        if mode not in (_MODE_VARINT, _MODE_PACKED):
            raise ValueError(f"modo de codificación desconocido para '{term}': {mode}")
        first_delta, pos = decode_varint(data, pos + 1)
        if mode == _MODE_PACKED:
            gaps, pos = _decode_packed_block(data, pos, entry.num_postings - 1, block_size)
        else:
            gaps, pos = _decode_varint_block(data, pos, entry.num_postings - 1)
        doc_ids = _cumulative_sum(previous_last + first_delta, gaps)
        frequencies, pos = _decode_varint_block(data, pos, entry.num_postings)
        for doc_id, term_frequency in zip(doc_ids, frequencies, strict=True):
            positions, pos = _decode_positions(data, pos)
            postings.append(
                Posting(doc_id=doc_id, term_frequency=term_frequency, positions=positions)
            )
        if pos - block_start != entry.byte_length or doc_ids[-1] != entry.last_doc_id:
            raise ValueError(f"bloque corrupto en '{term}': no coincide con su directorio")
        previous_last = entry.last_doc_id
    return PostingsList(term=term, document_frequency=num_postings, postings=tuple(postings))
