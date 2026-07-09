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
