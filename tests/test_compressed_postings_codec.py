"""Tests del decodificador de postings comprimidas.

No hay codificador en este repo (solo lee índices comprimidos, nunca los
escribe — ver `compressed_postings_codec.py`), así que las fixtures binarias se
construyen a mano byte a byte, exactamente con el mismo layout que documenta
`index-compression-codec`: esto verifica el decoder contra el formato
documentado, no contra un codificador hermano que aquí no existe.
"""

from __future__ import annotations

import pytest

from bm25_ranking_engine.compressed_postings_codec import decode_postings_list, decode_varint

_MODE_VARINT = 0
_MODE_PACKED = 1


def _varint(value: int) -> bytes:
    """Codifica un entero no negativo como varint LEB128 (helper de test)."""
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


def test_decode_varint_single_and_multi_byte_values() -> None:
    assert decode_varint(_varint(0) + b"\x00", 0) == (0, 1)
    assert decode_varint(_varint(127), 0) == (127, 1)
    assert decode_varint(_varint(128), 0) == (128, 2)
    assert decode_varint(_varint(300), 0) == (300, 2)


def test_decode_varint_truncated_buffer_raises() -> None:
    with pytest.raises(ValueError, match="varint truncado"):
        decode_varint(bytes([0x80]), 0)


def test_decode_postings_list_varint_mode() -> None:
    # doc_ids = [0, 3], frecuencias = [1, 2], posiciones = [(0,), (0, 4)]
    blob = bytearray()
    blob.append(_MODE_VARINT)
    blob.extend(_varint(2))  # num_postings
    blob.extend(_varint(0))  # first_doc_id
    blob.extend(_varint(3))  # gap único: doc_id 0 -> 3
    blob.extend(_varint(1))  # tf del posting 0
    blob.extend(_varint(2))  # tf del posting 1
    # posting 0: 1 posición, valor 0
    blob.extend(_varint(1))
    blob.extend(_varint(0))
    # posting 1: 2 posiciones, primera 0, gap 4 -> [0, 4]
    blob.extend(_varint(2))
    blob.extend(_varint(0))
    blob.extend(_varint(4))

    postings_list = decode_postings_list("cat", 2, bytes(blob), block_size=128)

    assert postings_list.term == "cat"
    assert postings_list.document_frequency == 2
    assert [p.doc_id for p in postings_list.postings] == [0, 3]
    assert [p.term_frequency for p in postings_list.postings] == [1, 2]
    assert postings_list.postings[0].positions == (0,)
    assert postings_list.postings[1].positions == (0, 4)


def test_decode_postings_list_packed_mode() -> None:
    # doc_id gaps [1, 2, 3, 4] empaquetados en un único bloque de bit_width=3:
    # bit_buffer = 1 | (2<<3) | (3<<6) | (4<<9) -> bytes little-endian [0xD1, 0x08]
    doc_id_gaps_block = bytes([3, 0xD1, 0x08])  # byte de ancho de bits + payload empaquetado

    blob = bytearray()
    blob.append(_MODE_PACKED)
    blob.extend(_varint(5))  # num_postings
    blob.extend(_varint(0))  # first_doc_id
    blob.extend(doc_id_gaps_block)
    blob.extend(_varint(1) * 5)  # 5 frecuencias, todas 1 (cada varint(1) es un byte)
    blob.extend(bytes([0]) * 5)  # 5 postings sin posiciones (num_positions=0)

    postings_list = decode_postings_list("dog", 5, bytes(blob), block_size=4)

    assert [p.doc_id for p in postings_list.postings] == [0, 1, 3, 6, 10]
    assert all(p.term_frequency == 1 for p in postings_list.postings)
    assert all(p.positions == () for p in postings_list.postings)


def test_decode_postings_list_empty_data_raises() -> None:
    with pytest.raises(ValueError, match="blob de postings vacío"):
        decode_postings_list("cat", 0, b"", block_size=128)


def test_decode_postings_list_unknown_mode_raises() -> None:
    blob = bytes([2]) + _varint(0)
    with pytest.raises(ValueError, match="modo de codificación desconocido"):
        decode_postings_list("cat", 0, blob, block_size=128)


def test_decode_postings_list_document_frequency_mismatch_raises() -> None:
    blob = bytes([_MODE_VARINT]) + _varint(0)  # declara 0 postings en el blob
    with pytest.raises(ValueError, match="inconsistencia de índice"):
        decode_postings_list("cat", 3, blob, block_size=128)


def test_decode_postings_list_zero_postings() -> None:
    blob = bytes([_MODE_VARINT]) + _varint(0)
    postings_list = decode_postings_list("cat", 0, blob, block_size=128)
    assert postings_list.document_frequency == 0
    assert postings_list.postings == ()


def test_decode_postings_list_truncated_packed_block_raises() -> None:
    blob = bytearray()
    blob.append(_MODE_PACKED)
    blob.extend(_varint(5))
    blob.extend(_varint(0))
    blob.extend(bytes([3, 0xD1]))  # falta el segundo byte del bloque empaquetado
    with pytest.raises(ValueError, match="bloque de bitpacking truncado"):
        decode_postings_list("dog", 5, bytes(blob), block_size=4)
