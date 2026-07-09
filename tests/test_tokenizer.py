"""Tests del tokenizador mínimo de conveniencia para la CLI."""

from __future__ import annotations

import unicodedata

from bm25_ranking_engine.tokenizer import tokenize


def test_tokenize_splits_on_punctuation_and_whitespace() -> None:
    assert tokenize("search engine, ranking!") == ["search", "engine", "ranking"]


def test_tokenize_casefolds() -> None:
    assert tokenize("Search ENGINE") == ["search", "engine"]


def test_tokenize_nfc_equivalence_between_composed_and_decomposed_accents() -> None:
    composed = "café"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed  # confirma que son representaciones distintas
    assert tokenize(composed) == tokenize(decomposed) == ["café"]


def test_tokenize_empty_string_returns_empty_list() -> None:
    assert tokenize("") == []
    assert tokenize("   ...   ") == []


def test_tokenize_keeps_digits_within_alphanumeric_tokens() -> None:
    assert tokenize("bm25 ranking") == ["bm25", "ranking"]
