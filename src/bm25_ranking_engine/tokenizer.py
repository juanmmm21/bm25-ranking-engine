"""Tokenizador mínimo, propio de este repo, para que la CLI de
`bm25-ranking-engine` sea usable de forma standalone con una query en texto
libre. No es el tokenizador de producción del ecosistema — ese rol lo cumplirá
`query-parser-autocomplete`, que además soportará operadores (frases exactas,
exclusión de términos, filtros) que este módulo deliberadamente no implementa.

Reutiliza, sin importarlo, el mismo criterio de normalización que
`inverted-index-builder` aplica al indexar (NFC + `casefold()` consciente de
locale, no `lower()`) para que los términos de una query en texto libre
coincidan con el vocabulario del índice sin depender del paquete Python de ese
repo (`../AGENTS.md`).
"""

from __future__ import annotations

import unicodedata


def tokenize(text: str) -> list[str]:
    """Normaliza a NFC, corta en tiradas máximas de letras/dígitos Unicode y
    aplica `casefold()` a cada token — puntuación, espacios y símbolos separan
    tokens."""
    normalized = unicodedata.normalize("NFC", text)
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current).casefold())
            current = []
    if current:
        tokens.append("".join(current).casefold())
    return tokens
