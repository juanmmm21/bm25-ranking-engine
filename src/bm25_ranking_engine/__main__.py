"""CLI de `bm25-ranking-engine`: `search` (rankea documentos frente a una query
en texto libre) y `term-idf` (document_frequency e IDF de un término concreto,
para depurar por qué un documento puntúa como puntúa)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from bm25_ranking_engine.models import BM25Params
from bm25_ranking_engine.pipeline import BM25RankingPipeline
from bm25_ranking_engine.tokenizer import tokenize


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bm25-ranking-engine",
        description="Rankea documentos de un índice invertido por relevancia BM25.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search", help="Rankea los documentos más relevantes para una query"
    )
    search_parser.add_argument(
        "index_dir", type=Path, help="Índice de inverted-index-builder o index-compression-codec"
    )
    search_parser.add_argument("query", nargs="+", help="Términos de la query (texto libre)")
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--k1", type=float, default=BM25Params().k1)
    search_parser.add_argument("--b", type=float, default=BM25Params().b)

    term_idf_parser = subparsers.add_parser(
        "term-idf", help="Muestra document_frequency e IDF de un término del índice"
    )
    term_idf_parser.add_argument("index_dir", type=Path)
    term_idf_parser.add_argument("term", type=str)

    return parser


def _run_search(args: argparse.Namespace) -> int:
    index_dir: Path = args.index_dir
    if not index_dir.exists():
        print(f"Error: no existe {index_dir}", file=sys.stderr)
        return 2

    query_terms = [token for raw_term in args.query for token in tokenize(raw_term)]
    if not query_terms:
        print("Error: la query no contiene ningún término indexable", file=sys.stderr)
        return 2

    params = BM25Params(k1=args.k1, b=args.b)
    with BM25RankingPipeline(index_dir, params) as pipeline:
        results = pipeline.search_tokens(query_terms, top_k=args.top_k)
        if not results:
            print("Sin resultados")
            return 0
        for rank, scored in enumerate(results, start=1):
            document = pipeline.document(scored.doc_id)
            title = document.title if document is not None else "(sin metadatos)"
            url = document.url if document is not None else "?"
            print(f"{rank}. doc_id={scored.doc_id} score={scored.score:.4f} — {title} ({url})")
    return 0


def _run_term_idf(args: argparse.Namespace) -> int:
    index_dir: Path = args.index_dir
    if not index_dir.exists():
        print(f"Error: no existe {index_dir}", file=sys.stderr)
        return 2

    normalized_terms = tokenize(args.term)
    term = normalized_terms[0] if normalized_terms else args.term

    with BM25RankingPipeline(index_dir) as pipeline:
        result = pipeline.term_document_frequency_and_idf(term)
        if result is None:
            print(f"'{term}' no aparece en el índice")
            return 0
        document_frequency, idf = result
        print(f"'{term}' — document_frequency={document_frequency} idf={idf:.4f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "search":
        return _run_search(args)
    if args.command == "term-idf":
        return _run_term_idf(args)

    parser.error(f"Comando desconocido: {args.command}")
    return 2  # inalcanzable: parser.error() termina el proceso con sys.exit()


if __name__ == "__main__":
    sys.exit(main())
