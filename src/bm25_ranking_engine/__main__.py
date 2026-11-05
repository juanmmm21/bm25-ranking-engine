"""CLI de `bm25-ranking-engine`: `search` (rankea documentos frente a una query
en texto libre) y `term-idf` (document_frequency e IDF de un término concreto,
para depurar por qué un documento puntúa como puntúa)."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from bm25_ranking_engine.dynamic_pruning import RetrievalStrategy
from bm25_ranking_engine.index_reader import open_index_reader
from bm25_ranking_engine.models import BM25Params
from bm25_ranking_engine.pipeline import BM25RankingPipeline
from bm25_ranking_engine.scoring import BM25Scorer
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
    search_parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in RetrievalStrategy],
        default=RetrievalStrategy.EXHAUSTIVE.value,
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Latencia p50/p99 de cada estrategia sobre un fichero de queries"
    )
    benchmark_parser.add_argument("index_dir", type=Path)
    benchmark_parser.add_argument("queries_path", type=Path, help="Una query por línea")
    benchmark_parser.add_argument("--top-k", type=int, default=10)
    benchmark_parser.add_argument("--repeat", type=int, default=3)

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
    with BM25RankingPipeline(index_dir, params, RetrievalStrategy(args.strategy)) as pipeline:
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


def _run_benchmark(args: argparse.Namespace) -> int:
    for path in (args.index_dir, args.queries_path):
        if not path.exists():
            print(f"Error: no existe {path}", file=sys.stderr)
            return 2
    queries = [
        tokens
        for line in args.queries_path.read_text(encoding="utf-8").splitlines()
        if (tokens := tokenize(line))
    ]
    if not queries:
        print("Error: el fichero no contiene ninguna query indexable", file=sys.stderr)
        return 2
    reader = open_index_reader(args.index_dir)
    try:
        scorer = BM25Scorer(reader)
        reference = [scorer.score(query, top_k=args.top_k) for query in queries]
        for strategy in RetrievalStrategy:
            latencies: list[float] = []
            scored = 0
            for _ in range(args.repeat):
                for query, expected in zip(queries, reference, strict=True):
                    started = time.perf_counter()
                    results, stats = scorer.score_with_stats(
                        query, top_k=args.top_k, strategy=strategy
                    )
                    latencies.append((time.perf_counter() - started) * 1000)
                    scored += stats.scored_documents
                    if results != expected:
                        print(f"Error: {strategy} difiere del exhaustivo", file=sys.stderr)
                        return 1
            ordered = sorted(latencies)
            p50 = ordered[len(ordered) // 2]
            p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
            print(
                f"{strategy.value:<15} p50={p50:.2f} ms p99={p99:.2f} ms "
                f"docs puntuados/query={scored / len(latencies):.0f}"
            )
    finally:
        reader.close()
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
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "term-idf":
        return _run_term_idf(args)

    parser.error(f"Comando desconocido: {args.command}")
    return 2  # inalcanzable: parser.error() termina el proceso con sys.exit()


if __name__ == "__main__":
    sys.exit(main())
