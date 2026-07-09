"""Tests de la CLI, reutilizando la misma fixture de índice de `test_pipeline.py`."""

from __future__ import annotations

from pathlib import Path

import pytest

from bm25_ranking_engine.__main__ import main
from tests.test_pipeline import _write_index


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "index"
    _write_index(directory)
    return directory


def test_search_prints_ranked_results(index_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["search", str(index_dir), "cat", "dog"])
    captured = capsys.readouterr()

    assert exit_code == 0
    lines = captured.out.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("1. doc_id=1")
    assert "Cats and Dogs" in lines[0]


def test_search_respects_top_k(index_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["search", str(index_dir), "cat", "dog", "--top-k", "1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1


def test_search_on_missing_directory_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["search", str(tmp_path / "does-not-exist"), "cat"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Error" in captured.err


def test_search_with_unindexable_query_reports_error(
    index_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["search", str(index_dir), "..."])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Error" in captured.err


def test_term_idf_prints_document_frequency_and_idf(
    index_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["term-idf", str(index_dir), "cat"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "document_frequency=2" in captured.out
    assert "idf=" in captured.out


def test_term_idf_reports_missing_term(index_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["term-idf", str(index_dir), "missing"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "no aparece en el índice" in captured.out
