import sys
from pathlib import Path

import pytest

# Permite importar spike_grafos.py quando o projeto não está como pacote instalado.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spike_grafos import (  # noqa: E402
    SYNTHETIC_TRIPLES,
    build_igraph,
    build_manual_graph,
    clean_markdown,
    dedupe_triples,
    extract_triples_heuristic,
    igraph_neighbors_out,
    manual_neighbors_in,
    manual_neighbors_out,
)


def test_clean_markdown_removes_links_and_sections():
    raw = """
# Title
Content with a [link](http://example.com).
## References
- ref1
"""
    cleaned = clean_markdown(raw)
    assert "http" not in cleaned
    assert "references" not in cleaned.lower()


def test_extract_triples_heuristic_basic():
    text = "Sleep improves mood. Poor sleep raises risk of hypertension."
    triples = extract_triples_heuristic(text)
    assert ("sleep", "improves", "mood") in triples
    assert ("poor sleep", "raises risk of", "hypertension") in triples


def test_dedupe_triples_removes_duplicates():
    triples = [("a", "rel", "b"), ("a", "rel", "b"), ("b", "rel", "c")]
    deduped = dedupe_triples(triples)
    assert len(deduped) == 2


def test_manual_neighbors():
    triples = [("a", "rel", "b"), ("b", "rel", "c")]
    g = build_manual_graph(triples)
    assert manual_neighbors_out(g, "a") == ["b"]
    assert manual_neighbors_in(g, "c") == ["b"]


def test_snapshot_contains_synthetic_triples():
    combined = dedupe_triples([*SYNTHETIC_TRIPLES, ("x", "rel", "y")])
    for t in SYNTHETIC_TRIPLES:
        assert t in combined


def test_igraph_neighbors_out():
    igraph = pytest.importorskip("igraph")
    g, msg = build_igraph([("a", "rel", "b"), ("b", "rel", "c")])
    assert msg is None
    assert igraph_neighbors_out(g, "a") == ["b"]
