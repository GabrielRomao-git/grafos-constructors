from __future__ import annotations

from collections import Counter
from pathlib import Path
import json


def main():
    p = Path("out/graph_manual.json")
    if not p.exists():
        raise SystemExit("Gere out/graph_manual.json com `make run` primeiro")
    data = json.loads(p.read_text())
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    print(f"Nos: {len(nodes)} | Arestas: {len(edges)}")
    freq = Counter(e["rel"] for e in edges)
    print("Top relações:", freq.most_common(5))


if __name__ == "__main__":
    main()
