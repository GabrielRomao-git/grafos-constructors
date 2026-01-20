"""
Gera CSVs para ingestão no Neo4j a partir do graph_manual.json.
Executa após rodar `python spike_grafos.py` (gera out/graph_manual.json).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def export_csv(
    manual_json_path: str | Path = "out/graph_manual.json",
    output_dir: str | Path = "neo4j/import",
) -> Dict[str, Path]:
    src_path = Path(manual_json_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {src_path}")

    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes: List[str] = data.get("nodes", [])
    edges: List[Dict[str, str]] = data.get("edges", [])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_id_map = {name: idx for idx, name in enumerate(nodes)}

    nodes_csv = out_dir / "nodes.csv"
    edges_csv = out_dir / "edges.csv"

    with nodes_csv.open("w", newline="", encoding="utf-8") as f_nodes:
        writer = csv.writer(f_nodes)
        writer.writerow(["node_id", "name"])
        for name, idx in node_id_map.items():
            writer.writerow([idx, name])

    with edges_csv.open("w", newline="", encoding="utf-8") as f_edges:
        writer = csv.writer(f_edges)
        writer.writerow(["src_id", "dst_id", "relation"])
        for edge in edges:
            writer.writerow(
                [
                    node_id_map.get(edge["src"]),
                    node_id_map.get(edge["dst"]),
                    edge["rel"],
                ]
            )

    return {"nodes": nodes_csv, "edges": edges_csv}


def print_cypher_examples():
    cypher = """
// No cypher-shell:
// :param csvBase => 'file:///nodes.csv' etc.
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CREATE (n:Entity {id: toInteger(row.node_id), name: row.name});

LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (s:Entity {id: toInteger(row.src_id)}), (d:Entity {id: toInteger(row.dst_id)})
CREATE (s)-[:RELATION {name: row.relation}]->(d);

// Exemplos
MATCH (n:Entity {name: 'sleep'})-[:RELATION]->(m) RETURN m LIMIT 10;
MATCH p=shortestPath((a:Entity {name: 'sleep'})-[:RELATION*..4]->(b:Entity {name:'hypertension'})) RETURN p;
MATCH ()-[r:RELATION]->() RETURN r.name AS rel, count(*) AS freq ORDER BY freq DESC LIMIT 10;
"""
    print(cypher.strip())


if __name__ == "__main__":
    paths = export_csv()
    print(f"CSV gerados em: {paths}")
    print_cypher_examples()
