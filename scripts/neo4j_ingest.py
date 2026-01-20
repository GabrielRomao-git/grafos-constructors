"""
Automatiza ingestão/consultas no Neo4j usando o compose existente.

Fluxo:
- Garante que out/graph_manual.json existe (roda spike_grafos.py se preciso).
- Gera CSVs em neo4j/import (export_csv).
- Executa cypher-shell dentro do container para limpar, carregar CSV e rodar consultas de sanidade.

Requisitos:
- Docker + docker compose
- Container Neo4j ativo: `docker compose -f docker-compose.neo4j.yml up -d`
- Variáveis: NEO4J_USER, NEO4J_PASSWORD, BOLT_URL (opcional), NEO4J_SAMPLE_NODE (opcional)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = ROOT / "docker-compose.neo4j.yml"
GRAPH_JSON = ROOT / "out" / "graph_manual.json"


def run(cmd: List[str], cwd: Path = ROOT) -> str:
    """Executa comando e retorna stdout; lança erro com stderr em caso de falha."""
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Erro ao executar: {' '.join(cmd)}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    return res.stdout.strip()


def ensure_graph_json():
    """Gera graph_manual.json se ausente."""
    if GRAPH_JSON.exists():
        return
    print("Gerando out/graph_manual.json via spike_grafos.py ...")
    run(["python", "spike_grafos.py"])
    if not GRAPH_JSON.exists():
        raise FileNotFoundError("Falha ao gerar out/graph_manual.json")


def generate_csv() -> Dict[str, Path]:
    """Gera nodes.csv/edges.csv em neo4j/import."""
    from neo4j_load import export_csv

    print("Gerando CSVs em neo4j/import ...")
    return export_csv(
        manual_json_path=GRAPH_JSON,
        output_dir=ROOT / "neo4j" / "import",
    )


def cypher_shell(query: str, user: str, password: str, bolt: str) -> str:
    """Executa cypher-shell dentro do container neo4j."""
    if not COMPOSE_FILE.exists():
        raise FileNotFoundError(f"docker-compose.neo4j.yml não encontrado em {COMPOSE_FILE}")
    cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "exec",
        "-T",
        "neo4j",
        "cypher-shell",
        "-u",
        user,
        "-p",
        password,
        "-a",
        bolt,
        query,
    ]
    return run(cmd)


def ingest_and_query(sample_node: str = "sleep") -> List[Tuple[str, str]]:
    """Limpa BD, carrega CSV e roda consultas básicas. Retorna lista (etapa, saída)."""
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "12345678")
    bolt = os.getenv("BOLT_URL", "bolt://localhost:7687")

    steps: List[Tuple[str, str]] = []
    cyphers = [
        ("clear", "MATCH (n) DETACH DELETE n;"),
        (
            "load_nodes",
            "LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row "
            "CREATE (n:Entity {id: toInteger(row.node_id), name: row.name});",
        ),
        (
            "load_edges",
            "LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row "
            "MATCH (s:Entity {id: toInteger(row.src_id)}), (d:Entity {id: toInteger(row.dst_id)}) "
            "CREATE (s)-[:RELATION {name: row.relation}]->(d);",
        ),
        ("count_nodes", "MATCH (n:Entity) RETURN count(n) AS nodes;"),
        ("count_edges", "MATCH ()-[r:RELATION]->() RETURN count(r) AS edges;"),
        (
            "top_relations",
            "MATCH ()-[r:RELATION]->() "
            "RETURN r.name AS relation, count(*) AS freq ORDER BY freq DESC LIMIT 5;",
        ),
        (
            "neighbors_sample",
            f"MATCH (n:Entity {{name: '{sample_node}'}})-[:RELATION]->(m) RETURN m.name AS neighbor LIMIT 5;",
        ),
    ]

    for label, query in cyphers:
        print(f"Executando {label} ...")
        out = cypher_shell(query, user=user, password=password, bolt=bolt)
        steps.append((label, out))
    return steps


def main():
    ensure_graph_json()
    generate_csv()
    sample_node = os.getenv("NEO4J_SAMPLE_NODE", "sleep")
    results = ingest_and_query(sample_node=sample_node)
    print("\n=== Resultados ===")
    for label, out in results:
        print(f"\n[{label}]\n{out}")


if __name__ == "__main__":
    main()
