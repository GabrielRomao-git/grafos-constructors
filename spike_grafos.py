"""
Protótipo de construção/consulta de grafos a partir de texto Markdown (.md).

Aborda:
- Extração de triplas via heurística (regex) e stub LLM.
- Construção de grafos em três variantes: NetworkX, estrutura manual, igraph.
- Consultas básicas e export/import simples.

"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Triple = Tuple[str, str, str]


def load_env_file(env_path: str | Path = ".env") -> None:
    """Carrega variáveis de um arquivo .env simples se existir."""
    path = Path(env_path)
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, val = stripped.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as exc:  # noqa: BLE001
        print(f"[env] Não foi possível carregar .env: {exc}")


# Configurações de domínio e qualidade
DOMAIN_RELATIONS = [
    "causes",
    "prevents",
    "increases",
    "reduces",
    "worsens",
    "improves",
    "associated with",
    "linked to",
    "related to",
    "leads to",
    "contributes to",
    "affects",
    "impacts",
    "raises risk of",
    "raises risk for",
    "risk of",
    "risk for",
    "aumenta risco de",
    "diminui risco de",
]

STOP_ENTITIES = {
    "thing",
    "things",
    "issue",
    "issues",
    "problem",
    "problems",
    "patient",
    "patients",
    "people",
    "person",
    "someone",
    "something",
    "anything",
    "entity",
}

IRRELEVANT_SECTION_TITLES = {"references", "keywords", "abbreviations"}
MAX_SENTENCE_CHARS = 400
MIN_ENTITY_LEN = 3


# Dataset sintético para regressão/validação rápida.
SYNTHETIC_TRIPLES: List[Triple] = [
    ("sleep", "impacts", "cardiovascular health"),
    ("sleep", "influences", "immune system"),
    ("poor sleep", "increases", "insulin resistance"),
    ("sleep apnea", "raises risk of", "hypertension"),
    ("sleep hygiene", "improves", "sleep quality"),
    ("melatonin", "regulates", "circadian rhythm"),
]


def load_markdown(md_path: str | Path) -> str:
    path = Path(md_path)
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def strip_irrelevant_sections(text: str) -> str:
    """
    Remove blocos do Markdown iniciados por títulos irrelevantes
    (References, Keywords, Abbreviations) até o próximo heading.
    """
    pattern = re.compile(
        rf"(?im)^#+\s+({'|'.join(IRRELEVANT_SECTION_TITLES)})\b.*?(?=^#|\Z)",
        flags=re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, "\n", text)


def clean_markdown(text: str) -> str:
    """Remove marcações simples e normaliza espaços."""
    text = strip_irrelevant_sections(text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[.*?\]\(.*?\)", " ", text)  # links markdown
    text = re.sub(r"http[s]?://\S+", " ", text)  # URLs soltas
    text = re.sub(r"\|.*?\|", " ", text)  # linhas de tabela simples
    text = re.sub(r"\s+", " ", text)
    return text.strip()


REL_PATTERNS = [
    r"(?P<src>[A-Za-z][A-Za-z0-9 ,&/().'-]{2,80})\s+(?P<rel>associated with|associated to|linked to|related to|leads to|contributes to|affects|impacts|causes|increases|reduces|worsens|improves|promotes|prevents|raises risk of|raises risk for)\s+(?P<dst>[A-Za-z][A-Za-z0-9 ,&/().'-]{2,80})",
    r"(?P<src>[A-Za-z][A-Za-z0-9 ,&/().'-]{2,80})\s+(?P<rel>risk of|risk for)\s+(?P<dst>[A-Za-z][A-Za-z0-9 ,&/().'-]{2,80})",
]


def _normalize_entity(value: str) -> str:
    value = re.sub(r"[^\w\s&/().'-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .;:-")
    return value.lower()


def _valid_entity(value: str) -> bool:
    return len(value) >= MIN_ENTITY_LEN and value not in STOP_ENTITIES


def validate_triple(src: str, rel: str, dst: str) -> bool:
    if not src or not dst or src == dst:
        return False
    if not _valid_entity(src) or not _valid_entity(dst):
        return False
    if not rel or rel.isspace():
        return False
    return True


def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def split_by_titles(text: str) -> List[str]:
    """Quebra por headings Markdown para reduzir contexto por bloco."""
    return re.split(r"(?im)^#+\s+.*$", text)


def chunk_text_by_chars(text: str, max_chars: int = 8000) -> List[str]:
    """Chunk simples por tamanho aproximado de tokens (~2000 tokens)."""
    words = text.split()
    chunks = []
    current: List[str] = []
    total = 0
    for w in words:
        if total + len(w) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = []
            total = 0
        current.append(w)
        total += len(w) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def extract_triples_heuristic(text: str, max_triples: int = 80) -> List[Triple]:
    sections = split_by_titles(text)
    triples: List[Triple] = []
    seen = set()
    discarded = 0

    for section in sections:
        sentences = split_sentences(section)
        for sentence in sentences:
            if len(sentence) < 20 or len(sentence) > MAX_SENTENCE_CHARS or sentence.isupper():
                continue
            for pattern in REL_PATTERNS:
                for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                    src = _normalize_entity(match.group("src"))
                    rel = match.group("rel").lower()
                    dst = _normalize_entity(match.group("dst"))
                    if not validate_triple(src, rel, dst):
                        discarded += 1
                        continue
                    key = (src, rel, dst)
                    if key in seen:
                        continue
                    seen.add(key)
                    triples.append(key)
                    if len(triples) >= max_triples:
                        print(f"[heuristic] Descartes: {discarded}")
                        return triples
    if discarded:
        print(f"[heuristic] Descartes: {discarded}")
    return triples


def dedupe_triples(triples: Iterable[Triple]) -> List[Triple]:
    seen = set()
    result: List[Triple] = []
    for t in triples:
        if t in seen:
            continue
        seen.add(t)
        result.append(t)
    return result


def build_manual_graph(triples: Iterable[Triple]) -> Dict[str, object]:
    nodes = set()
    edges = []
    out_idx: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    in_idx: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for idx, (src, rel, dst) in enumerate(triples):
        nodes.update([src, dst])
        edge = {"id": idx, "src": src, "rel": rel, "dst": dst}
        edges.append(edge)
        out_idx[src].append(edge)
        in_idx[dst].append(edge)

    return {"nodes": nodes, "edges": edges, "out": out_idx, "inp": in_idx}


def manual_neighbors_out(graph: Dict[str, object], node: str) -> List[str]:
    return [edge["dst"] for edge in graph["out"].get(node, [])]  # type: ignore[index]


def manual_neighbors_in(graph: Dict[str, object], node: str) -> List[str]:
    return [edge["src"] for edge in graph["inp"].get(node, [])]  # type: ignore[index]


def manual_filter_by_relation(graph: Dict[str, object], relation: str) -> List[Tuple[str, str]]:
    return [
        (edge["src"], edge["dst"])
        for edge in graph["edges"]  # type: ignore[index]
        if edge["rel"] == relation
    ]


def build_networkx(triples: Iterable[Triple]):
    import networkx as nx

    g = nx.DiGraph()
    for src, rel, dst in triples:
        g.add_edge(src, dst, relation=rel)
    return g


def build_igraph(triples: Iterable[Triple]):
    try:
        import igraph as ig
    except ImportError:
        return None, "python-igraph não está instalado; pule esta etapa ou instale-o."

    # Mapear nós para índices
    unique_nodes = {}
    for src, _, dst in triples:
        if src not in unique_nodes:
            unique_nodes[src] = len(unique_nodes)
        if dst not in unique_nodes:
            unique_nodes[dst] = len(unique_nodes)

    edges_idx = [(unique_nodes[src], unique_nodes[dst]) for src, _, dst in triples]
    relations = [rel for _, rel, _ in triples]

    g = ig.Graph(directed=True)
    g.add_vertices(len(unique_nodes))
    g.vs["name"] = list(unique_nodes.keys())
    g.add_edges(edges_idx)
    g.es["relation"] = relations
    return g, None


def igraph_neighbors_out(graph, node: str) -> List[str]:
    if graph is None:
        return []
    try:
        vid = graph.vs.find(name=node).index  # type: ignore[attr-defined]
    except Exception:
        return []
    return [graph.vs[n]["name"] for n in graph.neighbors(vid, mode="OUT")]  # type: ignore[attr-defined]


def igraph_neighbors_in(graph, node: str) -> List[str]:
    if graph is None:
        return []
    try:
        vid = graph.vs.find(name=node).index  # type: ignore[attr-defined]
    except Exception:
        return []
    return [graph.vs[n]["name"] for n in graph.neighbors(vid, mode="IN")]  # type: ignore[attr-defined]


def igraph_shortest_path(graph, source: str, target: str) -> List[str]:
    if graph is None:
        return []
    try:
        path = graph.get_shortest_paths(source, to=target, output="vpath")  # type: ignore[attr-defined]
        if not path or not path[0]:
            return []
        return [graph.vs[idx]["name"] for idx in path[0]]  # type: ignore[attr-defined]
    except Exception:
        return []


def relation_frequency(triples: Iterable[Triple]) -> Counter:
    return Counter(rel for _, rel, _ in triples)


def _parse_llm_json(content: str) -> List[Dict[str, object]]:
    """
    Tenta extrair JSON mesmo que venha dentro de code fences ou com texto extra.
    Retorna lista de dicionários ou lista vazia em caso de falha.
    """
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = cleaned.rstrip("` \n")
    # Tenta achar o primeiro bloco JSON válido dentro do texto
    candidates = [cleaned]
    match = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(1))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return [data]
            if isinstance(data, list):
                return data  # type: ignore[return-value]
        except Exception:
            continue
    print(f"[LLM] Resposta não pôde ser parseada como JSON: {cleaned[:200]}...")
    return []


def export_networkx_graphml(graph, output_path: str | Path):
    import networkx as nx

    nx.write_graphml(graph, Path(output_path))


def export_manual_json(graph: Dict[str, object], output_path: str | Path):
    data = {
        "nodes": sorted(graph["nodes"]),  # type: ignore[index]
        "edges": graph["edges"],  # type: ignore[index]
    }
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_igraph_graphml(graph, output_path: str):
    graph.write_graphml(str(output_path))


def extract_triples_llm(
    text: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    env_path: str | Path = ".env",
    max_triples: int = 30,
) -> List[Triple]:
    """
    Extração via LLM usando cliente OpenAI direto (compatível com Azure/OpenAI):
    - Requer OPENAI_API_KEY e OPENAI_BASE_URL configurados (env ou .env).
    - OPENAI_MODEL define o deployment/modelo (ex.: gpt-4o-mini).
    - Sem chave ou base_url, apenas registra aviso e não usa LLM.
    """
    load_env_file(env_path)
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    model_name = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    if not api_key:
        print("[LLM] OPENAI_API_KEY não configurada; pulei extração via LLM.")
        return []
    if not base_url:
        print("[LLM] OPENAI_BASE_URL não configurada; pulei extração via LLM.")
        return []

    try:
        from openai import OpenAI
    except Exception:
        print("[LLM] Dependência 'openai' ausente; instale-a para habilitar LLM.")
        return []

    client = OpenAI(base_url=base_url,api_key=api_key)

    prompt_template = (
            "Extraia triplas (origem, relação, destino) do texto abaixo.\n"
            "Responda SOMENTE em JSON válido, sem markdown, sem texto extra.\n"
            "Formato: [{{src: ..., rel: ..., dst: ...}}]\n"
            "Use relações curtas (1-5 palavras), tudo em minúsculas, cite apenas fatos explícitos.\n"
            "Texto:\n{chunk}"
        )

    triples: List[Triple] = []

    for chunk in chunk_text_by_chars(text, max_chars=8000)[:4]:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_template.format(chunk=chunk)}]
            )
            message = resp.choices[0].message if resp and resp.choices else None

            content = message.content if message else ""
            print(f"[LLM][content] {content}\n")

            data = _parse_llm_json(content)
            for item in data:
                src = _normalize_entity(str(item.get("src", "")))
                rel = str(item.get("rel", "")).lower()
                dst = _normalize_entity(str(item.get("dst", "")))
                if validate_triple(src, rel, dst):
                    triples.append((src, rel, dst))
                else:
                    print(f"[LLM] Descartando tripla inválida: {(src, rel, dst)}")
            if len(triples) >= max_triples:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"[LLM] Erro ao processar chunk: {exc}")
            continue

    return dedupe_triples(triples)[:max_triples]


def demo_pipeline(
    md_path: str | Path = "artigos/text.md",
    output_dir: str | Path = "out",
    max_triples_from_md: int = 60,
    env_path: str | Path = "env.example",
):
    load_env_file(env_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_text = load_markdown(md_path)
    cleaned = clean_markdown(raw_text)
    triples_md = extract_triples_heuristic(cleaned, max_triples=max_triples_from_md)
    triples_llm = extract_triples_llm(cleaned, max_triples=10, env_path=env_path)

    combined = dedupe_triples([*triples_md, *triples_llm, *SYNTHETIC_TRIPLES])

    # Manual graph
    manual_graph = build_manual_graph(combined)
    export_manual_json(manual_graph, out_dir / "graph_manual.json")

    # NetworkX
    nx_graph = build_networkx(combined)
    export_networkx_graphml(nx_graph, out_dir / "graph_networkx.graphml")

    # igraph (opcional)
    ig_graph, ig_msg = build_igraph(combined)
    if ig_graph is not None:
        export_igraph_graphml(ig_graph, out_dir / "graph_igraph.graphml")

    # Consultas exemplo (impressões simples)
    sample_node = combined[0][0] if combined else "sleep"
    print(f"Total de triplas (deduplicadas): {len(combined)}")
    print(f"Exemplo - vizinhos de saída (manual) para '{sample_node}': {manual_neighbors_out(manual_graph, sample_node)}")
    print(f"Exemplo - vizinhos de entrada (manual) para '{sample_node}': {manual_neighbors_in(manual_graph, sample_node)}")
    print(f"Frequência de relações: {relation_frequency(combined).most_common(5)}")

    if ig_msg:
        print(ig_msg)

    return {
        "triples_md": triples_md,
        "triples_llm": triples_llm,
        "combined": combined,
    }


if __name__ == "__main__":
    demo_pipeline()
