# Spike: Grafos a partir de Markdown (.md)

## Como rodar
- Ative o venv já criado.
- Executar o pipeline padrão (usa `artigos/text.md`, heurística + stub LLM, gera exportações em `out/`):
  - `python spike_grafos.py`
- Saídas:
  - `out/graph_manual.json`
  - `out/graph_networkx.graphml`
  - `out/graph_igraph.graphml` (se `python-igraph` estiver instalado)

## Pipeline implementado
- Leitura do `.md` e limpeza básica (`clean_markdown`).
- Extração de triplas heurística (regex com verbos como “affects”, “improves”, “associated with”) + stub LLM (`extract_triples_llm`) que retorna mock quando não há chave/dependências.
- Combinação com dataset sintético fixo para garantir resultados mínimos.
- Construção de grafos:
  - NetworkX (`DiGraph` com atributo `relation` na aresta).
  - Estrutura manual (`nodes`, `edges`, índices de entrada/saída).
  - igraph (opcional; ignora se lib não instalada).
- Consultas exemplo impressas no terminal:
  - Vizinhos de saída/entrada para um nó sample.
  - Frequência das relações.
- Export:
  - Manual → JSON (`graph_manual.json`).
  - NetworkX → GraphML (`graph_networkx.graphml`).
  - igraph → GraphML (`graph_igraph.graphml`, se lib disponível).

## Comparação (rápida)
- NetworkX: API simples, ótimo para POC, boa compatibilidade (GraphML). Menor performance em grafos muito grandes.
- Estrutura manual: zero dependências, totalmente controlável; precisa mais código para queries avançadas (caminhos, métricas).
- igraph: mais performática e compacta; API diferente, documentação sólida; instalação pode ser mais sensível em alguns ambientes.
- Neo4j (não implementado aqui): útil se precisar de persistência, consultas ricas (Cypher) e integração com GraphRAG; requer setup de DB/container.

## Recomendações iniciais
- Para spike/POC: use NetworkX + heurística atual; combine com stub LLM apenas se quiser enriquecer triplas.
- Se volume crescer ou precisar de métricas/algoritmos mais rápidos: considere igraph.
- Para persistência/queries complexas entre serviços: avaliar Neo4j + Cypher/GraphRAG.

### Automação Neo4j (nova)
- Subir DB: `make neo4j-up`
- Ingestão/consultas: `make neo4j-ingest` (gera `out/graph_manual.json` se faltar, exporta CSV para `neo4j/import/`, roda `LOAD CSV` via `cypher-shell` e imprime contagens/top relações/vizinhos de um nó exemplo).
- Variáveis: `NEO4J_USER`, `NEO4J_PASSWORD`, `BOLT_URL` (default `bolt://localhost:7687`), `NEO4J_SAMPLE_NODE` (ex.: `sleep`).

## Próximos passos possíveis
- Afinar regex/heurísticas com lista de verbos/domínios específicos e stopwords.
- Adicionar chunking do `.md` e streaming de triplas LLM se chave estiver disponível.
- Implementar testes unitários leves para funções de extração e consultas.
