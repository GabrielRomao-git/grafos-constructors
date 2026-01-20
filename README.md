# Grafos a partir de Markdown (.md)

Protótipo em Python para extrair triplas de um texto `.md`, construir grafos em diferentes abordagens e exportar resultados.

## Pré-requisitos
- Python 3.11+.
- Ambiente virtual (uv/venv).
- Bibliotecas (ver `requirements.txt`):
  - Obrigatórias: `networkx`, `python-igraph`, `pandas`, `PyYAML`.
  - Opcionais: `graph-tool` (instalação mais complexa), `langchain` + `openai` (ou compatível) para LLM real.
- Instalação sugerida: `pip install -r requirements.txt` (comente/remova pacotes que não quiser).
- Arquivo de entrada em `artigos/text.md` (pode trocar via argumento na função `demo_pipeline`).

## Como rodar (pipeline padrão)
1. Ative o venv.
2. Execute: `python spike_grafos.py`
3. Saídas geradas em `out/`:
   - `graph_manual.json`
   - `graph_networkx.graphml`
   - `graph_igraph.graphml`
4. Logs no terminal mostram contagem de triplas, vizinhos de exemplo e top relações.

## Ordem das etapas no código
1. **Leitura e limpeza**: `load_markdown` lê o `.md`; `clean_markdown` remove marcações simples, URLs e excesso de espaços.
2. **Extração de triplas (heurística)**: `extract_triples_heuristic` aplica regex em sentenças para achar padrões `origem -> relação -> destino`.
3. **Extração com LLM (stub)**: `extract_triples_llm` retorna mock se não houver chave/dep; pode ser trocado por chamada real (LangChain/OpenAI).
4. **Combinação e dedupe**: junta triplas heurísticas, LLM stub e dataset sintético (`SYNTHETIC_TRIPLES`) e remove duplicatas.
5. **Construção de grafos**:
   - Estrutura manual (`build_manual_graph`): dicionários com índices de entrada/saída.
   - NetworkX (`build_networkx`): `DiGraph` com atributo `relation`.
   - igraph opcional (`build_igraph`): se instalado, monta grafo dirigido e salva GraphML.
6. **Consultas exemplo**: imprime vizinhos de entrada/saída de um nó sample e frequência de relações.
7. **Export**:
   - Manual → JSON (`export_manual_json`)
   - NetworkX → GraphML (`export_networkx_graphml`)
   - igraph → GraphML (`export_igraph_graphml`, se igraph disponível)

## Arquivos principais
- `spike_grafos.py`: pipeline completo e funções utilitárias.
- `spike_report.md`: resumo das abordagens, prós/contras e recomendações.
- `artigos/text.md`: texto de entrada padrão.
- `out/`: exportações geradas após rodar o script.

## Ajustes rápidos
- Trocar arquivo de entrada: passe outro caminho para `demo_pipeline(md_path=...)`.
- Limitar/expandir triplas extraídas: ajuste `max_triples_from_md`.
- Refinar heurística: edite `REL_PATTERNS` e aplique stopwords/remoção de seções irrelevantes antes da extração.
- Ativar LLM real: instale `langchain` + cliente OpenAI/compatível, defina `OPENAI_API_KEY` e substitua o stub em `extract_triples_llm` pela chamada real.
- Habilitar igraph: instale `python-igraph` e reexecute o script.

## Neo4j
- Subir Neo4j: `make neo4j-up`
- Ingestão automatizada (gera triplas se faltar, cria CSV, executa LOAD CSV e consultas de sanidade): `make neo4j-ingest`
- Variáveis sugeridas (veja `env.example`): `NEO4J_USER`, `NEO4J_PASSWORD`, `BOLT_URL`, `NEO4J_SAMPLE_NODE` (para consulta exemplo)
- UI: http://localhost:7474 — Bolt: `bolt://localhost:7687`
- Exportar CSV isolado: `make neo4j-csv` (gera `neo4j/import/nodes.csv` e `neo4j/import/edges.csv`)

## LLM (opcional) (Nao foi configurado chave)
- Pacotes opcionais listados (comentados) em `requirements.txt`: `langchain-openai`, `openai`, `tiktoken`.
- Configure `OPENAI_API_KEY` no ambiente (veja `env.example`) antes de ativar a chamada real no `extract_triples_llm`.
- O fluxo implementa chunking (~2000 tokens) e dedupe combinado com heurística + `SYNTHETIC_TRIPLES`.

## Automação e testes
- `make run` para rodar o pipeline e gerar exportações em `out/`.
- `make report` para métricas rápidas (nós/arestas/top relações) lendo `out/graph_manual.json`.
- `make test` para executar os testes rápidos (heurística, limpeza, dedupe, consultas, snapshot sintético).
