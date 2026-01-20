.ONESHELL:
.PHONY: run test report neo4j-csv

run:
	python spike_grafos.py

test:
	pytest -q

report:
	python scripts/report.py

neo4j-csv:
	python scripts/neo4j_load.py

neo4j-up:
	docker compose -f docker-compose.neo4j.yml up -d

neo4j-ingest:
	python scripts/neo4j_ingest.py
