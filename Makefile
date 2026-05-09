.PHONY: install dev api ui ingest test lint clean

install:
	pip install -r requirements.txt

dev: api

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run streamlit_app.py

ingest:
	@if [ -z "$(FILE)" ]; then echo "Usage: make ingest FILE=data/raw/book.pdf"; exit 1; fi
	python scripts/ingest_book.py $(FILE)

ingest-dry:
	@if [ -z "$(FILE)" ]; then echo "Usage: make ingest-dry FILE=data/raw/book.pdf"; exit 1; fi
	python scripts/ingest_book.py $(FILE) --dry-run

test:
	pytest tests/ -v

lint:
	ruff check src/ graph/ configs/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
