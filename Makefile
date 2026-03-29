.PHONY: install test lint run-api clean

install:
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt

test:
	.venv/Scripts/pytest tests/ --cov=src

lint:
	.venv/Scripts/ruff check src/ api/ tests/

run-api:
	.venv/Scripts/uvicorn api.main:app --host 0.0.0.0 --port 8000

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
