# CLAUDE.md

ML pipeline for telecom customer churn prediction with FastAPI serving.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest tests/ --cov=src
```

## Run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Architecture

- `src/` contains ingest, feature_engineering, and model_training (with `benchmark_all`)
- API layer in `api/` serves predictions via FastAPI

## Style

- **Docstrings:** Google style
- **Commits:** Descriptive (e.g., "Add feature importance plotting to evaluation module")
- **Req pinning:** Exact `==`
