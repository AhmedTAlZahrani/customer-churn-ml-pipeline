# ── builder stage ─────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── runtime stage ─────────────────────────────
FROM python:3.11-slim-bookworm

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/model-info || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
