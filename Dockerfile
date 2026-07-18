# FlipInsight — Docker image
#
# Build-time:  ETL (clean.py) + ML model training (no DB connection needed)
# Start-time:  lifespan hook in api/main.py seeds PostgreSQL via init_db()
#
# This split is required for Railway (and any cloud platform) because
# managed PostgreSQL is not reachable during `docker build`.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (raw data included — needed for ETL + ML training)
COPY . .

# Build-time pipeline: clean data + train all 3 ML models
# NOTE: db.init_db is intentionally excluded here — it runs at startup via lifespan hook
RUN python3 etl/clean.py \
    && python3 -m ml.train_price_model \
    && python3 -m ml.train_category_model \
    && python3 -m ml.train_recommender

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Railway injects PORT; fall back to 8000 for local docker-compose
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
