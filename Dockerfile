# FlipInsight — single-container image: bakes in the cleaned data,
# SQLite DB, and trained ML models at build time so `docker run` starts
# instantly with no first-request warm-up.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (raw data included — needed to build the pipeline)
COPY . .

# Run the full pipeline once at build time: clean -> load DB -> train all 3 models
RUN python3 etl/clean.py \
    && python3 -m db.init_db \
    && python3 -m ml.train_price_model \
    && python3 -m ml.train_category_model \
    && python3 -m ml.train_recommender

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
