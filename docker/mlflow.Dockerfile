# MLflow tracking server with Postgres backend and GCS artifact support.
FROM ghcr.io/mlflow/mlflow:v3.17.1
RUN pip install --no-cache-dir "psycopg[binary]>=3.2" "google-cloud-storage>=2.18"
