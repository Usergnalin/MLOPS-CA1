FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow==3.18.0" "psycopg[binary]" boto3

EXPOSE 8000

CMD mlflow server \
    --host 0.0.0.0 --port 8000 --workers 1 \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION" \
    --serve-artifacts \
    --allowed-hosts "$MLFLOW_ALLOWED_HOSTS"