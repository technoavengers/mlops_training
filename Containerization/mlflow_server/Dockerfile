FROM python:3.9-slim

WORKDIR /app

RUN pip install mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow", "--host", "0.0.0.0", "--port", "5000"]