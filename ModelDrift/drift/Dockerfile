FROM python:3.9-slim

WORKDIR /app

COPY drift_server.py requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8090

CMD ["uvicorn", "drift_server:app", "--host", "0.0.0.0", "--port", "8009"]