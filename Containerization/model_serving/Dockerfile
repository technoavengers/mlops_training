FROM python:3.9-slim

WORKDIR /app

COPY shared/random_forest_model.pkl shared/scaler.pkl serving.py preprocessing.py requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "8000"]