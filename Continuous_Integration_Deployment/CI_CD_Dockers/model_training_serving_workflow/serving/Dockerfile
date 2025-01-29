FROM python:3.9-slim

WORKDIR /app

COPY scripts/serving.py scripts/preprocessing.py requirements.txt random_forest_model.pkl scaler.pkl /app/


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "8000"]