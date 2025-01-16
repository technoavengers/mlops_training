FROM python:3.9-slim

WORKDIR /app

COPY preprocessing.py training.py tracking.py params.yaml data/walmart.csv requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "training.py"]