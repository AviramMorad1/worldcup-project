FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY features.py .
COPY model.py .
COPY predictions.py .
COPY squad_features.py .
COPY main.py .

CMD ["python", "main.py"]
