FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('vader_lexicon', quiet=True)"

COPY . .

CMD ["python", "main.py"]
