"""
sentiment.py
------------
Sentiment analysis helpers for the World Cup preprocessor service.

Uses:
  - VADER  (vaderSentiment)  → compound, positive, negative, neutral
  - TextBlob                  → polarity, subjectivity
"""

import logging

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Module-level singleton — SentimentIntensityAnalyzer loads a lexicon on
# instantiation, so we create it once and reuse it for every call.
_vader = SentimentIntensityAnalyzer()

_EMPTY_RESULT: dict = {
    "compound": 0.0,
    "positive": 0.0,
    "negative": 0.0,
    "neutral":  1.0,
    "textblob_polarity":     0.0,
    "textblob_subjectivity": 0.0,
}


def get_basic_sentiment(text: str) -> dict:
    """Return a sentiment dict for *text*.

    Keys
    ----
    compound              float  VADER compound score          [-1, 1]
    positive              float  VADER positive ratio          [0, 1]
    negative              float  VADER negative ratio          [0, 1]
    neutral               float  VADER neutral ratio           [0, 1]
    textblob_polarity     float  TextBlob polarity             [-1, 1]
    textblob_subjectivity float  TextBlob subjectivity         [0, 1]
    """
    if not isinstance(text, str) or not text.strip():
        return dict(_EMPTY_RESULT)

    # --- VADER ---
    try:
        vs = _vader.polarity_scores(text)
        compound = round(vs["compound"], 4)
        positive = round(vs["pos"], 4)
        negative = round(vs["neg"], 4)
        neutral  = round(vs["neu"], 4)
    except Exception as exc:
        logger.warning("VADER failed for text snippet '%s...': %s", text[:60], exc)
        compound = positive = negative = 0.0
        neutral = 1.0

    # --- TextBlob ---
    try:
        blob = TextBlob(text)
        textblob_polarity     = round(blob.sentiment.polarity,     4)
        textblob_subjectivity = round(blob.sentiment.subjectivity, 4)
    except Exception as exc:
        logger.warning("TextBlob failed for text snippet '%s...': %s", text[:60], exc)
        textblob_polarity     = 0.0
        textblob_subjectivity = 0.0

    return {
        "compound":              compound,
        "positive":              positive,
        "negative":              negative,
        "neutral":               neutral,
        "textblob_polarity":     textblob_polarity,
        "textblob_subjectivity": textblob_subjectivity,
    }