_POSITIVE_WORDS = {
    "win", "wins", "great", "good", "strong", "amazing",
    "favorite", "favorites", "confident", "hope", "best",
}

_NEGATIVE_WORDS = {
    "lose", "loses", "bad", "weak", "injury", "injured",
    "terrible", "overrated", "worried", "fear",
}


def get_basic_sentiment(text: str) -> dict:
    if not isinstance(text, str) or not text:
        return {"compound": 0.0, "positive": 0, "negative": 0, "neutral": 0}

    tokens = text.lower().split()
    total_tokens = len(tokens)

    positive_count = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    negative_count = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    total_matched = positive_count + negative_count

    compound = (positive_count - negative_count) / max(total_matched, 1)
    compound = max(-1.0, min(1.0, compound))

    neutral = max(total_tokens - positive_count - negative_count, 0)

    return {
        "compound": round(compound, 4),
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral,
    }
