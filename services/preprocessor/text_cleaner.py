import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_KEEP_RE = re.compile(r"[^a-z0-9 ]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")

    text = _KEEP_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    if not text:
        return []
    return [token for token in text.split() if len(token) >= 2]
