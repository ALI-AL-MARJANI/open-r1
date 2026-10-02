"""Text utilities shared by the rewards and the evaluation metrics.

Answer normalisation, exact match and token F1 follow the official SQuAD v2.0
evaluation script (Rajpurkar et al., 2018) so that training rewards and reported
metrics use the same definition.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WORD = re.compile(r"\w+", re.UNICODE)
_PUNCTUATION = set(string.punctuation)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "or",
        "but",
        "not",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "as",
        "if",
        "than",
        "then",
        "so",
    }
)


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation and articles, collapse whitespace (SQuAD official)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def answer_tokens(text: str) -> list[str]:
    """Tokens of the normalised answer; empty list for an empty answer."""
    normalised = normalize_answer(text)
    return normalised.split() if normalised else []


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_prf(prediction: str, gold: str) -> tuple[float, float, float]:
    """Token-level precision, recall and F1 between a prediction and one gold answer."""
    pred_tokens = answer_tokens(prediction)
    gold_tokens = answer_tokens(gold)
    if not pred_tokens or not gold_tokens:
        # The official script gives 1.0 only when both are empty (no-answer case).
        score = float(pred_tokens == gold_tokens)
        return score, score, score
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0, 0.0, 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return precision, recall, 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between a prediction and one gold answer (SQuAD official)."""
    return token_prf(prediction, gold)[2]


def best_em_f1(prediction: str, gold_answers: list[str]) -> tuple[float, float]:
    """Maximum EM and F1 over the gold answers. An empty gold list means "no answer"."""
    golds = gold_answers or [""]
    return (
        max(exact_match(prediction, g) for g in golds),
        max(token_f1(prediction, g) for g in golds),
    )


def words(text: str) -> list[str]:
    """Word tokens of a raw string (no normalisation beyond the regex)."""
    return _WORD.findall(text)


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def is_verbatim(quote: str, source: str) -> bool:
    """True if `quote` is a substring of `source`, up to runs of whitespace.

    The comparison is case- and punctuation-sensitive on purpose: the task is to copy,
    and any fuzzier test would let paraphrases count as citations.
    """
    if not quote or not quote.strip():
        return False
    if quote in source:
        return True
    return collapse_whitespace(quote) in collapse_whitespace(source)


def _occurs_on_boundaries(text: str, phrase: str) -> bool:
    """Case-insensitive search for `phrase` in `text`, not cutting through a word or number.

    A match is rejected only when an alphanumeric character of the phrase touches an
    alphanumeric character of the text ("1924" in "19245"). "Rollo" matches in
    "Rollo's" and "$1,000,000" in "US$1,000,000".
    """
    haystack, needle = text.lower(), phrase.lower().strip()
    if not needle:
        return False
    start = haystack.find(needle)
    while start != -1:
        end = start + len(needle)
        cuts_left = start > 0 and haystack[start - 1].isalnum() and needle[0].isalnum()
        cuts_right = end < len(haystack) and haystack[end].isalnum() and needle[-1].isalnum()
        if not cuts_left and not cuts_right:
            return True
        start = haystack.find(needle, start + 1)
    return False


def contains_answer(text: str, gold_answers: list[str]) -> bool:
    """True if a gold answer occurs in `text`.

    Two tests are tried: the raw answer string on word boundaries, and the normalised
    answer on token boundaries of the normalised text (SQuAD normalisation removes
    punctuation, which would otherwise hide "Rollo" in "Rollo's").
    """
    haystack = f" {normalize_answer(text)} "
    for gold in gold_answers:
        if _occurs_on_boundaries(text, gold):
            return True
        needle = normalize_answer(gold)
        if needle and f" {needle} " in haystack:
            return True
    return False


def is_trivial_quote(quote: str, min_words: int) -> bool:
    """A quote is trivial if it is shorter than `min_words` or made only of stopwords."""
    tokens = [w.lower() for w in words(quote)]
    if len(tokens) < min_words:
        return True
    return all(t in STOPWORDS for t in tokens)
