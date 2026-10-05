"""Question-independent policies used to probe the reward suite for shortcuts.

Each policy maps a record to an output without reading the question. A sound reward
must score all of them far below a correct response. They are used by
`tests/grounded/test_reward_hacking.py` and `scripts/grounded/reward_hacking_report.py`.

`ALWAYS_ABSTAIN` and `ALWAYS_ANSWER` group the policies by the decision they hard-code.
"""

from __future__ import annotations

import json
from collections.abc import Callable

Policy = Callable[[dict], str]


def _respond(answer: str, quotes: list[tuple[str, str]], sufficient: bool = True) -> str:
    return json.dumps(
        {
            "extracted_quotes": [{"chunk_id": c, "exact_quote": q} for c, q in quotes],
            "is_context_sufficient": sufficient,
            "final_answer": answer,
        },
        ensure_ascii=False,
    )


def _chunks(record: dict) -> dict[str, str]:
    return json.loads(record["context_chunks"])


def always_abstain(record: dict) -> str:
    return _respond("", [], sufficient=False)


def abstain_with_keywords(record: dict) -> str:
    return _respond("The context does not contain sufficient information.", [], sufficient=False)


def cite_a_stopword(record: dict) -> str:
    return _respond("the", [("c1", "the")])


def copy_whole_passage(record: dict) -> str:
    chunk_id, text = next(iter(_chunks(record).items()))
    return _respond(text, [(chunk_id, text)])


def quote_every_chunk_in_full(record: dict) -> str:
    return _respond("yes", list(_chunks(record).items()))


def first_sentence_as_quote_and_answer(record: dict) -> str:
    chunk_id, text = next(iter(_chunks(record).items()))
    sentence = text.split(". ")[0]
    return _respond(sentence, [(chunk_id, sentence)])


def first_words_of_every_chunk(record: dict) -> str:
    quotes = [(cid, " ".join(text.split()[:8])) for cid, text in _chunks(record).items()]
    return _respond(quotes[0][1], quotes)


def fabricated_quote(record: dict) -> str:
    return _respond("Paris", [("c1", "The capital of the country is Paris")])


def answer_without_evidence(record: dict) -> str:
    return _respond("France", [])


def empty_object(record: dict) -> str:
    return "{}"


def not_json(record: dict) -> str:
    return "I think the answer is France."


def valid_json_then_padding(record: dict) -> str:
    return always_abstain(record) + " " + "padding " * 50


def reference(record: dict) -> str:
    """The reference response built from the gold labels (upper bound, not a trivial policy)."""
    return record["target"]


TRIVIAL_POLICIES: dict[str, Policy] = {
    fn.__name__: fn
    for fn in (
        always_abstain,
        abstain_with_keywords,
        cite_a_stopword,
        copy_whole_passage,
        quote_every_chunk_in_full,
        first_sentence_as_quote_and_answer,
        first_words_of_every_chunk,
        fabricated_quote,
        answer_without_evidence,
        empty_object,
        not_json,
        valid_json_then_padding,
    )
}

ALWAYS_ABSTAIN = ("always_abstain", "abstain_with_keywords")
ALWAYS_ANSWER = (
    "cite_a_stopword",
    "copy_whole_passage",
    "quote_every_chunk_in_full",
    "first_sentence_as_quote_and_answer",
    "first_words_of_every_chunk",
    "fabricated_quote",
)
