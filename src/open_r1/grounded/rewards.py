"""Verifiable rewards for grounded question answering with GRPO.

Every reward is computed from the completion and the gold labels of the example
(`is_answerable`, `gold_answers`, `gold_chunk_ids`), which the trainer forwards from
the dataset columns. No reward depends on a learned judge.

Design rules, each of which closes a shortcut found in the first version of this
reward suite (see `tests/grounded/test_reward_hacking.py`):

* Abstention is only rewarded when the question is labelled unanswerable.
* A quote only counts if it is verbatim, non-trivial and concise, so that citing a
  stopword or pasting the whole passage earns nothing.
* The answer itself is scored against the gold answers (SQuAD token F1).
* Rewards that are undefined for an example return `None`, which TRL excludes from
  the weighted sum for that sample. `None` is decided from the labels only, so all
  completions of a prompt are scored by the same set of rewards.

Signature expected by `trl.GRPOTrainer`:
    reward(completions, **dataset_columns) -> list[float | None]
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from .parsing import GroundedResponse, ParseStatus, Quote, completion_text, parse_response
from .text import (
    answer_tokens,
    best_em_f1,
    contains_answer,
    is_trivial_quote,
    is_verbatim,
    normalize_answer,
    words,
)

RewardFn = Callable[..., list]


@dataclass(frozen=True)
class RewardSettings:
    """Shaping constants of the reward suite. Set from the experiment YAML."""

    quote_min_words: int = 3  # shorter quotes, or stopword-only quotes, score 0
    quote_full_credit_words: int = 50  # quotes up to this length get full credit
    quote_zero_credit_words: int = 100  # linear decay to 0 at this length
    max_quotes: int = 3  # more quotes than this scale the grounding reward down
    max_extra_chars: int = 0  # non-whitespace characters tolerated outside the JSON object
    answer_copy_ratio: float = 0.8  # answer/quote length ratio above which faithfulness is capped
    answer_copy_cap: float = 0.5


@dataclass(frozen=True)
class Example:
    """Gold labels of one dataset row, decoded once per completion."""

    chunks: dict[str, str]
    gold_answers: list[str]
    gold_chunk_ids: list[str]
    is_answerable: bool


def _decode_json(value: Any, expected: type, default: Any) -> Any:
    if isinstance(value, expected):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return default
        if isinstance(decoded, expected):
            return decoded
    return default


def _iter_samples(completions: list, columns: dict[str, Any]) -> Iterator[tuple[GroundedResponse, Example]]:
    n = len(completions)
    chunks_col = columns.get("context_chunks") or [None] * n
    answers_col = columns.get("gold_answers") or [None] * n
    gold_ids_col = columns.get("gold_chunk_ids") or [None] * n
    answerable_col = columns.get("is_answerable") or [None] * n
    for completion, chunks, answers, gold_ids, answerable in zip(
        completions, chunks_col, answers_col, gold_ids_col, answerable_col
    ):
        gold_answers = [a for a in _decode_json(answers, list, []) if isinstance(a, str)]
        example = Example(
            chunks={k: v for k, v in _decode_json(chunks, dict, {}).items() if isinstance(v, str)},
            gold_answers=gold_answers,
            gold_chunk_ids=[c for c in _decode_json(gold_ids, list, []) if isinstance(c, str)],
            is_answerable=bool(answerable) if answerable is not None else bool(gold_answers),
        )
        yield parse_response(completion_text(completion)), example


def quote_quality(quote: Quote, source: str, settings: RewardSettings) -> float:
    """Score of one quote against a source text, in [0, 1].

    0 if the quote is not verbatim or is trivial; otherwise 1 up to
    `quote_full_credit_words`, decaying linearly to 0 at `quote_zero_credit_words`.
    """
    if is_trivial_quote(quote.text, settings.quote_min_words) or not is_verbatim(quote.text, source):
        return 0.0
    n_words = len(words(quote.text))
    if n_words <= settings.quote_full_credit_words:
        return 1.0
    span = settings.quote_zero_credit_words - settings.quote_full_credit_words
    return max(0.0, 1.0 - (n_words - settings.quote_full_credit_words) / span)


def _context(example: Example) -> str:
    # Chunks are joined with a non-whitespace separator that does not occur in the
    # datasets, so that a quote spanning two chunks is not accepted as verbatim.
    return " ␞ ".join(example.chunks.values())


def build_reward_functions(settings: RewardSettings | None = None) -> dict[str, RewardFn]:
    """Return the reward suite as a name -> function mapping."""
    settings = settings or RewardSettings()

    def format_reward(completions, **columns) -> list[float]:
        """Schema compliance, the only reward that does not need gold labels.

        0.0 no JSON object, 0.25 missing keys, 0.5 wrong types, 0.75 valid schema with
        text outside the JSON object or an inconsistent abstention (abstains but cites,
        or answers without citing), 1.0 otherwise.
        """
        scores = []
        for response, _ in _iter_samples(completions, columns):
            if response.status is ParseStatus.NO_JSON:
                scores.append(0.0)
            elif response.status is ParseStatus.MISSING_KEYS:
                scores.append(0.25)
            elif response.status is ParseStatus.BAD_TYPES:
                scores.append(0.5)
            elif response.extra_chars > settings.max_extra_chars or not response.consistent:
                scores.append(0.75)
            else:
                scores.append(1.0)
        return scores

    def answer_correctness_reward(completions, **columns) -> list[float]:
        """Correctness of the decision to answer and of the answer itself.

        Answerable question:   max token F1 against the gold answers; 0 if abstained.
        Unanswerable question: 1 if abstained, 0 if answered.
        """
        scores = []
        for response, example in _iter_samples(completions, columns):
            if not response.valid:
                scores.append(0.0)
            elif not example.is_answerable:
                scores.append(1.0 if response.abstained else 0.0)
            elif response.abstained:
                scores.append(0.0)
            else:
                scores.append(best_em_f1(response.answer, example.gold_answers)[1])
        return scores

    def quote_grounding_reward(completions, **columns) -> list[float | None]:
        """Quality of the cited evidence on answerable questions.

        reward = count_factor * (0.5 * precision + 0.5 * support)
          precision     mean quote quality (verbatim in the context, non-trivial, concise)
          support       1 if a quote of non-zero quality contains a gold answer
          count_factor  min(1, max_quotes / n_quotes)

        When no gold answer occurs in the context (yes/no questions), support is
        undefined and the reward is count_factor * precision.
        Undefined (None) on unanswerable questions: abstention there is scored by
        `answer_correctness_reward` only.
        """
        scores: list[float | None] = []
        for response, example in _iter_samples(completions, columns):
            if not example.is_answerable:
                scores.append(None)
                continue
            if not response.valid or response.abstained or not response.quotes:
                scores.append(0.0)
                continue
            context = _context(example)
            qualities = [quote_quality(q, context, settings) for q in response.quotes]
            precision = sum(qualities) / len(qualities)
            count_factor = min(1.0, settings.max_quotes / len(qualities))
            if contains_answer(context, example.gold_answers):
                support = float(
                    any(
                        quality > 0 and contains_answer(q.text, example.gold_answers)
                        for q, quality in zip(response.quotes, qualities)
                    )
                )
                scores.append(count_factor * (0.5 * precision + 0.5 * support))
            else:
                scores.append(count_factor * precision)
        return scores

    def chunk_routing_reward(completions, **columns) -> list[float | None]:
        """F1 between cited chunks and gold chunks on answerable questions.

        A citation counts for its chunk only if the quote is of non-zero quality in the
        chunk it names. Precision is over quotes, recall over gold chunks.
        Undefined (None) on unanswerable questions.
        """
        scores: list[float | None] = []
        for response, example in _iter_samples(completions, columns):
            if not example.is_answerable or not example.gold_chunk_ids:
                scores.append(None)
                continue
            if not response.valid or response.abstained or not response.quotes:
                scores.append(0.0)
                continue
            gold = set(example.gold_chunk_ids)
            correct_quotes = 0
            covered: set[str] = set()
            for quote in response.quotes:
                source = example.chunks.get(quote.chunk_id, "")
                if quote.chunk_id in gold and quote_quality(quote, source, settings) > 0:
                    correct_quotes += 1
                    covered.add(quote.chunk_id)
            precision = correct_quotes / len(response.quotes)
            recall = len(covered) / len(gold)
            scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
        return scores

    def answer_faithfulness_reward(completions, **columns) -> list[float | None]:
        """Whether the answer is contained in the cited evidence (label-free given answerability).

        1 if the normalised answer occurs in a quote of non-zero quality, otherwise the
        best fraction of answer tokens found in one such quote. Capped at
        `answer_copy_cap` when the answer is nearly as long as the quote it comes from,
        so that pasting the quote as the answer is not a free maximum.
        Undefined (None) on unanswerable questions and for yes/no answers, which are
        not expected to appear in the text.
        """
        scores: list[float | None] = []
        for response, example in _iter_samples(completions, columns):
            if not example.is_answerable:
                scores.append(None)
                continue
            if not response.valid or response.abstained or not response.quotes:
                scores.append(0.0)
                continue
            answer = answer_tokens(response.answer)
            if normalize_answer(response.answer) in {"yes", "no"}:
                scores.append(None)
                continue
            if not answer:
                scores.append(0.0)
                continue
            context = _context(example)
            best = 0.0
            for quote in response.quotes:
                if quote_quality(quote, context, settings) == 0:
                    continue
                quote_tokens = answer_tokens(quote.text)
                if contains_answer(quote.text, [response.answer]):
                    score = 1.0
                else:
                    score = sum(t in set(quote_tokens) for t in answer) / len(answer)
                if quote_tokens and len(answer) >= settings.answer_copy_ratio * len(quote_tokens):
                    score = min(score, settings.answer_copy_cap)
                best = max(best, score)
            scores.append(best)
        return scores

    def abstention_rate(completions, **columns) -> list[float]:
        """Monitoring signal (use with weight 0): 1 if the completion abstains."""
        return [float(response.abstained) for response, _ in _iter_samples(completions, columns)]

    functions = {
        "format": format_reward,
        "answer_correctness": answer_correctness_reward,
        "quote_grounding": quote_grounding_reward,
        "chunk_routing": chunk_routing_reward,
        "answer_faithfulness": answer_faithfulness_reward,
        "abstention_rate": abstention_rate,
    }
    for name, fn in functions.items():
        fn.__name__ = name  # TRL logs each reward under `rewards/<__name__>/mean`
    return functions


def weighted_reward(
    completions: list,
    columns: dict[str, Any],
    weights: dict[str, float],
    settings: RewardSettings | None = None,
) -> list[tuple[float, float]]:
    """Total reward and maximum attainable reward for each completion.

    Mirrors the aggregation of `trl.GRPOTrainer` (weighted sum that skips `None`).
    Used by the tests and by the evaluation script, not by the trainer.
    """
    functions = build_reward_functions(settings)
    per_function = {name: functions[name](completions, **columns) for name in weights}
    totals = []
    for i in range(len(completions)):
        total, maximum = 0.0, 0.0
        for name, weight in weights.items():
            value = per_function[name][i]
            if value is None:
                continue
            total += weight * value
            maximum += weight
        totals.append((total, maximum))
    return totals
