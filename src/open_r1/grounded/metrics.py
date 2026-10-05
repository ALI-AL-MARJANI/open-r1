"""Evaluation metrics, bootstrap confidence intervals and paired comparison.

Scoring is done in two steps so that confidence intervals can be computed by
resampling examples: `score_example` produces one row of raw counts per example,
and `aggregate` turns a list of rows into the reported metrics.

Metric definitions
------------------
format_rate            share of outputs that parse to a schema-valid response.
em / f1                SQuAD v2 official exact match and token F1. An abstention is the
                       empty prediction. An output that does not parse scores 0 on both,
                       including on unanswerable questions.
has_ans_* / no_ans_*   the same, restricted to answerable / unanswerable questions.
abstention_precision,  abstention as a detector of unanswerable questions
abstention_recall,     (positive class = unanswerable).
abstention_f1
answer_rate            share of outputs that give an answer (valid and not abstaining).
unverified_quote_rate  quotes that are not a verbatim substring of any context chunk,
                       over all quotes produced ("hallucinated citations", micro-average).
answers_with_unverified_quote
                       share of answering outputs with at least one such quote.
evidence_recall        share of answerable questions with a verbatim quote that
                       contains a gold answer string (0 for abstentions and invalid outputs).
chunk_precision,       cited chunk ids against gold chunk ids on answerable questions; a
chunk_recall, chunk_f1 citation counts only if its quote is verbatim in the chunk it names.
reward_fraction        total training reward divided by the maximum attainable reward.

HotpotQA adds the official supporting-fact metrics (`sp_em`, `sp_f1`, `joint_em`,
`joint_f1`; Yang et al., 2018). The predicted supporting sentences are derived from
the quotes: a sentence is predicted when a verbatim quote from its paragraph covers
at least half of it (or lies entirely inside it).

PubMedQA adds 3-way `accuracy` and `macro_f1` over yes / no / maybe, where the
predicted label is the first word of the answer if it is yes or no, and maybe otherwise
(including abstentions and invalid outputs).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from typing import Any

from .parsing import parse_response
from .rewards import RewardSettings, weighted_reward
from .text import best_em_f1, collapse_whitespace, contains_answer, is_verbatim, normalize_answer, token_prf

Row = dict[str, Any]
PUBMEDQA_LABELS = ("yes", "no", "maybe")


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def predicted_sentences(quotes: list[tuple[str, str]], chunk_sentences: dict[str, list[str]]) -> set[tuple[str, int]]:
    """Map (chunk_id, quote) pairs to the (chunk_id, sentence_index) pairs they cover."""
    predicted: set[tuple[str, int]] = set()
    for chunk_id, quote in quotes:
        quote_text = collapse_whitespace(quote)
        for index, sentence in enumerate(chunk_sentences.get(chunk_id, [])):
            sentence_text = collapse_whitespace(sentence)
            if not sentence_text:
                continue
            if sentence_text in quote_text or quote_text in sentence_text:
                predicted.add((chunk_id, index))
                continue
            # Partial coverage: longest prefix or suffix of the sentence inside the quote.
            half = len(sentence_text) // 2 + 1
            if sentence_text[:half] in quote_text or sentence_text[-half:] in quote_text:
                predicted.add((chunk_id, index))
    return predicted


def _hotpot_answer_prf(prediction: str, gold: str) -> tuple[float, float, float, float]:
    """Answer EM, precision, recall, F1 as in the official HotpotQA script."""
    norm_pred, norm_gold = normalize_answer(prediction), normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if (norm_pred in special or norm_gold in special) and norm_pred != norm_gold:
        return 0.0, 0.0, 0.0, 0.0
    precision, recall, f1 = token_prf(prediction, gold)
    return float(norm_pred == norm_gold), precision, recall, f1


def score_example(
    record: dict,
    output: str,
    reward_weights: dict[str, float] | None = None,
    settings: RewardSettings | None = None,
) -> Row:
    """Raw per-example counts for one model output."""
    response = parse_response(output)
    chunks: dict[str, str] = json.loads(record["context_chunks"])
    gold_answers: list[str] = json.loads(record["gold_answers"])
    gold_chunk_ids: set[str] = set(json.loads(record["gold_chunk_ids"]))
    is_answerable = bool(record["is_answerable"])
    answered = response.valid and not response.abstained

    prediction = response.answer if answered else ""
    if response.valid:
        em, f1 = best_em_f1(prediction, gold_answers if is_answerable else [])
    else:
        em, f1 = 0.0, 0.0

    quotes = response.quotes if response.valid else ()
    verbatim_any = [any(is_verbatim(q.text, text) for text in chunks.values()) for q in quotes]
    verbatim_in_named = [is_verbatim(q.text, chunks.get(q.chunk_id, "")) for q in quotes]
    correct_citations = [ok and q.chunk_id in gold_chunk_ids for q, ok in zip(quotes, verbatim_in_named)]
    covered = {q.chunk_id for q, ok in zip(quotes, correct_citations) if ok}

    row: Row = {
        "id": record["id"],
        "is_answerable": is_answerable,
        "format_valid": response.valid,
        "abstained": response.abstained,
        "answered": answered,
        "em": em,
        "f1": f1,
        "n_quotes": len(quotes),
        "n_unverified_quotes": sum(not ok for ok in verbatim_any),
        "evidence_found": bool(
            is_answerable
            and answered
            and any(ok and contains_answer(q.text, gold_answers) for q, ok in zip(quotes, verbatim_any))
        ),
        "chunk_precision": (sum(correct_citations) / len(quotes)) if (answered and quotes) else 0.0,
        "chunk_recall": (len(covered) / len(gold_chunk_ids)) if (answered and gold_chunk_ids) else 0.0,
    }

    if reward_weights:
        columns = {
            "context_chunks": [record["context_chunks"]],
            "gold_answers": [record["gold_answers"]],
            "gold_chunk_ids": [record["gold_chunk_ids"]],
            "is_answerable": [is_answerable],
        }
        row["reward"], row["reward_max"] = weighted_reward([output], columns, reward_weights, settings)[0]

    if record["dataset"] == "hotpotqa":
        gold_sentences = {(cid, int(idx)) for cid, idx in json.loads(record["gold_sentences"])}
        valid_quotes = [(q.chunk_id, q.text) for q, ok in zip(quotes, verbatim_in_named) if ok]
        predicted = predicted_sentences(valid_quotes, json.loads(record["chunk_sentences"])) if answered else set()
        true_positives = len(predicted & gold_sentences)
        sp_precision = true_positives / len(predicted) if predicted else 0.0
        sp_recall = true_positives / len(gold_sentences) if gold_sentences else 0.0
        ans_em, ans_precision, ans_recall, ans_f1 = _hotpot_answer_prf(prediction, gold_answers[0])
        if not response.valid:
            ans_em = ans_precision = ans_recall = ans_f1 = 0.0
        sp_em = float(predicted == gold_sentences)
        row.update(
            em=ans_em,
            f1=ans_f1,
            sp_em=sp_em,
            sp_f1=_f1(sp_precision, sp_recall),
            joint_em=ans_em * sp_em,
            joint_f1=_f1(ans_precision * sp_precision, ans_recall * sp_recall),
        )

    if record["dataset"] == "pubmedqa":
        first_word = normalize_answer(prediction).split(" ")[0] if answered else ""
        row["label"] = record["label"]
        row["predicted_label"] = first_word if first_word in ("yes", "no") else "maybe"
        row["em"] = row["f1"] = float(row["predicted_label"] == row["label"])

    return row


def aggregate(rows: list[Row]) -> dict[str, float | None]:
    """Reported metrics for a list of per-example rows. `None` marks an undefined metric."""
    answerable = [r for r in rows if r["is_answerable"]]
    unanswerable = [r for r in rows if not r["is_answerable"]]
    answering = [r for r in rows if r["answered"]]
    n_quotes = sum(r["n_quotes"] for r in rows)

    metrics: dict[str, float | None] = {
        "format_rate": _mean([float(r["format_valid"]) for r in rows]),
        "em": _mean([r["em"] for r in rows]),
        "f1": _mean([r["f1"] for r in rows]),
        "has_ans_em": _mean([r["em"] for r in answerable]),
        "has_ans_f1": _mean([r["f1"] for r in answerable]),
        "no_ans_em": _mean([r["em"] for r in unanswerable]),
        "answer_rate": _mean([float(r["answered"]) for r in rows]),
        "unverified_quote_rate": (sum(r["n_unverified_quotes"] for r in rows) / n_quotes) if n_quotes else None,
        "answers_with_unverified_quote": _mean([float(r["n_unverified_quotes"] > 0) for r in answering]),
        "evidence_recall": _mean([float(r["evidence_found"]) for r in answerable]),
        "chunk_precision": _mean([r["chunk_precision"] for r in answerable]),
        "chunk_recall": _mean([r["chunk_recall"] for r in answerable]),
        "chunk_f1": _mean([_f1(r["chunk_precision"], r["chunk_recall"]) for r in answerable]),
    }

    if unanswerable:
        abstained = [r for r in rows if r["abstained"]]
        true_positives = sum(1 for r in abstained if not r["is_answerable"])
        precision = true_positives / len(abstained) if abstained else 0.0
        recall = true_positives / len(unanswerable)
        metrics.update(abstention_precision=precision, abstention_recall=recall, abstention_f1=_f1(precision, recall))

    if rows and "reward" in rows[0]:
        total_max = sum(r["reward_max"] for r in rows)
        metrics["reward_fraction"] = (sum(r["reward"] for r in rows) / total_max) if total_max else None

    if rows and "sp_f1" in rows[0]:
        for key in ("sp_em", "sp_f1", "joint_em", "joint_f1"):
            metrics[key] = _mean([r[key] for r in rows])

    if rows and "predicted_label" in rows[0]:
        metrics["accuracy"] = _mean([float(r["predicted_label"] == r["label"]) for r in rows])
        per_label = []
        for label in PUBMEDQA_LABELS:
            tp = sum(1 for r in rows if r["predicted_label"] == label and r["label"] == label)
            n_predicted = sum(1 for r in rows if r["predicted_label"] == label)
            n_gold = sum(1 for r in rows if r["label"] == label)
            per_label.append(_f1(tp / n_predicted if n_predicted else 0.0, tp / n_gold if n_gold else 0.0))
        metrics["macro_f1"] = sum(per_label) / len(per_label)

    return metrics


def bootstrap_ci(
    rows: list[Row],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    aggregate_fn: Callable[[list[Row]], dict] = aggregate,
) -> dict[str, dict[str, float | None]]:
    """Point estimate and percentile bootstrap interval for every metric.

    Examples are resampled with replacement. A metric that is undefined in more than
    5% of the resamples gets no interval.
    """
    point = aggregate_fn(rows)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {key: [] for key in point}
    n = len(rows)
    for _ in range(n_resamples):
        resampled = aggregate_fn([rows[rng.randrange(n)] for _ in range(n)])
        for key, value in resampled.items():
            if value is not None and key in samples:
                samples[key].append(value)

    alpha = (1 - confidence) / 2
    result: dict[str, dict[str, float | None]] = {}
    for key, value in point.items():
        values = sorted(samples[key])
        if value is None or len(values) < 0.95 * n_resamples:
            result[key] = {"value": value, "ci_low": None, "ci_high": None}
        else:
            result[key] = {
                "value": value,
                "ci_low": values[int(alpha * (len(values) - 1))],
                "ci_high": values[int((1 - alpha) * (len(values) - 1))],
            }
    return result


def paired_bootstrap(
    rows_a: list[Row],
    rows_b: list[Row],
    metric: str,
    n_resamples: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap test of `metric(B) - metric(A)` on the same examples.

    Returns the observed difference, its 95% percentile interval, and a two-sided
    p-value: twice the share of resamples in which the difference has the opposite
    sign to the observed one (or is zero).
    """
    by_id = {r["id"]: r for r in rows_b}
    pairs = [(r, by_id[r["id"]]) for r in rows_a if r["id"] in by_id]
    if len(pairs) != len(rows_a) or len(pairs) != len(rows_b):
        raise ValueError("The two runs were not evaluated on the same examples.")

    def difference(sample: list[tuple[Row, Row]]) -> float | None:
        a = aggregate([p[0] for p in sample])[metric]
        b = aggregate([p[1] for p in sample])[metric]
        return None if a is None or b is None else b - a

    observed = difference(pairs)
    if observed is None:
        raise ValueError(f"Metric {metric!r} is undefined on these runs.")
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_resamples):
        value = difference([pairs[rng.randrange(n)] for _ in range(n)])
        if value is not None:
            diffs.append(value)
    diffs.sort()
    opposite = sum(1 for d in diffs if d * observed <= 0)
    return {
        "metric": metric,
        "difference": observed,
        "ci_low": diffs[int(0.025 * (len(diffs) - 1))],
        "ci_high": diffs[int(0.975 * (len(diffs) - 1))],
        "p_value": min(1.0, 2 * opposite / len(diffs)),
        "n_examples": n,
        "n_resamples": len(diffs),
    }
