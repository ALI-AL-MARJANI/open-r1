"""Conversion of public QA datasets to the grounded schema.

Every example becomes a flat record:

    id               str   source example id
    dataset          str   "squad_v2" | "hotpotqa" | "pubmedqa"
    question         str
    user_message     str   context chunks + question, as shown to the model
    context_chunks   str   JSON {chunk_id: text}
    gold_answers     str   JSON [answer, ...]; [] for unanswerable questions
    gold_chunk_ids   str   JSON [chunk_id, ...]; [] for unanswerable questions
    is_answerable    bool
    target           str   reference response (SQuAD only), used by the SFT baseline
    chunk_sentences  str   JSON {chunk_id: [sentence, ...]} (HotpotQA only)
    gold_sentences   str   JSON [[chunk_id, sentence_index], ...] (HotpotQA only)
    label            str   yes | no | maybe (PubMedQA only)

Chunk ids are position-neutral (`c1`, `c2`, ...) and assigned after shuffling, so the
id carries no information about which chunk is gold. All randomness is derived from
string seeds (`random.Random(str)` is stable across processes, unlike `hash()`).

The conversion functions are pure and are unit-tested without network access; the
`load_*` functions download from the Hugging Face Hub.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from typing import Any

from .prompts import build_target, build_user_message
from .text import contains_answer, words

SQUAD_V2 = "rajpurkar/squad_v2"
HOTPOTQA = "hotpotqa/hotpot_qa"
PUBMEDQA = "qiaojin/PubMedQA"

_SENTENCE_ENDINGS = (". ", "? ", "! ", ".\n", "?\n", "!\n")


def _rng(*parts: Any) -> random.Random:
    return random.Random("|".join(str(p) for p in parts))


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _assign_chunk_ids(texts: list[str], rng: random.Random) -> tuple[dict[str, str], list[int]]:
    """Shuffle `texts` and name them c1..cN. Returns the chunks and the permutation used."""
    order = list(range(len(texts)))
    rng.shuffle(order)
    chunks = {f"c{rank + 1}": texts[source_index] for rank, source_index in enumerate(order)}
    return chunks, order


# ── SQuAD v2 ──────────────────────────────────────────────────────────────────


def supporting_span(context: str, answer_text: str, answer_start: int, max_words: int) -> str:
    """Sentence of `context` that contains the gold answer span.

    Falls back to a window of at most `max_words` words around the answer when the
    sentence is longer than that. The result is always a substring of `context`
    that contains `answer_text`.
    """
    answer_end = answer_start + len(answer_text)
    if context[answer_start:answer_end] != answer_text:
        # Misaligned offset in the source data: locate the answer by search instead.
        answer_start = context.find(answer_text)
        if answer_start < 0:
            return ""
        answer_end = answer_start + len(answer_text)

    starts = [context.rfind(end, 0, answer_start) for end in _SENTENCE_ENDINGS]
    sentence_start = max(starts) + 2 if max(starts) >= 0 else 0
    ends = [pos + 1 for pos in (context.find(end, answer_end - 1) for end in _SENTENCE_ENDINGS) if pos >= 0]
    sentence_end = min(ends) if ends else len(context)
    sentence = context[sentence_start:sentence_end].strip()
    if answer_text in sentence and len(words(sentence)) <= max_words:
        return sentence

    # Fallback: widen a window around the answer one whitespace-separated token at a
    # time, alternating sides, while the word count stays within the limit.
    left, right = answer_start, answer_end
    grow_left = True
    while left > 0 or right < len(context):
        if (grow_left and left > 0) or right >= len(context):
            new_left = context.rfind(" ", 0, max(left - 1, 0)) + 1
            new_right = right
        else:
            next_space = context.find(" ", right + 1)
            new_left = left
            new_right = len(context) if next_space < 0 else next_space
        if len(words(context[new_left:new_right])) > max_words:
            break
        left, right = new_left, new_right
        grow_left = not grow_left
    return context[left:right].strip()


def build_title_index(examples: Iterable[dict]) -> dict[str, list[str]]:
    """Map each article title to its distinct paragraphs, in first-seen order."""
    index: dict[str, list[str]] = {}
    for example in examples:
        paragraphs = index.setdefault(example["title"], [])
        if example["context"] not in paragraphs:
            paragraphs.append(example["context"])
    return index


def convert_squad_example(
    example: dict,
    title_index: dict[str, list[str]] | None = None,
    n_distractors: int = 0,
    seed: int = 0,
    target_max_words: int = 50,
) -> dict:
    """SQuAD v2 example -> grounded record, optionally with same-article distractors.

    Distractors are other paragraphs of the same Wikipedia article. For answerable
    questions, paragraphs that contain a gold answer string are not used as
    distractors, so the gold paragraph stays the only labelled evidence.
    """
    gold_answers = list(dict.fromkeys(example["answers"]["text"]))
    is_answerable = bool(gold_answers)
    rng = _rng("squad_v2", seed, example["id"])

    texts = [example["context"]]
    if n_distractors > 0 and title_index is not None:
        candidates = [
            p
            for p in title_index.get(example["title"], [])
            if p != example["context"] and not contains_answer(p, gold_answers)
        ]
        rng.shuffle(candidates)
        texts += candidates[:n_distractors]

    chunks, order = _assign_chunk_ids(texts, rng)
    gold_chunk_id = f"c{order.index(0) + 1}"

    if is_answerable:
        quote = supporting_span(
            example["context"], example["answers"]["text"][0], example["answers"]["answer_start"][0], target_max_words
        )
        target = build_target(example["answers"]["text"][0], [(gold_chunk_id, quote)])
    else:
        target = build_target(None, [])

    return {
        "id": example["id"],
        "dataset": "squad_v2",
        "question": example["question"],
        "user_message": build_user_message(example["question"], chunks),
        "context_chunks": _dumps(chunks),
        "gold_answers": _dumps(gold_answers),
        "gold_chunk_ids": _dumps([gold_chunk_id] if is_answerable else []),
        "is_answerable": is_answerable,
        "target": target,
    }


# ── HotpotQA (distractor setting) ─────────────────────────────────────────────


def convert_hotpotqa_example(example: dict, seed: int = 0) -> dict:
    """HotpotQA distractor example -> grounded record (10 paragraphs, 2 of them gold).

    The paragraph title is prepended to the chunk text ("Title: sentences"), as in the
    usual reading-comprehension formatting of this dataset. Sentence boundaries are
    kept so that quotes can be mapped back to supporting-fact sentences.
    """
    titles: list[str] = example["context"]["title"]
    sentence_lists: list[list[str]] = example["context"]["sentences"]
    texts = [f"{title}: {''.join(sents).strip()}" for title, sents in zip(titles, sentence_lists)]

    chunks, order = _assign_chunk_ids(texts, _rng("hotpotqa", seed, example["id"]))
    chunk_of_paragraph = {source_index: f"c{rank + 1}" for rank, source_index in enumerate(order)}
    chunk_of_title = {title: chunk_of_paragraph[i] for i, title in enumerate(titles)}

    gold_sentences = [
        [chunk_of_title[title], int(sent_id)]
        for title, sent_id in zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"])
        if title in chunk_of_title
    ]
    gold_chunk_ids = sorted({cid for cid, _ in gold_sentences}, key=lambda c: int(c[1:]))
    chunk_sentences = {chunk_of_paragraph[i]: [s.strip() for s in sents] for i, sents in enumerate(sentence_lists)}

    return {
        "id": example["id"],
        "dataset": "hotpotqa",
        "question": example["question"],
        "user_message": build_user_message(example["question"], chunks),
        "context_chunks": _dumps(chunks),
        "gold_answers": _dumps([example["answer"]]),
        "gold_chunk_ids": _dumps(gold_chunk_ids),
        "is_answerable": True,
        "chunk_sentences": _dumps(chunk_sentences),
        "gold_sentences": _dumps(gold_sentences),
    }


# ── PubMedQA (reasoning-required setting) ─────────────────────────────────────


def convert_pubmedqa_example(example: dict) -> dict:
    """PubMedQA example -> grounded record.

    The context is the abstract without its conclusion (the "reasoning-required"
    setting of Jin et al., 2019), one chunk per labelled section, in the original
    order. The gold label is yes / no / maybe.
    """
    sections: list[str] = example["context"]["contexts"]
    chunks = {f"c{i + 1}": text.strip() for i, text in enumerate(sections)}
    label = str(example["final_decision"]).lower()
    return {
        "id": str(example["pubid"]),
        "dataset": "pubmedqa",
        "question": example["question"],
        "user_message": build_user_message(example["question"], chunks),
        "context_chunks": _dumps(chunks),
        "gold_answers": _dumps([label]),
        "gold_chunk_ids": _dumps(list(chunks)),
        "is_answerable": True,
        "label": label,
    }


# ── Sub-sampling ──────────────────────────────────────────────────────────────


def select_subset(records: list[dict], n: int, seed: int, answerable_fraction: float | None = None) -> list[dict]:
    """Deterministic subset of `records`.

    Records are sorted by id before a seeded shuffle, so the result does not depend on
    the input order. With `answerable_fraction`, the subset is stratified on
    `is_answerable` (e.g. 0.5 for a balanced SQuAD v2 subset).
    """
    ordered = sorted(records, key=lambda r: r["id"])
    rng = _rng("subset", seed)
    if answerable_fraction is None:
        rng.shuffle(ordered)
        return ordered[:n]
    answerable = [r for r in ordered if r["is_answerable"]]
    unanswerable = [r for r in ordered if not r["is_answerable"]]
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    n_answerable = round(n * answerable_fraction)
    subset = answerable[:n_answerable] + unanswerable[: n - n_answerable]
    rng.shuffle(subset)
    return subset


# ── Loaders (network) ─────────────────────────────────────────────────────────


def load_squad_v2(
    split: str,
    n: int | None = None,
    seed: int = 0,
    n_distractors: int = 0,
    answerable_fraction: float | None = None,
) -> list[dict]:
    """Load SQuAD v2, select the subset on the raw examples, then convert only those."""
    from datasets import load_dataset

    raw = load_dataset(SQUAD_V2, split=split)
    stubs = [{"id": i, "is_answerable": bool(a["text"])} for i, a in zip(raw["id"], raw["answers"])]
    if n is None:
        keep = {s["id"] for s in stubs}
    else:
        keep = {s["id"] for s in select_subset(stubs, n, seed, answerable_fraction)}
    title_index = build_title_index(raw) if n_distractors > 0 else None
    records = [convert_squad_example(ex, title_index, n_distractors, seed) for ex in raw if ex["id"] in keep]
    return sorted(records, key=lambda r: r["id"])


def load_hotpotqa(split: str = "validation", seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    raw = load_dataset(HOTPOTQA, "distractor", split=split)
    return [convert_hotpotqa_example(ex, seed) for ex in raw]


def load_pubmedqa() -> list[dict]:
    from datasets import load_dataset

    raw = load_dataset(PUBMEDQA, "pqa_labeled", split="train")
    return [convert_pubmedqa_example(ex) for ex in raw]


def load_records(
    dataset: str,
    split: str,
    n: int | None,
    seed: int,
    n_distractors: int = 0,
    answerable_fraction: float | None = None,
) -> list[dict]:
    """Load a dataset by name and return a deterministic subset of `n` records, sorted by id."""
    if dataset == "squad_v2":
        return load_squad_v2(split, n, seed, n_distractors, answerable_fraction)
    if dataset == "hotpotqa":
        records = load_hotpotqa(split, seed=seed)
    elif dataset == "pubmedqa":
        records = load_pubmedqa()
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    if n is not None:
        records = select_subset(records, n, seed, answerable_fraction)
    return sorted(records, key=lambda r: r["id"])
