import json
import subprocess
import sys

from open_r1.grounded.data import (
    build_title_index,
    convert_hotpotqa_example,
    convert_pubmedqa_example,
    convert_squad_example,
    select_subset,
    supporting_span,
)
from open_r1.grounded.parsing import parse_response
from open_r1.grounded.text import is_verbatim, words

HOTPOT = {
    "id": "hp1",
    "question": "Which composer taught at the school founded by the pianist?",
    "answer": "Gian Carlo Menotti",
    "supporting_facts": {"title": ["School A", "Composer B"], "sent_id": [1, 0]},
    "context": {
        "title": ["School A", "Composer B", "Distractor C", "Distractor D"],
        "sentences": [
            ["School A is a conservatory.", " It was founded by a pianist in 1924."],
            ["Gian Carlo Menotti taught at School A.", " He was born in Italy."],
            ["Distractor C is a river.", " It flows north."],
            ["Distractor D is a town."],
        ],
    },
}

PUBMED = {
    "pubid": 123,
    "question": "Does the treatment reduce mortality?",
    "context": {"contexts": ["Background text here.", "Methods text here.", "Results text here."], "labels": []},
    "long_answer": "The treatment reduces mortality.",
    "final_decision": "yes",
}


def test_squad_record_schema(records):
    for record in records:
        chunks = json.loads(record["context_chunks"])
        gold = json.loads(record["gold_chunk_ids"])
        assert list(chunks) == [f"c{i + 1}" for i in range(len(chunks))]
        assert len(chunks) == 3
        assert bool(gold) == record["is_answerable"] == bool(json.loads(record["gold_answers"]))
        for chunk_id, text in chunks.items():
            assert f'[CHUNK id="{chunk_id}"]\n{text}' in record["user_message"]
        assert record["user_message"].endswith(f"QUESTION: {record['question']}")


def test_squad_targets_are_valid_and_verbatim(records):
    for record in records:
        response = parse_response(record["target"])
        chunks = json.loads(record["context_chunks"])
        assert response.valid and response.consistent
        assert response.abstained == (not record["is_answerable"])
        for quote in response.quotes:
            assert quote.chunk_id in json.loads(record["gold_chunk_ids"])
            assert is_verbatim(quote.text, chunks[quote.chunk_id])
            assert response.answer in quote.text


def test_gold_chunk_position_is_not_constant(records):
    positions = {json.loads(r["gold_chunk_ids"])[0] for r in records if r["is_answerable"]}
    assert len(positions) > 1


def test_distractors_do_not_contain_the_gold_answer(raw_examples):
    leaking = "A paragraph about France and the French kingdom, which is not the gold paragraph."
    index = build_title_index(raw_examples)
    index["Normans"].append(leaking)
    record = convert_squad_example(raw_examples[0], index, n_distractors=10, seed=0)
    assert leaking not in json.loads(record["context_chunks"]).values()


def test_no_distractors_gives_a_single_chunk(raw_examples):
    record = convert_squad_example(raw_examples[0])
    assert list(json.loads(record["context_chunks"])) == ["c1"]
    assert json.loads(record["gold_chunk_ids"]) == ["c1"]


def test_conversion_is_deterministic_across_processes(raw_examples):
    """`hash()` is salted per process; the conversion must not depend on it."""
    code = (
        "import json, sys; sys.path[:0] = ['src', '.']\n"
        "from tests.grounded.conftest import RAW_EXAMPLES\n"
        "from open_r1.grounded.data import build_title_index, convert_squad_example\n"
        "index = build_title_index(RAW_EXAMPLES)\n"
        "print(json.dumps([convert_squad_example(e, index, 2, 7) for e in RAW_EXAMPLES], sort_keys=True))\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, env={"PYTHONHASHSEED": seed}
        ).stdout
        for seed in ("1", "2")
    }
    assert len(outputs) == 1
    index = build_title_index(raw_examples)
    here = json.dumps([convert_squad_example(e, index, 2, 7) for e in raw_examples], sort_keys=True)
    assert outputs == {here + "\n"}


def test_supporting_span_returns_the_sentence():
    context = "First sentence here. The answer is forty-two, as stated. Last sentence."
    span = supporting_span(context, "forty-two", context.find("forty-two"), max_words=50)
    assert span == "The answer is forty-two, as stated."


def test_supporting_span_clips_long_sentences():
    context = " ".join(f"w{i}" for i in range(200)) + " ANSWER " + " ".join(f"v{i}" for i in range(200)) + "."
    span = supporting_span(context, "ANSWER", context.find("ANSWER"), max_words=20)
    assert "ANSWER" in span and span in context
    assert len(words(span)) <= 20


def test_supporting_span_counts_words_like_the_reward_does():
    # Hyphens and thousands separators split into several regex words.
    filler = " ".join(["state-of-the-art 1,000,000"] * 60)
    context = f"{filler} ANSWER {filler}."
    span = supporting_span(context, "ANSWER", context.find("ANSWER"), max_words=50)
    assert "ANSWER" in span and span in context
    assert 30 < len(words(span)) <= 50


def test_supporting_span_recovers_from_a_wrong_offset():
    context = "Alpha beta. The capital is Paris. Gamma."
    assert supporting_span(context, "Paris", 0, max_words=50) == "The capital is Paris."
    assert supporting_span(context, "Berlin", 3, max_words=50) == ""


def test_select_subset_is_deterministic_and_stratified(records):
    shuffled = list(reversed(records))
    first = select_subset(records, 6, seed=1, answerable_fraction=0.5)
    second = select_subset(shuffled, 6, seed=1, answerable_fraction=0.5)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert sum(r["is_answerable"] for r in first) == 3
    assert {r["id"] for r in select_subset(records, 6, seed=2, answerable_fraction=0.5)} != {r["id"] for r in first}


def test_hotpotqa_conversion():
    record = convert_hotpotqa_example(HOTPOT, seed=0)
    chunks = json.loads(record["context_chunks"])
    sentences = json.loads(record["chunk_sentences"])
    gold_sentences = json.loads(record["gold_sentences"])
    gold_chunks = json.loads(record["gold_chunk_ids"])

    assert len(chunks) == 4 and record["is_answerable"]
    assert json.loads(record["gold_answers"]) == ["Gian Carlo Menotti"]
    assert len(gold_chunks) == 2
    assert {chunks[c].split(":")[0] for c in gold_chunks} == {"School A", "Composer B"}
    resolved = {sentences[cid][idx] for cid, idx in gold_sentences}
    assert resolved == {"It was founded by a pianist in 1924.", "Gian Carlo Menotti taught at School A."}
    for cid, sents in sentences.items():
        assert all(is_verbatim(s, chunks[cid]) for s in sents)


def test_pubmedqa_conversion():
    record = convert_pubmedqa_example(PUBMED)
    assert record["label"] == "yes"
    assert list(json.loads(record["context_chunks"])) == ["c1", "c2", "c3"]
    assert "The treatment reduces mortality." not in record["user_message"]  # conclusion is withheld
