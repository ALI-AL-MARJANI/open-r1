import json

import pytest

from open_r1.grounded.parsing import Quote
from open_r1.grounded.rewards import RewardSettings, build_reward_functions, quote_quality, weighted_reward

from .conftest import as_chat, columns_of

REWARDS = build_reward_functions()


def respond(answer, quotes, sufficient=True):
    return json.dumps(
        {
            "extracted_quotes": [{"chunk_id": c, "exact_quote": q} for c, q in quotes],
            "is_context_sufficient": sufficient,
            "final_answer": answer,
        }
    )


ABSTAIN = respond("", [], sufficient=False)


def score(name, record, output):
    return REWARDS[name](as_chat([output]), **columns_of([record]))[0]


def gold_chunk(record):
    return json.loads(record["gold_chunk_ids"])[0]


def distractor_chunk(record):
    gold = set(json.loads(record["gold_chunk_ids"]))
    return next(c for c in json.loads(record["context_chunks"]) if c not in gold)


@pytest.fixture()
def answerable(records):
    return records[0]  # "In what country is Normandy located?" -> France


@pytest.fixture()
def unanswerable(records):
    return next(r for r in records if not r["is_answerable"])


def test_reference_responses_get_the_maximum(records, reward_weights):
    totals = weighted_reward([r["target"] for r in records], columns_of(records), reward_weights)
    for (total, maximum), record in zip(totals, records):
        assert total == pytest.approx(maximum), record["id"]


def test_reward_names_are_stable():
    # TRL logs rewards under these names; the configs refer to them.
    assert {name: fn.__name__ for name, fn in REWARDS.items()} == {name: name for name in REWARDS}


def test_string_and_chat_completions_score_the_same(answerable):
    columns = columns_of([answerable])
    for fn in REWARDS.values():
        assert fn([answerable["target"]], **columns) == fn(as_chat([answerable["target"]]), **columns)


class TestFormat:
    def test_levels(self, answerable):
        assert score("format", answerable, "not json") == 0.0
        assert score("format", answerable, "{}") == 0.25
        assert score("format", answerable, respond(1, [])) == 0.5
        assert score("format", answerable, answerable["target"] + " trailing text") == 0.75
        assert score("format", answerable, respond("France", [])) == 0.75  # answers without a quote
        assert score("format", answerable, answerable["target"]) == 1.0
        assert score("format", answerable, ABSTAIN) == 1.0


class TestAnswerCorrectness:
    def test_answerable(self, answerable):
        quote = [(gold_chunk(answerable), "a region in France")]
        assert score("answer_correctness", answerable, respond("France", quote)) == 1.0
        assert score("answer_correctness", answerable, respond("Germany", quote)) == 0.0
        assert 0 < score("answer_correctness", answerable, respond("a region in France", quote)) < 1
        assert score("answer_correctness", answerable, ABSTAIN) == 0.0

    def test_unanswerable(self, unanswerable):
        assert score("answer_correctness", unanswerable, ABSTAIN) == 1.0
        assert score("answer_correctness", unanswerable, respond("France", [("c1", "x y z")])) == 0.0

    def test_invalid_output_scores_zero_even_when_unanswerable(self, unanswerable):
        assert score("answer_correctness", unanswerable, "I cannot answer.") == 0.0


class TestQuoteGrounding:
    def test_supporting_quote(self, answerable):
        out = respond("France", [(gold_chunk(answerable), "gave their name to Normandy, a region in France")])
        assert score("quote_grounding", answerable, out) == 1.0

    def test_verbatim_quote_without_the_answer_gets_half(self, answerable):
        out = respond("France", [(gold_chunk(answerable), "They were descended from Norse raiders")])
        assert score("quote_grounding", answerable, out) == 0.5

    def test_paraphrase_scores_zero(self, answerable):
        out = respond("France", [(gold_chunk(answerable), "Normandy is a region located in France")])
        assert score("quote_grounding", answerable, out) == 0.0

    def test_trivial_quotes_score_zero(self, answerable):
        for quote in ("the", "in the", "France", "of the and"):
            assert score("quote_grounding", answerable, respond("France", [(gold_chunk(answerable), quote)])) == 0.0

    def test_abstaining_on_an_answerable_question_scores_zero(self, answerable):
        assert score("quote_grounding", answerable, ABSTAIN) == 0.0

    def test_undefined_on_unanswerable_questions(self, unanswerable):
        assert score("quote_grounding", unanswerable, ABSTAIN) is None
        assert score("quote_grounding", unanswerable, "garbage") is None

    def test_many_quotes_are_scaled_down(self, answerable):
        chunk = gold_chunk(answerable)
        good = (chunk, "a region in France")
        fillers = [
            (chunk, "The Normans were the people"),
            (chunk, "They were descended from Norse raiders"),
            (chunk, "pirates from Denmark, Iceland and Norway"),
            (chunk, "Their leader Rollo agreed to swear fealty"),
            (chunk, "King Charles III of West Francia"),
        ]
        assert score("quote_grounding", answerable, respond("France", [good] + fillers)) == pytest.approx(0.5)

    def test_quote_spanning_two_chunks_is_not_verbatim(self, answerable):
        chunks = json.loads(answerable["context_chunks"])
        ids = list(chunks)
        spanning = chunks[ids[0]][-30:] + " " + chunks[ids[1]][:30]
        assert score("quote_grounding", answerable, respond("France", [(ids[0], spanning)])) == 0.0

    def test_long_quotes_lose_credit(self):
        settings = RewardSettings(quote_full_credit_words=5, quote_zero_credit_words=10)
        source = "one two three four five six seven eight nine ten eleven twelve"
        assert quote_quality(Quote("c1", "one two three four five"), source, settings) == 1.0
        assert 0 < quote_quality(Quote("c1", "one two three four five six seven"), source, settings) < 1
        assert quote_quality(Quote("c1", source), source, settings) == 0.0

    def test_yes_no_questions_use_precision_only(self, answerable):
        record = {**answerable, "gold_answers": json.dumps(["yes"])}
        out = respond("yes", [(gold_chunk(answerable), "They were descended from Norse raiders")])
        assert score("quote_grounding", record, out) == 1.0


class TestChunkRouting:
    def test_gold_chunk(self, answerable):
        out = respond("France", [(gold_chunk(answerable), "a region in France")])
        assert score("chunk_routing", answerable, out) == 1.0

    def test_right_quote_wrong_chunk_id(self, answerable):
        out = respond("France", [(distractor_chunk(answerable), "a region in France")])
        assert score("chunk_routing", answerable, out) == 0.0

    def test_quote_from_a_distractor(self, answerable):
        distractor = distractor_chunk(answerable)
        text = " ".join(json.loads(answerable["context_chunks"])[distractor].split()[:6])
        assert score("chunk_routing", answerable, respond("France", [(distractor, text)])) == 0.0

    def test_mixed_citations_get_partial_credit(self, answerable):
        distractor = distractor_chunk(answerable)
        text = " ".join(json.loads(answerable["context_chunks"])[distractor].split()[:6])
        out = respond("France", [(gold_chunk(answerable), "a region in France"), (distractor, text)])
        assert score("chunk_routing", answerable, out) == pytest.approx(2 / 3)

    def test_unknown_chunk_id(self, answerable):
        assert score("chunk_routing", answerable, respond("France", [("c99", "a region in France")])) == 0.0

    def test_undefined_on_unanswerable_questions(self, unanswerable):
        assert score("chunk_routing", unanswerable, ABSTAIN) is None


class TestAnswerFaithfulness:
    def test_answer_inside_the_quote(self, answerable):
        out = respond("France", [(gold_chunk(answerable), "a region in France")])
        assert score("answer_faithfulness", answerable, out) == 1.0

    def test_answer_absent_from_the_quote(self, answerable):
        out = respond("Germany", [(gold_chunk(answerable), "a region in France")])
        assert score("answer_faithfulness", answerable, out) == 0.0

    def test_answer_supported_only_by_an_unverified_quote(self, answerable):
        out = respond("Germany", [(gold_chunk(answerable), "Normandy is a region in Germany")])
        assert score("answer_faithfulness", answerable, out) == 0.0

    def test_copying_the_quote_as_the_answer_is_capped(self, answerable):
        quote = "gave their name to Normandy, a region in France"
        out = respond(quote, [(gold_chunk(answerable), quote)])
        assert score("answer_faithfulness", answerable, out) == RewardSettings().answer_copy_cap

    def test_undefined_for_yes_no_answers_and_unanswerable_questions(self, answerable, unanswerable):
        out = respond("Yes", [(gold_chunk(answerable), "a region in France")])
        assert score("answer_faithfulness", answerable, out) is None
        assert score("answer_faithfulness", unanswerable, ABSTAIN) is None


def test_abstention_rate_monitor(answerable):
    assert score("abstention_rate", answerable, ABSTAIN) == 1.0
    assert score("abstention_rate", answerable, answerable["target"]) == 0.0
    assert score("abstention_rate", answerable, "garbage") == 0.0


def test_none_depends_only_on_the_labels(records):
    """All completions of a prompt must be scored by the same set of rewards."""
    outputs = [ABSTAIN, "garbage", "{}", respond("x", [("c1", "one two three")]), respond(1, [])]
    for record in records:
        for name in ("quote_grounding", "chunk_routing"):
            undefined = {score(name, record, out) is None for out in outputs}
            assert len(undefined) == 1, (name, record["id"])


@pytest.mark.parametrize("bad_column", [None, 3, "not json", "[1, 2", {"a": 1}])
def test_malformed_dataset_columns_do_not_raise(answerable, bad_column):
    columns = {
        "context_chunks": [bad_column],
        "gold_answers": [bad_column],
        "gold_chunk_ids": [bad_column],
        "is_answerable": [True],
    }
    for fn in REWARDS.values():
        result = fn(as_chat([answerable["target"]]), **columns)
        assert len(result) == 1


def test_rewards_ignore_extra_trainer_kwargs(answerable):
    columns = columns_of([answerable])
    for fn in REWARDS.values():
        fn(as_chat([answerable["target"]]), prompts=[[]], completion_ids=[[1]], trainer_state=None, **columns)
