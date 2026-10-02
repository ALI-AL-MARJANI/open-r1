import json

import pytest

from open_r1.grounded.data import convert_hotpotqa_example, convert_pubmedqa_example
from open_r1.grounded.metrics import aggregate, bootstrap_ci, paired_bootstrap, predicted_sentences, score_example
from open_r1.grounded.text import best_em_f1, contains_answer, exact_match, is_verbatim, normalize_answer, token_f1

from .test_data import HOTPOT, PUBMED

ABSTAIN = '{"extracted_quotes": [], "is_context_sufficient": false, "final_answer": ""}'


def _respond(answer, quotes):
    return json.dumps(
        {
            "extracted_quotes": [{"chunk_id": c, "exact_quote": q} for c, q in quotes],
            "is_context_sufficient": True,
            "final_answer": answer,
        }
    )


class TestText:
    def test_normalisation_matches_the_squad_script(self):
        assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
        assert exact_match("the Eiffel tower", "Eiffel Tower.") == 1.0

    def test_token_f1(self):
        assert token_f1("Roger II of Sicily", "Roger II") == pytest.approx(2 * 0.5 * 1.0 / 1.5)
        assert token_f1("", "") == 1.0
        assert token_f1("", "Roger") == 0.0
        assert token_f1("the", "Roger") == 0.0  # normalises to an empty prediction

    def test_best_over_gold_answers_and_no_answer(self):
        assert best_em_f1("Romanesque", ["a unique Romanesque idiom", "Romanesque"]) == (1.0, 1.0)
        assert best_em_f1("", []) == (1.0, 1.0)
        assert best_em_f1("something", []) == (0.0, 0.0)

    def test_verbatim_is_strict_except_for_whitespace(self):
        source = "The quick  brown\nfox."
        assert is_verbatim("quick brown fox", source)
        assert not is_verbatim("quick Brown fox", source)
        assert not is_verbatim("quick brown dog", source)
        assert not is_verbatim("  ", source)

    def test_contains_answer_respects_token_boundaries(self):
        assert contains_answer("It was built in 1924.", ["1924"])
        assert not contains_answer("It was built in 19245.", ["1924"])
        assert not contains_answer("anything", [""])

    def test_contains_answer_survives_punctuation_next_to_the_answer(self):
        assert contains_answer("Before Rollo's arrival, the region was poor.", ["Rollo"])
        assert contains_answer("There is a US$1,000,000 prize.", ["$1,000,000"])
        assert contains_answer("It is called THE Eiffel tower.", ["Eiffel Tower"])
        assert not contains_answer("The Normans arrived.", ["Norman"])
        assert not contains_answer("anything", ["   "])


class TestSquadRows:
    def test_reference_outputs(self, records, reward_weights):
        rows = [score_example(r, r["target"], reward_weights) for r in records]
        metrics = aggregate(rows)
        for key in ("format_rate", "em", "f1", "has_ans_f1", "no_ans_em", "evidence_recall", "chunk_f1"):
            assert metrics[key] == 1.0, key
        assert metrics["unverified_quote_rate"] == 0.0
        assert metrics["abstention_f1"] == 1.0
        assert metrics["reward_fraction"] == pytest.approx(1.0)
        assert metrics["answer_rate"] == 0.5

    def test_always_abstain(self, records):
        metrics = aggregate([score_example(r, ABSTAIN) for r in records])
        assert metrics["em"] == 0.5 and metrics["has_ans_em"] == 0.0 and metrics["no_ans_em"] == 1.0
        assert metrics["abstention_recall"] == 1.0 and metrics["abstention_precision"] == 0.5
        assert metrics["unverified_quote_rate"] is None  # no quotes were produced
        assert metrics["answers_with_unverified_quote"] is None

    def test_invalid_output_is_wrong_everywhere(self, records):
        metrics = aggregate([score_example(r, "no json") for r in records])
        assert metrics["format_rate"] == 0.0 and metrics["em"] == 0.0 and metrics["no_ans_em"] == 0.0
        assert metrics["abstention_recall"] == 0.0

    def test_unverified_quotes_are_counted(self, records):
        record = records[0]
        gold = json.loads(record["gold_chunk_ids"])[0]
        out = _respond("France", [(gold, "a region in France"), (gold, "Normandy lies in northern France")])
        row = score_example(record, out)
        assert row["n_quotes"] == 2 and row["n_unverified_quotes"] == 1
        assert row["evidence_found"] and row["chunk_precision"] == 0.5 and row["chunk_recall"] == 1.0
        metrics = aggregate([row])
        assert metrics["unverified_quote_rate"] == 0.5
        assert metrics["answers_with_unverified_quote"] == 1.0


class TestHotpotRows:
    @pytest.fixture()
    def record(self):
        return convert_hotpotqa_example(HOTPOT, seed=0)

    def _chunk(self, record, title):
        return next(c for c, t in json.loads(record["context_chunks"]).items() if t.startswith(title))

    def test_exact_supporting_facts(self, record):
        quotes = [
            (self._chunk(record, "School A"), "It was founded by a pianist in 1924."),
            (self._chunk(record, "Composer B"), "Gian Carlo Menotti taught at School A."),
        ]
        row = score_example(record, _respond("Gian Carlo Menotti", quotes))
        assert row["em"] == row["sp_em"] == row["sp_f1"] == row["joint_em"] == row["joint_f1"] == 1.0

    def test_partial_supporting_facts(self, record):
        quotes = [
            (self._chunk(record, "Composer B"), "Gian Carlo Menotti taught at School A."),
            (self._chunk(record, "Distractor C"), "Distractor C is a river."),
        ]
        row = score_example(record, _respond("Menotti", quotes))
        assert row["sp_em"] == 0.0
        assert row["sp_f1"] == pytest.approx(0.5)  # precision 1/2, recall 1/2
        assert 0 < row["joint_f1"] < row["sp_f1"]
        assert row["chunk_precision"] == 0.5 and row["chunk_recall"] == 0.5

    def test_quote_inside_a_sentence_maps_to_it(self, record):
        chunk = self._chunk(record, "School A")
        sentences = json.loads(record["chunk_sentences"])
        assert predicted_sentences([(chunk, "founded by a pianist")], sentences) == {(chunk, 1)}
        two_sentences = "School A is a conservatory. It was founded by a pianist"
        assert predicted_sentences([(chunk, two_sentences)], sentences) == {
            (chunk, 0),
            (chunk, 1),
        }

    def test_yes_no_mismatch_scores_zero(self, record):
        yes_record = {**record, "gold_answers": json.dumps(["yes"])}
        quotes = [(self._chunk(record, "Composer B"), "Gian Carlo Menotti taught at School A.")]
        assert score_example(yes_record, _respond("no", quotes))["f1"] == 0.0
        assert score_example(yes_record, _respond("Yes", quotes))["em"] == 1.0

    def test_abstention_scores_zero(self, record):
        row = score_example(record, ABSTAIN)
        assert row["f1"] == row["sp_f1"] == row["joint_f1"] == 0.0


class TestPubmedRows:
    def test_label_mapping(self):
        record = convert_pubmedqa_example(PUBMED)
        quote = [("c3", "Results text here.")]
        assert score_example(record, _respond("Yes, it does.", quote))["predicted_label"] == "yes"
        assert score_example(record, _respond("No.", quote))["predicted_label"] == "no"
        assert score_example(record, _respond("Unclear", quote))["predicted_label"] == "maybe"
        assert score_example(record, ABSTAIN)["predicted_label"] == "maybe"
        assert score_example(record, "garbage")["predicted_label"] == "maybe"

    def test_accuracy_and_macro_f1(self):
        record = convert_pubmedqa_example(PUBMED)
        rows = [score_example(record, _respond("Yes", [("c3", "Results text here.")])), score_example(record, ABSTAIN)]
        metrics = aggregate(rows)
        assert metrics["accuracy"] == 0.5
        assert 0 < metrics["macro_f1"] < 1


class TestBootstrap:
    def test_interval_contains_the_point_estimate(self, records):
        outputs = [r["target"] if i % 3 else ABSTAIN for i, r in enumerate(records)]
        rows = [score_example(r, o) for r, o in zip(records, outputs)]
        ci = bootstrap_ci(rows, n_resamples=300, seed=0)
        assert ci["f1"]["ci_low"] <= ci["f1"]["value"] <= ci["f1"]["ci_high"]
        assert ci["f1"]["ci_low"] < ci["f1"]["ci_high"]
        assert bootstrap_ci(rows, n_resamples=300, seed=0) == ci

    def test_degenerate_metric_has_a_zero_width_interval(self, records):
        rows = [score_example(r, r["target"]) for r in records]
        ci = bootstrap_ci(rows, n_resamples=100)
        assert ci["format_rate"] == {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0}

    def test_paired_bootstrap_detects_a_clear_difference(self, records):
        records = records * 5
        for i, record in enumerate(records):
            record = dict(record, id=f"{record['id']}-{i}")
            records[i] = record
        good = [score_example(r, r["target"]) for r in records]
        bad = [score_example(r, ABSTAIN) for r in records]
        result = paired_bootstrap(bad, good, "f1", n_resamples=500)
        assert result["difference"] == pytest.approx(0.5)
        assert result["ci_low"] > 0 and result["p_value"] < 0.05
        same = paired_bootstrap(good, good, "f1", n_resamples=200)
        assert same["difference"] == 0.0 and same["p_value"] == 1.0

    def test_paired_bootstrap_requires_the_same_examples(self, records):
        rows = [score_example(r, r["target"]) for r in records]
        with pytest.raises(ValueError):
            paired_bootstrap(rows, rows[:-1], "f1")
