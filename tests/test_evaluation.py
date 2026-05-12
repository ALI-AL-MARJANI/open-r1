"""Unit tests for the Grounded-R1 evaluation pipeline."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate_grounded import (
    MetricsAccumulator,
    _best_f1_em,
    _exact_match,
    _is_grounded,
    _normalise,
    _parse_response,
    _token_f1,
)


# ── Answer normalisation ──────────────────────────────────────────────────────

class TestNormalise:
    def test_lowercases(self):
        assert _normalise("Paris") == "paris"

    def test_removes_articles(self):
        assert _normalise("the capital") == "capital"
        assert _normalise("a city") == "city"

    def test_removes_punctuation(self):
        assert _normalise("Paris, France.") == "paris france"

    def test_collapses_whitespace(self):
        assert _normalise("  Paris   France  ") == "paris france"


class TestTokenF1:
    def test_exact_match_is_1(self):
        assert _token_f1("Paris", "Paris") == 1.0

    def test_no_overlap_is_0(self):
        assert _token_f1("London", "Paris") == 0.0

    def test_partial_overlap(self):
        f1 = _token_f1("Paris is the capital", "Paris is the city")
        assert 0.5 < f1 < 1.0

    def test_empty_prediction(self):
        assert _token_f1("", "Paris") == 0.0

    def test_best_f1_picks_max_across_golds(self):
        f1, em = _best_f1_em("Denver Broncos", ["Denver Broncos", "The Denver Broncos"])
        assert f1 == 1.0
        assert em is True


# ── MetricsAccumulator ────────────────────────────────────────────────────────

def _make_squad_example(question, context, title, answer_texts):
    """Build a minimal SQuAD-style example dict."""
    return {
        "id": "test_id",
        "question": question,
        "context": context,
        "title": title,
        "answers": {"text": answer_texts, "answer_start": [0] * len(answer_texts)},
    }


def _make_output(
    is_sufficient: bool,
    final_answer: str,
    quotes: list[dict],
    reasoning: str = "I checked the context and it is sufficient.",
) -> str:
    return json.dumps({
        "reasoning_path": reasoning,
        "is_context_sufficient": is_sufficient,
        "final_answer": final_answer,
        "extracted_quotes": quotes,
    })


CONTEXT = "Paris is the capital of France and a major European city."


class TestMetricsAccumulatorAnswerable:
    def setup_method(self):
        self.acc = MetricsAccumulator()
        self.example = _make_squad_example(
            question="What is the capital of France?",
            context=CONTEXT,
            title="France",
            answer_texts=["Paris"],
        )

    def test_perfect_prediction(self):
        raw = _make_output(
            is_sufficient=True,
            final_answer="Paris",
            quotes=[{"chunk_id": "France_0", "exact_quote": "Paris is the capital of France"}],
        )
        self.acc.update(self.example, raw, json.loads(raw))
        m = self.acc.compute()

        assert m["format"]["format_rate"] == 1.0
        assert m["grounding"]["quote_accuracy"] == 1.0
        assert m["grounding"]["hallucination_rate"] == 0.0
        assert m["grounding"]["coverage_rate"] == 1.0
        assert m["answer_quality"]["exact_match"] == 1.0
        assert m["answer_quality"]["f1"] == 1.0
        # Sufficiency: model says sufficient, gold is answerable → tn_suff += 1
        assert m["sufficiency_classification"]["accuracy"] == 1.0

    def test_hallucinated_quote_penalised(self):
        raw = _make_output(
            is_sufficient=True,
            final_answer="Paris",
            quotes=[{"chunk_id": "France_0", "exact_quote": "London is the capital of England"}],
        )
        self.acc.update(self.example, raw, json.loads(raw))
        m = self.acc.compute()

        assert m["grounding"]["quote_accuracy"] == 0.0
        assert m["grounding"]["hallucination_rate"] == 1.0

    def test_invalid_json_zeros_out(self):
        self.acc.update(self.example, "not json", None)
        m = self.acc.compute()
        assert m["format"]["format_rate"] == 0.0
        assert m["grounding"]["quote_accuracy"] == 0.0


class TestMetricsAccumulatorUnanswerable:
    def setup_method(self):
        self.acc = MetricsAccumulator()
        self.example = _make_squad_example(
            question="What is the capital of Mars?",
            context=CONTEXT,
            title="France",
            answer_texts=[],  # unanswerable
        )

    def test_correct_abstention(self):
        raw = _make_output(
            is_sufficient=False,
            final_answer="The provided context does not contain sufficient information to answer this question.",
            quotes=[],
            reasoning="The context does not contain any information about the capital of Mars.",
        )
        self.acc.update(self.example, raw, json.loads(raw))
        m = self.acc.compute()

        assert m["grounding"]["abstention_rate"] == 1.0
        assert m["grounding"]["abstention_quality"] == 1.0
        assert m["sufficiency_classification"]["recall"] == 1.0
        # No answerable examples → coverage and answer quality undefined
        assert m["grounding"]["coverage_rate"] == 0.0

    def test_hallucinating_on_unanswerable(self):
        """Model says sufficient when gold is unanswerable — worst failure mode."""
        raw = _make_output(
            is_sufficient=True,
            final_answer="The capital of Mars is Olympus City.",
            quotes=[{"chunk_id": "France_0", "exact_quote": "Paris is the capital of France"}],
        )
        self.acc.update(self.example, raw, json.loads(raw))
        m = self.acc.compute()

        assert m["grounding"]["abstention_rate"] == 0.0
        assert m["sufficiency_classification"]["recall"] == 0.0
        # fn_insuff = 1, tp_insuff = 0 → recall = 0


class TestMetricsMixedBatch:
    def test_mixed_batch_aggregation(self):
        acc = MetricsAccumulator()
        answerable = _make_squad_example("Q1?", CONTEXT, "T", ["Paris"])
        unanswerable = _make_squad_example("Q2?", CONTEXT, "T", [])

        # Perfect answerable
        raw_a = _make_output(True, "Paris", [{"chunk_id": "T_0", "exact_quote": "Paris is the capital of France"}])
        acc.update(answerable, raw_a, json.loads(raw_a))

        # Perfect abstention
        raw_u = _make_output(False, "The provided context does not contain sufficient information.", [],
                             reasoning="The context does not mention the capital of Mars, insufficient.")
        acc.update(unanswerable, raw_u, json.loads(raw_u))

        m = acc.compute()
        assert m["total_examples"] == 2
        assert m["answerable"] == 1
        assert m["unanswerable"] == 1
        assert m["grounding"]["quote_accuracy"] == 1.0
        assert m["grounding"]["abstention_rate"] == 1.0
        assert m["sufficiency_classification"]["accuracy"] == 1.0
        assert m["answer_quality"]["exact_match"] == 1.0


# ── Parse response ────────────────────────────────────────────────────────────

class TestParseResponse:
    def test_bare_json(self):
        payload = {"reasoning_path": "test", "is_context_sufficient": True,
                   "final_answer": "Paris", "extracted_quotes": []}
        assert _parse_response(json.dumps(payload)) == payload

    def test_json_code_block(self):
        payload = {"reasoning_path": "x", "is_context_sufficient": False,
                   "final_answer": "N/A", "extracted_quotes": []}
        content = f"```json\n{json.dumps(payload)}\n```"
        assert _parse_response(content) == payload

    def test_garbage_returns_none(self):
        assert _parse_response("I cannot answer this.") is None
