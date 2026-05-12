"""Unit tests for domain-specific dataset preparation (v3 — PubMedQA)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_domain_dataset import convert_pubmedqa_example


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_pubmedqa(pubid, question, labels, sentences, long_answer, final_decision):
    return {
        "pubid": pubid,
        "question": question,
        "context": {"labels": labels, "sentences": sentences},
        "long_answer": long_answer,
        "final_decision": final_decision,
    }


LABELS_IMRAD = ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]
SENTENCES_IMRAD = [
    ["BRCA1 mutations have been associated with elevated cancer risk."],
    ["We analysed 1000 patients over 10 years using prospective cohort design."],
    ["Mutation carriers had 5.2x higher breast cancer incidence (p < 0.001)."],
    ["BRCA1 mutations significantly increase lifetime breast cancer risk."],
]
LONG_ANSWER = "Yes, BRCA1 mutations are strongly associated with increased breast cancer risk."

EXAMPLE_YES = _make_pubmedqa(
    pubid="12345678",
    question="Do BRCA1 mutations increase breast cancer risk?",
    labels=LABELS_IMRAD,
    sentences=SENTENCES_IMRAD,
    long_answer=LONG_ANSWER,
    final_decision="yes",
)

EXAMPLE_NO = _make_pubmedqa(
    pubid="87654321",
    question="Does aspirin prevent heart disease in healthy adults?",
    labels=["BACKGROUND", "RESULTS", "CONCLUSIONS"],
    sentences=[
        ["Aspirin inhibits platelet aggregation."],
        ["No significant reduction in cardiac events was found (HR=0.98, CI 0.91-1.05)."],
        ["Aspirin prophylaxis is not recommended for healthy adults without prior cardiac history."],
    ],
    long_answer="No, aspirin does not prevent heart disease in healthy adults.",
    final_decision="no",
)

EXAMPLE_MAYBE = _make_pubmedqa(
    pubid="11112222",
    question="Is intermittent fasting superior to caloric restriction for weight loss?",
    labels=["BACKGROUND", "RESULTS"],
    sentences=[
        ["Intermittent fasting (IF) and caloric restriction (CR) are two popular weight-loss strategies."],
        ["Both IF and CR produced similar weight loss of 5-8% over 6 months; differences were not significant."],
    ],
    long_answer="Evidence is mixed; neither approach shows clear superiority.",
    final_decision="maybe",
)


# ── Schema correctness ────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_output_columns_present(self):
        result = convert_pubmedqa_example(EXAMPLE_YES)
        assert set(result.keys()) == {"prompt", "context_raw", "context_chunks", "gold_chunk_ids", "solution"}

    def test_context_chunks_is_valid_json_dict(self):
        result = convert_pubmedqa_example(EXAMPLE_YES)
        chunks = json.loads(result["context_chunks"])
        assert isinstance(chunks, dict)
        assert len(chunks) == 4  # 4 IMRAD sections

    def test_chunk_ids_contain_pubid_and_label(self):
        result = convert_pubmedqa_example(EXAMPLE_YES)
        chunks = json.loads(result["context_chunks"])
        for cid in chunks:
            assert "12345678" in cid
        assert any("BACKGROUND" in cid for cid in chunks)
        assert any("CONCLUSIONS" in cid for cid in chunks)

    def test_solution_is_long_answer(self):
        result = convert_pubmedqa_example(EXAMPLE_YES)
        assert result["solution"] == LONG_ANSWER

    def test_context_raw_contains_all_sections(self):
        result = convert_pubmedqa_example(EXAMPLE_YES)
        chunks = json.loads(result["context_chunks"])
        for text in chunks.values():
            assert text[:20] in result["context_raw"]

    def test_prompt_contains_all_chunk_ids(self):
        result = convert_pubmedqa_example(EXAMPLE_YES)
        chunks = json.loads(result["context_chunks"])
        for cid in chunks:
            assert cid in result["prompt"]
        assert EXAMPLE_YES["question"] in result["prompt"]

    def test_chunk_ids_are_unique(self):
        result = convert_pubmedqa_example(EXAMPLE_YES)
        chunks = json.loads(result["context_chunks"])
        assert len(chunks) == len(set(chunks.keys()))


# ── Sufficiency mapping ───────────────────────────────────────────────────────

class TestSufficiencyMapping:
    def test_yes_decision_all_sections_are_gold(self):
        result = convert_pubmedqa_example(EXAMPLE_YES)
        chunks = json.loads(result["context_chunks"])
        gold_ids = json.loads(result["gold_chunk_ids"])
        assert set(gold_ids) == set(chunks.keys())

    def test_no_decision_all_sections_are_gold(self):
        result = convert_pubmedqa_example(EXAMPLE_NO)
        chunks = json.loads(result["context_chunks"])
        gold_ids = json.loads(result["gold_chunk_ids"])
        assert set(gold_ids) == set(chunks.keys())

    def test_maybe_decision_gold_ids_empty(self):
        result = convert_pubmedqa_example(EXAMPLE_MAYBE)
        gold_ids = json.loads(result["gold_chunk_ids"])
        assert gold_ids == []

    def test_maybe_solution_is_long_answer(self):
        result = convert_pubmedqa_example(EXAMPLE_MAYBE)
        assert result["solution"] == "Evidence is mixed; neither approach shows clear superiority."


# ── Duplicate label handling ──────────────────────────────────────────────────

class TestDuplicateLabels:
    def test_duplicate_section_labels_get_unique_ids(self):
        dup_example = _make_pubmedqa(
            pubid="99999999",
            question="Test duplicate labels?",
            labels=["RESULTS", "RESULTS"],  # same label twice
            sentences=[["First result."], ["Second result."]],
            long_answer="Results vary.",
            final_decision="maybe",
        )
        result = convert_pubmedqa_example(dup_example)
        chunks = json.loads(result["context_chunks"])
        assert len(chunks) == 2
        assert len(set(chunks.keys())) == 2  # unique ids


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_pubid_produces_same_shuffle(self):
        r1 = convert_pubmedqa_example(EXAMPLE_YES)
        r2 = convert_pubmedqa_example(EXAMPLE_YES)
        assert r1["prompt"] == r2["prompt"]


# ── Reward compatibility ──────────────────────────────────────────────────────

class TestRewardCompatibility:
    def setup_method(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from open_r1.grounded_rewards import chunk_routing_reward, quote_grounding_reward
        self.chunk_routing = chunk_routing_reward
        self.quote_grounding = quote_grounding_reward

    def test_verbatim_citation_from_abstract_scores_1(self):
        result = convert_pubmedqa_example(EXAMPLE_YES)
        gold_ids = json.loads(result["gold_chunk_ids"])
        chunks = json.loads(result["context_chunks"])
        gold_id = gold_ids[0]
        # Use exact text from the gold chunk
        exact_quote = chunks[gold_id][:50].strip()

        payload = json.dumps({
            "reasoning_path": "The results section contains direct evidence for BRCA1 risk.",
            "is_context_sufficient": True,
            "final_answer": "Yes, BRCA1 mutations increase cancer risk.",
            "extracted_quotes": [{"chunk_id": gold_id, "exact_quote": exact_quote}],
        })
        completions = [[{"role": "assistant", "content": payload}]]

        r_routing = self.chunk_routing(completions, [result["context_chunks"]], [result["gold_chunk_ids"]])
        r_grounding = self.quote_grounding(completions, context_raw=[result["context_raw"]])
        assert r_routing == [1.0]
        assert r_grounding == [1.0]

    def test_correct_abstention_for_maybe_scores_1(self):
        result = convert_pubmedqa_example(EXAMPLE_MAYBE)
        payload = json.dumps({
            "reasoning_path": "The context does not provide sufficient evidence to determine superiority.",
            "is_context_sufficient": False,
            "final_answer": "The provided context does not contain sufficient information to answer this question.",
            "extracted_quotes": [],
        })
        completions = [[{"role": "assistant", "content": payload}]]

        r_routing = self.chunk_routing(completions, [result["context_chunks"]], [result["gold_chunk_ids"]])
        assert r_routing == [1.0]

    def test_hallucinated_medical_fact_scores_0(self):
        """Reward = 0 if model cites a fact NOT in the abstract."""
        result = convert_pubmedqa_example(EXAMPLE_YES)
        gold_ids = json.loads(result["gold_chunk_ids"])

        payload = json.dumps({
            "reasoning_path": "Based on my knowledge of genetics...",
            "is_context_sufficient": True,
            "final_answer": "BRCA1 mutations triple the lifetime cancer risk.",
            "extracted_quotes": [
                # Fabricated quote — not in the abstract
                {"chunk_id": gold_ids[0], "exact_quote": "BRCA1 mutations triple the lifetime cancer risk"}
            ],
        })
        completions = [[{"role": "assistant", "content": payload}]]

        r_grounding = self.quote_grounding(completions, context_raw=[result["context_raw"]])
        assert r_grounding == [0.0]  # hallucinated medical fact → penalised
