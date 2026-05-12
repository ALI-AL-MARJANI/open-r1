"""Unit tests for the hard-negative injection pipeline (v2)."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from inject_hard_negatives import TitleIndex, augment_example


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_example(id_, title, context, question, answers):
    return {
        "id": id_,
        "title": title,
        "context": context,
        "question": question,
        "answers": {"text": answers, "answer_start": [0] * len(answers)},
    }


PARIS_CAPITAL = "Paris is the capital of France and a major European city."
PARIS_HISTORY = "Paris was founded by the Parisii tribe around 250 BC on the Île de la Cité."
PARIS_CULTURE = "Paris is home to the Louvre, one of the world's largest art museums."
BERLIN_CAPITAL = "Berlin is the capital of Germany, located in central Europe."
LONDON_CAPITAL = "London is the capital of the United Kingdom."

EXAMPLES = [
    _make_example("a1", "France", PARIS_CAPITAL, "What is the capital of France?", ["Paris"]),
    _make_example("a2", "France", PARIS_HISTORY, "When was Paris founded?", ["around 250 BC"]),
    _make_example("a3", "France", PARIS_CULTURE, "What museum is in Paris?", ["the Louvre"]),
    _make_example("b1", "Germany", BERLIN_CAPITAL, "What is the capital of Germany?", ["Berlin"]),
    _make_example("c1", "UK", LONDON_CAPITAL, "What is the capital of the UK?", ["London"]),
    # Unanswerable
    _make_example("a4", "France", PARIS_CAPITAL, "What is the capital of Mars?", []),
]


# ── TitleIndex tests ──────────────────────────────────────────────────────────

class TestTitleIndex:
    def setup_method(self):
        self.index = TitleIndex(EXAMPLES)

    def test_title_grouping(self):
        assert len(self.index._by_title["France"]) == 3
        assert len(self.index._by_title["Germany"]) == 1
        assert len(self.index._by_title["UK"]) == 1

    def test_deduplication(self):
        """Duplicate (title, context) pairs should be ignored."""
        dup = EXAMPLES + [_make_example("a1_dup", "France", PARIS_CAPITAL, "Q?", ["A"])]
        idx = TitleIndex(dup)
        assert len(idx._by_title["France"]) == 3

    def test_same_article_distractors_preferred(self):
        rng = random.Random(42)
        distractors = self.index.sample_distractors("France", PARIS_CAPITAL, n=2, rng=rng)
        texts = [d["context"] for d in distractors]
        assert PARIS_CAPITAL not in texts          # gold excluded
        assert all(d["title"] == "France" for d in distractors)  # same article preferred

    def test_fallback_when_same_title_exhausted(self):
        """Germany has only 1 passage; requesting 3 distractors should fall back."""
        rng = random.Random(42)
        distractors = self.index.sample_distractors("Germany", BERLIN_CAPITAL, n=3, rng=rng)
        texts = [d["context"] for d in distractors]
        assert BERLIN_CAPITAL not in texts
        assert len(distractors) == 3

    def test_n_greater_than_pool_returns_all(self):
        rng = random.Random(42)
        distractors = self.index.sample_distractors("France", PARIS_CAPITAL, n=100, rng=rng)
        # Only 2 other France passages exist + fallbacks from adjacent titles
        assert len(distractors) > 0
        assert all(d["context"] != PARIS_CAPITAL for d in distractors)


# ── augment_example tests ─────────────────────────────────────────────────────

class TestAugmentExample:
    def setup_method(self):
        self.index = TitleIndex(EXAMPLES)

    def test_output_columns(self):
        result = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        assert set(result.keys()) == {"prompt", "context_raw", "context_chunks", "gold_chunk_ids", "solution"}

    def test_gold_chunk_in_context_chunks(self):
        result = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        chunks = json.loads(result["context_chunks"])
        gold_ids = json.loads(result["gold_chunk_ids"])
        assert len(gold_ids) == 1
        gold_id = gold_ids[0]
        assert gold_id in chunks
        assert chunks[gold_id] == PARIS_CAPITAL

    def test_correct_number_of_chunks(self):
        result = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        chunks = json.loads(result["context_chunks"])
        assert len(chunks) == 3   # 1 gold + 2 distractors

    def test_gold_excluded_from_distractors(self):
        result = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        chunks = json.loads(result["context_chunks"])
        gold_ids = json.loads(result["gold_chunk_ids"])
        gold_text = chunks[gold_ids[0]]
        distractor_texts = [v for k, v in chunks.items() if k not in gold_ids]
        assert gold_text not in distractor_texts

    def test_context_raw_contains_all_chunks(self):
        result = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        chunks = json.loads(result["context_chunks"])
        for text in chunks.values():
            # Each chunk text should appear in context_raw
            key_phrase = text[:30]
            assert key_phrase in result["context_raw"]

    def test_prompt_contains_all_chunk_ids(self):
        result = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        chunks = json.loads(result["context_chunks"])
        for chunk_id in chunks:
            assert chunk_id in result["prompt"]
        assert "QUESTION:" in result["prompt"]
        assert EXAMPLES[0]["question"] in result["prompt"]

    def test_unanswerable_has_empty_gold_ids(self):
        unanswerable = EXAMPLES[-1]  # "What is the capital of Mars?"
        result = augment_example(unanswerable, self.index, n_negatives=2)
        gold_ids = json.loads(result["gold_chunk_ids"])
        assert gold_ids == []
        assert result["solution"] == ""

    def test_answerable_has_correct_solution(self):
        result = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        assert result["solution"] == "Paris"

    def test_deterministic_given_same_id(self):
        """Same example id → same augmentation (deterministic shuffle)."""
        r1 = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        r2 = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        assert r1["prompt"] == r2["prompt"]
        assert r1["context_chunks"] == r2["context_chunks"]

    def test_chunk_id_uniqueness(self):
        """All chunk ids within one example must be unique."""
        result = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        chunks = json.loads(result["context_chunks"])
        assert len(chunks) == len(set(chunks.keys()))

    def test_no_distractor_in_gold_ids(self):
        """Distractor chunk ids must NOT appear in gold_chunk_ids."""
        result = augment_example(EXAMPLES[0], self.index, n_negatives=2)
        chunks = json.loads(result["context_chunks"])
        gold_ids = set(json.loads(result["gold_chunk_ids"]))
        distractor_ids = set(chunks.keys()) - gold_ids
        assert len(distractor_ids) >= 1
        for dist_id in distractor_ids:
            assert dist_id not in gold_ids


# ── Integration: reward compatibility ────────────────────────────────────────

class TestRewardCompatibility:
    """Verify that augmented examples work correctly with chunk_routing_reward."""

    def setup_method(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from open_r1.grounded_rewards import chunk_routing_reward, quote_grounding_reward
        self.chunk_routing = chunk_routing_reward
        self.quote_grounding = quote_grounding_reward
        self.index = TitleIndex(EXAMPLES)

    def test_perfect_response_scores_1_on_augmented(self):
        example = EXAMPLES[0]
        aug = augment_example(example, self.index, n_negatives=2)
        gold_id = json.loads(aug["gold_chunk_ids"])[0]

        # Simulate perfect model output
        payload = json.dumps({
            "reasoning_path": "The first chunk contains the answer about the capital of France.",
            "is_context_sufficient": True,
            "final_answer": "Paris",
            "extracted_quotes": [{"chunk_id": gold_id, "exact_quote": "Paris is the capital of France"}],
        })
        completions = [[{"role": "assistant", "content": payload}]]

        r_routing = self.chunk_routing(completions, [aug["context_chunks"]], [aug["gold_chunk_ids"]])
        r_grounding = self.quote_grounding(completions, context_raw=[aug["context_raw"]])
        assert r_routing == [1.0]
        assert r_grounding == [1.0]

    def test_distractor_citation_scores_0_routing(self):
        example = EXAMPLES[0]
        aug = augment_example(example, self.index, n_negatives=2)
        chunks = json.loads(aug["context_chunks"])
        gold_ids = set(json.loads(aug["gold_chunk_ids"]))
        distractor_ids = [cid for cid in chunks if cid not in gold_ids]
        if not distractor_ids:
            return  # skip if no distractors (shouldn't happen with n=2)

        dist_id = distractor_ids[0]
        dist_text = chunks[dist_id]
        # Use first 40 chars of distractor text as quote
        dist_quote = dist_text[:40].strip()

        payload = json.dumps({
            "reasoning_path": "I found relevant info in a chunk.",
            "is_context_sufficient": True,
            "final_answer": "Some answer.",
            "extracted_quotes": [{"chunk_id": dist_id, "exact_quote": dist_quote}],
        })
        completions = [[{"role": "assistant", "content": payload}]]

        r_routing = self.chunk_routing(completions, [aug["context_chunks"]], [aug["gold_chunk_ids"]])
        # Quote is grounded (exists in context_raw) but from wrong chunk → routing = 0
        r_grounding = self.quote_grounding(completions, context_raw=[aug["context_raw"]])
        assert r_routing == [0.0]
        assert r_grounding == [1.0]   # grounded, but wrong chunk
