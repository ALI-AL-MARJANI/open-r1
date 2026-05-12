"""
v2 hard negative injection: add same-article distractors to SQuAD v2.

Motivation
----------
After v0/v1 training, the model learns to cite verbatim text. All 8 GRPO rollouts
eventually score ~1.0 → zero variance → no policy gradient. We need harder examples
that remain challenging for a well-trained model.

Key insight: SQuAD stores multiple paragraphs per Wikipedia article (title). We use
other paragraphs from the SAME article as hard-negative distractors. They share
vocabulary, entities, and topic with the gold passage — but don't contain the answer.
The model can't rely on topic matching; it must actually read each passage.

This creates a multi-passage retrieval + extraction task from SQuAD's single-passage
annotation, without any external LLM or human labelling.

Distractor selection strategy:
  1. Same-article passages (same Wikipedia title) — topically hard, preferred
  2. Adjacent-title passages (alphabetically nearby) — for titles with only 1 passage
  3. Random fallback — last resort

Output schema (same as v1, fully compatible with existing rewards):
  prompt          str   rebuilt with gold + distractors, shuffled
  context_raw     str   concatenation of all chunks (for quote_grounding_reward)
  context_chunks  str   JSON {chunk_id: text}  (for chunk_routing_reward)
  gold_chunk_ids  str   JSON [gold_id]  (unchanged semantics)
  solution        str   gold answer string

Usage:
  python scripts/inject_hard_negatives.py \\
      --n_negatives 3 \\
      --output_dir data/grounded-squad-v2-hard

  # Then train with v2 recipe:
  sbatch --nodes=2 slurm/train.slurm \\
      --model Grounded-R1-Qwen-1.5B --task grpo --config v2 --accelerator zero3
"""

import argparse
import json
import random
import re
from collections import defaultdict
from typing import Optional

from datasets import DatasetDict, load_dataset
from tqdm import tqdm


# ── Shared utilities (mirrors prepare_grounded_dataset.py) ───────────────────

def _make_chunk_id(title: str, suffix: str = "0") -> str:
    sanitised = re.sub(r"[^a-zA-Z0-9_]", "_", title).strip("_")
    return f"{sanitised}_{suffix}"


def _format_chunks_block(chunks: dict[str, str], order: list[str]) -> str:
    return "\n\n".join(
        f'[CHUNK id="{cid}"]\n{chunks[cid].strip()}\n[/CHUNK]'
        for cid in order
    )


def _build_prompt(question: str, chunks: dict[str, str], order: list[str]) -> str:
    return f"CONTEXT:\n{_format_chunks_block(chunks, order)}\n\nQUESTION: {question.strip()}"


# ── Title index ───────────────────────────────────────────────────────────────

class TitleIndex:
    """
    Groups SQuAD passages by Wikipedia article title.

    For articles with a single passage, falls back to alphabetically adjacent
    titles (likely related topics) then to random sampling.
    """

    def __init__(self, dataset):
        self._by_title: dict[str, list[dict]] = defaultdict(list)
        self._titles_sorted: list[str] = []
        self._all_passages: list[dict] = []

        seen_contexts: set[str] = set()
        for example in dataset:
            key = (example["title"], example["context"])
            if key in seen_contexts:
                continue
            seen_contexts.add(key)
            entry = {"title": example["title"], "context": example["context"]}
            self._by_title[example["title"]].append(entry)
            self._all_passages.append(entry)

        self._titles_sorted = sorted(self._by_title.keys())
        print(f"  Title index: {len(self._titles_sorted)} unique titles, "
              f"{len(self._all_passages)} unique passages")

    def sample_distractors(
        self,
        title: str,
        gold_context: str,
        n: int,
        rng: random.Random,
    ) -> list[dict]:
        """Return up to n distractor passages for the given title, excluding gold."""
        candidates = [
            p for p in self._by_title[title]
            if p["context"] != gold_context
        ]

        # Fall back to adjacent titles if same-article pool is exhausted
        if len(candidates) < n:
            title_idx = self._titles_sorted.index(title) if title in self._titles_sorted else 0
            for delta in range(1, 10):
                for direction in (-1, 1):
                    adj_idx = title_idx + direction * delta
                    if 0 <= adj_idx < len(self._titles_sorted):
                        adj_title = self._titles_sorted[adj_idx]
                        for p in self._by_title[adj_title]:
                            if p["context"] != gold_context and p not in candidates:
                                candidates.append(p)
                if len(candidates) >= n:
                    break

        # Final fallback: random from full pool
        if len(candidates) < n:
            pool = [p for p in self._all_passages if p["context"] != gold_context]
            candidates += rng.sample(pool, min(n - len(candidates), len(pool)))

        return rng.sample(candidates, min(n, len(candidates)))


# ── Example augmentation ──────────────────────────────────────────────────────

def augment_example(
    example: dict,
    index: TitleIndex,
    n_negatives: int,
) -> dict:
    """
    Augment one SQuAD example with same-article hard-negative passages.

    Gold chunk id: {sanitised_title}_0
    Distractor ids: {sanitised_title}_neg1, _neg2, ...

    These ids deliberately signal their role (neg) so the model cannot
    exploit id patterns — the shuffled order in the prompt prevents that anyway.
    """
    rng = random.Random(hash(example["id"]) % (2 ** 31))

    gold_chunk_id = _make_chunk_id(example["title"], suffix="0")
    distractors = index.sample_distractors(
        example["title"], example["context"], n=n_negatives, rng=rng
    )

    # Build chunks dict: gold first (then distractors)
    chunks: dict[str, str] = {gold_chunk_id: example["context"]}
    for i, dist in enumerate(distractors, start=1):
        dist_id = _make_chunk_id(dist["title"], suffix=f"neg{i}")
        # Ensure uniqueness in the rare case two distractors share a sanitised title
        while dist_id in chunks:
            dist_id += "_x"
        chunks[dist_id] = dist["context"]

    is_answerable = bool(example["answers"]["text"])
    gold_chunk_ids = [gold_chunk_id] if is_answerable else []

    # Shuffle to prevent position bias
    shuffled_ids = list(chunks.keys())
    rng.shuffle(shuffled_ids)

    return {
        "prompt": _build_prompt(example["question"], chunks, order=shuffled_ids),
        "context_raw": " ".join(chunks[cid] for cid in chunks),
        "context_chunks": json.dumps(chunks),
        "gold_chunk_ids": json.dumps(gold_chunk_ids),
        "solution": example["answers"]["text"][0] if is_answerable else "",
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(processed_split) -> dict:
    """Compute per-split stats for sanity checking."""
    n_chunks_per_example = []
    for example in processed_split:
        chunks = json.loads(example["context_chunks"])
        n_chunks_per_example.append(len(chunks))
    return {
        "total": len(processed_split),
        "avg_chunks": sum(n_chunks_per_example) / len(n_chunks_per_example),
        "min_chunks": min(n_chunks_per_example),
        "max_chunks": max(n_chunks_per_example),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inject hard-negative distractors into SQuAD v2")
    parser.add_argument(
        "--n_negatives", type=int, default=3,
        help="Number of same-article distractor passages per example (default: 3)",
    )
    parser.add_argument(
        "--output_dir", default="data/grounded-squad-v2-hard",
        help="Output directory for the augmented dataset",
    )
    parser.add_argument("--push_to_hub", default=None)
    args = parser.parse_args()

    print("Loading SQuAD v2...")
    raw = load_dataset("rajpurkar/squad_v2")

    print("Building title index from train split...")
    # Build index from train only to avoid train/val contamination
    train_index = TitleIndex(raw["train"])
    val_index = TitleIndex(raw["validation"])

    print(f"\nAugmenting with {args.n_negatives} same-article hard-negative distractors per example...")
    processed = DatasetDict()
    for split, index in [("train", train_index), ("validation", val_index)]:
        split_data = raw[split]
        augmented = split_data.map(
            lambda ex: augment_example(ex, index, args.n_negatives),
            remove_columns=split_data.column_names,
            desc=f"Augmenting {split}",
        )
        processed[split] = augmented
        stats = compute_stats(augmented)
        print(f"  {split}: {stats['total']:,} examples, "
              f"avg {stats['avg_chunks']:.1f} chunks/example "
              f"(min {stats['min_chunks']}, max {stats['max_chunks']})")

    # Sanity check on a sample
    sample = processed["train"][0]
    chunks = json.loads(sample["context_chunks"])
    gold_ids = json.loads(sample["gold_chunk_ids"])
    print(f"\n--- Sample (first 700 chars of prompt) ---")
    print(sample["prompt"][:700])
    print(f"\nChunks: {list(chunks.keys())}")
    print(f"Gold chunk ids: {gold_ids}")
    print(f"Solution: {sample['solution']!r}")

    print(f"\nSaving to {args.output_dir}...")
    processed.save_to_disk(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing to Hub: {args.push_to_hub}...")
        processed.push_to_hub(args.push_to_hub)

    print("\nDone. To train v2:")
    print(f"  python src/open_r1/grpo.py --config recipes/Grounded-R1-Qwen-1.5B/grpo/config_v2.yaml")


if __name__ == "__main__":
    main()
