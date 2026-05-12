"""
Prepare Grounded-R1 training datasets.

Supported sources:
  squad_v2   — single-chunk, ~86k answerable + ~43k unanswerable  (v0 baseline)
  hotpotqa   — 10-chunk multi-passage (2 gold + 8 distractors)    (v1 multi-chunk)

All output datasets share the same schema so the same training recipe can be used:

  prompt          str  — formatted user message: [CHUNK] blocks + question
  context_raw     str  — concatenated text of all chunks (for quote_grounding_reward)
  context_chunks  str  — JSON {chunk_id: text}  (for chunk_routing_reward)
  gold_chunk_ids  str  — JSON [chunk_id, ...]   (for chunk_routing_reward; [] = unanswerable)
  solution        str  — gold answer string (empty = unanswerable)

Usage:
  # v0 — single-chunk SQuAD v2
  python scripts/prepare_grounded_dataset.py \\
      --dataset squad_v2 --output_dir data/grounded-squad-v2

  # v1 — multi-chunk HotpotQA (distractor setting)
  python scripts/prepare_grounded_dataset.py \\
      --dataset hotpotqa --output_dir data/grounded-hotpotqa

  # Push to Hub
  python scripts/prepare_grounded_dataset.py \\
      --dataset hotpotqa --output_dir data/grounded-hotpotqa \\
      --push_to_hub your-org/grounded-hotpotqa
"""

import argparse
import json
import random
import re
from typing import Optional

from datasets import DatasetDict, load_dataset


# ── Shared utilities ──────────────────────────────────────────────────────────

def _make_chunk_id(title: str, idx: int = 0) -> str:
    """Sanitise a title into a valid, collision-resistant chunk identifier."""
    sanitised = re.sub(r"[^a-zA-Z0-9_]", "_", title).strip("_")
    return f"{sanitised}_{idx}"


def _format_chunks_block(chunks: dict[str, str], order: Optional[list[str]] = None) -> str:
    """Render an ordered list of chunks as [CHUNK] blocks for the prompt."""
    ids = order if order is not None else list(chunks.keys())
    blocks = [f'[CHUNK id="{cid}"]\n{chunks[cid].strip()}\n[/CHUNK]' for cid in ids]
    return "\n\n".join(blocks)


def _build_prompt(question: str, chunks: dict[str, str], order: Optional[list[str]] = None) -> str:
    context_block = _format_chunks_block(chunks, order)
    return f"CONTEXT:\n{context_block}\n\nQUESTION: {question.strip()}"


# ── SQuAD v2 ──────────────────────────────────────────────────────────────────

def convert_squad_example(example: dict) -> dict:
    """Single-chunk SQuAD v2 example → grounded-r1 schema."""
    chunk_id = _make_chunk_id(example["title"])
    chunks = {chunk_id: example["context"]}
    is_answerable = bool(example["answers"]["text"])
    gold_chunk_ids = [chunk_id] if is_answerable else []

    return {
        "prompt": _build_prompt(example["question"], chunks),
        "context_raw": example["context"],
        "context_chunks": json.dumps(chunks),
        "gold_chunk_ids": json.dumps(gold_chunk_ids),
        "solution": example["answers"]["text"][0] if is_answerable else "",
    }


def prepare_squad_v2(output_dir: str, push_to_hub: Optional[str]) -> DatasetDict:
    print("Loading SQuAD v2...")
    raw = load_dataset("rajpurkar/squad_v2")
    print("Converting...")
    processed = raw.map(
        convert_squad_example,
        remove_columns=raw["train"].column_names,
        desc="SQuAD v2",
    )
    _save_and_push(processed, output_dir, push_to_hub)
    return processed


# ── HotpotQA ─────────────────────────────────────────────────────────────────

def convert_hotpotqa_example(example: dict) -> dict:
    """
    Multi-chunk HotpotQA (distractor setting) → grounded-r1 schema.

    Each example has 10 paragraphs: 2 supporting (gold) + 8 distractors.
    Chunks are shuffled per-example to prevent position bias during training.
    """
    titles: list[str] = example["context"]["title"]
    sentences: list[list[str]] = example["context"]["sentences"]

    # Build chunk dict, handling rare title collisions with a suffix counter
    chunks: dict[str, str] = {}
    seen: dict[str, int] = {}
    ordered_ids: list[str] = []
    for title, sents in zip(titles, sentences):
        base_id = _make_chunk_id(title)
        count = seen.get(base_id, 0)
        chunk_id = base_id if count == 0 else f"{base_id}_{count}"
        seen[base_id] = count + 1
        chunks[chunk_id] = " ".join(sents)
        ordered_ids.append(chunk_id)

    # Identify gold chunk ids (supporting facts reference titles, not sentences)
    gold_titles = set(example["supporting_facts"]["title"])
    gold_chunk_ids = [
        cid for cid, title in zip(ordered_ids, titles)
        if title in gold_titles
    ]

    # Shuffle chunk order — deterministic per example to keep eval reproducible
    shuffled_ids = ordered_ids.copy()
    rng = random.Random(hash(example["id"]) % (2**31))
    rng.shuffle(shuffled_ids)

    return {
        "prompt": _build_prompt(example["question"], chunks, order=shuffled_ids),
        "context_raw": " ".join(chunks[cid] for cid in ordered_ids),
        "context_chunks": json.dumps(chunks),
        "gold_chunk_ids": json.dumps(gold_chunk_ids),
        "solution": example["answer"],
    }


def prepare_hotpotqa(output_dir: str, push_to_hub: Optional[str]) -> DatasetDict:
    print("Loading HotpotQA (distractor)...")
    # distractor config: 10 paragraphs per question (2 gold + 8 distractors)
    raw = load_dataset("hotpot_qa", "distractor")
    print("Converting...")
    processed = raw.map(
        convert_hotpotqa_example,
        remove_columns=raw["train"].column_names,
        desc="HotpotQA",
    )
    _save_and_push(processed, output_dir, push_to_hub)
    return processed


# ── Shared save / push ────────────────────────────────────────────────────────

def _save_and_push(processed: DatasetDict, output_dir: str, push_to_hub: Optional[str]):
    for split in processed:
        n = len(processed[split])
        print(f"  {split}: {n:,} examples")

    sample = processed[list(processed.keys())[0]][0]
    print("\n--- Sample prompt (first 700 chars) ---")
    print(sample["prompt"][:700])
    print(f"\ncontext_raw snippet : {sample['context_raw'][:150]}")
    print(f"gold_chunk_ids      : {sample['gold_chunk_ids']}")
    print(f"solution            : {sample['solution']!r}")

    print(f"\nSaving to {output_dir}...")
    processed.save_to_disk(output_dir)

    if push_to_hub:
        print(f"Pushing to Hub: {push_to_hub}...")
        processed.push_to_hub(push_to_hub)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Grounded-R1 training datasets")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["squad_v2", "hotpotqa"],
        help="Source dataset to convert",
    )
    parser.add_argument("--output_dir", required=True, help="Local output directory")
    parser.add_argument("--push_to_hub", default=None, help="HuggingFace Hub repo id")
    args = parser.parse_args()

    if args.dataset == "squad_v2":
        prepare_squad_v2(output_dir=args.output_dir, push_to_hub=args.push_to_hub)
        print("\nNext step (v0 training):")
        print("  python src/open_r1/grpo.py --config recipes/Grounded-R1-Qwen-1.5B/grpo/config_v0.yaml")
    elif args.dataset == "hotpotqa":
        prepare_hotpotqa(output_dir=args.output_dir, push_to_hub=args.push_to_hub)
        print("\nNext step (v1 training):")
        print("  python src/open_r1/grpo.py --config recipes/Grounded-R1-Qwen-1.5B/grpo/config_v1.yaml")


if __name__ == "__main__":
    main()
