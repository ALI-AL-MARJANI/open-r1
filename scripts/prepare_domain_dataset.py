"""
Prepare domain-specific Grounded-R1 datasets.

The grounded reward functions (grounded_format, quote_grounding, chunk_routing,
answer_faithfulness, reasoning_quality) are fully domain-agnostic — they operate
on substrings and JSON schema, not semantics. Only the dataset format and system
prompt vary per domain.

Supported domains:
  pubmedqa   — biomedical QA over PubMed abstracts (pqa_labeled: 1k expert,
                pqa_unlabeled: 61k; pqa_artificial: 211k silver)
               Source: qiaojin/PubMedQA on HuggingFace

Why PubMedQA is ideal for Grounded-R1:
  - "maybe" decisions map exactly to is_context_sufficient=False
  - Abstract sections (BACKGROUND/METHODS/RESULTS/CONCLUSIONS) become chunks
  - Hallucination from memorised medical knowledge is a real, high-stakes failure mode
  - Verbatim citation from abstracts is exactly what medical evidence synthesis requires

Output schema (same as v0/v1/v2 — fully compatible with existing rewards):
  prompt          str  — [CHUNK] abstract sections + clinical question
  context_raw     str  — all abstract text concatenated
  context_chunks  str  — JSON {chunk_id: section_text}
  gold_chunk_ids  str  — JSON [chunk_id, ...] (all sections for yes/no; [] for maybe)
  solution        str  — long_answer string

Usage:
  python scripts/prepare_domain_dataset.py \\
      --domain pubmedqa \\
      --config pqa_labeled \\
      --output_dir data/grounded-pubmedqa

  # Silver labels at scale (for RL training):
  python scripts/prepare_domain_dataset.py \\
      --domain pubmedqa \\
      --config pqa_artificial \\
      --output_dir data/grounded-pubmedqa-large

  # Push to Hub:
  python scripts/prepare_domain_dataset.py \\
      --domain pubmedqa --config pqa_labeled \\
      --output_dir data/grounded-pubmedqa \\
      --push_to_hub your-org/grounded-pubmedqa
"""

import argparse
import json
import random
import re
from typing import Optional

from datasets import DatasetDict, load_dataset


# ── Shared utilities ──────────────────────────────────────────────────────────

def _sanitise(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", text).strip("_")


def _format_chunks_block(chunks: dict[str, str], order: list[str]) -> str:
    return "\n\n".join(
        f'[CHUNK id="{cid}"]\n{chunks[cid].strip()}\n[/CHUNK]'
        for cid in order
    )


def _build_prompt(question: str, chunks: dict[str, str], order: list[str]) -> str:
    return f"CONTEXT:\n{_format_chunks_block(chunks, order)}\n\nQUESTION: {question.strip()}"


# ── PubMedQA ──────────────────────────────────────────────────────────────────

# Domain-specific system prompt addendum for biomedical contexts
BIOMEDICAL_SYSTEM_PROMPT_NOTE = (
    "You are operating in a biomedical research context. "
    "Cite ONLY from the provided abstract sections. "
    "Never apply memorised medical knowledge — your only source of truth is the provided context."
)

_PUBMEDQA_LABEL_ORDER = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]


def _pubmedqa_section_order(labels: list[str]) -> list[str]:
    """Return section labels in canonical IMRAD order, with unknowns appended."""
    known = [l for l in _PUBMEDQA_LABEL_ORDER if l in labels]
    unknown = [l for l in labels if l not in _PUBMEDQA_LABEL_ORDER]
    return known + unknown


def convert_pubmedqa_example(example: dict) -> dict:
    """
    Convert one PubMedQA example to grounded-r1 schema.

    Abstract sections → chunks:
      chunk_id: pubmed_{pubid}_{SECTION_LABEL}

    Sufficiency mapping:
      final_decision="yes"/"no" → is_context_sufficient=True (abstract answers the question)
      final_decision="maybe"    → is_context_sufficient=False (model should abstain)

    Gold chunks: ALL sections when answerable (any section may contain supporting evidence).
    This means chunk_routing rewards citing verbatim from the abstract vs hallucinating
    from memorised medical knowledge — which is exactly the high-stakes failure mode.
    """
    pubid = str(example["pubid"])
    labels: list[str] = example["context"]["labels"]
    sentences: list[list[str]] = example["context"]["sentences"]

    # Build one chunk per abstract section, handling rare duplicate labels
    chunks: dict[str, str] = {}
    ordered_ids: list[str] = []
    label_counts: dict[str, int] = {}
    for label, sents in zip(labels, sentences):
        label_up = label.upper()
        count = label_counts.get(label_up, 0)
        chunk_id = f"pubmed_{pubid}_{label_up}" if count == 0 else f"pubmed_{pubid}_{label_up}_{count}"
        label_counts[label_up] = count + 1
        chunks[chunk_id] = " ".join(sents).strip()
        ordered_ids.append(chunk_id)

    final_decision = str(example.get("final_decision", "maybe")).lower()
    is_answerable = final_decision in ("yes", "no")
    gold_chunk_ids = ordered_ids if is_answerable else []

    # Shuffle section order per example (prevents model exploiting CONCLUSIONS always last)
    rng = random.Random(hash(pubid) % (2 ** 31))
    shuffled_ids = ordered_ids.copy()
    rng.shuffle(shuffled_ids)

    return {
        "prompt": _build_prompt(example["question"], chunks, order=shuffled_ids),
        "context_raw": " ".join(chunks[cid] for cid in ordered_ids),
        "context_chunks": json.dumps(chunks),
        "gold_chunk_ids": json.dumps(gold_chunk_ids),
        "solution": example.get("long_answer", ""),
    }


def prepare_pubmedqa(config: str, output_dir: str, push_to_hub: Optional[str]) -> DatasetDict:
    """
    config options:
      pqa_labeled     — 1k expert labels (train: 500, test: 500) — use as eval
      pqa_unlabeled   — 61k, no final_decision labels — use with self-supervised rewards only
      pqa_artificial  — 211k silver labels — use for RL training at scale
    """
    print(f"Loading PubMedQA ({config})...")
    raw = load_dataset("qiaojin/PubMedQA", config)

    print("Converting to grounded-r1 format...")
    remove_cols = raw[list(raw.keys())[0]].column_names

    processed = raw.map(
        convert_pubmedqa_example,
        remove_columns=remove_cols,
        desc=f"PubMedQA/{config}",
    )

    _print_stats(processed, domain="pubmedqa", config=config)
    _save_and_push(processed, output_dir, push_to_hub)
    return processed


# ── Stats + save ──────────────────────────────────────────────────────────────

def _print_stats(processed: DatasetDict, domain: str, config: str):
    print(f"\n  Domain: {domain} / {config}")
    for split in processed:
        ds = processed[split]
        n = len(ds)
        n_answerable = sum(1 for g in ds["gold_chunk_ids"] if g != "[]")
        n_unanswerable = n - n_answerable

        # Avg chunks per example
        avg_chunks = sum(
            len(json.loads(c)) for c in ds["context_chunks"]
        ) / n if n > 0 else 0

        print(f"  {split}: {n:,} examples | "
              f"{n_answerable:,} answerable, {n_unanswerable:,} unanswerable | "
              f"avg {avg_chunks:.1f} chunks/example")

    # Print a sample
    first_split = list(processed.keys())[0]
    sample = processed[first_split][0]
    print(f"\n--- Sample prompt (first 600 chars) ---")
    print(sample["prompt"][:600])
    print(f"\ngold_chunk_ids : {sample['gold_chunk_ids'][:80]}")
    print(f"solution       : {sample['solution'][:120]!r}")


def _save_and_push(processed: DatasetDict, output_dir: str, push_to_hub: Optional[str]):
    print(f"\nSaving to {output_dir}...")
    processed.save_to_disk(output_dir)
    if push_to_hub:
        print(f"Pushing to Hub: {push_to_hub}...")
        processed.push_to_hub(push_to_hub)


# ── CLI ───────────────────────────────────────────────────────────────────────

DOMAIN_HANDLERS = {
    "pubmedqa": prepare_pubmedqa,
}

DOMAIN_CONFIGS = {
    "pubmedqa": ["pqa_labeled", "pqa_unlabeled", "pqa_artificial"],
}


def main():
    parser = argparse.ArgumentParser(description="Prepare domain-specific Grounded-R1 datasets")
    parser.add_argument(
        "--domain",
        required=True,
        choices=list(DOMAIN_HANDLERS.keys()),
        help="Source domain",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config (e.g. pqa_labeled for pubmedqa). Default: first available.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--push_to_hub", default=None)
    args = parser.parse_args()

    config = args.config or DOMAIN_CONFIGS[args.domain][0]
    print(f"Domain: {args.domain} | Config: {config}")

    DOMAIN_HANDLERS[args.domain](
        config=config,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
    )

    print(f"\nDone. To train on this domain:")
    print(f"  Update dataset_name in your recipe to: {args.output_dir}")
    print(f"  Then: python src/open_r1/grpo.py --config <recipe>.yaml")


if __name__ == "__main__":
    main()
