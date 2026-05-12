"""
Prepare a grounded-r1 training dataset from SQuAD v2.

SQuAD v2 is ideal for Grounded-R1 because it contains:
  - ~86k answerable questions   → is_context_sufficient=True examples
  - ~43k unanswerable questions → is_context_sufficient=False examples

Output dataset columns:
  prompt       — formatted user message: [CHUNK] context + question
  context_raw  — raw passage text used by quote_grounding_reward for substring matching
  solution     — first gold answer string (empty string if unanswerable)

Usage:
  python scripts/prepare_grounded_dataset.py \
    --output_dir data/grounded-squad-v2 \
    [--push_to_hub your-org/grounded-squad-v2]
"""

import argparse
import re

from datasets import DatasetDict, load_dataset

SYSTEM_PROMPT = """\
You are an expert information extraction system. Your task is to answer questions \
based STRICTLY on the provided context chunks.

CRITICAL RULES:
1. You MUST ONLY use information explicitly stated in the provided context.
2. Every claim in your final_answer MUST be directly supported by an exact extracted_quote.
3. Your "exact_quote" values MUST be verbatim substrings copied from the context — \
no paraphrasing, no omissions, no additions.
4. If the context does not contain enough information to answer the question, \
set "is_context_sufficient": false and leave "extracted_quotes" as an empty list [].

OUTPUT FORMAT — respond ONLY with this JSON schema (no markdown, no preamble):
{
  "reasoning_path": "<chain-of-thought: identify relevant chunks, assess sufficiency>",
  "is_context_sufficient": <true|false>,
  "final_answer": "<synthesised answer, or 'The provided context does not contain sufficient information to answer this question.' if is_context_sufficient is false>",
  "extracted_quotes": [
    {"chunk_id": "<source chunk id>", "exact_quote": "<verbatim text from the chunk>"}
  ]
}"""


def _make_chunk_id(title: str, idx: int = 0) -> str:
    """Sanitise a Wikipedia title into a valid chunk identifier."""
    sanitised = re.sub(r"[^a-zA-Z0-9_]", "_", title).strip("_")
    return f"{sanitised}_{idx}"


def _format_context(passage: str, chunk_id: str) -> str:
    return f'[CHUNK id="{chunk_id}"]\n{passage.strip()}\n[/CHUNK]'


def _build_user_message(question: str, passage: str, chunk_id: str) -> str:
    """Construct the full user message: formatted context + question."""
    context_block = _format_context(passage, chunk_id)
    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question.strip()}"
    )


def convert_squad_example(example: dict) -> dict:
    """Map a single SQuAD v2 example to the grounded-r1 training format."""
    chunk_id = _make_chunk_id(example["title"])
    prompt = _build_user_message(example["question"], example["context"], chunk_id)
    context_raw = example["context"]
    # answers["text"] is [] for unanswerable questions
    solution = example["answers"]["text"][0] if example["answers"]["text"] else ""
    return {
        "prompt": prompt,
        "context_raw": context_raw,
        "solution": solution,
    }


def prepare_dataset(push_to_hub: str | None = None, output_dir: str = "data/grounded-squad-v2") -> DatasetDict:
    print("Loading SQuAD v2...")
    raw = load_dataset("rajpurkar/squad_v2")

    print("Converting to grounded-r1 format...")
    processed = DatasetDict()
    for split in raw:
        processed[split] = raw[split].map(
            convert_squad_example,
            remove_columns=raw[split].column_names,
            desc=f"Processing {split}",
        )

    print(f"Train: {len(processed['train'])} examples")
    print(f"Validation: {len(processed['validation'])} examples")

    # Quick sanity check
    sample = processed["train"][0]
    print("\n--- Sample prompt (first 600 chars) ---")
    print(sample["prompt"][:600])
    print(f"\ncontext_raw (first 200 chars): {sample['context_raw'][:200]}")
    print(f"solution: {sample['solution']!r}")

    print(f"\nSaving to {output_dir}...")
    processed.save_to_disk(output_dir)

    if push_to_hub:
        print(f"Pushing to Hub: {push_to_hub}...")
        processed.push_to_hub(push_to_hub)

    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Grounded-R1 training data from SQuAD v2")
    parser.add_argument("--output_dir", default="data/grounded-squad-v2", help="Local output directory")
    parser.add_argument("--push_to_hub", default=None, help="HuggingFace Hub repo id to push the dataset")
    args = parser.parse_args()

    prepare_dataset(push_to_hub=args.push_to_hub, output_dir=args.output_dir)
    print("\nDone. To launch training:")
    print(f"  python src/open_r1/grpo.py --config recipes/Grounded-R1-Qwen-1.5B/grpo/config_v0.yaml")
