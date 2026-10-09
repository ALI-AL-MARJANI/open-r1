"""Helpers shared by the SFT and GRPO training scripts (model loading, LoRA, logging)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .data import load_records, select_subset
from .prompts import build_messages

LABEL_COLUMNS = ("context_chunks", "gold_answers", "gold_chunk_ids", "is_answerable")


def load_training_records(spec: dict[str, Any]) -> list[dict]:
    """Training pool described by the `data` section of a config.

    Examples whose prompt exceeds `max_prompt_words` are excluded to bound memory use.
    Twice the requested number of candidates is drawn so that `n_examples` remain
    after this filter, with the requested answerable fraction.
    """
    limit = spec.get("max_prompt_words")
    candidates = load_records(
        spec["dataset"],
        spec["split"],
        spec["n_examples"] * (2 if limit else 1),
        spec["seed"],
        n_distractors=spec.get("n_distractors", 0),
        answerable_fraction=spec.get("answerable_fraction"),
    )
    if limit:
        candidates = [r for r in candidates if len(r["user_message"].split()) <= limit]
    records = select_subset(candidates, spec["n_examples"], spec["seed"], spec.get("answerable_fraction"))
    if len(records) < spec["n_examples"]:
        raise ValueError(f"Only {len(records)} of {spec['n_examples']} examples fit max_prompt_words={limit}.")
    return sorted(records, key=lambda r: r["id"])


def grpo_rows(records: list[dict]) -> list[dict]:
    """Rows for `trl.GRPOTrainer`: a chat `prompt` plus the label columns read by the rewards."""
    return [{"prompt": build_messages(r["user_message"]), **{c: r[c] for c in LABEL_COLUMNS}} for r in records]


def sft_rows(records: list[dict]) -> list[dict]:
    """Rows for `trl.SFTTrainer` in prompt/completion form (loss on the completion only)."""
    return [
        {
            "prompt": build_messages(r["user_message"]),
            "completion": [{"role": "assistant", "content": r["target"]}],
        }
        for r in records
    ]


def load_model_and_tokenizer(spec: dict[str, Any]):
    """Load the base model and its tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec["name"], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["name"], dtype=getattr(torch, spec.get("dtype", "float32")))
    return model, tokenizer


def lora_config(spec: dict[str, Any]):
    from peft import LoraConfig

    return LoraConfig(
        task_type="CAUSAL_LM",
        r=spec["r"],
        lora_alpha=spec["alpha"],
        lora_dropout=spec.get("dropout", 0.0),
        target_modules=spec["target_modules"],
        bias="none",
    )


def make_jsonl_logger(path: str | Path):
    """Trainer callback that appends every logged metrics dict to a JSONL file.

    The file is the source of the training curves in `results/`; it is written
    incrementally so that an interrupted run still leaves its log.
    """
    from transformers import TrainerCallback

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class JsonlLogger(TrainerCallback):
        def __init__(self) -> None:
            self.start = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not state.is_world_process_zero:
                return
            entry = {"step": state.global_step, "elapsed_seconds": round(time.time() - self.start, 1), **logs}
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=float) + "\n")

    return JsonlLogger()


def trim_log(path: str | Path, checkpoint: str | Path | None) -> None:
    """Drop the log entries written after `checkpoint` (a `checkpoint-<step>` directory).

    A resumed run repeats the steps between its last checkpoint and the interruption;
    without this the JSONL log would contain them twice.
    """
    path = Path(path)
    if checkpoint is None or not path.exists():
        return
    last_step = int(Path(checkpoint).name.rsplit("-", 1)[1])
    with open(path, encoding="utf-8") as handle:
        kept = [line for line in handle if line.strip() and json.loads(line)["step"] <= last_step]
    path.write_text("".join(kept), encoding="utf-8")


def count_parameters(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
