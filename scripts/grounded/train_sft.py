"""Supervised fine-tuning baseline (LoRA) on reference responses built from gold answers.

    python scripts/grounded/train_sft.py --config configs/grounded/sft_qwen0.5b.yaml --seed 0

The reference response of an answerable question quotes the sentence that contains the
gold answer span and gives the gold answer; for an unanswerable question it abstains.
No model is involved in building the targets (see `open_r1.grounded.data`).

Outputs
-------
data/<run_name>/seed<seed>/                   LoRA adapter and trainer checkpoints (not committed)
results/sft/<run_name>/seed<seed>/run.json    config, seed, environment, runtime, final training metrics
results/sft/<run_name>/seed<seed>/train_log.jsonl

An interrupted run resumes from its last checkpoint with --resume.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from open_r1.grounded.config import load_config  # noqa: E402
from open_r1.grounded.runinfo import collect_run_info, write_json  # noqa: E402
from open_r1.grounded.training import (  # noqa: E402
    count_parameters,
    load_model_and_tokenizer,
    load_training_records,
    lora_config,
    make_jsonl_logger,
    sft_rows,
    trim_log,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in the output dir.")
    parser.add_argument("--checkpoint_root", default="data")
    parser.add_argument("--results_root", default="results/sft")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides.")
    args = parser.parse_args()

    from datasets import Dataset
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    config = load_config(args.config, args.set)
    run_name = config["run_name"]
    output_dir = Path(args.checkpoint_root) / run_name / f"seed{args.seed}"
    results_dir = Path(args.results_root) / run_name / f"seed{args.seed}"

    records = load_training_records(config["data"])
    dataset = Dataset.from_list(sft_rows(records))
    model, tokenizer = load_model_and_tokenizer(config["model"])

    log_path = results_dir / "train_log.jsonl"
    if not args.resume and log_path.exists():
        log_path.unlink()
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir=str(output_dir), seed=args.seed, **config["sft"]),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(config["lora"]),
        callbacks=[make_jsonl_logger(log_path)],
    )

    checkpoint = get_last_checkpoint(str(output_dir)) if args.resume and output_dir.is_dir() else None
    trim_log(log_path, checkpoint)
    start = time.time()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    runtime = time.time() - start

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    write_json(
        results_dir / "run.json",
        {
            "run_name": run_name,
            "seed": args.seed,
            "config": config,
            "n_training_examples": len(records),
            "n_answerable": sum(r["is_answerable"] for r in records),
            "parameters": count_parameters(trainer.model),
            "resumed_from": checkpoint,
            "train_runtime_seconds": round(runtime, 1),
            "global_step": trainer.state.global_step,
            "train_metrics": result.metrics,
            "adapter_dir": str(output_dir),
            "environment": collect_run_info(),
        },
    )
    print(f"Adapter saved to {output_dir}; logs in {results_dir}")


if __name__ == "__main__":
    main()
