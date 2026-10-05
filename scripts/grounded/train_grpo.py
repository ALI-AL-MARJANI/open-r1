"""GRPO training with the grounded reward suite (single GPU, LoRA).

    python scripts/grounded/train_grpo.py --config configs/grounded/grpo_qwen0.5b.yaml --seed 0

Outputs
-------
data/<run_name>/seed<seed>/                    LoRA adapter and trainer checkpoints (not committed)
results/grpo/<run_name>/seed<seed>/run.json    config, seed, environment, runtime, final training metrics
results/grpo/<run_name>/seed<seed>/train_log.jsonl   one line per logging step (rewards, KL, lengths, ...)

An interrupted run resumes from its last checkpoint with --resume.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from open_r1.grounded.config import load_config, reward_spec  # noqa: E402
from open_r1.grounded.rewards import build_reward_functions  # noqa: E402
from open_r1.grounded.runinfo import collect_run_info, write_json  # noqa: E402
from open_r1.grounded.training import (  # noqa: E402
    count_parameters,
    grpo_rows,
    load_model_and_tokenizer,
    load_training_records,
    lora_config,
    make_jsonl_logger,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Training seed (LoRA init, sampling, data order).")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in the output dir.")
    parser.add_argument("--checkpoint_root", default="data", help="Where adapters and checkpoints are written.")
    parser.add_argument("--results_root", default="results/grpo")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides.")
    args = parser.parse_args()

    from datasets import Dataset
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    config = load_config(args.config, args.set)
    run_name = config["run_name"]
    output_dir = Path(args.checkpoint_root) / run_name / f"seed{args.seed}"
    results_dir = Path(args.results_root) / run_name / f"seed{args.seed}"

    weights, settings = reward_spec(config)
    functions = build_reward_functions(settings)
    unknown = sorted(set(weights) - set(functions))
    if unknown:
        raise ValueError(f"Unknown rewards in config: {unknown}. Available: {sorted(functions)}")

    records = load_training_records(config["data"])
    dataset = Dataset.from_list(grpo_rows(records))
    model, tokenizer = load_model_and_tokenizer(config["model"])

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        seed=args.seed,
        reward_weights=list(weights.values()),
        **config["grpo"],
    )
    log_path = results_dir / "train_log.jsonl"
    if not args.resume and log_path.exists():
        log_path.unlink()
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[functions[name] for name in weights],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(config["lora"]),
        callbacks=[make_jsonl_logger(log_path)],
    )

    checkpoint = get_last_checkpoint(str(output_dir)) if args.resume and output_dir.is_dir() else None
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
