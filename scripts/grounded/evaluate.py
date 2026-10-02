"""Evaluate a model on the grounded QA protocol and write results to results/eval/.

Examples
--------
Zero-shot base model:
    python scripts/grounded/evaluate.py --run_name zero_shot_qwen0.5b --model Qwen/Qwen2.5-0.5B-Instruct

Few-shot (3 demonstrations from the SQuAD v2 training split):
    python scripts/grounded/evaluate.py --run_name few_shot_qwen0.5b --model Qwen/Qwen2.5-0.5B-Instruct --few_shot

LoRA adapter produced by train_sft.py or train_grpo.py:
    python scripts/grounded/evaluate.py --run_name grpo_qwen0.5b_seed0 \
        --model Qwen/Qwen2.5-0.5B-Instruct --adapter data/grpo_qwen0.5b/seed0

Model served by a local OpenAI-compatible endpoint (vLLM or Ollama):
    python scripts/grounded/evaluate.py --run_name zero_shot_qwen7b_ollama \
        --backend openai --base_url http://localhost:11434/v1 --model qwen2.5:7b

For each dataset this writes
    results/eval/<run_name>/<dataset>.json               metrics with 95% bootstrap CIs, config, environment
    results/eval/<run_name>/<dataset>.predictions.jsonl  one line per example: raw output and per-example scores
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from open_r1.grounded.config import load_config, reward_spec  # noqa: E402
from open_r1.grounded.data import load_records, load_squad_v2  # noqa: E402
from open_r1.grounded.generation import HFGenerator, OpenAICompatibleGenerator  # noqa: E402
from open_r1.grounded.metrics import bootstrap_ci, score_example  # noqa: E402
from open_r1.grounded.prompts import build_messages  # noqa: E402
from open_r1.grounded.runinfo import collect_run_info, write_json  # noqa: E402


def select_demonstrations(spec: dict) -> list[tuple[str, str]]:
    """Fixed few-shot demonstrations from the SQuAD v2 training split.

    Short contexts are preferred to keep the prompt small. The selection depends only
    on the `few_shot` section of the config, so every few-shot run uses the same ones.
    """
    pool = load_squad_v2("train", n=2000, seed=spec["seed"], n_distractors=spec["n_distractors"])
    max_words = spec["max_context_words"] * (spec["n_distractors"] + 1)
    short = [r for r in pool if len(r["user_message"].split()) <= max_words]
    answerable = [r for r in short if r["is_answerable"]][: spec["n_answerable"]]
    unanswerable = [r for r in short if not r["is_answerable"]][: spec["n_unanswerable"]]
    if len(answerable) < spec["n_answerable"] or len(unanswerable) < spec["n_unanswerable"]:
        raise RuntimeError("Not enough short training examples for the requested demonstrations.")
    # Interleave so the abstention example is not last.
    demos = answerable[:1] + unanswerable + answerable[1:]
    return [(r["user_message"], r["target"]) for r in demos]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/grounded/eval.yaml")
    parser.add_argument("--run_name", required=True, help="Sub-directory of results/eval/ for this run.")
    parser.add_argument("--model", required=True, help="Hub id or local path (hf), or served model name (openai).")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory (hf backend only).")
    parser.add_argument("--backend", choices=("hf", "openai"), default="hf")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None, help="Subset of the datasets in the config.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate fewer examples (smoke tests only).")
    parser.add_argument("--output_dir", default="results/eval")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides.")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    weights, settings = reward_spec(config)
    names = args.datasets or list(config["datasets"])
    generation = config["generation"]

    if args.backend == "openai":
        if not args.base_url:
            parser.error("--base_url is required with --backend openai")
        generator = OpenAICompatibleGenerator(args.base_url, args.model, generation["max_new_tokens"])
    else:
        generator = HFGenerator(
            args.model, args.adapter, generation["batch_size"], generation["max_new_tokens"], generation["dtype"]
        )

    demonstrations = select_demonstrations(config["few_shot"]) if args.few_shot else None
    run_dir = Path(args.output_dir) / args.run_name

    for name in names:
        spec = config["datasets"][name]
        n_examples = args.limit or spec.get("n_examples")
        records = load_records(
            spec["dataset"],
            spec["split"],
            n_examples,
            config["seed"],
            n_distractors=spec.get("n_distractors", 0),
            answerable_fraction=spec.get("answerable_fraction"),
        )
        conversations = [build_messages(r["user_message"], few_shot=demonstrations) for r in records]

        start = time.time()
        outputs = generator.generate(conversations)
        seconds = time.time() - start

        rows = [score_example(record, output, weights, settings) for record, output in zip(records, outputs)]
        metrics = bootstrap_ci(rows, n_resamples=config["bootstrap_resamples"], seed=config["seed"])

        write_json(
            run_dir / f"{name}.json",
            {
                "run_name": args.run_name,
                "dataset": name,
                "dataset_spec": {**spec, "n_examples": len(records), "subset_seed": config["seed"]},
                "is_smoke_test": args.limit is not None,
                "few_shot": bool(args.few_shot),
                "n_demonstrations": len(demonstrations or []),
                "generator": generator.description,
                "metrics": metrics,
                "confidence_interval": "95% percentile bootstrap over examples",
                "bootstrap_resamples": config["bootstrap_resamples"],
                "generation_seconds": round(seconds, 1),
                "seconds_per_example": round(seconds / max(len(records), 1), 3),
                "config": config,
                "environment": collect_run_info(),
            },
        )
        with open(run_dir / f"{name}.predictions.jsonl", "w", encoding="utf-8") as handle:
            for record, output, row in zip(records, outputs, rows):
                line = {"id": record["id"], "output": output, "scores": row}
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")

        shown = ("format_rate", "f1", "has_ans_f1", "no_ans_em", "unverified_quote_rate", "chunk_f1", "sp_f1")
        summary = ", ".join(
            f"{key}={metrics[key]['value']:.3f}" for key in shown if metrics.get(key, {}).get("value") is not None
        )
        print(f"[{args.run_name}/{name}] n={len(records)} {summary} ({seconds:.0f}s)")


if __name__ == "__main__":
    main()
