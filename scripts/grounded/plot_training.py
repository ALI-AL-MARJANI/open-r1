"""Plot training curves from the JSONL logs written by train_grpo.py.

    python scripts/grounded/plot_training.py --run grpo_qwen0.5b

Reads results/grpo/<run>/seed*/train_log.jsonl and writes
results/grpo/<run>/training_curves.png: one panel per reward (mean over the sampled
completions of a step), plus abstention rate, completion length and KL when logged.
One line per seed; curves are smoothed with a centred rolling mean (`--window` steps).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PANELS = [
    ("rewards/answer_correctness/mean", "answer_correctness"),
    ("rewards/quote_grounding/mean", "quote_grounding"),
    ("rewards/chunk_routing/mean", "chunk_routing"),
    ("rewards/answer_faithfulness/mean", "answer_faithfulness"),
    ("rewards/format/mean", "format"),
    ("rewards/abstention_rate/mean", "abstention rate (monitor)"),
    ("completions/mean_length", "completion length (tokens)"),
    ("completions/clipped_ratio", "truncated completions"),
    ("kl", "KL to initial policy"),
]


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    half = window // 2
    return [
        sum(values[max(0, i - half) : i + half + 1]) / len(values[max(0, i - half) : i + half + 1])
        for i in range(len(values))
    ]


def _read_log(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run name under --results_root.")
    parser.add_argument("--results_root", default="results/grpo")
    parser.add_argument("--window", type=int, default=5, help="Rolling-mean window, in logging steps.")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(args.results_root) / args.run
    logs = {path.parent.name: _read_log(path) for path in sorted(run_dir.glob("seed*/train_log.jsonl"))}
    if not logs:
        raise SystemExit(f"No train_log.jsonl found under {run_dir}")

    panels = [(key, title) for key, title in PANELS if any(key in entry for log in logs.values() for entry in log)]
    n_columns = 3
    n_rows = -(-len(panels) // n_columns)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(4.6 * n_columns, 3.0 * n_rows), squeeze=False)

    for axis, (key, title) in zip(axes.flat, panels):
        for seed, log in logs.items():
            points = [(entry["step"], entry[key]) for entry in log if key in entry]
            if points:
                steps, values = zip(*points)
                axis.plot(steps, _smooth(list(values), args.window), label=seed, linewidth=1.4)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.3)
    for axis in list(axes.flat)[len(panels) :]:
        axis.axis("off")
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(f"{args.run}: training curves (rolling mean over {args.window} logging steps)", fontsize=11)
    figure.tight_layout()

    output = run_dir / "training_curves.png"
    figure.savefig(output, dpi=150)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
