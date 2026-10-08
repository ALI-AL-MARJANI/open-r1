"""Paired bootstrap comparison of two evaluated runs on the same examples.

    python scripts/grounded/compare.py --a sft_qwen0.5b_seed0 --b grpo_qwen0.5b_seed0

Reads the per-example scores written by evaluate.py and tests, for each metric, whether
run B differs from run A (difference B - A, 95% interval, two-sided p-value).

Output: results/comparisons/<a>__vs__<b>__<dataset>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from open_r1.grounded.metrics import paired_bootstrap  # noqa: E402
from open_r1.grounded.runinfo import collect_run_info, write_json  # noqa: E402

DEFAULT_METRICS = ("f1", "has_ans_f1", "no_ans_em", "unverified_quote_rate", "evidence_recall", "chunk_f1")


def _load_rows(eval_dir: Path, run: str, dataset: str) -> list[dict]:
    path = eval_dir / run / f"{dataset}.predictions.jsonl"
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line)["scores"] for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="Baseline run name.")
    parser.add_argument("--b", required=True, help="Compared run name.")
    parser.add_argument("--dataset", default="squad_v2_distractors")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_dir", default="results/eval")
    parser.add_argument("--output_dir", default="results/comparisons")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    rows_a = _load_rows(eval_dir, args.a, args.dataset)
    rows_b = _load_rows(eval_dir, args.b, args.dataset)

    results = []
    for metric in args.metrics:
        try:
            result = paired_bootstrap(rows_a, rows_b, metric, args.n_resamples, args.seed)
        except (ValueError, KeyError) as error:
            print(f"{metric}: skipped ({error})")
            continue
        results.append(result)
        # A p-value of 0 only means that no resample crossed zero: report the resolution instead.
        resolution = 2 / args.n_resamples
        p_text = f"p < {resolution:g}" if result["p_value"] < resolution else f"p = {result['p_value']:.4f}"
        print(
            f"{metric}: B - A = {100 * result['difference']:+.1f} points "
            f"[{100 * result['ci_low']:+.1f}, {100 * result['ci_high']:+.1f}], {p_text}"
        )

    write_json(
        Path(args.output_dir) / f"{args.a}__vs__{args.b}__{args.dataset}.json",
        {
            "run_a": args.a,
            "run_b": args.b,
            "dataset": args.dataset,
            "test": "paired bootstrap over examples, percentile interval, two-sided p-value",
            "results": results,
            "environment": collect_run_info(),
        },
    )


if __name__ == "__main__":
    main()
