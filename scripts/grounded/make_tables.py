"""Build the result tables from results/eval/ (no number is typed by hand).

    python scripts/grounded/make_tables.py

Runs whose names differ only by a `_seed<k>` suffix are grouped: the table then shows
the mean and standard deviation across seeds. A single run is shown with its 95%
bootstrap interval over examples. Smoke-test runs (`--limit`) are ignored.

Outputs
-------
results/tables/<dataset>.md     one Markdown table per evaluation dataset
results/tables/summary.json     the same numbers in machine-readable form
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from open_r1.grounded.runinfo import write_json  # noqa: E402

# (metric key, column header, "higher is better")
COLUMNS = {
    "squad_v2_distractors": [
        ("format_rate", "Format", True),
        ("em", "EM", True),
        ("f1", "F1", True),
        ("has_ans_f1", "HasAns F1", True),
        ("no_ans_em", "NoAns acc.", True),
        ("abstention_f1", "Abstention F1", True),
        ("unverified_quote_rate", "Unverified quotes", False),
        ("evidence_recall", "Evidence recall", True),
        ("chunk_f1", "Chunk F1", True),
    ],
    "hotpotqa": [
        ("format_rate", "Format", True),
        ("em", "Ans EM", True),
        ("f1", "Ans F1", True),
        ("sp_em", "Sup EM", True),
        ("sp_f1", "Sup F1", True),
        ("joint_f1", "Joint F1", True),
        ("chunk_f1", "Chunk F1", True),
        ("unverified_quote_rate", "Unverified quotes", False),
    ],
    "pubmedqa": [
        ("format_rate", "Format", True),
        ("accuracy", "Accuracy", True),
        ("macro_f1", "Macro F1", True),
        ("answer_rate", "Answer rate", True),
        ("unverified_quote_rate", "Unverified quotes", False),
    ],
}
COLUMNS["squad_v2"] = COLUMNS["squad_v2_distractors"]
_SEED_SUFFIX = re.compile(r"_seed(\d+)$")


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}"


def _cell(runs: list[dict], key: str) -> tuple[str, dict]:
    values = [run["metrics"].get(key, {}).get("value") for run in runs]
    if any(v is None for v in values):
        return "n/a", {"value": None}
    if len(runs) == 1:
        metric = runs[0]["metrics"][key]
        if metric["ci_low"] is None:
            return _percent(values[0]), {"value": values[0]}
        text = f"{_percent(values[0])} [{_percent(metric['ci_low'])}, {_percent(metric['ci_high'])}]"
        return text, {"value": values[0], "ci_low": metric["ci_low"], "ci_high": metric["ci_high"]}
    mean, std = statistics.mean(values), statistics.stdev(values)
    return f"{_percent(mean)} ± {100 * std:.1f}", {"mean": mean, "std": std, "values": values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_dir", default="results/eval")
    parser.add_argument("--output_dir", default="results/tables")
    parser.add_argument("--include_smoke_tests", action="store_true", help="For testing this script only.")
    args = parser.parse_args()

    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(args.eval_dir).glob("*/*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("is_smoke_test") and not args.include_smoke_tests:
            continue
        groups[run["dataset"]][_SEED_SUFFIX.sub("", run["run_name"])].append(run)

    summary: dict[str, dict] = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, by_group in groups.items():
        columns = COLUMNS.get(dataset, COLUMNS["squad_v2"])
        titles = [f"{title} {'↑' if up else '↓'}" for _, title, up in columns]
        header = "| Run | Seeds | n | " + " | ".join(titles) + " |"
        lines = [header, "|---|---:|---:|" + "---:|" * len(columns)]
        summary[dataset] = {}
        for group, runs in sorted(by_group.items()):
            cells, raw = [], {}
            for key, _, _ in columns:
                text, numbers = _cell(runs, key)
                cells.append(text)
                raw[key] = numbers
            n_examples = runs[0]["dataset_spec"]["n_examples"]
            lines.append(f"| `{group}` | {len(runs)} | {n_examples} | " + " | ".join(cells) + " |")
            summary[dataset][group] = {
                "n_seeds": len(runs),
                "n_examples": n_examples,
                "git_shas": sorted({run["environment"]["git_sha"] or "unknown" for run in runs}),
                "metrics": raw,
            }
        note = (
            "\nValues are percentages. One seed: point estimate [95% bootstrap interval over examples]. "
            "Several seeds: mean ± standard deviation across training seeds.\n"
        )
        (output_dir / f"{dataset}.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")
        print(f"## {dataset}\n" + "\n".join(lines) + "\n")
    write_json(output_dir / "summary.json", summary)
    if not groups:
        print(f"No evaluation results found in {args.eval_dir}.")


if __name__ == "__main__":
    main()
