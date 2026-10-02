"""Score question-independent policies under the current and the first (v0) reward suites.

    python scripts/grounded/reward_hacking_report.py

Runs on CPU in a few seconds (it only downloads SQuAD v2). For each policy in
`open_r1.grounded.policies`, the script reports the share of the maximum reward it
obtains on the in-distribution evaluation subset (balanced SQuAD v2 dev with
distractors), overall and on each half, and the same quantity under the v0 rewards.

Outputs
-------
results/reward_hacking.json   raw numbers, config and environment
results/reward_hacking.md     the table shown in the README
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from open_r1.grounded.config import load_config, reward_spec  # noqa: E402
from open_r1.grounded.data import load_records  # noqa: E402
from open_r1.grounded.legacy_v0 import LEGACY_V0_WEIGHTS, legacy_reward  # noqa: E402
from open_r1.grounded.policies import TRIVIAL_POLICIES, reference  # noqa: E402
from open_r1.grounded.rewards import weighted_reward  # noqa: E402
from open_r1.grounded.runinfo import collect_run_info, write_json  # noqa: E402

COLUMNS = ("context_chunks", "gold_answers", "gold_chunk_ids", "is_answerable")


def _fraction(pairs: list[tuple[float, float]]) -> float:
    return sum(total for total, _ in pairs) / sum(maximum for _, maximum in pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/grounded/eval.yaml")
    parser.add_argument("--dataset", default="squad_v2_distractors")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    weights, settings = reward_spec(config)
    spec = config["datasets"][args.dataset]
    records = load_records(
        spec["dataset"],
        spec["split"],
        spec["n_examples"],
        config["seed"],
        n_distractors=spec.get("n_distractors", 0),
        answerable_fraction=spec.get("answerable_fraction"),
    )
    columns = {key: [r[key] for r in records] for key in COLUMNS}
    answerable = [i for i, r in enumerate(records) if r["is_answerable"]]
    unanswerable = [i for i, r in enumerate(records) if not r["is_answerable"]]

    rows = []
    for name, policy in {**TRIVIAL_POLICIES, "reference (gold labels)": reference}.items():
        outputs = [policy(r) for r in records]
        current = weighted_reward(outputs, columns, weights, settings)
        legacy = [
            legacy_reward(output, " ".join(json.loads(r["context_chunks"]).values()))
            for output, r in zip(outputs, records)
        ]
        rows.append(
            {
                "policy": name,
                "v0_reward_fraction": _fraction(legacy),
                "reward_fraction": _fraction(current),
                "reward_fraction_answerable": _fraction([current[i] for i in answerable]),
                "reward_fraction_unanswerable": _fraction([current[i] for i in unanswerable]),
            }
        )

    output_dir = Path(args.output_dir)
    write_json(
        output_dir / "reward_hacking.json",
        {
            "dataset": args.dataset,
            "dataset_spec": {**spec, "subset_seed": config["seed"]},
            "n_examples": len(records),
            "n_answerable": len(answerable),
            "reward_weights": weights,
            "reward_settings": settings.__dict__,
            "v0_reward_weights": LEGACY_V0_WEIGHTS,
            "definition": "sum of weighted reward over examples / sum of maximum attainable reward",
            "policies": rows,
            "environment": collect_run_info(),
        },
    )

    lines = [
        "| Policy (ignores the question) | v0 rewards | Current rewards | Answerable half | Unanswerable half |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | {row['v0_reward_fraction']:.1%} | {row['reward_fraction']:.1%} "
            f"| {row['reward_fraction_answerable']:.1%} | {row['reward_fraction_unanswerable']:.1%} |"
        )
    table = "\n".join(lines) + "\n"
    (output_dir / "reward_hacking.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
