# Evaluations run with the repetition penalty inherited from the model (1.1)

These files were produced by `scripts/grounded/evaluate.py` before commit `4f70054`.
Until that commit the script did not set `repetition_penalty`, so `transformers`
applied the value of Qwen2.5's `generation_config.json` (1.1) although sampling was
disabled. The `generator` field of these files says `"decoding": "greedy"` and does not
record the penalty; the files are otherwise unmodified.

| Directory | Produced at | Note |
|---|---|---|
| `eval/zero_shot_qwen0.5b/` | commit `d37267c`, 2026-10-05 | four datasets |
| `eval/few_shot_qwen0.5b/` | commit `4f8f005`, 2026-10-09 | four datasets |
| `eval/sft_qwen0.5b_seed0/` | commit `7c73d0a` with uncommitted documentation changes, 2026-10-11 | three datasets; the PubMedQA run was stopped when the penalty was found |
| `comparisons/` | commit `4b26b85` | paired test, zero-shot against few-shot |
| `tables/` | `python scripts/grounded/make_tables.py --eval_dir results/repetition_penalty_1.1/eval --output_dir results/repetition_penalty_1.1/tables` | |

They are kept for the "Decoding sensitivity" section of the README. The reported
protocol uses a penalty of 1.0 (`results/eval/`). To regenerate a file of this
directory with the current code:

```bash
python scripts/grounded/evaluate.py --run_name zero_shot_qwen0.5b --model Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir results/repetition_penalty_1.1/eval --set generation.repetition_penalty=1.1
```
