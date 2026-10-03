# Runbook

Exact commands to reproduce every table and figure of the README. Each command writes
its outputs under `results/` together with the configuration, seed, git SHA, library
versions and hardware (`environment` field of each JSON file).

Status of each step is tracked in the last section. A step marked *pending* has not
been run; its rows in the README read `TBD (run pending)`.

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install -r requirements-grounded.txt
make grounded-test                                      # 162 tests, < 1 s, CPU only
```

`requirements-grounded.txt` pins the versions used for development (TRL 0.29.1,
transformers 5.18.0, torch 2.14.1, peft 0.21.2, datasets 5.0.1). Do not install it
into the same environment as the upstream package (`pip install -e .` pins
`trl==0.18.0`); the grounded scripts do not need the package to be installed.

Models and datasets are downloaded from the Hugging Face Hub on first use
(Qwen2.5-0.5B: 1 GB, Qwen2.5-1.5B: 3 GB, SQuAD v2: 16 MB, HotpotQA distractor: 600 MB,
PubMedQA PQA-L: 1 MB). No paid API is used anywhere.

### Hardware and expected durations

Only the Apple M4 timings below were measured. GPU durations are estimates derived
from them and from the step sizes; measure your own with the 5-step pilot of section 3
before committing to a full run, and update this table.

| Step | Apple M4, 16 GB (MPS) | 1x T4 16 GB | 1x L4 / A10G 24 GB |
|---|---|---|---|
| Reward-hacking report | 10 s (measured, CPU) | same | same |
| Evaluation, 0.5B, 4 datasets (3,000 examples) | 65 min generation time (measured: 18 + 9 + 22 + 16 min) | ~15 min (estimate) | ~10 min (estimate) |
| Evaluation, 1.5B, 4 datasets | ~3x the 0.5B time (estimate) | ~30 min (estimate) | ~20 min (estimate) |
| SFT 0.5B, 250 steps | ~30 min (estimate) | ~10 min (estimate) | ~5 min (estimate) |
| GRPO 0.5B, 500 steps | not practical: 130-195 s per step measured on a 3-step pilot, with 8 GB of swap in use (about one day for 500 steps) | ~4-6 h (estimate) | ~2-3 h (estimate) |
| GRPO 1.5B, 300 steps | not recommended | ~8-12 h, memory-tight (estimate, untested) | ~4-6 h (estimate) |

Rented GPU prices change often; at roughly 0.5-1 USD per hour for an L4 or A10G, the
full plan below (3 seeds of the main run, 6 ablations, the 1.5B run and all
evaluations, about 30 GPU-hours on that class of card) costs in the order of
15-30 USD. On free T4 sessions (Kaggle, Colab) the same plan takes about twice as
long and needs `--resume` across sessions.

Overrides for specific hardware (append to any training command):

```bash
# T4 (no native bfloat16): float32 weights with fp16 autocast
--set model.dtype=float32 grpo.bf16=false grpo.fp16=true          # train_grpo.py
--set model.dtype=float32 sft.bf16=false sft.fp16=true            # train_sft.py

# Apple Silicon (MPS): float32, no mixed precision
--set model.dtype=float32 grpo.bf16=false                         # train_grpo.py
--set model.dtype=float32 sft.bf16=false                          # train_sft.py

# Less GPU memory: 4 sequences per backward pass instead of 8 (same 32 completions per step)
--set grpo.per_device_train_batch_size=4 grpo.gradient_accumulation_steps=8 grpo.gradient_checkpointing=true
```

The T4 and low-memory overrides have not been exercised; the MPS override is the one
used for the smoke tests during development.

## 1. Reward-hacking study (CPU, 10 s)

```bash
python scripts/grounded/reward_hacking_report.py
```

Writes `results/reward_hacking.json` and `results/reward_hacking.md` (README, "Reward
shortcuts"). **Status: done.**

## 2. Baselines without training

```bash
# Zero-shot
python scripts/grounded/evaluate.py --run_name zero_shot_qwen0.5b --model Qwen/Qwen2.5-0.5B-Instruct
python scripts/grounded/evaluate.py --run_name zero_shot_qwen1.5b --model Qwen/Qwen2.5-1.5B-Instruct

# Few-shot: 3 fixed demonstrations from the SQuAD v2 training split
python scripts/grounded/evaluate.py --run_name few_shot_qwen0.5b --model Qwen/Qwen2.5-0.5B-Instruct --few_shot
python scripts/grounded/evaluate.py --run_name few_shot_qwen1.5b --model Qwen/Qwen2.5-1.5B-Instruct --few_shot
```

Each command evaluates the four datasets of `configs/grounded/eval.yaml` and writes
`results/eval/<run_name>/<dataset>.json` and `<dataset>.predictions.jsonl`.
On a 16 GB machine add `--set generation.batch_size=8` for the two SQuAD sets and
`--datasets hotpotqa pubmedqa --set generation.batch_size=2` for the other two: with
batch size 8, the long HotpotQA prompts did not finish in 90 minutes on the M4.

Optional ceiling with a larger model served locally (Ollama exposes an
OpenAI-compatible endpoint; `vllm serve` works the same way on port 8000):

```bash
ollama pull qwen2.5:7b
python scripts/grounded/evaluate.py --run_name zero_shot_qwen7b_ollama \
    --backend openai --base_url http://localhost:11434/v1 --model qwen2.5:7b
```

The Ollama model is a 4-bit quantised build; report it as such.

## 3. Supervised baseline (SFT, LoRA)

```bash
for seed in 0 1 2; do
  python scripts/grounded/train_sft.py --config configs/grounded/sft_qwen0.5b.yaml --seed $seed
  python scripts/grounded/evaluate.py --run_name sft_qwen0.5b_seed$seed \
      --model Qwen/Qwen2.5-0.5B-Instruct --adapter data/sft_qwen0.5b/seed$seed
done
```

Adapters are written to `data/` (not committed); logs and provenance to
`results/sft/sft_qwen0.5b/seed<k>/`.

## 4. GRPO

[notebooks/grounded_r1_gpu_run.ipynb](notebooks/grounded_r1_gpu_run.ipynb) runs this section
for seed 0 on a free Kaggle or Colab T4 (pilot, full run, curves, evaluation, download of
`results/`). It has not been executed on a T4 yet.

Pilot first (5 steps, a few minutes) to check memory and measure the step time:

```bash
python scripts/grounded/train_grpo.py --config configs/grounded/grpo_qwen0.5b.yaml --seed 0 \
    --checkpoint_root data/pilot --results_root results/pilot --set grpo.max_steps=5 grpo.logging_steps=1
grep -o '"step_time": [0-9.]*' results/pilot/grpo_qwen0.5b/seed0/train_log.jsonl
rm -r data/pilot results/pilot
```

Main run, three seeds:

```bash
for seed in 0 1 2; do
  python scripts/grounded/train_grpo.py --config configs/grounded/grpo_qwen0.5b.yaml --seed $seed
  python scripts/grounded/evaluate.py --run_name grpo_qwen0.5b_seed$seed \
      --model Qwen/Qwen2.5-0.5B-Instruct --adapter data/grpo_qwen0.5b/seed$seed
done
python scripts/grounded/plot_training.py --run grpo_qwen0.5b      # results/grpo/grpo_qwen0.5b/training_curves.png
```

An interrupted run continues from its last checkpoint (saved every 100 steps) with
`--resume`. If the budget only allows one seed, run seed 0: the tables then show
bootstrap intervals over examples instead of a standard deviation over seeds.

The learning rate (`1e-5`) has not been tuned. If `rewards/answer_correctness/mean`
is flat after 100 steps, or `rewards/format/mean` collapses, rerun the pilot with
`--set grpo.learning_rate=5e-6` and `2e-5` for 100 steps and keep the better one for
all seeds; record the choice in the README.

## 5. Ablations (one seed each, same protocol as the main run)

```bash
for name in no_answer_correctness no_quote_grounding no_chunk_routing no_answer_faithfulness kl_beta dr_grpo; do
  python scripts/grounded/train_grpo.py --config configs/grounded/ablation_$name.yaml --seed 0
  python scripts/grounded/evaluate.py --run_name ablation_${name}_seed0 \
      --model Qwen/Qwen2.5-0.5B-Instruct --adapter data/ablation_$name/seed0
  python scripts/grounded/plot_training.py --run ablation_$name
done
```

`ablation_kl_beta` loads a frozen reference copy of the model and needs more memory.

## 6. Larger model (1.5B)

```bash
python scripts/grounded/train_sft.py  --config configs/grounded/sft_qwen1.5b.yaml  --seed 0
python scripts/grounded/evaluate.py --run_name sft_qwen1.5b_seed0 \
    --model Qwen/Qwen2.5-1.5B-Instruct --adapter data/sft_qwen1.5b/seed0

python scripts/grounded/train_grpo.py --config configs/grounded/grpo_qwen1.5b.yaml --seed 0
python scripts/grounded/evaluate.py --run_name grpo_qwen1.5b_seed0 \
    --model Qwen/Qwen2.5-1.5B-Instruct --adapter data/grpo_qwen1.5b/seed0
python scripts/grounded/plot_training.py --run grpo_qwen1.5b
```

## 7. Tables, significance tests, figures

```bash
python scripts/grounded/make_tables.py                       # results/tables/*.md, summary.json
python scripts/grounded/compare.py --a zero_shot_qwen0.5b --b grpo_qwen0.5b_seed0
python scripts/grounded/compare.py --a sft_qwen0.5b_seed0  --b grpo_qwen0.5b_seed0
python scripts/grounded/compare.py --a sft_qwen0.5b_seed0  --b grpo_qwen0.5b_seed0 --dataset hotpotqa \
    --metrics f1 sp_f1 joint_f1 chunk_f1 unverified_quote_rate
```

Copy the tables of `results/tables/` into the README (replacing the `TBD` rows) and
commit `results/` with an `exp:` commit. Do not edit numbers by hand.

For the qualitative analysis, compare the per-example files
`results/eval/zero_shot_qwen0.5b/squad_v2_distractors.predictions.jsonl` and
`results/eval/grpo_qwen0.5b_seed0/squad_v2_distractors.predictions.jsonl` (same ids,
same order) and pick examples by score difference, including regressions.

## 8. Publishing an adapter

```bash
huggingface-cli login
huggingface-cli upload <user>/grounded-r1-qwen2.5-0.5b-lora data/grpo_qwen0.5b/seed0 . \
    --exclude "checkpoint-*" "training_args.bin"
```

Add the link to the README once the upload exists.

## Status

| Step | Status | Output |
|---|---|---|
| 1. Reward-hacking study | done | `results/reward_hacking.{json,md}` |
| 2. Zero-shot Qwen2.5-0.5B | done (Apple M4) | `results/eval/zero_shot_qwen0.5b/` |
| 2. Zero-shot Qwen2.5-1.5B, few-shot 0.5B / 1.5B | pending | `results/eval/` |
| 3. SFT 0.5B, 3 seeds | pending | `results/sft/`, `results/eval/sft_qwen0.5b_seed*/` |
| 4. GRPO 0.5B, 3 seeds | pending | `results/grpo/`, `results/eval/grpo_qwen0.5b_seed*/` |
| 5. Ablations | pending | `results/eval/ablation_*_seed0/` |
| 6. SFT and GRPO 1.5B | pending | `results/eval/*_qwen1.5b_seed0/` |
| 7. Tables, tests, figures | partial (tables for the zero-shot 0.5B run only) | `results/tables/` |
| 8. Adapter on the Hub | pending | - |
