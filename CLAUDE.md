# CLAUDE.md — Grounded-R1 Project Context

## What This Project Is

Fork of `huggingface/open-r1` (DeepSeek-R1 GRPO reproduction). We repurpose the GRPO training pipeline from math reasoning → **hallucination-free information extraction**.

Core innovation: reward the model for verbatim citations (exact substring matching) instead of correct math answers.

## Key Architecture Decisions

### The grpo.py is NOT modified
The upstream `src/open_r1/grpo.py` is left untouched. Our work lives in:
- `src/open_r1/grounded_rewards.py` — new reward functions
- `scripts/prepare_grounded_dataset.py` — dataset preprocessing

This works because `GRPOTrainer` passes all non-`prompt` dataset columns as `**kwargs` to reward functions. So if the dataset has a `context_raw` column, it reaches `quote_grounding_reward(completions, context_raw, **kwargs)` automatically.

### Dataset columns required for grounded training
- `prompt` — Full user message: `[CHUNK id="..."]...[/CHUNK]` + question
- `context_raw` — Raw passage text (used by `quote_grounding_reward` for substring matching)
- `solution` — Gold answer string (optional, for evaluation only)

### Reward function signatures
All reward functions follow the trl GRPOTrainer signature:
```python
def reward_fn(completions: list[list[dict]], **kwargs) -> list[float | None]:
```
`completions[i]` is a list of message dicts; we always read `completions[i][0]["content"]`.

### The `context_raw` grounding check
Uses exact substring matching with whitespace normalisation:
```python
def _is_grounded(quote: str, context: str) -> bool:
    if quote in context: return True
    return " ".join(quote.split()) in " ".join(context.split())
```
**No fuzzy matching.** The model must learn to copy verbatim. This is the design intent.

### Reward registry
New rewards are registered in `src/open_r1/rewards.py` inside `REWARD_FUNCS_REGISTRY`. They are activated via the `reward_funcs` list in the YAML config.

## File Map

```
src/open_r1/
├── grpo.py                    # upstream training loop — DO NOT MODIFY
├── grounded_rewards.py        # all grounded reward functions
├── rewards.py                 # registry: grounded rewards imported + registered here
└── configs.py                 # GRPOScriptArguments, GRPOConfig dataclasses

scripts/
└── prepare_grounded_dataset.py  # SQuAD v2 → {prompt, context_raw, solution}

recipes/
└── Grounded-R1-Qwen-1.5B/grpo/config_v0.yaml  # training recipe YAML

tests/
└── test_grounded_rewards.py   # unit tests (16/16 green, run with pytest)
```

## Training Recipe Summary

Recipe: `recipes/Grounded-R1-Qwen-1.5B/grpo/config_v0.yaml`

- Model: `Qwen/Qwen2.5-1.5B-Instruct`, bf16, flash_attention_2
- Dataset: `data/grounded-squad-v2` (disk, from prepare script)
- Rewards: `grounded_format` ×1.0, `quote_grounding` ×2.0, `answer_faithfulness` ×1.0, `reasoning_quality` ×0.5
- GRPO: `num_generations=8`, `beta=0.04`, `temperature=0.9`
- Context: `max_prompt_length=1024`, `max_completion_length=512`
- vLLM: `use_vllm=true`
- Cluster: 2 nodes (1 training ZeRO-3, 1 vLLM server)

## System Prompt (used in recipe)

The model is instructed to output ONLY the JSON schema with:
- `reasoning_path`: CoT assessing context sufficiency
- `is_context_sufficient`: bool
- `final_answer`: synthesis or abstention phrase
- `extracted_quotes`: list of `{chunk_id, exact_quote}` verbatim substrings

## Dataset

SQuAD v2 (`rajpurkar/squad_v2`) chosen because:
- 86k answerable → trains `is_context_sufficient=true` + grounded quotes
- 43k unanswerable → trains `is_context_sufficient=false` + empty quotes (correct abstention = reward 1.0)

## Conventions

- Line length: 119 chars (see setup.cfg)
- Python 3.10+ type hints (`list[str]`, not `List[str]`)
- No docstrings on reward functions — behaviour is explained in the docstring block at module top
- Tests live in `tests/`, run with `pytest tests/test_grounded_rewards.py`

## Evaluation Pipeline

Script: `scripts/evaluate_grounded.py`

```bash
python scripts/evaluate_grounded.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --limit 500 \
    --output_file eval_results.json
```

Key metrics:
- `quote_accuracy` — fraction of extracted_quotes that are exact substrings (primary anti-hallucination signal)
- `hallucination_rate` — 1 - quote_accuracy (the number to minimise)
- `abstention_rate` — fraction of unanswerable examples where model correctly abstained
- `sufficiency_f1` — F1 of is_context_sufficient binary classification vs gold
- `answer_em` / `answer_f1` — standard SQuAD metrics (answerable examples only)

The `MetricsAccumulator` class handles per-example tracking + aggregate computation. Tests: `tests/test_evaluation.py`.

## Roadmap (next steps)

1. ~~Evaluate~~ — DONE (`scripts/evaluate_grounded.py`)
2. **Multi-chunk** — extend dataset pipeline to multi-passage contexts with chunk routing
3. **Synthetic hard negatives** — generate near-miss quotes (paraphrases) to sharpen the grounding signal
4. **Scale** — Qwen2.5-7B, then 32B with FSDP
