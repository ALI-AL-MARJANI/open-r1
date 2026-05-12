# CLAUDE.md — Grounded-R1 Project Context

## What This Project Is

Fork of `huggingface/open-r1` (DeepSeek-R1 GRPO reproduction). We repurpose the GRPO training pipeline from math reasoning → **hallucination-free information extraction**.

Core innovation: reward the model for verbatim citations (exact substring matching) instead of correct math answers. The upstream `src/open_r1/grpo.py` is **never modified** — all new code is additive.

---

## Architecture

`GRPOTrainer` passes all non-`prompt` dataset columns as `**kwargs` to reward functions. We exploit this: add `context_raw`, `context_chunks`, `gold_chunk_ids` columns to the dataset, and the reward functions receive them automatically with zero changes to the training loop.

### Required dataset columns

| Column | Type | Used by |
|--------|------|---------|
| `prompt` | str | training loop — full user message with `[CHUNK]` blocks + question |
| `context_raw` | str | `quote_grounding_reward` — concatenation of all chunk texts |
| `context_chunks` | str (JSON) | `chunk_routing_reward` — `{chunk_id: text}` dict |
| `gold_chunk_ids` | str (JSON) | `chunk_routing_reward` — `[chunk_id, ...]`; `[]` = unanswerable |
| `solution` | str | eval only — gold answer string |

### Reward function signature
```python
def reward_fn(completions: list[list[dict]], **kwargs) -> list[float | None]:
    content = completions[i][0]["content"]  # always read index [0]["content"]
```

### Core grounding primitive
```python
def _is_grounded(quote: str, context: str) -> bool:
    if quote in context: return True
    return " ".join(quote.split()) in " ".join(context.split())  # whitespace norm
```
**No fuzzy matching** — the model must copy verbatim. This is the design intent.

---

## File Map

```
src/open_r1/
├── grpo.py                    # upstream — DO NOT MODIFY
├── grounded_rewards.py        # all 5 reward functions
├── rewards.py                 # REWARD_FUNCS_REGISTRY — grounded rewards registered here
└── configs.py                 # GRPOScriptArguments reward_funcs help text updated

scripts/
├── prepare_grounded_dataset.py  # SQuAD v2 + HotpotQA → grounded schema
├── inject_hard_negatives.py     # v2: same-article distractor injection into SQuAD
├── prepare_domain_dataset.py    # v3: domain-specific (PubMedQA biomedical)
└── evaluate_grounded.py         # evaluation pipeline (10 metrics, no model changes needed)

recipes/
├── Grounded-R1-Qwen-1.5B/grpo/
│   ├── config_v0.yaml           # SQuAD v2, 4 rewards, 2 nodes
│   ├── config_v1.yaml           # HotpotQA 10-chunk, 5 rewards, 2 nodes
│   └── config_v2.yaml           # SQuAD v2 + hard negatives, 5 rewards, 2 nodes
├── Grounded-R1-Qwen-7B/grpo/
│   └── config_v0.yaml           # 4 nodes, ZeRO-3, tp=2, lr=2e-7
└── Grounded-R1-Qwen-32B/grpo/
    └── config_v0.yaml           # 16 nodes, FSDP, tp=8, paged_adamw_8bit, lr=5e-8

tests/
├── test_grounded_rewards.py     # reward functions    — 16 tests
├── test_evaluation.py           # eval pipeline       — 15 tests
├── test_hard_negatives.py       # v2 injection        — 23 tests
└── test_domain_dataset.py       # v3 PubMedQA         — 18 tests
                                 # TOTAL: 72 tests, all green
```

---

## Reward Stack

All registered in `rewards.py` under `REWARD_FUNCS_REGISTRY`.

| Key | Function | Requires | Signal |
|-----|----------|----------|--------|
| `grounded_format` | `grounded_format_reward` | — | Additive JSON schema check (+0.25 per layer). Max 1.0. |
| `quote_grounding` | `quote_grounding_reward` | `context_raw` | Fraction of quotes that are exact substrings of ANY chunk. Correct abstention → 1.0. |
| `chunk_routing` | `chunk_routing_reward` | `context_chunks`, `gold_chunk_ids` | Fraction of quotes from the correct gold chunk. Distractor citations → 0. |
| `answer_faithfulness` | `answer_faithfulness_reward` | — | Token overlap between `final_answer` and extracted quotes. |
| `reasoning_quality` | `reasoning_quality_reward` | — | Soft CoT: non-empty +0.5, sufficiency vocab +0.3, ≥20 words +0.2. |

**Two-level hierarchy**: `quote_grounding` catches hallucinations vs all context. `chunk_routing` adds the stricter requirement of citing from the right passage. Used together from v1 onwards.

---

## Dataset Pipelines

### `prepare_grounded_dataset.py --dataset squad_v2`
- Source: `rajpurkar/squad_v2` — 86k answerable + 43k unanswerable
- Single chunk per example, `gold_chunk_ids=[chunk_id]` or `[]`

### `prepare_grounded_dataset.py --dataset hotpotqa`
- Source: `hotpot_qa` distractor config — 10 chunks/question (2 gold + 8 distractors)
- Chunks shuffled deterministically per `hash(example_id)`
- Handles title collision with suffix counter

### `inject_hard_negatives.py`
- Input: raw SQuAD v2 from HF
- Adds same-article Wikipedia passages as distractors (`TitleIndex` groups by title)
- Fallback chain: same-title → adjacent-title → random
- `gold_chunk_id` unchanged; distractor ids get `_neg1`, `_neg2` suffix
- Prevents GRPO gradient collapse when model converges on easy examples

### `prepare_domain_dataset.py --domain pubmedqa`
- Source: `qiaojin/PubMedQA` — configs: `pqa_labeled` (1k expert), `pqa_artificial` (211k silver)
- IMRAD sections → one chunk each (`pubmed_{pubid}_{LABEL}`)
- `"yes"/"no"` → `gold_chunk_ids=all_sections` (abstraction: any abstract section is citable)
- `"maybe"` → `gold_chunk_ids=[]` (genuine uncertainty = correct abstention)
- Sections shuffled to prevent model exploiting CONCLUSIONS-always-last

---

## Evaluation Pipeline

```bash
python scripts/evaluate_grounded.py \
    --model_name_or_path <checkpoint> \
    --limit 500 \
    --output_file eval_results.json
```

10 metrics in 4 groups:

| Group | Metrics |
|-------|---------|
| Format | `format_rate` |
| Grounding | `quote_accuracy`, `hallucination_rate`, `coverage_rate`, `abstention_rate`, `abstention_quality` |
| Sufficiency | `accuracy`, `precision`, `recall`, `f1` |
| Answer quality | `exact_match`, `token_f1` (answerable only) |

Runs against raw SQuAD v2 validation (not preprocessed) — use to compare any checkpoint under identical conditions.

---

## Scaling Recipes

| Model | Nodes | Accel | vLLM tp | lr | key constraint |
|-------|-------|-------|---------|-----|----------------|
| Qwen2.5-1.5B | 2 | ZeRO-3 | 1 | 5e-7 | baseline |
| Qwen2.5-7B | 4 | ZeRO-3 | 2 | 2e-7 | bs=1/GPU |
| Qwen2.5-32B | 16 | **FSDP** | **8** | 5e-8 | `paged_adamw_8bit`, shorter completions (512) |

For 32B: FSDP (not ZeRO-3) is more stable. `paged_adamw_8bit` halves optimizer memory. Full node for vLLM server.

---

## Conventions

- Line length: 119 chars (see `setup.cfg`)
- Python 3.10+ type hints (`list[str]`, not `List[str]`)
- All reward function docstrings describe WHY, not WHAT
- Tests: `pytest tests/` — 72 tests total, must all pass before any commit
- Reward functions return `list[float]` not `list[float | None]` unless the signal truly can't be computed (rare)
