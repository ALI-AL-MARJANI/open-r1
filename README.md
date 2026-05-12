# Grounded-R1

Grounded-R1 applies the GRPO training pipeline from [open-r1](https://github.com/huggingface/open-r1) to a fundamentally different problem: **hallucination-free information extraction**. Instead of rewarding correct math answers, we reward exact verbatim citations.

> *The model learns it cannot invent facts — every claim must be a substring of the input context.*

---

## The Problem

RAG systems hallucinate. Fine-tuned extractors hallucinate. Even the best LLMs confabulate quotes that look real but aren't. Current mitigations (RLHF, constitutional AI, output parsing) treat the symptom. We attack the cause at training time.

## The Insight

GRPO's relative reward signal is perfectly suited to this. Within a group of 8 completions for the same prompt:
- completions that cite verbatim substrings of the context score high
- completions that paraphrase or invent score zero

The policy gradient pushes the model toward grounded responses *without needing human preference labels*.

## Architecture

```
Prompt (context chunks + question)
         │
         ▼
  ┌─────────────┐     vLLM rollout    ┌────────────────────────────────────────┐
  │  LLM Policy │ ─── 8 completions ─▶│           Reward Stack                 │
  └─────────────┘                     │                                        │
         ▲                            │  R1  grounded_format    ×1.0           │
         │  policy gradient           │  R2  quote_grounding    ×1.5  ← v0+v1  │
         │                            │  R3  chunk_routing      ×2.0  ← v1 NEW │
         └──────────── GRPO ──────────│  R4  answer_faithfulness×1.0           │
                    (trl)             │  R5  reasoning_quality  ×0.5           │
                                      └────────────────────────────────────────┘
```

### Output Format (enforced by rewards)

```json
{
  "reasoning_path": "Chain-of-thought: which chunks are relevant, is context sufficient?",
  "is_context_sufficient": true,
  "final_answer": "Synthesised answer supported by the quotes below.",
  "extracted_quotes": [
    {
      "chunk_id": "doc_001",
      "exact_quote": "Verbatim substring copied from the source context — word for word."
    }
  ]
}
```

### Reward Design

| Reward | v0 | v1 | Weight | Signal |
|--------|----|----|--------|--------|
| `grounded_format` | ✓ | ✓ | ×1.0 | Additive structural check: JSON → keys → types → quote schema |
| `quote_grounding` | ✓ | ✓ | ×1.5 | `exact_quote ∈ context_raw` (any chunk). Correct abstention = 1.0. |
| `chunk_routing` | | ✓ | ×2.0 | `exact_quote ∈ gold_chunk_text` AND `chunk_id ∈ gold_chunk_ids`. Distractor citations = 0. |
| `answer_faithfulness` | ✓ | ✓ | ×1.0 | Token overlap between `final_answer` and extracted quotes. |
| `reasoning_quality` | ✓ | ✓ | ×0.5 | Soft CoT reward: non-empty, sufficiency vocab, ≥20 words. |

**v1 curriculum**: `quote_grounding` catches any hallucination (quote must exist *somewhere* in the context). `chunk_routing` adds the stricter requirement that the citation comes from the *correct* supporting passage — not a distractor. With 8 distractors per HotpotQA question, this forces genuine multi-hop reasoning.

## Datasets & Scaling

| Version | Dataset | Chunks/q | Challenge | Models |
|---------|---------|----------|-----------|--------|
| v0 | SQuAD v2 | 1 | Format + no hallucination; 43k unanswerable | 1.5B |
| v1 | HotpotQA distractor | 10 (2 gold + 8 distractor) | Multi-hop chunk routing | 1.5B |
| v2 | SQuAD v2 + same-article hard negatives | 4 | Topic-similar distractors; no gradient collapse | 1.5B / 7B |
| v3 | PubMedQA (biomedical) | 4–5 IMRAD sections | Hallucination vs memorised medical knowledge | 7B / 32B |

## Quickstart

```bash
git clone https://github.com/your-org/grounded-r1
cd grounded-r1
pip install -e ".[dev]"
```

**v0 — single-chunk SQuAD v2, 1.5B:**
```bash
python scripts/prepare_grounded_dataset.py --dataset squad_v2 --output_dir data/grounded-squad-v2
sbatch --nodes=2 slurm/train.slurm --model Grounded-R1-Qwen-1.5B --task grpo --config v0 --accelerator zero3
```

**v2 — hard-negative distractors, 7B:**
```bash
python scripts/inject_hard_negatives.py --n_negatives 3 --output_dir data/grounded-squad-v2-hard
sbatch --nodes=4 slurm/train.slurm --model Grounded-R1-Qwen-7B --task grpo --config v0 --accelerator zero3 --tp 2
```

**v3 — biomedical, 32B:**
```bash
python scripts/prepare_domain_dataset.py --domain pubmedqa --config pqa_artificial --output_dir data/grounded-pubmedqa
sbatch --nodes=16 slurm/train.slurm --model Grounded-R1-Qwen-32B --task grpo --config v0 --accelerator fsdp --tp 8
```

**Evaluate:**
```bash
python scripts/evaluate_grounded.py --model_name_or_path data/Grounded-R1-Qwen-7B-v0 --output_file eval_results.json
```

## Project Structure

```
src/open_r1/
├── grpo.py                  # GRPO training loop (upstream, unmodified)
├── grounded_rewards.py      # ← all Grounded-R1 reward functions (5 signals)
├── rewards.py               # reward registry
└── configs.py               # config dataclasses

scripts/
├── prepare_grounded_dataset.py  # squad_v2 + hotpotqa → grounded format
├── inject_hard_negatives.py     # v2: same-article distractor injection
├── prepare_domain_dataset.py    # v3: domain-specific (PubMedQA biomedical)
└── evaluate_grounded.py         # evaluation pipeline (10 metrics)

recipes/
├── Grounded-R1-Qwen-1.5B/grpo/  config_v0.yaml  config_v1.yaml  config_v2.yaml
├── Grounded-R1-Qwen-7B/grpo/    config_v0.yaml
└── Grounded-R1-Qwen-32B/grpo/   config_v0.yaml  (FSDP, paged_adamw_8bit, tp=8)

tests/
├── test_grounded_rewards.py # reward unit tests        (16 tests)
├── test_evaluation.py       # eval pipeline tests      (15 tests)
├── test_hard_negatives.py   # v2 injection tests       (23 tests)
└── test_domain_dataset.py   # v3 PubMedQA tests        (18 tests)
```

## Evaluation

```
============================================================
  Grounded-R1 Evaluation
============================================================
  GROUNDING
    quote accuracy         : 87.3%  ← fraction verbatim substrings
    hallucination rate     : 12.7%  ← lower is better
    abstention rate        : 82.5%  ← unanswerable correctly declined

  SUFFICIENCY CLASSIFICATION
    F1 (insufficient)      : 84.2%

  ANSWER QUALITY
    exact match            : 61.3%
    token F1               : 74.8%
============================================================
```
*(illustrative — real results coming after training)*

## Roadmap

- [x] **Reward stack** — format, grounding, faithfulness, reasoning (v0)
- [x] **Eval pipeline** — quote accuracy, hallucination rate, F1, sufficiency classification
- [x] **v1** — Multi-chunk HotpotQA, `chunk_routing_reward`
- [x] **v2** — Same-article hard-negative distractor injection (`inject_hard_negatives.py`)
- [x] **v3** — 7B/32B recipes (ZeRO-3 / FSDP), PubMedQA biomedical domain

## Why This Matters

RAG is now the dominant LLM deployment pattern. Every enterprise RAG system suffers from extraction hallucinations. A model trained to never fabricate citations — verified at training time by exact string matching — is a direct, deployable solution.

## Credits

Built on top of [`huggingface/open-r1`](https://github.com/huggingface/open-r1), which reproduces the DeepSeek-R1 GRPO training pipeline.
