# Grounded-R1

**Teaching LLMs to cite their sources — not hallucinate them.**

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
  ┌─────────────┐     vLLM rollout    ┌──────────────────────────────────┐
  │  LLM Policy │ ─── 8 completions ─▶│         Reward Stack             │
  └─────────────┘                     │                                  │
         ▲                            │  R1  grounded_format   ×1.0      │
         │  policy gradient           │  R2  quote_grounding   ×2.0  ◀── CORE
         │                            │  R3  answer_faithfulness×1.0     │
         └──────────── GRPO ──────────│  R4  reasoning_quality ×0.5      │
                    (trl)             └──────────────────────────────────┘
```

### Output Format (enforced by rewards)

The model must output **only** this JSON schema:

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

| Reward | Weight | Signal |
|--------|--------|--------|
| `grounded_format` | ×1.0 | Additive structural check: valid JSON → correct keys → correct types → well-formed quotes |
| `quote_grounding` | ×2.0 | **Core.** `exact_quote ∈ context_raw` (substring). Correct abstention (`is_context_sufficient=false`, empty quotes) = 1.0. Any hallucinated quote = proportional penalty. |
| `answer_faithfulness` | ×1.0 | Token overlap between `final_answer` content words and extracted quotes. Prevents using grounded quotes as cover while hallucinating the synthesis. |
| `reasoning_quality` | ×0.5 | Soft reward for non-trivial CoT: non-empty (+0.5), sufficiency vocabulary (+0.3), ≥20 words (+0.2). |

The `quote_grounding` reward is where the anti-hallucination pressure lives. The others shape format and output quality.

## Stack

| Component | Choice |
|-----------|--------|
| RL framework | `trl` GRPOTrainer |
| Rollout inference | `vLLM` (PagedAttention, batched generation) |
| Base model (MVP) | `Qwen/Qwen2.5-1.5B-Instruct` |
| Training dataset | SQuAD v2 (86k answerable + 43k unanswerable) |
| Compute | Slurm, multi-GPU, ZeRO-3 |

SQuAD v2 is a deliberate choice: its 43k unanswerable questions force the model to learn **when not to answer** — producing `is_context_sufficient: false` with an empty quotes list instead of hallucinating.

## Quickstart

### 1. Install

```bash
git clone https://github.com/your-org/grounded-r1
cd grounded-r1
pip install -e ".[dev]"
```

### 2. Prepare dataset

```bash
python scripts/prepare_grounded_dataset.py --output_dir data/grounded-squad-v2
```

This converts SQuAD v2 into the grounded format with `prompt` and `context_raw` columns.

### 3. Train (local, single GPU — for testing)

```bash
python src/open_r1/grpo.py \
  --config recipes/Grounded-R1-Qwen-1.5B/grpo/config_v0.yaml
```

### 4. Train (cluster, 2 nodes: 1 training + 1 vLLM)

```bash
sbatch --nodes=2 slurm/train.slurm \
  --model Grounded-R1-Qwen-1.5B \
  --task grpo \
  --config v0 \
  --accelerator zero3
```

## Project Structure

```
src/open_r1/
├── grpo.py                  # GRPO training loop (upstream, unmodified)
├── grounded_rewards.py      # ← Grounded-R1 reward functions
├── rewards.py               # Reward registry (grounded rewards registered here)
└── configs.py               # Config dataclasses

scripts/
└── prepare_grounded_dataset.py  # SQuAD v2 → grounded format

recipes/
└── Grounded-R1-Qwen-1.5B/grpo/config_v0.yaml

tests/
└── test_grounded_rewards.py
```

## Evaluation

```bash
python scripts/evaluate_grounded.py \
    --model_name_or_path path/to/checkpoint \
    --limit 500 \
    --output_file eval_results.json
```

```
============================================================
  Grounded-R1 Evaluation — checkpoint
============================================================
  Examples : 500 (323 answerable, 177 unanswerable)

  FORMAT
    valid JSON schema      : 94.2%

  GROUNDING
    quote accuracy         : 87.3%  ← fraction of quotes that are verbatim substrings
    hallucination rate     : 12.7%  ← lower is better
    coverage rate          : 91.0%  ← answerable examples with ≥1 grounded quote
    abstention rate        : 82.5%  ← unanswerable examples correctly abstained
    abstention quality     : 96.1%  ← abstentions with correct phrase

  SUFFICIENCY CLASSIFICATION
    accuracy               : 88.4%
    F1 (insufficient)      : 84.2%

  ANSWER QUALITY
    exact match            : 61.3%  (n=282)
    token F1               : 74.8%
============================================================
```
*(numbers above are illustrative — real results coming after training)*

## Roadmap

- [ ] **v0** — SQuAD v2, Qwen2.5-1.5B, single-chunk contexts
- [x] **Eval** — Quote accuracy, hallucination rate, F1, sufficiency classification
- [ ] **v1** — Multi-chunk contexts (NaturalQuestions, TriviaQA), chunk_id routing reward
- [ ] **v2** — Synthetic hard negatives (near-miss quotes to sharpen grounding signal)
- [ ] **v3** — Larger models (7B, 32B), domain-specific datasets (biomedical, legal, finance)

## Why This Matters

RAG is now the dominant LLM deployment pattern. Every enterprise RAG system suffers from extraction hallucinations. A model trained to never fabricate citations — verified at training time by exact string matching — is a direct, deployable solution.

## Credits

Built on top of [`huggingface/open-r1`](https://github.com/huggingface/open-r1), which reproduces the DeepSeek-R1 GRPO training pipeline.
