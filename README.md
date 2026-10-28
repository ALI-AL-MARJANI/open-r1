<div align="center">

# Grounded-R1

**Teaching a 0.5B model to quote its sources word for word,<br>and finding out how easily the rewards could be cheated first.**


</div>

---

Language models often back an answer with a "quote" that isn't in the source. When the answer is a span of the text,
a quote can be checked by plain string matching: exact, free, and no judge model needed. The plan was to use that check
as a reward to train **Qwen2.5-0.5B** with **GRPO**. Before spending GPU hours, one question had to be answered: *can a
model get the reward without doing the work?*

## In short

| | |
|:---:|:---|
| **100%**<br>of the reward | What a model that always answers "I don't know" got from the first reward version. |
| **37.5%** | What it gets now, after the redesign. Tests keep it there. |
| **20.5% → 0.8%** | Quotes not found in the source: zero-shot vs. fine-tuned (SFT) model. |
| **99.5%** | Share of questions the same SFT model refuses, because of one default decoding setting. |

## The reward was broken before training started

Twelve dumb policies were scored against the rewards, with no model involved: always refuse, copy the first passage,
cite the word "the", invent a quote. Five of them earned **100%** of the maximum reward. A model trained on that reward
would have learned to cheat, not to cite.

Each shortcut was closed, then locked with a test. The best cheating policy now gets 37.5%, and only because refusing is
the right answer on half the questions. The real answers, built from gold labels, still score 99.9%
([`results/reward_hacking.md`](results/reward_hacking.md)).

| Cheating policy | First rewards | Now |
|:---|---:|---:|
| Always refuse | 100.0% | 37.5% |
| Copy the first passage | 100.0% | 16.7% |
| Cite a stopword (`"the"`) | 77.8% | 12.5% |
| Invent a quote | 55.6% | 12.5% |

## What GRPO would have to beat

Three baselines on 1,000 SQuAD v2 questions (half of them unanswerable, with distracting paragraphs), with 95% bootstrap
intervals and paired tests ([`results/tables/`](results/tables/)):

| Model | F1 ↑ | Quotes not in the source ↓ |
|:---|---:|---:|
| Zero-shot | 7.7 | 20.5% |
| 3-shot | 18.3 | 16.3% |
| SFT (LoRA, 1 seed) | **59.9** | **0.8%** |

## One default setting flipped the results

Qwen's config ships a repetition penalty of 1.1, and `transformers` applies it even with greedy decoding. With it, the
SFT model refused **99.5%** of the questions: F1 on answerable ones fell from 50.4 to 0.8. The penalty is now set
explicitly and recorded in every result file ([`results/repetition_penalty_1.1/`](results/repetition_penalty_1.1/README.md)).

## Next steps

The GRPO pipeline (`trl.GRPOTrainer`, one main configuration and six ablations) is written and tested. It is ready to run;
the results above are the reward audit and the baselines it has to beat.

| Run | Estimated compute |
|:---|---:|
| GRPO, Qwen2.5-0.5B, 1 seed (500 steps) | ~2 h on one A100 |
| 3 seeds + 6 ablations | ~18 h on one A100 |
| SFT seeds 1 and 2, for mean ± std | ~1 h on one A100 |
