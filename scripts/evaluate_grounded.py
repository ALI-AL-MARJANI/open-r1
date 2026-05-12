"""
Evaluation pipeline for Grounded-R1 models.

Runs inference on SQuAD v2 validation set and computes:

  Grounding metrics (novel — these are what matter for anti-hallucination):
    quote_accuracy        fraction of extracted quotes that are exact substrings of context
    hallucination_rate    1 - quote_accuracy  (the number to minimise)
    coverage_rate         fraction of answerable examples with ≥1 grounded quote
    abstention_rate       fraction of unanswerable examples where model correctly abstains

  Sufficiency classification (binary: is_context_sufficient vs gold unanswerable):
    sufficiency_accuracy  overall accuracy
    sufficiency_precision precision for "insufficient" class
    sufficiency_recall    recall for "insufficient" class
    sufficiency_f1        F1 for "insufficient" class

  Answer quality (standard SQuAD metrics):
    answer_em             exact match (answerable examples only)
    answer_f1             token-level F1 (answerable examples only)

  Format:
    format_rate           fraction of outputs with valid JSON + correct schema

Usage:
    python scripts/evaluate_grounded.py \\
        --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \\
        --limit 500 \\
        --output_file eval_results.json

    # Against a trained checkpoint:
    python scripts/evaluate_grounded.py \\
        --model_name_or_path data/Grounded-R1-Qwen-1.5B-v0 \\
        --output_file eval_results_grounded_r1.json
"""

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── System prompt (must match training recipe) ────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert information extraction system. Your task is to answer questions "
    "based STRICTLY on the provided context chunks.\n\n"
    "CRITICAL RULES:\n"
    "1. You MUST ONLY use information explicitly stated in the provided context.\n"
    "2. Every claim in your final_answer MUST be directly supported by an exact extracted_quote.\n"
    "3. Your \"exact_quote\" values MUST be verbatim substrings copied from the context — "
    "no paraphrasing, no omissions, no additions.\n"
    "4. If the context does not contain enough information to answer the question, "
    "set \"is_context_sufficient\": false and leave \"extracted_quotes\" as an empty list [].\n\n"
    "OUTPUT FORMAT — respond ONLY with this JSON schema (no markdown, no preamble):\n"
    "{\n"
    "  \"reasoning_path\": \"<chain-of-thought: identify relevant chunks, assess sufficiency>\",\n"
    "  \"is_context_sufficient\": <true|false>,\n"
    "  \"final_answer\": \"<synthesised answer, or abstention message if insufficient>\",\n"
    "  \"extracted_quotes\": [\n"
    "    {\"chunk_id\": \"<source chunk id>\", \"exact_quote\": \"<verbatim text from the chunk>\"}\n"
    "  ]\n"
    "}"
)

_INSUFFICIENT_PHRASES = frozenset({
    "not contain", "insufficient", "cannot answer", "no information",
    "not enough", "does not provide", "not mentioned", "not found",
    "unable to answer", "no relevant", "context does not", "does not contain",
})


# ── Prompt formatting ─────────────────────────────────────────────────────────

def _make_chunk_id(title: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", title).strip("_") + "_0"


def format_user_message(question: str, passage: str, title: str) -> str:
    chunk_id = _make_chunk_id(title)
    return (
        f'CONTEXT:\n[CHUNK id="{chunk_id}"]\n{passage.strip()}\n[/CHUNK]\n\n'
        f"QUESTION: {question.strip()}"
    )


# ── Answer normalisation (SQuAD standard) ────────────────────────────────────

def _normalise(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalise(pred).split()
    gold_toks = _normalise(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if common == 0:
        return 0.0
    p = common / len(pred_toks)
    r = common / len(gold_toks)
    return 2 * p * r / (p + r)


def _exact_match(pred: str, gold: str) -> bool:
    return _normalise(pred) == _normalise(gold)


def _best_f1_em(pred: str, gold_answers: list[str]) -> tuple[float, bool]:
    """Return best F1 and EM across all gold answers."""
    if not gold_answers:
        return 0.0, False
    f1 = max(_token_f1(pred, g) for g in gold_answers)
    em = any(_exact_match(pred, g) for g in gold_answers)
    return f1, em


# ── JSON parsing ──────────────────────────────────────────────────────────────

_REQUIRED_KEYS = frozenset({"reasoning_path", "is_context_sufficient", "final_answer", "extracted_quotes"})
_REQUIRED_QUOTE_KEYS = frozenset({"chunk_id", "exact_quote"})


def _parse_response(content: str) -> Optional[dict]:
    m = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _is_valid_schema(parsed: dict) -> bool:
    if not _REQUIRED_KEYS.issubset(parsed.keys()):
        return False
    if not (
        isinstance(parsed["reasoning_path"], str)
        and isinstance(parsed["is_context_sufficient"], bool)
        and isinstance(parsed["final_answer"], str)
        and isinstance(parsed["extracted_quotes"], list)
    ):
        return False
    return all(
        isinstance(q, dict) and _REQUIRED_QUOTE_KEYS.issubset(q.keys())
        for q in parsed["extracted_quotes"]
    )


def _is_grounded(quote: str, context: str) -> bool:
    if not quote:
        return False
    if quote in context:
        return True
    return " ".join(quote.split()) in " ".join(context.split())


# ── Model inference ───────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, device: str):
    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model: {model_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    user_message: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,  # greedy for eval
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=temperature > 0,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Metrics accumulator ───────────────────────────────────────────────────────

class MetricsAccumulator:
    def __init__(self):
        self.total = 0
        self.valid_format = 0

        # Grounding
        self.quote_scores: list[float] = []       # per-example quote accuracy
        self.answerable_with_quotes = 0           # for coverage
        self.answerable_total = 0

        # Sufficiency classification
        self.tp_insuff = 0   # gold unanswerable, model said insufficient
        self.fp_insuff = 0   # gold answerable,   model said insufficient
        self.tn_suff = 0     # gold answerable,   model said sufficient
        self.fn_insuff = 0   # gold unanswerable, model said sufficient

        # Answer quality (answerable only)
        self.f1_scores: list[float] = []
        self.em_scores: list[bool] = []

        # Abstention quality (unanswerable only)
        self.correct_abstentions = 0
        self.unanswerable_total = 0

        self.examples: list[dict] = []  # for per-example output file

    def update(self, example: dict, raw_output: str, parsed: Optional[dict]):
        self.total += 1
        is_unanswerable = not example["answers"]["text"]  # SQuAD: empty = unanswerable
        gold_answers = example["answers"]["text"]
        context = example["context"]

        record = {
            "id": example["id"],
            "question": example["question"],
            "is_unanswerable": is_unanswerable,
            "gold_answers": gold_answers,
            "raw_output": raw_output,
            "valid_format": False,
            "is_context_sufficient": None,
            "quote_accuracy": None,
            "answer_f1": None,
            "answer_em": None,
        }

        if parsed is None or not _is_valid_schema(parsed):
            self.examples.append(record)
            return

        self.valid_format += 1
        record["valid_format"] = True

        predicted_sufficient = parsed["is_context_sufficient"]
        quotes = parsed["extracted_quotes"]
        final_answer = parsed["final_answer"]
        record["is_context_sufficient"] = predicted_sufficient
        record["final_answer"] = final_answer

        # ── Sufficiency classification ─────────────────────────────────────
        # is_context_sufficient=True  ↔  not unanswerable
        # is_context_sufficient=False ↔  unanswerable
        if is_unanswerable:
            self.unanswerable_total += 1
            if not predicted_sufficient:
                self.tp_insuff += 1
            else:
                self.fn_insuff += 1
        else:
            self.answerable_total += 1
            if not predicted_sufficient:
                self.fp_insuff += 1
            else:
                self.tn_suff += 1

        # ── Quote grounding ────────────────────────────────────────────────
        valid_quotes = [q for q in quotes if isinstance(q, dict) and q.get("exact_quote", "").strip()]
        if valid_quotes:
            grounded = sum(1 for q in valid_quotes if _is_grounded(q["exact_quote"], context))
            quote_acc = grounded / len(valid_quotes)
            self.quote_scores.append(quote_acc)
            record["quote_accuracy"] = quote_acc
            if not is_unanswerable:
                self.answerable_with_quotes += 1
        elif not is_unanswerable and predicted_sufficient:
            # Claimed sufficient but no quotes — worst case
            self.quote_scores.append(0.0)
            record["quote_accuracy"] = 0.0

        # ── Abstention quality ─────────────────────────────────────────────
        if is_unanswerable and not predicted_sufficient:
            answer_lower = final_answer.lower()
            if any(p in answer_lower for p in _INSUFFICIENT_PHRASES):
                self.correct_abstentions += 1

        # ── Answer quality (answerable only) ──────────────────────────────
        if not is_unanswerable and predicted_sufficient and gold_answers:
            f1, em = _best_f1_em(final_answer, gold_answers)
            self.f1_scores.append(f1)
            self.em_scores.append(em)
            record["answer_f1"] = f1
            record["answer_em"] = em

        self.examples.append(record)

    def compute(self) -> dict:
        def safe_div(a, b): return a / b if b > 0 else 0.0

        format_rate = safe_div(self.valid_format, self.total)
        quote_accuracy = sum(self.quote_scores) / len(self.quote_scores) if self.quote_scores else 0.0
        coverage_rate = safe_div(self.answerable_with_quotes, self.answerable_total)
        abstention_rate = safe_div(self.tp_insuff, self.unanswerable_total)
        correct_abstention_quality = safe_div(self.correct_abstentions, self.tp_insuff) if self.tp_insuff > 0 else 0.0

        suff_precision = safe_div(self.tp_insuff, self.tp_insuff + self.fp_insuff)
        suff_recall = safe_div(self.tp_insuff, self.tp_insuff + self.fn_insuff)
        suff_f1 = safe_div(2 * suff_precision * suff_recall, suff_precision + suff_recall)
        suff_accuracy = safe_div(self.tp_insuff + self.tn_suff, self.total)

        answer_f1 = sum(self.f1_scores) / len(self.f1_scores) if self.f1_scores else 0.0
        answer_em = sum(self.em_scores) / len(self.em_scores) if self.em_scores else 0.0

        return {
            "total_examples": self.total,
            "answerable": self.answerable_total,
            "unanswerable": self.unanswerable_total,
            "format": {
                "format_rate": round(format_rate, 4),
            },
            "grounding": {
                "quote_accuracy": round(quote_accuracy, 4),
                "hallucination_rate": round(1 - quote_accuracy, 4),
                "coverage_rate": round(coverage_rate, 4),
                "abstention_rate": round(abstention_rate, 4),
                "abstention_quality": round(correct_abstention_quality, 4),
            },
            "sufficiency_classification": {
                "accuracy": round(suff_accuracy, 4),
                "precision": round(suff_precision, 4),
                "recall": round(suff_recall, 4),
                "f1": round(suff_f1, 4),
            },
            "answer_quality": {
                "exact_match": round(answer_em, 4),
                "f1": round(answer_f1, 4),
                "n_scored": len(self.f1_scores),
            },
        }


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_report(metrics: dict, model_name: str):
    print("\n" + "=" * 60)
    print(f"  Grounded-R1 Evaluation — {Path(model_name).name}")
    print("=" * 60)
    print(f"  Examples : {metrics['total_examples']} "
          f"({metrics['answerable']} answerable, {metrics['unanswerable']} unanswerable)")
    print()
    print("  FORMAT")
    print(f"    valid JSON schema      : {metrics['format']['format_rate']:.1%}")
    print()
    print("  GROUNDING  (primary anti-hallucination metrics)")
    g = metrics["grounding"]
    print(f"    quote accuracy         : {g['quote_accuracy']:.1%}  ← fraction of quotes that are verbatim substrings")
    print(f"    hallucination rate     : {g['hallucination_rate']:.1%}  ← lower is better")
    print(f"    coverage rate          : {g['coverage_rate']:.1%}  ← answerable examples with ≥1 grounded quote")
    print(f"    abstention rate        : {g['abstention_rate']:.1%}  ← unanswerable examples correctly abstained")
    print(f"    abstention quality     : {g['abstention_quality']:.1%}  ← abstentions with correct phrase")
    print()
    print("  SUFFICIENCY CLASSIFICATION  (is_context_sufficient vs gold)")
    s = metrics["sufficiency_classification"]
    print(f"    accuracy               : {s['accuracy']:.1%}")
    print(f"    precision (insuff.)    : {s['precision']:.1%}")
    print(f"    recall    (insuff.)    : {s['recall']:.1%}")
    print(f"    F1        (insuff.)    : {s['f1']:.1%}")
    print()
    print("  ANSWER QUALITY  (answerable examples, gold-matched answers only)")
    a = metrics["answer_quality"]
    print(f"    exact match            : {a['exact_match']:.1%}  (n={a['n_scored']})")
    print(f"    token F1               : {a['f1']:.1%}")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Grounded-R1 model on SQuAD v2")
    parser.add_argument("--model_name_or_path", required=True, help="HF model ID or local checkpoint path")
    parser.add_argument("--split", default="validation", help="Dataset split (default: validation)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_file", default=None, help="Save full results to this JSON file")
    parser.add_argument("--print_examples", type=int, default=3, help="Print N example outputs (0 to disable)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, device)

    # Load SQuAD v2 validation
    print(f"\nLoading SQuAD v2 ({args.split})...")
    dataset = load_dataset("rajpurkar/squad_v2", split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Evaluating {len(dataset)} examples...\n")

    acc = MetricsAccumulator()
    printed = 0

    for example in tqdm(dataset, desc="Evaluating"):
        user_msg = format_user_message(example["question"], example["context"], example["title"])
        raw_output = generate_response(model, tokenizer, user_msg, max_new_tokens=args.max_new_tokens)
        parsed = _parse_response(raw_output)
        acc.update(example, raw_output, parsed)

        if args.print_examples and printed < args.print_examples:
            print(f"\n--- Example {printed + 1} ---")
            print(f"Q: {example['question']}")
            print(f"Gold: {example['answers']['text'] or '[unanswerable]'}")
            print(f"Output:\n{raw_output[:600]}")
            printed += 1

    metrics = acc.compute()
    print_report(metrics, args.model_name_or_path)

    if args.output_file:
        output = {
            "model": args.model_name_or_path,
            "split": args.split,
            "limit": args.limit,
            "metrics": metrics,
            "examples": acc.examples,
        }
        Path(args.output_file).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"Full results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
