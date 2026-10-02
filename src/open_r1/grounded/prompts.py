"""Prompt construction shared by training, the SFT baseline and evaluation."""

from __future__ import annotations

import json

SYSTEM_PROMPT = """You answer questions using only the provided context chunks.

Rules:
1. Copy the evidence first: each "exact_quote" must be a verbatim substring of one chunk, \
and "chunk_id" must be the id of that chunk. Quote the shortest passage that supports the answer.
2. "final_answer" must be supported by the quotes and as short as possible.
3. If the context does not contain the answer, set "is_context_sufficient" to false, \
"extracted_quotes" to [] and "final_answer" to "".

Respond with one JSON object and nothing else:
{"extracted_quotes": [{"chunk_id": "<id>", "exact_quote": "<verbatim text>"}], \
"is_context_sufficient": <true|false>, "final_answer": "<answer>"}"""


def format_chunks(chunks: dict[str, str], order: list[str] | None = None) -> str:
    ids = order if order is not None else list(chunks)
    return "\n\n".join(f'[CHUNK id="{cid}"]\n{chunks[cid].strip()}\n[/CHUNK]' for cid in ids)


def build_user_message(question: str, chunks: dict[str, str], order: list[str] | None = None) -> str:
    return f"CONTEXT:\n{format_chunks(chunks, order)}\n\nQUESTION: {question.strip()}"


def build_target(answer: str | None, quotes: list[tuple[str, str]]) -> str:
    """Serialise a reference response. `answer=None` produces an abstention."""
    if answer is None:
        payload = {"extracted_quotes": [], "is_context_sufficient": False, "final_answer": ""}
    else:
        payload = {
            "extracted_quotes": [{"chunk_id": cid, "exact_quote": text} for cid, text in quotes],
            "is_context_sufficient": True,
            "final_answer": answer,
        }
    return json.dumps(payload, ensure_ascii=False)


def build_messages(
    user_message: str,
    few_shot: list[tuple[str, str]] | None = None,
    target: str | None = None,
) -> list[dict[str, str]]:
    """Chat messages for one example.

    `few_shot` is a list of (user message, reference response) demonstrations inserted
    as earlier turns. `target` appends the assistant turn (SFT only).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demo_user, demo_assistant in few_shot or []:
        messages.append({"role": "user", "content": demo_user})
        messages.append({"role": "assistant", "content": demo_assistant})
    messages.append({"role": "user", "content": user_message})
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages
