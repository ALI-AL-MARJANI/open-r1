"""Parsing of model completions into a validated grounded response.

The parser never raises on model output: anything that is not a well-typed response
is reported through `ParseStatus` so that the rewards can score it as a failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

REQUIRED_KEYS = ("extracted_quotes", "is_context_sufficient", "final_answer")
_DECODER = json.JSONDecoder()
# Upper bound on the number of "{" positions tried, to keep parsing linear on junk output.
_MAX_DECODE_ATTEMPTS = 16


class ParseStatus(IntEnum):
    """How far a completion got through validation (ordered from worst to best)."""

    NO_JSON = 0  # no JSON object could be decoded
    MISSING_KEYS = 1  # a JSON object, but not all required keys
    BAD_TYPES = 2  # required keys present with wrong types
    VALID = 3  # schema-valid


@dataclass(frozen=True)
class Quote:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class GroundedResponse:
    """Outcome of parsing one completion.

    `is_sufficient`, `answer` and `quotes` are only meaningful when `status` is VALID.
    `extra_chars` counts the non-whitespace characters outside the JSON object.
    """

    status: ParseStatus
    is_sufficient: bool = False
    answer: str = ""
    quotes: tuple[Quote, ...] = field(default_factory=tuple)
    extra_chars: int = 0

    @property
    def valid(self) -> bool:
        return self.status is ParseStatus.VALID

    @property
    def abstained(self) -> bool:
        return self.valid and not self.is_sufficient

    @property
    def consistent(self) -> bool:
        """An abstention cites nothing; an answer cites at least one quote."""
        return self.valid and (bool(self.quotes) == self.is_sufficient)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped


def _decode_first_object(text: str) -> tuple[dict[str, Any] | None, int]:
    """Decode the first JSON object in `text`.

    Uses `JSONDecoder.raw_decode` from each "{" in turn, so braces in surrounding prose
    or after the object do not break parsing. Returns the object and the number of
    non-whitespace characters outside it.
    """
    start = text.find("{")
    attempts = 0
    while start != -1 and attempts < _MAX_DECODE_ATTEMPTS:
        try:
            obj, end = _DECODER.raw_decode(text, start)
        except (json.JSONDecodeError, RecursionError):
            obj = None
        if isinstance(obj, dict):
            outside = text[:start] + text[end:]
            return obj, len("".join(outside.split()))
        attempts += 1
        start = text.find("{", start + 1)
    return None, 0


def _validate_quotes(raw: Any) -> tuple[Quote, ...] | None:
    if not isinstance(raw, list):
        return None
    quotes = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        chunk_id, text = item.get("chunk_id"), item.get("exact_quote")
        if not isinstance(chunk_id, str) or not isinstance(text, str) or not text.strip():
            return None
        quotes.append(Quote(chunk_id=chunk_id, text=text))
    return tuple(quotes)


def parse_response(content: Any) -> GroundedResponse:
    """Parse one completion. Accepts any input type and never raises."""
    if not isinstance(content, str):
        return GroundedResponse(status=ParseStatus.NO_JSON)

    obj, extra_chars = _decode_first_object(_strip_code_fence(content))
    if obj is None:
        return GroundedResponse(status=ParseStatus.NO_JSON)
    if any(key not in obj for key in REQUIRED_KEYS):
        return GroundedResponse(status=ParseStatus.MISSING_KEYS, extra_chars=extra_chars)

    quotes = _validate_quotes(obj["extracted_quotes"])
    is_sufficient = obj["is_context_sufficient"]
    answer = obj["final_answer"]
    if quotes is None or not isinstance(is_sufficient, bool) or not isinstance(answer, str):
        return GroundedResponse(status=ParseStatus.BAD_TYPES, extra_chars=extra_chars)

    return GroundedResponse(
        status=ParseStatus.VALID,
        is_sufficient=is_sufficient,
        answer=answer,
        quotes=quotes,
        extra_chars=extra_chars,
    )


def completion_text(completion: Any) -> Any:
    """Extract the assistant text from a TRL completion.

    TRL passes a plain string for text prompts and a list of chat messages
    (`[{"role": "assistant", "content": ...}]`) for conversational prompts.
    """
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content")
    return completion
