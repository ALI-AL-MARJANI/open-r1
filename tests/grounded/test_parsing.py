import json
import random
import string

import pytest

from open_r1.grounded.parsing import ParseStatus, completion_text, parse_response

VALID = {
    "extracted_quotes": [{"chunk_id": "c1", "exact_quote": "a region in France"}],
    "is_context_sufficient": True,
    "final_answer": "France",
}


def test_valid_response():
    response = parse_response(json.dumps(VALID))
    assert response.status is ParseStatus.VALID
    assert response.answer == "France"
    assert response.quotes[0].chunk_id == "c1"
    assert response.extra_chars == 0
    assert response.consistent and not response.abstained


def test_code_fence_is_accepted():
    response = parse_response(f"```json\n{json.dumps(VALID)}\n```")
    assert response.valid and response.extra_chars == 0


def test_text_with_braces_after_the_object_does_not_break_parsing():
    # The first version used a greedy `{.*}` regex, which failed on this input.
    response = parse_response(json.dumps(VALID) + "\nNote: {this is not json}")
    assert response.valid
    assert response.extra_chars > 0


def test_prose_with_braces_before_the_object():
    response = parse_response("Using the set {a, b}: " + json.dumps(VALID))
    assert response.valid and response.extra_chars > 0


def test_missing_keys():
    assert parse_response('{"final_answer": "x"}').status is ParseStatus.MISSING_KEYS
    assert parse_response("{}").status is ParseStatus.MISSING_KEYS


@pytest.mark.parametrize(
    "patch",
    [
        {"is_context_sufficient": "true"},
        {"is_context_sufficient": 1},
        {"final_answer": 42},
        {"final_answer": None},
        {"extracted_quotes": "c1"},
        {"extracted_quotes": ["a quote"]},
        {"extracted_quotes": [{"chunk_id": "c1", "exact_quote": 3.14}]},
        {"extracted_quotes": [{"chunk_id": "c1", "exact_quote": None}]},
        {"extracted_quotes": [{"chunk_id": 1, "exact_quote": "text"}]},
        {"extracted_quotes": [{"chunk_id": "c1", "exact_quote": "   "}]},
        {"extracted_quotes": [{"chunk_id": "c1"}]},
    ],
)
def test_wrong_types_are_reported_not_raised(patch):
    assert parse_response(json.dumps({**VALID, **patch})).status is ParseStatus.BAD_TYPES


NOT_JSON = [None, 12, [], {}, "", "   ", "no json here", "{", "}{", '{"a": ', "[1, 2]", "{" * 5000]


@pytest.mark.parametrize("content", NOT_JSON)
def test_non_json_inputs(content):
    assert parse_response(content).status is ParseStatus.NO_JSON


def test_abstention_and_consistency():
    abstain = parse_response('{"extracted_quotes": [], "is_context_sufficient": false, "final_answer": ""}')
    assert abstain.abstained and abstain.consistent
    cites_but_abstains = parse_response(json.dumps({**VALID, "is_context_sufficient": False}))
    assert cites_but_abstains.abstained and not cites_but_abstains.consistent
    answers_without_quote = parse_response(json.dumps({**VALID, "extracted_quotes": []}))
    assert answers_without_quote.valid and not answers_without_quote.consistent


def test_fuzz_never_raises():
    """Random mutations of a valid response and random strings must never raise."""
    rng = random.Random(0)
    base = json.dumps(VALID)
    alphabet = string.printable + '{}[]":,\\'
    for _ in range(3000):
        chars = list(base)
        for _ in range(rng.randint(1, 8)):
            action = rng.choice(("delete", "insert", "replace", "truncate"))
            if not chars:
                break
            position = rng.randrange(len(chars))
            if action == "delete":
                del chars[position]
            elif action == "insert":
                chars.insert(position, rng.choice(alphabet))
            elif action == "replace":
                chars[position] = rng.choice(alphabet)
            else:
                chars = chars[:position]
        response = parse_response("".join(chars))
        assert isinstance(response.status, ParseStatus)
        if response.valid:
            assert isinstance(response.answer, str)
            assert all(isinstance(q.text, str) and isinstance(q.chunk_id, str) for q in response.quotes)
    for _ in range(1000):
        parse_response("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 200))))


def test_completion_text_handles_both_trl_formats():
    assert completion_text("plain") == "plain"
    assert completion_text([{"role": "assistant", "content": "chat"}]) == "chat"
    assert completion_text([]) == []
